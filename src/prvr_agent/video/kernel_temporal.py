from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KernelTemporalResult:
    """Result of task-agnostic kernel temporal segmentation.

    Change points are sequence indices at which a new segment starts.  They are
    intentionally represented in feature coordinates so the caller can map them
    to physical time using the video duration without making the segmenter depend
    on frame rate or benchmark-specific timestamps.
    """

    change_points: tuple[int, ...]
    costs: tuple[float, ...]
    selected_change_points: int


def _as_feature_matrix(features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("kernel temporal segmentation expects [time, dim] features")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("kernel temporal segmentation requires a non-empty feature matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("kernel temporal features must be finite")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Zero vectors carry no semantic direction. Keeping them at zero is safer than
    # injecting arbitrary directions and still gives a valid positive semidefinite
    # linear kernel.
    return matrix / np.maximum(norms, 1e-12)


def _segment_scatter_costs(kernel: np.ndarray) -> np.ndarray:
    """Pre-compute within-segment kernel scatter for every contiguous interval.

    For a segment [i, j], the kernel k-means scatter is

        sum_t K_tt - (1 / length) * sum_{s,t} K_st.

    Prefix sums make each interval O(1) after an O(N^2) pre-computation.  This is
    the objective used by Kernel Temporal Segmentation, implemented here from the
    mathematical objective rather than vendoring third-party KTS source code.
    """

    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    if not np.isfinite(kernel).all():
        raise ValueError("kernel must be finite")

    n = int(kernel.shape[0])
    diagonal_prefix = np.concatenate(([0.0], np.cumsum(np.diag(kernel))))
    block_prefix = np.zeros((n + 1, n + 1), dtype=np.float64)
    block_prefix[1:, 1:] = np.cumsum(np.cumsum(kernel, axis=0), axis=1)

    scatter = np.zeros((n, n), dtype=np.float64)
    for start in range(n):
        for end in range(start, n):
            length = end - start + 1
            block_sum = (
                block_prefix[end + 1, end + 1]
                - block_prefix[start, end + 1]
                - block_prefix[end + 1, start]
                + block_prefix[start, start]
            )
            diag_sum = diagonal_prefix[end + 1] - diagonal_prefix[start]
            value = diag_sum - block_sum / float(length)
            # Numerical roundoff can produce tiny negative values for a perfectly
            # constant segment. The theoretical scatter is non-negative.
            scatter[start, end] = max(0.0, float(value))
    return scatter


def _dynamic_program(
    scatter: np.ndarray,
    *,
    max_change_points: int,
    min_segment_length: int,
    max_segment_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the globally optimal contiguous segmentation for 0..M changes."""

    n = int(scatter.shape[0])
    max_segments = max_change_points + 1
    inf = float("inf")
    dp = np.full((max_segments + 1, n + 1), inf, dtype=np.float64)
    previous = np.full((max_segments + 1, n + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    for segments in range(1, max_segments + 1):
        min_end = segments * min_segment_length
        max_end = min(n, segments * max_segment_length)
        for end in range(min_end, max_end + 1):
            prev_min = max(
                (segments - 1) * min_segment_length,
                end - max_segment_length,
            )
            prev_max = min(
                end - min_segment_length,
                (segments - 1) * max_segment_length,
            )
            if prev_min > prev_max:
                continue
            for prev in range(prev_min, prev_max + 1):
                prefix_cost = dp[segments - 1, prev]
                if not math.isfinite(float(prefix_cost)):
                    continue
                candidate = prefix_cost + scatter[prev, end - 1]
                if candidate < dp[segments, end]:
                    dp[segments, end] = candidate
                    previous[segments, end] = prev
    return dp, previous


def kernel_temporal_segment(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    max_change_points: int = 8,
    penalty_scale: float = 0.6,
    min_segment_length: int = 2,
    max_segment_length: int | None = None,
) -> KernelTemporalResult:
    """Globally segment an ordered semantic feature sequence with a KTS objective.

    The number of change points is selected from 0..``max_change_points`` using
    the standard KTS model-complexity form

        score / N + lambda * m/(2N) * (log(N/m) + 1).

    ``penalty_scale`` is exposed because the optimal value is representation and
    dataset dependent.  A linear kernel over L2-normalized semantic features keeps
    the method lightweight and query-agnostic for the APEI sidekick.
    """

    matrix = _as_feature_matrix(features)
    n = int(matrix.shape[0])

    if max_change_points < 0:
        raise ValueError("max_change_points must be non-negative")
    if not math.isfinite(penalty_scale) or penalty_scale < 0:
        raise ValueError("penalty_scale must be finite and non-negative")
    if min_segment_length <= 0:
        raise ValueError("min_segment_length must be positive")
    if max_segment_length is None:
        max_segment_length = n
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive")
    if min_segment_length > max_segment_length:
        raise ValueError("min_segment_length must not exceed max_segment_length")

    # There cannot be more feasible segments than the sequence can support.
    feasible_max_changes = max(0, n // min_segment_length - 1)
    max_change_points = min(int(max_change_points), feasible_max_changes)

    kernel = matrix @ matrix.T
    scatter = _segment_scatter_costs(kernel)
    dp, previous = _dynamic_program(
        scatter,
        max_change_points=max_change_points,
        min_segment_length=int(min_segment_length),
        max_segment_length=int(max_segment_length),
    )

    costs: list[float] = []
    for changes in range(max_change_points + 1):
        raw_score = float(dp[changes + 1, n])
        if not math.isfinite(raw_score):
            costs.append(float("inf"))
            continue
        penalty = 0.0
        if changes > 0:
            penalty = (
                penalty_scale
                * changes
                / (2.0 * n)
                * (math.log(float(n) / changes) + 1.0)
            )
        costs.append(raw_score / float(n) + penalty)

    finite_indices = [idx for idx, value in enumerate(costs) if math.isfinite(value)]
    if not finite_indices:
        raise ValueError("no feasible kernel temporal segmentation under the length constraints")
    selected = min(finite_indices, key=lambda idx: (costs[idx], idx))

    change_points: list[int] = []
    end = n
    for segments in range(selected + 1, 1, -1):
        prev = int(previous[segments, end])
        if prev <= 0:
            raise RuntimeError("failed to backtrack kernel temporal segmentation")
        change_points.append(prev)
        end = prev
    change_points.reverse()

    return KernelTemporalResult(
        change_points=tuple(change_points),
        costs=tuple(float(value) for value in costs),
        selected_change_points=int(selected),
    )


def change_points_to_fractions(change_points: Sequence[int], sequence_length: int) -> list[float]:
    """Map feature-space boundaries to normalized video-time boundaries.

    A change point ``c`` means the new segment starts at semantic bin ``c``.
    With ``N`` uniform temporal bins, the physical boundary is therefore ``c/N``
    of the video duration, not ``c/(N-1)``.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    output: list[float] = []
    previous = 0
    for value in change_points:
        point = int(value)
        if point <= previous or point >= sequence_length:
            raise ValueError("change points must be strictly increasing and inside the sequence")
        output.append(point / float(sequence_length))
        previous = point
    return output
