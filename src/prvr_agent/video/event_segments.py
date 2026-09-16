from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class VisualScanPoint:
    """One cheap raw-pixel sidekick observation for temporal structure discovery."""

    timestamp: float
    change_score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("scan timestamp must be finite and non-negative")
        if not math.isfinite(self.change_score) or self.change_score < 0:
            raise ValueError("change_score must be finite and non-negative")


@dataclass(frozen=True)
class SemanticScanPoint:
    """Query-agnostic semantic change derived from ordered video representations."""

    timestamp: float
    change_score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("semantic scan timestamp must be finite and non-negative")
        if not math.isfinite(self.change_score) or self.change_score < 0:
            raise ValueError("semantic change_score must be finite and non-negative")


@dataclass(frozen=True)
class SidekickScanPoint:
    """Robustly normalized fusion of cheap visual and semantic temporal signals."""

    timestamp: float
    visual_score: float
    semantic_score: float
    fused_score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("sidekick timestamp must be finite and non-negative")
        for name, value in (
            ("visual_score", self.visual_score),
            ("semantic_score", self.semantic_score),
            ("fused_score", self.fused_score),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class EventSegment:
    """Variable-length event proposal produced without retrieval-peak guidance."""

    index: int
    start: float
    end: float
    visual_salience: float
    semantic_salience: float = 0.0
    sidekick_salience: float = 0.0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("segment index must be non-negative")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("segment bounds must be finite")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("event segment must have a positive duration")
        for name, value in (
            ("visual_salience", self.visual_salience),
            ("semantic_salience", self.semantic_salience),
            ("sidekick_salience", self.sidekick_salience),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def _robust_normalize(
    values: Sequence[float],
    *,
    baseline_quantile: float = 0.25,
    scale_quantile: float = 0.90,
) -> list[float]:
    """Contrast-normalize a non-negative temporal signal per video.

    A simple percentile divisor can flatten a trace whose low background is
    nearly constant (e.g. 0.02 everywhere plus one 0.90 transition), because the
    90th percentile may still equal the background. We first subtract a robust
    lower baseline and then scale the residual. This preserves relative change
    peaks while suppressing constant camera/noise floors.
    """

    if not values:
        return []
    if not 0.0 <= baseline_quantile < scale_quantile <= 1.0:
        raise ValueError("normalization quantiles must satisfy 0 <= baseline < scale <= 1")
    raw = [max(0.0, float(value)) for value in values]
    if any(not math.isfinite(value) for value in raw):
        raise ValueError("sidekick scan contains non-finite values")
    if not any(value > 0.0 for value in raw):
        return [0.0 for _ in raw]

    baseline = _quantile(raw, baseline_quantile)
    centered = [max(0.0, value - baseline) for value in raw]
    positive = [value for value in centered if value > 0.0]
    if not positive:
        return [0.0 for _ in raw]

    scale = _quantile(positive, scale_quantile)
    if scale <= 1e-12:
        scale = max(positive)
    scale = max(scale, 1e-12)
    return [max(0.0, min(1.0, value / scale)) for value in centered]


def semantic_scores_to_scan_points(
    change_scores: Sequence[float],
    duration: float,
) -> list[SemanticScanPoint]:
    """Map DreamPRVR semantic *boundary* scores onto physical video time.

    DreamPRVR uniformly samples/averages an ordered frame-feature sequence into N
    semantic bins. ``change_scores[i]`` for i>0 compares the region before bin i
    with bin i, so its physical boundary is i/N of the video duration. Mapping it
    as i/(N-1) would incorrectly treat semantic bins as endpoint samples and shift
    every interior boundary later in time.
    """

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    scores = [float(value) for value in change_scores]
    if any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("semantic change scores must be finite and non-negative")
    if not scores:
        return []

    denominator = float(len(scores))
    return [
        SemanticScanPoint(
            timestamp=float(duration) * idx / denominator,
            change_score=score,
        )
        for idx, score in enumerate(scores)
    ]


def _interpolate(points: Sequence[tuple[float, float]], timestamp: float) -> float:
    if not points:
        return 0.0
    if timestamp <= points[0][0]:
        return points[0][1]
    if timestamp >= points[-1][0]:
        return points[-1][1]

    for left, right in zip(points, points[1:]):
        if left[0] <= timestamp <= right[0]:
            span = right[0] - left[0]
            if span <= 1e-12:
                return max(left[1], right[1])
            alpha = (timestamp - left[0]) / span
            return left[1] * (1.0 - alpha) + right[1] * alpha
    return points[-1][1]


def fuse_sidekick_scans(
    duration: float,
    visual_points: Sequence[VisualScanPoint],
    semantic_points: Sequence[SemanticScanPoint] = (),
    *,
    visual_weight: float = 0.5,
    semantic_weight: float = 0.5,
) -> list[SidekickScanPoint]:
    """Fuse raw-pixel and semantic change into one query-agnostic temporal scan.

    Each modality is contrast-normalized per video, then aligned on the union of
    timestamps. Fusion uses a weighted noisy-OR so a strong signal from either
    modality can preserve a boundary candidate. If semantic features are absent,
    the result falls back to normalized visual change.
    """

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not math.isfinite(visual_weight) or not math.isfinite(semantic_weight):
        raise ValueError("sidekick fusion weights must be finite")
    if visual_weight < 0 or semantic_weight < 0:
        raise ValueError("sidekick fusion weights must be non-negative")
    if visual_weight + semantic_weight <= 0:
        raise ValueError("at least one sidekick fusion weight must be positive")

    visual = sorted(
        [point for point in visual_points if 0.0 <= point.timestamp <= duration],
        key=lambda point: point.timestamp,
    )
    semantic = sorted(
        [point for point in semantic_points if 0.0 <= point.timestamp <= duration],
        key=lambda point: point.timestamp,
    )
    if not visual and not semantic:
        return [SidekickScanPoint(0.0, 0.0, 0.0, 0.0)]

    visual_norm = _robust_normalize([point.change_score for point in visual])
    semantic_norm = _robust_normalize([point.change_score for point in semantic])
    visual_series = [(point.timestamp, score) for point, score in zip(visual, visual_norm)]
    semantic_series = [(point.timestamp, score) for point, score in zip(semantic, semantic_norm)]

    timestamps = sorted(
        {0.0, float(duration)}
        | {float(point.timestamp) for point in visual}
        | {float(point.timestamp) for point in semantic}
    )

    active_visual_weight = visual_weight if visual_series else 0.0
    active_semantic_weight = semantic_weight if semantic_series else 0.0
    total_weight = active_visual_weight + active_semantic_weight
    if total_weight <= 0:
        active_visual_weight = 1.0 if visual_series else 0.0
        active_semantic_weight = 1.0 if semantic_series else 0.0
        total_weight = active_visual_weight + active_semantic_weight
    visual_exp = active_visual_weight / total_weight
    semantic_exp = active_semantic_weight / total_weight

    output: list[SidekickScanPoint] = []
    for timestamp in timestamps:
        visual_score = _interpolate(visual_series, timestamp) if visual_series else 0.0
        semantic_score = _interpolate(semantic_series, timestamp) if semantic_series else 0.0
        visual_survival = (1.0 - visual_score) ** visual_exp if visual_exp > 0 else 1.0
        semantic_survival = (1.0 - semantic_score) ** semantic_exp if semantic_exp > 0 else 1.0
        fused = 1.0 - visual_survival * semantic_survival
        output.append(
            SidekickScanPoint(
                timestamp=timestamp,
                visual_score=max(0.0, min(1.0, visual_score)),
                semantic_score=max(0.0, min(1.0, semantic_score)),
                fused_score=max(0.0, min(1.0, fused)),
            )
        )
    return output


def _as_sidekick_points(
    duration: float,
    scan_points: Sequence[VisualScanPoint | SidekickScanPoint],
) -> list[SidekickScanPoint]:
    if not scan_points:
        return fuse_sidekick_scans(duration, [])
    if all(isinstance(point, SidekickScanPoint) for point in scan_points):
        return sorted(
            [point for point in scan_points if 0.0 <= point.timestamp <= duration],
            key=lambda point: point.timestamp,
        )
    if all(isinstance(point, VisualScanPoint) for point in scan_points):
        return fuse_sidekick_scans(duration, list(scan_points))
    raise TypeError("scan_points must be uniformly VisualScanPoint or SidekickScanPoint")


def _preferred_boundaries(
    duration: float,
    boundaries: Sequence[float],
    *,
    min_segment_seconds: float,
) -> list[float]:
    """Validate and sparsify global semantic boundaries before local fusion."""

    raw: list[float] = []
    for value in boundaries:
        boundary = float(value)
        if not math.isfinite(boundary):
            raise ValueError("preferred event boundaries must be finite")
        if boundary <= 0.0 or boundary >= duration:
            raise ValueError("preferred event boundaries must lie strictly inside the video")
        raw.append(boundary)

    selected: list[float] = []
    for boundary in sorted(set(raw)):
        if boundary < min_segment_seconds or duration - boundary < min_segment_seconds:
            continue
        if selected and boundary - selected[-1] < min_segment_seconds:
            continue
        selected.append(boundary)
    return selected


def build_event_segments(
    duration: float,
    scan_points: Sequence[VisualScanPoint | SidekickScanPoint],
    *,
    preferred_boundaries: Sequence[float] = (),
    min_segment_seconds: float = 4.0,
    max_segment_seconds: float = 24.0,
    boundary_quantile: float = 0.80,
    max_segments: int = 96,
) -> list[EventSegment]:
    """Convert hybrid local novelty plus global semantic structure into segments.

    ``preferred_boundaries`` are query-agnostic global change points, e.g. from a
    KTS dynamic program. Local visual/semantic novelty remains useful for sharp
    transitions omitted by global model selection. Neither signal establishes
    query relevance; both only define where the expensive observer should look.
    """

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not math.isfinite(min_segment_seconds) or min_segment_seconds <= 0:
        raise ValueError("min_segment_seconds must be finite and positive")
    if not math.isfinite(max_segment_seconds) or max_segment_seconds <= 0:
        raise ValueError("max_segment_seconds must be finite and positive")
    if min_segment_seconds > max_segment_seconds:
        raise ValueError("min_segment_seconds must not exceed max_segment_seconds")
    if not math.isfinite(boundary_quantile) or not 0.0 <= boundary_quantile <= 1.0:
        raise ValueError("boundary_quantile must be finite and in [0, 1]")
    if max_segments <= 0:
        raise ValueError("max_segments must be positive")

    duration = float(duration)
    points = _as_sidekick_points(duration, scan_points)
    if not points:
        points = [SidekickScanPoint(0.0, 0.0, 0.0, 0.0)]

    global_boundaries = _preferred_boundaries(
        duration,
        preferred_boundaries,
        min_segment_seconds=min_segment_seconds,
    )

    positive_scores = [point.fused_score for point in points if point.fused_score > 0.0]
    boundary_threshold = _quantile(positive_scores, boundary_quantile) if positive_scores else float("inf")

    local_boundaries: list[float] = []
    for idx, point in enumerate(points):
        if point.timestamp <= 0.0 or point.timestamp >= duration:
            continue
        left = points[idx - 1].fused_score if idx > 0 else -1.0
        right = points[idx + 1].fused_score if idx + 1 < len(points) else -1.0
        is_strict_local_peak = (
            point.fused_score >= left
            and point.fused_score >= right
            and (point.fused_score > left or point.fused_score > right)
        )
        if point.fused_score < boundary_threshold or not is_strict_local_peak:
            continue
        candidate = float(point.timestamp)
        if candidate < min_segment_seconds or duration - candidate < min_segment_seconds:
            continue
        # Give the global KTS structure precedence when two proposals describe the
        # same temporal transition at slightly different timestamps.
        if any(abs(candidate - global_boundary) < min_segment_seconds for global_boundary in global_boundaries):
            continue
        local_boundaries.append(candidate)

    candidate_boundaries = sorted(set(global_boundaries + local_boundaries))

    boundaries = [0.0]
    for candidate in candidate_boundaries:
        # If a meaningful boundary lies just beyond the maximum-duration limit,
        # place the forced split early enough to preserve that boundary instead of
        # blindly splitting at last+max and then discarding the semantic boundary.
        while candidate - boundaries[-1] > max_segment_seconds:
            forced = min(
                boundaries[-1] + max_segment_seconds,
                candidate - min_segment_seconds,
            )
            if forced - boundaries[-1] < min_segment_seconds:
                break
            boundaries.append(forced)
        if candidate - boundaries[-1] < min_segment_seconds:
            continue
        if duration - candidate < min_segment_seconds:
            continue
        boundaries.append(candidate)

    while duration - boundaries[-1] > max_segment_seconds:
        forced = min(
            boundaries[-1] + max_segment_seconds,
            duration - min_segment_seconds,
        )
        if forced - boundaries[-1] < min_segment_seconds:
            break
        boundaries.append(forced)
    boundaries.append(duration)

    canonical: list[float] = []
    for value in boundaries:
        if not canonical or value - canonical[-1] > 1e-9:
            canonical.append(value)
    if canonical[-1] != duration:
        canonical[-1] = duration

    if any(right - left < min_segment_seconds - 1e-9 for left, right in zip(canonical, canonical[1:])):
        raise RuntimeError("event segmentation produced a segment shorter than min_segment_seconds")
    if any(right - left > max_segment_seconds + 1e-9 for left, right in zip(canonical, canonical[1:])):
        raise RuntimeError("event segmentation produced a segment longer than max_segment_seconds")

    if len(canonical) - 1 > max_segments:
        raise ValueError(
            f"event-aware segmentation requires {len(canonical) - 1} segments; "
            f"max_segments={max_segments}. Increase max_segments or max_segment_seconds explicitly."
        )

    segments: list[EventSegment] = []
    for index, (start, end) in enumerate(zip(canonical, canonical[1:])):
        local = [point for point in points if start - 1e-9 <= point.timestamp <= end + 1e-9]
        visual_salience = max((point.visual_score for point in local), default=0.0)
        semantic_salience = max((point.semantic_score for point in local), default=0.0)
        sidekick_salience = max((point.fused_score for point in local), default=0.0)
        segments.append(
            EventSegment(
                index=index,
                start=float(start),
                end=float(end),
                visual_salience=visual_salience,
                semantic_salience=semantic_salience,
                sidekick_salience=sidekick_salience,
            )
        )
    return segments
