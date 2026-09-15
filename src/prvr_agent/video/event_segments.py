from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VisualScanPoint:
    """One cheap sidekick observation used only for temporal structure discovery."""

    timestamp: float
    change_score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("scan timestamp must be finite and non-negative")
        if not math.isfinite(self.change_score) or self.change_score < 0:
            raise ValueError("change_score must be finite and non-negative")


@dataclass(frozen=True)
class EventSegment:
    """Variable-length event proposal produced without retrieval-peak guidance."""

    index: int
    start: float
    end: float
    visual_salience: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("segment index must be non-negative")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("segment bounds must be finite")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("event segment must have a positive duration")
        if not math.isfinite(self.visual_salience) or not 0.0 <= self.visual_salience <= 1.0:
            raise ValueError("visual_salience must be finite and in [0, 1]")


def _quantile(values: list[float], q: float) -> float:
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


def build_event_segments(
    duration: float,
    scan_points: list[VisualScanPoint],
    *,
    min_segment_seconds: float = 4.0,
    max_segment_seconds: float = 24.0,
    boundary_quantile: float = 0.80,
    max_segments: int = 96,
) -> list[EventSegment]:
    """Convert a dense visual-change scan into variable-length event segments.

    The sidekick proposes boundaries from local visual-change maxima and inserts
    forced boundaries when a segment would become too long. These boundaries are
    proposals only: CQHG relevance is still decided by the multimodal observer.
    This avoids reintroducing query-similarity/argmax peak guidance while giving
    short events a denser, event-aware observation lattice.
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
    points = sorted(
        [point for point in scan_points if 0.0 <= point.timestamp <= duration],
        key=lambda point: point.timestamp,
    )
    if not points:
        points = [VisualScanPoint(timestamp=0.0, change_score=0.0)]

    positive_scores = [point.change_score for point in points if point.change_score > 0.0]
    boundary_threshold = _quantile(positive_scores, boundary_quantile) if positive_scores else float("inf")

    # Normalize salience per video so the sidekick can contribute to refinement
    # priority even when absolute pixel-difference magnitudes are small.
    salience_scale = _quantile(positive_scores, 0.90) if positive_scores else 0.0
    if salience_scale <= 1e-12:
        salience_scale = max(positive_scores, default=1.0)
    salience_scale = max(salience_scale, 1e-12)

    candidate_boundaries: list[float] = []
    for idx, point in enumerate(points):
        if point.timestamp <= 0.0 or point.timestamp >= duration:
            continue
        left = points[idx - 1].change_score if idx > 0 else -1.0
        right = points[idx + 1].change_score if idx + 1 < len(points) else -1.0
        if point.change_score >= boundary_threshold and point.change_score >= left and point.change_score >= right:
            candidate_boundaries.append(float(point.timestamp))

    boundaries = [0.0]
    for candidate in candidate_boundaries:
        # Never allow a visually quiet region to grow without bound. Forced
        # splits preserve temporal locality while still preferring event peaks.
        while candidate - boundaries[-1] > max_segment_seconds:
            forced = boundaries[-1] + max_segment_seconds
            if duration - forced < min_segment_seconds:
                break
            boundaries.append(forced)
        if candidate - boundaries[-1] < min_segment_seconds:
            continue
        if duration - candidate < min_segment_seconds:
            continue
        boundaries.append(candidate)

    while duration - boundaries[-1] > max_segment_seconds:
        forced = boundaries[-1] + max_segment_seconds
        if duration - forced < min_segment_seconds:
            break
        boundaries.append(forced)
    boundaries.append(duration)

    # Remove numerical duplicates while preserving order.
    canonical: list[float] = []
    for value in boundaries:
        if not canonical or value - canonical[-1] > 1e-9:
            canonical.append(value)
    if canonical[-1] != duration:
        canonical[-1] = duration

    if len(canonical) - 1 > max_segments:
        raise ValueError(
            f"event-aware segmentation requires {len(canonical) - 1} segments; "
            f"max_segments={max_segments}. Increase max_segments or max_segment_seconds explicitly."
        )

    segments: list[EventSegment] = []
    for index, (start, end) in enumerate(zip(canonical, canonical[1:])):
        local_scores = [
            point.change_score
            for point in points
            if start - 1e-9 <= point.timestamp <= end + 1e-9
        ]
        raw_salience = max(local_scores, default=0.0)
        visual_salience = max(0.0, min(1.0, raw_salience / salience_scale))
        segments.append(
            EventSegment(
                index=index,
                start=float(start),
                end=float(end),
                visual_salience=visual_salience,
            )
        )
    return segments
