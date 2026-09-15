import pytest

from prvr_agent.video.event_segments import (
    SemanticScanPoint,
    VisualScanPoint,
    build_event_segments,
    fuse_sidekick_scans,
    semantic_scores_to_scan_points,
)


def test_semantic_trace_maps_uniform_positions_to_video_time():
    points = semantic_scores_to_scan_points([0.0, 0.1, 1.0], 20.0)
    assert [point.timestamp for point in points] == [0.0, 10.0, 20.0]
    assert points[-1].change_score == 1.0


def test_semantic_signal_can_create_boundary_when_pixels_are_quiet():
    visual = [
        VisualScanPoint(timestamp=0.0, change_score=0.0),
        VisualScanPoint(timestamp=5.0, change_score=0.0),
        VisualScanPoint(timestamp=10.0, change_score=0.0),
        VisualScanPoint(timestamp=15.0, change_score=0.0),
        VisualScanPoint(timestamp=20.0, change_score=0.0),
    ]
    semantic = [
        SemanticScanPoint(timestamp=0.0, change_score=0.0),
        SemanticScanPoint(timestamp=5.0, change_score=0.05),
        SemanticScanPoint(timestamp=10.0, change_score=1.0),
        SemanticScanPoint(timestamp=15.0, change_score=0.05),
        SemanticScanPoint(timestamp=20.0, change_score=0.0),
    ]
    fused = fuse_sidekick_scans(
        20.0,
        visual,
        semantic,
        visual_weight=0.5,
        semantic_weight=0.5,
    )
    assert max(point.semantic_score for point in fused) == 1.0
    assert max(point.fused_score for point in fused) == 1.0

    segments = build_event_segments(
        20.0,
        fused,
        min_segment_seconds=4.0,
        max_segment_seconds=20.0,
        boundary_quantile=0.8,
        max_segments=8,
    )
    assert len(segments) == 2
    assert segments[0].end == pytest.approx(10.0)
    assert segments[1].start == pytest.approx(10.0)
    assert max(segment.semantic_salience for segment in segments) == 1.0


def test_visual_only_fallback_remains_available():
    visual = [
        VisualScanPoint(timestamp=0.0, change_score=0.0),
        VisualScanPoint(timestamp=5.0, change_score=0.2),
        VisualScanPoint(timestamp=10.0, change_score=1.0),
    ]
    fused = fuse_sidekick_scans(10.0, visual, [])
    assert all(point.semantic_score == 0.0 for point in fused)
    assert max(point.visual_score for point in fused) == 1.0
    assert max(point.fused_score for point in fused) == 1.0


def test_hybrid_noisy_or_preserves_strong_signal_from_either_modality():
    visual = [
        VisualScanPoint(timestamp=0.0, change_score=0.0),
        VisualScanPoint(timestamp=5.0, change_score=1.0),
        VisualScanPoint(timestamp=10.0, change_score=0.0),
    ]
    semantic = [
        SemanticScanPoint(timestamp=0.0, change_score=0.0),
        SemanticScanPoint(timestamp=5.0, change_score=0.0),
        SemanticScanPoint(timestamp=10.0, change_score=1.0),
    ]
    fused = fuse_sidekick_scans(10.0, visual, semantic)
    by_time = {point.timestamp: point for point in fused}
    assert by_time[5.0].fused_score == pytest.approx(1.0)
    assert by_time[10.0].fused_score == pytest.approx(1.0)


def test_sidekick_fusion_rejects_invalid_weights():
    with pytest.raises(ValueError):
        fuse_sidekick_scans(10.0, [], [], visual_weight=0.0, semantic_weight=0.0)
    with pytest.raises(ValueError):
        fuse_sidekick_scans(10.0, [], [], visual_weight=-1.0, semantic_weight=1.0)
