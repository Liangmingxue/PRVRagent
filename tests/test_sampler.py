from pathlib import Path

import pytest

from prvr_agent.video.event_segments import (
    VisualScanPoint,
    build_event_segments,
    semantic_scores_to_scan_points,
)
from prvr_agent.video.sampler import (
    FRAME_DIRECTORY_FPS_ENV,
    DecordFrameSampler,
    FrameDirectorySampler,
    TimeWindow,
    build_overlapping_windows,
    open_temporal_visual_source,
    uniform_bin_center_indices,
)


def test_time_window_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        TimeWindow(-1.0, 1.0)
    with pytest.raises(ValueError):
        TimeWindow(2.0, 1.0)
    with pytest.raises(ValueError):
        TimeWindow(0.0, float("nan"))


def test_time_window_clamps_to_duration():
    assert TimeWindow(2.0, 20.0).clamp(10.0) == TimeWindow(2.0, 10.0)
    with pytest.raises(ValueError):
        TimeWindow(0.0, 1.0).clamp(0.0)


def test_overlapping_windows_cover_full_video_without_gaps():
    windows = build_overlapping_windows(100.0, target_seconds=20.0, overlap=0.25, max_windows=12)
    assert windows[0].start == 0.0
    assert windows[-1].end == 100.0
    assert all(a.end >= b.start for a, b in zip(windows, windows[1:]))
    assert all(window.end > window.start for window in windows)
    assert all(window.end - window.start <= 20.0 + 1e-9 for window in windows)


def test_long_video_never_silently_widens_local_chunks():
    with pytest.raises(ValueError, match="requires"):
        build_overlapping_windows(1000.0, target_seconds=20.0, overlap=0.25, max_windows=8)

    windows = build_overlapping_windows(1000.0, target_seconds=20.0, overlap=0.25, max_windows=80)
    assert windows[0].start == 0.0
    assert windows[-1].end == 1000.0
    assert all(window.end - window.start <= 20.0 + 1e-9 for window in windows)
    assert all(a.end >= b.start for a, b in zip(windows, windows[1:]))


def test_chunk_builder_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        build_overlapping_windows(10.0, overlap=1.0)


def test_bin_center_sampling_avoids_wasting_sparse_samples_on_boundaries():
    indices = uniform_bin_center_indices(0, 19, 4)
    assert indices == [2, 7, 12, 17]
    assert indices[0] > 0
    assert indices[-1] < 19


def test_bin_center_sampling_returns_every_frame_when_budget_is_dense():
    assert uniform_bin_center_indices(5, 8, 10) == [5, 6, 7, 8]


def test_frame_directory_sampler_naturally_orders_frames_and_uses_explicit_fps(tmp_path: Path):
    frame_dir = tmp_path / "clip_01"
    frame_dir.mkdir()
    for name in ["10.jpg", "2.jpg", "1.jpg"]:
        (frame_dir / name).write_bytes(b"placeholder")

    source = FrameDirectorySampler(frame_dir, fps=3.0)
    assert [path.name for path in source.frame_paths] == ["1.jpg", "2.jpg", "10.jpg"]
    assert source.frame_count == 3
    assert source.fps == 3.0
    assert source.duration == pytest.approx(1.0)


def test_frame_directory_source_requires_explicit_dataset_fps(tmp_path: Path, monkeypatch):
    frame_dir = tmp_path / "clip_01"
    frame_dir.mkdir()
    (frame_dir / "0001.jpg").write_bytes(b"placeholder")
    monkeypatch.delenv(FRAME_DIRECTORY_FPS_ENV, raising=False)

    with pytest.raises(ValueError, match="frame_directory_fps"):
        open_temporal_visual_source(frame_dir)


def test_existing_observer_sampler_dispatches_frame_directory_via_env(tmp_path: Path, monkeypatch):
    frame_dir = tmp_path / "clip_01"
    frame_dir.mkdir()
    for idx in range(3):
        (frame_dir / f"{idx:04d}.jpg").write_bytes(b"placeholder")
    monkeypatch.setenv(FRAME_DIRECTORY_FPS_ENV, "3")

    source = DecordFrameSampler(frame_dir)
    assert isinstance(source, FrameDirectorySampler)
    assert source.fps == 3.0
    assert source.frame_count == 3


def test_semantic_change_scores_map_to_uniform_bin_boundaries():
    points = semantic_scores_to_scan_points([0.0, 0.1, 0.9, 0.2, 0.1], duration=100.0)
    assert [point.timestamp for point in points] == pytest.approx([0.0, 20.0, 40.0, 60.0, 80.0])
    # The strong change at index 2 is the boundary before semantic bin 2: 2/5 T.
    assert points[2].timestamp == pytest.approx(40.0)


def test_event_segmenter_uses_visual_change_peak_as_boundary():
    points = [VisualScanPoint(timestamp=float(t), change_score=0.02) for t in range(0, 31, 2)]
    # A strong transition around 12 s should become an adaptive event boundary.
    points[6] = VisualScanPoint(timestamp=12.0, change_score=0.90)
    segments = build_event_segments(
        30.0,
        points,
        min_segment_seconds=4.0,
        max_segment_seconds=20.0,
        boundary_quantile=0.80,
        max_segments=8,
    )
    assert segments[0].start == 0.0
    assert segments[-1].end == 30.0
    assert any(abs(segment.end - 12.0) < 1e-6 for segment in segments)
    assert all(segment.end - segment.start <= 20.0 + 1e-9 for segment in segments)


def test_global_semantic_boundary_survives_forced_locality_split():
    # A naive forced split at 20 s would make the meaningful 22 s boundary too
    # close and discard it. The event builder should instead force at 18 s so the
    # 22 s structural boundary can be retained while respecting max duration.
    segments = build_event_segments(
        40.0,
        [VisualScanPoint(timestamp=0.0, change_score=0.0)],
        preferred_boundaries=[22.0],
        min_segment_seconds=4.0,
        max_segment_seconds=20.0,
        max_segments=8,
    )
    boundaries = [segment.end for segment in segments[:-1]]
    assert 22.0 in boundaries
    assert all(4.0 - 1e-9 <= segment.end - segment.start <= 20.0 + 1e-9 for segment in segments)


def test_event_segmenter_forces_locality_even_without_visual_peaks():
    points = [VisualScanPoint(timestamp=float(t), change_score=0.0) for t in range(0, 61, 5)]
    segments = build_event_segments(
        60.0,
        points,
        min_segment_seconds=4.0,
        max_segment_seconds=20.0,
        max_segments=8,
    )
    assert [(segment.start, segment.end) for segment in segments] == [
        (0.0, 20.0),
        (20.0, 40.0),
        (40.0, 60.0),
    ]


def test_event_segmenter_fails_instead_of_silently_coarsening_resolution():
    with pytest.raises(ValueError, match="requires"):
        build_event_segments(
            200.0,
            [VisualScanPoint(timestamp=0.0, change_score=0.0)],
            min_segment_seconds=4.0,
            max_segment_seconds=20.0,
            max_segments=4,
        )
