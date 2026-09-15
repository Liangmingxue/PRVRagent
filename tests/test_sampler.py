import pytest

from prvr_agent.video.sampler import TimeWindow, build_overlapping_windows


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
    windows = build_overlapping_windows(100.0, target_seconds=24.0, overlap=0.25, max_windows=12)
    assert windows[0].start == 0.0
    assert windows[-1].end == 100.0
    assert all(a.end >= b.start for a, b in zip(windows, windows[1:]))
    assert all(window.end > window.start for window in windows)


def test_long_video_expands_chunk_width_instead_of_dropping_tail():
    windows = build_overlapping_windows(1000.0, target_seconds=24.0, overlap=0.25, max_windows=8)
    assert len(windows) == 8
    assert windows[0].start == 0.0
    assert windows[-1].end == 1000.0
    assert all(a.end >= b.start for a, b in zip(windows, windows[1:]))


def test_chunk_builder_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        build_overlapping_windows(10.0, overlap=1.0)
