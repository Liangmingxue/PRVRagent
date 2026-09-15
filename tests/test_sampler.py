import pytest

from prvr_agent.video.sampler import TimeWindow


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
