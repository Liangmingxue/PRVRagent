import numpy as np
import pytest

from prvr_agent.video.kernel_temporal import (
    change_points_to_fractions,
    kernel_temporal_segment,
)


def test_kernel_temporal_dp_finds_piecewise_semantic_transition():
    features = np.asarray(
        [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4,
        dtype=np.float64,
    )
    result = kernel_temporal_segment(
        features,
        max_change_points=3,
        penalty_scale=0.6,
        min_segment_length=2,
    )
    assert result.change_points == (4,)
    assert result.selected_change_points == 1
    assert change_points_to_fractions(result.change_points, len(features)) == [0.5]


def test_kernel_temporal_dp_does_not_split_constant_features():
    features = np.asarray([[1.0, 0.0]] * 10, dtype=np.float64)
    result = kernel_temporal_segment(
        features,
        max_change_points=4,
        penalty_scale=0.6,
        min_segment_length=2,
    )
    assert result.change_points == ()
    assert result.selected_change_points == 0


def test_kernel_temporal_respects_minimum_segment_length():
    features = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    result = kernel_temporal_segment(
        features,
        max_change_points=3,
        penalty_scale=0.0,
        min_segment_length=2,
    )
    assert all(point >= 2 for point in result.change_points)
    assert all(
        right - left >= 2
        for left, right in zip((0,) + result.change_points, result.change_points + (len(features),))
    )


def test_change_point_fraction_uses_bin_boundary_not_endpoint_interpolation():
    # Five semantic bins span [0, T].  The boundary before bin 2 is 2/5 of T,
    # whereas endpoint interpolation would incorrectly place it at 2/4 of T.
    assert change_points_to_fractions([2], 5) == [0.4]


def test_change_point_fraction_rejects_non_monotonic_or_endpoint_boundaries():
    with pytest.raises(ValueError):
        change_points_to_fractions([2, 2], 5)
    with pytest.raises(ValueError):
        change_points_to_fractions([5], 5)
