import numpy as np
import pytest

from prvr_agent.evaluation import (
    build_query_to_video_ground_truth,
    evaluate_prvr_scores,
    recall_at_k,
)


def test_ground_truth_uses_video_prefix_before_hash():
    q2v = build_query_to_video_ground_truth(
        ["v0", "v1", "v2"],
        ["v1#0", "v0#caption-a", "v2#17"],
    )
    assert q2v == {0: [1], 1: [0], 2: [2]}


def test_recall_uses_best_valid_ground_truth_rank():
    scores = np.asarray([
        [0.9, 0.8, 0.1, 0.0],
        [0.4, 0.5, 0.6, 0.7],
    ])
    recalls = recall_at_k(
        scores,
        {
            0: [1, 3],  # best relevant video is rank 2
            1: [3],     # relevant video is rank 1
        },
        ks=(1, 2, 4),
    )
    assert recalls[1] == pytest.approx(50.0)
    assert recalls[2] == pytest.approx(100.0)
    assert recalls[4] == pytest.approx(100.0)


def test_dreamprvr_style_metrics_include_r100_in_rsum():
    video_ids = [f"v{i}" for i in range(120)]
    query_ids = ["v0#q0", "v50#q1", "v119#q2"]

    scores = np.zeros((3, 120), dtype=np.float64)
    # q0 -> ground truth rank 1
    scores[0] = np.linspace(0.0, -1.0, 120)
    # q1 -> ground truth rank 5
    order_q1 = list(range(120))
    order_q1.remove(50)
    order_q1.insert(4, 50)
    for rank, video_index in enumerate(order_q1):
        scores[1, video_index] = -float(rank)
    # q2 -> ground truth rank 50, so it contributes only to R@100.
    order_q2 = list(range(120))
    order_q2.remove(119)
    order_q2.insert(49, 119)
    for rank, video_index in enumerate(order_q2):
        scores[2, video_index] = -float(rank)

    metrics = evaluate_prvr_scores(
        scores,
        video_ids=video_ids,
        query_ids=query_ids,
    )
    assert metrics.r1 == pytest.approx(100.0 / 3.0)
    assert metrics.r5 == pytest.approx(200.0 / 3.0)
    assert metrics.r10 == pytest.approx(200.0 / 3.0)
    assert metrics.r100 == pytest.approx(100.0)
    assert metrics.rsum == pytest.approx(
        metrics.r1 + metrics.r5 + metrics.r10 + metrics.r100
    )


def test_metrics_reject_missing_ground_truth_video():
    with pytest.raises(ValueError, match="absent"):
        build_query_to_video_ground_truth(["v0"], ["v1#q"])


def test_metrics_reject_non_finite_score():
    with pytest.raises(ValueError, match="finite"):
        recall_at_k(np.asarray([[float("nan")]]), {0: [0]})
