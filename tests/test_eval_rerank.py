import numpy as np
import pytest

from prvr_agent.evaluation import rerank_score_matrix_rows, rerank_topk_score_slots
from prvr_agent.pipeline import RerankedCandidate
from prvr_agent.prospective import ProspectiveAssessment
from prvr_agent.schemas import Candidate


def _reranked(video_index: int, base_score: float, final_score: float) -> RerankedCandidate:
    candidate = Candidate(
        video_id=f"v{video_index}",
        video_index=video_index,
        base_score=base_score,
        clip_score=base_score,
        frame_score=base_score,
    )
    assessment = ProspectiveAssessment(
        beliefs=(),
        graph_score=0.0,
        world_score=0.0,
    )
    return RerankedCandidate(
        candidate=candidate,
        final_score=final_score,
        graph_score=0.0,
        world_score=0.0,
        assessment=assessment,
    )


def test_topk_reranking_permutes_only_original_score_slots():
    base = np.asarray([0.90, 0.80, 0.70, 0.60, 0.10])
    reranked = [
        _reranked(0, 0.90, 0.20),
        _reranked(1, 0.80, 0.95),
        _reranked(2, 0.70, 0.50),
    ]
    output = rerank_topk_score_slots(base, reranked)

    # APEI orders v1 > v2 > v0, but the numerical slots remain 0.9, 0.8, 0.7.
    assert output.tolist() == pytest.approx([0.70, 0.90, 0.80, 0.60, 0.10])
    assert sorted(output[:3].tolist()) == pytest.approx(sorted(base[:3].tolist()))
    assert output[3:].tolist() == pytest.approx(base[3:].tolist())


def test_topk_reranking_rejects_shortlist_that_is_not_base_topk():
    base = np.asarray([0.90, 0.80, 0.70, 0.60])
    reranked = [
        _reranked(0, 0.90, 0.2),
        _reranked(2, 0.70, 0.9),
    ]
    with pytest.raises(ValueError, match="exactly the DreamPRVR Top-K"):
        rerank_topk_score_slots(base, reranked)


def test_matrix_reranking_preserves_untouched_tail_per_query():
    base = np.asarray([
        [0.9, 0.8, 0.7, 0.1],
        [0.1, 0.8, 0.9, 0.7],
    ])
    rows = [
        [
            _reranked(0, 0.9, 0.1),
            _reranked(1, 0.8, 0.9),
        ],
        [
            _reranked(2, 0.9, 0.2),
            _reranked(1, 0.8, 0.8),
        ],
    ]
    output = rerank_score_matrix_rows(base, rows)
    assert output[0].tolist() == pytest.approx([0.8, 0.9, 0.7, 0.1])
    assert output[1].tolist() == pytest.approx([0.1, 0.9, 0.8, 0.7])
