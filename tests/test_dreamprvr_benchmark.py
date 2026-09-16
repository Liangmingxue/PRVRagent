import numpy as np
import pytest

torch = pytest.importorskip("torch")

from prvr_agent.integration import rerank_dreamprvr_query_loader
from prvr_agent.pipeline import RerankedCandidate
from prvr_agent.prospective import ProspectiveAssessment
from prvr_agent.retriever import DreamPRVRAdapter


class FakeDreamPRVR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def encode_query(self, query_feat, query_mask):
        return query_feat

    def get_pred_from_raw_query(self, query_feat, query_mask, _, clip_context, frame_context):
        query_vectors = self.encode_query(query_feat, query_mask)
        clip_scores = DreamPRVRAdapter._max_scores(query_vectors, clip_context)
        frame_scores = DreamPRVRAdapter._max_scores(query_vectors, frame_context)
        return clip_scores, frame_scores


class FakeQueryDataset:
    captions = {
        "v0#0": "person performs event zero",
        "v1#0": "person performs event one",
    }


class FakeQueryLoader:
    dataset = FakeQueryDataset()

    def __iter__(self):
        yield (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.ones(2, 1),
            (0, 1),
            ("v0#0", "v1#0"),
        )


class ReverseShortlistReranker:
    def rerank(self, query, candidates):
        assert query
        assessment = ProspectiveAssessment(beliefs=(), graph_score=0.0, world_score=0.0)
        return [
            RerankedCandidate(
                candidate=candidate,
                final_score=-float(candidate.base_score),
                graph_score=0.0,
                world_score=0.0,
                assessment=assessment,
            )
            for candidate in candidates
        ]


def _context():
    features = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.8, 0.2]],
        ]
    )
    return {
        "video_metas": ["v0", "v1", "v2"],
        "video_proposal_feat": features.clone(),
        "video_feat": features.clone(),
    }


def test_bridge_uses_upstream_full_scores_and_rank_slot_shortlist_reranking():
    result = rerank_dreamprvr_query_loader(
        model=FakeDreamPRVR(),
        query_loader=FakeQueryLoader(),
        context_info=_context(),
        reranker=ReverseShortlistReranker(),
        clip_scale_weight=0.5,
        frame_scale_weight=0.5,
        top_k=2,
    )

    assert result.base_scores.shape == (2, 3)
    assert result.reranked_scores.shape == (2, 3)
    assert result.video_ids == ("v0", "v1", "v2")
    assert result.query_ids == ("v0#0", "v1#0")
    assert result.base_metrics.r1 == pytest.approx(100.0)
    assert result.reranked_metrics.r1 == pytest.approx(0.0)

    # Rank-slot reranking must preserve the multiset of each full score row. Only
    # which shortlisted video occupies the high score slot is allowed to change.
    for base_row, reranked_row in zip(result.base_scores, result.reranked_scores):
        assert np.sort(base_row).tolist() == pytest.approx(np.sort(reranked_row).tolist())


def test_bridge_can_limit_queries_for_local_vlm_smoke_tests():
    result = rerank_dreamprvr_query_loader(
        model=FakeDreamPRVR(),
        query_loader=FakeQueryLoader(),
        context_info=_context(),
        reranker=ReverseShortlistReranker(),
        clip_scale_weight=0.5,
        frame_scale_weight=0.5,
        top_k=2,
        max_queries=1,
    )
    assert result.base_scores.shape == (1, 3)
    assert result.query_ids == ("v0#0",)
