import pytest

torch = pytest.importorskip("torch")

from prvr_agent.retriever import DreamPRVRAdapter


class FakeModel:
    def encode_query(self, query_feat, query_mask):
        return query_feat


def test_retriever_returns_video_scores_without_peak_locations():
    context = {
        "video_metas": ["v0", "v1"],
        "video_proposal_feat": torch.tensor([
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.2, 0.8], [0.9, 0.1]],
        ]),
        "video_feat": torch.tensor([
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.1, 0.9], [0.8, 0.2]],
        ]),
    }
    adapter = DreamPRVRAdapter(FakeModel(), context, clip_scale_weight=0.5, frame_scale_weight=0.5)
    batch = adapter.retrieve(torch.tensor([[1.0, 0.0]]), None, top_k=2)
    candidates = batch.candidates[0]
    assert len(candidates) == 2
    assert candidates[0].base_score >= candidates[1].base_score
    assert not hasattr(candidates[0], "clip_peak_index")
    assert not hasattr(candidates[0], "frame_peak_index")
