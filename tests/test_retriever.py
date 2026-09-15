import pytest

torch = pytest.importorskip("torch")

from prvr_agent.retriever import DreamPRVRAdapter


class FakeModel:
    def encode_query(self, query_feat, query_mask):
        return query_feat


def test_retriever_returns_scores_without_peak_fields():
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
    batch = DreamPRVRAdapter(FakeModel(), context).retrieve(torch.tensor([[1.0, 0.0]]), None, top_k=2)
    candidate = batch.candidates[0][0]
    assert not hasattr(candidate, "clip_peak_index")
    assert not hasattr(candidate, "frame_peak_index")
