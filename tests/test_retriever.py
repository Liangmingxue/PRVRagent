import pytest

torch = pytest.importorskip("torch")

from prvr_agent.retriever import DreamPRVRAdapter


class FakeModel:
    def encode_query(self, query_feat, query_mask):
        return query_feat


def test_retriever_preserves_peak_indices():
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
    by_id = {c.video_id: c for c in batch.candidates[0]}
    assert by_id["v0"].clip_peak_index == 0
    assert by_id["v0"].frame_peak_index == 0
    assert by_id["v1"].clip_peak_index == 1
    assert by_id["v1"].frame_peak_index == 1
