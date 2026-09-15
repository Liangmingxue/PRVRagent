import pytest

torch = pytest.importorskip("torch")

from prvr_agent.retriever import DreamPRVRAdapter


class FakeModel:
    def __init__(self):
        self.training = True
        self.seen_training_state = None

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def encode_query(self, query_feat, query_mask):
        self.seen_training_state = self.training
        return query_feat


def _context():
    return {
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


def test_retriever_returns_scores_without_peak_fields_and_uses_eval():
    model = FakeModel()
    batch = DreamPRVRAdapter(model, _context()).retrieve(torch.tensor([[1.0, 0.0]]), None, top_k=2)
    candidate = batch.candidates[0][0]
    assert not hasattr(candidate, "clip_peak_index")
    assert not hasattr(candidate, "frame_peak_index")
    assert model.seen_training_state is False
    assert model.training is True


def test_retriever_rejects_video_count_mismatch():
    with pytest.raises(ValueError):
        DreamPRVRAdapter(FakeModel(), _context(), video_ids=["only-one"])


def test_retriever_rejects_invalid_fusion_weights():
    with pytest.raises(ValueError):
        DreamPRVRAdapter(FakeModel(), _context(), clip_scale_weight=-1.0, frame_scale_weight=1.0)
