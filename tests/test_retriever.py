import pytest

torch = pytest.importorskip("torch")

from prvr_agent.retriever import DreamPRVRAdapter
from prvr_agent.retriever.dreamprvr_adapter import SEMANTIC_SIDEKICK_METADATA_KEY


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
        "video_mask": torch.tensor([
            [1.0, 1.0],
            [1.0, 1.0],
        ]),
    }


def test_retriever_returns_scores_without_peak_fields_and_uses_eval():
    model = FakeModel()
    batch = DreamPRVRAdapter(model, _context()).retrieve(torch.tensor([[1.0, 0.0]]), None, top_k=2)
    candidate = batch.candidates[0][0]
    assert not hasattr(candidate, "clip_peak_index")
    assert not hasattr(candidate, "frame_peak_index")
    assert SEMANTIC_SIDEKICK_METADATA_KEY in candidate.metadata
    assert model.seen_training_state is False
    assert model.training is True


def test_semantic_sidekick_detects_feature_transition_without_query_peak():
    features = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    scores = DreamPRVRAdapter._semantic_change_curve(features, radius=2)
    assert len(scores) == 4
    assert scores[0] == 0.0
    assert max(scores[1:]) > 0.9


def test_semantic_sidekick_exports_global_kernel_boundary_without_query_peak():
    context = {
        "video_metas": ["v0"],
        "video_proposal_feat": torch.tensor([[[1.0, 0.0]]]),
        "video_feat": torch.tensor([[
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]]),
        "video_mask": torch.ones(1, 8),
    }
    candidate = DreamPRVRAdapter(FakeModel(), context).retrieve(
        torch.tensor([[1.0, 0.0]]), None, top_k=1
    ).candidates[0][0]
    payload = candidate.metadata[SEMANTIC_SIDEKICK_METADATA_KEY]
    assert payload["kernel_boundary_fractions"] == [0.5]
    assert payload["kernel_selected_change_points"] == 1
    assert len(payload["local_change_scores"]) == 8
    assert payload["change_scores"][4] >= 2.0


def test_semantic_sidekick_mask_excludes_padded_tail():
    context = {
        "video_metas": ["v0"],
        "video_proposal_feat": torch.tensor([[[1.0, 0.0]]]),
        "video_feat": torch.tensor([[
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [100.0, 100.0],
            [-100.0, -100.0],
        ]]),
        "video_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]),
    }
    adapter = DreamPRVRAdapter(FakeModel(), context)
    candidate = adapter.retrieve(torch.tensor([[1.0, 0.0]]), None, top_k=1).candidates[0][0]
    payload = candidate.metadata[SEMANTIC_SIDEKICK_METADATA_KEY]
    assert payload["valid_length"] == 4
    assert len(payload["change_scores"]) == 4
    assert len(payload["local_change_scores"]) == 4


def test_semantic_sidekick_rejects_sparse_mask_that_breaks_chronology():
    context = {
        "video_metas": ["v0"],
        "video_proposal_feat": torch.tensor([[[1.0, 0.0]]]),
        "video_feat": torch.tensor([[
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [1.0, 1.0],
        ]]),
        "video_mask": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
    }
    adapter = DreamPRVRAdapter(FakeModel(), context)
    with pytest.raises(ValueError, match="contiguous prefix"):
        adapter.retrieve(torch.tensor([[1.0, 0.0]]), None, top_k=1)


def test_custom_context_without_mask_falls_back_without_semantic_trace():
    context = _context()
    context.pop("video_mask")
    candidate = DreamPRVRAdapter(FakeModel(), context).retrieve(
        torch.tensor([[1.0, 0.0]]), None, top_k=1
    ).candidates[0][0]
    assert SEMANTIC_SIDEKICK_METADATA_KEY not in candidate.metadata


def test_retriever_rejects_video_count_mismatch():
    with pytest.raises(ValueError):
        DreamPRVRAdapter(FakeModel(), _context(), video_ids=["only-one"])


def test_retriever_rejects_invalid_fusion_weights():
    with pytest.raises(ValueError):
        DreamPRVRAdapter(FakeModel(), _context(), clip_scale_weight=-1.0, frame_scale_weight=1.0)


def test_retriever_rejects_non_integer_top_k():
    adapter = DreamPRVRAdapter(FakeModel(), _context())
    for top_k in (True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            adapter.retrieve(torch.tensor([[1.0, 0.0]]), None, top_k=top_k)


def test_retriever_rejects_bad_video_mask_shape():
    context = _context()
    context["video_mask"] = torch.ones(2, 3)
    with pytest.raises(ValueError, match="video_mask"):
        DreamPRVRAdapter(FakeModel(), context)
