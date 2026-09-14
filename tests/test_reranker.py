from prvr_agent.reranker import (
    ScoreFusionConfig,
    VerificationScore,
    aggregate_evidence,
    fuse_candidate_score,
    summarize_evidence,
)
from prvr_agent.schemas import EvidenceResult


def _e(mode: str, support: float, contradiction: float, **kwargs) -> EvidenceResult:
    return EvidenceResult(
        mode=mode,
        matched=True,
        support=support,
        contradiction=contradiction,
        entity_consistency=kwargs.get("entity_consistency", 1.0),
        temporal_consistency=kwargs.get("temporal_consistency", 1.0),
        action_completeness=kwargs.get("action_completeness", 1.0),
        uncertainty=kwargs.get("uncertainty", 0.1),
        verified_event_ids=kwargs.get("verified_event_ids", []),
        verified_relation_ids=kwargs.get("verified_relation_ids", []),
    )


def test_contradiction_lowers_score():
    cfg = ScoreFusionConfig()
    good = VerificationScore(1, 1, 1, 1, 0, 0)
    bad = VerificationScore(1, 1, 1, 1, 1, 0)
    assert fuse_candidate_score(0.5, good, cfg) > fuse_candidate_score(0.5, bad, cfg)


def test_refute_contradiction_is_not_query_contradiction():
    score = summarize_evidence(_e("support", 0.8, 0.0), _e("refute", 0.1, 0.9))
    assert score.contradiction == 0.1


def test_aggregate_does_not_frankenstein_components_across_rounds():
    supports = [
        _e("support", 1.0, 0.0, temporal_consistency=0.0, entity_consistency=0.0, action_completeness=0.4),
        _e("support", 0.4, 0.0, temporal_consistency=1.0, entity_consistency=1.0, action_completeness=1.0),
    ]
    refutes = [_e("refute", 0.0, 0.0), _e("refute", 0.0, 0.0)]
    score = aggregate_evidence(supports, refutes)
    assert not (score.atomic == 1.0 and score.temporal == 1.0 and score.identity == 1.0 and score.completeness == 1.0)


def test_event_and_relation_coverage_caps_confidence():
    support = _e(
        "support",
        0.95,
        0.0,
        verified_event_ids=["E1"],
        verified_relation_ids=[],
        action_completeness=0.95,
        temporal_consistency=0.95,
    )
    score = summarize_evidence(
        support,
        _e("refute", 0.0, 0.0),
        expected_event_ids={"E1", "E2"},
        expected_temporal_ids={"T1"},
        expected_identity_ids=set(),
    )
    assert score.atomic == 0.5
    assert score.completeness == 0.5
    assert score.temporal == 0.0
    assert score.identity == 1.0
