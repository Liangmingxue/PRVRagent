from prvr_agent.reranker import ScoreFusionConfig, VerificationScore, fuse_candidate_score, summarize_evidence
from prvr_agent.schemas import EvidenceResult


def _e(mode: str, support: float, contradiction: float) -> EvidenceResult:
    return EvidenceResult(
        mode=mode,
        matched=True,
        support=support,
        contradiction=contradiction,
        entity_consistency=1.0,
        temporal_consistency=1.0,
        action_completeness=1.0,
        uncertainty=0.1,
    )


def test_contradiction_lowers_score():
    cfg = ScoreFusionConfig()
    good = VerificationScore(1, 1, 1, 1, 0, 0)
    bad = VerificationScore(1, 1, 1, 1, 1, 0)
    assert fuse_candidate_score(0.5, good, cfg) > fuse_candidate_score(0.5, bad, cfg)


def test_refute_contradiction_is_not_query_contradiction():
    score = summarize_evidence(_e("support", 0.8, 0.0), _e("refute", 0.1, 0.9))
    assert score.contradiction == 0.1
