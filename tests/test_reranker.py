from prvr_agent.reranker import ScoreFusionConfig, VerificationScore, fuse_candidate_score


def test_contradiction_lowers_score():
    cfg = ScoreFusionConfig()
    good = VerificationScore(1, 1, 1, 1, 0, 0)
    bad = VerificationScore(1, 1, 1, 1, 1, 0)
    assert fuse_candidate_score(0.5, good, cfg) > fuse_candidate_score(0.5, bad, cfg)
