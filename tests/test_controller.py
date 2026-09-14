from prvr_agent.controller import CandidateEvidenceState, VerificationBudgetConfig, decide_next_action
from prvr_agent.reranker import VerificationScore
from prvr_agent.schemas import Candidate


def _candidate():
    return Candidate(video_id="v1", video_index=0, base_score=0.9, clip_score=0.9, frame_score=0.9, clip_peak_index=10, frame_peak_index=10)


def test_controller_can_early_stop_positive():
    state = CandidateEvidenceState(candidate=_candidate(), round_idx=1, retrieval_margin=1.0)
    score = VerificationScore(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    d = decide_next_action(state, score, VerificationBudgetConfig())
    assert d.action == "stop"


def test_controller_can_early_stop_strong_refutation():
    state = CandidateEvidenceState(candidate=_candidate(), round_idx=1, retrieval_margin=0.01, peak_gap_seconds=20.0)
    score = VerificationScore(0.1, 0.1, 0.1, 0.1, 0.95, 0.05)
    d = decide_next_action(state, score, VerificationBudgetConfig())
    assert d.action == "stop"
    assert "counterevidence" in d.reason


def test_negative_stop_waits_for_required_alternative_seed():
    state = CandidateEvidenceState(
        candidate=_candidate(),
        round_idx=1,
        retrieval_margin=0.01,
        peak_gap_seconds=20.0,
        min_rounds_before_negative_stop=2,
    )
    score = VerificationScore(0.1, 0.1, 0.1, 0.1, 0.95, 0.05)
    first = decide_next_action(state, score, VerificationBudgetConfig())
    assert first.action == "expand"

    state.round_idx = 2
    second = decide_next_action(state, score, VerificationBudgetConfig())
    assert second.action == "stop"
    assert "counterevidence" in second.reason
