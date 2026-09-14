from prvr_agent.controller import CandidateEvidenceState, VerificationBudgetConfig, decide_next_action
from prvr_agent.reranker import VerificationScore
from prvr_agent.schemas import Candidate


def _candidate():
    return Candidate(video_id="v1", video_index=0, base_score=0.9, clip_score=0.9, frame_score=0.9, clip_peak_index=10, frame_peak_index=10)


def test_controller_can_early_stop():
    state = CandidateEvidenceState(candidate=_candidate(), round_idx=1, retrieval_margin=1.0)
    score = VerificationScore(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    d = decide_next_action(state, score, VerificationBudgetConfig())
    assert d.action == "stop"
