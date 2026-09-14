from prvr_agent.agents.hypothesis_planner import RuleBasedHypothesisPlanner
from prvr_agent.pipeline import PeakMappingConfig, PipelineConfig, PRVRAgentReranker
from prvr_agent.schemas import Candidate, EvidenceResult


class FakeBackend:
    def verify(self, *, candidate, mode, **kwargs):
        good = candidate.video_id == "good"
        if mode == "support":
            return EvidenceResult(
                mode=mode,
                matched=good,
                support=0.95 if good else 0.25,
                contradiction=0.0 if good else 0.6,
                entity_consistency=1.0 if good else 0.2,
                temporal_consistency=1.0 if good else 0.2,
                action_completeness=1.0 if good else 0.3,
                uncertainty=0.05,
            )
        return EvidenceResult(
            mode=mode,
            matched=not good,
            support=0.05 if good else 0.9,
            contradiction=0.8 if good else 0.0,
            uncertainty=0.05,
        )


def test_pipeline_can_promote_verified_candidate():
    cfg = PipelineConfig(
        top_k_verify=2,
        peak_mapping=PeakMappingConfig(frame_seconds_per_index=1.0, clip_seconds_per_index=2.0),
    )
    reranker = PRVRAgentReranker(
        RuleBasedHypothesisPlanner(),
        FakeBackend(),
        lambda video_id: f"/{video_id}.mp4",
        cfg=cfg,
    )
    candidates = [
        Candidate(video_id="bad", video_index=0, base_score=0.80, clip_score=0.8, frame_score=0.8, clip_peak_index=5, frame_peak_index=10),
        Candidate(video_id="good", video_index=1, base_score=0.78, clip_score=0.78, frame_score=0.78, clip_peak_index=5, frame_peak_index=10),
    ]
    ranked = reranker.rerank("a person closes a door then sits", candidates)
    assert ranked[0].candidate.video_id == "good"
