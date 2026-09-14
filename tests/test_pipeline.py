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
                verified_event_ids=["E1", "E2"] if good else ["E1"],
                verified_relation_ids=["T1"] if good else [],
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
        peak_mapping=PeakMappingConfig(
            mode="fixed_stride", frame_seconds_per_index=1.0, clip_seconds_per_index=2.0
        ),
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


def test_unverified_candidate_uses_same_base_score_scale():
    cfg = PipelineConfig(
        top_k_verify=1,
        peak_mapping=PeakMappingConfig(
            mode="fixed_stride", frame_seconds_per_index=1.0, clip_seconds_per_index=1.0
        ),
    )
    reranker = PRVRAgentReranker(
        RuleBasedHypothesisPlanner(), FakeBackend(), lambda video_id: f"/{video_id}.mp4", cfg=cfg
    )
    candidates = [
        Candidate(video_id="good", video_index=0, base_score=0.9, clip_score=0.9, frame_score=0.9, clip_peak_index=1, frame_peak_index=1),
        Candidate(video_id="other", video_index=1, base_score=0.89, clip_score=0.89, frame_score=0.89, clip_peak_index=1, frame_peak_index=1),
    ]
    ranked = reranker.rerank("person walks", candidates)
    other = next(item for item in ranked if item.candidate.video_id == "other")
    assert other.final_score == cfg.fusion.base_weight * 0.89


class DurationBackend(FakeBackend):
    def video_duration(self, video_path):
        return 100.0


def test_relative_peak_mapping_uses_per_video_duration_and_metadata():
    cfg = PipelineConfig(top_k_verify=1, peak_mapping=PeakMappingConfig(mode="relative"))
    reranker = PRVRAgentReranker(
        RuleBasedHypothesisPlanner(), DurationBackend(), lambda video_id: f"/{video_id}.mp4", cfg=cfg
    )
    candidate = Candidate(
        video_id="good", video_index=0, base_score=0.9, clip_score=0.9, frame_score=0.9,
        clip_peak_index=15, frame_peak_index=63,
        metadata={"clip_num_locations": 32, "frame_valid_locations": 128},
    )
    clip_t, frame_t, duration = reranker._peak_times(candidate, "/good.mp4")
    assert duration == 100.0
    assert abs(clip_t - (15.5 / 32 * 100)) < 1e-6
    assert abs(frame_t - (63.5 / 128 * 100)) < 1e-6
