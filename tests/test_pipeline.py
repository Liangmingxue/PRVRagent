from prvr_agent.agents.hypothesis_planner import RuleBasedHypothesisPlanner
from prvr_agent.agents.world_model import RuleBasedEventWorldPlanner
from prvr_agent.pipeline import PipelineConfig, PRVRAgentReranker
from prvr_agent.prospective import ProspectiveConfig
from prvr_agent.schemas import Candidate, WorldEvidence, WorldEvidenceBundle


class FakeWorldBackend:
    def assess(self, *, candidate, worlds, **kwargs):
        good = candidate.video_id == "good"
        evidence = []
        for world in worlds.worlds:
            evidence.append(
                WorldEvidence(
                    world_id=world.id,
                    support=0.95 if good else 0.05,
                    contradiction=0.0 if good else 0.9,
                    uncertainty=0.05,
                )
            )
        return WorldEvidenceBundle(candidate_video_id=candidate.video_id, evidence=evidence)


def test_pipeline_promotes_candidate_supported_by_prospective_worlds():
    cfg = PipelineConfig(
        num_worlds=3,
        coarse_frames=8,
        scoring=ProspectiveConfig(base_weight=0.5, world_weight=0.5),
    )
    reranker = PRVRAgentReranker(
        RuleBasedHypothesisPlanner(),
        RuleBasedEventWorldPlanner(),
        FakeWorldBackend(),
        lambda video_id: f"/{video_id}.mp4",
        cfg=cfg,
    )
    candidates = [
        Candidate(video_id="bad", video_index=0, base_score=0.80, clip_score=0.8, frame_score=0.8),
        Candidate(video_id="good", video_index=1, base_score=0.78, clip_score=0.78, frame_score=0.78),
    ]
    ranked = reranker.rerank("a person closes a door then sits", candidates)
    assert ranked[0].candidate.video_id == "good"
