from prvr_agent.agents.hypothesis_planner import RuleBasedHypothesisPlanner
from prvr_agent.pipeline import PipelineConfig, PRVRAgentReranker
from prvr_agent.schemas import (
    Candidate,
    CandidateWorldObservation,
    ProspectiveEventWorld,
    ProspectiveWorldSet,
    WorldEvidence,
)


class FakeWorldModeler:
    def imagine(self, graph, *, num_worlds):
        return ProspectiveWorldSet(
            query=graph.query,
            worlds=[
                ProspectiveEventWorld(
                    id="W1",
                    preconditions=["preparation"],
                    query_anchor=graph.positive_hypothesis,
                    consequences=["follow-up"],
                    prior=1.0,
                )
            ],
        )


class FakeObserver:
    def observe(self, *, candidate, **kwargs):
        good = candidate.video_id == "good"
        return CandidateWorldObservation(
            query_satisfaction=0.95 if good else 0.2,
            counterfactual_risk=0.05 if good else 0.9,
            world_evidence=[
                WorldEvidence(
                    world_id="W1",
                    support=0.8 if good else 0.1,
                    contradiction=0.0 if good else 0.8,
                )
            ],
            uncertainty=0.05,
        )


def test_pipeline_can_promote_complete_candidate_without_peak_seeding():
    reranker = PRVRAgentReranker(
        RuleBasedHypothesisPlanner(),
        FakeWorldModeler(),
        FakeObserver(),
        lambda video_id: f"/{video_id}.mp4",
        cfg=PipelineConfig(top_k_reason=2, num_worlds=1, coarse_frames=8),
    )
    candidates = [
        Candidate(video_id="bad", video_index=0, base_score=0.80, clip_score=0.8, frame_score=0.8),
        Candidate(video_id="good", video_index=1, base_score=0.78, clip_score=0.78, frame_score=0.78),
    ]
    ranked = reranker.rerank("a person closes a door then sits", candidates)
    assert ranked[0].candidate.video_id == "good"
    assert ranked[0].world_beliefs[0].posterior == 1.0
