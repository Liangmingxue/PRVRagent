from prvr_agent.agents.hypothesis_planner import RuleBasedHypothesisPlanner
from prvr_agent.agents.world_model import RuleBasedEventWorldPlanner
from prvr_agent.pipeline import PipelineConfig, PRVRAgentReranker
from prvr_agent.prospective import ProspectiveConfig
from prvr_agent.schemas import Candidate, WorldEvidence, WorldEvidenceBundle


class FakeWorldBackend:
    def assess(self, *, candidate, worlds, graph, **kwargs):
        good = candidate.video_id == "good"
        evidence = [
            WorldEvidence(
                world_id=world.id,
                support=0.95 if good else 0.05,
                contradiction=0.0 if good else 0.9,
                uncertainty=0.05,
            )
            for world in worlds.worlds
        ]
        relation_ids = [
            rel.id for rel in list(graph.temporal_constraints) + list(graph.identity_constraints)
        ]
        return WorldEvidenceBundle(
            candidate_video_id=candidate.video_id,
            query_support=0.95 if good else 0.2,
            query_contradiction=0.0 if good else 0.9,
            query_uncertainty=0.05,
            verified_event_ids=[event.id for event in graph.atomic_events]
            if good
            else [graph.atomic_events[0].id],
            verified_relation_ids=relation_ids if good else [],
            supported_counterfactual_ids=[]
            if good
            else [graph.counterfactuals[0].id],
            evidence=evidence,
        )


def test_pipeline_promotes_candidate_supported_by_cqhg_and_worlds():
    cfg = PipelineConfig(
        num_worlds=3,
        frames_per_chunk=4,
        target_chunk_seconds=20.0,
        max_chunks=16,
        chunks_per_request=8,
        refinement_frames_per_chunk=12,
        max_refinement_chunks=3,
        scoring=ProspectiveConfig(base_weight=0.5, graph_weight=0.25, world_weight=0.25),
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
    assert ranked[0].graph_score > ranked[1].graph_score
    assert ranked[0].world_score > ranked[1].world_score


def test_pipeline_rejects_excessive_coarse_request_frame_budget():
    try:
        PipelineConfig(frames_per_chunk=9, chunks_per_request=8).validate()
    except ValueError as exc:
        assert "must not exceed" in str(exc)
    else:
        raise AssertionError("expected excessive visual frame budget to fail")


def test_pipeline_rejects_refinement_that_is_not_denser_than_coarse_pass():
    try:
        PipelineConfig(frames_per_chunk=8, refinement_frames_per_chunk=8).validate()
    except ValueError as exc:
        assert "must exceed" in str(exc)
    else:
        raise AssertionError("expected non-dense refinement to fail")


def test_pipeline_allows_multiple_refinements_because_each_is_isolated():
    # Refinement requests are intentionally sent one segment at a time, so the
    # total refinement budget is not a per-request image-budget violation.
    PipelineConfig(refinement_frames_per_chunk=16, max_refinement_chunks=5).validate()


def test_pipeline_rejects_invalid_event_sidekick_and_confirmation_settings():
    try:
        PipelineConfig(sidekick_scan_fps=0.0).validate()
    except ValueError as exc:
        assert "sidekick_scan_fps" in str(exc)
    else:
        raise AssertionError("expected invalid sidekick rate to fail")

    try:
        PipelineConfig(event_min_seconds=25.0, target_chunk_seconds=20.0).validate()
    except ValueError as exc:
        assert "event_min_seconds" in str(exc)
    else:
        raise AssertionError("expected invalid event duration bounds to fail")

    try:
        PipelineConfig(confirmation_max_segments=4).validate()
    except ValueError as exc:
        assert "confirmation_max_segments" in str(exc)
    else:
        raise AssertionError("expected oversized confirmation span to fail")
