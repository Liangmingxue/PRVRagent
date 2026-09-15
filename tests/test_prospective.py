from prvr_agent.prospective import ProspectiveConfig, revise_world_beliefs
from prvr_agent.schemas import (
    AtomicEvent,
    EventWorld,
    EventWorldSet,
    QueryHypothesisGraph,
    TemporalConstraint,
    WorldEvidence,
    WorldEvidenceBundle,
)


def _graph():
    return QueryHypothesisGraph(
        query="person opens a door",
        atomic_events=[AtomicEvent(id="E1", subject="person", action="opens", object="door")],
        positive_hypothesis="person opens a door",
    )


def test_supported_world_gains_posterior():
    worlds = EventWorldSet(
        query="person opens a door",
        worlds=[
            EventWorld(id="H1", query_anchor_event_ids=["E1"], prior=0.5),
            EventWorld(id="H2", query_anchor_event_ids=["E1"], prior=0.5),
        ],
    )
    evidence = WorldEvidenceBundle(
        candidate_video_id="v",
        query_support=0.9,
        query_contradiction=0.0,
        query_uncertainty=0.1,
        verified_event_ids=["E1"],
        evidence=[
            WorldEvidence(world_id="H1", support=0.9, contradiction=0.0, uncertainty=0.1),
            WorldEvidence(world_id="H2", support=0.1, contradiction=0.8, uncertainty=0.1),
        ],
    )
    out = revise_world_beliefs(_graph(), worlds, evidence, ProspectiveConfig())
    by_id = {b.world_id: b for b in out.beliefs}
    assert by_id["H1"].posterior > by_id["H1"].prior
    assert by_id["H2"].posterior < by_id["H2"].prior
    assert out.graph_score > 0
    assert out.world_score > 0


def test_counterfactual_evidence_lowers_graph_score():
    worlds = EventWorldSet(
        query="person opens a door",
        worlds=[EventWorld(id="H1", query_anchor_event_ids=["E1"], prior=1.0)],
    )
    positive = WorldEvidenceBundle(
        candidate_video_id="v1",
        query_support=0.9,
        query_contradiction=0.0,
        query_uncertainty=0.1,
        verified_event_ids=["E1"],
        evidence=[WorldEvidence(world_id="H1", support=0.5, contradiction=0.0)],
    )
    near_miss = WorldEvidenceBundle(
        candidate_video_id="v2",
        query_support=0.4,
        query_contradiction=0.9,
        query_uncertainty=0.1,
        verified_event_ids=["E1"],
        evidence=[WorldEvidence(world_id="H1", support=0.5, contradiction=0.0)],
    )
    cfg = ProspectiveConfig()
    good = revise_world_beliefs(_graph(), worlds, positive, cfg)
    bad = revise_world_beliefs(_graph(), worlds, near_miss, cfg)
    assert good.graph_score > bad.graph_score
    assert bad.world_score < good.world_score


def test_uncertainty_is_neutral_not_negative_evidence():
    worlds = EventWorldSet(
        query="person opens a door",
        worlds=[EventWorld(id="H1", query_anchor_event_ids=["E1"], prior=1.0)],
    )
    unknown = WorldEvidenceBundle(
        candidate_video_id="v",
        query_support=0.0,
        query_contradiction=0.0,
        query_uncertainty=1.0,
        verified_event_ids=[],
        evidence=[
            WorldEvidence(
                world_id="H1",
                support=0.0,
                contradiction=0.0,
                uncertainty=1.0,
            )
        ],
    )
    out = revise_world_beliefs(_graph(), worlds, unknown)
    assert out.graph_score == 0.0
    assert out.world_score == 0.0


def test_temporal_relation_must_be_verified_for_complete_cqhg_support():
    graph = QueryHypothesisGraph(
        query="person closes a door then sits",
        atomic_events=[
            AtomicEvent(id="E1", subject="person", action="closes", object="door"),
            AtomicEvent(id="E2", subject="person", action="sits"),
        ],
        temporal_constraints=[
            TemporalConstraint(id="T1", event_a="E1", relation="before", event_b="E2")
        ],
        positive_hypothesis="person closes a door before sitting",
    )
    worlds = EventWorldSet(
        query=graph.query,
        worlds=[EventWorld(id="H1", query_anchor_event_ids=["E1", "E2"], prior=1.0)],
    )
    missing_relation = WorldEvidenceBundle(
        candidate_video_id="v",
        query_support=1.0,
        query_contradiction=0.0,
        query_uncertainty=0.0,
        verified_event_ids=["E1", "E2"],
        verified_relation_ids=[],
        evidence=[WorldEvidence(world_id="H1", support=0.5, contradiction=0.0)],
    )
    verified_relation = missing_relation.model_copy(update={"verified_relation_ids": ["T1"]})
    weak = revise_world_beliefs(graph, worlds, missing_relation)
    strong = revise_world_beliefs(graph, worlds, verified_relation)
    assert weak.graph_score == 0.0
    assert strong.graph_score == 1.0


def test_world_evidence_ids_must_match():
    worlds = EventWorldSet(
        query="person opens a door",
        worlds=[EventWorld(id="H1", query_anchor_event_ids=["E1"], prior=1.0)],
    )
    evidence = WorldEvidenceBundle(
        candidate_video_id="v",
        query_support=0.5,
        query_contradiction=0.0,
        verified_event_ids=["E1"],
        evidence=[WorldEvidence(world_id="H2", support=0.5, contradiction=0.0)],
    )
    try:
        revise_world_beliefs(_graph(), worlds, evidence)
    except ValueError as exc:
        assert "missing=['H1']" in str(exc)
    else:
        raise AssertionError("expected mismatch to fail")
