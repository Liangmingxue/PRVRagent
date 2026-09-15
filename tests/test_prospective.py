from prvr_agent.prospective import ProspectiveConfig, revise_world_beliefs
from prvr_agent.schemas import (
    AtomicEvent,
    EventWorld,
    EventWorldSet,
    QueryHypothesisGraph,
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
