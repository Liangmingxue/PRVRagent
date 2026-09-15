from prvr_agent.prospective import ProspectiveConfig, revise_world_beliefs
from prvr_agent.schemas import EventWorld, EventWorldSet, WorldEvidence, WorldEvidenceBundle


def test_supported_world_gains_posterior():
    worlds = EventWorldSet(
        query="q",
        worlds=[
            EventWorld(id="H1", query_anchor_event_ids=["E1"], prior=0.5),
            EventWorld(id="H2", query_anchor_event_ids=["E1"], prior=0.5),
        ],
    )
    evidence = WorldEvidenceBundle(
        candidate_video_id="v",
        evidence=[
            WorldEvidence(world_id="H1", support=0.9, contradiction=0.0, uncertainty=0.1),
            WorldEvidence(world_id="H2", support=0.1, contradiction=0.8, uncertainty=0.1),
        ],
    )
    out = revise_world_beliefs(worlds, evidence, ProspectiveConfig())
    by_id = {b.world_id: b for b in out.beliefs}
    assert by_id["H1"].posterior > by_id["H1"].prior
    assert by_id["H2"].posterior < by_id["H2"].prior
    assert out.world_score > 0


def test_world_evidence_ids_must_match():
    worlds = EventWorldSet(
        query="q",
        worlds=[EventWorld(id="H1", query_anchor_event_ids=["E1"], prior=1.0)],
    )
    evidence = WorldEvidenceBundle(
        candidate_video_id="v",
        evidence=[WorldEvidence(world_id="H2", support=0.5, contradiction=0.0)],
    )
    try:
        revise_world_beliefs(worlds, evidence)
    except ValueError as exc:
        assert "missing=['H1']" in str(exc)
    else:
        raise AssertionError("expected mismatch to fail")
