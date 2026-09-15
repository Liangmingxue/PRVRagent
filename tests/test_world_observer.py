from prvr_agent.agents.world_observer import aggregate_chunk_evidence
from prvr_agent.prospective import ProspectiveConfig, revise_world_beliefs
from prvr_agent.schemas import (
    AtomicEvent,
    ChunkEvidence,
    ChunkEvidenceBundle,
    EventWorld,
    EventWorldSet,
    QueryHypothesisGraph,
    TemporalConstraint,
    WorldEvidence,
)


def _graph_and_worlds():
    graph = QueryHypothesisGraph(
        query="person closes a door then sits",
        atomic_events=[
            AtomicEvent(id="E1", subject="person", action="closes", object="door"),
            AtomicEvent(id="E2", subject="person", action="sits"),
        ],
        temporal_constraints=[
            TemporalConstraint(id="T1", event_a="E1", relation="before", event_b="E2")
        ],
        positive_hypothesis="the same local event contains closing the door before sitting",
    )
    worlds = EventWorldSet(
        query=graph.query,
        worlds=[
            EventWorld(
                id="H1",
                query_anchor_event_ids=["E1", "E2"],
                query_anchor_relation_ids=["T1"],
                prior=1.0,
            )
        ],
    )
    return graph, worlds


def _chunk(index, start, end, *, events, relations=(), support=0.7, contradiction=0.0, world_support=0.5):
    return ChunkEvidence(
        chunk_index=index,
        start_time=start,
        end_time=end,
        query_support=support,
        query_contradiction=contradiction,
        query_uncertainty=0.05,
        verified_event_ids=list(events),
        verified_relation_ids=list(relations),
        evidence=[
            WorldEvidence(
                world_id="H1",
                support=world_support,
                contradiction=0.0,
                uncertainty=0.05,
            )
        ],
    )


def test_distant_atomic_events_are_not_stitched_into_complete_query():
    graph, worlds = _graph_and_worlds()
    raw = ChunkEvidenceBundle(
        candidate_video_id="v",
        chunks=[
            _chunk(0, 0.0, 20.0, events=["E1"], support=0.9),
            _chunk(1, 20.0, 40.0, events=["E2"], support=0.9),
        ],
    )
    aggregated = aggregate_chunk_evidence(graph, worlds, raw, context_radius=1)
    assert set(aggregated.verified_event_ids) in ({"E1"}, {"E2"})
    assert aggregated.verified_relation_ids == []
    assessment = revise_world_beliefs(graph, worlds, aggregated, ProspectiveConfig())
    assert assessment.graph_score == 0.0


def test_coherent_complete_chunk_becomes_hard_query_anchor():
    graph, worlds = _graph_and_worlds()
    raw = ChunkEvidenceBundle(
        candidate_video_id="v",
        chunks=[
            _chunk(0, 0.0, 20.0, events=["E1"], support=0.8),
            _chunk(
                1,
                15.0,
                35.0,
                events=["E1", "E2"],
                relations=["T1"],
                support=0.95,
                world_support=0.9,
            ),
            _chunk(2, 30.0, 50.0, events=["E2"], support=0.8),
        ],
    )
    aggregated = aggregate_chunk_evidence(graph, worlds, raw, context_radius=1)
    assert aggregated.anchor_chunk_index == 1
    assert aggregated.verified_event_ids == ["E1", "E2"]
    assert aggregated.verified_relation_ids == ["T1"]
    assessment = revise_world_beliefs(graph, worlds, aggregated)
    assert assessment.graph_score > 0.8


def test_distant_world_context_is_not_allowed_to_drive_local_revision():
    graph, worlds = _graph_and_worlds()
    raw = ChunkEvidenceBundle(
        candidate_video_id="v",
        chunks=[
            _chunk(
                0,
                0.0,
                20.0,
                events=["E1", "E2"],
                relations=["T1"],
                support=0.95,
                world_support=0.2,
            ),
            _chunk(1, 15.0, 35.0, events=["E1"], support=0.2, world_support=0.2),
            _chunk(2, 30.0, 50.0, events=["E2"], support=0.2, world_support=1.0),
        ],
    )
    aggregated = aggregate_chunk_evidence(graph, worlds, raw, context_radius=1)
    assert aggregated.anchor_chunk_index == 0
    assert aggregated.evidence[0].support < 0.5
