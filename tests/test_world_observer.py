from prvr_agent.agents.world_observer import (
    aggregate_chunk_evidence,
    chunk_refinement_priority,
    select_confirmation_span_indices,
    select_refinement_chunk_indices,
)
from prvr_agent.prospective import ProspectiveConfig, revise_world_beliefs
from prvr_agent.schemas import (
    AtomicEvent,
    ChunkEvidence,
    ChunkEvidenceBundle,
    EventTimeRange,
    EventWorld,
    EventWorldSet,
    QueryHypothesisGraph,
    TemporalConstraint,
    TemporalSegmentTrace,
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


def _chunk(
    index,
    start,
    end,
    *,
    events,
    relations=(),
    support=0.7,
    contradiction=0.0,
    uncertainty=0.05,
    world_support=0.5,
    world_uncertainty=0.05,
):
    ranges = []
    if "E1" in events:
        ranges.append(EventTimeRange(event_id="E1", start_time=start + 1.0, end_time=start + 2.0))
    if "E2" in events:
        ranges.append(EventTimeRange(event_id="E2", start_time=max(start + 3.0, end - 2.0), end_time=max(start + 3.5, end - 1.0)))
    return ChunkEvidence(
        chunk_index=index,
        start_time=start,
        end_time=end,
        query_support=support,
        query_contradiction=contradiction,
        query_uncertainty=uncertainty,
        verified_event_ids=list(events),
        verified_relation_ids=list(relations),
        event_time_ranges=ranges,
        evidence=[
            WorldEvidence(
                world_id="H1",
                support=world_support,
                contradiction=0.0,
                uncertainty=world_uncertainty,
            )
        ],
    )


def test_distant_atomic_events_are_not_stitched_into_complete_query():
    graph, worlds = _graph_and_worlds()
    raw = ChunkEvidenceBundle(
        candidate_video_id="v",
        chunks=[
            _chunk(0, 0.0, 20.0, events=["E1"], support=0.9),
            _chunk(1, 80.0, 100.0, events=["E2"], support=0.9),
        ],
    )
    aggregated = aggregate_chunk_evidence(graph, worlds, raw, context_radius=1)
    assert set(aggregated.verified_event_ids) in ({"E1"}, {"E2"})
    assert aggregated.verified_relation_ids == []
    assessment = revise_world_beliefs(graph, worlds, aggregated, ProspectiveConfig())
    assert assessment.graph_score == 0.0


def test_coherent_complete_chunk_becomes_hard_query_anchor_without_confirmation():
    graph, worlds = _graph_and_worlds()
    raw = ChunkEvidenceBundle(
        candidate_video_id="v",
        chunks=[
            _chunk(0, 0.0, 20.0, events=["E1"], support=0.8),
            _chunk(
                1,
                20.0,
                40.0,
                events=["E1", "E2"],
                relations=["T1"],
                support=0.95,
                world_support=0.9,
            ),
            _chunk(2, 40.0, 60.0, events=["E2"], support=0.8),
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
            _chunk(1, 20.0, 40.0, events=["E1"], support=0.2, world_support=0.2),
            _chunk(2, 40.0, 60.0, events=["E2"], support=0.2, world_support=1.0),
        ],
    )
    aggregated = aggregate_chunk_evidence(graph, worlds, raw, context_radius=1)
    assert aggregated.anchor_chunk_index == 0
    assert aggregated.evidence[0].support < 0.5


def test_partial_query_segment_is_prioritized_for_dense_refinement():
    graph, _ = _graph_and_worlds()
    partial = _chunk(
        1,
        20.0,
        40.0,
        events=["E1"],
        support=0.55,
        uncertainty=0.55,
        world_support=0.65,
        world_uncertainty=0.35,
    )
    irrelevant = _chunk(
        0,
        0.0,
        20.0,
        events=[],
        support=0.05,
        uncertainty=0.10,
        world_support=0.05,
        world_uncertainty=0.10,
    )
    assert chunk_refinement_priority(graph, partial) > chunk_refinement_priority(graph, irrelevant)
    selected = select_refinement_chunk_indices(graph, [irrelevant, partial], max_chunks=1, threshold=0.30)
    assert selected == [1]


def test_visual_sidekick_can_refine_short_event_even_when_qwen_is_overconfident():
    graph, _ = _graph_and_worlds()
    missed = _chunk(
        3,
        60.0,
        80.0,
        events=[],
        support=0.0,
        uncertainty=0.05,
        world_support=0.0,
        world_uncertainty=0.05,
    )
    assert chunk_refinement_priority(graph, missed, visual_salience=1.0) >= 0.35
    selected = select_refinement_chunk_indices(
        graph,
        [missed],
        visual_salience_by_index={3: 1.0},
        max_chunks=1,
        threshold=0.30,
    )
    assert selected == [3]


def test_high_uncertainty_no_evidence_segments_are_refined_with_temporal_diversity():
    graph, _ = _graph_and_worlds()
    chunks = [
        _chunk(
            idx,
            idx * 20.0,
            idx * 20.0 + 20.0,
            events=[],
            support=0.0,
            uncertainty=0.95,
            world_support=0.0,
            world_uncertainty=0.95,
        )
        for idx in range(6)
    ]
    selected = select_refinement_chunk_indices(
        graph,
        chunks,
        visual_salience_by_index={idx: 0.5 for idx in range(6)},
        max_chunks=3,
        threshold=0.30,
    )
    assert len(selected) == 3
    assert 0 in selected
    assert 5 in selected


def test_confirmation_span_can_join_adjacent_partial_events_but_not_distant_ones():
    graph, _ = _graph_and_worlds()
    adjacent = [
        _chunk(0, 0.0, 10.0, events=["E1"], support=0.7),
        _chunk(1, 10.0, 20.0, events=["E2"], support=0.7),
        _chunk(2, 20.0, 30.0, events=[], support=0.1),
    ]
    assert select_confirmation_span_indices(
        graph,
        adjacent,
        max_segments=2,
        max_span_seconds=25.0,
    ) == [0, 1]

    distant = [
        _chunk(0, 0.0, 10.0, events=["E1"], support=0.7),
        _chunk(1, 80.0, 90.0, events=["E2"], support=0.7),
    ]
    selected = select_confirmation_span_indices(
        graph,
        distant,
        max_segments=2,
        max_span_seconds=25.0,
    )
    assert len(selected) == 1


def test_isolated_confirmation_is_the_only_source_of_final_hard_evidence():
    graph, worlds = _graph_and_worlds()
    raw = ChunkEvidenceBundle(
        candidate_video_id="v",
        chunks=[
            _chunk(0, 0.0, 10.0, events=["E1"], support=0.7),
            _chunk(1, 10.0, 20.0, events=["E2"], support=0.7),
        ],
    )
    confirmed = _chunk(
        0,
        0.0,
        20.0,
        events=["E1", "E2"],
        relations=["T1"],
        support=0.96,
        world_support=0.85,
    )
    aggregated = aggregate_chunk_evidence(
        graph,
        worlds,
        raw,
        refined_chunk_indices=[0],
        confirmed_evidence=confirmed,
        confirmed_chunk_indices=[0, 1],
        segment_trace=[
            TemporalSegmentTrace(segment_index=0, start_time=0.0, end_time=10.0, visual_salience=0.8),
            TemporalSegmentTrace(segment_index=1, start_time=10.0, end_time=20.0, visual_salience=0.7),
        ],
    )
    assert aggregated.verified_event_ids == ["E1", "E2"]
    assert aggregated.verified_relation_ids == ["T1"]
    assert aggregated.confirmed_chunk_indices == [0, 1]
    assert aggregated.confirmation_start_time == 0.0
    assert aggregated.confirmation_end_time == 20.0
    assert aggregated.evidence[0].support == 0.85
