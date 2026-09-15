import pytest

from prvr_agent.agents.world_model import MAX_EVENT_WORLDS, RuleBasedEventWorldPlanner, _validate_query_anchors
from prvr_agent.schemas import (
    AtomicEvent,
    EventWorld,
    EventWorldSet,
    QueryHypothesisGraph,
    TemporalConstraint,
)


def graph():
    return QueryHypothesisGraph(
        query="person opens a door",
        atomic_events=[AtomicEvent(id="E1", subject="person", action="opens", object="door")],
        positive_hypothesis="person opens a door",
    )


def relation_graph():
    return QueryHypothesisGraph(
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


def test_rule_worlds_preserve_query_anchor():
    worlds = RuleBasedEventWorldPlanner().imagine(graph(), num_worlds=3)
    assert len(worlds.worlds) == 3
    assert all(world.query_anchor_event_ids == ["E1"] for world in worlds.worlds)
    assert all(world.query_anchor_relation_ids == [] for world in worlds.worlds)


def test_rule_worlds_preserve_relation_anchor():
    worlds = RuleBasedEventWorldPlanner().imagine(relation_graph(), num_worlds=2)
    assert all(world.query_anchor_event_ids == ["E1", "E2"] for world in worlds.worlds)
    assert all(world.query_anchor_relation_ids == ["T1"] for world in worlds.worlds)


def test_anchor_drift_is_rejected():
    bad = EventWorldSet(
        query="person opens a door",
        worlds=[EventWorld(id="H1", query_anchor_event_ids=["E2"], prior=1.0)],
    )
    with pytest.raises(ValueError):
        _validate_query_anchors(graph(), bad)


def test_relation_anchor_drift_is_rejected():
    bad = EventWorldSet(
        query=relation_graph().query,
        worlds=[
            EventWorld(
                id="H1",
                query_anchor_event_ids=["E1", "E2"],
                query_anchor_relation_ids=[],
                prior=1.0,
            )
        ],
    )
    with pytest.raises(ValueError):
        _validate_query_anchors(relation_graph(), bad)


def test_duplicate_anchor_ids_are_rejected():
    with pytest.raises(ValueError):
        EventWorld(id="H1", query_anchor_event_ids=["E1", "E1"], prior=1.0)
    with pytest.raises(ValueError):
        EventWorld(
            id="H1",
            query_anchor_event_ids=["E1"],
            query_anchor_relation_ids=["T1", "T1"],
            prior=1.0,
        )


def test_world_count_is_bounded():
    with pytest.raises(ValueError):
        RuleBasedEventWorldPlanner().imagine(graph(), num_worlds=MAX_EVENT_WORLDS + 1)
