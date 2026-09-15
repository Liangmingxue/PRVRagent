import pytest

from prvr_agent.agents.world_model import MAX_EVENT_WORLDS, RuleBasedEventWorldPlanner, _validate_query_anchors
from prvr_agent.schemas import AtomicEvent, EventWorld, EventWorldSet, QueryHypothesisGraph


def graph():
    return QueryHypothesisGraph(
        query="person opens a door",
        atomic_events=[AtomicEvent(id="E1", subject="person", action="opens", object="door")],
        positive_hypothesis="person opens a door",
    )


def test_rule_worlds_preserve_query_anchor():
    worlds = RuleBasedEventWorldPlanner().imagine(graph(), num_worlds=3)
    assert len(worlds.worlds) == 3
    assert all(world.query_anchor_event_ids == ["E1"] for world in worlds.worlds)


def test_anchor_drift_is_rejected():
    bad = EventWorldSet(
        query="person opens a door",
        worlds=[EventWorld(id="H1", query_anchor_event_ids=["E2"], prior=1.0)],
    )
    with pytest.raises(ValueError):
        _validate_query_anchors(graph(), bad)


def test_duplicate_anchor_ids_are_rejected():
    with pytest.raises(ValueError):
        EventWorld(id="H1", query_anchor_event_ids=["E1", "E1"], prior=1.0)


def test_world_count_is_bounded():
    with pytest.raises(ValueError):
        RuleBasedEventWorldPlanner().imagine(graph(), num_worlds=MAX_EVENT_WORLDS + 1)
