import pytest

from prvr_agent.schemas import (
    AtomicEvent,
    ProspectiveEventWorld,
    ProspectiveWorldSet,
    QueryHypothesisGraph,
    TemporalConstraint,
)


def test_graph_rejects_unknown_event_reference():
    with pytest.raises(ValueError):
        QueryHypothesisGraph(
            query="x",
            atomic_events=[AtomicEvent(id="E1", subject="p", action="a")],
            temporal_constraints=[TemporalConstraint(id="T1", event_a="E1", relation="before", event_b="E2")],
            positive_hypothesis="x",
        )


def test_world_set_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        ProspectiveWorldSet(
            query="x",
            worlds=[
                ProspectiveEventWorld(id="W1", query_anchor="x", prior=0.5),
                ProspectiveEventWorld(id="W1", query_anchor="x", prior=0.5),
            ],
        )
