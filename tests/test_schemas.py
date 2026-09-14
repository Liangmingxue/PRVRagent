import pytest

from prvr_agent.schemas import AtomicEvent, QueryHypothesisGraph, TemporalConstraint


def test_graph_rejects_unknown_event_reference():
    with pytest.raises(ValueError):
        QueryHypothesisGraph(
            query="x",
            atomic_events=[AtomicEvent(id="E1", subject="p", action="a")],
            temporal_constraints=[TemporalConstraint(event_a="E1", relation="before", event_b="E2")],
            positive_hypothesis="x",
        )
