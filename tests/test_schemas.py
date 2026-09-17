import pytest

from prvr_agent.schemas import (
    AtomicEvent,
    CounterfactualHypothesis,
    QueryHypothesisGraph,
    TemporalConstraint,
)


def test_graph_rejects_unknown_event_reference():
    with pytest.raises(ValueError):
        QueryHypothesisGraph(
            query="x",
            atomic_events=[AtomicEvent(id="E1", subject="p", action="a")],
            temporal_constraints=[
                TemporalConstraint(id="T1", event_a="E1", relation="before", event_b="E2")
            ],
            positive_hypothesis="x",
        )


def test_graph_rejects_duplicate_relation_and_counterfactual_ids():
    events = [
        AtomicEvent(id="E1", subject="p", action="a"),
        AtomicEvent(id="E2", subject="p", action="b"),
    ]
    with pytest.raises(ValueError):
        QueryHypothesisGraph(
            query="x",
            atomic_events=events,
            temporal_constraints=[
                TemporalConstraint(id="T1", event_a="E1", relation="before", event_b="E2"),
                TemporalConstraint(id="T1", event_a="E2", relation="after", event_b="E1"),
            ],
            positive_hypothesis="x",
        )

    with pytest.raises(ValueError):
        QueryHypothesisGraph(
            query="x",
            atomic_events=events,
            positive_hypothesis="x",
            counterfactuals=[
                CounterfactualHypothesis(id="CF1", type="partial_event", description="a"),
                CounterfactualHypothesis(id="CF1", type="wrong_action", description="b"),
            ],
        )


def test_graph_rejects_empty_query_and_extra_fields():
    with pytest.raises(ValueError):
        QueryHypothesisGraph(
            query="",
            atomic_events=[AtomicEvent(id="E1", subject="p", action="a")],
            positive_hypothesis="x",
        )

    with pytest.raises(ValueError):
        AtomicEvent(id="E1", subject="p", action="a", unexpected="ignored-no-more")
