from __future__ import annotations

import json
from typing import Protocol

from prvr_agent.llm_config import LLMConfig, create_openai_compatible_client
from prvr_agent.schemas import EventWorld, EventWorldSet, QueryHypothesisGraph


class EventWorldPlanner(Protocol):
    def imagine(self, graph: QueryHypothesisGraph, *, num_worlds: int = 3) -> EventWorldSet: ...


def _validate_query_anchors(graph: QueryHypothesisGraph, worlds: EventWorldSet) -> EventWorldSet:
    expected = {event.id for event in graph.atomic_events}
    if worlds.query != graph.query:
        raise ValueError("event-world query must exactly match the CQHG query")
    for world in worlds.worlds:
        anchors = set(world.query_anchor_event_ids)
        if anchors != expected:
            raise ValueError(
                f"world {world.id!r} must preserve every CQHG atomic event id exactly; "
                f"expected={sorted(expected)}, got={sorted(anchors)}"
            )
    return worlds


class RuleBasedEventWorldPlanner:
    """Dependency-free fallback for tests and ablations, not the research model."""

    def imagine(self, graph: QueryHypothesisGraph, *, num_worlds: int = 3) -> EventWorldSet:
        if num_worlds <= 0:
            raise ValueError("num_worlds must be positive")
        anchor_ids = [event.id for event in graph.atomic_events]
        templates = [
            (
                ["The participants and objects needed by the query are already present."],
                ["The queried event produces an immediately observable post-event state."],
            ),
            (
                ["The queried event starts with little visible lead-in context."],
                ["The video continues with an activity compatible with the queried event."],
            ),
            (
                ["A plausible preparation step establishes the state required by the queried event."],
                ["A plausible follow-up step changes or uses the state created by the queried event."],
            ),
        ]
        prior = 1.0 / num_worlds
        worlds = []
        for idx in range(num_worlds):
            preconditions, consequences = templates[idx % len(templates)]
            worlds.append(
                EventWorld(
                    id=f"H{idx + 1}",
                    preconditions=list(preconditions),
                    query_anchor_event_ids=anchor_ids,
                    consequences=list(consequences),
                    prior=prior,
                    rationale="Rule-based prospective world used for smoke testing.",
                )
            )
        return EventWorldSet(query=graph.query, worlds=worlds)


class OpenAIEventWorldPlanner:
    """CQHG-conditioned prospective event-world generator for local Qwen3-VL/vLLM."""

    def __init__(self, client=None, model: str | None = None, *, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig.from_env()
        self.client = client or create_openai_compatible_client(cfg)
        self.model = model or cfg.model
        self.temperature = cfg.temperature

    @classmethod
    def from_env(cls) -> "OpenAIEventWorldPlanner":
        return cls(config=LLMConfig.from_env())

    def imagine(self, graph: QueryHypothesisGraph, *, num_worlds: int = 3) -> EventWorldSet:
        if num_worlds <= 0:
            raise ValueError("num_worlds must be positive")
        schema = EventWorldSet.model_json_schema()
        graph_json = graph.model_dump_json(indent=2)
        system = (
            "You perform abductive prospective event-world modeling for partially relevant video retrieval. "
            "Given a Counterfactual Query Hypothesis Graph (CQHG), imagine multiple plausible temporal event worlds "
            "in which the query event could occur. Each world must be structured as plausible preconditions -> "
            "the immutable CQHG query event -> plausible consequences. Generate alternatives, not paraphrases. "
            "The CQHG atomic events are hard semantic anchors: do not add, remove, rename, replace, or reinterpret "
            "them. Preconditions and consequences are soft context and must never be treated as required query facts. "
            "Return only JSON matching the supplied schema."
        )
        user = (
            f"Generate exactly {num_worlds} plausible event worlds.\n\nCQHG:\n{graph_json}\n\n"
            "For every world, query_anchor_event_ids must contain every CQHG atomic event id exactly once. "
            "Assign a positive prior to each world; priors need not sum exactly to one because the caller normalizes them.\n\n"
            f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        worlds = EventWorldSet.model_validate_json(raw)
        if len(worlds.worlds) != num_worlds:
            raise ValueError(f"expected exactly {num_worlds} event worlds, got {len(worlds.worlds)}")
        return _validate_query_anchors(graph, worlds)
