from __future__ import annotations

import json
import re
from typing import Protocol

from prvr_agent.llm_config import LLMConfig, create_openai_compatible_client
from prvr_agent.schemas import AtomicEvent, CounterfactualHypothesis, QueryHypothesisGraph, TemporalConstraint


class HypothesisPlanner(Protocol):
    def plan(self, query: str) -> QueryHypothesisGraph: ...


class RuleBasedHypothesisPlanner:
    """Dependency-free fallback used for smoke tests and ablations."""

    _splitter = re.compile(r"\b(?:then|after|before|and then)\b", re.IGNORECASE)

    def plan(self, query: str) -> QueryHypothesisGraph:
        parts = [p.strip(" ,.") for p in self._splitter.split(query) if p.strip(" ,.")]
        if not parts:
            parts = [query.strip()]
        events = [AtomicEvent(id=f"E{i+1}", subject="unknown", action=part) for i, part in enumerate(parts)]
        temporal: list[TemporalConstraint] = []
        lowered = query.lower()
        if len(events) >= 2:
            relation = "after" if " after " in lowered else "before"
            temporal.append(TemporalConstraint(event_a=events[0].id, relation=relation, event_b=events[1].id))
        counterfactuals = [
            CounterfactualHypothesis(
                id="CF1",
                type="partial_event",
                description="Only a subset of the requested events is visible.",
            )
        ]
        if temporal:
            counterfactuals.append(
                CounterfactualHypothesis(
                    id="CF2",
                    type="temporal_reversal",
                    description="The requested events occur in the opposite temporal order.",
                )
            )
        return QueryHypothesisGraph(
            query=query,
            atomic_events=events,
            temporal_constraints=temporal,
            positive_hypothesis=f"The video contains a coherent local moment satisfying: {query}",
            counterfactuals=counterfactuals,
        )


class OpenAIHypothesisPlanner:
    """Structured planner backed by an OpenAI-compatible server such as vLLM."""

    def __init__(self, client=None, model: str | None = None, *, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig.from_env()
        self.client = client or create_openai_compatible_client(cfg)
        self.model = model or cfg.model
        self.temperature = cfg.temperature

    @classmethod
    def from_env(cls) -> "OpenAIHypothesisPlanner":
        return cls(config=LLMConfig.from_env())

    def plan(self, query: str) -> QueryHypothesisGraph:
        schema = QueryHypothesisGraph.model_json_schema()
        system = (
            "You decompose a partially relevant video retrieval query into a testable event graph. "
            "Preserve the query meaning. Separate atomic visual events from temporal and identity constraints. "
            "Also generate plausible counterfactual near-miss hypotheses that could create high local similarity "
            "without fully satisfying the query. Return only JSON matching the supplied schema."
        )
        user = f"Query:\n{query}\n\nJSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return QueryHypothesisGraph.model_validate_json(content)
