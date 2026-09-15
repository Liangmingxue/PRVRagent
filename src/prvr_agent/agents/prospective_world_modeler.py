from __future__ import annotations

import json
from typing import Protocol

from prvr_agent.llm_config import LLMConfig, create_openai_compatible_client
from prvr_agent.schemas import ProspectiveEventWorld, ProspectiveWorldSet, QueryHypothesisGraph
from prvr_agent.structured_output import request_structured_json


class ProspectiveWorldModeler(Protocol):
    def imagine(self, graph: QueryHypothesisGraph, *, num_worlds: int) -> ProspectiveWorldSet: ...


def _anchor_worlds(graph: QueryHypothesisGraph, world_set: ProspectiveWorldSet, num_worlds: int) -> ProspectiveWorldSet:
    if num_worlds <= 0:
        raise ValueError("num_worlds must be positive")
    worlds = world_set.worlds[:num_worlds]
    if not worlds:
        raise ValueError("world modeler returned no prospective worlds")
    anchored = [world.model_copy(update={"query_anchor": graph.positive_hypothesis}) for world in worlds]
    return ProspectiveWorldSet(query=graph.query, worlds=anchored)


class RuleBasedProspectiveWorldModeler:
    """Simple deterministic fallback for tests; it is not the research model."""

    _templates = (
        (
            ["The required actors and objects become available in the scene."],
            ["The scene continues consistently after the queried event."],
        ),
        (
            ["A short preparation or approach leads into the queried event."],
            ["A natural follow-up action occurs after the queried event."],
        ),
        (
            ["The queried event begins with minimal visible preparation."],
            ["The participants transition to another compatible activity."],
        ),
    )

    def imagine(self, graph: QueryHypothesisGraph, *, num_worlds: int = 3) -> ProspectiveWorldSet:
        if num_worlds <= 0:
            raise ValueError("num_worlds must be positive")
        count = min(num_worlds, len(self._templates))
        prior = 1.0 / count
        worlds = []
        for idx, (preconditions, consequences) in enumerate(self._templates[:count], start=1):
            worlds.append(
                ProspectiveEventWorld(
                    id=f"W{idx}",
                    preconditions=preconditions,
                    query_anchor=graph.positive_hypothesis,
                    consequences=consequences,
                    prior=prior,
                    rationale="Rule-based smoke-test world.",
                )
            )
        return ProspectiveWorldSet(query=graph.query, worlds=worlds)


class OpenAIProspectiveWorldModeler:
    """APEI query-side event-world generator backed by an OpenAI-compatible LLM."""

    def __init__(self, client=None, model: str | None = None, *, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig.from_env()
        cfg.validate()
        self.client = client or create_openai_compatible_client(cfg)
        self.model = model or cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.validation_retries = cfg.validation_retries

    @classmethod
    def from_env(cls) -> "OpenAIProspectiveWorldModeler":
        return cls(config=LLMConfig.from_env())

    def imagine(self, graph: QueryHypothesisGraph, *, num_worlds: int = 3) -> ProspectiveWorldSet:
        if num_worlds <= 0:
            raise ValueError("num_worlds must be positive")
        schema = ProspectiveWorldSet.model_json_schema()
        system = (
            "You are the Abductive Prospective Event Imagination agent for PRVR. Starting from a CQHG that "
            "defines what MUST be true, generate multiple plausible event worlds describing how that anchored "
            "event could be situated in a longer video. Each world has soft preconditions, the fixed query event, "
            "and soft consequences. Generate plausible alternatives rather than one deterministic story. "
            "Do NOT turn optional context into relevance requirements, do NOT generate counterfactual failures, "
            "and do NOT change the query anchor. Priors should reflect relative plausibility. Return only JSON."
        )
        user = (
            f"Generate up to {num_worlds} prospective event worlds.\n"
            "CQHG:\n"
            + graph.model_dump_json(indent=2)
            + "\n\nThe query_anchor in every world MUST equal this string exactly:\n"
            + graph.positive_hypothesis
            + "\n\nJSON schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        result = request_structured_json(
            client=self.client,
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_model=ProspectiveWorldSet,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            validation_retries=self.validation_retries,
        )
        return _anchor_worlds(graph, result, num_worlds)
