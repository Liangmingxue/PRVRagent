from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agents.hypothesis_planner import HypothesisPlanner
from .agents.prospective_world_modeler import ProspectiveWorldModeler
from .agents.world_observer import WorldObservationBackend
from .schemas import Candidate
from .world_reasoning import (
    ScoreFusionConfig,
    WorldBelief,
    WorldRevisionConfig,
    fuse_candidate_score,
    prospective_world_score,
    revise_world_beliefs,
)


@dataclass(frozen=True)
class PipelineConfig:
    top_k_reason: int = 20
    num_worlds: int = 3
    coarse_frames: int = 12
    revision: WorldRevisionConfig = WorldRevisionConfig()
    fusion: ScoreFusionConfig = ScoreFusionConfig()


@dataclass(frozen=True)
class RerankedCandidate:
    candidate: Candidate
    final_score: float
    query_satisfaction: float
    counterfactual_risk: float
    world_score: float
    world_beliefs: tuple[WorldBelief, ...]
    uncertainty: float


class PRVRAgentReranker:
    """CQHG + abductive prospective event-world reranker.

    The pipeline intentionally does not consume DreamPRVR argmax locations. It
    reasons over query-side event worlds and sparse global candidate evidence.
    """

    def __init__(
        self,
        planner: HypothesisPlanner,
        world_modeler: ProspectiveWorldModeler,
        observer: WorldObservationBackend,
        video_path_resolver: Callable[[str], str | Path],
        *,
        cfg: PipelineConfig | None = None,
    ) -> None:
        self.planner = planner
        self.world_modeler = world_modeler
        self.observer = observer
        self.video_path_resolver = video_path_resolver
        self.cfg = cfg or PipelineConfig()
        if self.cfg.top_k_reason <= 0:
            raise ValueError("top_k_reason must be positive")
        if self.cfg.num_worlds <= 0:
            raise ValueError("num_worlds must be positive")
        if self.cfg.coarse_frames <= 0:
            raise ValueError("coarse_frames must be positive")

    def rerank(self, query: str, candidates: list[Candidate]) -> list[RerankedCandidate]:
        if not candidates:
            return []
        graph = self.planner.plan(query)
        worlds = self.world_modeler.imagine(graph, num_worlds=self.cfg.num_worlds)
        sorted_base = sorted(candidates, key=lambda item: item.base_score, reverse=True)
        reason_set = sorted_base[: self.cfg.top_k_reason]
        output: list[RerankedCandidate] = []

        for candidate in reason_set:
            observation = self.observer.observe(
                query=query,
                graph=graph,
                worlds=worlds,
                candidate=candidate,
                video_path=str(self.video_path_resolver(candidate.video_id)),
                num_frames=self.cfg.coarse_frames,
            )
            beliefs = revise_world_beliefs(worlds, observation, self.cfg.revision)
            world_score = prospective_world_score(beliefs)
            final_score = fuse_candidate_score(candidate.base_score, observation, world_score, self.cfg.fusion)
            output.append(
                RerankedCandidate(
                    candidate=candidate,
                    final_score=final_score,
                    query_satisfaction=observation.query_satisfaction,
                    counterfactual_risk=observation.counterfactual_risk,
                    world_score=world_score,
                    world_beliefs=beliefs,
                    uncertainty=observation.uncertainty,
                )
            )

        for candidate in sorted_base[len(reason_set) :]:
            output.append(
                RerankedCandidate(
                    candidate=candidate,
                    final_score=float(candidate.base_score),
                    query_satisfaction=0.5,
                    counterfactual_risk=0.0,
                    world_score=0.0,
                    world_beliefs=(),
                    uncertainty=1.0,
                )
            )
        return sorted(output, key=lambda item: item.final_score, reverse=True)
