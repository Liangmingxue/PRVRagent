from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agents.hypothesis_planner import HypothesisPlanner
from .agents.world_model import EventWorldPlanner
from .agents.world_observer import WorldEvidenceBackend
from .prospective import ProspectiveAssessment, ProspectiveConfig, fuse_prospective_score, revise_world_beliefs
from .schemas import Candidate


@dataclass(frozen=True)
class PipelineConfig:
    num_worlds: int = 3
    coarse_frames: int = 8
    scoring: ProspectiveConfig = ProspectiveConfig()

    def validate(self) -> None:
        if self.num_worlds <= 0:
            raise ValueError("num_worlds must be positive")
        if self.coarse_frames <= 0:
            raise ValueError("coarse_frames must be positive")
        self.scoring.validate()


@dataclass(frozen=True)
class RerankedCandidate:
    candidate: Candidate
    final_score: float
    graph_score: float
    world_score: float
    assessment: ProspectiveAssessment


class PRVRAgentReranker:
    """Two-part PRVR reasoning: CQHG satisfaction + prospective event worlds.

    The former peak-seeded support/refute loop is gone. One coarse candidate
    observation now serves both contributions: it evaluates whether the hard CQHG
    is satisfied and revises the soft prospective worlds before score fusion.
    """

    def __init__(
        self,
        planner: HypothesisPlanner,
        world_planner: EventWorldPlanner,
        world_evidence_backend: WorldEvidenceBackend,
        video_path_resolver: Callable[[str], str | Path],
        *,
        cfg: PipelineConfig | None = None,
    ) -> None:
        self.planner = planner
        self.world_planner = world_planner
        self.world_evidence_backend = world_evidence_backend
        self.video_path_resolver = video_path_resolver
        self.cfg = cfg or PipelineConfig()
        self.cfg.validate()

    def rerank(self, query: str, candidates: list[Candidate]) -> list[RerankedCandidate]:
        if not candidates:
            return []

        graph = self.planner.plan(query)
        worlds = self.world_planner.imagine(graph, num_worlds=self.cfg.num_worlds)
        output: list[RerankedCandidate] = []

        for candidate in candidates:
            video_path = str(self.video_path_resolver(candidate.video_id))
            evidence = self.world_evidence_backend.assess(
                query=query,
                graph=graph,
                worlds=worlds,
                candidate=candidate,
                video_path=video_path,
                num_frames=self.cfg.coarse_frames,
            )
            assessment = revise_world_beliefs(graph, worlds, evidence, self.cfg.scoring)
            final_score = fuse_prospective_score(candidate.base_score, assessment, self.cfg.scoring)
            output.append(
                RerankedCandidate(
                    candidate=candidate,
                    final_score=final_score,
                    graph_score=assessment.graph_score,
                    world_score=assessment.world_score,
                    assessment=assessment,
                )
            )

        return sorted(output, key=lambda item: item.final_score, reverse=True)
