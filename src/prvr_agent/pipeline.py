from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agents.hypothesis_planner import HypothesisPlanner
from .agents.world_model import MAX_EVENT_WORLDS, EventWorldPlanner
from .agents.world_observer import MAX_COARSE_FRAMES, WorldEvidenceBackend
from .prospective import ProspectiveAssessment, ProspectiveConfig, fuse_prospective_score, revise_world_beliefs
from .schemas import Candidate


@dataclass(frozen=True)
class PipelineConfig:
    num_worlds: int = 3
    coarse_frames: int = 8
    scoring: ProspectiveConfig = ProspectiveConfig()

    def validate(self) -> None:
        if not 1 <= self.num_worlds <= MAX_EVENT_WORLDS:
            raise ValueError(f"num_worlds must be in [1, {MAX_EVENT_WORLDS}]")
        if not 1 <= self.coarse_frames <= MAX_COARSE_FRAMES:
            raise ValueError(f"coarse_frames must be in [1, {MAX_COARSE_FRAMES}]")
        self.scoring.validate()


@dataclass(frozen=True)
class RerankedCandidate:
    candidate: Candidate
    final_score: float
    graph_score: float
    world_score: float
    assessment: ProspectiveAssessment


class PRVRAgentReranker:
    """Two-part PRVR reasoning: CQHG satisfaction + prospective event worlds."""

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
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        graph = self.planner.plan(query)
        if graph.query != query:
            raise ValueError("hypothesis planner changed the retrieval query")
        worlds = self.world_planner.imagine(graph, num_worlds=self.cfg.num_worlds)
        if worlds.query != graph.query:
            raise ValueError("event worlds do not match the CQHG query")
        if len(worlds.worlds) != self.cfg.num_worlds:
            raise ValueError(
                f"event-world planner returned {len(worlds.worlds)} worlds; expected {self.cfg.num_worlds}"
            )

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
            if evidence.candidate_video_id != candidate.video_id:
                raise ValueError(
                    f"evidence candidate id {evidence.candidate_video_id!r} does not match {candidate.video_id!r}"
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
