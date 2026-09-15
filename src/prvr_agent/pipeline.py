from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agents.hypothesis_planner import HypothesisPlanner
from .agents.world_model import MAX_EVENT_WORLDS, EventWorldPlanner
from .agents.world_observer import (
    MAX_CHUNKS,
    MAX_CHUNKS_PER_REQUEST,
    MAX_FRAMES_PER_CHUNK,
    MAX_TOTAL_FRAMES_PER_REQUEST,
    WorldEvidenceBackend,
)
from .prospective import ProspectiveAssessment, ProspectiveConfig, fuse_prospective_score, revise_world_beliefs
from .schemas import Candidate


@dataclass(frozen=True)
class PipelineConfig:
    num_worlds: int = 3
    frames_per_chunk: int = 4
    target_chunk_seconds: float = 20.0
    chunk_overlap: float = 0.25
    max_chunks: int = 64
    chunks_per_request: int = 8
    context_radius: int = 1
    scoring: ProspectiveConfig = ProspectiveConfig()

    def validate(self) -> None:
        if not 1 <= self.num_worlds <= MAX_EVENT_WORLDS:
            raise ValueError(f"num_worlds must be in [1, {MAX_EVENT_WORLDS}]")
        if not 1 <= self.frames_per_chunk <= MAX_FRAMES_PER_CHUNK:
            raise ValueError(f"frames_per_chunk must be in [1, {MAX_FRAMES_PER_CHUNK}]")
        if not math.isfinite(self.target_chunk_seconds) or self.target_chunk_seconds <= 0:
            raise ValueError("target_chunk_seconds must be finite and positive")
        if not math.isfinite(self.chunk_overlap) or not 0.0 <= self.chunk_overlap < 1.0:
            raise ValueError("chunk_overlap must be in [0, 1)")
        if not 1 <= self.max_chunks <= MAX_CHUNKS:
            raise ValueError(f"max_chunks must be in [1, {MAX_CHUNKS}]")
        if not 1 <= self.chunks_per_request <= MAX_CHUNKS_PER_REQUEST:
            raise ValueError(f"chunks_per_request must be in [1, {MAX_CHUNKS_PER_REQUEST}]")
        if self.chunks_per_request * self.frames_per_chunk > MAX_TOTAL_FRAMES_PER_REQUEST:
            raise ValueError(
                f"chunks_per_request * frames_per_chunk must not exceed {MAX_TOTAL_FRAMES_PER_REQUEST}"
            )
        if self.context_radius < 0:
            raise ValueError("context_radius must be non-negative")
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
                frames_per_chunk=self.cfg.frames_per_chunk,
                target_chunk_seconds=self.cfg.target_chunk_seconds,
                chunk_overlap=self.cfg.chunk_overlap,
                max_chunks=self.cfg.max_chunks,
                chunks_per_request=self.cfg.chunks_per_request,
                context_radius=self.cfg.context_radius,
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
