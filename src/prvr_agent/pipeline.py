from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .agents.hypothesis_planner import HypothesisPlanner
from .agents.world_model import MAX_EVENT_WORLDS, EventWorldPlanner
from .agents.world_observer import (
    MAX_CHUNKS,
    MAX_CHUNKS_PER_REQUEST,
    MAX_CONFIRMATION_SEGMENTS,
    MAX_FRAMES_PER_CHUNK,
    MAX_REFINEMENT_CHUNKS,
    MAX_SIDEKICK_SCAN_FRAMES,
    MAX_TOTAL_FRAMES_PER_REQUEST,
    WorldEvidenceBackend,
)
from .prospective import ProspectiveAssessment, ProspectiveConfig, fuse_prospective_score, revise_world_beliefs
from .schemas import Candidate


@dataclass(frozen=True)
class PipelineConfig:
    num_worlds: int = 3

    # Coarse VLM observation over event-aware variable-length segments.
    frames_per_chunk: int = 4
    target_chunk_seconds: float = 20.0
    chunk_overlap: float = 0.25  # API compatibility; adaptive segments replace fixed overlap.
    max_chunks: int = 64
    chunks_per_request: int = 8
    context_radius: int = 1

    # Query-agnostic hybrid sidekick: dense raw-pixel change + cached DreamPRVR
    # semantic feature novelty. Weights are validation-time hyperparameters rather
    # than literature-derived constants.
    sidekick_scan_fps: float = 2.0
    sidekick_max_frames: int = 512
    sidekick_visual_weight: float = 0.5
    sidekick_semantic_weight: float = 0.5
    event_min_seconds: float = 4.0
    event_boundary_quantile: float = 0.80

    # Bounded coarse-to-fine observation.  A zero default threshold deliberately
    # keeps every coarse event segment eligible for the small refinement budget;
    # the selector then balances evidence priority and temporal diversity.  This
    # avoids a hard pre-filter failure in which an overconfident coarse pass could
    # suppress all denser observations.  ``max_refinement_chunks`` still caps cost.
    refinement_frames_per_chunk: int = 12
    max_refinement_chunks: int = 3
    refinement_threshold: float = 0.0

    # Clean single-span confirmation that produces final hard CQHG evidence.
    confirmation_frames: int = 16
    confirmation_max_segments: int = 2
    confirmation_max_seconds: float = 40.0

    scoring: ProspectiveConfig = ProspectiveConfig()

    def validate(self) -> None:
        if not 1 <= self.num_worlds <= MAX_EVENT_WORLDS:
            raise ValueError(f"num_worlds must be in [1, {MAX_EVENT_WORLDS}]")
        if not 1 <= self.frames_per_chunk <= MAX_FRAMES_PER_CHUNK:
            raise ValueError(f"frames_per_chunk must be in [1, {MAX_FRAMES_PER_CHUNK}]")
        if not 1 <= self.refinement_frames_per_chunk <= MAX_FRAMES_PER_CHUNK:
            raise ValueError(f"refinement_frames_per_chunk must be in [1, {MAX_FRAMES_PER_CHUNK}]")
        if not 0 <= self.max_refinement_chunks <= MAX_REFINEMENT_CHUNKS:
            raise ValueError(f"max_refinement_chunks must be in [0, {MAX_REFINEMENT_CHUNKS}]")
        if self.max_refinement_chunks > 0 and self.refinement_frames_per_chunk <= self.frames_per_chunk:
            raise ValueError("refinement_frames_per_chunk must exceed frames_per_chunk")
        if not math.isfinite(self.refinement_threshold) or not 0.0 <= self.refinement_threshold <= 1.0:
            raise ValueError("refinement_threshold must be finite and in [0, 1]")
        if not math.isfinite(self.target_chunk_seconds) or self.target_chunk_seconds <= 0:
            raise ValueError("target_chunk_seconds must be finite and positive")
        if not math.isfinite(self.chunk_overlap) or not 0.0 <= self.chunk_overlap < 1.0:
            raise ValueError("chunk_overlap must be finite and in [0, 1)")
        if not 1 <= self.max_chunks <= MAX_CHUNKS:
            raise ValueError(f"max_chunks must be in [1, {MAX_CHUNKS}]")
        if not 1 <= self.chunks_per_request <= MAX_CHUNKS_PER_REQUEST:
            raise ValueError(f"chunks_per_request must be in [1, {MAX_CHUNKS_PER_REQUEST}]")
        if self.chunks_per_request * self.frames_per_chunk > MAX_TOTAL_FRAMES_PER_REQUEST:
            raise ValueError(f"chunks_per_request * frames_per_chunk must not exceed {MAX_TOTAL_FRAMES_PER_REQUEST}")
        if self.context_radius < 0:
            raise ValueError("context_radius must be non-negative")

        if not math.isfinite(self.sidekick_scan_fps) or self.sidekick_scan_fps <= 0:
            raise ValueError("sidekick_scan_fps must be finite and positive")
        if not 2 <= self.sidekick_max_frames <= MAX_SIDEKICK_SCAN_FRAMES:
            raise ValueError(f"sidekick_max_frames must be in [2, {MAX_SIDEKICK_SCAN_FRAMES}]")
        for name, value in (
            ("sidekick_visual_weight", self.sidekick_visual_weight),
            ("sidekick_semantic_weight", self.sidekick_semantic_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.sidekick_visual_weight + self.sidekick_semantic_weight <= 0:
            raise ValueError("at least one sidekick fusion weight must be positive")
        if not math.isfinite(self.event_min_seconds) or self.event_min_seconds <= 0:
            raise ValueError("event_min_seconds must be finite and positive")
        if self.event_min_seconds > self.target_chunk_seconds:
            raise ValueError("event_min_seconds must not exceed target_chunk_seconds")
        if not math.isfinite(self.event_boundary_quantile) or not 0.0 <= self.event_boundary_quantile <= 1.0:
            raise ValueError("event_boundary_quantile must be finite and in [0, 1]")

        if not 1 <= self.confirmation_frames <= MAX_FRAMES_PER_CHUNK:
            raise ValueError(f"confirmation_frames must be in [1, {MAX_FRAMES_PER_CHUNK}]")
        if not 1 <= self.confirmation_max_segments <= MAX_CONFIRMATION_SEGMENTS:
            raise ValueError(f"confirmation_max_segments must be in [1, {MAX_CONFIRMATION_SEGMENTS}]")
        if not math.isfinite(self.confirmation_max_seconds) or self.confirmation_max_seconds <= 0:
            raise ValueError("confirmation_max_seconds must be finite and positive")
        if self.confirmation_max_seconds < self.event_min_seconds:
            raise ValueError("confirmation_max_seconds must be >= event_min_seconds")

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
        cfg: Optional[PipelineConfig] = None,
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
                refinement_frames_per_chunk=self.cfg.refinement_frames_per_chunk,
                max_refinement_chunks=self.cfg.max_refinement_chunks,
                refinement_threshold=self.cfg.refinement_threshold,
                sidekick_scan_fps=self.cfg.sidekick_scan_fps,
                sidekick_max_frames=self.cfg.sidekick_max_frames,
                sidekick_visual_weight=self.cfg.sidekick_visual_weight,
                sidekick_semantic_weight=self.cfg.sidekick_semantic_weight,
                event_min_seconds=self.cfg.event_min_seconds,
                event_boundary_quantile=self.cfg.event_boundary_quantile,
                confirmation_frames=self.cfg.confirmation_frames,
                confirmation_max_segments=self.cfg.confirmation_max_segments,
                confirmation_max_seconds=self.cfg.confirmation_max_seconds,
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
