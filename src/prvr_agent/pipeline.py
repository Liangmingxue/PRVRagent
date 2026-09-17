from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Callable

from .agents.hypothesis_planner import HypothesisPlanner
from .agents.verifier import EvidenceBackend
from .controller import CandidateEvidenceState, VerificationBudgetConfig, decide_next_action
from .reranker import ScoreFusionConfig, aggregate_evidence, fuse_candidate_score
from .schemas import Candidate
from .video.sampler import TimeWindow


@dataclass(frozen=True)
class PeakMappingConfig:
    frame_seconds_per_index: float | None = None
    clip_seconds_per_index: float | None = None

    def validate(self) -> None:
        for name, value in (
            ("frame_seconds_per_index", self.frame_seconds_per_index),
            ("clip_seconds_per_index", self.clip_seconds_per_index),
        ):
            if value is None or not isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be configured from the dataset feature-extraction pipeline before verification"
                )


@dataclass(frozen=True)
class PipelineConfig:
    top_k_verify: int = 20
    frames_per_round: int = 8
    budget: VerificationBudgetConfig = VerificationBudgetConfig()
    fusion: ScoreFusionConfig = ScoreFusionConfig()
    peak_mapping: PeakMappingConfig = PeakMappingConfig()


@dataclass(frozen=True)
class RerankedCandidate:
    candidate: Candidate
    final_score: float
    rounds: int
    uncertainty: float


class PRVRAgentReranker:
    def __init__(
        self,
        planner: HypothesisPlanner,
        evidence_backend: EvidenceBackend,
        video_path_resolver: Callable[[str], str | Path],
        *,
        cfg: PipelineConfig | None = None,
    ) -> None:
        self.planner = planner
        self.evidence_backend = evidence_backend
        self.video_path_resolver = video_path_resolver
        self.cfg = cfg or PipelineConfig()
        self.cfg.peak_mapping.validate()
        for name in ("top_k_verify", "frames_per_round"):
            value = getattr(self.cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def _peak_times(self, candidate: Candidate) -> tuple[float, float]:
        mapping = self.cfg.peak_mapping
        assert mapping.clip_seconds_per_index is not None
        assert mapping.frame_seconds_per_index is not None
        return (
            candidate.clip_peak_index * mapping.clip_seconds_per_index,
            candidate.frame_peak_index * mapping.frame_seconds_per_index,
        )

    def _window(self, seed: float, round_idx: int) -> TimeWindow:
        width = self.cfg.budget.initial_window_seconds * (self.cfg.budget.expansion_factor ** round_idx)
        return TimeWindow(max(0.0, seed - width / 2), seed + width / 2)

    @staticmethod
    def _local_margin(sorted_candidates: list[Candidate], idx: int) -> float:
        gaps: list[float] = []
        if idx > 0:
            gaps.append(abs(sorted_candidates[idx - 1].base_score - sorted_candidates[idx].base_score))
        if idx + 1 < len(sorted_candidates):
            gaps.append(abs(sorted_candidates[idx].base_score - sorted_candidates[idx + 1].base_score))
        return min(gaps) if gaps else 1.0

    def rerank(self, query: str, candidates: list[Candidate]) -> list[RerankedCandidate]:
        if not candidates:
            return []
        graph = self.planner.plan(query)
        sorted_base = sorted(candidates, key=lambda c: c.base_score, reverse=True)
        verify_set = sorted_base[: self.cfg.top_k_verify]
        output: list[RerankedCandidate] = []

        for idx, candidate in enumerate(verify_set):
            clip_t, frame_t = self._peak_times(candidate)
            state = CandidateEvidenceState(
                candidate=candidate,
                retrieval_margin=self._local_margin(sorted_base, idx),
                peak_gap_seconds=abs(clip_t - frame_t),
            )
            seed = 0.5 * (clip_t + frame_t)
            last_uncertainty = 1.0
            last_verification = None
            video_path = str(self.video_path_resolver(candidate.video_id))

            for round_idx in range(self.cfg.budget.max_rounds):
                window = self._window(seed, round_idx)
                support = self.evidence_backend.verify(
                    query=query,
                    graph=graph,
                    candidate=candidate,
                    video_path=video_path,
                    window=window,
                    mode="support",
                    num_frames=self.cfg.frames_per_round,
                )
                refute = self.evidence_backend.verify(
                    query=query,
                    graph=graph,
                    candidate=candidate,
                    video_path=video_path,
                    window=window,
                    mode="refute",
                    num_frames=self.cfg.frames_per_round,
                )
                state.add_round((window.start, window.end), support, refute)
                last_verification = aggregate_evidence(state.support_evidence, state.refute_evidence)
                decision = decide_next_action(state, last_verification, self.cfg.budget)
                last_uncertainty = decision.uncertainty
                if decision.action == "stop":
                    break

            if last_verification is None:  # pragma: no cover
                continue
            final_score = fuse_candidate_score(candidate.base_score, last_verification, self.cfg.fusion)
            output.append(
                RerankedCandidate(
                    candidate=candidate,
                    final_score=final_score,
                    rounds=state.round_idx,
                    uncertainty=last_uncertainty,
                )
            )

        for candidate in sorted_base[len(verify_set) :]:
            output.append(
                RerankedCandidate(
                    candidate=candidate,
                    final_score=self.cfg.fusion.base_weight * candidate.base_score,
                    rounds=0,
                    uncertainty=1.0,
                )
            )
        return sorted(output, key=lambda x: x.final_score, reverse=True)
