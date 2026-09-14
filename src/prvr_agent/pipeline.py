from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agents.hypothesis_planner import HypothesisPlanner
from .agents.verifier import EvidenceBackend
from .controller import CandidateEvidenceState, VerificationBudgetConfig, decide_next_action
from .reranker import ScoreFusionConfig, fuse_candidate_score, summarize_evidence
from .schemas import Candidate
from .video.sampler import TimeWindow


@dataclass(frozen=True)
class PeakMappingConfig:
    frame_seconds_per_index: float = 1.0
    clip_seconds_per_index: float = 1.0


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
    def __init__(self, planner: HypothesisPlanner, evidence_backend: EvidenceBackend, video_path_resolver: Callable[[str], str | Path], *, cfg: PipelineConfig | None = None) -> None:
        self.planner = planner
        self.evidence_backend = evidence_backend
        self.video_path_resolver = video_path_resolver
        self.cfg = cfg or PipelineConfig()

    def _seed_time(self, candidate: Candidate) -> float:
        clip_t = candidate.clip_peak_index * self.cfg.peak_mapping.clip_seconds_per_index
        frame_t = candidate.frame_peak_index * self.cfg.peak_mapping.frame_seconds_per_index
        return 0.5 * (clip_t + frame_t)

    def _window(self, seed: float, round_idx: int) -> TimeWindow:
        width = self.cfg.budget.initial_window_seconds * (self.cfg.budget.expansion_factor ** round_idx)
        return TimeWindow(max(0.0, seed - width / 2), seed + width / 2)

    def rerank(self, query: str, candidates: list[Candidate]) -> list[RerankedCandidate]:
        if not candidates:
            return []
        graph = self.planner.plan(query)
        sorted_base = sorted(candidates, key=lambda c: c.base_score, reverse=True)
        verify_set = sorted_base[: self.cfg.top_k_verify]
        output: list[RerankedCandidate] = []

        for idx, candidate in enumerate(verify_set):
            next_score = sorted_base[idx + 1].base_score if idx + 1 < len(sorted_base) else candidate.base_score
            state = CandidateEvidenceState(candidate=candidate, retrieval_margin=max(0.0, candidate.base_score - next_score))
            seed = self._seed_time(candidate)
            last_uncertainty = 1.0
            last_verification = None
            for round_idx in range(self.cfg.budget.max_rounds):
                window = self._window(seed, round_idx)
                video_path = str(self.video_path_resolver(candidate.video_id))
                support = self.evidence_backend.verify(
                    query=query, graph=graph, candidate=candidate, video_path=video_path,
                    window=window, mode="support", num_frames=self.cfg.frames_per_round,
                )
                refute = self.evidence_backend.verify(
                    query=query, graph=graph, candidate=candidate, video_path=video_path,
                    window=window, mode="refute", num_frames=self.cfg.frames_per_round,
                )
                state.add_round((window.start, window.end), support, refute)
                last_verification = summarize_evidence(support, refute)
                decision = decide_next_action(state, last_verification, self.cfg.budget)
                last_uncertainty = decision.uncertainty
                if decision.action == "stop":
                    break
            if last_verification is None:  # pragma: no cover
                continue
            final_score = fuse_candidate_score(candidate.base_score, last_verification, self.cfg.fusion)
            output.append(RerankedCandidate(candidate=candidate, final_score=final_score, rounds=state.round_idx, uncertainty=last_uncertainty))

        for candidate in sorted_base[len(verify_set):]:
            output.append(RerankedCandidate(candidate=candidate, final_score=candidate.base_score, rounds=0, uncertainty=1.0))
        return sorted(output, key=lambda x: x.final_score, reverse=True)
