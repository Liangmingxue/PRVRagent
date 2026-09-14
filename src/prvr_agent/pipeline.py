from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .agents.hypothesis_planner import HypothesisPlanner
from .agents.verifier import EvidenceBackend
from .controller import CandidateEvidenceState, VerificationBudgetConfig, decide_next_action
from .reranker import ScoreFusionConfig, aggregate_evidence, fuse_candidate_score
from .schemas import Candidate
from .video.sampler import TimeWindow


@dataclass(frozen=True)
class PeakMappingConfig:
    """Map DreamPRVR feature locations to raw-video time.

    In ``auto`` mode, fixed strides are used only when both are supplied. Otherwise
    the mapper uses relative bin centers plus the raw-video duration. Relative mode
    matches DreamPRVR's public preprocessing, which uniformly/averagely resamples
    each video's feature sequence instead of using one global seconds-per-index.
    """

    mode: str = "auto"
    frame_seconds_per_index: Optional[float] = None
    clip_seconds_per_index: Optional[float] = None

    def resolved_mode(self) -> str:
        if self.mode not in {"auto", "relative", "fixed_stride"}:
            raise ValueError("peak mapping mode must be auto, relative, or fixed_stride")
        if self.mode == "auto":
            if self.frame_seconds_per_index is None and self.clip_seconds_per_index is None:
                return "relative"
            if self.frame_seconds_per_index is not None and self.clip_seconds_per_index is not None:
                return "fixed_stride"
            raise ValueError("set both fixed strides or neither")
        return self.mode

    def validate(self) -> None:
        mode = self.resolved_mode()
        if mode == "fixed_stride":
            for name, value in (
                ("frame_seconds_per_index", self.frame_seconds_per_index),
                ("clip_seconds_per_index", self.clip_seconds_per_index),
            ):
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive in fixed_stride mode")


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
        video_path_resolver: Callable[[str], object],
        *,
        cfg: Optional[PipelineConfig] = None,
    ) -> None:
        self.planner = planner
        self.evidence_backend = evidence_backend
        self.video_path_resolver = video_path_resolver
        self.cfg = cfg or PipelineConfig()
        self.cfg.peak_mapping.validate()
        if self.cfg.top_k_verify <= 0:
            raise ValueError("top_k_verify must be positive")
        if self.cfg.frames_per_round <= 0:
            raise ValueError("frames_per_round must be positive")

    def _video_duration(self, video_path: str) -> float:
        duration_fn = getattr(self.evidence_backend, "video_duration", None)
        if not callable(duration_fn):
            raise ValueError(
                "relative peak mapping requires an evidence backend exposing video_duration(video_path); "
                "otherwise configure fixed_stride peak mapping"
            )
        duration = float(duration_fn(video_path))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"invalid video duration {duration!r} for {video_path}")
        return duration

    @staticmethod
    def _relative_time(index: int, count: int, duration: float, name: str) -> float:
        if count <= 0:
            raise ValueError(f"{name} location count must be positive")
        if index < 0 or index >= count:
            raise ValueError(f"{name} peak index {index} is outside valid location count {count}")
        return ((index + 0.5) / count) * duration

    def _peak_times(self, candidate: Candidate, video_path: str) -> Tuple[float, float, Optional[float]]:
        mapping = self.cfg.peak_mapping
        mode = mapping.resolved_mode()
        if mode == "fixed_stride":
            assert mapping.clip_seconds_per_index is not None
            assert mapping.frame_seconds_per_index is not None
            return (
                candidate.clip_peak_index * mapping.clip_seconds_per_index,
                candidate.frame_peak_index * mapping.frame_seconds_per_index,
                None,
            )

        duration = self._video_duration(video_path)
        try:
            clip_count = int(candidate.metadata["clip_num_locations"])
            frame_count = int(candidate.metadata["frame_valid_locations"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "relative peak mapping requires candidate metadata 'clip_num_locations' and "
                "'frame_valid_locations'; obtain candidates from DreamPRVRAdapter"
            ) from exc
        return (
            self._relative_time(candidate.clip_peak_index, clip_count, duration, "clip"),
            self._relative_time(candidate.frame_peak_index, frame_count, duration, "frame"),
            duration,
        )

    @staticmethod
    def _local_margin(sorted_candidates: list[Candidate], idx: int) -> float:
        gaps = []
        if idx > 0:
            gaps.append(abs(sorted_candidates[idx - 1].base_score - sorted_candidates[idx].base_score))
        if idx + 1 < len(sorted_candidates):
            gaps.append(abs(sorted_candidates[idx].base_score - sorted_candidates[idx + 1].base_score))
        return min(gaps) if gaps else 1.0

    def _peak_strengths(self, candidate: Candidate) -> Tuple[float, float]:
        clip_peak_score = float(candidate.metadata.get("clip_peak_score", candidate.clip_score))
        frame_peak_score = float(candidate.metadata.get("frame_peak_score", candidate.frame_score))
        clip_weight = float(candidate.metadata.get("clip_scale_weight", 1.0))
        frame_weight = float(candidate.metadata.get("frame_scale_weight", 1.0))
        return clip_peak_score * clip_weight, frame_peak_score * frame_weight

    def _window_for_round(
        self,
        candidate: Candidate,
        clip_t: float,
        frame_t: float,
        round_idx: int,
        duration: Optional[float],
    ) -> TimeWindow:
        initial = self.cfg.budget.initial_window_seconds
        clip_strength, frame_strength = self._peak_strengths(candidate)
        primary, secondary = (clip_t, frame_t) if clip_strength >= frame_strength else (frame_t, clip_t)
        dual_seed = abs(clip_t - frame_t) > initial / 2.0

        if round_idx == 0:
            seed = primary
            width = initial
        elif round_idx == 1 and dual_seed:
            seed = secondary
            width = initial
        else:
            expansion_idx = round_idx if not dual_seed else round_idx - 1
            seed = 0.5 * (clip_t + frame_t) if dual_seed else primary
            width = initial * (self.cfg.budget.expansion_factor ** expansion_idx)

        window = TimeWindow(max(0.0, seed - width / 2.0), seed + width / 2.0)
        return window.clamp(duration) if duration is not None else window

    def rerank(self, query: str, candidates: list[Candidate]) -> list[RerankedCandidate]:
        if not candidates:
            return []
        graph = self.planner.plan(query)
        expected_event_ids = {event.id for event in graph.atomic_events}
        expected_temporal_ids = {rel.id for rel in graph.temporal_constraints}
        expected_identity_ids = {rel.id for rel in graph.identity_constraints}

        sorted_base = sorted(candidates, key=lambda c: c.base_score, reverse=True)
        verify_set = sorted_base[: self.cfg.top_k_verify]
        output = []

        for idx, candidate in enumerate(verify_set):
            video_path = str(self.video_path_resolver(candidate.video_id))
            clip_t, frame_t, duration = self._peak_times(candidate, video_path)
            state = CandidateEvidenceState(
                candidate=candidate,
                retrieval_margin=self._local_margin(sorted_base, idx),
                peak_gap_seconds=abs(clip_t - frame_t),
            )
            last_uncertainty = 1.0
            last_verification = None

            for round_idx in range(self.cfg.budget.max_rounds):
                window = self._window_for_round(candidate, clip_t, frame_t, round_idx, duration)
                interval = (window.start, window.end)
                if state.has_seen(interval):
                    break
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
                state.add_round(interval, support, refute)
                last_verification = aggregate_evidence(
                    state.support_evidence,
                    state.refute_evidence,
                    expected_event_ids=expected_event_ids,
                    expected_temporal_ids=expected_temporal_ids,
                    expected_identity_ids=expected_identity_ids,
                )
                decision = decide_next_action(state, last_verification, self.cfg.budget)
                last_uncertainty = decision.uncertainty
                if decision.action == "stop":
                    break

            if last_verification is None:
                output.append(
                    RerankedCandidate(
                        candidate=candidate,
                        final_score=self.cfg.fusion.base_weight * candidate.base_score,
                        rounds=0,
                        uncertainty=1.0,
                    )
                )
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
        return sorted(output, key=lambda item: item.final_score, reverse=True)
