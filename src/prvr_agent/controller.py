from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from .reranker import VerificationScore
from .schemas import Candidate, EvidenceResult


@dataclass(frozen=True)
class VerificationBudgetConfig:
    max_rounds: int = 3
    initial_window_seconds: float = 12.0
    expansion_factor: float = 2.0
    early_stop_uncertainty: float = 0.18
    peak_disagreement_seconds_scale: float = 0.10
    retrieval_margin_scale: float = 0.20
    reject_contradiction: float = 0.80
    reject_positive_max: float = 0.40

    def __post_init__(self) -> None:
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.initial_window_seconds <= 0:
            raise ValueError("initial_window_seconds must be positive")
        if self.expansion_factor < 1.0:
            raise ValueError("expansion_factor must be >= 1")
        for name in (
            "early_stop_uncertainty",
            "reject_contradiction",
            "reject_positive_max",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.peak_disagreement_seconds_scale < 0 or self.retrieval_margin_scale <= 0:
            raise ValueError("uncertainty scales must be positive")


@dataclass
class CandidateEvidenceState:
    candidate: Candidate
    seen_intervals: list[tuple[float, float]] = field(default_factory=list)
    support_evidence: list[EvidenceResult] = field(default_factory=list)
    refute_evidence: list[EvidenceResult] = field(default_factory=list)
    round_idx: int = 0
    retrieval_margin: float = 0.0
    peak_gap_seconds: float = 0.0

    def add_round(self, interval: tuple[float, float], support: EvidenceResult, refute: EvidenceResult) -> None:
        self.round_idx += 1
        self.seen_intervals.append(interval)
        self.support_evidence.append(support)
        self.refute_evidence.append(refute)

    def has_seen(self, interval: tuple[float, float], tol: float = 1e-3) -> bool:
        return any(abs(a - interval[0]) <= tol and abs(b - interval[1]) <= tol for a, b in self.seen_intervals)


@dataclass(frozen=True)
class ControlDecision:
    action: str
    uncertainty: float
    reason: str


def positive_confidence(score: VerificationScore) -> float:
    return mean((score.atomic, score.temporal, score.identity, score.completeness))


def joint_uncertainty(
    state: CandidateEvidenceState,
    score: VerificationScore,
    cfg: VerificationBudgetConfig,
) -> float:
    peak_disagreement = min(1.0, state.peak_gap_seconds * cfg.peak_disagreement_seconds_scale)
    margin_uncertainty = max(
        0.0,
        1.0 - min(1.0, state.retrieval_margin / max(cfg.retrieval_margin_scale, 1e-6)),
    )
    evidence_uncertainty = score.uncertainty
    evidence_conflict = min(1.0, 2.0 * min(positive_confidence(score), score.contradiction))
    return min(
        1.0,
        0.30 * peak_disagreement
        + 0.20 * margin_uncertainty
        + 0.35 * evidence_uncertainty
        + 0.15 * evidence_conflict,
    )


def decide_next_action(
    state: CandidateEvidenceState,
    score: VerificationScore,
    cfg: VerificationBudgetConfig,
) -> ControlDecision:
    u = joint_uncertainty(state, score, cfg)
    if state.round_idx >= cfg.max_rounds:
        return ControlDecision("stop", u, "verification budget exhausted")

    pos = positive_confidence(score)
    if (
        score.uncertainty <= cfg.early_stop_uncertainty
        and score.contradiction >= cfg.reject_contradiction
        and pos <= cfg.reject_positive_max
    ):
        return ControlDecision("stop", u, "strong low-uncertainty counterevidence")

    if u <= cfg.early_stop_uncertainty and score.completeness >= 0.8 and score.contradiction <= 0.2:
        return ControlDecision("stop", u, "evidence is sufficiently complete and consistent")
    return ControlDecision("expand", u, "more local evidence is needed")
