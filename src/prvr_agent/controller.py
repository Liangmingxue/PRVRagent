from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class ControlDecision:
    action: str  # stop | expand
    uncertainty: float
    reason: str


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
    contradiction_conflict = min(1.0, score.contradiction)
    return min(
        1.0,
        0.30 * peak_disagreement
        + 0.20 * margin_uncertainty
        + 0.35 * evidence_uncertainty
        + 0.15 * contradiction_conflict,
    )


def decide_next_action(
    state: CandidateEvidenceState,
    score: VerificationScore,
    cfg: VerificationBudgetConfig,
) -> ControlDecision:
    u = joint_uncertainty(state, score, cfg)
    if state.round_idx >= cfg.max_rounds:
        return ControlDecision("stop", u, "verification budget exhausted")
    if u <= cfg.early_stop_uncertainty and score.completeness >= 0.8 and score.contradiction <= 0.2:
        return ControlDecision("stop", u, "evidence is sufficiently complete and consistent")
    return ControlDecision("expand", u, "more local evidence is needed")
