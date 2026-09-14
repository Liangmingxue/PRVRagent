from __future__ import annotations

from dataclasses import dataclass

from .schemas import EvidenceResult


@dataclass(frozen=True)
class VerificationScore:
    atomic: float
    temporal: float
    identity: float
    completeness: float
    contradiction: float
    uncertainty: float


@dataclass(frozen=True)
class ScoreFusionConfig:
    base_weight: float = 0.65
    verification_weight: float = 0.35
    atomic_weight: float = 0.25
    temporal_weight: float = 0.20
    identity_weight: float = 0.15
    completeness_weight: float = 0.25
    contradiction_weight: float = 0.30
    uncertainty_penalty: float = 0.10


def summarize_evidence(support: EvidenceResult, refute: EvidenceResult) -> VerificationScore:
    atomic = max(0.0, min(1.0, support.support))
    temporal = max(0.0, min(1.0, support.temporal_consistency))
    identity = max(0.0, min(1.0, support.entity_consistency))
    completeness = max(0.0, min(1.0, support.action_completeness))
    contradiction = max(refute.support, support.contradiction, refute.contradiction)
    uncertainty = max(support.uncertainty, refute.uncertainty)
    return VerificationScore(
        atomic=atomic,
        temporal=temporal,
        identity=identity,
        completeness=completeness,
        contradiction=contradiction,
        uncertainty=uncertainty,
    )


def verification_score(v: VerificationScore, cfg: ScoreFusionConfig) -> float:
    positive = (
        cfg.atomic_weight * v.atomic
        + cfg.temporal_weight * v.temporal
        + cfg.identity_weight * v.identity
        + cfg.completeness_weight * v.completeness
    )
    negative = cfg.contradiction_weight * v.contradiction + cfg.uncertainty_penalty * v.uncertainty
    return positive - negative


def fuse_candidate_score(base_score: float, verification: VerificationScore, cfg: ScoreFusionConfig) -> float:
    return cfg.base_weight * float(base_score) + cfg.verification_weight * verification_score(verification, cfg)
