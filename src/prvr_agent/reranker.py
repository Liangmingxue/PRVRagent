from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Optional, Set

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

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.base_weight == 0 and self.verification_weight == 0:
            raise ValueError("base_weight and verification_weight cannot both be zero")


def _coverage(found_ids, expected_ids: Optional[Set[str]]) -> Optional[float]:
    if expected_ids is None:
        return None
    if not expected_ids:
        return 1.0
    return len(set(found_ids).intersection(expected_ids)) / len(expected_ids)


def summarize_evidence(
    support: EvidenceResult,
    refute: EvidenceResult,
    *,
    expected_event_ids: Optional[Set[str]] = None,
    expected_temporal_ids: Optional[Set[str]] = None,
    expected_identity_ids: Optional[Set[str]] = None,
) -> VerificationScore:
    """Summarize one coherent support/refute observation round."""

    event_coverage = _coverage(support.verified_event_ids, expected_event_ids)
    temporal_coverage = _coverage(support.verified_relation_ids, expected_temporal_ids)
    identity_coverage = _coverage(support.verified_relation_ids, expected_identity_ids)

    atomic = max(0.0, min(1.0, support.support))
    completeness = max(0.0, min(1.0, support.action_completeness))
    if event_coverage is not None:
        atomic = min(atomic, event_coverage)
        completeness = min(completeness, event_coverage)

    temporal = max(0.0, min(1.0, support.temporal_consistency))
    if temporal_coverage is not None:
        temporal = 1.0 if not expected_temporal_ids else min(temporal, temporal_coverage)

    identity = max(0.0, min(1.0, support.entity_consistency))
    if identity_coverage is not None:
        identity = 1.0 if not expected_identity_ids else min(identity, identity_coverage)

    return VerificationScore(
        atomic=atomic,
        temporal=temporal,
        identity=identity,
        completeness=completeness,
        contradiction=max(refute.support, support.contradiction),
        uncertainty=max(support.uncertainty, refute.uncertainty),
    )


def _positive_confidence(v: VerificationScore) -> float:
    return mean((v.atomic, v.temporal, v.identity, v.completeness))


def aggregate_evidence(
    supports: list[EvidenceResult],
    refutes: list[EvidenceResult],
    *,
    expected_event_ids: Optional[Set[str]] = None,
    expected_temporal_ids: Optional[Set[str]] = None,
    expected_identity_ids: Optional[Set[str]] = None,
) -> VerificationScore:
    if not supports or not refutes or len(supports) != len(refutes):
        raise ValueError("support/refute evidence must be non-empty and aligned by round")
    per_round = [
        summarize_evidence(
            support,
            refute,
            expected_event_ids=expected_event_ids,
            expected_temporal_ids=expected_temporal_ids,
            expected_identity_ids=expected_identity_ids,
        )
        for support, refute in zip(supports, refutes)
    ]

    best = max(per_round, key=lambda value: _positive_confidence(value) - 0.25 * value.uncertainty)
    return VerificationScore(
        atomic=best.atomic,
        temporal=best.temporal,
        identity=best.identity,
        completeness=best.completeness,
        contradiction=max(v.contradiction for v in per_round),
        uncertainty=mean(v.uncertainty for v in per_round),
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
