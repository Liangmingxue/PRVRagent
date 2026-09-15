from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import EventWorldSet, WorldEvidenceBundle


@dataclass(frozen=True)
class ProspectiveConfig:
    """Scoring parameters for evidence-grounded event-world revision."""

    base_weight: float = 0.75
    world_weight: float = 0.25
    support_scale: float = 2.0
    contradiction_scale: float = 2.0
    uncertainty_penalty: float = 0.15

    def validate(self) -> None:
        if self.base_weight < 0 or self.world_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if self.base_weight + self.world_weight <= 0:
            raise ValueError("at least one fusion weight must be positive")
        if self.support_scale < 0 or self.contradiction_scale < 0:
            raise ValueError("belief-update scales must be non-negative")
        if self.uncertainty_penalty < 0:
            raise ValueError("uncertainty_penalty must be non-negative")


@dataclass(frozen=True)
class WorldBelief:
    world_id: str
    prior: float
    posterior: float
    support: float
    contradiction: float
    uncertainty: float


@dataclass(frozen=True)
class ProspectiveAssessment:
    beliefs: tuple[WorldBelief, ...]
    world_score: float


def _normalized_priors(worlds: EventWorldSet) -> dict[str, float]:
    total = sum(float(world.prior) for world in worlds.worlds)
    if total <= 0:
        raise ValueError("event-world priors must sum to a positive value")
    return {world.id: float(world.prior) / total for world in worlds.worlds}


def revise_world_beliefs(
    worlds: EventWorldSet,
    evidence: WorldEvidenceBundle,
    cfg: ProspectiveConfig | None = None,
) -> ProspectiveAssessment:
    """Update event-world priors with coarse visual support/contradiction evidence.

    This is a belief-revision step rather than query expansion: CQHG-anchored
    worlds remain fixed while their posterior probabilities change per candidate.
    """

    cfg = cfg or ProspectiveConfig()
    cfg.validate()
    priors = _normalized_priors(worlds)
    by_id = {item.world_id: item for item in evidence.evidence}
    expected = set(priors)
    observed = set(by_id)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"world evidence ids do not match imagined worlds; missing={missing}, extra={extra}")

    logits: dict[str, float] = {}
    for world_id, prior in priors.items():
        item = by_id[world_id]
        logits[world_id] = (
            math.log(max(prior, 1e-12))
            + cfg.support_scale * item.support
            - cfg.contradiction_scale * item.contradiction
        )

    max_logit = max(logits.values())
    unnormalized = {world_id: math.exp(logit - max_logit) for world_id, logit in logits.items()}
    normalizer = sum(unnormalized.values())

    beliefs: list[WorldBelief] = []
    world_score = 0.0
    for world in worlds.worlds:
        item = by_id[world.id]
        posterior = unnormalized[world.id] / normalizer
        beliefs.append(
            WorldBelief(
                world_id=world.id,
                prior=priors[world.id],
                posterior=posterior,
                support=item.support,
                contradiction=item.contradiction,
                uncertainty=item.uncertainty,
            )
        )
        world_score += posterior * (
            item.support - item.contradiction - cfg.uncertainty_penalty * item.uncertainty
        )

    return ProspectiveAssessment(beliefs=tuple(beliefs), world_score=world_score)


def fuse_prospective_score(base_score: float, assessment: ProspectiveAssessment, cfg: ProspectiveConfig) -> float:
    cfg.validate()
    return cfg.base_weight * float(base_score) + cfg.world_weight * float(assessment.world_score)
