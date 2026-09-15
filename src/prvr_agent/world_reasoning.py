from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import CandidateWorldObservation, ProspectiveWorldSet


@dataclass(frozen=True)
class WorldRevisionConfig:
    support_scale: float = 2.0
    contradiction_scale: float = 2.0
    epsilon: float = 1e-8


@dataclass(frozen=True)
class WorldBelief:
    world_id: str
    prior: float
    posterior: float
    support: float
    contradiction: float


@dataclass(frozen=True)
class ScoreFusionConfig:
    """Small additive reranking terms on top of the DreamPRVR base score."""

    cqhg_weight: float = 0.20
    counterfactual_weight: float = 0.15
    world_weight: float = 0.10


def revise_world_beliefs(
    worlds: ProspectiveWorldSet,
    observation: CandidateWorldObservation,
    cfg: WorldRevisionConfig | None = None,
) -> tuple[WorldBelief, ...]:
    cfg = cfg or WorldRevisionConfig()
    evidence = {item.world_id: item for item in observation.world_evidence}
    world_ids = {world.id for world in worlds.worlds}
    unknown = set(evidence) - world_ids
    if unknown:
        raise ValueError(f"Observation references unknown prospective world ids: {sorted(unknown)}")

    logits: list[float] = []
    aligned = []
    for world in worlds.worlds:
        item = evidence.get(world.id)
        support = item.support if item is not None else 0.0
        contradiction = item.contradiction if item is not None else 0.0
        prior = max(float(world.prior), cfg.epsilon)
        logit = math.log(prior) + cfg.support_scale * support - cfg.contradiction_scale * contradiction
        logits.append(logit)
        aligned.append((world, support, contradiction))

    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    normalizer = sum(exp_values)
    posteriors = [value / normalizer for value in exp_values]

    return tuple(
        WorldBelief(
            world_id=world.id,
            prior=float(world.prior),
            posterior=posterior,
            support=support,
            contradiction=contradiction,
        )
        for (world, support, contradiction), posterior in zip(aligned, posteriors)
    )


def prospective_world_score(beliefs: tuple[WorldBelief, ...]) -> float:
    if not beliefs:
        return 0.0
    return sum(item.posterior * (item.support - item.contradiction) for item in beliefs)


def fuse_candidate_score(
    base_score: float,
    observation: CandidateWorldObservation,
    world_score: float,
    cfg: ScoreFusionConfig | None = None,
) -> float:
    cfg = cfg or ScoreFusionConfig()
    cqhg_delta = cfg.cqhg_weight * (observation.query_satisfaction - 0.5)
    counterfactual_delta = cfg.counterfactual_weight * observation.counterfactual_risk
    world_delta = cfg.world_weight * world_score
    return float(base_score) + cqhg_delta - counterfactual_delta + world_delta
