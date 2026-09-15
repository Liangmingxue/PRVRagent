from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import EventWorldSet, QueryHypothesisGraph, WorldEvidenceBundle


@dataclass(frozen=True)
class ProspectiveConfig:
    """Scoring parameters for CQHG evidence and event-world belief revision."""

    base_weight: float = 0.60
    graph_weight: float = 0.20
    world_weight: float = 0.20
    support_scale: float = 2.0
    contradiction_scale: float = 2.0
    uncertainty_penalty: float = 0.15

    def validate(self) -> None:
        values = {
            "base_weight": self.base_weight,
            "graph_weight": self.graph_weight,
            "world_weight": self.world_weight,
            "support_scale": self.support_scale,
            "contradiction_scale": self.contradiction_scale,
            "uncertainty_penalty": self.uncertainty_penalty,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.base_weight < 0 or self.graph_weight < 0 or self.world_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if self.base_weight + self.graph_weight + self.world_weight <= 0:
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
    graph_score: float
    world_score: float


def _normalized_priors(worlds: EventWorldSet) -> dict[str, float]:
    total = sum(float(world.prior) for world in worlds.worlds)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("event-world priors must sum to a finite positive value")
    return {world.id: float(world.prior) / total for world in worlds.worlds}


def _validate_world_anchors(graph: QueryHypothesisGraph, worlds: EventWorldSet) -> None:
    if worlds.query != graph.query:
        raise ValueError("event worlds do not match the CQHG query")
    expected_events = {event.id for event in graph.atomic_events}
    expected_relations = {
        rel.id for rel in list(graph.temporal_constraints) + list(graph.identity_constraints)
    }
    for world in worlds.worlds:
        if set(world.query_anchor_event_ids) != expected_events or len(world.query_anchor_event_ids) != len(expected_events):
            raise ValueError(f"world {world.id!r} does not preserve the complete CQHG event anchors")
        if set(world.query_anchor_relation_ids) != expected_relations or len(world.query_anchor_relation_ids) != len(expected_relations):
            raise ValueError(f"world {world.id!r} does not preserve the complete CQHG relation anchors")


def score_cqhg_evidence(
    graph: QueryHypothesisGraph,
    evidence: WorldEvidenceBundle,
    cfg: ProspectiveConfig,
) -> float:
    """Score complete CQHG satisfaction, including hard relation coverage.

    Uncertainty reduces the strength of a judgment toward zero; it is not itself
    negative evidence. This matters in PRVR because sparse observation of a long
    video can simply fail to see the relevant local moment.
    """

    valid_event_ids = {event.id for event in graph.atomic_events}
    verified_events = valid_event_ids.intersection(evidence.verified_event_ids)
    event_coverage = len(verified_events) / len(valid_event_ids)

    valid_relation_ids = {
        rel.id for rel in list(graph.temporal_constraints) + list(graph.identity_constraints)
    }
    if valid_relation_ids:
        verified_relations = valid_relation_ids.intersection(evidence.verified_relation_ids)
        relation_coverage = len(verified_relations) / len(valid_relation_ids)
    else:
        relation_coverage = 1.0

    positive = min(float(evidence.query_support), event_coverage, relation_coverage)
    contradiction = float(evidence.query_contradiction)
    confidence = max(0.0, 1.0 - float(evidence.query_uncertainty))
    return confidence * (positive - contradiction)


def revise_world_beliefs(
    graph: QueryHypothesisGraph,
    worlds: EventWorldSet,
    evidence: WorldEvidenceBundle,
    cfg: ProspectiveConfig | None = None,
) -> ProspectiveAssessment:
    """Revise possible-world priors with candidate evidence while preserving CQHG semantics."""

    cfg = cfg or ProspectiveConfig()
    cfg.validate()
    _validate_world_anchors(graph, worlds)
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
            - cfg.uncertainty_penalty * item.uncertainty
        )

    max_logit = max(logits.values())
    unnormalized = {world_id: math.exp(logit - max_logit) for world_id, logit in logits.items()}
    normalizer = sum(unnormalized.values())
    if not math.isfinite(normalizer) or normalizer <= 0:  # pragma: no cover - defensive
        raise ValueError("event-world posterior normalization failed")

    beliefs: list[WorldBelief] = []
    world_score = 0.0
    positive_world_gate = max(0.0, 1.0 - float(evidence.query_contradiction))
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
        evidence_confidence = max(0.0, 1.0 - item.uncertainty)
        local_score = evidence_confidence * (item.support - item.contradiction)
        # Soft imagined context may help when the hard query is unresolved, but it
        # must not rescue a candidate with explicit CQHG contradiction evidence.
        if local_score > 0:
            local_score *= positive_world_gate
        world_score += posterior * local_score

    return ProspectiveAssessment(
        beliefs=tuple(beliefs),
        graph_score=score_cqhg_evidence(graph, evidence, cfg),
        world_score=world_score,
    )


def fuse_prospective_score(base_score: float, assessment: ProspectiveAssessment, cfg: ProspectiveConfig) -> float:
    cfg.validate()
    if not math.isfinite(float(base_score)):
        raise ValueError("base_score must be finite")
    score = (
        cfg.base_weight * float(base_score)
        + cfg.graph_weight * float(assessment.graph_score)
        + cfg.world_weight * float(assessment.world_score)
    )
    if not math.isfinite(score):
        raise ValueError("fused prospective score is non-finite")
    return score
