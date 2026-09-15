from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AtomicEvent(StrictModel):
    id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    action: str = Field(min_length=1)
    object: str | None = None
    attributes: list[str] = Field(default_factory=list)


class TemporalConstraint(StrictModel):
    id: str = Field(min_length=1)
    event_a: str = Field(min_length=1)
    relation: Literal["before", "after", "during", "overlap"]
    event_b: str = Field(min_length=1)


class IdentityConstraint(StrictModel):
    id: str = Field(min_length=1)
    event_a: str = Field(min_length=1)
    event_b: str = Field(min_length=1)
    same_actor: bool = True


class CounterfactualHypothesis(StrictModel):
    id: str = Field(min_length=1)
    type: Literal[
        "temporal_reversal",
        "identity_break",
        "partial_event",
        "wrong_object",
        "wrong_action",
        "other",
    ]
    description: str = Field(min_length=1)


class QueryHypothesisGraph(StrictModel):
    query: str = Field(min_length=1)
    atomic_events: list[AtomicEvent] = Field(min_length=1)
    temporal_constraints: list[TemporalConstraint] = Field(default_factory=list)
    identity_constraints: list[IdentityConstraint] = Field(default_factory=list)
    positive_hypothesis: str = Field(min_length=1)
    counterfactuals: list[CounterfactualHypothesis] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "QueryHypothesisGraph":
        event_ids = [event.id for event in self.atomic_events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Atomic event ids must be unique")
        valid_event_ids = set(event_ids)

        relation_ids: list[str] = []
        for rel in self.temporal_constraints:
            relation_ids.append(rel.id)
            if rel.event_a not in valid_event_ids or rel.event_b not in valid_event_ids:
                raise ValueError("Temporal constraints must reference existing atomic event ids")
            if rel.event_a == rel.event_b:
                raise ValueError("Temporal constraints cannot relate an event to itself")
        for rel in self.identity_constraints:
            relation_ids.append(rel.id)
            if rel.event_a not in valid_event_ids or rel.event_b not in valid_event_ids:
                raise ValueError("Identity constraints must reference existing atomic event ids")
            if rel.event_a == rel.event_b:
                raise ValueError("Identity constraints cannot relate an event to itself")
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError("Temporal and identity relation ids must be globally unique")

        counterfactual_ids = [cf.id for cf in self.counterfactuals]
        if len(set(counterfactual_ids)) != len(counterfactual_ids):
            raise ValueError("Counterfactual hypothesis ids must be unique")
        return self


class Candidate(StrictModel):
    video_id: str = Field(min_length=1)
    video_index: int = Field(ge=0)
    base_score: float
    clip_score: float
    frame_score: float
    metadata: dict[str, object] = Field(default_factory=dict)


class ProspectiveEventWorld(StrictModel):
    """One plausible context in which the anchored query event could unfold."""

    id: str = Field(min_length=1)
    preconditions: list[str] = Field(default_factory=list)
    query_anchor: str = Field(min_length=1)
    consequences: list[str] = Field(default_factory=list)
    prior: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class ProspectiveWorldSet(StrictModel):
    query: str = Field(min_length=1)
    worlds: list[ProspectiveEventWorld] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_worlds(self) -> "ProspectiveWorldSet":
        world_ids = [world.id for world in self.worlds]
        if len(set(world_ids)) != len(world_ids):
            raise ValueError("Prospective world ids must be unique")
        if sum(world.prior for world in self.worlds) <= 0:
            raise ValueError("At least one prospective world must have positive prior mass")
        return self


class WorldEvidence(StrictModel):
    world_id: str = Field(min_length=1)
    support: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(ge=0.0, le=1.0)
    observations: list[str] = Field(default_factory=list)


class CandidateWorldObservation(StrictModel):
    """Coarse candidate-video evidence for CQHG satisfaction and event-world revision."""

    query_satisfaction: float = Field(
        ge=0.0,
        le=1.0,
        description="Evidence that the hard CQHG positive hypothesis is actually satisfied.",
    )
    counterfactual_risk: float = Field(
        ge=0.0,
        le=1.0,
        description="Evidence that a CQHG counterfactual/near-miss is present instead of a full match.",
    )
    world_evidence: list[WorldEvidence] = Field(default_factory=list)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    observations: list[str] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def validate_world_ids(self) -> "CandidateWorldObservation":
        ids = [item.world_id for item in self.world_evidence]
        if len(set(ids)) != len(ids):
            raise ValueError("world_evidence entries must have unique world ids")
        return self
