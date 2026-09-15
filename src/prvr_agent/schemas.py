from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base schema that rejects silent field drift and non-finite values."""

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
        if len(event_ids) != len(set(event_ids)):
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
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("Temporal and identity constraint ids must be globally unique")

        counterfactual_ids = [cf.id for cf in self.counterfactuals]
        if len(counterfactual_ids) != len(set(counterfactual_ids)):
            raise ValueError("Counterfactual hypothesis ids must be unique")
        return self


class Candidate(StrictModel):
    """DreamPRVR candidate without any peak-specific state."""

    video_id: str = Field(min_length=1)
    video_index: int = Field(ge=0)
    base_score: float
    clip_score: float
    frame_score: float
    metadata: dict[str, object] = Field(default_factory=dict)


class EventWorld(StrictModel):
    """One plausible global event trajectory anchored by the full CQHG semantics."""

    id: str = Field(min_length=1)
    preconditions: list[str] = Field(default_factory=list)
    query_anchor_event_ids: list[str] = Field(min_length=1)
    query_anchor_relation_ids: list[str] = Field(default_factory=list)
    consequences: list[str] = Field(default_factory=list)
    prior: float = Field(gt=0.0, le=1.0)
    rationale: str = ""

    @model_validator(mode="after")
    def validate_anchor_ids(self) -> "EventWorld":
        if len(self.query_anchor_event_ids) != len(set(self.query_anchor_event_ids)):
            raise ValueError("Each CQHG anchor event id must appear exactly once in an event world")
        if len(self.query_anchor_relation_ids) != len(set(self.query_anchor_relation_ids)):
            raise ValueError("Each CQHG anchor relation id must appear exactly once in an event world")
        return self


class EventWorldSet(StrictModel):
    query: str = Field(min_length=1)
    worlds: list[EventWorld] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_worlds(self) -> "EventWorldSet":
        world_ids = [world.id for world in self.worlds]
        if len(world_ids) != len(set(world_ids)):
            raise ValueError("Event world ids must be unique")
        return self


class WorldEvidence(StrictModel):
    """Coarse visual evidence for one imagined event world."""

    world_id: str = Field(min_length=1)
    support: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    observations: list[str] = Field(default_factory=list)


class WorldEvidenceBundle(StrictModel):
    """One coarse candidate observation serving both CQHG and APEI."""

    candidate_video_id: str = Field(min_length=1)
    query_support: float = Field(ge=0.0, le=1.0)
    query_contradiction: float = Field(ge=0.0, le=1.0)
    query_uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    verified_event_ids: list[str] = Field(default_factory=list)
    verified_relation_ids: list[str] = Field(default_factory=list)
    supported_counterfactual_ids: list[str] = Field(default_factory=list)
    evidence: list[WorldEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> "WorldEvidenceBundle":
        world_ids = [item.world_id for item in self.evidence]
        if len(world_ids) != len(set(world_ids)):
            raise ValueError("Each world may appear at most once in an evidence bundle")
        for name, ids in (
            ("verified_event_ids", self.verified_event_ids),
            ("verified_relation_ids", self.verified_relation_ids),
            ("supported_counterfactual_ids", self.supported_counterfactual_ids),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} must not contain duplicate ids")
        return self
