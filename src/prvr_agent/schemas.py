from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AtomicEvent(BaseModel):
    id: str
    subject: str
    action: str
    object: str | None = None
    attributes: list[str] = Field(default_factory=list)


class TemporalConstraint(BaseModel):
    event_a: str
    relation: Literal["before", "after", "during", "overlap"]
    event_b: str


class IdentityConstraint(BaseModel):
    event_a: str
    event_b: str
    same_actor: bool = True


class CounterfactualHypothesis(BaseModel):
    id: str
    type: Literal[
        "temporal_reversal",
        "identity_break",
        "partial_event",
        "wrong_object",
        "wrong_action",
        "other",
    ]
    description: str


class QueryHypothesisGraph(BaseModel):
    query: str
    atomic_events: list[AtomicEvent]
    temporal_constraints: list[TemporalConstraint] = Field(default_factory=list)
    identity_constraints: list[IdentityConstraint] = Field(default_factory=list)
    positive_hypothesis: str
    counterfactuals: list[CounterfactualHypothesis] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "QueryHypothesisGraph":
        ids = {e.id for e in self.atomic_events}
        if len(ids) != len(self.atomic_events):
            raise ValueError("Atomic event ids must be unique")
        for rel in self.temporal_constraints:
            if rel.event_a not in ids or rel.event_b not in ids:
                raise ValueError("Temporal constraints must reference existing atomic event ids")
        for rel in self.identity_constraints:
            if rel.event_a not in ids or rel.event_b not in ids:
                raise ValueError("Identity constraints must reference existing atomic event ids")
        return self


class Candidate(BaseModel):
    """DreamPRVR candidate without any peak-specific state."""

    video_id: str
    video_index: int
    base_score: float
    clip_score: float
    frame_score: float
    metadata: dict = Field(default_factory=dict)


class EventWorld(BaseModel):
    """One plausible global event trajectory anchored by the CQHG query event."""

    id: str
    preconditions: list[str] = Field(default_factory=list)
    query_anchor_event_ids: list[str] = Field(default_factory=list)
    consequences: list[str] = Field(default_factory=list)
    prior: float = Field(gt=0.0, le=1.0)
    rationale: str = ""


class EventWorldSet(BaseModel):
    query: str
    worlds: list[EventWorld] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_worlds(self) -> "EventWorldSet":
        world_ids = [world.id for world in self.worlds]
        if len(world_ids) != len(set(world_ids)):
            raise ValueError("Event world ids must be unique")
        return self


class WorldEvidence(BaseModel):
    """Coarse visual evidence for one imagined event world."""

    world_id: str
    support: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    observations: list[str] = Field(default_factory=list)


class WorldEvidenceBundle(BaseModel):
    """One coarse candidate observation serving both CQHG and APEI.

    `query_support` measures evidence that the hard CQHG positive hypothesis is
    completely satisfied. `query_contradiction` measures evidence for a CQHG
    counterfactual/near-miss. The per-world evidence is soft prospective context.
    """

    candidate_video_id: str
    query_support: float = Field(ge=0.0, le=1.0)
    query_contradiction: float = Field(ge=0.0, le=1.0)
    query_uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    verified_event_ids: list[str] = Field(default_factory=list)
    supported_counterfactual_ids: list[str] = Field(default_factory=list)
    evidence: list[WorldEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> "WorldEvidenceBundle":
        ids = [item.world_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("Each world may appear at most once in an evidence bundle")
        return self
