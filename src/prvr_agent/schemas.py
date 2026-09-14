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
        for rel in self.temporal_constraints:
            if rel.event_a not in ids or rel.event_b not in ids:
                raise ValueError("Temporal constraints must reference existing atomic event ids")
        for rel in self.identity_constraints:
            if rel.event_a not in ids or rel.event_b not in ids:
                raise ValueError("Identity constraints must reference existing atomic event ids")
        return self


class Candidate(BaseModel):
    video_id: str
    video_index: int
    base_score: float
    clip_score: float
    frame_score: float
    clip_peak_index: int
    frame_peak_index: int
    metadata: dict = Field(default_factory=dict)


class EvidenceResult(BaseModel):
    mode: Literal["support", "refute"]
    matched: bool
    support: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(ge=0.0, le=1.0)
    start_time: float | None = None
    end_time: float | None = None
    observations: list[str] = Field(default_factory=list)
    verified_event_ids: list[str] = Field(default_factory=list)
    verified_relation_ids: list[str] = Field(default_factory=list)
    entity_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    action_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""

    @model_validator(mode="after")
    def validate_interval(self) -> "EvidenceResult":
        if self.start_time is not None and self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")
        return self
