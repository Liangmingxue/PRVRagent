from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AtomicEvent(StrictModel):
    id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    action: str = Field(min_length=1)
    object: Optional[str] = None
    attributes: List[str] = Field(default_factory=list)


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
    atomic_events: List[AtomicEvent] = Field(min_length=1)
    temporal_constraints: List[TemporalConstraint] = Field(default_factory=list)
    identity_constraints: List[IdentityConstraint] = Field(default_factory=list)
    positive_hypothesis: str = Field(min_length=1)
    counterfactuals: List[CounterfactualHypothesis] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "QueryHypothesisGraph":
        event_ids = [event.id for event in self.atomic_events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Atomic event ids must be unique")
        ids = set(event_ids)

        relation_ids: List[str] = []
        for rel in self.temporal_constraints:
            relation_ids.append(rel.id)
            if rel.event_a not in ids or rel.event_b not in ids:
                raise ValueError("Temporal constraints must reference existing atomic event ids")
            if rel.event_a == rel.event_b:
                raise ValueError("Temporal constraints cannot relate an event to itself")
        for rel in self.identity_constraints:
            relation_ids.append(rel.id)
            if rel.event_a not in ids or rel.event_b not in ids:
                raise ValueError("Identity constraints must reference existing atomic event ids")
            if rel.event_a == rel.event_b:
                raise ValueError("Identity constraints cannot relate an event to itself")
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError("Temporal and identity constraint ids must be globally unique")

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
    clip_peak_index: int = Field(ge=0)
    frame_peak_index: int = Field(ge=0)
    metadata: Dict[str, object] = Field(default_factory=dict)


class EvidenceResult(StrictModel):
    mode: Literal["support", "refute"]
    matched: bool = Field(
        description=(
            "Whether the current agent's target hypothesis matches the observed frames. "
            "For support mode the target is the positive query hypothesis; for refute mode "
            "the target is a counterfactual/near-miss hypothesis."
        )
    )
    support: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Evidence FOR the current agent's target hypothesis. In refute mode, high support "
            "means strong evidence that a counterfactual/semantic near-miss is present."
        ),
    )
    contradiction: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Evidence AGAINST the current agent's target hypothesis. In refute mode, high "
            "contradiction means the falsifier's counterfactual is itself contradicted."
        ),
    )
    start_time: Optional[float] = Field(default=None, ge=0.0)
    end_time: Optional[float] = Field(default=None, ge=0.0)
    observations: List[str] = Field(default_factory=list)
    verified_event_ids: List[str] = Field(default_factory=list)
    verified_relation_ids: List[str] = Field(default_factory=list)
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
