"""PRVR-Agent: counterfactual query reasoning and prospective event-world modeling."""

from .schemas import (
    AtomicEvent,
    Candidate,
    CandidateWorldObservation,
    CounterfactualHypothesis,
    IdentityConstraint,
    ProspectiveEventWorld,
    ProspectiveWorldSet,
    QueryHypothesisGraph,
    TemporalConstraint,
    WorldEvidence,
)
from .world_reasoning import (
    ScoreFusionConfig,
    WorldBelief,
    WorldRevisionConfig,
    fuse_candidate_score,
    prospective_world_score,
    revise_world_beliefs,
)

__all__ = [
    "AtomicEvent",
    "Candidate",
    "CandidateWorldObservation",
    "CounterfactualHypothesis",
    "IdentityConstraint",
    "ProspectiveEventWorld",
    "ProspectiveWorldSet",
    "QueryHypothesisGraph",
    "TemporalConstraint",
    "WorldEvidence",
    "ScoreFusionConfig",
    "WorldBelief",
    "WorldRevisionConfig",
    "fuse_candidate_score",
    "prospective_world_score",
    "revise_world_beliefs",
]
