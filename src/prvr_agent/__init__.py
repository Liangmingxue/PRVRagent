"""PRVR-Agent: counterfactual query reasoning and prospective event-world modeling."""

from .prospective import (
    ProspectiveAssessment,
    ProspectiveConfig,
    WorldBelief,
    fuse_prospective_score,
    revise_world_beliefs,
    score_chunk_cqhg_evidence,
)
from .schemas import (
    AtomicEvent,
    Candidate,
    ChunkEvidence,
    ChunkEvidenceBundle,
    CounterfactualHypothesis,
    EventTimeRange,
    EventWorld,
    EventWorldSet,
    IdentityConstraint,
    QueryHypothesisGraph,
    TemporalConstraint,
    TemporalSegmentTrace,
    WorldEvidence,
    WorldEvidenceBundle,
)

__all__ = [
    "AtomicEvent",
    "Candidate",
    "ChunkEvidence",
    "ChunkEvidenceBundle",
    "CounterfactualHypothesis",
    "EventTimeRange",
    "EventWorld",
    "EventWorldSet",
    "IdentityConstraint",
    "QueryHypothesisGraph",
    "TemporalConstraint",
    "TemporalSegmentTrace",
    "WorldEvidence",
    "WorldEvidenceBundle",
    "ProspectiveAssessment",
    "ProspectiveConfig",
    "WorldBelief",
    "fuse_prospective_score",
    "revise_world_beliefs",
    "score_chunk_cqhg_evidence",
]
