from prvr_agent.agents.world_observer import select_refinement_chunk_indices
from prvr_agent.pipeline import PipelineConfig
from prvr_agent.schemas import (
    AtomicEvent,
    ChunkEvidence,
    EventWorld,
    QueryHypothesisGraph,
    WorldEvidence,
)


def _graph():
    return QueryHypothesisGraph(
        query="a person closes a door then sits",
        atomic_events=[
            AtomicEvent(id="E1", subject="person", action="closes", object="door"),
            AtomicEvent(id="E2", subject="person", action="sits"),
        ],
        positive_hypothesis="the person closes the door and then sits",
    )


def _empty_chunk(index: int) -> ChunkEvidence:
    return ChunkEvidence(
        chunk_index=index,
        start_time=float(index * 10),
        end_time=float((index + 1) * 10),
        query_support=0.0,
        query_contradiction=0.0,
        query_uncertainty=0.0,
        evidence=[
            WorldEvidence(
                world_id="H1",
                support=0.0,
                contradiction=0.0,
                uncertainty=0.0,
            )
        ],
    )


def test_default_pipeline_does_not_hard_prefilter_all_refinements():
    cfg = PipelineConfig()
    assert cfg.refinement_threshold == 0.0

    chunks = [_empty_chunk(index) for index in range(6)]
    selected = select_refinement_chunk_indices(
        _graph(),
        chunks,
        max_chunks=cfg.max_refinement_chunks,
        threshold=cfg.refinement_threshold,
    )

    # Even an overconfident coarse pass with no detected evidence must spend the
    # bounded refinement budget rather than silently terminating observation.
    assert len(selected) == cfg.max_refinement_chunks
    # With tied evidence scores, the existing diversity bonus should cover both
    # temporal extremes before allocating the remaining slot.
    assert 0 in selected
    assert 5 in selected


def test_positive_threshold_remains_available_for_ablation():
    chunks = [_empty_chunk(index) for index in range(4)]
    selected = select_refinement_chunk_indices(
        _graph(),
        chunks,
        max_chunks=3,
        threshold=0.30,
    )
    assert selected == []
