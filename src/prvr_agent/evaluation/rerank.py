from __future__ import annotations

from typing import Sequence

import numpy as np

from prvr_agent.pipeline import RerankedCandidate


def rerank_topk_score_slots(
    base_scores: Sequence[float] | np.ndarray,
    reranked: Sequence[RerankedCandidate],
    *,
    require_exact_base_topk: bool = True,
) -> np.ndarray:
    """Apply an APEI ordering without mixing incompatible score scales.

    ``PRVRAgentReranker`` produces a fused score only for the expensive shortlist.
    Directly writing those fused values into an otherwise untouched DreamPRVR row
    would compare transformed Top-K scores with raw scores outside Top-K.  Instead
    this function treats the original Top-K scores as *rank slots*: APEI decides
    which shortlisted video occupies each slot, while the multiset of DreamPRVR
    scores and every score outside the shortlist remain unchanged.

    This makes benchmark evaluation a true Top-K reranking experiment. It does
    not let an agent-only score accidentally promote/demote the entire shortlist
    merely because its numerical scale differs from DreamPRVR cosine similarity.
    """

    row = np.asarray(base_scores, dtype=np.float64)
    if row.ndim != 1 or row.size == 0:
        raise ValueError("base_scores must be a non-empty 1D score row")
    if not np.isfinite(row).all():
        raise ValueError("base_scores must be finite")
    if not reranked:
        return row.copy()

    video_indices = [int(item.candidate.video_index) for item in reranked]
    if len(video_indices) != len(set(video_indices)):
        raise ValueError("reranked candidates must reference unique video indices")
    if any(index < 0 or index >= row.size for index in video_indices):
        raise ValueError("reranked candidate contains an out-of-range video index")
    if any(not np.isfinite(float(item.final_score)) for item in reranked):
        raise ValueError("reranked candidate scores must be finite")

    k = len(video_indices)
    if require_exact_base_topk:
        expected = set(np.argsort(-row, kind="stable")[:k].tolist())
        observed = set(video_indices)
        if observed != expected:
            raise ValueError(
                "reranked candidates must be exactly the DreamPRVR Top-K when "
                "require_exact_base_topk=True"
            )

    # APEI decides the within-shortlist order. Ties fall back to the original base
    # score and then video index for deterministic diagnostics.
    ordered_candidates = sorted(
        reranked,
        key=lambda item: (
            -float(item.final_score),
            -float(item.candidate.base_score),
            int(item.candidate.video_index),
        ),
    )
    target_indices = [int(item.candidate.video_index) for item in ordered_candidates]

    # Preserve the original score slots exactly. With distinct base scores this
    # preserves Top-K membership and its boundary relative to the untouched tail.
    score_slots = sorted((float(row[index]) for index in video_indices), reverse=True)
    output = row.copy()
    for video_index, score_slot in zip(target_indices, score_slots):
        output[video_index] = score_slot
    return output


def rerank_score_matrix_rows(
    base_score_matrix: Sequence[Sequence[float]] | np.ndarray,
    reranked_rows: Sequence[Sequence[RerankedCandidate]],
    *,
    require_exact_base_topk: bool = True,
) -> np.ndarray:
    """Apply scale-safe shortlist reranking to every query row in a matrix."""

    matrix = np.asarray(base_score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("base_score_matrix must be 2D")
    if len(reranked_rows) != matrix.shape[0]:
        raise ValueError("reranked_rows must contain exactly one shortlist per query")

    output = matrix.copy()
    for query_index, reranked in enumerate(reranked_rows):
        output[query_index] = rerank_topk_score_slots(
            matrix[query_index],
            reranked,
            require_exact_base_topk=require_exact_base_topk,
        )
    return output
