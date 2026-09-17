from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PRVRMetrics:
    """Official DreamPRVR-style text-to-video recall summary.

    DreamPRVR reports R@1, R@5, R@10, R@100 and defines ``Rsum`` as the sum of
    all four recalls.  Values are percentages in [0, 100].
    """

    r1: float
    r5: float
    r10: float
    r100: float
    rsum: float

    def as_dict(self) -> dict[str, float]:
        return {
            "R@1": self.r1,
            "R@5": self.r5,
            "R@10": self.r10,
            "R@100": self.r100,
            "Rsum": self.rsum,
        }


def _video_id_from_query_id(query_id: str) -> str:
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("query ids must be non-empty strings")
    return query_id.split("#", 1)[0]


def build_query_to_video_ground_truth(
    video_ids: Sequence[str],
    query_ids: Sequence[str],
) -> dict[int, list[int]]:
    """Reproduce DreamPRVR's query-to-video ground-truth construction.

    PRVR benchmark caption/query identifiers use ``video_id#...``.  Multiple
    collection positions may theoretically share the same video id, so the
    mapping stores a list of valid video indices just like upstream DreamPRVR.
    """

    canonical_video_ids = [str(video_id) for video_id in video_ids]
    if not canonical_video_ids:
        raise ValueError("video_ids must not be empty")
    if len(canonical_video_ids) != len(set(canonical_video_ids)):
        # Upstream can technically represent duplicate ids, but a retrieval score
        # matrix over duplicate collection entries is ambiguous for our runner.
        raise ValueError("video_ids must be unique in the retrieval collection")

    video_to_index = {video_id: idx for idx, video_id in enumerate(canonical_video_ids)}
    q2v: dict[int, list[int]] = {}
    for query_index, query_id in enumerate(query_ids):
        video_id = _video_id_from_query_id(query_id)
        if video_id not in video_to_index:
            raise ValueError(
                f"query {query_id!r} references video {video_id!r}, which is absent from the collection"
            )
        q2v[query_index] = [video_to_index[video_id]]
    if not q2v:
        raise ValueError("query_ids must not be empty")
    return q2v


def recall_at_k(
    scores: Sequence[Sequence[float]] | np.ndarray,
    query_to_video_gt: Mapping[int, Sequence[int]],
    *,
    ks: Sequence[int] = (1, 5, 10, 100),
    higher_is_better: bool = True,
) -> dict[int, float]:
    """Compute best-ground-truth text-to-video recall at the requested cutoffs.

    This follows DreamPRVR's evaluation semantics: for each query, rank collection
    videos, take the best rank among all valid videos, and report the percentage
    of queries whose best rank is no larger than ``k``.
    """

    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("scores must be a [num_queries, num_videos] matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("scores must contain at least one query and one video")
    if not np.isfinite(matrix).all():
        raise ValueError("scores must be finite")

    cutoffs = [int(k) for k in ks]
    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("all recall cutoffs must be positive")
    if len(cutoffs) != len(set(cutoffs)):
        raise ValueError("recall cutoffs must be unique")

    num_queries, num_videos = matrix.shape
    if set(query_to_video_gt) != set(range(num_queries)):
        raise ValueError("query_to_video_gt must contain exactly one entry for every query row")

    best_ranks = np.empty(num_queries, dtype=np.int64)
    for query_index in range(num_queries):
        gt_indices = [int(index) for index in query_to_video_gt[query_index]]
        if not gt_indices:
            raise ValueError(f"query {query_index} has no ground-truth video")
        if any(index < 0 or index >= num_videos for index in gt_indices):
            raise ValueError(f"query {query_index} contains an out-of-range ground-truth index")

        row = matrix[query_index]
        # DreamPRVR passes negative similarity to an ascending argsort.  Sorting
        # negative similarity here reproduces the same ranking direction while a
        # stable sort makes exact ties deterministic for tests and diagnostics.
        ordering = np.argsort(-row if higher_is_better else row, kind="stable")
        inverse_rank = np.empty(num_videos, dtype=np.int64)
        inverse_rank[ordering] = np.arange(1, num_videos + 1, dtype=np.int64)
        best_ranks[query_index] = min(int(inverse_rank[index]) for index in gt_indices)

    return {
        cutoff: 100.0 * float(np.count_nonzero(best_ranks <= cutoff)) / float(num_queries)
        for cutoff in cutoffs
    }


def evaluate_prvr_scores(
    scores: Sequence[Sequence[float]] | np.ndarray,
    *,
    video_ids: Sequence[str],
    query_ids: Sequence[str],
    higher_is_better: bool = True,
) -> PRVRMetrics:
    """Evaluate a full PRVR score matrix using DreamPRVR's reported metrics."""

    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("scores must be a 2D matrix")
    if matrix.shape != (len(query_ids), len(video_ids)):
        raise ValueError(
            "score matrix shape must equal (len(query_ids), len(video_ids)); "
            f"got {matrix.shape}, expected {(len(query_ids), len(video_ids))}"
        )

    q2v = build_query_to_video_ground_truth(video_ids, query_ids)
    recalls = recall_at_k(
        matrix,
        q2v,
        ks=(1, 5, 10, 100),
        higher_is_better=higher_is_better,
    )
    rsum = recalls[1] + recalls[5] + recalls[10] + recalls[100]
    return PRVRMetrics(
        r1=recalls[1],
        r5=recalls[5],
        r10=recalls[10],
        r100=recalls[100],
        rsum=rsum,
    )
