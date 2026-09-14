from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from prvr_agent.schemas import Candidate


@dataclass(frozen=True)
class RetrievalBatch:
    candidates: list[list[Candidate]]


class DreamPRVRAdapter:
    """Non-invasive adapter around a loaded DreamPRVR model and cached context features.

    This class deliberately does not vendor DreamPRVR. It expects an upstream model exposing
    ``encode_query`` and cached context tensors named ``video_proposal_feat`` and ``video_feat``.
    It recomputes the per-location similarities so the argmax locations are preserved rather than
    discarded by the original retrieval API.
    """

    def __init__(
        self,
        model,
        context_info: dict,
        video_ids: Sequence[str] | None = None,
        *,
        clip_scale_weight: float = 0.5,
        frame_scale_weight: float = 0.5,
    ) -> None:
        self.model = model
        self.context_info = context_info
        self.video_ids = list(video_ids or context_info.get("video_metas") or [])
        self.clip_scale_weight = float(clip_scale_weight)
        self.frame_scale_weight = float(frame_scale_weight)
        if not self.video_ids:
            raise ValueError("video_ids or context_info['video_metas'] is required")

    @staticmethod
    def _scores_and_peaks(query_vectors, context_features):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DreamPRVRAdapter requires the optional 'torch' dependency") from exc

        if query_vectors.ndim == 1:
            query_vectors = query_vectors.unsqueeze(0)
        if context_features.ndim != 3:
            raise ValueError(
                f"Expected inference context with shape [video, location, dim], got {tuple(context_features.shape)}"
            )
        q = F.normalize(query_vectors, dim=-1)
        ctx = F.normalize(context_features, dim=-1)
        location_scores = torch.einsum("qd,vld->qvl", q, ctx)
        return location_scores.max(dim=-1)

    def retrieve(self, query_feat, query_mask, *, top_k: int = 20) -> RetrievalBatch:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DreamPRVRAdapter requires the optional 'torch' dependency") from exc

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        with torch.no_grad():
            query_vectors = self.model.encode_query(query_feat, query_mask)
            clip_scores, clip_peaks = self._scores_and_peaks(
                query_vectors, self.context_info["video_proposal_feat"]
            )
            frame_scores, frame_peaks = self._scores_and_peaks(
                query_vectors, self.context_info["video_feat"]
            )
            fused = self.clip_scale_weight * clip_scores + self.frame_scale_weight * frame_scores
            k = min(int(top_k), fused.shape[1])
            _, top_indices = fused.topk(k=k, dim=1)

        all_candidates: list[list[Candidate]] = []
        for q_idx in range(fused.shape[0]):
            row: list[Candidate] = []
            for vid_idx in top_indices[q_idx].tolist():
                row.append(
                    Candidate(
                        video_id=str(self.video_ids[vid_idx]),
                        video_index=int(vid_idx),
                        base_score=float(fused[q_idx, vid_idx].item()),
                        clip_score=float(clip_scores[q_idx, vid_idx].item()),
                        frame_score=float(frame_scores[q_idx, vid_idx].item()),
                        clip_peak_index=int(clip_peaks[q_idx, vid_idx].item()),
                        frame_peak_index=int(frame_peaks[q_idx, vid_idx].item()),
                    )
                )
            all_candidates.append(row)
        return RetrievalBatch(candidates=all_candidates)
