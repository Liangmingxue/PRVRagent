from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from prvr_agent.schemas import Candidate


@dataclass(frozen=True)
class RetrievalBatch:
    candidates: list[list[Candidate]]


class DreamPRVRAdapter:
    """Non-invasive adapter around a loaded DreamPRVR model and cached features.

    DreamPRVR remains responsible for its original clip/frame max-similarity scores.
    The agent consumes only video-level candidates; no local argmax peak is exported.
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
        required = {"video_proposal_feat", "video_feat"}
        missing = sorted(required.difference(context_info))
        if missing:
            raise ValueError(f"context_info is missing required keys: {missing}")

        self.model = model
        self.context_info = context_info
        if video_ids is not None:
            self.video_ids = list(video_ids)
        else:
            self.video_ids = list(context_info.get("video_metas") or [])
        self.clip_scale_weight = float(clip_scale_weight)
        self.frame_scale_weight = float(frame_scale_weight)

        if not self.video_ids:
            raise ValueError("video_ids or context_info['video_metas'] is required")
        if not math.isfinite(self.clip_scale_weight) or not math.isfinite(self.frame_scale_weight):
            raise ValueError("retrieval fusion weights must be finite")
        if self.clip_scale_weight < 0 or self.frame_scale_weight < 0:
            raise ValueError("retrieval fusion weights must be non-negative")
        if self.clip_scale_weight + self.frame_scale_weight <= 0:
            raise ValueError("at least one retrieval fusion weight must be positive")

        clip_features = context_info["video_proposal_feat"]
        frame_features = context_info["video_feat"]
        if getattr(clip_features, "ndim", None) != 3 or getattr(frame_features, "ndim", None) != 3:
            raise ValueError("DreamPRVR context features must both be [video, location, dim] tensors")
        num_videos = int(clip_features.shape[0])
        if int(frame_features.shape[0]) != num_videos:
            raise ValueError("clip/frame context tensors contain different numbers of videos")
        if len(self.video_ids) != num_videos:
            raise ValueError(
                f"video_ids length {len(self.video_ids)} does not match context video count {num_videos}"
            )
        if int(clip_features.shape[1]) <= 0 or int(frame_features.shape[1]) <= 0:
            raise ValueError("DreamPRVR context tensors must contain at least one location per video")
        if int(clip_features.shape[2]) != int(frame_features.shape[2]):
            raise ValueError("clip/frame context embedding dimensions do not match")

    @staticmethod
    def _max_scores(query_vectors, context_features):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DreamPRVRAdapter requires the optional 'torch' dependency") from exc

        if query_vectors.ndim == 1:
            query_vectors = query_vectors.unsqueeze(0)
        if query_vectors.ndim != 2:
            raise ValueError(f"Expected query vectors [query, dim], got {tuple(query_vectors.shape)}")
        if context_features.ndim != 3:
            raise ValueError(
                f"Expected inference context [video, location, dim], got {tuple(context_features.shape)}"
            )
        if query_vectors.shape[-1] != context_features.shape[-1]:
            raise ValueError("query/context embedding dimensions do not match")
        if context_features.shape[1] <= 0:
            raise ValueError("context feature tensor contains no locations")

        q = F.normalize(query_vectors, dim=-1)
        ctx = F.normalize(context_features, dim=-1)
        location_scores = torch.einsum("qd,vld->qvl", q, ctx)
        if not torch.isfinite(location_scores).all():
            raise ValueError("DreamPRVR similarity computation produced non-finite scores")
        return location_scores.max(dim=-1).values

    def retrieve(self, query_feat, query_mask, *, top_k: int = 20) -> RetrievalBatch:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DreamPRVRAdapter requires the optional 'torch' dependency") from exc

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        had_training_attr = hasattr(self.model, "training")
        was_training = bool(getattr(self.model, "training", False))
        if hasattr(self.model, "eval"):
            self.model.eval()
        try:
            with torch.no_grad():
                query_vectors = self.model.encode_query(query_feat, query_mask)
                clip_scores = self._max_scores(query_vectors, self.context_info["video_proposal_feat"])
                frame_scores = self._max_scores(query_vectors, self.context_info["video_feat"])
                if clip_scores.shape != frame_scores.shape:
                    raise ValueError("clip/frame score tensors have incompatible shapes")
                fused = self.clip_scale_weight * clip_scores + self.frame_scale_weight * frame_scores
                if not torch.isfinite(fused).all():
                    raise ValueError("DreamPRVR fused retrieval scores are non-finite")
                k = min(int(top_k), int(fused.shape[1]))
                _, top_indices = fused.topk(k=k, dim=1)
        finally:
            if had_training_attr and was_training and hasattr(self.model, "train"):
                self.model.train()

        all_candidates: list[list[Candidate]] = []
        for q_idx in range(int(fused.shape[0])):
            row: list[Candidate] = []
            for vid_idx in top_indices[q_idx].tolist():
                row.append(
                    Candidate(
                        video_id=str(self.video_ids[vid_idx]),
                        video_index=int(vid_idx),
                        base_score=float(fused[q_idx, vid_idx].item()),
                        clip_score=float(clip_scores[q_idx, vid_idx].item()),
                        frame_score=float(frame_scores[q_idx, vid_idx].item()),
                        metadata={
                            "clip_scale_weight": self.clip_scale_weight,
                            "frame_scale_weight": self.frame_scale_weight,
                        },
                    )
                )
            all_candidates.append(row)
        return RetrievalBatch(candidates=all_candidates)
