from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from prvr_agent.schemas import Candidate


@dataclass(frozen=True)
class RetrievalBatch:
    candidates: list[list[Candidate]]


class DreamPRVRAdapter:
    """Non-invasive adapter around a loaded DreamPRVR model and cached context features.

    The adapter keeps upstream DreamPRVR video-level scores unchanged while exposing
    valid (non-padding) activation peaks for downstream verification.
    """

    def __init__(
        self,
        model,
        context_info: dict,
        video_ids: Optional[Sequence[str]] = None,
        *,
        clip_scale_weight: float = 0.5,
        frame_scale_weight: float = 0.5,
    ) -> None:
        self.model = model
        self.context_info = context_info
        required = {"video_proposal_feat", "video_feat"}
        missing = sorted(required.difference(context_info))
        if missing:
            raise ValueError(f"context_info is missing required keys: {missing}")
        self.video_ids = list(video_ids or context_info.get("video_metas") or [])
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

        num_videos = int(context_info["video_proposal_feat"].shape[0])
        if int(context_info["video_feat"].shape[0]) != num_videos:
            raise ValueError("clip/frame context tensors contain different numbers of videos")
        if len(self.video_ids) != num_videos:
            raise ValueError(
                f"video_ids length {len(self.video_ids)} does not match context video count {num_videos}"
            )

    @staticmethod
    def _normalize_mask(valid_mask, *, target_length: int, num_videos: int, device):
        if valid_mask is None:
            return None
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DreamPRVRAdapter requires the optional 'torch' dependency") from exc
        if valid_mask.ndim != 2 or valid_mask.shape[0] != num_videos:
            raise ValueError(
                f"Expected valid mask [video, location], got {tuple(valid_mask.shape)}"
            )
        mask = valid_mask.to(device=device).bool()
        if mask.shape[1] < target_length:
            pad = torch.zeros(
                (num_videos, target_length - mask.shape[1]), dtype=torch.bool, device=device
            )
            mask = torch.cat([mask, pad], dim=1)
        elif mask.shape[1] > target_length:
            mask = mask[:, :target_length]
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every video must contain at least one valid frame location")
        return mask

    @staticmethod
    def _scores_and_peaks(query_vectors, context_features, valid_mask=None):
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

        q = F.normalize(query_vectors, dim=-1)
        ctx = F.normalize(context_features, dim=-1)
        location_scores = torch.einsum("qd,vld->qvl", q, ctx)

        upstream_scores = location_scores.max(dim=-1).values
        mask = DreamPRVRAdapter._normalize_mask(
            valid_mask,
            target_length=context_features.shape[1],
            num_videos=context_features.shape[0],
            device=context_features.device,
        )
        if mask is None:
            valid_scores = location_scores
        else:
            valid_scores = location_scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        peak_scores, peaks = valid_scores.max(dim=-1)
        return upstream_scores, peaks, peak_scores, mask

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
                clip_scores, clip_peaks, clip_peak_scores, _ = self._scores_and_peaks(
                    query_vectors, self.context_info["video_proposal_feat"]
                )
                frame_scores, frame_peaks, frame_peak_scores, frame_mask = self._scores_and_peaks(
                    query_vectors,
                    self.context_info["video_feat"],
                    self.context_info.get("video_mask"),
                )
                fused = self.clip_scale_weight * clip_scores + self.frame_scale_weight * frame_scores
                k = min(int(top_k), int(fused.shape[1]))
                _, top_indices = fused.topk(k=k, dim=1)
        finally:
            if had_training_attr and was_training and hasattr(self.model, "train"):
                self.model.train()

        clip_locations = int(self.context_info["video_proposal_feat"].shape[1])
        all_candidates = []
        for q_idx in range(int(fused.shape[0])):
            row = []
            for vid_idx in top_indices[q_idx].tolist():
                frame_valid_locations = (
                    int(frame_mask[vid_idx].sum().item())
                    if frame_mask is not None
                    else int(self.context_info["video_feat"].shape[1])
                )
                row.append(
                    Candidate(
                        video_id=str(self.video_ids[vid_idx]),
                        video_index=int(vid_idx),
                        base_score=float(fused[q_idx, vid_idx].item()),
                        clip_score=float(clip_scores[q_idx, vid_idx].item()),
                        frame_score=float(frame_scores[q_idx, vid_idx].item()),
                        clip_peak_index=int(clip_peaks[q_idx, vid_idx].item()),
                        frame_peak_index=int(frame_peaks[q_idx, vid_idx].item()),
                        metadata={
                            "clip_num_locations": clip_locations,
                            "frame_valid_locations": frame_valid_locations,
                            "clip_peak_score": float(clip_peak_scores[q_idx, vid_idx].item()),
                            "frame_peak_score": float(frame_peak_scores[q_idx, vid_idx].item()),
                            "clip_scale_weight": self.clip_scale_weight,
                            "frame_scale_weight": self.frame_scale_weight,
                        },
                    )
                )
            all_candidates.append(row)
        return RetrievalBatch(candidates=all_candidates)
