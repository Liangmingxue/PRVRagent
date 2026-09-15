from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from prvr_agent.schemas import Candidate


SEMANTIC_SIDEKICK_METADATA_KEY = "apei_semantic_sidekick"


@dataclass(frozen=True)
class RetrievalBatch:
    candidates: list[list[Candidate]]


class DreamPRVRAdapter:
    """Non-invasive adapter around a loaded DreamPRVR model and cached features.

    DreamPRVR remains responsible for its original clip/frame max-similarity scores.
    The agent consumes only video-level candidates; no query-conditioned local
    argmax peak is exported.

    When the upstream ``video_mask`` is available, the adapter additionally
    exports a *query-agnostic* temporal semantic-change trace derived from the
    already-computed DreamPRVR frame representations. APEI uses this trace only
    as a cheap sidekick for event-aware observation, never as retrieval evidence.
    """

    def __init__(
        self,
        model,
        context_info: dict,
        video_ids: Sequence[str] | None = None,
        *,
        clip_scale_weight: float = 0.5,
        frame_scale_weight: float = 0.5,
        semantic_sidekick_radius: int = 2,
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
        self.semantic_sidekick_radius = int(semantic_sidekick_radius)
        self._semantic_sidekick_cache: dict[int, dict[str, object]] = {}

        if not self.video_ids:
            raise ValueError("video_ids or context_info['video_metas'] is required")
        if not math.isfinite(self.clip_scale_weight) or not math.isfinite(self.frame_scale_weight):
            raise ValueError("retrieval fusion weights must be finite")
        if self.clip_scale_weight < 0 or self.frame_scale_weight < 0:
            raise ValueError("retrieval fusion weights must be non-negative")
        if self.clip_scale_weight + self.frame_scale_weight <= 0:
            raise ValueError("at least one retrieval fusion weight must be positive")
        if self.semantic_sidekick_radius <= 0:
            raise ValueError("semantic_sidekick_radius must be positive")

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

        video_mask = context_info.get("video_mask")
        if video_mask is not None:
            if getattr(video_mask, "ndim", None) != 2:
                raise ValueError("DreamPRVR video_mask must be [video, location]")
            if int(video_mask.shape[0]) != num_videos or int(video_mask.shape[1]) != int(frame_features.shape[1]):
                raise ValueError("DreamPRVR video_mask shape must match video_feat temporal dimensions")

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
            raise ValueError(f"Expected inference context [video, location, dim], got {tuple(context_features.shape)}")
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

    @staticmethod
    def _semantic_change_curve(frame_features, *, radius: int = 2) -> list[float]:
        """Return a query-agnostic semantic novelty curve for ordered frame features.

        Each boundary combines adjacent cosine change with a short left-vs-right
        context change. This is a lightweight KTS-inspired feature-change signal,
        not an implementation of the full KTS dynamic-programming objective.
        """

        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DreamPRVRAdapter requires the optional 'torch' dependency") from exc

        if radius <= 0:
            raise ValueError("radius must be positive")
        if getattr(frame_features, "ndim", None) != 2:
            raise ValueError("semantic sidekick expects [location, dim] frame features")
        if int(frame_features.shape[0]) <= 0:
            return []
        if not torch.isfinite(frame_features).all():
            raise ValueError("DreamPRVR frame features contain non-finite values")

        x = F.normalize(frame_features.float(), dim=-1)
        length = int(x.shape[0])
        scores = x.new_zeros(length)
        if length == 1:
            return [0.0]

        adjacent = 1.0 - torch.sum(x[1:] * x[:-1], dim=-1)
        scores[1:] = torch.clamp(adjacent, min=0.0, max=2.0)

        for boundary in range(1, length):
            left = x[max(0, boundary - radius) : boundary].mean(dim=0, keepdim=True)
            right = x[boundary : min(length, boundary + radius)].mean(dim=0, keepdim=True)
            left = F.normalize(left, dim=-1)
            right = F.normalize(right, dim=-1)
            context_change = 1.0 - torch.sum(left * right, dim=-1).squeeze(0)
            scores[boundary] = torch.maximum(
                scores[boundary],
                torch.clamp(context_change, min=0.0, max=2.0),
            )

        return [float(value) for value in scores.detach().cpu().tolist()]

    def _semantic_sidekick_metadata(self, video_index: int) -> dict[str, object] | None:
        """Build/cache a semantic sidekick trace for one candidate video."""

        if video_index in self._semantic_sidekick_cache:
            return dict(self._semantic_sidekick_cache[video_index])

        video_mask = self.context_info.get("video_mask")
        if video_mask is None:
            return None

        frame_features = self.context_info["video_feat"][video_index]
        mask_row = video_mask[video_index]
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DreamPRVRAdapter requires the optional 'torch' dependency") from exc

        if not torch.isfinite(mask_row).all():
            raise ValueError("DreamPRVR video_mask contains non-finite values")
        valid = mask_row > 0
        valid_length = int(valid.sum().item())
        if valid_length <= 0:
            return None

        # The public DreamPRVR collate path uses a contiguous valid prefix followed
        # by padding. Enforce that contract: compressing an arbitrary sparse mask
        # would make temporally distant features adjacent and invent a false
        # semantic transition for the sidekick.
        expected_valid = torch.arange(valid.numel(), device=valid.device) < valid_length
        if not torch.equal(valid.reshape(-1), expected_valid):
            raise ValueError("DreamPRVR video_mask valid positions must form one contiguous prefix")
        valid_features = frame_features[:valid_length]

        scores = self._semantic_change_curve(valid_features, radius=self.semantic_sidekick_radius)
        payload: dict[str, object] = {
            "source": "dreamprvr_encoded_frame_feat",
            "valid_length": valid_length,
            "context_radius": self.semantic_sidekick_radius,
            "change_scores": scores,
        }
        self._semantic_sidekick_cache[video_index] = payload
        return dict(payload)

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
                metadata: dict[str, object] = {
                    "clip_scale_weight": self.clip_scale_weight,
                    "frame_scale_weight": self.frame_scale_weight,
                }
                semantic_sidekick = self._semantic_sidekick_metadata(int(vid_idx))
                if semantic_sidekick is not None:
                    metadata[SEMANTIC_SIDEKICK_METADATA_KEY] = semantic_sidekick

                row.append(
                    Candidate(
                        video_id=str(self.video_ids[vid_idx]),
                        video_index=int(vid_idx),
                        base_score=float(fused[q_idx, vid_idx].item()),
                        clip_score=float(clip_scores[q_idx, vid_idx].item()),
                        frame_score=float(frame_scores[q_idx, vid_idx].item()),
                        metadata=metadata,
                    )
                )
            all_candidates.append(row)
        return RetrievalBatch(candidates=all_candidates)
