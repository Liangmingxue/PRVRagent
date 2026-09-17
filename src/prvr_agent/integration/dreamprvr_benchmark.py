from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from prvr_agent.evaluation import PRVRMetrics, evaluate_prvr_scores, rerank_topk_score_slots
from prvr_agent.pipeline import PRVRAgentReranker, RerankedCandidate
from prvr_agent.retriever import DreamPRVRAdapter


@dataclass(frozen=True)
class DreamPRVRBenchmarkResult:
    """Full-collection baseline and APEI-reranked score matrices."""

    base_scores: np.ndarray
    reranked_scores: np.ndarray
    video_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    base_metrics: PRVRMetrics
    reranked_metrics: PRVRMetrics
    top_k: int

    def metric_delta(self) -> dict[str, float]:
        base = self.base_metrics.as_dict()
        reranked = self.reranked_metrics.as_dict()
        return {key: float(reranked[key] - base[key]) for key in base}


def _unwrap_model(model):
    """Use the underlying DreamPRVR module when wrapped by DataParallel/DDP."""

    return getattr(model, "module", model)


def _model_device(model):
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration):
        return None
    return parameter.device


def _move_tensor(value, device):
    if device is None or not hasattr(value, "to"):
        return value
    return value.to(device)


def _query_text_lookup(query_loader) -> dict[str, str]:
    dataset = getattr(query_loader, "dataset", None)
    captions = getattr(dataset, "captions", None)
    if not isinstance(captions, dict):
        raise ValueError(
            "DreamPRVR query loader dataset must expose a captions dict mapping cap_id -> query text"
        )
    output: dict[str, str] = {}
    for key, value in captions.items():
        query_id = str(key)
        text = str(value).strip()
        if not query_id or not text:
            raise ValueError("DreamPRVR caption ids and texts must be non-empty")
        output[query_id] = text
    return output


def _video_ids_from_context(context_info: dict) -> list[str]:
    video_metas = context_info.get("video_metas")
    if not isinstance(video_metas, (list, tuple)) or not video_metas:
        raise ValueError("context_info['video_metas'] must be a non-empty list/tuple")
    video_ids = [str(item) for item in video_metas]
    if any(not item for item in video_ids):
        raise ValueError("DreamPRVR video ids must be non-empty")
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("DreamPRVR context contains duplicate video ids")
    return video_ids


def _validate_context_for_model(context_info: dict, device):
    required = ("video_proposal_feat", "video_feat")
    missing = [key for key in required if key not in context_info]
    if missing:
        raise ValueError(f"context_info is missing required keys: {missing}")

    prepared = dict(context_info)
    for key in ("video_proposal_feat", "video_feat", "video_mask"):
        if key in prepared and prepared[key] is not None:
            prepared[key] = _move_tensor(prepared[key], device)
    return prepared


def _upstream_full_scores(
    model,
    query_feat,
    query_mask,
    context_info: dict,
    *,
    clip_scale_weight: float,
    frame_scale_weight: float,
):
    """Use DreamPRVR's public validation scoring path as benchmark authority."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DreamPRVR benchmark integration requires torch") from exc

    if not hasattr(model, "get_pred_from_raw_query"):
        raise AttributeError("DreamPRVR model must expose get_pred_from_raw_query")

    with torch.no_grad():
        clip_scores, frame_scores = model.get_pred_from_raw_query(
            query_feat,
            query_mask,
            None,
            context_info["video_proposal_feat"],
            context_info["video_feat"],
        )
        fused = clip_scale_weight * clip_scores + frame_scale_weight * frame_scores
    if getattr(fused, "ndim", None) != 2:
        raise ValueError("DreamPRVR fused validation scores must be [query, video]")
    if not torch.isfinite(fused).all():
        raise ValueError("DreamPRVR produced non-finite validation scores")
    return fused


def rerank_dreamprvr_query_loader(
    *,
    model,
    query_loader,
    context_info: dict,
    reranker: PRVRAgentReranker,
    clip_scale_weight: float,
    frame_scale_weight: float,
    top_k: int = 20,
    max_queries: Optional[int] = None,
) -> DreamPRVRBenchmarkResult:
    """Run a DreamPRVR validation/test query loader through APEI Top-K reranking.

    This bridge deliberately reuses DreamPRVR's own ``get_pred_from_raw_query``
    output for the full score matrix.  ``DreamPRVRAdapter`` is used only to build
    the identical Top-K candidate objects plus query-agnostic semantic sidekick
    metadata.  ``rerank_topk_score_slots`` then changes only the ordering within
    the original shortlist, so the untouched collection tail remains on the same
    DreamPRVR score scale.

    The public DreamPRVR ``TxtDataSet4PRVR`` keeps raw query strings in
    ``query_loader.dataset.captions``; this function uses that mapping instead of
    trying to reconstruct text from precomputed RoBERTa features.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if max_queries is not None and (
        isinstance(max_queries, bool) or not isinstance(max_queries, int) or max_queries <= 0
    ):
        raise ValueError("max_queries must be a positive integer when provided")
    for name, value in (
        ("clip_scale_weight", clip_scale_weight),
        ("frame_scale_weight", frame_scale_weight),
    ):
        if not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(clip_scale_weight) + float(frame_scale_weight) <= 0:
        raise ValueError("at least one DreamPRVR fusion weight must be positive")

    core_model = _unwrap_model(model)
    if hasattr(core_model, "eval"):
        core_model.eval()
    device = _model_device(core_model)
    prepared_context = _validate_context_for_model(context_info, device)
    video_ids = _video_ids_from_context(prepared_context)
    query_text = _query_text_lookup(query_loader)

    adapter = DreamPRVRAdapter(
        core_model,
        prepared_context,
        video_ids=video_ids,
        clip_scale_weight=float(clip_scale_weight),
        frame_scale_weight=float(frame_scale_weight),
    )

    base_rows: list[np.ndarray] = []
    reranked_rows: list[np.ndarray] = []
    query_ids: list[str] = []

    processed = 0
    for batch in query_loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 4:
            raise ValueError("DreamPRVR text validation batch must contain query_feat, query_mask, idxs, cap_ids")
        query_feat = _move_tensor(batch[0], device)
        query_mask = _move_tensor(batch[1], device)
        cap_ids = [str(item) for item in batch[-1]]
        if not cap_ids:
            continue

        fused = _upstream_full_scores(
            core_model,
            query_feat,
            query_mask,
            prepared_context,
            clip_scale_weight=float(clip_scale_weight),
            frame_scale_weight=float(frame_scale_weight),
        )
        candidate_batch = adapter.retrieve(query_feat, query_mask, top_k=top_k)
        if len(candidate_batch.candidates) != len(cap_ids) or int(fused.shape[0]) != len(cap_ids):
            raise ValueError("DreamPRVR batch query count is inconsistent across ids, scores, and candidates")
        if int(fused.shape[1]) != len(video_ids):
            raise ValueError("DreamPRVR score column count does not match context video ids")

        fused_np = fused.detach().cpu().numpy().astype(np.float64, copy=False)
        for row_index, query_id in enumerate(cap_ids):
            if max_queries is not None and processed >= max_queries:
                break
            text = query_text.get(query_id)
            if text is None:
                raise KeyError(f"query id {query_id!r} is missing from query_loader.dataset.captions")

            candidates = candidate_batch.candidates[row_index]
            # Enforce exact shortlist equivalence between our adapter and the
            # authoritative upstream DreamPRVR scoring path before spending VLM
            # compute. A mismatch signals integration drift and invalidates the
            # reranking experiment.
            expected_topk = set(
                np.argsort(-fused_np[row_index], kind="stable")[: len(candidates)].tolist()
            )
            observed_topk = {int(item.video_index) for item in candidates}
            if observed_topk != expected_topk:
                raise ValueError(
                    "DreamPRVRAdapter Top-K differs from upstream get_pred_from_raw_query; "
                    "verify adapter score equivalence before benchmark reporting"
                )

            reranked: Sequence[RerankedCandidate] = reranker.rerank(text, candidates)
            base_row = fused_np[row_index].copy()
            reranked_row = rerank_topk_score_slots(base_row, reranked)
            base_rows.append(base_row)
            reranked_rows.append(reranked_row)
            query_ids.append(query_id)
            processed += 1

        if max_queries is not None and processed >= max_queries:
            break

    if not base_rows:
        raise ValueError("query loader produced no benchmark queries")

    base_matrix = np.stack(base_rows, axis=0)
    reranked_matrix = np.stack(reranked_rows, axis=0)
    base_metrics = evaluate_prvr_scores(base_matrix, video_ids=video_ids, query_ids=query_ids)
    reranked_metrics = evaluate_prvr_scores(reranked_matrix, video_ids=video_ids, query_ids=query_ids)

    return DreamPRVRBenchmarkResult(
        base_scores=base_matrix,
        reranked_scores=reranked_matrix,
        video_ids=tuple(video_ids),
        query_ids=tuple(query_ids),
        base_metrics=base_metrics,
        reranked_metrics=reranked_metrics,
        top_k=int(top_k),
    )
