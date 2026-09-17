from __future__ import annotations

import contextlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import numpy as np

from prvr_agent.agents.hypothesis_planner import OpenAIHypothesisPlanner
from prvr_agent.agents.world_model import OpenAIEventWorldPlanner
from prvr_agent.agents.world_observer import OpenAIWorldEvidenceBackend
from prvr_agent.evaluation import IndexedFrameDirectoryResolver, IndexedVideoPathResolver
from prvr_agent.pipeline import PRVRAgentReranker, PipelineConfig

from .dreamprvr_benchmark import DreamPRVRBenchmarkResult, rerank_dreamprvr_query_loader


@dataclass(frozen=True)
class DreamPRVRDatasetSpec:
    """Official DreamPRVR dataset identity plus the APEI visual-source type."""

    name: str
    official_code: str
    collection: str
    visual_feature: str
    source_kind: str


_DATASET_SPECS = {
    "tvr": DreamPRVRDatasetSpec(
        name="tvr",
        official_code="tvr",
        collection="tvr",
        visual_feature="i3d_resnet",
        source_kind="extracted_frames",
    ),
    "activitynet": DreamPRVRDatasetSpec(
        name="activitynet",
        official_code="act",
        collection="activitynet",
        visual_feature="i3d",
        source_kind="raw_video",
    ),
    "charades": DreamPRVRDatasetSpec(
        name="charades",
        official_code="cha",
        collection="charades",
        visual_feature="i3d_rgb_lgi",
        source_kind="raw_video",
    ),
}
_DATASET_ALIASES = {
    "tvr": "tvr",
    "act": "activitynet",
    "activitynet": "activitynet",
    "activitynet-captions": "activitynet",
    "cha": "charades",
    "charades": "charades",
    "charades-sta": "charades",
}


def resolve_dreamprvr_dataset(value: str) -> DreamPRVRDatasetSpec:
    """Resolve CLI-friendly aliases without changing DreamPRVR collection ids."""

    key = str(value).strip().lower().replace("_", "-")
    canonical = _DATASET_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(_DATASET_SPECS))
        raise ValueError(f"unsupported DreamPRVR dataset {value!r}; choose one of: {supported}")
    return _DATASET_SPECS[canonical]


@dataclass(frozen=True)
class DreamPRVRVisualSource:
    """Dataset-specific resolver and timing metadata used by APEI."""

    dataset: DreamPRVRDatasetSpec
    resolver: Callable[[str], Path]
    indexed_video_ids: tuple[str, ...]
    frame_directory_fps: Optional[float]

    @property
    def source_kind(self) -> str:
        return self.dataset.source_kind


def build_dreamprvr_visual_source(
    dataset: str,
    roots: Sequence[str | Path],
    *,
    frame_directory_fps: Optional[float] = None,
) -> DreamPRVRVisualSource:
    """Build the benchmark visual resolver for TVR, ActivityNet, or Charades.

    TVR is intentionally wired to extracted frame directories and requires an
    explicit extraction FPS. ActivityNet Captions and Charades-STA are wired to
    raw videos. This keeps timestamp semantics visible instead of silently
    applying TVQA's 3 FPS convention to unrelated frame releases.
    """

    spec = resolve_dreamprvr_dataset(dataset)
    if spec.source_kind == "extracted_frames":
        if frame_directory_fps is None:
            raise ValueError("TVR extracted frames require --frame-directory-fps")
        fps = float(frame_directory_fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("frame_directory_fps must be finite and positive")
        resolver = IndexedFrameDirectoryResolver.from_roots(roots)
        return DreamPRVRVisualSource(
            dataset=spec,
            resolver=resolver,
            indexed_video_ids=resolver.video_ids(),
            frame_directory_fps=fps,
        )

    if frame_directory_fps is not None:
        raise ValueError(
            f"frame_directory_fps is only valid for TVR extracted frames, not {spec.name} raw videos"
        )
    resolver = IndexedVideoPathResolver.from_roots(roots)
    return DreamPRVRVisualSource(
        dataset=spec,
        resolver=resolver,
        indexed_video_ids=resolver.video_ids(),
        frame_directory_fps=None,
    )


def validate_visual_source_coverage(
    source: DreamPRVRVisualSource,
    expected_video_ids: Sequence[str],
) -> None:
    """Fail before any Qwen calls when a benchmark video cannot be observed."""

    expected = [str(video_id) for video_id in expected_video_ids]
    if not expected or any(not video_id for video_id in expected):
        raise ValueError("DreamPRVR expected video ids must be non-empty")
    if len(expected) != len(set(expected)):
        raise ValueError("DreamPRVR expected video ids contain duplicates")
    indexed = set(source.indexed_video_ids)
    missing = [video_id for video_id in expected if video_id not in indexed]
    if missing:
        preview = ", ".join(repr(video_id) for video_id in missing[:20])
        raise FileNotFoundError(
            f"{source.dataset.name} visual source is missing {len(missing)} of "
            f"{len(expected)} benchmark videos; first missing ids: {preview}"
        )


@dataclass(frozen=True)
class _DreamPRVRBindings:
    get_datasets: Callable[[dict[str, Any]], tuple]
    get_models: Callable[[dict[str, Any]], Any]
    get_validations: Callable[[dict[str, Any]], Any]
    source_root: Optional[Path] = None


@contextlib.contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:  # pragma: no cover - defensive against third-party mutation
            pass


def _validate_import_origin(module: ModuleType, source_root: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise ImportError(f"official DreamPRVR module {module.__name__!r} has no filesystem origin")
    resolved = Path(module_file).resolve()
    try:
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise ImportError(
            f"module {module.__name__!r} was imported from {resolved}, not {source_root}; "
            "start a clean Python process to avoid mixing DreamPRVR checkouts"
        ) from exc


def _load_dreamprvr_bindings(dreamprvr_root: str | Path) -> _DreamPRVRBindings:
    root = Path(dreamprvr_root).expanduser().resolve()
    source_root = root / "src"
    required = (
        source_root / "Datasets" / "builder.py",
        source_root / "Models" / "builder.py",
        source_root / "Validations" / "builder.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "DreamPRVR checkout is missing official builder files: " + ", ".join(missing)
        )

    # The upstream project uses top-level imports such as ``from Models...``.
    # Put only its src directory at the front while importing those builders.
    with _prepend_sys_path(source_root):
        dataset_module = importlib.import_module("Datasets.builder")
        model_module = importlib.import_module("Models.builder")
        validation_module = importlib.import_module("Validations.builder")
    for module in (dataset_module, model_module, validation_module):
        _validate_import_origin(module, source_root)
    return _DreamPRVRBindings(
        get_datasets=dataset_module.get_datasets,
        get_models=model_module.get_models,
        get_validations=validation_module.get_validations,
        source_root=source_root,
    )


def _load_official_checkpoint(path: str | Path):
    """Load the official checkpoint through PyTorch's restricted weights loader."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DreamPRVR benchmark entrypoint requires torch") from exc

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DreamPRVR checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - old unsupported torch builds
        raise RuntimeError(
            "this torch build lacks weights_only checkpoint loading; use torch>=2.0 "
            "instead of falling back to unrestricted pickle loading"
        ) from exc
    except Exception as exc:
        raise ValueError(
            "the checkpoint could not be loaded in restricted mode; expected the official "
            "DreamPRVR config/state_dict checkpoint format"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("DreamPRVR checkpoint must be a mapping")
    required = {"config", "state_dict"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"DreamPRVR checkpoint is missing required keys: {missing}")
    if not isinstance(payload["config"], Mapping) or not isinstance(payload["state_dict"], Mapping):
        raise ValueError("DreamPRVR checkpoint config and state_dict must be mappings")
    return payload, checkpoint_path


def _strict_load_state_dict(model, state_dict: Mapping[str, Any]) -> None:
    """Accept the official state dict with at most one DataParallel prefix."""

    model_keys = set(model.state_dict())
    state = dict(state_dict)
    state_keys = set(state)
    if state_keys != model_keys and state and all(str(key).startswith("module.") for key in state):
        stripped = {str(key)[7:]: value for key, value in state.items()}
        if set(stripped) == model_keys:
            state = stripped
            state_keys = set(state)
    if state_keys != model_keys:
        missing = sorted(model_keys.difference(state_keys))[:10]
        unexpected = sorted(state_keys.difference(model_keys))[:10]
        raise ValueError(
            "DreamPRVR checkpoint does not exactly match the constructed model; "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.load_state_dict(state, strict=True)


def _move_nested_to_device(value, device):
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_nested_to_device(item, device) for key, item in value.items()}
    if hasattr(value, "to"):
        return value.contiguous().to(device, non_blocking=device.type == "cuda")
    return value


@contextlib.contextmanager
def _official_validation_device(compute_context_info, device) -> Iterator[None]:
    """Make upstream's hard-coded ``gpu(batch)`` follow the selected device.

    The computation itself remains the official ``compute_context_info`` method.
    Only its module-global transfer helper is temporarily replaced, which also
    enables deterministic CPU integration tests and explicit ``cuda:N`` use.
    """

    function = getattr(compute_context_info, "__func__", compute_context_info)
    globals_dict = getattr(function, "__globals__", None)
    if not isinstance(globals_dict, dict) or "gpu" not in globals_dict:
        yield
        return
    original = globals_dict["gpu"]
    globals_dict["gpu"] = lambda value: _move_nested_to_device(value, device)
    try:
        yield
    finally:
        globals_dict["gpu"] = original


@dataclass(frozen=True)
class DreamPRVRRuntime:
    """Loaded official DreamPRVR test objects needed by the reranking bridge."""

    dataset: DreamPRVRDatasetSpec
    config: dict[str, Any]
    model: Any
    validation: Any
    test_context_dataloader: Any
    test_query_eval_loader: Any
    context_info: dict[str, Any]
    checkpoint_path: Path
    device: str


def load_dreamprvr_runtime(
    *,
    dreamprvr_root: str | Path,
    checkpoint: str | Path,
    data_root: str | Path,
    dataset: str,
    device: str = "auto",
    num_workers: Optional[int] = 4,
    eval_query_batch_size: Optional[int] = None,
    eval_context_batch_size: Optional[int] = None,
) -> DreamPRVRRuntime:
    """Load checkpoint, official datasets/model, and official test context."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DreamPRVR benchmark entrypoint requires torch") from exc

    spec = resolve_dreamprvr_dataset(dataset)
    payload, checkpoint_path = _load_official_checkpoint(checkpoint)
    cfg = dict(payload["config"])
    if str(cfg.get("collection", "")) != spec.collection:
        raise ValueError(
            f"checkpoint collection {cfg.get('collection')!r} does not match requested dataset {spec.name!r}"
        )
    if str(cfg.get("visual_feature", "")) != spec.visual_feature:
        raise ValueError(
            f"checkpoint visual_feature {cfg.get('visual_feature')!r} does not match "
            f"official {spec.name} feature {spec.visual_feature!r}"
        )

    resolved_data_root = Path(data_root).expanduser().resolve()
    if not resolved_data_root.is_dir():
        raise NotADirectoryError(f"DreamPRVR data root is not a directory: {resolved_data_root}")
    collection_root = resolved_data_root / spec.collection
    if not collection_root.is_dir():
        raise FileNotFoundError(
            f"DreamPRVR data root must contain {spec.collection!r}: {collection_root}"
        )
    cfg["data_root"] = str(resolved_data_root)
    if num_workers is not None:
        if int(num_workers) < 0:
            raise ValueError("num_workers must be non-negative")
        cfg["num_workers"] = int(num_workers)
    if eval_query_batch_size is not None:
        if int(eval_query_batch_size) <= 0:
            raise ValueError("eval_query_batch_size must be positive")
        cfg["eval_query_bsz"] = int(eval_query_batch_size)
    if eval_context_batch_size is not None:
        if int(eval_context_batch_size) <= 0:
            raise ValueError("eval_context_batch_size must be positive")
        cfg["eval_context_bsz"] = int(eval_context_batch_size)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        if torch_device.index is not None:
            torch.cuda.set_device(torch_device)
        cfg["pin_memory"] = True
    else:
        cfg["pin_memory"] = False

    bindings = _load_dreamprvr_bindings(dreamprvr_root)
    import_path = (
        _prepend_sys_path(bindings.source_root)
        if bindings.source_root is not None
        else contextlib.nullcontext()
    )
    # Keep the upstream src directory available while constructors and data
    # methods run because the public code uses absolute top-level imports.
    with import_path:
        datasets = bindings.get_datasets(cfg)
        if not isinstance(datasets, (tuple, list)) or len(datasets) != 6:
            raise ValueError("official DreamPRVR get_datasets(cfg) must return exactly six values")
        (
            returned_cfg,
            _train_loader,
            _context_dataloader,
            _query_eval_loader,
            test_context_dataloader,
            test_query_eval_loader,
        ) = datasets
        if not isinstance(returned_cfg, Mapping):
            raise ValueError("official DreamPRVR get_datasets returned an invalid config")
        cfg = dict(returned_cfg)

        model = bindings.get_models(cfg)
        _strict_load_state_dict(model, payload["state_dict"])
        model = model.to(torch_device)
        model.eval()
        validation = bindings.get_validations(cfg)
        if not hasattr(validation, "compute_context_info"):
            raise AttributeError("official DreamPRVR validation object lacks compute_context_info")
        with torch.no_grad(), _official_validation_device(validation.compute_context_info, torch_device):
            context_info = validation.compute_context_info(model, test_context_dataloader)
    if not isinstance(context_info, dict):
        raise ValueError("official DreamPRVR compute_context_info must return a dictionary")

    return DreamPRVRRuntime(
        dataset=spec,
        config=cfg,
        model=model,
        validation=validation,
        test_context_dataloader=test_context_dataloader,
        test_query_eval_loader=test_query_eval_loader,
        context_info=context_info,
        checkpoint_path=checkpoint_path,
        device=str(torch_device),
    )


@dataclass(frozen=True)
class DreamPRVROneClickResult:
    benchmark: DreamPRVRBenchmarkResult
    metadata: dict[str, Any]


def run_dreamprvr_benchmark(
    *,
    dreamprvr_root: str | Path,
    checkpoint: str | Path,
    data_root: str | Path,
    dataset: str,
    visual_roots: Sequence[str | Path],
    frame_directory_fps: Optional[float] = None,
    top_k: int = 20,
    max_queries: Optional[int] = None,
    device: str = "auto",
    num_workers: Optional[int] = 4,
    eval_query_batch_size: Optional[int] = None,
    eval_context_batch_size: Optional[int] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    reranker: Optional[PRVRAgentReranker] = None,
) -> DreamPRVROneClickResult:
    """Run the official DreamPRVR test split through CQHG + APEI in one call."""

    source = build_dreamprvr_visual_source(
        dataset,
        visual_roots,
        frame_directory_fps=frame_directory_fps,
    )
    runtime = load_dreamprvr_runtime(
        dreamprvr_root=dreamprvr_root,
        checkpoint=checkpoint,
        data_root=data_root,
        dataset=dataset,
        device=device,
        num_workers=num_workers,
        eval_query_batch_size=eval_query_batch_size,
        eval_context_batch_size=eval_context_batch_size,
    )
    video_ids = runtime.context_info.get("video_metas")
    if not isinstance(video_ids, (list, tuple)):
        raise ValueError("official DreamPRVR context_info lacks video_metas")
    validate_visual_source_coverage(source, video_ids)

    if reranker is None:
        reranker = PRVRAgentReranker(
            planner=OpenAIHypothesisPlanner.from_env(),
            world_planner=OpenAIEventWorldPlanner.from_env(),
            world_evidence_backend=OpenAIWorldEvidenceBackend(
                frame_directory_fps=source.frame_directory_fps,
            ),
            video_path_resolver=source.resolver,
            cfg=pipeline_config or PipelineConfig(),
        )

    clip_weight = float(runtime.config["clip_scale_w"])
    frame_weight = float(runtime.config["frame_scale_w"])
    benchmark = rerank_dreamprvr_query_loader(
        model=runtime.model,
        query_loader=runtime.test_query_eval_loader,
        context_info=runtime.context_info,
        reranker=reranker,
        clip_scale_weight=clip_weight,
        frame_scale_weight=frame_weight,
        top_k=top_k,
        max_queries=max_queries,
    )
    metadata = {
        "schema_version": 1,
        "dataset": runtime.dataset.name,
        "dreamprvr_dataset_code": runtime.dataset.official_code,
        "dreamprvr_collection": runtime.dataset.collection,
        "visual_source_kind": source.source_kind,
        "frame_directory_fps": source.frame_directory_fps,
        "checkpoint": str(runtime.checkpoint_path),
        "dreamprvr_root": str(Path(dreamprvr_root).expanduser().resolve()),
        "data_root": str(Path(data_root).expanduser().resolve()),
        "visual_roots": [str(Path(root).expanduser().resolve()) for root in visual_roots],
        "device": runtime.device,
        "clip_scale_weight": clip_weight,
        "frame_scale_weight": frame_weight,
        "top_k": int(benchmark.top_k),
        "max_queries": max_queries,
        "query_count": len(benchmark.query_ids),
        "video_count": len(benchmark.video_ids),
    }
    return DreamPRVROneClickResult(benchmark=benchmark, metadata=metadata)


def save_dreamprvr_benchmark_result(
    run: DreamPRVROneClickResult,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist score matrices, ordered ids, and a self-describing metrics manifest."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    base_name = "dreamprvr_base_scores.npy"
    reranked_name = "dreamprvr_cqhg_apei_scores.npy"
    video_ids_name = "video_ids.json"
    query_ids_name = "query_ids.json"
    manifest_name = "benchmark_result.json"

    targets = [
        destination / base_name,
        destination / reranked_name,
        destination / video_ids_name,
        destination / query_ids_name,
        destination / manifest_name,
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "benchmark output already exists; use a new output directory or explicit overwrite: "
            + ", ".join(existing)
        )

    np.save(destination / base_name, run.benchmark.base_scores, allow_pickle=False)
    np.save(destination / reranked_name, run.benchmark.reranked_scores, allow_pickle=False)
    (destination / video_ids_name).write_text(
        json.dumps(list(run.benchmark.video_ids), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (destination / query_ids_name).write_text(
        json.dumps(list(run.benchmark.query_ids), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = dict(run.metadata)
    manifest.update(
        {
            "base_metrics": run.benchmark.base_metrics.as_dict(),
            "reranked_metrics": run.benchmark.reranked_metrics.as_dict(),
            "metric_delta": run.benchmark.metric_delta(),
            "files": {
                "base_scores": base_name,
                "reranked_scores": reranked_name,
                "video_ids": video_ids_name,
                "query_ids": query_ids_name,
            },
        }
    )
    manifest_path = destination / manifest_name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
