import json
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
except ImportError:  # Keep filesystem/result tests active in lightweight CI jobs.
    torch = None

from prvr_agent.evaluation import PRVRMetrics
from prvr_agent.agents.world_observer import OpenAIWorldEvidenceBackend
from prvr_agent.integration.dreamprvr_benchmark import DreamPRVRBenchmarkResult
from prvr_agent.integration.dreamprvr_entrypoint import (
    DreamPRVROneClickResult,
    _DreamPRVRBindings,
    _strict_load_state_dict,
    build_dreamprvr_visual_source,
    load_dreamprvr_runtime,
    resolve_dreamprvr_dataset,
    save_dreamprvr_benchmark_result,
    validate_visual_source_coverage,
)
from prvr_agent.llm_config import LLMConfig


def test_dataset_aliases_match_official_dreamprvr_codes():
    assert resolve_dreamprvr_dataset("TVR").official_code == "tvr"
    assert resolve_dreamprvr_dataset("act").collection == "activitynet"
    assert resolve_dreamprvr_dataset("activitynet-captions").visual_feature == "i3d"
    assert resolve_dreamprvr_dataset("charades-sta").official_code == "cha"
    with pytest.raises(ValueError, match="unsupported"):
        resolve_dreamprvr_dataset("unknown")


def test_dataset_specific_visual_sources_require_correct_media_type(tmp_path: Path):
    tvr_clip = tmp_path / "tvr" / "friends_frames" / "friends_clip_0"
    tvr_clip.mkdir(parents=True)
    (tvr_clip / "000001.jpg").write_bytes(b"frame")

    with pytest.raises(ValueError, match="frame-directory-fps"):
        build_dreamprvr_visual_source("tvr", [tmp_path / "tvr"])
    tvr = build_dreamprvr_visual_source(
        "tvr",
        [tmp_path / "tvr"],
        frame_directory_fps=3.0,
    )
    assert tvr.source_kind == "extracted_frames"
    assert tvr.frame_directory_fps == pytest.approx(3.0)
    assert tvr.resolver("friends_clip_0") == tvr_clip.resolve()

    activitynet_root = tmp_path / "activitynet_videos"
    activitynet_root.mkdir()
    activitynet_video = activitynet_root / "v_demo.mp4"
    activitynet_video.write_bytes(b"video")
    activitynet = build_dreamprvr_visual_source("act", [activitynet_root])
    assert activitynet.source_kind == "raw_video"
    assert activitynet.resolver("v_demo") == activitynet_video.resolve()
    with pytest.raises(ValueError, match="only valid for TVR"):
        build_dreamprvr_visual_source("act", [activitynet_root], frame_directory_fps=3.0)

    charades_root = tmp_path / "charades_videos"
    charades_root.mkdir()
    charades_video = charades_root / "ABC123.mp4"
    charades_video.write_bytes(b"video")
    charades = build_dreamprvr_visual_source("cha", [charades_root])
    assert charades.resolver("ABC123") == charades_video.resolve()


def test_visual_source_coverage_fails_before_vlm_calls(tmp_path: Path):
    root = tmp_path / "videos"
    root.mkdir()
    (root / "present.mp4").write_bytes(b"video")
    source = build_dreamprvr_visual_source("activitynet", [root])
    validate_visual_source_coverage(source, ["present"])
    with pytest.raises(FileNotFoundError, match="missing 1"):
        validate_visual_source_coverage(source, ["present", "absent"])


def test_world_observer_uses_explicit_frame_directory_fps(tmp_path: Path, monkeypatch):
    frames = tmp_path / "clip"
    frames.mkdir()
    (frames / "000001.jpg").write_bytes(b"frame")
    monkeypatch.delenv("PRVR_FRAME_DIRECTORY_FPS", raising=False)
    backend = OpenAIWorldEvidenceBackend(
        client=object(),
        model="test-model",
        config=LLMConfig(model="test-model"),
        frame_directory_fps=3.0,
    )
    source = backend._get_sampler(str(frames))
    assert source.fps == pytest.approx(3.0)
    assert source.frame_count == 1


if torch is not None:
    class _TinyDreamPRVR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_strict_state_dict_accepts_only_exact_or_dataparallel_keys():
    model = _TinyDreamPRVR()
    _strict_load_state_dict(model, {"module.anchor": torch.tensor(2.0)})
    assert model.anchor.item() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="does not exactly match"):
        _strict_load_state_dict(model, {"wrong": torch.tensor(1.0)})


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_official_checkpoint_loader_uses_expected_safe_format(tmp_path: Path):
    from prvr_agent.integration.dreamprvr_entrypoint import _load_official_checkpoint

    checkpoint = tmp_path / "official.ckpt"
    torch.save(
        {
            "config": {"collection": "activitynet", "visual_feature": "i3d"},
            "state_dict": {"anchor": torch.tensor(1.0)},
            "optimizer": {"state": {}, "param_groups": []},
            "epoch": 3,
            "model_val": [1.0, 2.0],
        },
        checkpoint,
    )
    payload, resolved = _load_official_checkpoint(checkpoint)
    assert resolved == checkpoint.resolve()
    assert payload["config"]["collection"] == "activitynet"


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_runtime_wraps_official_get_datasets_and_compute_context_info(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "features"
    (data_root / "activitynet").mkdir(parents=True)
    checkpoint = tmp_path / "official.ckpt"
    checkpoint.write_bytes(b"injected checkpoint")
    calls = []

    checkpoint_payload = {
        "config": {
            "collection": "activitynet",
            "visual_feature": "i3d",
            "clip_scale_w": 0.6,
            "frame_scale_w": 0.4,
            "eval_query_bsz": 50,
            "eval_context_bsz": 100,
            "num_workers": 24,
            "pin_memory": True,
        },
        "state_dict": {"anchor": torch.tensor(5.0)},
    }

    class FakeValidation:
        def compute_context_info(self, model, context_loader):
            calls.append(("compute_context_info", context_loader, model.anchor.item()))
            feature = torch.tensor([[[1.0, 0.0]]])
            return {
                "video_metas": ["v0"],
                "video_proposal_feat": feature,
                "video_feat": feature,
                "video_mask": torch.ones(1, 1),
            }

    test_context_loader = object()
    test_query_loader = object()

    def get_datasets(cfg):
        calls.append(("get_datasets", dict(cfg)))
        return cfg, object(), object(), object(), test_context_loader, test_query_loader

    def get_models(cfg):
        calls.append(("get_models", dict(cfg)))
        return _TinyDreamPRVR()

    def get_validations(cfg):
        calls.append(("get_validations", dict(cfg)))
        return FakeValidation()

    monkeypatch.setattr(
        "prvr_agent.integration.dreamprvr_entrypoint._load_official_checkpoint",
        lambda path: (checkpoint_payload, checkpoint.resolve()),
    )
    monkeypatch.setattr(
        "prvr_agent.integration.dreamprvr_entrypoint._load_dreamprvr_bindings",
        lambda root: _DreamPRVRBindings(get_datasets, get_models, get_validations),
    )

    runtime = load_dreamprvr_runtime(
        dreamprvr_root=tmp_path / "official_repo",
        checkpoint=checkpoint,
        data_root=data_root,
        dataset="act",
        device="cpu",
        num_workers=0,
        eval_query_batch_size=7,
        eval_context_batch_size=3,
    )

    assert runtime.test_context_dataloader is test_context_loader
    assert runtime.test_query_eval_loader is test_query_loader
    assert runtime.context_info["video_metas"] == ["v0"]
    assert runtime.model.anchor.item() == pytest.approx(5.0)
    dataset_cfg = next(item[1] for item in calls if item[0] == "get_datasets")
    assert dataset_cfg["data_root"] == str(data_root.resolve())
    assert dataset_cfg["num_workers"] == 0
    assert dataset_cfg["eval_query_bsz"] == 7
    assert dataset_cfg["eval_context_bsz"] == 3
    assert dataset_cfg["pin_memory"] is False
    assert [item[0] for item in calls] == [
        "get_datasets",
        "get_models",
        "get_validations",
        "compute_context_info",
    ]


def test_benchmark_result_writer_preserves_full_matrices_and_manifest(tmp_path: Path):
    metrics = PRVRMetrics(r1=100.0, r5=100.0, r10=100.0, r100=100.0, rsum=400.0)
    benchmark = DreamPRVRBenchmarkResult(
        base_scores=np.asarray([[0.9, 0.1]], dtype=np.float64),
        reranked_scores=np.asarray([[0.1, 0.9]], dtype=np.float64),
        video_ids=("v0", "v1"),
        query_ids=("v0#0",),
        base_metrics=metrics,
        reranked_metrics=metrics,
        top_k=2,
    )
    run = DreamPRVROneClickResult(
        benchmark=benchmark,
        metadata={"schema_version": 1, "dataset": "activitynet", "top_k": 2},
    )

    manifest_path = save_dreamprvr_benchmark_result(run, tmp_path / "result")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["base_metrics"]["Rsum"] == pytest.approx(400.0)
    assert manifest["metric_delta"]["R@1"] == pytest.approx(0.0)
    assert np.load(manifest_path.parent / manifest["files"]["base_scores"]).shape == (1, 2)
    assert json.loads((manifest_path.parent / "video_ids.json").read_text()) == ["v0", "v1"]
    with pytest.raises(FileExistsError, match="already exists"):
        save_dreamprvr_benchmark_result(run, manifest_path.parent)
