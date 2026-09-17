from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents.hypothesis_planner import OpenAIHypothesisPlanner, RuleBasedHypothesisPlanner
from .agents.world_model import OpenAIEventWorldPlanner, RuleBasedEventWorldPlanner
from .llm_config import LLMConfig, create_openai_compatible_client, resolve_served_model


def _load_json_list(path: str | Path, *, name: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise ValueError(f"{name} must be a JSON array of strings")
    if not payload:
        raise ValueError(f"{name} must not be empty")
    return payload


def _coverage_output(resolver, roots: list[str], expected_path: str | None) -> dict[str, object]:
    output: dict[str, object] = {
        "indexed_sources": len(resolver),
        "roots": [str(Path(root).expanduser().resolve()) for root in roots],
    }
    if expected_path:
        expected = _load_json_list(expected_path, name="expected_video_ids")
        indexed = set(resolver.video_ids())
        missing = [video_id for video_id in expected if video_id not in indexed]
        output.update(
            {
                "expected_videos": len(expected),
                "missing_count": len(missing),
                "missing_video_ids": missing[:50],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="PRVR-Agent utility CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="Print a rule-based Counterfactual Query Hypothesis Graph")
    p.add_argument("query")

    p_llm = sub.add_parser("plan-llm", help="Generate a CQHG with the configured local LLM")
    p_llm.add_argument("query")

    p_world = sub.add_parser("imagine", help="Generate rule-based prospective event worlds for smoke testing")
    p_world.add_argument("query")
    p_world.add_argument("--num-worlds", type=int, default=3)

    p_world_llm = sub.add_parser("imagine-llm", help="Generate CQHG-conditioned event worlds with the local LLM")
    p_world_llm.add_argument("query")
    p_world_llm.add_argument("--num-worlds", type=int, default=3)

    sub.add_parser("check-llm", help="Check the configured OpenAI-compatible vLLM endpoint and model id")

    p_metrics = sub.add_parser(
        "eval-metrics",
        help="Evaluate a full PRVR score matrix with DreamPRVR-compatible recall metrics",
    )
    p_metrics.add_argument("--scores", required=True, help="NumPy .npy [num_queries, num_videos] score matrix")
    p_metrics.add_argument("--video-ids", required=True, help="JSON array in the score-matrix video-column order")
    p_metrics.add_argument("--query-ids", required=True, help="JSON array in the score-matrix query-row order")

    p_videos = sub.add_parser(
        "index-videos",
        help="Index a raw-video tree by filename stem and optionally verify benchmark coverage",
    )
    p_videos.add_argument("roots", nargs="+", help="One or more raw-video root directories")
    p_videos.add_argument(
        "--expected-video-ids",
        help="Optional JSON array of benchmark video ids that must all be present",
    )

    p_frames = sub.add_parser(
        "index-frames",
        help="Index extracted-frame directories by directory name and optionally verify benchmark coverage",
    )
    p_frames.add_argument("roots", nargs="+", help="One or more frame root directories")
    p_frames.add_argument(
        "--expected-video-ids",
        help="Optional JSON array of benchmark video ids that must all be present",
    )

    p_benchmark = sub.add_parser(
        "benchmark-dreamprvr",
        help="Run official DreamPRVR test retrieval followed by CQHG/APEI Top-K reranking",
    )
    p_benchmark.add_argument(
        "--dreamprvr-root",
        required=True,
        help="Official CVPR26-DreamPRVR checkout containing src/",
    )
    p_benchmark.add_argument(
        "--checkpoint",
        required=True,
        help="Official DreamPRVR checkpoint for the selected dataset",
    )
    p_benchmark.add_argument(
        "--data-root",
        required=True,
        help="DreamPRVR feature root containing activitynet/, charades/, and/or tvr/",
    )
    p_benchmark.add_argument(
        "--dataset",
        required=True,
        help="tvr, activitynet/act, or charades/cha",
    )
    p_benchmark.add_argument(
        "--visual-root",
        dest="visual_roots",
        action="append",
        required=True,
        help="Raw-video root (ActivityNet/Charades) or extracted-frame root (TVR); repeatable",
    )
    p_benchmark.add_argument(
        "--frame-directory-fps",
        type=float,
        help="Required for TVR extracted frames; use 3 only for the matching 3-FPS release",
    )
    p_benchmark.add_argument("--top-k", type=int, default=20)
    p_benchmark.add_argument(
        "--max-queries",
        type=int,
        help="Limit expensive Qwen reranking; use 10 for the first smoke test",
    )
    p_benchmark.add_argument(
        "--device",
        default="auto",
        help="DreamPRVR device, for example auto, cpu, cuda, or cuda:0",
    )
    p_benchmark.add_argument("--num-workers", type=int, default=4)
    p_benchmark.add_argument("--eval-query-batch-size", type=int)
    p_benchmark.add_argument("--eval-context-batch-size", type=int)
    p_benchmark.add_argument(
        "--output-dir",
        required=True,
        help="Directory for full baseline/reranked matrices, ids, and metrics manifest",
    )
    p_benchmark.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace benchmark files already present in --output-dir",
    )

    args = parser.parse_args()
    if args.cmd == "plan":
        graph = RuleBasedHypothesisPlanner().plan(args.query)
        print(json.dumps(graph.model_dump(), indent=2, ensure_ascii=False))
    elif args.cmd == "plan-llm":
        graph = OpenAIHypothesisPlanner.from_env().plan(args.query)
        print(json.dumps(graph.model_dump(), indent=2, ensure_ascii=False))
    elif args.cmd == "imagine":
        graph = RuleBasedHypothesisPlanner().plan(args.query)
        worlds = RuleBasedEventWorldPlanner().imagine(graph, num_worlds=args.num_worlds)
        print(json.dumps(worlds.model_dump(), indent=2, ensure_ascii=False))
    elif args.cmd == "imagine-llm":
        graph = OpenAIHypothesisPlanner.from_env().plan(args.query)
        worlds = OpenAIEventWorldPlanner.from_env().imagine(graph, num_worlds=args.num_worlds)
        print(json.dumps(worlds.model_dump(), indent=2, ensure_ascii=False))
    elif args.cmd == "check-llm":
        cfg = LLMConfig.from_env()
        client = create_openai_compatible_client(cfg)
        model = resolve_served_model(client, cfg.model)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "base_url": cfg.normalized_base_url(),
                    "model": model,
                    "timeout": cfg.timeout,
                },
                indent=2,
            )
        )
    elif args.cmd == "eval-metrics":
        # Keep heavy benchmark-only imports out of lightweight planner commands.
        import numpy as np

        from .evaluation import evaluate_prvr_scores

        scores = np.load(args.scores, allow_pickle=False)
        video_ids = _load_json_list(args.video_ids, name="video_ids")
        query_ids = _load_json_list(args.query_ids, name="query_ids")
        metrics = evaluate_prvr_scores(
            scores,
            video_ids=video_ids,
            query_ids=query_ids,
        )
        print(json.dumps(metrics.as_dict(), indent=2))
    elif args.cmd == "index-videos":
        from .evaluation import IndexedVideoPathResolver

        resolver = IndexedVideoPathResolver.from_roots(args.roots)
        output = _coverage_output(resolver, args.roots, args.expected_video_ids)
        output["source_kind"] = "raw_video"
        print(json.dumps(output, indent=2, ensure_ascii=False))
        if output.get("missing_count", 0):
            raise SystemExit(2)
    elif args.cmd == "index-frames":
        from .evaluation import IndexedFrameDirectoryResolver

        resolver = IndexedFrameDirectoryResolver.from_roots(args.roots)
        output = _coverage_output(resolver, args.roots, args.expected_video_ids)
        output["source_kind"] = "extracted_frames"
        print(json.dumps(output, indent=2, ensure_ascii=False))
        if output.get("missing_count", 0):
            raise SystemExit(2)
    elif args.cmd == "benchmark-dreamprvr":
        # Importing the integration lazily keeps planner/index commands usable in
        # lightweight environments without the DreamPRVR torch stack.
        from .integration import run_dreamprvr_benchmark, save_dreamprvr_benchmark_result

        run = run_dreamprvr_benchmark(
            dreamprvr_root=args.dreamprvr_root,
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            dataset=args.dataset,
            visual_roots=args.visual_roots,
            frame_directory_fps=args.frame_directory_fps,
            top_k=args.top_k,
            max_queries=args.max_queries,
            device=args.device,
            num_workers=args.num_workers,
            eval_query_batch_size=args.eval_query_batch_size,
            eval_context_batch_size=args.eval_context_batch_size,
        )
        manifest = save_dreamprvr_benchmark_result(
            run,
            args.output_dir,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "manifest": str(manifest),
                    "base_metrics": run.benchmark.base_metrics.as_dict(),
                    "reranked_metrics": run.benchmark.reranked_metrics.as_dict(),
                    "metric_delta": run.benchmark.metric_delta(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
