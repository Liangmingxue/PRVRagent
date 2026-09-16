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
        output: dict[str, object] = {
            "indexed_videos": len(resolver),
            "roots": [str(Path(root).expanduser().resolve()) for root in args.roots],
        }
        if args.expected_video_ids:
            expected = _load_json_list(args.expected_video_ids, name="expected_video_ids")
            indexed = set(resolver.video_ids())
            missing = [video_id for video_id in expected if video_id not in indexed]
            output.update(
                {
                    "expected_videos": len(expected),
                    "missing_count": len(missing),
                    "missing_video_ids": missing[:50],
                }
            )
            if missing:
                print(json.dumps(output, indent=2, ensure_ascii=False))
                raise SystemExit(2)
        print(json.dumps(output, indent=2, ensure_ascii=False))
