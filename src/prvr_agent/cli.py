from __future__ import annotations

import argparse
import json

from .agents.hypothesis_planner import OpenAIHypothesisPlanner, RuleBasedHypothesisPlanner
from .llm_config import LLMConfig, create_openai_compatible_client, resolve_served_model


def main() -> None:
    parser = argparse.ArgumentParser(description="PRVR-Agent utility CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="Print a rule-based query hypothesis graph for smoke testing")
    p.add_argument("query")

    p_llm = sub.add_parser("plan-llm", help="Generate a query hypothesis graph with the configured local LLM")
    p_llm.add_argument("query")

    sub.add_parser("check-llm", help="Check the configured OpenAI-compatible vLLM endpoint and model id")

    args = parser.parse_args()
    if args.cmd == "plan":
        graph = RuleBasedHypothesisPlanner().plan(args.query)
        print(json.dumps(graph.model_dump(), indent=2, ensure_ascii=False))
    elif args.cmd == "plan-llm":
        graph = OpenAIHypothesisPlanner.from_env().plan(args.query)
        print(json.dumps(graph.model_dump(), indent=2, ensure_ascii=False))
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
