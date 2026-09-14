from __future__ import annotations

import argparse
import json

from .agents.hypothesis_planner import RuleBasedHypothesisPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description="PRVR-Agent utility CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="Print a rule-based query hypothesis graph for smoke testing")
    p.add_argument("query")
    args = parser.parse_args()
    if args.cmd == "plan":
        graph = RuleBasedHypothesisPlanner().plan(args.query)
        print(json.dumps(graph.model_dump(), indent=2, ensure_ascii=False))
