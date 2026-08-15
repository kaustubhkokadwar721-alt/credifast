"""Command-line interface for local CrediFast scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .policy import DecisionPolicy
from .service import evaluate_application


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a demo applicant JSON file")
    parser.add_argument("input", type=Path, help="Path to applicant JSON")
    parser.add_argument("--policy", type=Path, help="Optional decision-policy JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    values = json.loads(args.input.read_text(encoding="utf-8"))
    policy = DecisionPolicy.from_json(args.policy) if args.policy else None
    print(json.dumps(evaluate_application(values, policy), indent=2))


if __name__ == "__main__":
    main()
