"""CLI for batch or single-application research champion scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .modeling.inference import score_history_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score applications with the CrediFast champion")
    parser.add_argument(
        "--application-input", type=Path, default=Path("data/raw/application_test.csv")
    )
    parser.add_argument(
        "--history-input", type=Path, default=Path("data/processed/history_features.parquet")
    )
    parser.add_argument(
        "--model-input", type=Path, default=Path("artifacts/models/history_lightgbm.joblib")
    )
    parser.add_argument(
        "--calibrator-input",
        type=Path,
        default=Path("artifacts/models/history_calibrator.joblib"),
    )
    parser.add_argument(
        "--application-model-input",
        type=Path,
        default=Path("artifacts/models/application_lightgbm.joblib"),
    )
    parser.add_argument(
        "--gap-strategy-input",
        type=Path,
        default=Path("configs/model_gap_strategy.json"),
    )
    parser.add_argument("--application-id", type=int)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--no-explanations", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/history_demo_scores.jsonl")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records, timings = score_history_files(
        args.application_input,
        args.history_input,
        args.model_input,
        args.calibrator_input,
        application_model_path=args.application_model_input,
        gap_strategy_path=args.gap_strategy_input,
        application_id=args.application_id,
        limit=None if args.application_id is not None else args.limit,
        explain=not args.no_explanations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    first = records[0]
    print(
        f"Scored {len(records)} applications to {args.output}; "
        f"first_id={first['application_id']} "
        f"probability={first['repayment_difficulty_probability']:.4f} "
        f"grade={first['risk_grade']} route={first['review_route']} "
        f"total_ms_per_row={timings['total_milliseconds_per_row']:.3f}"
    )


if __name__ == "__main__":
    main()
