"""Run the predeclared temporal challenger campaign without scoring the holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from credifast.modeling.temporal_experiments import run_temporal_challenger_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--application",
        type=Path,
        default=Path("data/raw/application_train.csv"),
    )
    parser.add_argument(
        "--history-v1",
        type=Path,
        default=Path("data/processed/history_features.parquet"),
    )
    parser.add_argument(
        "--history-v2",
        type=Path,
        default=Path("data/processed/history_temporal_features.parquet"),
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=Path("artifacts/models/history_lightgbm.joblib"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/temporal_challenger_experiment.json"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/temporal_challenger_experiments.json"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("artifacts/temporal_challenger_comparison.csv"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path(
            "data/processed/temporal_challenger_calibration_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=Path("artifacts/models/research"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_temporal_challenger_experiments(
        args.application,
        args.history_v1,
        args.history_v2,
        args.baseline_model,
        args.config,
        metrics_output=args.metrics_output,
        comparison_output=args.comparison_output,
        predictions_output=args.predictions_output,
        model_output_dir=args.model_output_dir,
    )
    best = report["best_calibration_experiment"]
    summary = {
        "status": report["status"],
        "best_calibration_experiment": best,
        "best_average_precision": report["results"][best]["metrics"][
            "average_precision"
        ],
        "research_gate_passing_experiments": report[
            "research_gate_passing_experiments"
        ],
        "sealed_holdout_evaluated": report["population"][
            "sealed_holdout_evaluated"
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
