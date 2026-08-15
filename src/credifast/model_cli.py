"""CLI entrypoint for reproducible CrediFast model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .modeling.application_baseline import train_application_baseline
from .modeling.application_lightgbm import train_application_lightgbm
from .modeling.benchmark import benchmark_history_inference
from .modeling.calibration import calibrate_history_model
from .modeling.explanation_stability import evaluate_explanation_stability
from .modeling.fairness import run_fairness_audit
from .modeling.final_evaluation import run_final_holdout_evaluation
from .modeling.gap_strategy import evaluate_gap_strategy
from .modeling.history_lightgbm import train_history_lightgbm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CrediFast model training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser(
        "application-baseline", help="Train the application-only logistic baseline"
    )
    baseline.add_argument("--input", type=Path, default=Path("data/raw/application_train.csv"))
    baseline.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/models/application_logistic.joblib"),
    )
    baseline.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/application_baseline_metrics.json"),
    )
    baseline.add_argument(
        "--features-output",
        type=Path,
        default=Path("artifacts/application_baseline_features.json"),
    )
    baseline.add_argument(
        "--splits-output",
        type=Path,
        default=Path("data/processed/application_splits.parquet"),
    )
    baseline.add_argument("--seed", type=int, default=42)

    lightgbm = subparsers.add_parser(
        "application-lightgbm", help="Train the application-only LightGBM challenger"
    )
    lightgbm.add_argument("--input", type=Path, default=Path("data/raw/application_train.csv"))
    lightgbm.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/models/application_lightgbm.joblib"),
    )
    lightgbm.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/application_lightgbm_metrics.json"),
    )
    lightgbm.add_argument(
        "--features-output",
        type=Path,
        default=Path("artifacts/application_lightgbm_features.json"),
    )
    lightgbm.add_argument("--seed", type=int, default=42)

    history = subparsers.add_parser(
        "history-lightgbm", help="Train the history-enriched LightGBM challenger"
    )
    history.add_argument(
        "--application-input", type=Path, default=Path("data/raw/application_train.csv")
    )
    history.add_argument(
        "--history-input", type=Path, default=Path("data/processed/history_features.parquet")
    )
    history.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/models/history_lightgbm.joblib"),
    )
    history.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/history_lightgbm_metrics.json"),
    )
    history.add_argument(
        "--features-output",
        type=Path,
        default=Path("artifacts/history_lightgbm_features.json"),
    )
    history.add_argument("--seed", type=int, default=42)

    calibration = subparsers.add_parser(
        "history-calibrate", help="Select and fit probability calibration for the history champion"
    )
    calibration.add_argument(
        "--application-input", type=Path, default=Path("data/raw/application_train.csv")
    )
    calibration.add_argument(
        "--history-input", type=Path, default=Path("data/processed/history_features.parquet")
    )
    calibration.add_argument(
        "--model-input", type=Path, default=Path("artifacts/models/history_lightgbm.joblib")
    )
    calibration.add_argument(
        "--calibrator-output",
        type=Path,
        default=Path("artifacts/models/history_calibrator.joblib"),
    )
    calibration.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/history_calibration_metrics.json"),
    )
    calibration.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/processed/history_calibration_predictions.parquet"),
    )
    calibration.add_argument(
        "--bundle-manifest-output",
        type=Path,
        default=Path("artifacts/champion_bundle_manifest.json"),
    )
    calibration.add_argument("--seed", type=int, default=42)

    fairness = subparsers.add_parser(
        "history-fairness", help="Audit gender and age-band metrics on held-back calibration rows"
    )
    fairness.add_argument(
        "--application-input", type=Path, default=Path("data/raw/application_train.csv")
    )
    fairness.add_argument(
        "--predictions-input",
        type=Path,
        default=Path("data/processed/history_calibration_predictions.parquet"),
    )
    fairness.add_argument(
        "--feature-inventory-input",
        type=Path,
        default=Path("artifacts/history_lightgbm_features.json"),
    )
    fairness.add_argument(
        "--output", type=Path, default=Path("artifacts/history_fairness_audit.json")
    )

    benchmark = subparsers.add_parser(
        "history-benchmark", help="Benchmark warm champion scoring with and without reasons"
    )
    benchmark.add_argument(
        "--application-input", type=Path, default=Path("data/raw/application_test.csv")
    )
    benchmark.add_argument(
        "--history-input", type=Path, default=Path("data/processed/history_features.parquet")
    )
    benchmark.add_argument(
        "--model-input", type=Path, default=Path("artifacts/models/history_lightgbm.joblib")
    )
    benchmark.add_argument(
        "--calibrator-input",
        type=Path,
        default=Path("artifacts/models/history_calibrator.joblib"),
    )
    benchmark.add_argument(
        "--output", type=Path, default=Path("artifacts/inference_benchmark.json")
    )

    stability = subparsers.add_parser(
        "history-explanation-stability",
        help="Probe reason-code stability under small amount perturbations",
    )
    stability.add_argument(
        "--application-input", type=Path, default=Path("data/raw/application_test.csv")
    )
    stability.add_argument(
        "--history-input", type=Path, default=Path("data/processed/history_features.parquet")
    )
    stability.add_argument(
        "--model-input", type=Path, default=Path("artifacts/models/history_lightgbm.joblib")
    )
    stability.add_argument(
        "--calibrator-input",
        type=Path,
        default=Path("artifacts/models/history_calibrator.joblib"),
    )
    stability.add_argument(
        "--output", type=Path, default=Path("artifacts/explanation_stability.json")
    )
    stability.add_argument("--rows", type=int, default=250)
    stability.add_argument("--perturbation-rate", type=float, default=0.005)

    gap_strategy = subparsers.add_parser(
        "history-gap-strategy",
        help="Select and evaluate segment-specific application/history fallback blends",
    )
    gap_strategy.add_argument(
        "--application-input", type=Path, default=Path("data/raw/application_train.csv")
    )
    gap_strategy.add_argument(
        "--history-input", type=Path, default=Path("data/processed/history_features.parquet")
    )
    gap_strategy.add_argument(
        "--predictions-input",
        type=Path,
        default=Path("data/processed/history_calibration_predictions.parquet"),
    )
    gap_strategy.add_argument(
        "--application-model-input",
        type=Path,
        default=Path("artifacts/models/application_lightgbm.joblib"),
    )
    gap_strategy.add_argument(
        "--metrics-output", type=Path, default=Path("artifacts/history_gap_strategy_metrics.json")
    )
    gap_strategy.add_argument(
        "--policy-output", type=Path, default=Path("configs/model_gap_strategy.json")
    )
    gap_strategy.add_argument("--bootstrap-iterations", type=int, default=500)
    gap_strategy.add_argument("--seed", type=int, default=42)

    final_evaluation = subparsers.add_parser(
        "final-holdout",
        help="Evaluate the frozen operational candidate once on the sealed holdout",
    )
    final_evaluation.add_argument(
        "--application-input", type=Path, default=Path("data/raw/application_train.csv")
    )
    final_evaluation.add_argument(
        "--history-input", type=Path, default=Path("data/processed/history_features.parquet")
    )
    final_evaluation.add_argument(
        "--history-model-input",
        type=Path,
        default=Path("artifacts/models/history_lightgbm.joblib"),
    )
    final_evaluation.add_argument(
        "--application-model-input",
        type=Path,
        default=Path("artifacts/models/application_lightgbm.joblib"),
    )
    final_evaluation.add_argument(
        "--calibrator-input",
        type=Path,
        default=Path("artifacts/models/history_calibrator.joblib"),
    )
    final_evaluation.add_argument(
        "--gap-policy-input", type=Path, default=Path("configs/model_gap_strategy.json")
    )
    final_evaluation.add_argument(
        "--metrics-output", type=Path, default=Path("artifacts/final_holdout_evaluation.json")
    )
    final_evaluation.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/processed/final_holdout_predictions.parquet"),
    )
    final_evaluation.add_argument("--bootstrap-iterations", type=int, default=1000)
    final_evaluation.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "application-baseline":
        metrics = train_application_baseline(
            args.input,
            model_output=args.model_output,
            metrics_output=args.metrics_output,
            features_output=args.features_output,
            splits_output=args.splits_output,
            random_seed=args.seed,
        )
        summary = metrics["calibration_partition_metrics"]
        print(
            "Application baseline complete: "
            f"ROC-AUC={summary['roc_auc']:.4f} "
            f"AP={summary['average_precision']:.4f} "
            f"Brier={summary['brier_score']:.4f} "
            f"selected_features={metrics['feature_summary']['selected']}"
        )
    elif args.command == "application-lightgbm":
        metrics = train_application_lightgbm(
            args.input,
            model_output=args.model_output,
            metrics_output=args.metrics_output,
            features_output=args.features_output,
            random_seed=args.seed,
        )
        summary = metrics["calibration_partition_metrics"]
        print(
            "Application LightGBM complete: "
            f"ROC-AUC={summary['roc_auc']:.4f} "
            f"AP={summary['average_precision']:.4f} "
            f"Brier={summary['brier_score']:.4f} "
            f"best_iteration={metrics['best_iteration']}"
        )
    elif args.command == "history-lightgbm":
        metrics = train_history_lightgbm(
            args.application_input,
            args.history_input,
            model_output=args.model_output,
            metrics_output=args.metrics_output,
            features_output=args.features_output,
            random_seed=args.seed,
        )
        summary = metrics["calibration_partition_metrics"]
        print(
            "History LightGBM complete: "
            f"ROC-AUC={summary['roc_auc']:.4f} "
            f"AP={summary['average_precision']:.4f} "
            f"Brier={summary['brier_score']:.4f} "
            f"best_iteration={metrics['best_iteration']}"
        )
    elif args.command == "history-calibrate":
        metrics = calibrate_history_model(
            args.application_input,
            args.history_input,
            args.model_input,
            calibrator_output=args.calibrator_output,
            metrics_output=args.metrics_output,
            predictions_output=args.predictions_output,
            bundle_manifest_output=args.bundle_manifest_output,
            random_seed=args.seed,
        )
        selected = metrics["selected_method"]
        summary = metrics["candidate_evaluation_metrics"][selected]["calibration"]
        print(
            "History calibration complete: "
            f"selected={selected} "
            f"Brier={summary['brier_score']:.4f} "
            f"ECE={summary['expected_calibration_error']:.4f} "
            f"slope={summary['calibration_slope']:.3f}"
        )
    elif args.command == "history-fairness":
        report = run_fairness_audit(
            args.application_input,
            args.predictions_input,
            args.feature_inventory_input,
            output_path=args.output,
        )
        gender_gap = report["dimensions"]["gender"]["observed_gaps"].get(
            "true_positive_rate_at_review_capacity", {}
        )
        print(
            "History fairness audit complete: "
            f"assessment={report['assessment']} rows={report['population']['rows']} "
            f"gender_tpr_gap={gender_gap.get('absolute_gap', 'not-comparable')}"
        )
    elif args.command == "history-benchmark":
        report = benchmark_history_inference(
            args.application_input,
            args.history_input,
            args.model_input,
            args.calibrator_input,
            output_path=args.output,
        )
        gates = report["gates"]
        print(
            "History inference benchmark complete: "
            f"probability_p95_ms={gates['observed_probability_p95_ms']:.2f} "
            f"explained_p95_ms={gates['observed_explained_p95_ms']:.2f}"
        )
    elif args.command == "history-explanation-stability":
        report = evaluate_explanation_stability(
            args.application_input,
            args.history_input,
            args.model_input,
            args.calibrator_input,
            output_path=args.output,
            rows=args.rows,
            perturbation_rate=args.perturbation_rate,
        )
        reasons = report["reason_stability"]
        predictions = report["prediction_stability"]
        print(
            "Explanation stability complete: "
            f"top_reason_match={reasons['top_adverse_reason_match_rate']:.3f} "
            f"top3_jaccard={reasons['mean_top_three_jaccard']:.3f} "
            f"probability_p95_delta={predictions['absolute_probability_change_p95']:.5f}"
        )
    elif args.command == "history-gap-strategy":
        report = evaluate_gap_strategy(
            args.application_input,
            args.history_input,
            args.predictions_input,
            args.application_model_input,
            metrics_output=args.metrics_output,
            policy_output=args.policy_output,
            bootstrap_iterations=args.bootstrap_iterations,
            random_seed=args.seed,
        )
        metrics = report["operational_candidate_overall"]["evaluation_metrics"]
        print(
            "History gap strategy complete: "
            f"weights={report['promotion_gate']['operational_history_weights']} "
            f"ROC-AUC={metrics['ranking']['roc_auc']:.4f} "
            f"Brier={metrics['calibration']['brier_score']:.4f}"
        )
    elif args.command == "final-holdout":
        report = run_final_holdout_evaluation(
            args.application_input,
            args.history_input,
            args.history_model_input,
            args.application_model_input,
            args.calibrator_input,
            args.gap_policy_input,
            metrics_output=args.metrics_output,
            predictions_output=args.predictions_output,
            bootstrap_iterations=args.bootstrap_iterations,
            random_seed=args.seed,
        )
        metrics = report["model_metrics"]["frozen_operational_candidate"]
        print(
            "Final holdout evaluation complete: "
            f"all_gates_passed={report['acceptance_gates']['all_passed']} "
            f"ROC-AUC={metrics['ranking']['roc_auc']:.4f} "
            f"AP={metrics['ranking']['average_precision']:.4f} "
            f"Brier={metrics['calibration']['brier_score']:.4f}"
        )


if __name__ == "__main__":
    main()
