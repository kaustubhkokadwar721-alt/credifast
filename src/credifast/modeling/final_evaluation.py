"""One-time evaluation of the frozen research candidate on the sealed holdout."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .application_baseline import assign_splits
from .application_features import engineer_application_features
from .application_lightgbm import _sha256
from .gap_strategy import _evaluation_metrics, _paired_bootstrap_delta
from .inference import _history_segment, merge_history_with_gaps


def evaluate_acceptance_gates(metrics: dict) -> dict:
    """Apply only the targets documented before holdout was opened."""

    ranking = metrics["ranking"]
    calibration = metrics["calibration"]
    base_rate = float(calibration["base_rate"])
    average_precision_multiple = float(ranking["average_precision"] / base_rate)
    base_rate_brier = float(base_rate * (1.0 - base_rate))
    checks = {
        "roc_auc_at_least_0_75": bool(ranking["roc_auc"] >= 0.75),
        "average_precision_at_least_2_5x_prevalence": bool(
            average_precision_multiple >= 2.5
        ),
        "brier_better_than_base_rate_predictor": bool(
            calibration["brier_score"] < base_rate_brier
        ),
        "calibration_slope_between_0_90_and_1_10": bool(
            0.90 <= calibration["calibration_slope"] <= 1.10
        ),
    }
    return {
        "all_passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "roc_auc": ranking["roc_auc"],
            "average_precision": ranking["average_precision"],
            "target_prevalence": base_rate,
            "average_precision_multiple_of_prevalence": average_precision_multiple,
            "brier_score": calibration["brier_score"],
            "base_rate_brier_score": base_rate_brier,
            "calibration_slope": calibration["calibration_slope"],
        },
    }


def _score_bundle(bundle: dict, engineered: pd.DataFrame) -> np.ndarray:
    selected = bundle["feature_selection"]["selected"]
    missing = sorted(set(selected) - set(engineered.columns))
    if missing:
        raise ValueError(f"holdout input is missing {len(missing)} model features")
    matrix = bundle["preprocessor"].transform(engineered[selected])
    return bundle["model"].predict_proba(matrix)[:, 1]


def run_final_holdout_evaluation(
    application_path: str | Path,
    history_path: str | Path,
    history_model_path: str | Path,
    application_model_path: str | Path,
    calibrator_path: str | Path,
    gap_policy_path: str | Path,
    *,
    metrics_output: str | Path,
    predictions_output: str | Path,
    bootstrap_iterations: int = 1_000,
    random_seed: int = 42,
) -> dict:
    """Score holdout exactly once with immutable, preselected artifacts."""

    application_source = Path(application_path)
    history_source = Path(history_path)
    history_model_source = Path(history_model_path)
    application_model_source = Path(application_model_path)
    calibrator_source = Path(calibrator_path)
    policy_source = Path(gap_policy_path)
    metrics_path = Path(metrics_output)
    if metrics_path.exists():
        raise FileExistsError(
            f"final holdout artifact already exists at {metrics_path}; refusing to overwrite it"
        )

    application = pd.read_csv(application_source, low_memory=False)
    history = pd.read_parquet(history_source)
    history_bundle = joblib.load(history_model_source)
    application_bundle = joblib.load(application_model_source)
    calibrator_bundle = joblib.load(calibrator_source)
    policy = json.loads(policy_source.read_text(encoding="utf-8"))
    if calibrator_bundle.get("method") != "identity":
        raise ValueError("frozen gap policy expects the selected identity calibrator")
    if policy.get("status") != "research_candidate_holdout_not_evaluated":
        raise ValueError("gap policy is not in the expected pre-holdout frozen state")
    if policy.get("component_probability_space") != "raw_model_probability":
        raise ValueError("unsupported component probability space")

    splits = assign_splits(application["TARGET"], random_seed=random_seed)
    holdout_application = application.loc[splits.eq("holdout")].copy()
    requested_ids = set(holdout_application["SK_ID_CURR"])
    holdout_history = history.loc[history["SK_ID_CURR"].isin(requested_ids)].copy()
    expected_history = history_bundle.get("history_features")
    if not expected_history:
        raise ValueError("history model bundle does not declare feature-store schema")
    frame, history_flags, missing_store_row = merge_history_with_gaps(
        holdout_application,
        holdout_history,
        list(expected_history),
    )
    engineered = engineer_application_features(frame)
    history_probability = _score_bundle(history_bundle, engineered)
    application_probability = _score_bundle(application_bundle, engineered)
    source_count = frame[history_flags].sum(axis=1).astype(int)
    segment = source_count.map(_history_segment)
    history_weight = segment.map(policy["history_weights"]).to_numpy(dtype=float)
    operational_probability = (
        history_weight * history_probability
        + (1.0 - history_weight) * application_probability
    )
    target = frame["TARGET"].to_numpy(dtype=int)

    model_metrics = {
        "application_only": _evaluation_metrics(target, application_probability),
        "history_only": _evaluation_metrics(target, history_probability),
        "frozen_operational_candidate": _evaluation_metrics(target, operational_probability),
    }
    segment_metrics = {}
    for segment_name in sorted(segment.unique()):
        mask = segment.eq(segment_name).to_numpy()
        segment_target = target[mask]
        segment_metrics[str(segment_name)] = {
            "rows": int(mask.sum()),
            "events": int(segment_target.sum()),
            "history_weight": float(policy["history_weights"][str(segment_name)]),
            "application_only": _evaluation_metrics(
                segment_target, application_probability[mask]
            ),
            "history_only": _evaluation_metrics(segment_target, history_probability[mask]),
            "frozen_operational_candidate": _evaluation_metrics(
                segment_target, operational_probability[mask]
            ),
        }
    exact_source_metrics = {}
    for count in sorted(source_count.unique()):
        mask = source_count.eq(count).to_numpy()
        count_target = target[mask]
        events = int(count_target.sum())
        exact_source_metrics[str(count)] = {
            "status": (
                "evaluated" if int(mask.sum()) >= 500 and events >= 30 else "exploratory_low_sample"
            ),
            "rows": int(mask.sum()),
            "events": events,
            "frozen_operational_candidate": _evaluation_metrics(
                count_target, operational_probability[mask]
            ),
        }

    candidate_metrics = model_metrics["frozen_operational_candidate"]
    report = {
        "run_version": "1.0.0",
        "status": "final_random_holdout_evaluation_complete_no_retuning",
        "evaluation_protocol": {
            "split": "sealed_holdout",
            "split_seed": random_seed,
            "rows": len(frame),
            "events": int(target.sum()),
            "bootstrap_iterations": bootstrap_iterations,
            "policy_version": policy["policy_version"],
            "policy_weights": policy["history_weights"],
            "note": (
                "All artifacts and segment weights were frozen before this evaluation. Results "
                "must not trigger another tuning cycle on this holdout."
            ),
        },
        "acceptance_gates": evaluate_acceptance_gates(candidate_metrics),
        "model_metrics": model_metrics,
        "paired_uncertainty": {
            "candidate_vs_history": _paired_bootstrap_delta(
                target,
                operational_probability,
                history_probability,
                iterations=bootstrap_iterations,
                random_seed=random_seed + 301,
            ),
            "candidate_vs_application": _paired_bootstrap_delta(
                target,
                operational_probability,
                application_probability,
                iterations=bootstrap_iterations,
                random_seed=random_seed + 302,
            ),
        },
        "history_segments": segment_metrics,
        "exact_source_count": exact_source_metrics,
        "data_quality": {
            "feature_store_rows_missing": int(missing_store_row.sum()),
            "feature_store_row_missing_rate": float(missing_store_row.mean()),
            "availability_flag_columns": history_flags,
        },
        "controls": {
            "protected_audit_fields_used_for_prediction": False,
            "post_holdout_retuning_allowed": False,
            "production_approved": False,
        },
        "limitations": [
            "This is a random competition holdout, not out-of-time or lender-population evidence.",
            "Subgroup and exact-source estimates may be unstable where event counts are low.",
            "Passing technical targets does not establish regulatory, fairness, or business approval.",
        ],
        "artifacts": {
            "history_model": {
                "path": str(history_model_source),
                "sha256": _sha256(history_model_source),
            },
            "application_model": {
                "path": str(application_model_source),
                "sha256": _sha256(application_model_source),
            },
            "identity_calibrator": {
                "path": str(calibrator_source),
                "sha256": _sha256(calibrator_source),
            },
            "gap_policy": {"path": str(policy_source), "sha256": _sha256(policy_source)},
            "history_features": {
                "path": str(history_source),
                "sha256": _sha256(history_source),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    prediction_frame = pd.DataFrame(
        {
            "SK_ID_CURR": frame["SK_ID_CURR"].to_numpy(),
            "TARGET": target,
            "history_segment": segment.to_numpy(),
            "history_source_count": source_count.to_numpy(),
            "history_weight": history_weight,
            "application_probability": application_probability,
            "history_probability": history_probability,
            "operational_probability": operational_probability,
            "feature_store_row_missing": missing_store_row.to_numpy(),
        }
    )
    prediction_path = Path(predictions_output)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_parquet(prediction_path, index=False)
    return report
