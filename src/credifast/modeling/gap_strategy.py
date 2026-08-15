"""Select and evaluate history-coverage fallback blends without touching holdout."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .application_baseline import evaluate_probabilities
from .application_features import engineer_application_features
from .application_lightgbm import _sha256
from .calibration import calibration_diagnostics

DEFAULT_HISTORY_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)


def blend_probabilities(
    application_probability: np.ndarray,
    history_probability: np.ndarray,
    history_weight: float,
) -> np.ndarray:
    """Return a convex blend, validating inputs rather than silently clipping them."""

    if not 0.0 <= history_weight <= 1.0:
        raise ValueError("history_weight must be between zero and one")
    application = np.asarray(application_probability, dtype=float)
    history = np.asarray(history_probability, dtype=float)
    if application.shape != history.shape:
        raise ValueError("application and history probabilities must have matching shapes")
    if not (
        np.isfinite(application).all()
        and np.isfinite(history).all()
        and ((0.0 < application) & (application < 1.0)).all()
        and ((0.0 < history) & (history < 1.0)).all()
    ):
        raise ValueError("component probabilities must be finite and strictly between zero and one")
    return history_weight * history + (1.0 - history_weight) * application


def select_history_weight(
    target: np.ndarray,
    application_probability: np.ndarray,
    history_probability: np.ndarray,
    *,
    candidates: tuple[float, ...] = DEFAULT_HISTORY_WEIGHTS,
) -> tuple[float, list[dict]]:
    """Choose a coarse, regularized blend by Brier score then log loss.

    Coarse weights reduce selection variance in small thin-file segments. A final
    tie-break favors more application-model weight because that model does not
    depend on the historical feature store that defines the gap.
    """

    y = np.asarray(target, dtype=int)
    if len(y) == 0 or set(np.unique(y)) - {0, 1}:
        raise ValueError("target must be a non-empty binary array")
    rows = []
    for weight in candidates:
        probability = blend_probabilities(
            application_probability, history_probability, float(weight)
        )
        rows.append(
            {
                "history_weight": float(weight),
                "application_weight": float(1.0 - weight),
                "brier_score": float(brier_score_loss(y, probability)),
                "log_loss": float(log_loss(y, probability)),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["brier_score"],
            row["log_loss"],
            row["history_weight"],
        ),
    )
    return float(selected["history_weight"]), rows


def _paired_bootstrap_delta(
    target: np.ndarray,
    candidate_probability: np.ndarray,
    reference_probability: np.ndarray,
    *,
    iterations: int,
    random_seed: int,
) -> dict:
    """Estimate paired uncertainty while preserving event/non-event counts."""

    y = np.asarray(target, dtype=int)
    candidate = np.asarray(candidate_probability, dtype=float)
    reference = np.asarray(reference_probability, dtype=float)
    class_positions = [np.flatnonzero(y == value) for value in (0, 1)]
    if min(map(len, class_positions)) < 2:
        return {"status": "insufficient_class_count"}
    rng = np.random.default_rng(random_seed)
    brier_deltas = np.empty(iterations)
    auc_deltas = np.empty(iterations)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(positions, size=len(positions), replace=True) for positions in class_positions]
        )
        sampled_target = y[sampled]
        brier_deltas[iteration] = brier_score_loss(
            sampled_target, candidate[sampled]
        ) - brier_score_loss(sampled_target, reference[sampled])
        auc_deltas[iteration] = roc_auc_score(
            sampled_target, candidate[sampled]
        ) - roc_auc_score(sampled_target, reference[sampled])
    observed_brier = float(
        brier_score_loss(y, candidate) - brier_score_loss(y, reference)
    )
    observed_auc = float(roc_auc_score(y, candidate) - roc_auc_score(y, reference))
    return {
        "status": "estimated",
        "iterations": iterations,
        "interpretation": (
            "Negative Brier delta and positive ROC-AUC delta favor the candidate hybrid."
        ),
        "brier_delta": {
            "observed": observed_brier,
            "confidence_interval_95": [
                float(np.quantile(brier_deltas, 0.025)),
                float(np.quantile(brier_deltas, 0.975)),
            ],
            "bootstrap_probability_candidate_better": float((brier_deltas < 0.0).mean()),
        },
        "roc_auc_delta": {
            "observed": observed_auc,
            "confidence_interval_95": [
                float(np.quantile(auc_deltas, 0.025)),
                float(np.quantile(auc_deltas, 0.975)),
            ],
            "bootstrap_probability_candidate_better": float((auc_deltas > 0.0).mean()),
        },
    }


def _evaluation_metrics(target: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "calibration": calibration_diagnostics(target, probability),
        "ranking": evaluate_probabilities(pd.Series(target), probability),
    }


def _validate_inputs(application: pd.DataFrame, predictions: pd.DataFrame) -> None:
    required_application = {"SK_ID_CURR", "TARGET"}
    required_prediction = {
        "SK_ID_CURR",
        "TARGET",
        "calibration_role",
        "history_segment",
        "raw_probability",
    }
    if not required_application.issubset(application):
        raise ValueError("application input requires SK_ID_CURR and TARGET")
    if not required_prediction.issubset(predictions):
        raise ValueError("calibration predictions are missing required columns")
    if application["SK_ID_CURR"].duplicated().any() or predictions["SK_ID_CURR"].duplicated().any():
        raise ValueError("gap strategy requires unique application keys")
    roles = set(predictions["calibration_role"].dropna().unique())
    if roles != {"calibrator_fit", "calibrator_evaluation"}:
        raise ValueError("calibration roles must contain disjoint fit and evaluation rows")
    if not predictions["raw_probability"].between(0.0, 1.0, inclusive="neither").all():
        raise ValueError("history raw probabilities must be strictly between zero and one")


def evaluate_gap_strategy(
    application_path: str | Path,
    history_path: str | Path,
    predictions_path: str | Path,
    application_model_path: str | Path,
    *,
    metrics_output: str | Path,
    policy_output: str | Path,
    bootstrap_iterations: int = 500,
    random_seed: int = 42,
) -> dict:
    """Select segment weights on calibration-fit and evaluate once on calibration-eval."""

    application_source = Path(application_path)
    history_source = Path(history_path)
    predictions_source = Path(predictions_path)
    model_source = Path(application_model_path)
    application = pd.read_csv(application_source, low_memory=False)
    history = pd.read_parquet(history_source)
    predictions = pd.read_parquet(predictions_source)
    _validate_inputs(application, predictions)
    if "SK_ID_CURR" not in history or history["SK_ID_CURR"].duplicated().any():
        raise ValueError("history feature store requires a unique SK_ID_CURR key")
    source_flags = [column for column in history if column.startswith("HAS_")]
    if not source_flags:
        raise ValueError("history feature store requires HAS_* availability flags")

    application_bundle = joblib.load(model_source)
    required_bundle = {"preprocessor", "model", "feature_selection", "model_name"}
    if not required_bundle.issubset(application_bundle):
        raise ValueError("application model bundle is incomplete")
    calibration_application = predictions[["SK_ID_CURR"]].merge(
        application,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    if calibration_application["TARGET"].isna().any():
        raise ValueError("one or more prediction IDs are absent from application input")
    if not np.array_equal(
        calibration_application["TARGET"].to_numpy(dtype=int),
        predictions["TARGET"].to_numpy(dtype=int),
    ):
        raise ValueError("targets disagree between application and prediction artifacts")

    engineered = engineer_application_features(calibration_application)
    selected = application_bundle["feature_selection"]["selected"]
    missing_features = sorted(set(selected) - set(engineered.columns))
    if missing_features:
        raise ValueError(f"application model input is missing {len(missing_features)} features")
    matrix = application_bundle["preprocessor"].transform(engineered[selected])
    application_probability = application_bundle["model"].predict_proba(matrix)[:, 1]

    frame = predictions.copy()
    frame["application_probability"] = application_probability
    frame["history_probability"] = frame["raw_probability"].astype(float)
    history_coverage = history[["SK_ID_CURR", *source_flags]].copy()
    history_coverage["history_source_count"] = history_coverage[source_flags].fillna(0).sum(axis=1)
    frame = frame.merge(
        history_coverage[["SK_ID_CURR", "history_source_count"]],
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
        indicator="history_store_join",
    )
    frame["history_store_row_missing"] = frame["history_store_join"].eq("left_only")
    frame["history_source_count"] = frame["history_source_count"].fillna(0).astype(int)
    frame = frame.drop(columns="history_store_join")
    segment_results = {}
    selected_weights = {}
    for segment_index, segment_name in enumerate(sorted(frame["history_segment"].unique())):
        segment = frame.loc[frame["history_segment"].eq(segment_name)].copy()
        fit = segment.loc[segment["calibration_role"].eq("calibrator_fit")]
        evaluation = segment.loc[segment["calibration_role"].eq("calibrator_evaluation")]
        weight, candidates = select_history_weight(
            fit["TARGET"].to_numpy(dtype=int),
            fit["application_probability"].to_numpy(),
            fit["history_probability"].to_numpy(),
        )
        selected_weights[str(segment_name)] = weight
        hybrid_probability = blend_probabilities(
            evaluation["application_probability"].to_numpy(),
            evaluation["history_probability"].to_numpy(),
            weight,
        )
        target = evaluation["TARGET"].to_numpy(dtype=int)
        app_probability = evaluation["application_probability"].to_numpy()
        history_probability = evaluation["history_probability"].to_numpy()
        segment_results[str(segment_name)] = {
            "selection_population": {
                "rows": len(fit),
                "base_rate": float(fit["TARGET"].mean()),
                "candidate_metrics": candidates,
                "selected_history_weight": weight,
            },
            "evaluation_population": {
                "rows": len(evaluation),
                "base_rate": float(evaluation["TARGET"].mean()),
            },
            "evaluation_metrics": {
                "application_only": _evaluation_metrics(target, app_probability),
                "history_only": _evaluation_metrics(target, history_probability),
                "selected_hybrid": _evaluation_metrics(target, hybrid_probability),
            },
            "paired_uncertainty": {
                "hybrid_vs_application": _paired_bootstrap_delta(
                    target,
                    hybrid_probability,
                    app_probability,
                    iterations=bootstrap_iterations,
                    random_seed=random_seed + segment_index * 10 + 1,
                ),
                "hybrid_vs_history": _paired_bootstrap_delta(
                    target,
                    hybrid_probability,
                    history_probability,
                    iterations=bootstrap_iterations,
                    random_seed=random_seed + segment_index * 10 + 2,
                ),
            },
        }

    evaluation = frame.loc[frame["calibration_role"].eq("calibrator_evaluation")].copy()
    evaluation["history_weight"] = evaluation["history_segment"].map(selected_weights)
    evaluation["hybrid_probability"] = [
        blend_probabilities(np.array([app]), np.array([history]), float(weight))[0]
        for app, history, weight in zip(
            evaluation["application_probability"],
            evaluation["history_probability"],
            evaluation["history_weight"],
            strict=True,
        )
    ]
    target = evaluation["TARGET"].to_numpy(dtype=int)
    hybrid_probability = evaluation["hybrid_probability"].to_numpy()
    app_probability = evaluation["application_probability"].to_numpy()
    history_probability = evaluation["history_probability"].to_numpy()
    overall = {
        "evaluation_population": {
            "rows": len(evaluation),
            "base_rate": float(evaluation["TARGET"].mean()),
        },
        "evaluation_metrics": {
            "application_only": _evaluation_metrics(target, app_probability),
            "history_only": _evaluation_metrics(target, history_probability),
            "selected_segment_hybrid": _evaluation_metrics(target, hybrid_probability),
        },
        "paired_uncertainty": {
            "hybrid_vs_application": _paired_bootstrap_delta(
                target,
                hybrid_probability,
                app_probability,
                iterations=bootstrap_iterations,
                random_seed=random_seed + 101,
            ),
            "hybrid_vs_history": _paired_bootstrap_delta(
                target,
                hybrid_probability,
                history_probability,
                iterations=bootstrap_iterations,
                random_seed=random_seed + 102,
            ),
        },
    }

    promotion_decisions = {}
    operational_weights = {}
    for segment_name, segment_report in segment_results.items():
        selected_weight = selected_weights[segment_name]
        metrics = segment_report["evaluation_metrics"]
        uncertainty = segment_report["paired_uncertainty"]["hybrid_vs_history"]
        history_metrics = metrics["history_only"]
        hybrid_metrics = metrics["selected_hybrid"]
        criteria = {
            "brier_improves": (
                hybrid_metrics["calibration"]["brier_score"]
                < history_metrics["calibration"]["brier_score"]
            ),
            "log_loss_improves": (
                hybrid_metrics["calibration"]["log_loss"]
                < history_metrics["calibration"]["log_loss"]
            ),
            "average_precision_not_worse": (
                hybrid_metrics["ranking"]["average_precision"]
                >= history_metrics["ranking"]["average_precision"]
            ),
            "bootstrap_brier_support_at_least_90_percent": (
                uncertainty["brier_delta"]["bootstrap_probability_candidate_better"] >= 0.90
            ),
        }
        candidate_differs_from_champion = selected_weight != 1.0
        promote = candidate_differs_from_champion and all(criteria.values())
        operational_weights[segment_name] = selected_weight if promote else 1.0
        promotion_decisions[segment_name] = {
            "selected_candidate_history_weight": selected_weight,
            "promoted_history_weight": operational_weights[segment_name],
            "candidate_promoted": promote,
            "decision": (
                "promote_fit_selected_blend"
                if promote
                else "retain_existing_history_champion"
            ),
            "criteria": criteria,
        }

    evaluation["operational_history_weight"] = evaluation["history_segment"].map(
        operational_weights
    )
    evaluation["operational_probability"] = [
        blend_probabilities(np.array([app]), np.array([history]), float(weight))[0]
        for app, history, weight in zip(
            evaluation["application_probability"],
            evaluation["history_probability"],
            evaluation["operational_history_weight"],
            strict=True,
        )
    ]
    operational_probability = evaluation["operational_probability"].to_numpy()
    operational_overall = {
        "evaluation_metrics": _evaluation_metrics(target, operational_probability),
        "paired_uncertainty_vs_history": _paired_bootstrap_delta(
            target,
            operational_probability,
            history_probability,
            iterations=bootstrap_iterations,
            random_seed=random_seed + 201,
        ),
    }
    exact_source_count = {}
    for source_count in sorted(evaluation["history_source_count"].unique()):
        stratum = evaluation.loc[evaluation["history_source_count"].eq(source_count)]
        stratum_target = stratum["TARGET"].to_numpy(dtype=int)
        events = int(stratum_target.sum())
        exact_source_count[str(source_count)] = {
            "status": (
                "evaluated" if len(stratum) >= 500 and events >= 30 else "exploratory_low_sample"
            ),
            "rows": len(stratum),
            "events": events,
            "base_rate": float(stratum_target.mean()),
            "application_only": _evaluation_metrics(
                stratum_target, stratum["application_probability"].to_numpy()
            ),
            "history_only": _evaluation_metrics(
                stratum_target, stratum["history_probability"].to_numpy()
            ),
            "operational_strategy": _evaluation_metrics(
                stratum_target, stratum["operational_probability"].to_numpy()
            ),
        }

    report = {
        "run_version": "1.0.0",
        "objective": (
            "Maximize probability quality under incomplete history while preserving a sealed holdout."
        ),
        "selection_design": {
            "selection_role": "calibrator_fit",
            "evaluation_role": "calibrator_evaluation",
            "candidate_history_weights": list(DEFAULT_HISTORY_WEIGHTS),
            "selection_rule": "minimum Brier score, then log loss, then lower history weight",
            "bootstrap_iterations": bootstrap_iterations,
            "component_probability_space": "raw model probabilities",
            "rationale": (
                "Both component models were trained only on development rows. Coarse blend weights "
                "limit variance in the smallest history segment."
            ),
        },
        "selected_history_weights": selected_weights,
        "promotion_gate": {
            "purpose": (
                "Retain the existing history champion unless the fit-selected candidate also "
                "improves proper scoring rules and ranking quality on disjoint evaluation rows."
            ),
            "criteria": (
                "Brier and log loss improve, average precision does not worsen, and paired "
                "bootstrap support for Brier improvement is at least 90%."
            ),
            "decisions": promotion_decisions,
            "operational_history_weights": operational_weights,
            "note": (
                "This is validation-based champion selection. The operational candidate still "
                "requires one final evaluation on the sealed holdout."
            ),
        },
        "segments": segment_results,
        "overall": overall,
        "operational_candidate_overall": operational_overall,
        "exact_source_count_evaluation": exact_source_count,
        "feature_store_gap_interpretation": {
            "empirical_zero_source_rows": int(evaluation["history_source_count"].eq(0).sum()),
            "missing_feature_store_rows_in_evaluation": int(
                evaluation["history_store_row_missing"].sum()
            ),
            "note": (
                "Zero-source applicants are the empirical proxy for absent history. A missing "
                "feature-store row is additionally treated as a pipeline-quality fault at inference."
            ),
        },
        "holdout_evaluated": False,
        "limitations": [
            "Calibration-evaluation is an internal validation partition, not the sealed holdout.",
            "Thin-file uncertainty is wider because that segment has fewer events and rows.",
            "The strategy does not infer that absent history means good or bad credit behavior.",
            "The prototype is not approved for autonomous lending or adverse action.",
        ],
        "artifacts": {
            "application_model": {
                "path": str(model_source),
                "sha256": _sha256(model_source),
            },
            "calibration_predictions": {
                "path": str(predictions_source),
                "sha256": _sha256(predictions_source),
            },
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
    metrics_path = Path(metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    policy = {
        "policy_version": "history-gap-strategy-0.1.0",
        "status": "research_candidate_holdout_not_evaluated",
        "component_probability_space": "raw_model_probability",
        "history_weights": operational_weights,
        "fit_selected_candidate_history_weights": selected_weights,
        "promotion_decisions": promotion_decisions,
        "selection_population": "calibrator_fit",
        "evaluation_population": "calibrator_evaluation",
        "selection_rule": report["selection_design"]["selection_rule"],
        "missing_feature_store_row": {
            "history_segment": "thin_0_to_1_sources",
            "availability_flags": 0,
            "force_route": "MANUAL_REVIEW_DATA_LIMITED",
            "quality_flag": "FEATURE_STORE_ROW_MISSING",
        },
        "controls": {
            "holdout_evaluated": False,
            "production_approved": False,
            "autonomous_decision_allowed": False,
        },
    }
    policy_path = Path(policy_output)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    return report
