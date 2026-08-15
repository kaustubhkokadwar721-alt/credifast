"""Measure what the collectible-input pivot costs, on the calibration partition only.

Trains four feature policies under identical splits, hyperparameters and early-stopping
protocol, then reports paired bootstrap intervals against the no-external-score baseline.

The V1 final holdout is spent. This script never reads holdout rows and never writes to
``artifacts/final_holdout_evaluation.json``. Results select nothing beyond further research.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\run_collectible_experiments.py
"""

from __future__ import annotations

import gc
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credifast.modeling.application_baseline import (
    assign_splits,
    evaluate_probabilities,
)
from credifast.modeling.application_features import (
    engineer_application_features,
    select_responsible_features,
)
from credifast.modeling.application_lightgbm import _preprocessor
from credifast.modeling.collectible_features import (
    POLICY_VERSION,
    VARIANT_DESCRIPTIONS,
    add_obligation_ratios,
    recompute_missing_count,
    restrict_selection,
)

APPLICATION_PATH = ROOT / "data" / "raw" / "application_train.csv"
HISTORY_PATH = ROOT / "data" / "processed" / "history_features.parquet"
METRICS_OUTPUT = ROOT / "artifacts" / "collectible_experiments.json"
COMPARISON_OUTPUT = ROOT / "artifacts" / "collectible_comparison.csv"
PREDICTIONS_OUTPUT = ROOT / "data" / "processed" / "collectible_calibration_predictions.parquet"

RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 300
BASELINE_VARIANT = "v1_no_external_score"
VARIANTS = ("v1_reference", "v1_no_external_score", "v3_core", "v3_extended")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _model(n_estimators: int) -> LGBMClassifier:
    """Identical hyperparameters to the frozen V1 history challenger."""

    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def _train_variant(
    variant: str,
    engineered: pd.DataFrame,
    target: pd.Series,
    development_mask: pd.Series,
    calibration_mask: pd.Series,
    selection,
) -> tuple[np.ndarray, dict]:
    features = restrict_selection(
        variant, selection.selected, selection.numeric, selection.categorical
    )
    _log(f"{variant}: {len(features.selected)} features ({len(features.dropped)} dropped)")

    frame = engineered
    if "APPLICATION_MISSING_COUNT" in features.selected:
        counted = tuple(f for f in features.selected if f != "APPLICATION_MISSING_COUNT")
        frame = engineered.copy(deep=False)
        frame["APPLICATION_MISSING_COUNT"] = recompute_missing_count(engineered, counted)

    numeric = list(features.numeric)
    categorical = list(features.categorical)
    columns = list(features.selected)

    development_indices = np.flatnonzero(development_mask.to_numpy())
    fit_indices, early_stop_indices = train_test_split(
        development_indices,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=target.iloc[development_indices],
    )

    tuning_preprocessor = _preprocessor(numeric, categorical)
    tuning_train = tuning_preprocessor.fit_transform(frame.loc[fit_indices, columns])
    tuning_validation = tuning_preprocessor.transform(frame.loc[early_stop_indices, columns])
    tuning_model = _model(n_estimators=2500)
    tuning_model.fit(
        tuning_train,
        target.loc[fit_indices].to_numpy(),
        eval_X=tuning_validation,
        eval_y=target.loc[early_stop_indices].to_numpy(),
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=125, verbose=False)],
    )
    best_iteration = int(tuning_model.best_iteration_ or tuning_model.n_estimators)
    del tuning_train, tuning_validation, tuning_preprocessor, tuning_model
    gc.collect()

    final_preprocessor = _preprocessor(numeric, categorical)
    development_matrix = final_preprocessor.fit_transform(frame.loc[development_mask, columns])
    calibration_matrix = final_preprocessor.transform(frame.loc[calibration_mask, columns])
    final_model = _model(n_estimators=best_iteration)
    final_model.fit(development_matrix, target.loc[development_mask].to_numpy())
    probability = final_model.predict_proba(calibration_matrix)[:, 1]

    importance = pd.Series(
        final_model.feature_importances_[: len(columns)], index=columns
    ).sort_values(ascending=False)

    record = {
        "variant": variant,
        "description": VARIANT_DESCRIPTIONS[variant],
        "feature_count": len(features.selected),
        "dropped_count": len(features.dropped),
        "transformed_columns": int(development_matrix.shape[1]),
        "best_iteration": best_iteration,
        "selected_features": list(features.selected),
        "dropped_features": list(features.dropped),
        "top_20_gain_features": {k: float(v) for k, v in importance.head(20).items()},
        "calibration_partition_metrics": evaluate_probabilities(
            target.loc[calibration_mask], probability
        ),
    }
    del development_matrix, calibration_matrix, final_preprocessor, final_model
    gc.collect()
    return probability, record


def _paired_bootstrap(
    y: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    metric,
) -> dict:
    """Class-stratified paired bootstrap over applicants."""

    rng = np.random.default_rng(RANDOM_SEED)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sample = np.concatenate(
            [
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            ]
        )
        deltas[iteration] = metric(y[sample], candidate[sample]) - metric(
            y[sample], baseline[sample]
        )
    return {
        "mean_delta": float(deltas.mean()),
        "ci_lower_2_5": float(np.percentile(deltas, 2.5)),
        "ci_upper_97_5": float(np.percentile(deltas, 97.5)),
        "probability_positive": float((deltas > 0).mean()),
        "iterations": BOOTSTRAP_ITERATIONS,
    }


def main() -> int:
    if not APPLICATION_PATH.exists() or not HISTORY_PATH.exists():
        _log(f"missing input: {APPLICATION_PATH} or {HISTORY_PATH}")
        return 1

    _log("loading application and history")
    application = pd.read_csv(APPLICATION_PATH, low_memory=False)
    history = pd.read_parquet(HISTORY_PATH)
    if "TARGET" in history.columns:
        raise ValueError("history features must not contain TARGET")

    frame = application.merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    applicant_ids = frame["SK_ID_CURR"].to_numpy()
    target = frame["TARGET"].astype("int8")
    splits = assign_splits(target, random_seed=RANDOM_SEED)
    del application, history
    gc.collect()

    _log("engineering features")
    engineered = engineer_application_features(frame)
    engineered = add_obligation_ratios(engineered)
    del frame
    gc.collect()

    development_mask = splits.eq("development")
    calibration_mask = splits.eq("calibration")
    holdout_mask = splits.eq("holdout")
    _log(
        f"rows development={int(development_mask.sum())} "
        f"calibration={int(calibration_mask.sum())} "
        f"holdout={int(holdout_mask.sum())} (holdout untouched)"
    )

    selection = select_responsible_features(engineered.loc[development_mask])
    _log(f"responsible selection: {len(selection.selected)} features")

    calibration_target = target.loc[calibration_mask].to_numpy(dtype=int)
    calibration_ids = applicant_ids[calibration_mask.to_numpy()]
    holdout_ids = set(applicant_ids[holdout_mask.to_numpy()].tolist())
    if holdout_ids & set(calibration_ids.tolist()):
        raise ValueError("calibration and holdout applicant IDs overlap")

    probabilities: dict[str, np.ndarray] = {}
    records: list[dict] = []
    for variant in VARIANTS:
        started = time.time()
        probability, record = _train_variant(
            variant, engineered, target, development_mask, calibration_mask, selection
        )
        record["train_seconds"] = round(time.time() - started, 1)
        probabilities[variant] = probability
        records.append(record)
        metrics = record["calibration_partition_metrics"]
        _log(
            f"{variant}: ROC-AUC {metrics['roc_auc']:.4f} "
            f"AP {metrics['average_precision']:.4f} "
            f"Brier {metrics['brier_score']:.5f} "
            f"({record['train_seconds']}s)"
        )

    from sklearn.metrics import average_precision_score, roc_auc_score

    baseline_probability = probabilities[BASELINE_VARIANT]
    for record in records:
        variant = record["variant"]
        if variant == BASELINE_VARIANT:
            record["bootstrap_vs_baseline"] = None
            continue
        _log(f"bootstrapping {variant} vs {BASELINE_VARIANT}")
        record["bootstrap_vs_baseline"] = {
            "baseline": BASELINE_VARIANT,
            "average_precision": _paired_bootstrap(
                calibration_target,
                probabilities[variant],
                baseline_probability,
                average_precision_score,
            ),
            "roc_auc": _paired_bootstrap(
                calibration_target,
                probabilities[variant],
                baseline_probability,
                roc_auc_score,
            ),
        }

    report = {
        "run_version": "1.0.0",
        "policy_version": POLICY_VERSION,
        "random_seed": RANDOM_SEED,
        "baseline_variant": BASELINE_VARIANT,
        "partition": "calibration",
        "holdout_evaluated": False,
        "holdout_note": (
            "The V1 final holdout is spent. These results select nothing beyond further "
            "research and cannot be reported as unbiased final performance."
        ),
        "split_rows": {
            "development": int(development_mask.sum()),
            "calibration": int(calibration_mask.sum()),
            "holdout": int(holdout_mask.sum()),
        },
        "variants": records,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    comparison = pd.DataFrame(
        [
            {
                "variant": record["variant"],
                "features": record["feature_count"],
                "roc_auc": record["calibration_partition_metrics"]["roc_auc"],
                "average_precision": record["calibration_partition_metrics"][
                    "average_precision"
                ],
                "brier_score": record["calibration_partition_metrics"]["brier_score"],
                "log_loss": record["calibration_partition_metrics"]["log_loss"],
                "top_10_capture": record["calibration_partition_metrics"]["top_10_percent"][
                    "event_capture_rate"
                ],
                "top_10_lift": record["calibration_partition_metrics"]["top_10_percent"]["lift"],
                "top_20_capture": record["calibration_partition_metrics"]["top_20_percent"][
                    "event_capture_rate"
                ],
                "best_iteration": record["best_iteration"],
                "train_seconds": record["train_seconds"],
            }
            for record in records
        ]
    )
    comparison.to_csv(COMPARISON_OUTPUT, index=False)

    predictions = pd.DataFrame({"SK_ID_CURR": calibration_ids, "TARGET": calibration_target})
    for variant, probability in probabilities.items():
        predictions[f"probability_{variant}"] = probability
    PREDICTIONS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTIONS_OUTPUT, index=False)

    _log(f"wrote {METRICS_OUTPUT}")
    _log(f"wrote {COMPARISON_OUTPUT}")
    _log(f"wrote {PREDICTIONS_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
