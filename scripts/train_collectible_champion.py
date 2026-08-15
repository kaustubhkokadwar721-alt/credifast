"""Train, persist and calibrate the collectible-input candidate.

Produces a servable model bundle plus an honest calibrator selection, using the same
protocol as the frozen V1 champion: fit on development, split the calibration partition
50/50, fit each calibrator on one half and choose on the disjoint half.

The V1 final holdout is spent. This script never reads holdout rows. The resulting bundle
is a research candidate pending a new untouched evaluation set; it is not validated.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\train_collectible_champion.py --variant v3_extended
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credifast.modeling.application_baseline import assign_splits, evaluate_probabilities
from credifast.modeling.application_features import (
    engineer_application_features,
    select_responsible_features,
)
from credifast.modeling.application_lightgbm import _preprocessor, _sha256
from credifast.modeling.calibration import compare_calibrators
from credifast.modeling.collectible_features import (
    POLICY_VERSION,
    VARIANT_DESCRIPTIONS,
    add_obligation_ratios,
    recompute_missing_count,
    restrict_selection,
)

APPLICATION_PATH = ROOT / "data" / "raw" / "application_train.csv"
HISTORY_PATH = ROOT / "data" / "processed" / "history_features.parquet"
MODEL_OUTPUT = ROOT / "artifacts" / "models" / "collectible_lightgbm.joblib"
CALIBRATOR_OUTPUT = ROOT / "artifacts" / "models" / "collectible_calibrator.joblib"
METRICS_OUTPUT = ROOT / "artifacts" / "collectible_champion_metrics.json"

RANDOM_SEED = 42
MODEL_VERSION = "0.3.0"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _model(n_estimators: int) -> LGBMClassifier:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="v3_extended", choices=["v3_core", "v3_extended"])
    arguments = parser.parse_args()
    variant = arguments.variant

    if not APPLICATION_PATH.exists() or not HISTORY_PATH.exists():
        _log("missing local application or history feature store")
        return 1

    _log("loading sources")
    application = pd.read_csv(APPLICATION_PATH, low_memory=False)
    history = pd.read_parquet(HISTORY_PATH)
    if "TARGET" in history.columns:
        raise ValueError("history features must not contain TARGET")
    frame = application.merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    target = frame["TARGET"].astype("int8")
    splits = assign_splits(target, random_seed=RANDOM_SEED)
    del application, history
    gc.collect()

    engineered = add_obligation_ratios(engineer_application_features(frame))
    del frame
    gc.collect()

    development_mask = splits.eq("development")
    calibration_mask = splits.eq("calibration")
    selection = select_responsible_features(engineered.loc[development_mask])
    features = restrict_selection(
        variant, selection.selected, selection.numeric, selection.categorical
    )
    _log(f"{variant}: {len(features.selected)} features")

    counted = tuple(f for f in features.selected if f != "APPLICATION_MISSING_COUNT")
    engineered["APPLICATION_MISSING_COUNT"] = recompute_missing_count(engineered, counted)

    columns = list(features.selected)
    numeric = list(features.numeric)
    categorical = list(features.categorical)

    development_indices = np.flatnonzero(development_mask.to_numpy())
    fit_indices, early_stop_indices = train_test_split(
        development_indices,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=target.iloc[development_indices],
    )
    tuning_preprocessor = _preprocessor(numeric, categorical)
    tuning_train = tuning_preprocessor.fit_transform(engineered.loc[fit_indices, columns])
    tuning_validation = tuning_preprocessor.transform(
        engineered.loc[early_stop_indices, columns]
    )
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
    _log(f"early stopping chose {best_iteration} trees")

    final_preprocessor = _preprocessor(numeric, categorical)
    development_matrix = final_preprocessor.fit_transform(
        engineered.loc[development_mask, columns]
    )
    calibration_matrix = final_preprocessor.transform(engineered.loc[calibration_mask, columns])
    final_model = _model(n_estimators=best_iteration)
    final_model.fit(development_matrix, target.loc[development_mask].to_numpy())
    raw_probability = final_model.predict_proba(calibration_matrix)[:, 1]
    calibration_target = target.loc[calibration_mask].to_numpy(dtype=int)
    _log("model fitted; selecting calibrator")

    local_positions = np.arange(len(calibration_target))
    calibrator_fit, calibrator_evaluation = train_test_split(
        local_positions,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=calibration_target,
    )
    selected_method, candidate_results, fitted_candidates = compare_calibrators(
        raw_probability,
        calibration_target,
        fit_indices=calibrator_fit,
        evaluation_indices=calibrator_evaluation,
        random_seed=RANDOM_SEED,
    )
    _log(f"calibrator selected: {selected_method}")
    calibrator = fitted_candidates[selected_method]
    calibrated_probability = calibrator.predict(raw_probability)

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": final_preprocessor,
            "model": final_model,
            "feature_selection": features.to_dict(),
            "selected_features": columns,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "model_name": f"collectible-input-lightgbm-{variant}",
            "model_version": MODEL_VERSION,
            "policy_version": POLICY_VERSION,
            "random_seed": RANDOM_SEED,
            "best_iteration": best_iteration,
            "validated": False,
            "status": "research candidate pending new untouched evaluation data",
        },
        MODEL_OUTPUT,
    )
    joblib.dump(
        {
            "calibrator": calibrator,
            "method": selected_method,
            "model_version": MODEL_VERSION,
            "random_seed": RANDOM_SEED,
        },
        CALIBRATOR_OUTPUT,
    )

    metrics = {
        "run_version": "1.0.0",
        "variant": variant,
        "description": VARIANT_DESCRIPTIONS[variant],
        "policy_version": POLICY_VERSION,
        "model_version": MODEL_VERSION,
        "random_seed": RANDOM_SEED,
        "best_iteration": best_iteration,
        "feature_count": len(columns),
        "selected_features": columns,
        "holdout_evaluated": False,
        "validated": False,
        "status_note": (
            "Selected on the calibration partition only. The V1 final holdout is spent, so this "
            "candidate has no unbiased final evidence and must not be described as validated."
        ),
        "calibrator": {
            "selected_method": selected_method,
            "fit_rows": len(calibrator_fit),
            "evaluation_rows": len(calibrator_evaluation),
            "candidates": candidate_results,
        },
        "calibration_partition_metrics_raw": evaluate_probabilities(
            pd.Series(calibration_target), raw_probability
        ),
        "calibration_partition_metrics_calibrated": evaluate_probabilities(
            pd.Series(calibration_target), calibrated_probability
        ),
        "artifacts": {
            "model": {"path": str(MODEL_OUTPUT), "sha256": _sha256(MODEL_OUTPUT)},
            "calibrator": {"path": str(CALIBRATOR_OUTPUT), "sha256": _sha256(CALIBRATOR_OUTPUT)},
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    raw = metrics["calibration_partition_metrics_raw"]
    done = metrics["calibration_partition_metrics_calibrated"]
    _log(f"raw        ROC-AUC {raw['roc_auc']:.4f} AP {raw['average_precision']:.4f} "
         f"Brier {raw['brier_score']:.5f}")
    _log(f"calibrated ROC-AUC {done['roc_auc']:.4f} AP {done['average_precision']:.4f} "
         f"Brier {done['brier_score']:.5f}")
    _log(f"wrote {MODEL_OUTPUT}")
    _log(f"wrote {CALIBRATOR_OUTPUT}")
    _log(f"wrote {METRICS_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
