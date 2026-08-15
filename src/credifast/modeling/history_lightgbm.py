"""Train a history-enriched LightGBM challenger while preserving the holdout."""

from __future__ import annotations

import gc
import json
import platform
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from .application_baseline import assign_splits, evaluate_probabilities
from .application_features import engineer_application_features, select_responsible_features
from .application_lightgbm import _preprocessor, _sha256


def _history_model(n_estimators: int, random_seed: int) -> LGBMClassifier:
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
        random_state=random_seed,
        n_jobs=-1,
        verbosity=-1,
    )


def train_history_lightgbm(
    application_path: str | Path,
    history_path: str | Path,
    *,
    model_output: str | Path,
    metrics_output: str | Path,
    features_output: str | Path,
    random_seed: int = 42,
) -> dict:
    """Fit on development and report only calibration metrics."""

    application_source = Path(application_path)
    history_source = Path(history_path)
    application = pd.read_csv(application_source, low_memory=False)
    history = pd.read_parquet(history_source)
    if "TARGET" not in application or "SK_ID_CURR" not in application:
        raise ValueError("history LightGBM requires TARGET and SK_ID_CURR")
    if application["SK_ID_CURR"].duplicated().any():
        raise ValueError("application source requires unique SK_ID_CURR values")
    if history["SK_ID_CURR"].duplicated().any():
        raise ValueError("history source requires unique SK_ID_CURR values")
    if "TARGET" in history:
        raise ValueError("history features must not contain TARGET")
    if not application["TARGET"].isin([0, 1]).all():
        raise ValueError("history LightGBM requires binary TARGET values")

    frame = application.merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    history_columns = [column for column in history.columns if column != "SK_ID_CURR"]
    matched = frame[history_columns].notna().any(axis=1)
    target = frame["TARGET"].astype("int8")
    splits = assign_splits(target, random_seed=random_seed)
    engineered = engineer_application_features(frame)
    development_mask = splits.eq("development")
    calibration_mask = splits.eq("calibration")
    selection = select_responsible_features(engineered.loc[development_mask])
    selected = list(selection.selected)
    numeric = list(selection.numeric)
    categorical = list(selection.categorical)

    development_indices = np.flatnonzero(development_mask.to_numpy())
    fit_indices, early_stop_indices = train_test_split(
        development_indices,
        test_size=0.15,
        random_state=random_seed,
        stratify=target.iloc[development_indices],
    )
    tuning_preprocessor = _preprocessor(numeric, categorical)
    tuning_train = tuning_preprocessor.fit_transform(engineered.loc[fit_indices, selected])
    tuning_validation = tuning_preprocessor.transform(engineered.loc[early_stop_indices, selected])
    tuning_model = _history_model(n_estimators=2500, random_seed=random_seed)
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
    development_matrix = final_preprocessor.fit_transform(
        engineered.loc[development_mask, selected]
    )
    calibration_matrix = final_preprocessor.transform(engineered.loc[calibration_mask, selected])
    final_model = _history_model(n_estimators=best_iteration, random_seed=random_seed)
    final_model.fit(development_matrix, target.loc[development_mask].to_numpy())
    calibration_probability = final_model.predict_proba(calibration_matrix)[:, 1]

    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": final_preprocessor,
            "model": final_model,
            "feature_selection": selection.to_dict(),
            "history_features": history_columns,
            "model_name": "history-enriched-lightgbm-challenger",
            "model_version": "0.2.0",
            "random_seed": random_seed,
            "best_iteration": best_iteration,
        },
        model_path,
    )

    metrics = {
        "run_version": "1.0.0",
        "model_name": "history-enriched-lightgbm-challenger",
        "model_version": "0.2.0",
        "application_source": application_source.name,
        "history_source": history_source.name,
        "random_seed": random_seed,
        "best_iteration": best_iteration,
        "join_summary": {
            "application_rows": len(application),
            "history_rows": len(history),
            "application_rows_matched": int(matched.sum()),
            "application_match_rate": float(matched.mean()),
        },
        "feature_summary": {
            "history_features_available": len(history_columns),
            "selected": len(selection.selected),
            "numeric": len(selection.numeric),
            "categorical": len(selection.categorical),
            "transformed": int(development_matrix.shape[1]),
        },
        "calibration_partition_metrics": evaluate_probabilities(
            target.loc[calibration_mask], calibration_probability
        ),
        "holdout_evaluated": False,
        "holdout_note": (
            "The final holdout remains unscored until champion selection and calibration are frozen."
        ),
        "artifacts": {
            "model": {"path": str(model_path), "sha256": _sha256(model_path)},
            "history_features": {
                "path": str(history_source),
                "sha256": _sha256(history_source),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    metrics_path = Path(metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    features_path = Path(features_output)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.write_text(json.dumps(selection.to_dict(), indent=2) + "\n", encoding="utf-8")
    return metrics
