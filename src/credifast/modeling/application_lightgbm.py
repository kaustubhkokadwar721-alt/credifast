"""Train the application-only LightGBM challenger without touching the holdout."""

from __future__ import annotations

import gc
import hashlib
import json
import platform
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .application_baseline import assign_splits, evaluate_probabilities
from .application_features import engineer_application_features, select_responsible_features


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _model(n_estimators: int, random_seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def train_application_lightgbm(
    input_path: str | Path,
    *,
    model_output: str | Path,
    metrics_output: str | Path,
    features_output: str | Path,
    random_seed: int = 42,
) -> dict:
    source = Path(input_path)
    frame = pd.read_csv(source, low_memory=False)
    if "TARGET" not in frame or "SK_ID_CURR" not in frame:
        raise ValueError("application LightGBM requires TARGET and SK_ID_CURR")
    if frame["SK_ID_CURR"].duplicated().any():
        raise ValueError("application LightGBM requires unique SK_ID_CURR values")
    if not frame["TARGET"].isin([0, 1]).all():
        raise ValueError("application LightGBM requires binary TARGET values")

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
    tuning_model = _model(n_estimators=2000, random_seed=random_seed)
    tuning_model.fit(
        tuning_train,
        target.loc[fit_indices].to_numpy(),
        eval_X=tuning_validation,
        eval_y=target.loc[early_stop_indices].to_numpy(),
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    best_iteration = int(tuning_model.best_iteration_ or tuning_model.n_estimators)
    del tuning_train, tuning_validation, tuning_preprocessor, tuning_model
    gc.collect()

    final_preprocessor = _preprocessor(numeric, categorical)
    development_matrix = final_preprocessor.fit_transform(
        engineered.loc[development_mask, selected]
    )
    calibration_matrix = final_preprocessor.transform(engineered.loc[calibration_mask, selected])
    final_model = _model(n_estimators=best_iteration, random_seed=random_seed)
    final_model.fit(development_matrix, target.loc[development_mask].to_numpy())
    calibration_probability = final_model.predict_proba(calibration_matrix)[:, 1]

    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": final_preprocessor,
            "model": final_model,
            "feature_selection": selection.to_dict(),
            "model_name": "application-lightgbm-challenger",
            "model_version": "0.1.0",
            "random_seed": random_seed,
            "best_iteration": best_iteration,
        },
        model_path,
    )

    metrics = {
        "run_version": "1.0.0",
        "model_name": "application-lightgbm-challenger",
        "model_version": "0.1.0",
        "source_file": source.name,
        "random_seed": random_seed,
        "best_iteration": best_iteration,
        "feature_summary": {
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
        "artifact": {"path": str(model_path), "sha256": _sha256(model_path)},
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
