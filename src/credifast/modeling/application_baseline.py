"""Train the first responsible application-only logistic baseline."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .application_features import engineer_application_features, select_responsible_features


def assign_splits(
    target: pd.Series,
    *,
    random_seed: int = 42,
) -> pd.Series:
    indices = np.arange(len(target))
    development, remainder = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_seed,
        stratify=target,
    )
    calibration, holdout = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_seed,
        stratify=target.iloc[remainder],
    )
    assignments = np.empty(len(target), dtype=object)
    assignments[development] = "development"
    assignments[calibration] = "calibration"
    assignments[holdout] = "holdout"
    return pd.Series(assignments, index=target.index, name="split")


def _ranking_metrics(target: np.ndarray, probability: np.ndarray, share: float) -> dict:
    threshold = float(np.quantile(probability, 1.0 - share))
    selected = probability >= threshold
    selected_target = target[selected]
    event_rate = float(target.mean())
    return {
        "population_share": float(selected.mean()),
        "probability_threshold": threshold,
        "selected_event_rate": float(selected_target.mean()),
        "lift": float(selected_target.mean() / event_rate),
        "event_capture_rate": float(selected_target.sum() / target.sum()),
    }


def evaluate_probabilities(target: pd.Series, probability: np.ndarray) -> dict:
    y = target.to_numpy(dtype=int)
    base_rate = float(y.mean())
    constant_probability = np.full_like(probability, base_rate, dtype=float)
    binary_prediction = probability >= 0.50
    bins = pd.DataFrame({"target": y, "probability": probability})
    bins["bin"] = pd.qcut(bins["probability"], q=10, duplicates="drop")
    calibration = [
        {
            "count": len(group),
            "mean_probability": float(group["probability"].mean()),
            "observed_rate": float(group["target"].mean()),
        }
        for _, group in bins.groupby("bin", observed=True)
    ]
    brier = float(brier_score_loss(y, probability))
    base_brier = float(brier_score_loss(y, constant_probability))
    return {
        "rows": len(y),
        "base_rate": base_rate,
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": brier,
        "base_rate_brier_score": base_brier,
        "brier_skill_score": float(1.0 - brier / base_brier),
        "log_loss": float(log_loss(y, probability)),
        "precision_at_0_5": float(precision_score(y, binary_prediction, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, binary_prediction, zero_division=0)),
        "top_10_percent": _ranking_metrics(y, probability, 0.10),
        "top_20_percent": _ranking_metrics(y, probability, 0.20),
        "calibration_deciles": calibration,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def train_application_baseline(
    input_path: str | Path,
    *,
    model_output: str | Path,
    metrics_output: str | Path,
    features_output: str | Path,
    splits_output: str | Path,
    random_seed: int = 42,
) -> dict:
    source = Path(input_path)
    frame = pd.read_csv(source, low_memory=False)
    if "TARGET" not in frame or "SK_ID_CURR" not in frame:
        raise ValueError("application baseline requires TARGET and SK_ID_CURR")
    if frame["SK_ID_CURR"].duplicated().any():
        raise ValueError("application baseline requires unique SK_ID_CURR values")
    if not frame["TARGET"].isin([0, 1]).all():
        raise ValueError("application baseline requires binary TARGET values")

    target = frame["TARGET"].astype("int8")
    splits = assign_splits(target, random_seed=random_seed)
    engineered = engineer_application_features(frame)
    development_mask = splits.eq("development")
    calibration_mask = splits.eq("calibration")
    selection = select_responsible_features(engineered.loc[development_mask])

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=100,
                    sparse_output=True,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(selection.numeric)),
            ("categorical", categorical_pipeline, list(selection.categorical)),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    max_iter=400,
                    solver="lbfgs",
                    random_state=random_seed,
                ),
            ),
        ]
    )
    pipeline.fit(
        engineered.loc[development_mask, list(selection.selected)],
        target.loc[development_mask],
    )
    calibration_probability = pipeline.predict_proba(
        engineered.loc[calibration_mask, list(selection.selected)]
    )[:, 1]

    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_selection": selection.to_dict(),
            "model_name": "application-logistic-baseline",
            "model_version": "0.1.0",
            "random_seed": random_seed,
        },
        model_path,
    )

    split_frame = pd.DataFrame(
        {"SK_ID_CURR": frame["SK_ID_CURR"], "TARGET": target, "split": splits}
    )
    split_path = Path(splits_output)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_frame.to_parquet(split_path, index=False)

    split_summary = {
        split_name: {
            "rows": int(mask.sum()),
            "target_count": int(target.loc[mask].sum()),
            "target_rate": float(target.loc[mask].mean()),
        }
        for split_name in ("development", "calibration", "holdout")
        for mask in [splits.eq(split_name)]
    }
    metrics = {
        "run_version": "1.0.0",
        "model_name": "application-logistic-baseline",
        "model_version": "0.1.0",
        "source_file": source.name,
        "random_seed": random_seed,
        "split_summary": split_summary,
        "feature_summary": {
            "selected": len(selection.selected),
            "numeric": len(selection.numeric),
            "categorical": len(selection.categorical),
            "excluded_by_policy": len(selection.excluded_by_policy),
            "excluded_high_missingness": len(selection.excluded_high_missingness),
            "excluded_constant": len(selection.excluded_constant),
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
        },
    }

    metrics_path = Path(metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    features_path = Path(features_output)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.write_text(json.dumps(selection.to_dict(), indent=2) + "\n", encoding="utf-8")
    return metrics
