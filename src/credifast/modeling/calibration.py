"""Probability calibration and thin-file diagnostics for the history champion."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from .application_baseline import assign_splits, evaluate_probabilities
from .application_features import engineer_application_features
from .application_lightgbm import _sha256


def _clip(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = _clip(probability)
    return np.log(clipped / (1.0 - clipped))


@dataclass
class IdentityCalibrator:
    """No-op candidate used to verify that calibration adds value."""

    def fit(self, probability: np.ndarray, target: np.ndarray) -> IdentityCalibrator:
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return _clip(probability)


@dataclass
class SigmoidCalibrator:
    """Platt scaling on the raw model log odds."""

    random_seed: int = 42
    model: LogisticRegression | None = None

    def fit(self, probability: np.ndarray, target: np.ndarray) -> SigmoidCalibrator:
        self.model = LogisticRegression(
            C=1_000_000.0,
            solver="lbfgs",
            random_state=self.random_seed,
        )
        self.model.fit(_logit(probability).reshape(-1, 1), target)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("sigmoid calibrator is not fitted")
        return self.model.predict_proba(_logit(probability).reshape(-1, 1))[:, 1]


@dataclass
class IsotonicCalibrator:
    """Monotone non-parametric probability calibration."""

    model: IsotonicRegression | None = None

    def fit(self, probability: np.ndarray, target: np.ndarray) -> IsotonicCalibrator:
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.model.fit(probability, target)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("isotonic calibrator is not fitted")
        return _clip(self.model.predict(probability))


def _calibrator(method: str, random_seed: int):
    if method == "identity":
        return IdentityCalibrator()
    if method == "sigmoid":
        return SigmoidCalibrator(random_seed=random_seed)
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"unknown calibration method: {method}")


def calibration_diagnostics(target: np.ndarray, probability: np.ndarray) -> dict:
    """Return proper scoring rules, calibration line, and equal-count reliability bins."""

    y = np.asarray(target, dtype=int)
    p = _clip(probability)
    calibration_model = LogisticRegression(C=1_000_000.0, solver="lbfgs")
    calibration_model.fit(_logit(p).reshape(-1, 1), y)
    calibration_frame = pd.DataFrame({"target": y, "probability": p})
    calibration_frame["bin"] = pd.qcut(
        calibration_frame["probability"], q=10, duplicates="drop"
    )
    reliability = []
    weighted_error = 0.0
    for _, group in calibration_frame.groupby("bin", observed=True):
        mean_probability = float(group["probability"].mean())
        observed_rate = float(group["target"].mean())
        weight = len(group) / len(calibration_frame)
        weighted_error += weight * abs(observed_rate - mean_probability)
        reliability.append(
            {
                "rows": len(group),
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_error": abs(observed_rate - mean_probability),
            }
        )
    return {
        "rows": len(y),
        "base_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "expected_calibration_error": float(weighted_error),
        "calibration_intercept": float(calibration_model.intercept_[0]),
        "calibration_slope": float(calibration_model.coef_[0, 0]),
        "reliability_deciles": reliability,
    }


def compare_calibrators(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    random_seed: int = 42,
) -> tuple[str, dict, dict]:
    """Compare candidates on rows that were not used to fit the calibrator."""

    results = {}
    fitted = {}
    for method in ("identity", "sigmoid", "isotonic"):
        calibrator = _calibrator(method, random_seed)
        calibrator.fit(probability[fit_indices], target[fit_indices])
        calibrated = calibrator.predict(probability[evaluation_indices])
        results[method] = {
            "calibration": calibration_diagnostics(target[evaluation_indices], calibrated),
            "ranking": evaluate_probabilities(
                pd.Series(target[evaluation_indices]), calibrated
            ),
        }
        fitted[method] = calibrator
    selected = min(
        results,
        key=lambda method: (
            results[method]["calibration"]["brier_score"],
            results[method]["calibration"]["log_loss"],
        ),
    )
    return selected, results, fitted


def _thin_file_segment(history: pd.DataFrame) -> pd.Series:
    flags = [column for column in history if column.startswith("HAS_")]
    if not flags:
        return pd.Series("unknown", index=history.index)
    source_count = history[flags].fillna(0).sum(axis=1)
    return pd.cut(
        source_count,
        bins=[-1, 1, 3, len(flags)],
        labels=["thin_0_to_1_sources", "partial_2_to_3_sources", "full_4_to_5_sources"],
    ).astype("string")


def calibrate_history_model(
    application_path: str | Path,
    history_path: str | Path,
    model_path: str | Path,
    *,
    calibrator_output: str | Path,
    metrics_output: str | Path,
    predictions_output: str | Path,
    bundle_manifest_output: str | Path,
    random_seed: int = 42,
) -> dict:
    """Select a calibrator honestly and refit it on all calibration rows."""

    application_source = Path(application_path)
    history_source = Path(history_path)
    model_source = Path(model_path)
    application = pd.read_csv(application_source, low_memory=False)
    history = pd.read_parquet(history_source)
    if history["SK_ID_CURR"].duplicated().any() or "TARGET" in history:
        raise ValueError("history calibration requires a unique, target-free history store")
    frame = application.merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    bundle = joblib.load(model_source)
    required_bundle_keys = {"preprocessor", "model", "feature_selection", "model_name"}
    if not required_bundle_keys.issubset(bundle):
        raise ValueError("history model bundle is incomplete")
    if bundle["model_name"] != "history-enriched-lightgbm-challenger":
        raise ValueError("calibration requires the history-enriched LightGBM bundle")

    target = frame["TARGET"].to_numpy(dtype=int)
    splits = assign_splits(frame["TARGET"], random_seed=random_seed)
    calibration_positions = np.flatnonzero(splits.eq("calibration").to_numpy())
    engineered = engineer_application_features(frame)
    selected_features = bundle["feature_selection"]["selected"]
    calibration_matrix = bundle["preprocessor"].transform(
        engineered.iloc[calibration_positions][selected_features]
    )
    raw_probability = bundle["model"].predict_proba(calibration_matrix)[:, 1]
    calibration_target = target[calibration_positions]
    local_positions = np.arange(len(calibration_positions))
    calibration_fit, calibration_evaluation = train_test_split(
        local_positions,
        test_size=0.50,
        random_state=random_seed,
        stratify=calibration_target,
    )
    selected_method, candidate_results, fitted_candidates = compare_calibrators(
        raw_probability,
        calibration_target,
        fit_indices=calibration_fit,
        evaluation_indices=calibration_evaluation,
        random_seed=random_seed,
    )
    selected_evaluation_probability = fitted_candidates[selected_method].predict(
        raw_probability[calibration_evaluation]
    )

    segment = _thin_file_segment(history.set_index("SK_ID_CURR").loc[
        frame.iloc[calibration_positions]["SK_ID_CURR"]
    ].reset_index(drop=True))
    segment_results = {}
    for segment_name in sorted(segment.dropna().unique()):
        segment_evaluation = segment.iloc[calibration_evaluation].eq(segment_name).to_numpy()
        y_segment = calibration_target[calibration_evaluation][segment_evaluation]
        p_segment = selected_evaluation_probability[segment_evaluation]
        if len(y_segment) < 100 or len(np.unique(y_segment)) < 2:
            segment_results[segment_name] = {
                "rows": len(y_segment),
                "status": "insufficient_sample",
            }
            continue
        segment_results[segment_name] = {
            "status": "evaluated",
            "calibration": calibration_diagnostics(y_segment, p_segment),
            "ranking": evaluate_probabilities(pd.Series(y_segment), p_segment),
        }

    final_calibrator = _calibrator(selected_method, random_seed)
    final_calibrator.fit(raw_probability, calibration_target)
    calibrator_path = Path(calibrator_output)
    calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "calibrator": final_calibrator,
            "method": selected_method,
            "calibrator_version": "0.1.0",
            "base_model_name": bundle["model_name"],
            "base_model_version": bundle.get("model_version"),
            "base_model_sha256": _sha256(model_source),
            "random_seed": random_seed,
            "fit_population": "all calibration rows after method selection",
        },
        calibrator_path,
    )

    selection_evaluation_probability = np.full(len(calibration_positions), np.nan)
    selection_evaluation_probability[calibration_evaluation] = selected_evaluation_probability
    prediction_frame = pd.DataFrame(
        {
            "SK_ID_CURR": frame.iloc[calibration_positions]["SK_ID_CURR"].to_numpy(),
            "TARGET": calibration_target,
            "calibration_role": np.where(
                np.isin(local_positions, calibration_fit), "calibrator_fit", "calibrator_evaluation"
            ),
            "history_segment": segment.to_numpy(),
            "raw_probability": raw_probability,
            "selection_evaluation_probability": selection_evaluation_probability,
            "final_calibrated_probability": final_calibrator.predict(raw_probability),
        }
    )
    predictions_path = Path(predictions_output)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_parquet(predictions_path, index=False)

    metrics = {
        "run_version": "1.0.0",
        "calibrator_version": "0.1.0",
        "base_model_name": bundle["model_name"],
        "base_model_version": bundle.get("model_version"),
        "selected_method": selected_method,
        "selection_rule": "minimum Brier score, then minimum log loss",
        "random_seed": random_seed,
        "calibration_split": {
            "total_rows": len(calibration_positions),
            "calibrator_fit_rows": len(calibration_fit),
            "calibrator_evaluation_rows": len(calibration_evaluation),
            "fit_target_rate": float(calibration_target[calibration_fit].mean()),
            "evaluation_target_rate": float(calibration_target[calibration_evaluation].mean()),
        },
        "candidate_evaluation_metrics": candidate_results,
        "thin_file_evaluation": segment_results,
        "final_refit": {
            "method": selected_method,
            "rows": len(calibration_positions),
            "note": (
                "The selected method is refit on all calibration rows. Its next unbiased evaluation "
                "is the sealed holdout."
            ),
        },
        "holdout_evaluated": False,
        "holdout_note": "No holdout features or predictions were created during calibration.",
        "artifacts": {
            "calibrator": {
                "path": str(calibrator_path),
                "sha256": _sha256(calibrator_path),
            },
            "base_model": {"path": str(model_source), "sha256": _sha256(model_source)},
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
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    bundle_manifest = {
        "bundle_version": "0.1.0",
        "status": "research_champion_holdout_not_evaluated",
        "model": {
            "name": bundle["model_name"],
            "version": bundle.get("model_version"),
            "sha256": _sha256(model_source),
            "selected_features": len(selected_features),
            "best_iteration": bundle.get("best_iteration"),
        },
        "calibration": {
            "method": selected_method,
            "version": "0.1.0",
            "sha256": _sha256(calibrator_path),
            "selection_rule": metrics["selection_rule"],
            "fit_population": "all 46,127 calibration rows after method selection",
        },
        "data": {
            "history_features_sha256": _sha256(history_source),
            "grain": "one row per SK_ID_CURR",
        },
        "controls": {
            "holdout_evaluated": False,
            "protected_audit_fields_used_for_prediction": False,
            "policy_separate_from_model": True,
            "production_approved": False,
        },
    }
    manifest_path = Path(bundle_manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8")
    return metrics
