"""Inference adapter for the frozen history LightGBM research champion."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credifast.scoring import probability_to_score, risk_grade

from .application_features import engineer_application_features
from .confidence import HistoryConfidencePolicy, assess_history_confidence
from .reasons import aggregate_reason_contributions


def merge_history_with_gaps(
    application: pd.DataFrame,
    history: pd.DataFrame,
    expected_history_features: list[str],
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Left-join history and preserve the distinction between missing rows and zeros."""

    if "SK_ID_CURR" not in application or "SK_ID_CURR" not in history:
        raise ValueError("application and history inputs require SK_ID_CURR")
    if application["SK_ID_CURR"].duplicated().any():
        raise ValueError("application input must have unique SK_ID_CURR")
    if history["SK_ID_CURR"].duplicated().any() or "TARGET" in history:
        raise ValueError("history input must have unique keys and no TARGET")
    missing_columns = sorted(set(expected_history_features) - set(history.columns))
    if missing_columns:
        raise ValueError(
            f"history feature-store schema is missing {len(missing_columns)} model features"
        )
    history_flags = [column for column in expected_history_features if column.startswith("HAS_")]
    if not history_flags:
        raise ValueError("history model bundle must contain HAS_* availability flags")
    application_for_join = application.copy()
    history_for_join = history.loc[:, ["SK_ID_CURR", *expected_history_features]].copy()
    frame = application_for_join.merge(
        history_for_join,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
        indicator="_history_store_join",
    )
    missing_row = frame["_history_store_join"].eq("left_only")
    frame = frame.drop(columns="_history_store_join")
    frame[history_flags] = frame[history_flags].fillna(0).astype("int8")
    return frame, history_flags, missing_row


def _history_segment(source_count: int) -> str:
    if source_count <= 1:
        return "thin_0_to_1_sources"
    if source_count <= 3:
        return "partial_2_to_3_sources"
    return "full_4_to_5_sources"


class HistoryModelScorer:
    """Load the frozen model/calibrator pair and score application-grain rows."""

    def __init__(
        self,
        model_path: str | Path,
        calibrator_path: str | Path,
        *,
        confidence_policy: HistoryConfidencePolicy | None = None,
        application_model_path: str | Path | None = None,
        gap_strategy_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.calibrator_path = Path(calibrator_path)
        self.model_bundle = joblib.load(self.model_path)
        self.calibrator_bundle = joblib.load(self.calibrator_path)
        required_model_keys = {"preprocessor", "model", "feature_selection", "model_name"}
        if not required_model_keys.issubset(self.model_bundle):
            raise ValueError("history model bundle is incomplete")
        required_calibrator_keys = {"calibrator", "method", "base_model_name"}
        if not required_calibrator_keys.issubset(self.calibrator_bundle):
            raise ValueError("history calibrator bundle is incomplete")
        if self.calibrator_bundle["base_model_name"] != self.model_bundle["model_name"]:
            raise ValueError("model and calibrator bundle names do not match")
        self.confidence_policy = confidence_policy or HistoryConfidencePolicy()
        if (application_model_path is None) != (gap_strategy_path is None):
            raise ValueError("application model and gap strategy must be provided together")
        self.application_model_bundle = None
        self.gap_strategy = None
        if application_model_path is not None and gap_strategy_path is not None:
            self.application_model_bundle = joblib.load(Path(application_model_path))
            required_application_keys = {
                "preprocessor",
                "model",
                "feature_selection",
                "model_name",
            }
            if not required_application_keys.issubset(self.application_model_bundle):
                raise ValueError("application model bundle is incomplete")
            self.gap_strategy = json.loads(Path(gap_strategy_path).read_text(encoding="utf-8"))
            required_segments = {
                "thin_0_to_1_sources",
                "partial_2_to_3_sources",
                "full_4_to_5_sources",
            }
            weights = self.gap_strategy.get("history_weights", {})
            if set(weights) != required_segments:
                raise ValueError("gap strategy does not define all history segments")
            if self.gap_strategy.get("component_probability_space") != "raw_model_probability":
                raise ValueError("unsupported gap-strategy probability space")

    def score(
        self,
        application: pd.DataFrame,
        history: pd.DataFrame,
        *,
        explain: bool = True,
    ) -> tuple[list[dict], dict]:
        """Score application rows with one-to-one target-free history features."""

        started = time.perf_counter()
        expected_history = self.model_bundle.get("history_features")
        if not expected_history:
            raise ValueError("history model bundle does not declare its feature-store schema")
        frame, history_flags, missing_history_row = merge_history_with_gaps(
            application,
            history,
            list(expected_history),
        )

        engineered = engineer_application_features(frame)
        selected = self.model_bundle["feature_selection"]["selected"]
        missing_features = sorted(set(selected) - set(engineered.columns))
        if missing_features:
            raise ValueError(f"scoring input is missing {len(missing_features)} model features")
        matrix = self.model_bundle["preprocessor"].transform(engineered[selected])
        raw_probability = self.model_bundle["model"].predict_proba(matrix)[:, 1]
        calibrated_probability = self.calibrator_bundle["calibrator"].predict(raw_probability)
        application_probability: np.ndarray | None = None
        history_weights = np.ones(len(frame), dtype=float)
        final_probability = calibrated_probability
        if self.application_model_bundle is not None and self.gap_strategy is not None:
            application_selected = self.application_model_bundle["feature_selection"]["selected"]
            missing_application = sorted(set(application_selected) - set(engineered.columns))
            if missing_application:
                raise ValueError(
                    f"application scoring input is missing {len(missing_application)} model features"
                )
            application_matrix = self.application_model_bundle["preprocessor"].transform(
                engineered[application_selected]
            )
            application_probability = self.application_model_bundle["model"].predict_proba(
                application_matrix
            )[:, 1]
            source_counts = frame[history_flags].sum(axis=1).astype(int)
            segments = source_counts.map(_history_segment)
            history_weights = segments.map(self.gap_strategy["history_weights"]).to_numpy(
                dtype=float
            )
            final_probability = (
                history_weights * raw_probability
                + (1.0 - history_weights) * application_probability
            )
        probability_elapsed = time.perf_counter() - started

        reason_rows: list[dict | None] = [None] * len(frame)
        application_reason_rows: list[dict | None] = [None] * len(frame)
        explanation_elapsed = 0.0
        if explain:
            explanation_started = time.perf_counter()
            contributions = self.model_bundle["model"].booster_.predict(
                matrix, pred_contrib=True
            )
            transformed_names = self.model_bundle["preprocessor"].get_feature_names_out()
            if contributions.shape[1] != len(transformed_names) + 1:
                raise ValueError("TreeSHAP output does not match transformed feature schema")
            reason_rows = [
                aggregate_reason_contributions(transformed_names, row[:-1])
                for row in contributions
            ]
            if self.application_model_bundle is not None:
                application_contributions = self.application_model_bundle["model"].booster_.predict(
                    application_matrix, pred_contrib=True
                )
                application_names = self.application_model_bundle[
                    "preprocessor"
                ].get_feature_names_out()
                if application_contributions.shape[1] != len(application_names) + 1:
                    raise ValueError("application TreeSHAP output does not match feature schema")
                application_reason_rows = [
                    aggregate_reason_contributions(application_names, row[:-1])
                    for row in application_contributions
                ]
            explanation_elapsed = time.perf_counter() - explanation_started

        records = []
        for position, (_, row) in enumerate(frame.iterrows()):
            probability = float(final_probability[position])
            flags = {column: row[column] for column in history_flags}
            confidence = assess_history_confidence(
                probability,
                flags,
                policy=self.confidence_policy,
            )
            quality_flags = []
            confidence_reasons = list(confidence.reasons)
            if bool(missing_history_row.iloc[position]):
                quality_flags.append("FEATURE_STORE_ROW_MISSING")
                confidence_reasons.insert(
                    0,
                    "No feature-store row was found; history values remain missing and flags are zero",
                )
            dominant_component = (
                "history" if history_weights[position] >= 0.5 else "application"
            )
            dominant_reasons = (
                reason_rows[position]
                if dominant_component == "history"
                else application_reason_rows[position]
            )
            record = {
                    "application_id": int(row["SK_ID_CURR"]),
                    "repayment_difficulty_probability": probability,
                    "raw_model_probability": float(raw_probability[position]),
                    "history_component_probability": float(raw_probability[position]),
                    "application_component_probability": (
                        None
                        if application_probability is None
                        else float(application_probability[position])
                    ),
                    "history_weight": float(history_weights[position]),
                    "internal_credit_score": probability_to_score(probability),
                    "risk_grade": risk_grade(probability),
                    "history_segment": confidence.history_segment,
                    "available_history_sources": confidence.available_sources,
                    "confidence": confidence.confidence,
                    "review_route": confidence.route,
                    "confidence_reasons": confidence_reasons,
                    "feature_store_row_missing": bool(missing_history_row.iloc[position]),
                    "data_quality_flags": quality_flags,
                    "reasons": dominant_reasons,
                    "explanation_scope": (
                        "single_history_model"
                        if self.application_model_bundle is None
                        else "dominant_component_not_full_blend"
                    ),
                    "dominant_explanation_component": dominant_component,
                    "component_reasons": (
                        None
                        if self.application_model_bundle is None
                        else {
                            "history": reason_rows[position],
                            "application": application_reason_rows[position],
                        }
                    ),
                    "versions": {
                        "model_name": self.model_bundle["model_name"],
                        "model_version": self.model_bundle.get("model_version"),
                        "calibration_method": self.calibrator_bundle["method"],
                        "calibrator_version": self.calibrator_bundle.get("calibrator_version"),
                        "confidence_policy": confidence.policy_version,
                        "gap_strategy": (
                            None
                            if self.gap_strategy is None
                            else self.gap_strategy.get("policy_version")
                        ),
                    },
                    "disclaimer": (
                        "Research-only competition model output; not approved for autonomous "
                        "lending, pricing, or adverse action."
                    ),
                }
            records.append(record)
        total_elapsed = time.perf_counter() - started
        timings = {
            "rows": len(records),
            "probability_scoring_seconds": probability_elapsed,
            "explanation_seconds": explanation_elapsed,
            "total_seconds": total_elapsed,
            "probability_milliseconds_per_row": 1000.0 * probability_elapsed / len(records),
            "total_milliseconds_per_row": 1000.0 * total_elapsed / len(records),
            "explanations_enabled": explain,
        }
        return records, timings


def score_history_files(
    application_path: str | Path,
    history_path: str | Path,
    model_path: str | Path,
    calibrator_path: str | Path,
    *,
    application_model_path: str | Path | None = None,
    gap_strategy_path: str | Path | None = None,
    application_id: int | None = None,
    limit: int | None = None,
    explain: bool = True,
) -> tuple[list[dict], dict]:
    application = pd.read_csv(application_path, low_memory=False)
    if application_id is not None:
        application = application.loc[application["SK_ID_CURR"].eq(application_id)]
        if application.empty:
            raise ValueError(f"application ID {application_id} was not found")
    application = application.sort_values("SK_ID_CURR")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        application = application.head(limit)
    history = pd.read_parquet(history_path)
    requested_ids = set(application["SK_ID_CURR"])
    history = history.loc[history["SK_ID_CURR"].isin(requested_ids)]
    scorer = HistoryModelScorer(
        model_path,
        calibrator_path,
        application_model_path=application_model_path,
        gap_strategy_path=gap_strategy_path,
    )
    return scorer.score(application, history, explain=explain)
