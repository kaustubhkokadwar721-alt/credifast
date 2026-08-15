"""Shared local runtime for the frozen CrediFast research candidate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .input_sources import build_input_source_schema
from .modeling.inference import HistoryModelScorer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERRIDE_COLUMNS = {
    "annual_income": "AMT_INCOME_TOTAL",
    "requested_credit": "AMT_CREDIT",
    "annual_annuity": "AMT_ANNUITY",
    "goods_price": "AMT_GOODS_PRICE",
}


@dataclass(frozen=True)
class RuntimePaths:
    application: Path = PROJECT_ROOT / "data/raw/application_test.csv"
    history: Path = PROJECT_ROOT / "data/processed/history_features.parquet"
    history_model: Path = PROJECT_ROOT / "artifacts/models/history_lightgbm.joblib"
    application_model: Path = PROJECT_ROOT / "artifacts/models/application_lightgbm.joblib"
    calibrator: Path = PROJECT_ROOT / "artifacts/models/history_calibrator.joblib"
    gap_policy: Path = PROJECT_ROOT / "configs/model_gap_strategy.json"
    demo_profiles: Path = PROJECT_ROOT / "configs/demo_profiles.json"
    final_metrics: Path = PROJECT_ROOT / "artifacts/final_holdout_evaluation.json"
    fairness_audit: Path = PROJECT_ROOT / "artifacts/history_fairness_audit.json"

    def required(self) -> dict[str, Path]:
        return {
            "application_data": self.application,
            "history_feature_store": self.history,
            "history_model": self.history_model,
            "application_model": self.application_model,
            "calibrator": self.calibrator,
            "gap_policy": self.gap_policy,
            "demo_profiles": self.demo_profiles,
            "final_metrics": self.final_metrics,
            "fairness_audit": self.fairness_audit,
        }


def validate_overrides(overrides: dict[str, float | None] | None) -> dict[str, float]:
    clean = {}
    for name, value in (overrides or {}).items():
        if name not in OVERRIDE_COLUMNS:
            raise ValueError(f"unsupported override: {name}")
        if value is None:
            continue
        numeric = float(value)
        if not 0.0 < numeric < 1_000_000_000.0:
            raise ValueError(f"{name} must be positive and below 1,000,000,000")
        clean[name] = numeric
    return clean


def affordability_context(
    *,
    annual_income: float,
    annual_annuity: float,
    requested_credit: float,
    maximum_ratio: float = 0.40,
) -> dict[str, Any]:
    """Return a transparent proposed-repayment screen, separate from ML risk."""

    if min(annual_income, annual_annuity, requested_credit) <= 0:
        raise ValueError("affordability inputs must be positive")
    ratio = annual_annuity / annual_income
    screen_passes = ratio <= maximum_ratio
    affordable_annuity = maximum_ratio * annual_income
    estimated_credit = requested_credit * affordable_annuity / annual_annuity
    return {
        "status": "SCREEN_PASS" if screen_passes else "SCREEN_REVIEW",
        "proposed_repayment_to_income": ratio,
        "maximum_research_ratio": maximum_ratio,
        "estimated_credit_at_maximum_ratio": max(0.0, estimated_credit),
        "note": (
            "Research affordability screen using proposed annual annuity divided by annual income; "
            "it excludes unobserved obligations and is not a lending decision."
        ),
    }


class LocalModelRuntime:
    """Load local artifacts once and provide deterministic applicant-level scoring."""

    def __init__(self, paths: RuntimePaths | None = None) -> None:
        self.paths = paths or RuntimePaths()
        missing = [name for name, path in self.paths.required().items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "CrediFast local runtime is missing required artifacts: " + ", ".join(missing)
            )
        application = pd.read_csv(self.paths.application, low_memory=False)
        if application["SK_ID_CURR"].duplicated().any():
            raise ValueError("application demo source contains duplicate SK_ID_CURR values")
        self.application = application.set_index("SK_ID_CURR", drop=False)
        self.history_columns = pq.read_schema(self.paths.history).names
        self.profile_bundle = json.loads(self.paths.demo_profiles.read_text(encoding="utf-8"))
        self.final_metrics = json.loads(self.paths.final_metrics.read_text(encoding="utf-8"))
        self.fairness_audit = json.loads(self.paths.fairness_audit.read_text(encoding="utf-8"))
        self.scorer = HistoryModelScorer(
            self.paths.history_model,
            self.paths.calibrator,
            application_model_path=self.paths.application_model,
            gap_strategy_path=self.paths.gap_policy,
        )

    def profiles(self) -> list[dict[str, Any]]:
        return [dict(profile) for profile in self.profile_bundle["profiles"]]

    def profile(self, key: str) -> dict[str, Any]:
        for profile in self.profiles():
            if profile["key"] == key:
                return profile
        raise ValueError(f"unknown demo profile: {key}")

    def application_inputs(self, application_id: int) -> dict[str, float | None]:
        """Return the editable application fields used by the reviewer workbench."""

        if int(application_id) not in self.application.index:
            raise ValueError(f"application ID {application_id} was not found in the demo population")
        row = self.application.loc[int(application_id)]
        return {
            name: None if pd.isna(row[column]) else float(row[column])
            for name, column in OVERRIDE_COLUMNS.items()
        }

    def status(self) -> dict[str, Any]:
        metrics = self.final_metrics["model_metrics"]["frozen_operational_candidate"]
        return {
            "ready": True,
            "application_rows": len(self.application),
            "demo_profiles": len(self.profile_bundle["profiles"]),
            "model": {
                "name": self.scorer.model_bundle["model_name"],
                "version": self.scorer.model_bundle.get("model_version"),
                "calibration": self.scorer.calibrator_bundle["method"],
                "gap_strategy": self.scorer.gap_strategy["policy_version"],
            },
            "final_holdout": {
                "rows": self.final_metrics["evaluation_protocol"]["rows"],
                "roc_auc": metrics["ranking"]["roc_auc"],
                "average_precision": metrics["ranking"]["average_precision"],
                "brier_score": metrics["calibration"]["brier_score"],
                "calibration_slope": metrics["calibration"]["calibration_slope"],
                "all_gates_passed": self.final_metrics["acceptance_gates"]["all_passed"],
            },
            "fairness_assessment": self.fairness_audit["assessment"],
            "production_approved": False,
        }

    def input_schema(self) -> dict[str, Any]:
        """Return versioned input provenance for the frozen selected feature set."""

        selected = self.scorer.model_bundle["feature_selection"]["selected"]
        return build_input_source_schema(selected)

    def _history_row(self, application_id: int, simulate_unavailable: bool) -> pd.DataFrame:
        if simulate_unavailable:
            return pd.DataFrame(columns=self.history_columns)
        history = pd.read_parquet(
            self.paths.history,
            filters=[("SK_ID_CURR", "=", int(application_id))],
        )
        return history

    def score(
        self,
        application_id: int,
        *,
        overrides: dict[str, float | None] | None = None,
        explain: bool = True,
        simulate_history_unavailable: bool = False,
    ) -> dict[str, Any]:
        if int(application_id) not in self.application.index:
            raise ValueError(f"application ID {application_id} was not found in the demo population")
        clean_overrides = validate_overrides(overrides)
        application = self.application.loc[[int(application_id)]].copy().reset_index(drop=True)
        original = {
            name: (
                None
                if pd.isna(application.iloc[0][column])
                else float(application.iloc[0][column])
            )
            for name, column in OVERRIDE_COLUMNS.items()
        }
        for name, value in clean_overrides.items():
            application.loc[:, OVERRIDE_COLUMNS[name]] = value
        effective = {
            name: (
                None
                if pd.isna(application.iloc[0][column])
                else float(application.iloc[0][column])
            )
            for name, column in OVERRIDE_COLUMNS.items()
        }
        required_affordability = ("annual_income", "annual_annuity", "requested_credit")
        if any(effective[name] is None for name in required_affordability):
            raise ValueError("selected application is missing required affordability inputs")
        history = self._history_row(int(application_id), simulate_history_unavailable)
        records, timings = self.scorer.score(application, history, explain=explain)
        result = records[0]
        affordability = affordability_context(
            annual_income=float(effective["annual_income"]),
            annual_annuity=float(effective["annual_annuity"]),
            requested_credit=float(effective["requested_credit"]),
        )
        probability = result["repayment_difficulty_probability"]
        if result["review_route"] == "MANUAL_REVIEW_DATA_LIMITED":
            review_route = "DATA_LIMITED_REVIEW"
        elif probability >= 0.20:
            review_route = "HIGH_RISK_REVIEW"
        elif affordability["status"] == "SCREEN_REVIEW":
            review_route = "AFFORDABILITY_REVIEW"
        elif result["review_route"] == "MANUAL_REVIEW_NEAR_RISK_BOUNDARY":
            review_route = "BOUNDARY_REVIEW"
        else:
            review_route = "STANDARD_REVIEW"
        result.update(
            {
                "review_route": review_route,
                "affordability": affordability,
                "input_snapshot": {
                    "original": original,
                    "effective": effective,
                    "overrides": clean_overrides,
                    "historical_snapshot": (
                        "SIMULATED_UNAVAILABLE" if simulate_history_unavailable else "LOCAL_STORE"
                    ),
                },
                "timings": timings,
            }
        )
        return result

    def score_profile(self, key: str, *, explain: bool = True) -> dict[str, Any]:
        profile = self.profile(key)
        result = self.score(
            int(profile["application_id"]),
            overrides=profile.get("overrides"),
            explain=explain,
            simulate_history_unavailable=bool(profile.get("simulate_history_unavailable")),
        )
        result["demo_profile"] = profile
        return result


@lru_cache(maxsize=1)
def get_local_model_runtime() -> LocalModelRuntime:
    return LocalModelRuntime()
