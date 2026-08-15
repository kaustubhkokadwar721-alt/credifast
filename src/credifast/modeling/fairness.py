"""Audit-only group diagnostics for the frozen research champion."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .calibration import calibration_diagnostics


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def group_fairness_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    review_threshold: float,
    minimum_rows: int = 200,
    minimum_events: int = 20,
) -> dict:
    """Measure ranking, calibration, and a capacity-based review route."""

    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    selected = p >= review_threshold
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    selected_positives = int((selected & y.astype(bool)).sum())
    selected_negatives = int((selected & ~y.astype(bool)).sum())
    eligible = len(y) >= minimum_rows and positives >= minimum_events and negatives >= minimum_events
    result = {
        "rows": len(y),
        "events": positives,
        "non_events": negatives,
        "event_rate": float(y.mean()) if len(y) else None,
        "event_rate_95_percent_interval": _wilson_interval(positives, len(y)),
        "mean_probability": float(p.mean()) if len(p) else None,
        "review_rate": float(selected.mean()) if len(selected) else None,
        "review_rate_95_percent_interval": _wilson_interval(int(selected.sum()), len(selected)),
        "true_positive_rate_at_review_capacity": (
            selected_positives / positives if positives else None
        ),
        "true_positive_rate_95_percent_interval": _wilson_interval(
            selected_positives, positives
        ),
        "false_positive_rate_at_review_capacity": (
            selected_negatives / negatives if negatives else None
        ),
        "false_positive_rate_95_percent_interval": _wilson_interval(
            selected_negatives, negatives
        ),
        "eligible_for_comparison": eligible,
        "suppression_rule": {
            "minimum_rows": minimum_rows,
            "minimum_events": minimum_events,
        },
    }
    if eligible:
        diagnostics = calibration_diagnostics(y, p)
        result.update(
            {
                "roc_auc": float(roc_auc_score(y, p)),
                "average_precision": float(average_precision_score(y, p)),
                "brier_score": float(brier_score_loss(y, p)),
                "expected_calibration_error": diagnostics["expected_calibration_error"],
                "calibration_intercept": diagnostics["calibration_intercept"],
                "calibration_slope": diagnostics["calibration_slope"],
            }
        )
    else:
        result["suppressed_metrics"] = [
            "roc_auc",
            "average_precision",
            "brier_score",
            "expected_calibration_error",
            "calibration_intercept",
            "calibration_slope",
        ]
    return result


def _comparison_gaps(groups: dict[str, dict]) -> dict:
    metrics = (
        "event_rate",
        "mean_probability",
        "review_rate",
        "true_positive_rate_at_review_capacity",
        "false_positive_rate_at_review_capacity",
        "roc_auc",
        "average_precision",
        "brier_score",
        "expected_calibration_error",
    )
    gaps = {}
    eligible = {
        name: values for name, values in groups.items() if values["eligible_for_comparison"]
    }
    for metric in metrics:
        available = {
            name: values.get(metric)
            for name, values in eligible.items()
            if values.get(metric) is not None
        }
        if len(available) < 2:
            continue
        high_group = max(available, key=available.get)
        low_group = min(available, key=available.get)
        gaps[metric] = {
            "maximum_group": high_group,
            "maximum": float(available[high_group]),
            "minimum_group": low_group,
            "minimum": float(available[low_group]),
            "absolute_gap": float(available[high_group] - available[low_group]),
        }
    return gaps


def run_fairness_audit(
    application_path: str | Path,
    predictions_path: str | Path,
    feature_inventory_path: str | Path,
    *,
    output_path: str | Path,
) -> dict:
    """Audit gender and age bands without exposing protected fields to prediction."""

    application_source = Path(application_path)
    predictions_source = Path(predictions_path)
    feature_inventory_source = Path(feature_inventory_path)
    application = pd.read_csv(
        application_source,
        usecols=["SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH"],
    )
    predictions = pd.read_parquet(predictions_source)
    predictions = predictions.loc[predictions["calibration_role"].eq("calibrator_evaluation")]
    if predictions["selection_evaluation_probability"].isna().any():
        raise ValueError("fairness audit requires out-of-calibrator-fit evaluation probabilities")
    frame = predictions.merge(application, on="SK_ID_CURR", how="left", validate="one_to_one")
    if frame[["CODE_GENDER", "DAYS_BIRTH"]].isna().any().any():
        raise ValueError("audit fields failed to join for one or more evaluation rows")

    feature_inventory = json.loads(feature_inventory_source.read_text(encoding="utf-8"))
    selected = set(feature_inventory["selected"])
    excluded_by_policy = set(feature_inventory["excluded_by_policy"])
    protected_controls = {
        "CODE_GENDER": {
            "selected_for_prediction": "CODE_GENDER" in selected,
            "excluded_by_policy": "CODE_GENDER" in excluded_by_policy,
        },
        "DAYS_BIRTH": {
            "selected_for_prediction": "DAYS_BIRTH" in selected,
            "excluded_by_policy": "DAYS_BIRTH" in excluded_by_policy,
        },
    }
    if any(values["selected_for_prediction"] for values in protected_controls.values()):
        raise ValueError("protected audit fields must not be selected for prediction")

    probability = frame["selection_evaluation_probability"].to_numpy(dtype=float)
    target = frame["TARGET"].to_numpy(dtype=int)
    review_threshold = float(np.quantile(probability, 0.80))
    frame["gender_audit_group"] = frame["CODE_GENDER"].replace({"XNA": "Unknown"})
    age_years = -frame["DAYS_BIRTH"] / 365.25
    frame["age_audit_group"] = pd.cut(
        age_years,
        bins=[0, 24, 34, 44, 54, 64, np.inf],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype("string")

    dimensions = {}
    for dimension, column in (
        ("gender", "gender_audit_group"),
        ("age_band", "age_audit_group"),
    ):
        groups = {}
        for group_name, group in frame.groupby(column, dropna=False):
            positions = group.index
            groups[str(group_name)] = group_fairness_metrics(
                frame.loc[positions, "TARGET"].to_numpy(dtype=int),
                frame.loc[positions, "selection_evaluation_probability"].to_numpy(dtype=float),
                review_threshold=review_threshold,
            )
        dimensions[dimension] = {"groups": groups, "observed_gaps": _comparison_gaps(groups)}

    overall = group_fairness_metrics(
        target,
        probability,
        review_threshold=review_threshold,
        minimum_rows=1,
        minimum_events=1,
    )
    report = {
        "run_version": "1.0.0",
        "model_scope": "history-enriched LightGBM with selected identity calibration",
        "assessment": "share_with_caveats",
        "population": {
            "partition": "calibration evaluation half",
            "rows": len(frame),
            "target_rate": float(target.mean()),
            "probability_field": "selection_evaluation_probability",
        },
        "protected_field_controls": protected_controls,
        "review_capacity_definition": {
            "population_share": 0.20,
            "global_probability_threshold": review_threshold,
            "meaning": (
                "Audit proxy for a fixed human-review queue; not an approval, decline, or pricing threshold."
            ),
        },
        "overall": overall,
        "dimensions": dimensions,
        "validation": {
            "join_rows": len(frame),
            "unique_application_ids": int(frame["SK_ID_CURR"].nunique()),
            "duplicate_application_ids": int(frame["SK_ID_CURR"].duplicated().sum()),
            "small_group_suppression": "rows below 200 or fewer than 20 events/non-events",
            "confidence_intervals": "95% Wilson intervals for rates",
            "holdout_evaluated": False,
        },
        "required_caveats": [
            "Gender and age are audit-only and excluded from prediction, but other inputs may remain correlated proxies.",
            "Observed gaps are descriptive and do not establish discrimination, causality, or legal compliance.",
            "The review route is a fixed-capacity audit proxy, not a lending decision threshold.",
            "Competition data and a random split do not represent a production Indian lending population.",
            "Independent legal, compliance, and model-risk review is required before any real lending use.",
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
