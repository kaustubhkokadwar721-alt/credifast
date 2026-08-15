"""Small-perturbation stability check for controlled reason codes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import HistoryModelScorer

PERTURBED_COLUMNS = ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE")


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def evaluate_explanation_stability(
    application_path: str | Path,
    history_path: str | Path,
    model_path: str | Path,
    calibrator_path: str | Path,
    *,
    output_path: str | Path,
    rows: int = 250,
    perturbation_rate: float = 0.005,
) -> dict:
    """Compare scores and top reasons after a small simultaneous amount perturbation."""

    if rows < 1 or not 0.0 < perturbation_rate < 0.10:
        raise ValueError("rows must be positive and perturbation_rate must be between zero and 0.10")
    application = pd.read_csv(application_path, low_memory=False).sort_values("SK_ID_CURR").head(rows)
    missing = sorted(set(PERTURBED_COLUMNS) - set(application.columns))
    if missing:
        raise ValueError(f"application source is missing perturbation fields: {missing}")
    history = pd.read_parquet(history_path)
    history = history.loc[history["SK_ID_CURR"].isin(set(application["SK_ID_CURR"]))]
    perturbed = application.copy()
    for column in PERTURBED_COLUMNS:
        perturbed[column] = perturbed[column] * (1.0 + perturbation_rate)

    scorer = HistoryModelScorer(model_path, calibrator_path)
    original_records, _ = scorer.score(application, history, explain=True)
    perturbed_records, _ = scorer.score(perturbed, history, explain=True)
    probability_changes = []
    top_one_matches = []
    exact_top_three_matches = []
    jaccard_scores = []
    grade_changes = []
    route_changes = []
    for original, changed in zip(original_records, perturbed_records, strict=True):
        original_codes = [item["code"] for item in original["reasons"]["adverse"]]
        changed_codes = [item["code"] for item in changed["reasons"]["adverse"]]
        probability_changes.append(
            abs(
                original["repayment_difficulty_probability"]
                - changed["repayment_difficulty_probability"]
            )
        )
        top_one_matches.append(
            bool(original_codes and changed_codes and original_codes[0] == changed_codes[0])
        )
        exact_top_three_matches.append(original_codes == changed_codes)
        jaccard_scores.append(_jaccard(set(original_codes), set(changed_codes)))
        grade_changes.append(original["risk_grade"] != changed["risk_grade"])
        route_changes.append(original["review_route"] != changed["review_route"])

    probability_values = np.asarray(probability_changes)
    report = {
        "run_version": "1.0.0",
        "rows": len(original_records),
        "perturbation": {
            "columns": list(PERTURBED_COLUMNS),
            "relative_increase": perturbation_rate,
            "interpretation": "small simultaneous change used as a local robustness probe",
        },
        "reason_stability": {
            "top_adverse_reason_match_rate": float(np.mean(top_one_matches)),
            "exact_ordered_top_three_match_rate": float(np.mean(exact_top_three_matches)),
            "mean_top_three_jaccard": float(np.mean(jaccard_scores)),
        },
        "prediction_stability": {
            "absolute_probability_change_median": float(np.median(probability_values)),
            "absolute_probability_change_p95": float(np.quantile(probability_values, 0.95)),
            "absolute_probability_change_maximum": float(np.max(probability_values)),
            "risk_grade_change_rate": float(np.mean(grade_changes)),
            "review_route_change_rate": float(np.mean(route_changes)),
        },
        "holdout_evaluated": False,
        "limitations": [
            "This probes one local perturbation pattern and does not establish global explanation robustness.",
            "TreeSHAP reasons explain the model, not causality or applicant actionability.",
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
