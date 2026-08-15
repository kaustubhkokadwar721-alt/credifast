"""Post-prediction group audit for the collectible-input candidate.

Uses the same method as the frozen V1 audit -- identical group metrics, identical Wilson
intervals, identical small-group suppression, identical fixed-capacity review proxy -- so the
two are directly comparable. The V3 feature set changed substantially, so V1's fairness
evidence must not be assumed to carry over; this measures whether it did.

Audited on the calibrator-evaluation half only, matching V1: those rows were not used to fit
the calibrator, so the probabilities are out-of-fit.

Protected fields are audit-only. The script fails if either is present in the model's
feature list.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\audit_collectible_fairness.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credifast.modeling.application_baseline import assign_splits
from credifast.modeling.application_features import engineer_application_features
from credifast.modeling.collectible_features import add_obligation_ratios, recompute_missing_count
from credifast.modeling.collectible_inference import align_features, load_bundle
from credifast.modeling.fairness import _comparison_gaps, group_fairness_metrics

APPLICATION_PATH = ROOT / "data" / "raw" / "application_train.csv"
HISTORY_PATH = ROOT / "data" / "processed" / "history_features.parquet"
V1_AUDIT_PATH = ROOT / "artifacts" / "history_fairness_audit.json"
OUTPUT_PATH = ROOT / "artifacts" / "collectible_fairness_audit.json"

RANDOM_SEED = 42
REVIEW_CAPACITY = 0.20
PROTECTED_FIELDS = ("CODE_GENDER", "DAYS_BIRTH")

# Gaps large enough that shipping without naming them would be misleading.
GAP_ATTENTION_THRESHOLDS = {
    "review_rate": 0.10,
    "true_positive_rate_at_review_capacity": 0.10,
    "false_positive_rate_at_review_capacity": 0.10,
    "roc_auc": 0.05,
    "expected_calibration_error": 0.02,
}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _source_segment(frame: pd.DataFrame, selected: list[str]) -> pd.Series:
    """Label applicants by how many of THIS model's source families they carry.

    Counting every ``HAS_*`` column in the engineered frame would count the five V1 families
    and mislabel the segments, because V3 consumes only three.
    """

    flags = [column for column in selected if column.startswith("HAS_")]
    if not flags:
        return pd.Series("unknown", index=frame.index, dtype="string")
    count = frame[flags].fillna(0).sum(axis=1)
    total = len(flags)
    return pd.cut(
        count,
        bins=[-0.5, 1.5, total - 0.5, np.inf],
        labels=[
            "thin (0-1 sources)",
            f"partial (2 of {total})",
            f"full ({total} of {total})",
        ],
    ).astype("string")


def _compare_with_v1(dimensions: dict) -> dict:
    """Report how each observed gap moved relative to the frozen V1 audit."""

    if not V1_AUDIT_PATH.exists():
        return {"available": False, "reason": f"{V1_AUDIT_PATH.name} not found"}
    v1 = json.loads(V1_AUDIT_PATH.read_text(encoding="utf-8"))
    comparison: dict = {"available": True, "source": V1_AUDIT_PATH.name, "dimensions": {}}
    for dimension, values in dimensions.items():
        v1_gaps = v1.get("dimensions", {}).get(dimension, {}).get("observed_gaps")
        if not v1_gaps:
            continue
        rows = {}
        for metric, gap in values["observed_gaps"].items():
            if metric not in v1_gaps:
                continue
            v3_value = gap["absolute_gap"]
            v1_value = v1_gaps[metric]["absolute_gap"]
            rows[metric] = {
                "v1_absolute_gap": v1_value,
                "v3_absolute_gap": v3_value,
                "change": v3_value - v1_value,
                "direction": (
                    "wider" if v3_value > v1_value + 1e-9
                    else "narrower" if v3_value < v1_value - 1e-9
                    else "unchanged"
                ),
            }
        comparison["dimensions"][dimension] = rows
    return comparison


def _attention_items(dimensions: dict) -> list[dict]:
    flagged = []
    for dimension, values in dimensions.items():
        for metric, gap in values["observed_gaps"].items():
            threshold = GAP_ATTENTION_THRESHOLDS.get(metric)
            if threshold is None or abs(gap["absolute_gap"]) < threshold:
                continue
            flagged.append(
                {
                    "dimension": dimension,
                    "metric": metric,
                    "absolute_gap": gap["absolute_gap"],
                    "threshold": threshold,
                    "widest_group": gap["maximum_group"],
                    "narrowest_group": gap["minimum_group"],
                }
            )
    return sorted(flagged, key=lambda item: -abs(item["absolute_gap"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    arguments = parser.parse_args()

    bundle, calibration_bundle = load_bundle()
    selected = list(bundle["selected_features"])

    leaked = [field for field in PROTECTED_FIELDS if field in selected]
    if leaked:
        raise ValueError(f"protected audit fields must not be model inputs: {leaked}")

    _log("loading sources")
    application = pd.read_csv(APPLICATION_PATH, low_memory=False)
    history = pd.read_parquet(HISTORY_PATH)
    frame = application.merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    target = frame["TARGET"].astype("int8")
    splits = assign_splits(target, random_seed=RANDOM_SEED)
    audit_fields = frame[["SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH"]].copy()
    del application, history
    gc.collect()

    engineered = add_obligation_ratios(engineer_application_features(frame))
    counted = tuple(f for f in selected if f != "APPLICATION_MISSING_COUNT")
    engineered["APPLICATION_MISSING_COUNT"] = recompute_missing_count(engineered, counted)
    del frame
    gc.collect()

    calibration_mask = splits.eq("calibration")
    aligned, _ = align_features(engineered.loc[calibration_mask, selected], selected)
    for column in bundle["numeric_features"]:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    raw = bundle["model"].predict_proba(bundle["preprocessor"].transform(aligned))[:, 1]
    probability = calibration_bundle["calibrator"].predict(raw)
    calibration_target = target.loc[calibration_mask].to_numpy(dtype=int)

    # Reproduce the champion's calibrator split so the audit uses out-of-fit rows only.
    positions = np.arange(len(calibration_target))
    _, evaluation_positions = train_test_split(
        positions,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=calibration_target,
    )
    _log(f"auditing {len(evaluation_positions)} calibrator-evaluation rows")

    audit = pd.DataFrame(
        {
            "SK_ID_CURR": engineered.loc[calibration_mask, "SK_ID_CURR"].to_numpy(),
            "TARGET": calibration_target,
            "probability": probability,
        }
    ).iloc[evaluation_positions]
    segments = _source_segment(engineered.loc[calibration_mask], selected).to_numpy()[
        evaluation_positions
    ]
    audit["source_segment"] = segments
    audit = audit.merge(audit_fields, on="SK_ID_CURR", how="left", validate="one_to_one")
    if audit[["CODE_GENDER", "DAYS_BIRTH"]].isna().any().any():
        raise ValueError("audit fields failed to join for one or more evaluation rows")

    review_threshold = float(np.quantile(audit["probability"], 1.0 - REVIEW_CAPACITY))
    audit["gender_audit_group"] = audit["CODE_GENDER"].replace({"XNA": "Unknown"})
    age_years = -audit["DAYS_BIRTH"] / 365.25
    audit["age_audit_group"] = pd.cut(
        age_years,
        bins=[0, 24, 34, 44, 54, 64, np.inf],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype("string")

    dimensions = {}
    for dimension, column in (
        ("gender", "gender_audit_group"),
        ("age_band", "age_audit_group"),
        ("source_availability", "source_segment"),
    ):
        groups = {}
        for group_name, group in audit.groupby(column, dropna=False):
            groups[str(group_name)] = group_fairness_metrics(
                group["TARGET"].to_numpy(dtype=int),
                group["probability"].to_numpy(dtype=float),
                review_threshold=review_threshold,
            )
        dimensions[dimension] = {"groups": groups, "observed_gaps": _comparison_gaps(groups)}
        _log(f"  {dimension}: {len(groups)} groups")

    overall = group_fairness_metrics(
        audit["TARGET"].to_numpy(dtype=int),
        audit["probability"].to_numpy(dtype=float),
        review_threshold=review_threshold,
        minimum_rows=1,
        minimum_events=1,
    )

    report = {
        "run_version": "1.0.0",
        "model_scope": (
            f"{bundle['model_name']} v{bundle['model_version']} with "
            f"{calibration_bundle['method']} calibration"
        ),
        "assessment": "share_with_caveats",
        "validated": False,
        "population": {
            "partition": "calibration evaluation half",
            "rows": len(audit),
            "target_rate": float(audit["TARGET"].mean()),
            "probability_field": "calibrated_probability",
        },
        "protected_field_controls": {
            field: {"selected_for_prediction": field in selected, "audit_only": True}
            for field in PROTECTED_FIELDS
        },
        "review_capacity_definition": {
            "population_share": REVIEW_CAPACITY,
            "global_probability_threshold": review_threshold,
            "meaning": (
                "Audit proxy for a fixed human-review queue; not an approval, decline, or "
                "pricing threshold."
            ),
        },
        "overall": overall,
        "dimensions": dimensions,
        "gaps_needing_attention": _attention_items(dimensions),
        "comparison_with_frozen_v1": _compare_with_v1(dimensions),
        "validation": {
            "join_rows": len(audit),
            "unique_application_ids": int(audit["SK_ID_CURR"].nunique()),
            "duplicate_application_ids": int(audit["SK_ID_CURR"].duplicated().sum()),
            "small_group_suppression": "rows below 200 or fewer than 20 events/non-events",
            "confidence_intervals": "95% Wilson intervals for rates",
            "holdout_evaluated": False,
        },
        "required_caveats": [
            (
                "Gender and age are audit-only and excluded from prediction, but other inputs "
                "may remain correlated proxies."
            ),
            (
                "Observed gaps are descriptive and do not establish discrimination, causality, "
                "or legal compliance."
            ),
            "The review route is a fixed-capacity audit proxy, not a lending decision threshold.",
            (
                "Competition data and a random split do not represent a production Indian "
                "lending population."
            ),
            "This candidate has no unbiased final evaluation; the V1 holdout is spent.",
            (
                "Independent legal, compliance, and model-risk review is required before any "
                "real lending use."
            ),
        ],
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    _log(f"review threshold {review_threshold:.6f}")
    for item in report["gaps_needing_attention"]:
        _log(
            f"  ATTENTION {item['dimension']}/{item['metric']}: "
            f"{item['absolute_gap']:+.4f} ({item['widest_group']} vs {item['narrowest_group']})"
        )
    _log(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
