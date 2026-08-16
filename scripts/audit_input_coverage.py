"""Audit which intake fields the collectible-input model actually depends on.

Two measurements, both on the calibration partition. The final holdout is spent and is
never read.

1. Gain importance -- what the model used when data was present. Uses the booster's gain,
   not the split-count default that ``feature_importances_`` returns.
2. Missingness ablation -- what breaks when a field is absent. This drives the policy.
   Gain says nothing about absence, and only absence justifies making a field mandatory.

The ablation masks by INTAKE FIELD, because a field is the only thing a reviewer can be
required to supply. One field often feeds several features, and those are masked together.
Masking reproduces the parser's real absent-data behaviour: numeric features go missing,
availability flags go to zero, and APPLICATION_MISSING_COUNT is recomputed. Nothing is
imputed as zero and no model is refitted.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\audit_input_coverage.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ui"))

# The routing rule is imported, never re-implemented, so the audit cannot drift from the UI.
from new_application import _route

from credifast.data.workbook_intake import APPLICATION_COLUMN_MAP, REQUIRED_APPLICATION_COLUMNS
from credifast.modeling.application_baseline import assign_splits
from credifast.modeling.application_features import engineer_application_features
from credifast.modeling.collectible_features import add_obligation_ratios, recompute_missing_count
from credifast.modeling.collectible_inference import align_features, load_bundle

APPLICATION_PATH = ROOT / "data" / "raw" / "application_train.csv"
HISTORY_PATH = ROOT / "data" / "processed" / "history_features.parquet"
JSON_OUTPUT = ROOT / "artifacts" / "input_coverage_audit.json"
POLICY_OUTPUT = ROOT / "configs" / "intake_requirements.json"

RANDOM_SEED = 42

# Tier boundary on route-change share. Justified in docs/INPUT_COVERAGE_AUDIT.md.
EXPECTED_ROUTE_CHANGE_THRESHOLD = 0.01

AVAILABILITY_FLAGS = {
    "tradelines": "HAS_BUREAU_HISTORY",
    "account_months.revolving": "HAS_CREDIT_CARD_HISTORY",
    "account_months.installment": "HAS_POS_HISTORY",
}

# Derived features and the declared terms they are computed from.
DERIVED_SOURCES = {
    "CREDIT_TO_INCOME": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
    "ANNUITY_TO_INCOME": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    "CREDIT_TO_ANNUITY": ("AMT_CREDIT", "AMT_ANNUITY"),
    "CREDIT_TO_GOODS": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
    "DOWN_PAYMENT_RATE": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
    "EXTERNAL_DEBT_TO_INCOME": ("BUREAU_DEBT_SUM", "AMT_INCOME_TOTAL"),
    "TOTAL_OBLIGATION_TO_INCOME": ("AMT_ANNUITY", "BUREAU_ANNUITY_SUM", "AMT_INCOME_TOTAL"),
    "EMPLOYMENT_SENTINEL": ("EMPLOYMENT_YEARS",),
}

# Sheet columns that feed named bureau aggregates, from the expressions in workbook_intake.
TRADELINE_COLUMN_FEATURES = {
    "status": (
        "BUREAU_ACTIVE_COUNT",
        "BUREAU_CLOSED_COUNT",
        "BUREAU_SOLD_COUNT",
        "BUREAU_BAD_DEBT_COUNT",
        "BUREAU_ACTIVE_RATE",
    ),
    "credit_type": ("BUREAU_CREDIT_TYPE_COUNT",),
    "opened_days_ago": (
        "BUREAU_DAYS_SINCE_CREDIT_MEAN",
        "BUREAU_DAYS_SINCE_CREDIT_MIN",
        "BUREAU_DAYS_SINCE_CREDIT_MAX",
        "BUREAU_RECENT_12M_COUNT",
        "BUREAU_RECENT_24M_COUNT",
    ),
    "sanctioned_amount": (
        "BUREAU_CREDIT_SUM",
        "BUREAU_CREDIT_MEAN",
        "BUREAU_CREDIT_MAX",
        "BUREAU_DEBT_TO_CREDIT",
        "BUREAU_OVERDUE_TO_CREDIT",
    ),
    "credit_limit": ("BUREAU_LIMIT_SUM",),
    "current_balance": (
        "BUREAU_DEBT_SUM",
        "BUREAU_DEBT_MEAN",
        "BUREAU_DEBT_MAX",
        "BUREAU_DEBT_TO_CREDIT",
    ),
    "overdue_amount": (
        "BUREAU_OVERDUE_SUM",
        "BUREAU_OVERDUE_MAX",
        "BUREAU_OVERDUE_TO_CREDIT",
    ),
    "days_overdue": (
        "BUREAU_DAYS_OVERDUE_MEAN",
        "BUREAU_DAYS_OVERDUE_MAX",
        "BUREAU_OVERDUE_CREDIT_COUNT",
        "BUREAU_OVERDUE_RATE",
    ),
    "annuity": ("BUREAU_ANNUITY_SUM",),
    "times_prolonged": ("BUREAU_PROLONG_SUM", "BUREAU_PROLONG_MAX"),
}

ACCOUNT_MONTH_COLUMN_FEATURES = {
    "dpd_days": (
        "CC_DPD_MEAN",
        "CC_DPD_MAX",
        "CC_DPD_MONTH_COUNT",
        "CC_DPD_MONTH_RATE",
        "CC_RECENT_12M_DPD_COUNT",
        "POS_DPD_MEAN",
        "POS_DPD_MAX",
        "POS_DPD_SUM",
        "POS_DPD_MONTH_COUNT",
        "POS_DPD_MONTH_RATE",
        "POS_RECENT_12M_DPD_COUNT",
    ),
    "dpd_days_tolerant": (
        "CC_DPD_DEF_MONTH_COUNT",
        "CC_DPD_DEF_MONTH_RATE",
        "POS_DPD_DEF_MEAN",
        "POS_DPD_DEF_MAX",
        "POS_DPD_DEF_MONTH_COUNT",
        "POS_DPD_DEF_MONTH_RATE",
    ),
    "balance": ("CC_BALANCE_MEAN", "CC_BALANCE_MAX", "CC_BALANCE_SUM"),
    "credit_limit": ("CC_LIMIT_MEAN", "CC_LIMIT_MAX"),
    "balance_and_limit": ("CC_UTILIZATION_MEAN", "CC_UTILIZATION_MAX"),
    "contract_status": ("CC_ACTIVE_MONTH_COUNT", "POS_ACTIVE_MONTH_COUNT", "POS_COMPLETED_MONTH_COUNT"),
    "installments_total": ("POS_INSTALLMENT_COUNT_MEAN", "POS_INSTALLMENT_COUNT_MAX"),
    "installments_remaining": ("POS_FUTURE_INSTALLMENTS_MEAN", "POS_FUTURE_INSTALLMENTS_MIN"),
}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def classify(feature: str) -> tuple[str, str]:
    """Return (classification, producing intake field or sheet)."""

    reverse_application = {model: sheet for sheet, model in APPLICATION_COLUMN_MAP.items()}
    if feature in reverse_application:
        return "DIRECT", f"application.{reverse_application[feature]}"
    if feature in DERIVED_SOURCES:
        return "DERIVED", " + ".join(DERIVED_SOURCES[feature])
    if feature == "APPLICATION_MISSING_COUNT":
        return "DERIVED", "computed during feature alignment, not supplied at intake"
    if feature == "HAS_BUREAU_HISTORY":
        return "DERIVED", "presence of rows in tradelines"
    if feature in {"HAS_CREDIT_CARD_HISTORY", "HAS_POS_HISTORY"}:
        return "DERIVED", "presence of rows in account_months"
    if feature.startswith("BUREAU_"):
        return "AGGREGATED", "tradelines"
    if feature.startswith("BB_"):
        return "AGGREGATED", "dpd_history"
    if feature.startswith("AMT_REQ_CREDIT_BUREAU_"):
        return "AGGREGATED", "enquiries"
    if feature.startswith(("CC_", "POS_")):
        return "AGGREGATED", "account_months"
    return "UNREACHABLE", "no intake path produces this feature"


def gain_importance(bundle: dict) -> dict[str, float]:
    """Map booster gain back to source features through the preprocessor."""

    model = bundle["model"]
    numeric = list(bundle["numeric_features"])
    categorical = list(bundle["categorical_features"])
    gains = model.booster_.feature_importance(importance_type="gain")

    # ColumnTransformer emits numeric columns, then SimpleImputer(add_indicator=True)
    # missingness indicators for the numeric columns that had missing values at fit time,
    # then the categorical columns. The indicator positions are recorded on the imputer.
    numeric_imputer = bundle["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
    indicator_sources = (
        [numeric[i] for i in numeric_imputer.indicator_.features_]
        if getattr(numeric_imputer, "indicator_", None) is not None
        else []
    )
    layout = numeric + [f"{name}__missing" for name in indicator_sources] + categorical
    if len(layout) != len(gains):
        raise ValueError(
            f"transformed layout {len(layout)} does not match booster width {len(gains)}"
        )

    totals: dict[str, float] = defaultdict(float)
    for column, gain in zip(layout, gains, strict=True):
        source = column.removesuffix("__missing")
        totals[source] += float(gain)
    return dict(totals)


def build_field_masks(selected: list[str]) -> dict[str, dict]:
    """Map each intake field to the features masked when it is absent."""

    masks: dict[str, dict] = {}

    for sheet_column, model_column in APPLICATION_COLUMN_MAP.items():
        features = {model_column}
        for derived, sources in DERIVED_SOURCES.items():
            if model_column in sources:
                features.add(derived)
        masks[f"application.{sheet_column}"] = {
            "features": sorted(f for f in features if f in selected),
            "flags": {},
            "scope": "application field",
        }

    masks["sheet.tradelines"] = {
        "features": sorted(
            f
            for f in selected
            if f.startswith("BUREAU_") or f in {"EXTERNAL_DEBT_TO_INCOME", "TOTAL_OBLIGATION_TO_INCOME"}
        ),
        "flags": {"HAS_BUREAU_HISTORY": 0},
        "scope": "whole sheet",
    }
    masks["sheet.dpd_history"] = {
        "features": sorted(f for f in selected if f.startswith("BB_")),
        "flags": {},
        "scope": "whole sheet",
    }
    masks["sheet.account_months"] = {
        "features": sorted(f for f in selected if f.startswith(("CC_", "POS_"))),
        "flags": {"HAS_CREDIT_CARD_HISTORY": 0, "HAS_POS_HISTORY": 0},
        "scope": "whole sheet",
    }
    masks["sheet.enquiries"] = {
        "features": sorted(f for f in selected if f.startswith("AMT_REQ_CREDIT_BUREAU_")),
        "flags": {},
        "scope": "whole sheet",
    }

    for column, features in TRADELINE_COLUMN_FEATURES.items():
        chosen = sorted(f for f in features if f in selected)
        if column == "annuity":
            chosen = sorted({*chosen, "TOTAL_OBLIGATION_TO_INCOME"} & set(selected))
        if column == "current_balance":
            chosen = sorted({*chosen, "EXTERNAL_DEBT_TO_INCOME"} & set(selected))
        masks[f"tradelines.{column}"] = {
            "features": chosen,
            "flags": {},
            "scope": "sheet column",
        }

    for column, features in ACCOUNT_MONTH_COLUMN_FEATURES.items():
        masks[f"account_months.{column}"] = {
            "features": sorted(f for f in features if f in selected),
            "flags": {},
            "scope": "sheet column",
        }
    return masks


def main() -> int:
    bundle, calibration_bundle = load_bundle()
    selected = list(bundle["selected_features"])
    _log(f"model {bundle['model_name']} v{bundle['model_version']}, {len(selected)} features")

    # --- Part 1 -------------------------------------------------------------------------
    coverage = []
    counts: dict[str, int] = defaultdict(int)
    for feature in selected:
        classification, producer = classify(feature)
        counts[classification] += 1
        coverage.append(
            {"feature": feature, "classification": classification, "produced_by": producer}
        )
    unreachable = [row["feature"] for row in coverage if row["classification"] == "UNREACHABLE"]

    _log("PART 1 coverage:")
    for name in ("DIRECT", "AGGREGATED", "DERIVED", "UNREACHABLE"):
        _log(f"  {name:12s} {counts[name]}")
    _log(f"  UNREACHABLE features: {unreachable or 'none'}")
    not_supplied_at_intake = [
        row["feature"]
        for row in coverage
        if row["produced_by"].startswith("computed during feature alignment")
    ]
    _log(f"  not emitted by the parser: {not_supplied_at_intake}")

    # --- data ---------------------------------------------------------------------------
    _log("loading calibration partition")
    application = pd.read_csv(APPLICATION_PATH, low_memory=False)
    history = pd.read_parquet(HISTORY_PATH)
    frame = application.merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    target = frame["TARGET"].astype("int8")
    splits = assign_splits(target, random_seed=RANDOM_SEED)
    del application, history
    gc.collect()

    engineered = add_obligation_ratios(engineer_application_features(frame))
    del frame
    gc.collect()
    calibration_mask = splits.eq("calibration")
    base_row = engineered.loc[calibration_mask]
    y = target.loc[calibration_mask].to_numpy(dtype=int)
    _log(f"calibration rows {len(y)}, holdout untouched")

    counted = tuple(f for f in selected if f != "APPLICATION_MISSING_COUNT")

    def score(row: pd.DataFrame) -> np.ndarray:
        aligned, _ = align_features(row, selected)
        for column in bundle["numeric_features"]:
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
        raw = bundle["model"].predict_proba(bundle["preprocessor"].transform(aligned))[:, 1]
        return calibration_bundle["calibrator"].predict(raw)

    baseline = score(base_row)
    base_ap = average_precision_score(y, baseline)
    base_auc = roc_auc_score(y, baseline)
    flags = ["HAS_BUREAU_HISTORY", "HAS_CREDIT_CARD_HISTORY", "HAS_POS_HISTORY"]
    base_sources = base_row[flags].fillna(0).sum(axis=1).to_numpy()
    base_routes = np.array(
        [_route(int(s), float(p))[0] for s, p in zip(base_sources, baseline, strict=True)]
    )
    _log(f"baseline AP {base_ap:.4f} ROC-AUC {base_auc:.4f}")

    # --- Part 2 -------------------------------------------------------------------------
    gains = gain_importance(bundle)
    total_gain = sum(gains.values()) or 1.0
    masks = build_field_masks(selected)
    _log(f"PART 2 ablating {len(masks)} intake fields")

    results = []
    for field, spec in masks.items():
        if not spec["features"] and not spec["flags"]:
            continue
        ablated = base_row.copy()
        for feature in spec["features"]:
            if feature in ablated.columns:
                ablated[feature] = np.nan
        for flag, value in spec["flags"].items():
            if flag in ablated.columns:
                ablated[flag] = value
        # The parser recomputes this over whatever survives, so the ablation must too.
        ablated["APPLICATION_MISSING_COUNT"] = recompute_missing_count(ablated, counted)

        probability = score(ablated)
        sources = ablated[flags].fillna(0).sum(axis=1).to_numpy()
        routes = np.array(
            [_route(int(s), float(p))[0] for s, p in zip(sources, probability, strict=True)]
        )
        field_gain = sum(gains.get(f, 0.0) for f in spec["features"])
        results.append(
            {
                "field": field,
                "scope": spec["scope"],
                "features_masked": len(spec["features"]),
                "gain_share": field_gain / total_gain,
                "delta_average_precision": float(average_precision_score(y, probability) - base_ap),
                "delta_roc_auc": float(roc_auc_score(y, probability) - base_auc),
                "mean_absolute_probability_shift": float(np.abs(probability - baseline).mean()),
                "route_change_share": float((routes != base_routes).mean()),
            }
        )
        del ablated
        gc.collect()

    results.sort(key=lambda row: -row["route_change_share"])
    _log("PART 2 ablation, sorted by route-change share:")
    print(
        f"{'field':38s} {'gain%':>7s} {'dAP':>9s} {'dAUC':>9s} {'meanShift':>10s} {'routeChg':>9s}"
    )
    for row in results:
        print(
            f"{row['field']:38s} {row['gain_share'] * 100:6.2f}% "
            f"{row['delta_average_precision']:+9.5f} {row['delta_roc_auc']:+9.5f} "
            f"{row['mean_absolute_probability_shift']:10.5f} {row['route_change_share'] * 100:8.3f}%"
        )

    # --- Part 3 -------------------------------------------------------------------------
    parser_required = {f"application.{name}" for name in REQUIRED_APPLICATION_COLUMNS}
    policy = {}
    for row in results:
        field = row["field"]
        if field in parser_required:
            tier, reason = "REQUIRED", "the parser rejects an intake package without it"
        elif row["route_change_share"] >= EXPECTED_ROUTE_CHANGE_THRESHOLD:
            tier = "EXPECTED"
            reason = (
                f"absence moves {row['route_change_share'] * 100:.2f}% of cases between review "
                "queues, at or above the 1.00% boundary"
            )
        else:
            tier = "OPTIONAL"
            reason = (
                f"absence moves {row['route_change_share'] * 100:.2f}% of cases between review "
                "queues, below the 1.00% boundary"
            )
        policy[field] = {
            "tier": tier,
            "reason": reason,
            "route_change_share": row["route_change_share"],
            "gain_share": row["gain_share"],
            "delta_average_precision": row["delta_average_precision"],
        }

    tier_counts: dict[str, int] = defaultdict(int)
    for entry in policy.values():
        tier_counts[entry["tier"]] += 1
    _log(f"PART 3 tiers: {dict(tier_counts)}")

    report = {
        "run_version": "1.0.0",
        "model": f"{bundle['model_name']} v{bundle['model_version']}",
        "validated": False,
        "partition": "calibration",
        "holdout_evaluated": False,
        "calibration_rows": len(y),
        "baseline": {"average_precision": float(base_ap), "roc_auc": float(base_auc)},
        "expected_route_change_threshold": EXPECTED_ROUTE_CHANGE_THRESHOLD,
        "coverage_counts": dict(counts),
        "unreachable_features": unreachable,
        "not_emitted_by_parser": not_supplied_at_intake,
        "coverage": coverage,
        "gain_share_by_feature": {
            feature: gains.get(feature, 0.0) / total_gain for feature in selected
        },
        "ablation": results,
        "policy": policy,
        "limitations": [
            (
                "Measured on the calibration partition of a research candidate with no "
                "unbiased final evaluation. The V1 final holdout is spent."
            ),
            (
                "Ablation masks features at inference time; it does not retrain a model "
                "without the field, so it measures dependence of this fitted model, not the "
                "achievable accuracy of a model built without that field."
            ),
            (
                "Route-change share depends on the research routing boundary and the "
                "thin/partial rule; changing either changes the tiering."
            ),
            (
                "Tiering measures dependence of the RISK MODEL only. A field can be optional "
                "for the model and still be required by the separate affordability screen; "
                "tradelines.annuity is exactly that case."
            ),
        ],
    }
    JSON_OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    POLICY_OUTPUT.write_text(
        json.dumps(
            {
                "policy_version": "1.0.0",
                "source": "artifacts/input_coverage_audit.json",
                "expected_route_change_threshold": EXPECTED_ROUTE_CHANGE_THRESHOLD,
                "tiers": {field: entry["tier"] for field, entry in policy.items()},
                "detail": policy,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _log(f"wrote {JSON_OUTPUT}")
    _log(f"wrote {POLICY_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
