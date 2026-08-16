"""Apply the measured intake field policy to routing and to the intake form.

Tiers come from configs/intake_requirements.json, produced by
scripts/audit_input_coverage.py. They are measured, not asserted: a field is EXPECTED
because its absence moved at least 1.00% of calibration cases between review queues.

Absence is detected from the produced feature row rather than from the sheets, so an
uploaded workbook, a demonstration case and a directly entered record are all judged the
same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

POLICY_PATH = Path(__file__).resolve().parents[1] / "configs" / "intake_requirements.json"

# Sentinel features that go missing when an EXPECTED field is absent. Sub-field sentinels
# are only meaningful when their parent family was supplied at all, otherwise a thin file
# would be reported as several separate gaps for evidence it never claimed to have.
EXPECTED_SENTINELS = {
    "sheet.tradelines": {
        "flag": "HAS_BUREAU_HISTORY",
        "label": "credit report",
        "absent_copy": "No credit report was supplied.",
        "resolve": "Obtain a credit information report for the applicant.",
    },
    "sheet.account_months": {
        "flag_any": ("HAS_CREDIT_CARD_HISTORY", "HAS_POS_HISTORY"),
        "label": "per-account monthly detail",
        "absent_copy": "No per-account monthly detail was supplied.",
        "resolve": "Obtain the monthly section of the credit report for each account.",
    },
    "tradelines.opened_days_ago": {
        "feature": "BUREAU_DAYS_SINCE_CREDIT_MEAN",
        "requires": "HAS_BUREAU_HISTORY",
        "label": "account open dates",
        "absent_copy": "Accounts were supplied without open dates, so account recency is unknown.",
        "resolve": "Add the open date to each account on the credit report.",
    },
    "account_months.balance_and_limit": {
        "feature": "CC_UTILIZATION_MEAN",
        "requires": "HAS_CREDIT_CARD_HISTORY",
        "label": "revolving balance and limit",
        "absent_copy": (
            "Revolving months were supplied without both balance and limit, so utilisation "
            "cannot be computed."
        ),
        "resolve": "Add monthly balance and credit limit for each revolving account.",
    },
    "application.goods_price": {
        "feature": "AMT_GOODS_PRICE",
        "label": "financed asset value",
        "absent_copy": "No financed asset value was supplied.",
        "resolve": "Add the invoice, quote or declared asset value.",
    },
}

# Measured as OPTIONAL for the risk model, but the sole source of the external side of the
# obligation ratio. The tiering measures risk-model dependence only.
AFFORDABILITY_CRITICAL = {
    "tradelines.annuity": {
        "feature": "BUREAU_ANNUITY_SUM",
        "requires": "HAS_BUREAU_HISTORY",
        "copy": (
            "Accounts were supplied without instalment amounts. The obligation ratio cannot "
            "be computed, so the affordability screen has nothing to measure. The risk "
            "estimate is unaffected — this field is optional for the model and necessary for "
            "affordability."
        ),
    }
}


def load_policy() -> dict:
    if not POLICY_PATH.exists():
        return {"tiers": {}, "detail": {}}
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def required_application_fields(policy: dict) -> list[str]:
    """Application sheet columns the parser rejects an intake package without."""

    return sorted(
        field.split(".", 1)[1]
        for field, tier in policy.get("tiers", {}).items()
        if tier == "REQUIRED" and field.startswith("application.")
    )


def _present(row: pd.Series, feature: str) -> bool:
    return feature in row.index and not pd.isna(row[feature])


def missing_expected(intake) -> list[dict]:
    """Return the EXPECTED fields this intake package did not supply."""

    row = intake.features.iloc[0]
    gaps = []
    for field, spec in EXPECTED_SENTINELS.items():
        if "flag" in spec:
            absent = row.get(spec["flag"], 0) != 1
        elif "flag_any" in spec:
            absent = all(row.get(flag, 0) != 1 for flag in spec["flag_any"])
        else:
            parent = spec.get("requires")
            if parent is not None and row.get(parent, 0) != 1:
                continue  # the family itself is absent; reported by its own entry
            absent = not _present(row, spec["feature"])
        if absent:
            gaps.append(
                {
                    "field": field,
                    "label": spec["label"],
                    "copy": spec["absent_copy"],
                    "resolve": spec["resolve"],
                }
            )
    return gaps


def affordability_gaps(intake) -> list[str]:
    """Fields the risk model tolerates but the affordability screen needs."""

    row = intake.features.iloc[0]
    notes = []
    for spec in AFFORDABILITY_CRITICAL.values():
        parent = spec.get("requires")
        if parent is not None and row.get(parent, 0) != 1:
            continue
        if not _present(row, spec["feature"]):
            notes.append(spec["copy"])
    return notes
