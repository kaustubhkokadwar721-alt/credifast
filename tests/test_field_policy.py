"""The measured field policy must actually govern routing, not just be documented."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))

from credifast.data.workbook_intake import build_features

pytest.importorskip("streamlit", reason="field policy lives in the Streamlit UI layer")

from field_policy import (
    affordability_gaps,
    load_policy,
    missing_expected,
    required_application_fields,
)
from new_application import _route

POLICY_PATH = ROOT / "configs" / "intake_requirements.json"

BASE_APPLICATION = {
    "applicant_ref": "POLICY-001",
    "annual_income": 720_000.0,
    "requested_credit": 600_000.0,
    "annual_annuity": 168_000.0,
    "goods_price": 580_000.0,
    "contract_type": "Cash loans",
    "income_type": "Working",
    "education_type": "Higher education",
    "housing_type": "House / apartment",
    "owns_car": "N",
    "car_age_years": None,
    "owns_realty": "N",
    "employment_years": 3.5,
    "address_vintage_years": 4.0,
    "id_document_age_years": 3.0,
    "has_email": "Y",
    "has_work_phone": "N",
}

TRADELINES = pd.DataFrame(
    [
        {
            "account_ref": "TL-01",
            "credit_type": "Consumer credit",
            "status": "Active",
            "opened_days_ago": 640,
            "sanctioned_amount": 300_000.0,
            "credit_limit": 0.0,
            "current_balance": 186_000.0,
            "overdue_amount": 0.0,
            "days_overdue": 0,
            "annuity": 36_000.0,
            "times_prolonged": 0,
        }
    ]
)

ACCOUNT_MONTHS = pd.DataFrame(
    [
        {
            "account_ref": "CC-1",
            "account_kind": "revolving",
            "months_ago": month,
            "dpd_days": 0,
            "dpd_days_tolerant": 0,
            "contract_status": "Active",
            "balance": 40_000.0,
            "credit_limit": 150_000.0,
            "installments_total": None,
            "installments_remaining": None,
        }
        for month in range(4)
    ]
    + [
        {
            "account_ref": "POS-1",
            "account_kind": "installment",
            "months_ago": month,
            "dpd_days": 0,
            "dpd_days_tolerant": 0,
            "contract_status": "Active",
            "balance": None,
            "credit_limit": None,
            "installments_total": 36,
            "installments_remaining": 20,
        }
        for month in range(4)
    ]
)


def _sheets(**overrides) -> dict[str, pd.DataFrame]:
    application = dict(BASE_APPLICATION)
    application.update(overrides.pop("application", {}))
    sheets = {
        "application": pd.DataFrame([application]),
        "tradelines": TRADELINES.copy(),
        "account_months": ACCOUNT_MONTHS.copy(),
    }
    sheets.update(overrides)
    return sheets


@pytest.mark.skipif(not POLICY_PATH.exists(), reason="requires the generated field policy")
def test_required_fields_come_from_the_measured_policy():
    required = required_application_fields(load_policy())
    assert "annual_income" in required
    assert "requested_credit" in required
    # Measured OPTIONAL fields must never appear as required.
    assert "has_email" not in required
    assert "id_document_age_years" not in required


def test_complete_package_reports_no_expected_gaps():
    intake = build_features(_sheets())
    assert missing_expected(intake) == []


def test_absent_expected_field_forces_data_limited_review():
    """goods_price moved 1.036% of calibration routes, so its absence must be routed on."""

    intake = build_features(_sheets(application={"goods_price": None}))
    gaps = missing_expected(intake)
    assert [gap["field"] for gap in gaps] == ["application.goods_price"]

    # Three source families present, low estimate: without the policy this is STANDARD.
    route, _, _ = _route(3, 0.02, len(gaps))
    assert route == "DATA-LIMITED MANUAL REVIEW"
    assert _route(3, 0.02, 0)[0] == "STANDARD MANUAL REVIEW"


def test_thin_file_reports_family_gaps_without_double_counting_subfields():
    """A file with no history must not also be blamed for missing sub-fields of it."""

    intake = build_features({"application": pd.DataFrame([BASE_APPLICATION])})
    fields = {gap["field"] for gap in missing_expected(intake)}
    assert fields == {"sheet.tradelines", "sheet.account_months"}
    assert "tradelines.opened_days_ago" not in fields
    assert "account_months.balance_and_limit" not in fields


def test_missing_open_dates_is_reported_when_accounts_exist():
    tradelines = TRADELINES.copy()
    tradelines["opened_days_ago"] = None
    intake = build_features(_sheets(tradelines=tradelines))
    assert "tradelines.opened_days_ago" in {gap["field"] for gap in missing_expected(intake)}


def test_instalments_are_optional_for_routing_but_flagged_for_affordability():
    """The tiering measures risk-model dependence only; affordability is separate."""

    tradelines = TRADELINES.copy()
    tradelines["annuity"] = None
    intake = build_features(_sheets(tradelines=tradelines))

    assert missing_expected(intake) == []  # routing is untouched
    assert pd.isna(intake.features.iloc[0]["TOTAL_OBLIGATION_TO_INCOME"])
    notes = affordability_gaps(intake)
    assert notes and "obligation ratio cannot be computed" in notes[0]


def test_expected_gap_never_overrides_the_thin_file_rule():
    intake = build_features({"application": pd.DataFrame([BASE_APPLICATION])})
    gaps = missing_expected(intake)
    route, _, _ = _route(0, 0.9, len(gaps))
    assert route == "DATA-LIMITED MANUAL REVIEW"
