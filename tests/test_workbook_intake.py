"""Workbook intake tests, including train/serve consistency against the V1 feature store."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from credifast.data.workbook_intake import (
    IntakeError,
    build_features,
    expand_dpd_strips,
    read_intake,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
HISTORY_STORE = ROOT / "data" / "processed" / "history_features.parquet"

BASE_APPLICATION = {
    "applicant_ref": "TEST-001",
    "annual_income": 600000.0,
    "requested_credit": 900000.0,
    "annual_annuity": 120000.0,
    "goods_price": 850000.0,
    "contract_type": "Cash loans",
    "income_type": "Working",
    "education_type": "Higher education",
    "housing_type": "House / apartment",
    "owns_car": "Y",
    "car_age_years": 6,
    "owns_realty": "N",
    "employment_years": 4.5,
    "address_vintage_years": 9.0,
    "id_document_age_years": 3.0,
    "has_email": "Y",
    "has_work_phone": "N",
}


def _sheets(**overrides) -> dict[str, pd.DataFrame]:
    application = dict(BASE_APPLICATION)
    application.update(overrides.pop("application", {}))
    sheets = {"application": pd.DataFrame([application])}
    sheets.update({name: frame for name, frame in overrides.items()})
    return sheets


def test_application_only_produces_thin_file_row():
    result = build_features(_sheets())
    row = result.features.iloc[0]
    assert row["HAS_BUREAU_HISTORY"] == 0
    assert row["AMT_INCOME_TOTAL"] == 600000.0
    assert row["EMPLOYMENT_SENTINEL"] == 0
    assert any("no bureau record" in finding for finding in result.findings)


def test_missing_history_is_not_observed_zero():
    row = build_features(_sheets()).features.iloc[0]
    for column in ("BUREAU_DEBT_SUM", "BUREAU_ANNUITY_SUM", "BB_DELINQUENT_MONTHS"):
        assert column not in row or pd.isna(row.get(column)), (
            f"{column} must stay missing when no bureau record is supplied, never zero"
        )
    assert pd.isna(row["TOTAL_OBLIGATION_TO_INCOME"])


def test_required_field_absence_is_rejected():
    with pytest.raises(IntakeError, match="missing required application field"):
        build_features(_sheets(application={"annual_income": None}))


def test_non_positive_income_is_rejected():
    with pytest.raises(IntakeError, match="annual_income must be positive"):
        build_features(_sheets(application={"annual_income": 0}))


def test_obligation_ratio_uses_reported_emis():
    tradelines = pd.DataFrame(
        [
            {
                "account_ref": "A1",
                "credit_type": "Consumer credit",
                "status": "Active",
                "opened_days_ago": 400,
                "sanctioned_amount": 200000.0,
                "credit_limit": 0.0,
                "current_balance": 90000.0,
                "overdue_amount": 0.0,
                "days_overdue": 0,
                "annuity": 60000.0,
                "times_prolonged": 0,
            },
            {
                "account_ref": "A2",
                "credit_type": "Credit card",
                "status": "Active",
                "opened_days_ago": 900,
                "sanctioned_amount": 100000.0,
                "credit_limit": 100000.0,
                "current_balance": 30000.0,
                "overdue_amount": 0.0,
                "days_overdue": 0,
                "annuity": 20000.0,
                "times_prolonged": 0,
            },
        ]
    )
    row = build_features(_sheets(tradelines=tradelines)).features.iloc[0]
    assert row["BUREAU_ANNUITY_SUM"] == 80000.0
    # (proposed 120000 + reported 80000) / 600000
    assert row["TOTAL_OBLIGATION_TO_INCOME"] == pytest.approx(200000.0 / 600000.0)
    assert row["EXTERNAL_DEBT_TO_INCOME"] == pytest.approx(120000.0 / 600000.0)
    assert row["BUREAU_ACTIVE_RATE"] == 1.0
    assert row["BUREAU_RECENT_12M_COUNT"] == 0


def test_dpd_strip_expands_to_account_months():
    frame = pd.DataFrame([{"account_ref": "A1", "dpd_strip": "001C X"}])
    expanded = expand_dpd_strips(frame)
    assert list(expanded["months_ago"]) == [0, 1, 2, 3, 5]
    assert list(expanded["status"]) == ["0", "0", "1", "C", "X"]


def test_enquiry_bands_are_disjoint_not_cumulative():
    """Home Credit's enquiry fields count their own period alone.

    Verified in the source data: applicant 100059 has WEEK=1 with MON=0 and QRT=0, which
    cannot happen if the windows nest.
    """

    enquiries = pd.DataFrame([{"days_ago": 0}, {"days_ago": 5}, {"days_ago": 60}, {"days_ago": 200}])
    row = build_features(_sheets(enquiries=enquiries)).features.iloc[0]
    assert row["AMT_REQ_CREDIT_BUREAU_HOUR"] == 1
    assert row["AMT_REQ_CREDIT_BUREAU_DAY"] == 0
    assert row["AMT_REQ_CREDIT_BUREAU_WEEK"] == 1
    assert row["AMT_REQ_CREDIT_BUREAU_MON"] == 0
    assert row["AMT_REQ_CREDIT_BUREAU_QRT"] == 1
    assert row["AMT_REQ_CREDIT_BUREAU_YEAR"] == 1


def test_enquiries_beyond_a_year_are_reported_not_silently_dropped():
    enquiries = pd.DataFrame([{"days_ago": 400}, {"days_ago": 500}])
    result = build_features(_sheets(enquiries=enquiries))
    assert result.features.iloc[0]["AMT_REQ_CREDIT_BUREAU_YEAR"] == 0
    assert any("older than 365 days" in finding for finding in result.findings)


def test_csv_directory_round_trip(tmp_path):
    sheets = _sheets()
    for name, frame in sheets.items():
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    result = read_intake(tmp_path)
    assert result.features.iloc[0]["AMT_CREDIT"] == 900000.0


def test_account_months_absent_leaves_behaviour_missing():
    result = build_features(_sheets())
    row = result.features.iloc[0]
    assert row["HAS_CREDIT_CARD_HISTORY"] == 0
    assert row["HAS_POS_HISTORY"] == 0
    assert "CC_UTILIZATION_MEAN" not in row or pd.isna(row.get("CC_UTILIZATION_MEAN"))
    assert any("extended model cannot be served" in finding for finding in result.findings)


def test_revolving_and_installment_months_are_separated():
    account_months = pd.DataFrame(
        [
            {
                "account_ref": "CC-1",
                "account_kind": "revolving",
                "months_ago": 1,
                "dpd_days": 0,
                "dpd_days_tolerant": 0,
                "contract_status": "Active",
                "balance": 40000.0,
                "credit_limit": 100000.0,
            },
            {
                "account_ref": "CC-1",
                "account_kind": "revolving",
                "months_ago": 2,
                "dpd_days": 12,
                "dpd_days_tolerant": 0,
                "contract_status": "Active",
                "balance": 60000.0,
                "credit_limit": 100000.0,
            },
            {
                "account_ref": "POS-1",
                "account_kind": "installment",
                "months_ago": 3,
                "dpd_days": 0,
                "dpd_days_tolerant": 0,
                "contract_status": "Active",
                "installments_total": 24,
                "installments_remaining": 9,
            },
        ]
    )
    row = build_features(_sheets(account_months=account_months)).features.iloc[0]
    assert row["CC_MONTH_COUNT"] == 2
    assert row["CC_CONTRACT_COUNT"] == 1
    assert row["CC_UTILIZATION_MEAN"] == pytest.approx(0.5)
    assert row["CC_UTILIZATION_MAX"] == pytest.approx(0.6)
    assert row["CC_DPD_MAX"] == 12
    assert row["CC_DPD_MONTH_RATE"] == pytest.approx(0.5)
    assert row["CC_RECENT_12M_DPD_COUNT"] == 1
    # Installment months must not leak into the revolving family or vice versa.
    assert row["POS_MONTH_COUNT"] == 1
    assert row["POS_FUTURE_INSTALLMENTS_MIN"] == 9
    assert row["POS_DPD_MAX"] == 0
    assert row["HAS_CREDIT_CARD_HISTORY"] == 1
    assert row["HAS_POS_HISTORY"] == 1


@pytest.mark.skipif(
    not (RAW_DIR / "bureau.csv").exists() or not HISTORY_STORE.exists(),
    reason="requires local Kaggle data and the built V1 feature store",
)
def test_workbook_aggregation_matches_v1_feature_store():
    """The served row must equal what the model was trained on, feature by feature."""

    import duckdb

    store = pd.read_parquet(HISTORY_STORE)
    connection = duckdb.connect()
    applicant = connection.execute(
        f"""
        SELECT b.SK_ID_CURR
        FROM read_csv_auto('{(RAW_DIR / "bureau.csv").as_posix()}', header=true) b
        JOIN read_csv_auto('{(RAW_DIR / "bureau_balance.csv").as_posix()}', header=true) bb
          USING (SK_ID_BUREAU)
        GROUP BY b.SK_ID_CURR
        HAVING COUNT(DISTINCT b.SK_ID_BUREAU) >= 4
        LIMIT 1
        """
    ).fetchone()[0]

    tradelines = connection.execute(
        f"""
        SELECT
            SK_ID_BUREAU        AS account_ref,
            CREDIT_TYPE         AS credit_type,
            CREDIT_ACTIVE       AS status,
            -DAYS_CREDIT        AS opened_days_ago,
            AMT_CREDIT_SUM      AS sanctioned_amount,
            AMT_CREDIT_SUM_LIMIT AS credit_limit,
            AMT_CREDIT_SUM_DEBT AS current_balance,
            AMT_CREDIT_SUM_OVERDUE AS overdue_amount,
            CREDIT_DAY_OVERDUE  AS days_overdue,
            AMT_ANNUITY         AS annuity,
            CNT_CREDIT_PROLONG  AS times_prolonged
        FROM read_csv_auto('{(RAW_DIR / "bureau.csv").as_posix()}', header=true)
        WHERE SK_ID_CURR = {applicant}
        """
    ).df()
    dpd = connection.execute(
        f"""
        SELECT
            bb.SK_ID_BUREAU   AS account_ref,
            -bb.MONTHS_BALANCE AS months_ago,
            bb.STATUS          AS status
        FROM read_csv_auto('{(RAW_DIR / "bureau_balance.csv").as_posix()}', header=true) bb
        JOIN read_csv_auto('{(RAW_DIR / "bureau.csv").as_posix()}', header=true) b
          USING (SK_ID_BUREAU)
        WHERE b.SK_ID_CURR = {applicant}
        """
    ).df()
    connection.close()

    result = build_features(_sheets(tradelines=tradelines, dpd_history=dpd))
    served = result.features.iloc[0]
    expected = store.loc[store["SK_ID_CURR"] == applicant].iloc[0]

    compared = [
        column
        for column in store.columns
        if column.startswith(("BUREAU_", "BB_")) and column in served.index
    ]
    assert compared, "no bureau features available to compare"

    mismatches = []
    for column in compared:
        want, got = expected[column], served[column]
        if pd.isna(want) and pd.isna(got):
            continue
        if pd.isna(want) or pd.isna(got) or not math.isclose(
            float(want), float(got), rel_tol=1e-9, abs_tol=1e-6
        ):
            mismatches.append(f"{column}: store={want!r} workbook={got!r}")
    assert not mismatches, "train/serve skew:\n" + "\n".join(mismatches)


@pytest.mark.skipif(
    not (RAW_DIR / "credit_card_balance.csv").exists() or not HISTORY_STORE.exists(),
    reason="requires local Kaggle data and the built V1 feature store",
)
def test_account_month_aggregation_matches_v1_feature_store():
    """CC_*/POS_* served from account_months must equal the trained feature store."""

    import duckdb

    store = pd.read_parquet(HISTORY_STORE)
    connection = duckdb.connect()
    card_csv = (RAW_DIR / "credit_card_balance.csv").as_posix()
    pos_csv = (RAW_DIR / "POS_CASH_balance.csv").as_posix()

    applicant = connection.execute(
        f"""
        SELECT SK_ID_CURR
        FROM read_csv_auto('{card_csv}', header=true)
        WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM read_csv_auto('{pos_csv}', header=true))
        GROUP BY SK_ID_CURR
        HAVING COUNT(*) >= 20 AND MAX(SK_DPD) > 0
        LIMIT 1
        """
    ).fetchone()[0]

    revolving = connection.execute(
        f"""
        SELECT 'CC-' || CAST(SK_ID_PREV AS VARCHAR) AS account_ref,
               'revolving' AS account_kind, -MONTHS_BALANCE AS months_ago,
               SK_DPD AS dpd_days, SK_DPD_DEF AS dpd_days_tolerant,
               NAME_CONTRACT_STATUS AS contract_status,
               AMT_BALANCE AS balance, AMT_CREDIT_LIMIT_ACTUAL AS credit_limit,
               NULL AS installments_total, NULL AS installments_remaining
        FROM read_csv_auto('{card_csv}', header=true)
        WHERE SK_ID_CURR = {applicant}
        """
    ).df()
    installment = connection.execute(
        f"""
        SELECT 'POS-' || CAST(SK_ID_PREV AS VARCHAR) AS account_ref,
               'installment' AS account_kind, -MONTHS_BALANCE AS months_ago,
               SK_DPD AS dpd_days, SK_DPD_DEF AS dpd_days_tolerant,
               NAME_CONTRACT_STATUS AS contract_status,
               NULL AS balance, NULL AS credit_limit,
               CNT_INSTALMENT AS installments_total,
               CNT_INSTALMENT_FUTURE AS installments_remaining
        FROM read_csv_auto('{pos_csv}', header=true)
        WHERE SK_ID_CURR = {applicant}
        """
    ).df()
    connection.close()

    account_months = pd.concat([revolving, installment], ignore_index=True)
    served = build_features(_sheets(account_months=account_months)).features.iloc[0]
    expected = store.loc[store["SK_ID_CURR"] == applicant].iloc[0]

    compared = [
        column
        for column in store.columns
        if column.startswith(("CC_", "POS_")) and column in served.index
    ]
    assert compared, "no tradeline-behaviour features available to compare"

    mismatches = []
    for column in compared:
        want, got = expected[column], served[column]
        if pd.isna(want) and pd.isna(got):
            continue
        if pd.isna(want) or pd.isna(got) or not math.isclose(
            float(want), float(got), rel_tol=1e-9, abs_tol=1e-6
        ):
            mismatches.append(f"{column}: store={want!r} workbook={got!r}")
    assert not mismatches, "train/serve skew:\n" + "\n".join(mismatches)
