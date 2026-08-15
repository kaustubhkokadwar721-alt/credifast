"""Export a real applicant's record as an intake workbook the reviewer can upload.

This produces the demo's input artifact: the same applicant the frozen model already knows,
rendered in the collectible workbook schema so the intake path can be exercised end to end
without any live connector.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\export_demo_workbook.py --application-id 176483
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "examples" / "intake_workbook"

APPLICATION_FIELDS = {
    "AMT_INCOME_TOTAL": "annual_income",
    "AMT_CREDIT": "requested_credit",
    "AMT_ANNUITY": "annual_annuity",
    "AMT_GOODS_PRICE": "goods_price",
    "NAME_CONTRACT_TYPE": "contract_type",
    "NAME_INCOME_TYPE": "income_type",
    "NAME_EDUCATION_TYPE": "education_type",
    "NAME_HOUSING_TYPE": "housing_type",
    "FLAG_OWN_CAR": "owns_car",
    "FLAG_OWN_REALTY": "owns_realty",
    "OWN_CAR_AGE": "car_age_years",
}


def _csv(name: str) -> str:
    return f"read_csv_auto('{(RAW_DIR / name).as_posix()}', header=true)"


def export(application_id: int, output: Path, source: str) -> None:
    connection = duckdb.connect()
    application = connection.execute(
        f"SELECT * FROM {_csv(source)} WHERE SK_ID_CURR = {application_id}"
    ).df()
    if application.empty:
        raise SystemExit(f"application {application_id} not found in {source}")
    row = application.iloc[0]

    record = {"applicant_ref": f"HC-{application_id}"}
    for model_column, sheet_column in APPLICATION_FIELDS.items():
        record[sheet_column] = row.get(model_column)

    # Full precision, not rounded: rounding here would show up as train/serve skew in
    # scripts/verify_collectible_serving.py rather than as an export artifact.
    days_employed = row.get("DAYS_EMPLOYED")
    record["employment_years"] = (
        None if days_employed is None or days_employed == 365243 else -days_employed / 365.25
    )
    record["address_vintage_years"] = -row["DAYS_REGISTRATION"] / 365.25
    record["id_document_age_years"] = -row["DAYS_ID_PUBLISH"] / 365.25
    record["has_email"] = "Y" if row.get("FLAG_EMAIL") == 1 else "N"
    record["has_work_phone"] = "Y" if row.get("FLAG_WORK_PHONE") == 1 else "N"

    tradelines = connection.execute(
        f"""
        SELECT
            SK_ID_BUREAU           AS account_ref,
            CREDIT_TYPE            AS credit_type,
            CREDIT_ACTIVE          AS status,
            -DAYS_CREDIT           AS opened_days_ago,
            AMT_CREDIT_SUM         AS sanctioned_amount,
            AMT_CREDIT_SUM_LIMIT   AS credit_limit,
            AMT_CREDIT_SUM_DEBT    AS current_balance,
            AMT_CREDIT_SUM_OVERDUE AS overdue_amount,
            CREDIT_DAY_OVERDUE     AS days_overdue,
            AMT_ANNUITY            AS annuity,
            CNT_CREDIT_PROLONG     AS times_prolonged
        FROM {_csv("bureau.csv")}
        WHERE SK_ID_CURR = {application_id}
        ORDER BY DAYS_CREDIT DESC
        """
    ).df()
    dpd = connection.execute(
        f"""
        SELECT
            bb.SK_ID_BUREAU    AS account_ref,
            -bb.MONTHS_BALANCE AS months_ago,
            bb.STATUS          AS status
        FROM {_csv("bureau_balance.csv")} bb
        JOIN {_csv("bureau.csv")} b USING (SK_ID_BUREAU)
        WHERE b.SK_ID_CURR = {application_id}
        ORDER BY bb.SK_ID_BUREAU, months_ago
        """
    ).df()
    # Per-account monthly detail. In Home Credit this is own-ledger data keyed by SK_ID_PREV,
    # which is why it carries its own account refs rather than the bureau tradeline keys.
    revolving = connection.execute(
        f"""
        SELECT
            'CC-' || CAST(SK_ID_PREV AS VARCHAR) AS account_ref,
            'revolving'                          AS account_kind,
            -MONTHS_BALANCE                      AS months_ago,
            SK_DPD                               AS dpd_days,
            SK_DPD_DEF                           AS dpd_days_tolerant,
            NAME_CONTRACT_STATUS                 AS contract_status,
            AMT_BALANCE                          AS balance,
            AMT_CREDIT_LIMIT_ACTUAL              AS credit_limit,
            NULL                                 AS installments_total,
            NULL                                 AS installments_remaining
        FROM {_csv("credit_card_balance.csv")}
        WHERE SK_ID_CURR = {application_id}
        """
    ).df()
    installment = connection.execute(
        f"""
        SELECT
            'POS-' || CAST(SK_ID_PREV AS VARCHAR) AS account_ref,
            'installment'                         AS account_kind,
            -MONTHS_BALANCE                       AS months_ago,
            SK_DPD                                AS dpd_days,
            SK_DPD_DEF                            AS dpd_days_tolerant,
            NAME_CONTRACT_STATUS                  AS contract_status,
            NULL                                  AS balance,
            NULL                                  AS credit_limit,
            CNT_INSTALMENT                        AS installments_total,
            CNT_INSTALMENT_FUTURE                 AS installments_remaining
        FROM {_csv("POS_CASH_balance.csv")}
        WHERE SK_ID_CURR = {application_id}
        """
    ).df()
    account_months = pd.concat([revolving, installment], ignore_index=True)
    connection.close()

    # Enquiry counts live on the application row as per-band totals; expand them into
    # individual dated records so the sheet carries facts rather than pre-aggregates.
    # The bands are disjoint (see ENQUIRY_BANDS), so each record is placed at a day offset
    # that falls inside its own band and no other.
    enquiries = []
    for column, days in (
        ("AMT_REQ_CREDIT_BUREAU_HOUR", 0),
        ("AMT_REQ_CREDIT_BUREAU_DAY", 1),
        ("AMT_REQ_CREDIT_BUREAU_WEEK", 5),
        ("AMT_REQ_CREDIT_BUREAU_MON", 20),
        ("AMT_REQ_CREDIT_BUREAU_QRT", 70),
        ("AMT_REQ_CREDIT_BUREAU_YEAR", 300),
    ):
        count = row.get(column)
        if count is None or pd.isna(count):
            continue
        enquiries.extend(
            {"days_ago": days, "purpose": "Credit application"} for _ in range(int(count))
        )

    sheets = {
        "application": pd.DataFrame([record]),
        "tradelines": tradelines,
        "dpd_history": dpd,
        "account_months": account_months,
        "enquiries": pd.DataFrame(enquiries, columns=["days_ago", "purpose"]),
    }

    output.mkdir(parents=True, exist_ok=True)
    for name, frame in sheets.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    try:
        with pd.ExcelWriter(output.with_suffix(".xlsx")) as writer:
            for name, frame in sheets.items():
                frame.to_sheet = frame.to_excel(writer, sheet_name=name, index=False)
        print(f"wrote {output.with_suffix('.xlsx')}")
    except (ImportError, ValueError) as exc:
        print(f"skipped .xlsx ({exc}); CSV sheets written instead")

    print(f"wrote {output}/ with {len(sheets)} sheets")
    print(
        f"  tradelines={len(tradelines)} dpd_months={len(dpd)} "
        f"account_months={len(account_months)} enquiries={len(sheets['enquiries'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-id", type=int, default=176483)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source",
        default="application_test.csv",
        choices=["application_test.csv", "application_train.csv"],
    )
    arguments = parser.parse_args()
    export(arguments.application_id, arguments.output, arguments.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
