"""Leakage-resistant aggregation of Home Credit history tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb

REQUIRED_FILES = (
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> str:
    return f"read_csv_auto('{_sql_path(path)}', header=true, sample_size=200000)"


def _feature_query(raw_dir: Path) -> str:
    app_train = _source(raw_dir / "application_train.csv")
    app_test = _source(raw_dir / "application_test.csv")
    bureau = _source(raw_dir / "bureau.csv")
    bureau_balance = _source(raw_dir / "bureau_balance.csv")
    previous = _source(raw_dir / "previous_application.csv")
    pos = _source(raw_dir / "POS_CASH_balance.csv")
    credit_card = _source(raw_dir / "credit_card_balance.csv")
    installments = _source(raw_dir / "installments_payments.csv")
    return f"""
WITH
base_ids AS (
    SELECT SK_ID_CURR FROM {app_train}
    UNION
    SELECT SK_ID_CURR FROM {app_test}
),
bureau_balance_by_loan AS (
    SELECT
        SK_ID_BUREAU,
        COUNT(*) AS BB_MONTHS_OBSERVED,
        MIN(MONTHS_BALANCE) AS BB_OLDEST_MONTH,
        MAX(MONTHS_BALANCE) AS BB_LATEST_MONTH,
        SUM(CASE WHEN STATUS IN ('1', '2', '3', '4', '5') THEN 1 ELSE 0 END)
            AS BB_DELINQUENT_MONTHS,
        SUM(CASE WHEN STATUS IN ('3', '4', '5') THEN 1 ELSE 0 END)
            AS BB_SEVERE_DELINQUENT_MONTHS,
        MAX(COALESCE(TRY_CAST(STATUS AS INTEGER), 0)) AS BB_MAX_DELINQUENCY_LEVEL,
        SUM(CASE WHEN STATUS = '5' THEN 1 ELSE 0 END) AS BB_DEFAULT_MONTHS,
        SUM(CASE WHEN STATUS = 'C' THEN 1 ELSE 0 END) AS BB_CLOSED_MONTHS,
        SUM(CASE WHEN STATUS = 'X' THEN 1 ELSE 0 END) AS BB_UNKNOWN_MONTHS,
        SUM(CASE WHEN MONTHS_BALANCE >= -12 AND STATUS IN ('1', '2', '3', '4', '5')
                 THEN 1 ELSE 0 END) AS BB_RECENT_12M_DELINQUENT_MONTHS
    FROM {bureau_balance}
    GROUP BY SK_ID_BUREAU
),
bureau_features AS (
    SELECT
        b.SK_ID_CURR,
        COUNT(*) AS BUREAU_CREDIT_COUNT,
        COUNT(DISTINCT b.SK_ID_BUREAU) AS BUREAU_UNIQUE_CREDIT_COUNT,
        COUNT(DISTINCT b.CREDIT_TYPE) AS BUREAU_CREDIT_TYPE_COUNT,
        SUM(CASE WHEN b.CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS BUREAU_ACTIVE_COUNT,
        SUM(CASE WHEN b.CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END) AS BUREAU_CLOSED_COUNT,
        SUM(CASE WHEN b.CREDIT_ACTIVE = 'Sold' THEN 1 ELSE 0 END) AS BUREAU_SOLD_COUNT,
        SUM(CASE WHEN b.CREDIT_ACTIVE = 'Bad debt' THEN 1 ELSE 0 END) AS BUREAU_BAD_DEBT_COUNT,
        SUM(CASE WHEN b.DAYS_CREDIT >= -365 THEN 1 ELSE 0 END) AS BUREAU_RECENT_12M_COUNT,
        SUM(CASE WHEN b.DAYS_CREDIT >= -730 THEN 1 ELSE 0 END) AS BUREAU_RECENT_24M_COUNT,
        AVG(-b.DAYS_CREDIT) AS BUREAU_DAYS_SINCE_CREDIT_MEAN,
        MIN(-b.DAYS_CREDIT) AS BUREAU_DAYS_SINCE_CREDIT_MIN,
        MAX(-b.DAYS_CREDIT) AS BUREAU_DAYS_SINCE_CREDIT_MAX,
        AVG(b.CREDIT_DAY_OVERDUE) AS BUREAU_DAYS_OVERDUE_MEAN,
        MAX(b.CREDIT_DAY_OVERDUE) AS BUREAU_DAYS_OVERDUE_MAX,
        SUM(CASE WHEN b.CREDIT_DAY_OVERDUE > 0 THEN 1 ELSE 0 END) AS BUREAU_OVERDUE_CREDIT_COUNT,
        SUM(b.CNT_CREDIT_PROLONG) AS BUREAU_PROLONG_SUM,
        MAX(b.CNT_CREDIT_PROLONG) AS BUREAU_PROLONG_MAX,
        SUM(b.AMT_CREDIT_SUM) AS BUREAU_CREDIT_SUM,
        AVG(b.AMT_CREDIT_SUM) AS BUREAU_CREDIT_MEAN,
        MAX(b.AMT_CREDIT_SUM) AS BUREAU_CREDIT_MAX,
        SUM(b.AMT_CREDIT_SUM_DEBT) AS BUREAU_DEBT_SUM,
        AVG(b.AMT_CREDIT_SUM_DEBT) AS BUREAU_DEBT_MEAN,
        MAX(b.AMT_CREDIT_SUM_DEBT) AS BUREAU_DEBT_MAX,
        SUM(b.AMT_CREDIT_SUM_OVERDUE) AS BUREAU_OVERDUE_SUM,
        MAX(b.AMT_CREDIT_SUM_OVERDUE) AS BUREAU_OVERDUE_MAX,
        SUM(b.AMT_CREDIT_SUM_LIMIT) AS BUREAU_LIMIT_SUM,
        SUM(b.AMT_ANNUITY) AS BUREAU_ANNUITY_SUM,
        SUM(b.AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(b.AMT_CREDIT_SUM), 0)
            AS BUREAU_DEBT_TO_CREDIT,
        SUM(b.AMT_CREDIT_SUM_OVERDUE) / NULLIF(SUM(b.AMT_CREDIT_SUM), 0)
            AS BUREAU_OVERDUE_TO_CREDIT,
        SUM(CASE WHEN b.CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
            AS BUREAU_ACTIVE_RATE,
        SUM(CASE WHEN b.CREDIT_DAY_OVERDUE > 0 THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
            AS BUREAU_OVERDUE_RATE,
        SUM(bb.BB_MONTHS_OBSERVED) AS BB_MONTHS_OBSERVED,
        SUM(bb.BB_DELINQUENT_MONTHS) AS BB_DELINQUENT_MONTHS,
        SUM(bb.BB_SEVERE_DELINQUENT_MONTHS) AS BB_SEVERE_DELINQUENT_MONTHS,
        MAX(bb.BB_MAX_DELINQUENCY_LEVEL) AS BB_MAX_DELINQUENCY_LEVEL,
        SUM(bb.BB_DEFAULT_MONTHS) AS BB_DEFAULT_MONTHS,
        SUM(bb.BB_CLOSED_MONTHS) AS BB_CLOSED_MONTHS,
        SUM(bb.BB_UNKNOWN_MONTHS) AS BB_UNKNOWN_MONTHS,
        SUM(bb.BB_RECENT_12M_DELINQUENT_MONTHS) AS BB_RECENT_12M_DELINQUENT_MONTHS,
        SUM(bb.BB_DELINQUENT_MONTHS)::DOUBLE / NULLIF(SUM(bb.BB_MONTHS_OBSERVED), 0)
            AS BB_DELINQUENT_MONTH_RATE,
        SUM(bb.BB_SEVERE_DELINQUENT_MONTHS)::DOUBLE / NULLIF(SUM(bb.BB_MONTHS_OBSERVED), 0)
            AS BB_SEVERE_DELINQUENT_MONTH_RATE
    FROM {bureau} b
    LEFT JOIN bureau_balance_by_loan bb USING (SK_ID_BUREAU)
    GROUP BY b.SK_ID_CURR
),
previous_features AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS PREV_APPLICATION_COUNT,
        COUNT(DISTINCT SK_ID_PREV) AS PREV_UNIQUE_APPLICATION_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS PREV_APPROVED_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END) AS PREV_REFUSED_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 1 ELSE 0 END) AS PREV_CANCELED_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Unused offer' THEN 1 ELSE 0 END)
            AS PREV_UNUSED_OFFER_COUNT,
        SUM(CASE WHEN DAYS_DECISION >= -365 THEN 1 ELSE 0 END) AS PREV_RECENT_12M_COUNT,
        SUM(CASE WHEN DAYS_DECISION >= -730 THEN 1 ELSE 0 END) AS PREV_RECENT_24M_COUNT,
        MIN(-DAYS_DECISION) AS PREV_DAYS_SINCE_DECISION_MIN,
        AVG(-DAYS_DECISION) AS PREV_DAYS_SINCE_DECISION_MEAN,
        MAX(-DAYS_DECISION) AS PREV_DAYS_SINCE_DECISION_MAX,
        AVG(AMT_APPLICATION) AS PREV_APPLICATION_AMOUNT_MEAN,
        MAX(AMT_APPLICATION) AS PREV_APPLICATION_AMOUNT_MAX,
        SUM(AMT_CREDIT) AS PREV_CREDIT_SUM,
        AVG(AMT_CREDIT) AS PREV_CREDIT_MEAN,
        MAX(AMT_CREDIT) AS PREV_CREDIT_MAX,
        AVG(AMT_ANNUITY) AS PREV_ANNUITY_MEAN,
        MAX(AMT_ANNUITY) AS PREV_ANNUITY_MAX,
        AVG(AMT_DOWN_PAYMENT) AS PREV_DOWN_PAYMENT_MEAN,
        AVG(RATE_DOWN_PAYMENT) AS PREV_DOWN_PAYMENT_RATE_MEAN,
        AVG(CNT_PAYMENT) AS PREV_PAYMENT_COUNT_MEAN,
        MAX(CNT_PAYMENT) AS PREV_PAYMENT_COUNT_MAX,
        AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0)) AS PREV_CREDIT_TO_APPLICATION_MEAN,
        AVG(AMT_ANNUITY / NULLIF(AMT_CREDIT, 0)) AS PREV_ANNUITY_TO_CREDIT_MEAN,
        SUM(CASE WHEN NAME_YIELD_GROUP = 'high' THEN 1 ELSE 0 END) AS PREV_HIGH_YIELD_COUNT,
        SUM(CASE WHEN NAME_YIELD_GROUP = 'middle' THEN 1 ELSE 0 END) AS PREV_MIDDLE_YIELD_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_TYPE = 'Cash loans' THEN 1 ELSE 0 END) AS PREV_CASH_LOAN_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_TYPE = 'Consumer loans' THEN 1 ELSE 0 END)
            AS PREV_CONSUMER_LOAN_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
            AS PREV_APPROVAL_RATE,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
            AS PREV_REFUSAL_RATE
    FROM {previous}
    GROUP BY SK_ID_CURR
),
pos_features AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS POS_MONTH_COUNT,
        COUNT(DISTINCT SK_ID_PREV) AS POS_CONTRACT_COUNT,
        MIN(MONTHS_BALANCE) AS POS_OLDEST_MONTH,
        MAX(MONTHS_BALANCE) AS POS_LATEST_MONTH,
        AVG(CNT_INSTALMENT) AS POS_INSTALLMENT_COUNT_MEAN,
        MAX(CNT_INSTALMENT) AS POS_INSTALLMENT_COUNT_MAX,
        AVG(CNT_INSTALMENT_FUTURE) AS POS_FUTURE_INSTALLMENTS_MEAN,
        MIN(CNT_INSTALMENT_FUTURE) AS POS_FUTURE_INSTALLMENTS_MIN,
        AVG(SK_DPD) AS POS_DPD_MEAN,
        MAX(SK_DPD) AS POS_DPD_MAX,
        SUM(SK_DPD) AS POS_DPD_SUM,
        AVG(SK_DPD_DEF) AS POS_DPD_DEF_MEAN,
        MAX(SK_DPD_DEF) AS POS_DPD_DEF_MAX,
        SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END) AS POS_DPD_MONTH_COUNT,
        SUM(CASE WHEN SK_DPD_DEF > 0 THEN 1 ELSE 0 END) AS POS_DPD_DEF_MONTH_COUNT,
        SUM(CASE WHEN MONTHS_BALANCE >= -12 AND SK_DPD > 0 THEN 1 ELSE 0 END)
            AS POS_RECENT_12M_DPD_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN 1 ELSE 0 END) AS POS_ACTIVE_MONTH_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Completed' THEN 1 ELSE 0 END)
            AS POS_COMPLETED_MONTH_COUNT,
        SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS POS_DPD_MONTH_RATE,
        SUM(CASE WHEN SK_DPD_DEF > 0 THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
            AS POS_DPD_DEF_MONTH_RATE
    FROM {pos}
    GROUP BY SK_ID_CURR
),
credit_card_features AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS CC_MONTH_COUNT,
        COUNT(DISTINCT SK_ID_PREV) AS CC_CONTRACT_COUNT,
        MIN(MONTHS_BALANCE) AS CC_OLDEST_MONTH,
        MAX(MONTHS_BALANCE) AS CC_LATEST_MONTH,
        AVG(AMT_BALANCE) AS CC_BALANCE_MEAN,
        MAX(AMT_BALANCE) AS CC_BALANCE_MAX,
        SUM(AMT_BALANCE) AS CC_BALANCE_SUM,
        AVG(AMT_CREDIT_LIMIT_ACTUAL) AS CC_LIMIT_MEAN,
        MAX(AMT_CREDIT_LIMIT_ACTUAL) AS CC_LIMIT_MAX,
        AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) AS CC_UTILIZATION_MEAN,
        MAX(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) AS CC_UTILIZATION_MAX,
        AVG(AMT_DRAWINGS_CURRENT) AS CC_DRAWINGS_MEAN,
        SUM(AMT_DRAWINGS_CURRENT) AS CC_DRAWINGS_SUM,
        SUM(AMT_DRAWINGS_ATM_CURRENT) AS CC_ATM_DRAWINGS_SUM,
        SUM(AMT_DRAWINGS_POS_CURRENT) AS CC_POS_DRAWINGS_SUM,
        SUM(AMT_PAYMENT_CURRENT) AS CC_PAYMENT_CURRENT_SUM,
        SUM(AMT_PAYMENT_TOTAL_CURRENT) AS CC_PAYMENT_TOTAL_SUM,
        AVG(AMT_PAYMENT_TOTAL_CURRENT / NULLIF(AMT_INST_MIN_REGULARITY, 0))
            AS CC_PAYMENT_TO_MINIMUM_MEAN,
        AVG(AMT_TOTAL_RECEIVABLE) AS CC_RECEIVABLE_MEAN,
        MAX(AMT_TOTAL_RECEIVABLE) AS CC_RECEIVABLE_MAX,
        SUM(CNT_DRAWINGS_CURRENT) AS CC_DRAWING_COUNT_SUM,
        SUM(CNT_DRAWINGS_ATM_CURRENT) AS CC_ATM_DRAWING_COUNT_SUM,
        SUM(CNT_DRAWINGS_POS_CURRENT) AS CC_POS_DRAWING_COUNT_SUM,
        AVG(SK_DPD) AS CC_DPD_MEAN,
        MAX(SK_DPD) AS CC_DPD_MAX,
        SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END) AS CC_DPD_MONTH_COUNT,
        SUM(CASE WHEN SK_DPD_DEF > 0 THEN 1 ELSE 0 END) AS CC_DPD_DEF_MONTH_COUNT,
        SUM(CASE WHEN MONTHS_BALANCE >= -12 AND SK_DPD > 0 THEN 1 ELSE 0 END)
            AS CC_RECENT_12M_DPD_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN 1 ELSE 0 END) AS CC_ACTIVE_MONTH_COUNT,
        SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS CC_DPD_MONTH_RATE,
        SUM(CASE WHEN SK_DPD_DEF > 0 THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
            AS CC_DPD_DEF_MONTH_RATE
    FROM {credit_card}
    GROUP BY SK_ID_CURR
),
installment_features AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS INST_PAYMENT_COUNT,
        COUNT(DISTINCT SK_ID_PREV) AS INST_CONTRACT_COUNT,
        COUNT(DISTINCT NUM_INSTALMENT_NUMBER) AS INST_UNIQUE_NUMBER_COUNT,
        AVG(NUM_INSTALMENT_VERSION) AS INST_VERSION_MEAN,
        SUM(AMT_INSTALMENT) AS INST_DUE_SUM,
        SUM(AMT_PAYMENT) AS INST_PAID_SUM,
        AVG(AMT_INSTALMENT) AS INST_DUE_MEAN,
        AVG(AMT_PAYMENT) AS INST_PAID_MEAN,
        SUM(AMT_PAYMENT) / NULLIF(SUM(AMT_INSTALMENT), 0) AS INST_PAYMENT_RATIO,
        AVG(GREATEST(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)) AS INST_LATE_DAYS_MEAN,
        MAX(GREATEST(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)) AS INST_LATE_DAYS_MAX,
        SUM(GREATEST(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)) AS INST_LATE_DAYS_SUM,
        AVG(GREATEST(AMT_INSTALMENT - AMT_PAYMENT, 0)) AS INST_SHORTFALL_MEAN,
        MAX(GREATEST(AMT_INSTALMENT - AMT_PAYMENT, 0)) AS INST_SHORTFALL_MAX,
        SUM(GREATEST(AMT_INSTALMENT - AMT_PAYMENT, 0)) AS INST_SHORTFALL_SUM,
        SUM(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END)
            AS INST_LATE_PAYMENT_COUNT,
        SUM(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1 ELSE 0 END)
            AS INST_SHORT_PAYMENT_COUNT,
        SUM(CASE WHEN DAYS_INSTALMENT >= -365 AND DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
                 THEN 1 ELSE 0 END) AS INST_RECENT_12M_LATE_COUNT,
        SUM(CASE WHEN DAYS_INSTALMENT >= -730 AND DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
                 THEN 1 ELSE 0 END) AS INST_RECENT_24M_LATE_COUNT,
        SUM(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
            AS INST_LATE_PAYMENT_RATE,
        SUM(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
            AS INST_SHORT_PAYMENT_RATE,
        MIN(-DAYS_INSTALMENT) AS INST_DAYS_SINCE_DUE_MIN,
        MAX(-DAYS_INSTALMENT) AS INST_DAYS_SINCE_DUE_MAX
    FROM {installments}
    GROUP BY SK_ID_CURR
)
SELECT
    ids.SK_ID_CURR,
    CASE WHEN b.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS HAS_BUREAU_HISTORY,
    CASE WHEN p.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS HAS_PREVIOUS_APPLICATION,
    CASE WHEN pos.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS HAS_POS_HISTORY,
    CASE WHEN cc.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS HAS_CREDIT_CARD_HISTORY,
    CASE WHEN inst.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS HAS_INSTALLMENT_HISTORY,
    b.* EXCLUDE (SK_ID_CURR),
    p.* EXCLUDE (SK_ID_CURR),
    pos.* EXCLUDE (SK_ID_CURR),
    cc.* EXCLUDE (SK_ID_CURR),
    inst.* EXCLUDE (SK_ID_CURR)
FROM base_ids ids
LEFT JOIN bureau_features b USING (SK_ID_CURR)
LEFT JOIN previous_features p USING (SK_ID_CURR)
LEFT JOIN pos_features pos USING (SK_ID_CURR)
LEFT JOIN credit_card_features cc USING (SK_ID_CURR)
LEFT JOIN installment_features inst USING (SK_ID_CURR)
ORDER BY ids.SK_ID_CURR
"""


def build_history_features(
    raw_dir: str | Path,
    *,
    output_path: str | Path,
    report_path: str | Path,
    memory_limit: str = "4GB",
    threads: int = 4,
) -> dict:
    """Aggregate all history sources to the application grain and write Parquet."""

    raw = Path(raw_dir)
    missing = [name for name in REQUIRED_FILES if not (raw / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required history inputs: {', '.join(missing)}")

    output = Path(output_path)
    report_output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output.parent.parent / "interim" / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.building{output.suffix}")
    if temporary.exists():
        temporary.unlink()

    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute(f"SET threads = {max(1, threads)}")
        connection.execute(f"SET temp_directory = '{_sql_path(temp_dir)}'")
        query = _feature_query(raw)
        connection.execute(
            f"COPY ({query}) TO '{_sql_path(temporary)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        summary = connection.execute(
            f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT SK_ID_CURR) AS unique_keys,
                COUNT(*) - COUNT(DISTINCT SK_ID_CURR) AS duplicate_keys,
                COUNT(*) FILTER (WHERE HAS_BUREAU_HISTORY = 1) AS bureau_rows,
                COUNT(*) FILTER (WHERE HAS_PREVIOUS_APPLICATION = 1) AS previous_rows,
                COUNT(*) FILTER (WHERE HAS_POS_HISTORY = 1) AS pos_rows,
                COUNT(*) FILTER (WHERE HAS_CREDIT_CARD_HISTORY = 1) AS credit_card_rows,
                COUNT(*) FILTER (WHERE HAS_INSTALLMENT_HISTORY = 1) AS installment_rows
            FROM read_parquet('{_sql_path(temporary)}')
            """
        ).fetchone()
        feature_count = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(temporary)}')"
        ).fetchall()
    finally:
        connection.close()

    if summary is None or summary[0] != summary[1] or summary[2] != 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("history feature output violated the one-row-per-application contract")
    temporary.replace(output)
    row_count = int(summary[0])
    coverage = {
        "bureau": {"rows": int(summary[3]), "rate": float(summary[3] / row_count)},
        "previous_application": {
            "rows": int(summary[4]),
            "rate": float(summary[4] / row_count),
        },
        "pos": {"rows": int(summary[5]), "rate": float(summary[5] / row_count)},
        "credit_card": {"rows": int(summary[6]), "rate": float(summary[6] / row_count)},
        "installments": {"rows": int(summary[7]), "rate": float(summary[7] / row_count)},
    }
    report = {
        "run_version": "1.0.0",
        "dataset": "home-credit-default-risk",
        "grain": "one row per SK_ID_CURR across train and test applications",
        "target_used": False,
        "source_files": list(REQUIRED_FILES),
        "row_count": row_count,
        "unique_keys": int(summary[1]),
        "duplicate_keys": int(summary[2]),
        "column_count": len(feature_count),
        "feature_count_excluding_key": len(feature_count) - 1,
        "source_coverage": coverage,
        "artifact": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "leakage_controls": [
            "TARGET is never selected by the aggregation query.",
            "Every source is aggregated to SK_ID_CURR before joining applications.",
            "Only historical relative-day and relative-month fields are summarized.",
            "The aggregation is identical for training and test application IDs.",
        ],
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
