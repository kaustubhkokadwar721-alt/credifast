"""Temporal and credit-cycling history features for challenger experiments.

This module writes only additive V2 features. The frozen V1 feature store remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb

from .history_features import REQUIRED_FILES, _source, _sql_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _temporal_feature_query(raw_dir: Path) -> str:
    app_train = _source(raw_dir / "application_train.csv")
    app_test = _source(raw_dir / "application_test.csv")
    bureau = _source(raw_dir / "bureau.csv")
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
installment_events AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        NUM_INSTALMENT_NUMBER,
        MAX(DAYS_INSTALMENT) AS DUE_DAY,
        MAX(DAYS_ENTRY_PAYMENT) AS PAYMENT_DAY,
        MAX(AMT_INSTALMENT) AS DUE_AMOUNT,
        SUM(AMT_PAYMENT) AS PAID_AMOUNT,
        GREATEST(MAX(DAYS_ENTRY_PAYMENT) - MAX(DAYS_INSTALMENT), 0) AS LATE_DAYS,
        GREATEST(MAX(AMT_INSTALMENT) - SUM(AMT_PAYMENT), 0) AS SHORTFALL,
        CASE WHEN MAX(DAYS_ENTRY_PAYMENT) > MAX(DAYS_INSTALMENT) THEN 1 ELSE 0 END AS IS_LATE,
        CASE WHEN SUM(AMT_PAYMENT) < MAX(AMT_INSTALMENT) THEN 1 ELSE 0 END AS IS_SHORT,
        CASE WHEN MAX(DAYS_ENTRY_PAYMENT) <= MAX(DAYS_INSTALMENT)
                  AND SUM(AMT_PAYMENT) >= MAX(AMT_INSTALMENT) THEN 1 ELSE 0 END AS IS_ON_TIME
    FROM {installments}
    GROUP BY SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_NUMBER
),
installment_sequence AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_PREV ORDER BY DUE_DAY
        ) - ROW_NUMBER() OVER (
            PARTITION BY SK_ID_PREV, IS_LATE ORDER BY DUE_DAY
        ) AS LATE_RUN_GROUP,
        SUM(IS_LATE) OVER (
            PARTITION BY SK_ID_PREV ORDER BY DUE_DAY DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS RECENT_LATE_SEEN,
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_PREV ORDER BY DUE_DAY DESC
        ) AS RECENT_EVENT_NUMBER,
        LAG(IS_LATE) OVER (PARTITION BY SK_ID_PREV ORDER BY DUE_DAY) AS PRIOR_IS_LATE
    FROM installment_events
),
installment_late_runs AS (
    SELECT SK_ID_PREV, LATE_RUN_GROUP, COUNT(*) AS RUN_LENGTH
    FROM installment_sequence
    WHERE IS_LATE = 1
    GROUP BY SK_ID_PREV, LATE_RUN_GROUP
),
installment_loan_features AS (
    SELECT
        e.SK_ID_CURR,
        e.SK_ID_PREV,
        SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 90.0) * e.IS_LATE)
            / NULLIF(SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 90.0)), 0)
            AS TINST_LOAN_WEIGHTED_LATE_RATE_90D,
        SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 180.0) * e.IS_LATE)
            / NULLIF(SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 180.0)), 0)
            AS TINST_LOAN_WEIGHTED_LATE_RATE_180D,
        SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 365.0) * e.IS_LATE)
            / NULLIF(SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 365.0)), 0)
            AS TINST_LOAN_WEIGHTED_LATE_RATE_365D,
        SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 90.0) * e.SHORTFALL)
            / NULLIF(SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 90.0) * e.DUE_AMOUNT), 0)
            AS TINST_LOAN_WEIGHTED_SHORTFALL_RATE_90D,
        SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 180.0) * e.SHORTFALL)
            / NULLIF(SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 180.0) * e.DUE_AMOUNT), 0)
            AS TINST_LOAN_WEIGHTED_SHORTFALL_RATE_180D,
        SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 365.0) * e.SHORTFALL)
            / NULLIF(SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 365.0) * e.DUE_AMOUNT), 0)
            AS TINST_LOAN_WEIGHTED_SHORTFALL_RATE_365D,
        SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 180.0) * e.LATE_DAYS)
            / NULLIF(SUM(EXP(-LN(2.0) * (-e.DUE_DAY) / 180.0)), 0)
            AS TINST_LOAN_WEIGHTED_LATE_DAYS_180D,
        SUM(CASE WHEN e.IS_ON_TIME = 1 AND e.RECENT_LATE_SEEN = 0 THEN 1 ELSE 0 END)
            AS TINST_LOAN_CURRENT_ON_TIME_STREAK,
        SUM(CASE WHEN e.IS_LATE = 1 AND e.RECENT_LATE_SEEN = e.RECENT_EVENT_NUMBER
                 THEN 1 ELSE 0 END) AS TINST_LOAN_CURRENT_LATE_STREAK,
        SUM(CASE WHEN e.PRIOR_IS_LATE = 1 AND e.IS_LATE = 0 THEN 1 ELSE 0 END)
            AS TINST_LOAN_CURE_COUNT,
        SUM(CASE WHEN e.PRIOR_IS_LATE = 0 AND e.IS_LATE = 1 THEN 1 ELSE 0 END)
            AS TINST_LOAN_RELAPSE_COUNT,
        MIN(CASE WHEN e.IS_LATE = 1 THEN -e.DUE_DAY END) AS TINST_LOAN_DAYS_SINCE_LATE,
        COALESCE((
            SELECT MAX(r.RUN_LENGTH)
            FROM installment_late_runs r
            WHERE r.SK_ID_PREV = e.SK_ID_PREV
        ), 0) AS TINST_LOAN_MAX_CONSECUTIVE_LATE
    FROM installment_sequence e
    GROUP BY e.SK_ID_CURR, e.SK_ID_PREV
),
installment_customer_windows AS (
    SELECT
        SK_ID_CURR,
        AVG(CASE WHEN -DUE_DAY <= 90 THEN IS_LATE END) AS TINST_LATE_RATE_90D,
        AVG(CASE WHEN -DUE_DAY > 90 AND -DUE_DAY <= 365 THEN IS_LATE END)
            AS TINST_LATE_RATE_91_365D,
        AVG(CASE WHEN -DUE_DAY <= 180 THEN IS_LATE END) AS TINST_LATE_RATE_180D,
        AVG(CASE WHEN -DUE_DAY > 180 AND -DUE_DAY <= 730 THEN IS_LATE END)
            AS TINST_LATE_RATE_181_730D,
        AVG(CASE WHEN -DUE_DAY <= 365 THEN IS_SHORT END) AS TINST_SHORT_RATE_365D,
        SUM(CASE WHEN -DUE_DAY <= 90 THEN SHORTFALL END)
            / NULLIF(SUM(CASE WHEN -DUE_DAY <= 90 THEN DUE_AMOUNT END), 0)
            AS TINST_SHORTFALL_RATE_90D,
        SUM(CASE WHEN -DUE_DAY <= 365 THEN SHORTFALL END)
            / NULLIF(SUM(CASE WHEN -DUE_DAY <= 365 THEN DUE_AMOUNT END), 0)
            AS TINST_SHORTFALL_RATE_365D,
        AVG(CASE WHEN -DUE_DAY <= 90 THEN LATE_DAYS END) AS TINST_LATE_DAYS_MEAN_90D,
        MAX(CASE WHEN -DUE_DAY <= 365 THEN LATE_DAYS END) AS TINST_LATE_DAYS_MAX_365D,
        SUM(CASE WHEN -DUE_DAY <= 30 THEN 1 ELSE 0 END) AS TINST_EVENT_COUNT_30D,
        SUM(CASE WHEN -DUE_DAY <= 90 THEN 1 ELSE 0 END) AS TINST_EVENT_COUNT_90D,
        SUM(CASE WHEN -DUE_DAY <= 365 THEN 1 ELSE 0 END) AS TINST_EVENT_COUNT_365D,
        SUM(CASE WHEN -DUE_DAY <= 90 AND IS_LATE = 1 THEN 1 ELSE 0 END)
            AS TINST_LATE_COUNT_90D,
        SUM(CASE WHEN -DUE_DAY <= 90 AND IS_SHORT = 1 THEN 1 ELSE 0 END)
            AS TINST_SHORT_COUNT_90D,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 1 THEN IS_LATE END) AS TINST_RECENT_EVENT_1_LATE,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 1 THEN IS_SHORT END) AS TINST_RECENT_EVENT_1_SHORT,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 1 THEN LATE_DAYS END) AS TINST_RECENT_EVENT_1_LATE_DAYS,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 1 THEN PAID_AMOUNT / NULLIF(DUE_AMOUNT, 0) END)
            AS TINST_RECENT_EVENT_1_PAYMENT_RATIO,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 2 THEN IS_LATE END) AS TINST_RECENT_EVENT_2_LATE,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 2 THEN IS_SHORT END) AS TINST_RECENT_EVENT_2_SHORT,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 2 THEN LATE_DAYS END) AS TINST_RECENT_EVENT_2_LATE_DAYS,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 2 THEN PAID_AMOUNT / NULLIF(DUE_AMOUNT, 0) END)
            AS TINST_RECENT_EVENT_2_PAYMENT_RATIO,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 3 THEN IS_LATE END) AS TINST_RECENT_EVENT_3_LATE,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 3 THEN IS_SHORT END) AS TINST_RECENT_EVENT_3_SHORT,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 3 THEN LATE_DAYS END) AS TINST_RECENT_EVENT_3_LATE_DAYS,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 3 THEN PAID_AMOUNT / NULLIF(DUE_AMOUNT, 0) END)
            AS TINST_RECENT_EVENT_3_PAYMENT_RATIO,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 4 THEN IS_LATE END) AS TINST_RECENT_EVENT_4_LATE,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 4 THEN IS_SHORT END) AS TINST_RECENT_EVENT_4_SHORT,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 4 THEN LATE_DAYS END) AS TINST_RECENT_EVENT_4_LATE_DAYS,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 4 THEN PAID_AMOUNT / NULLIF(DUE_AMOUNT, 0) END)
            AS TINST_RECENT_EVENT_4_PAYMENT_RATIO,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 5 THEN IS_LATE END) AS TINST_RECENT_EVENT_5_LATE,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 5 THEN IS_SHORT END) AS TINST_RECENT_EVENT_5_SHORT,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 5 THEN LATE_DAYS END) AS TINST_RECENT_EVENT_5_LATE_DAYS,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 5 THEN PAID_AMOUNT / NULLIF(DUE_AMOUNT, 0) END)
            AS TINST_RECENT_EVENT_5_PAYMENT_RATIO,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 6 THEN IS_LATE END) AS TINST_RECENT_EVENT_6_LATE,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 6 THEN IS_SHORT END) AS TINST_RECENT_EVENT_6_SHORT,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 6 THEN LATE_DAYS END) AS TINST_RECENT_EVENT_6_LATE_DAYS,
        MAX(CASE WHEN RECENT_EVENT_NUMBER = 6 THEN PAID_AMOUNT / NULLIF(DUE_AMOUNT, 0) END)
            AS TINST_RECENT_EVENT_6_PAYMENT_RATIO,
        MIN(CASE WHEN IS_LATE = 1 THEN -DUE_DAY END) AS TINST_DAYS_SINCE_LAST_LATE,
        MIN(CASE WHEN IS_SHORT = 1 THEN -DUE_DAY END) AS TINST_DAYS_SINCE_LAST_SHORT
    FROM installment_sequence
    GROUP BY SK_ID_CURR
),
installment_features AS (
    SELECT
        l.SK_ID_CURR,
        AVG(l.TINST_LOAN_WEIGHTED_LATE_RATE_90D) AS TINST_WEIGHTED_LATE_RATE_90D,
        AVG(l.TINST_LOAN_WEIGHTED_LATE_RATE_180D) AS TINST_WEIGHTED_LATE_RATE_180D,
        AVG(l.TINST_LOAN_WEIGHTED_LATE_RATE_365D) AS TINST_WEIGHTED_LATE_RATE_365D,
        MAX(l.TINST_LOAN_WEIGHTED_LATE_RATE_180D) AS TINST_WEIGHTED_LATE_RATE_MAX_180D,
        AVG(l.TINST_LOAN_WEIGHTED_SHORTFALL_RATE_90D)
            AS TINST_WEIGHTED_SHORTFALL_RATE_90D,
        AVG(l.TINST_LOAN_WEIGHTED_SHORTFALL_RATE_180D)
            AS TINST_WEIGHTED_SHORTFALL_RATE_180D,
        AVG(l.TINST_LOAN_WEIGHTED_SHORTFALL_RATE_365D)
            AS TINST_WEIGHTED_SHORTFALL_RATE_365D,
        AVG(l.TINST_LOAN_WEIGHTED_LATE_DAYS_180D) AS TINST_WEIGHTED_LATE_DAYS_180D,
        MAX(l.TINST_LOAN_CURRENT_ON_TIME_STREAK) AS TINST_CURRENT_ON_TIME_STREAK_MAX,
        MAX(l.TINST_LOAN_CURRENT_LATE_STREAK) AS TINST_CURRENT_LATE_STREAK_MAX,
        MAX(l.TINST_LOAN_MAX_CONSECUTIVE_LATE) AS TINST_MAX_CONSECUTIVE_LATE,
        SUM(l.TINST_LOAN_CURE_COUNT) AS TINST_CURE_COUNT,
        SUM(l.TINST_LOAN_RELAPSE_COUNT) AS TINST_RELAPSE_COUNT,
        SUM(l.TINST_LOAN_CURE_COUNT)::DOUBLE
            / NULLIF(SUM(l.TINST_LOAN_CURE_COUNT + l.TINST_LOAN_RELAPSE_COUNT), 0)
            AS TINST_CURE_RATE,
        MIN(l.TINST_LOAN_DAYS_SINCE_LATE) AS TINST_DAYS_SINCE_LATE_MIN,
        w.* EXCLUDE (SK_ID_CURR),
        w.TINST_LATE_RATE_90D - w.TINST_LATE_RATE_91_365D AS TINST_LATE_DELTA_90_365D,
        w.TINST_LATE_RATE_180D - w.TINST_LATE_RATE_181_730D AS TINST_LATE_DELTA_180_730D
    FROM installment_loan_features l
    JOIN installment_customer_windows w USING (SK_ID_CURR)
    GROUP BY ALL
),
previous_ordered AS (
    SELECT
        *,
        LAG(DAYS_DECISION) OVER (
            PARTITION BY SK_ID_CURR ORDER BY DAYS_DECISION
        ) AS PRIOR_DECISION_DAY,
        COUNT(*) OVER (
            PARTITION BY SK_ID_CURR ORDER BY DAYS_DECISION
            RANGE BETWEEN 30 PRECEDING AND CURRENT ROW
        ) AS APPLICATIONS_IN_30D
    FROM {previous}
),
previous_features AS (
    SELECT
        SK_ID_CURR,
        SUM(CASE WHEN -DAYS_DECISION <= 30 THEN 1 ELSE 0 END) AS TPREV_APPLICATION_COUNT_30D,
        SUM(CASE WHEN -DAYS_DECISION <= 90 THEN 1 ELSE 0 END) AS TPREV_APPLICATION_COUNT_90D,
        SUM(CASE WHEN -DAYS_DECISION <= 180 THEN 1 ELSE 0 END) AS TPREV_APPLICATION_COUNT_180D,
        SUM(CASE WHEN -DAYS_DECISION <= 365 THEN 1 ELSE 0 END) AS TPREV_APPLICATION_COUNT_365D,
        SUM(CASE WHEN -DAYS_DECISION <= 90 AND NAME_CONTRACT_STATUS = 'Refused'
                 THEN 1 ELSE 0 END) AS TPREV_REFUSED_COUNT_90D,
        SUM(CASE WHEN -DAYS_DECISION <= 180 AND NAME_CONTRACT_STATUS = 'Approved'
                 THEN 1 ELSE 0 END) AS TPREV_APPROVED_COUNT_180D,
        SUM(CASE WHEN -DAYS_DECISION <= 180 AND NAME_CONTRACT_TYPE = 'Cash loans'
                 THEN 1 ELSE 0 END) AS TPREV_CASH_LOAN_COUNT_180D,
        MAX(APPLICATIONS_IN_30D) AS TPREV_APPLICATION_BURST_MAX_30D,
        MIN(ABS(DAYS_DECISION - PRIOR_DECISION_DAY)) AS TPREV_DECISION_GAP_MIN_DAYS,
        AVG(ABS(DAYS_DECISION - PRIOR_DECISION_DAY)) AS TPREV_DECISION_GAP_MEAN_DAYS,
        SUM(CASE WHEN NAME_CASH_LOAN_PURPOSE = 'Payments on other loans'
                 THEN 1 ELSE 0 END) AS TPREV_OTHER_LOAN_PURPOSE_COUNT,
        AVG(CASE WHEN -DAYS_DECISION <= 365 THEN AMT_ANNUITY END)
            AS TPREV_ANNUITY_MEAN_365D,
        SUM(CASE WHEN -DAYS_DECISION <= 365 THEN AMT_CREDIT END)
            AS TPREV_CREDIT_SUM_365D
    FROM previous_ordered
    GROUP BY SK_ID_CURR
),
cash_after_stress AS (
    SELECT
        p.SK_ID_CURR,
        COUNT(DISTINCT p.SK_ID_PREV) AS TPREV_CASH_AFTER_STRESS_90D_COUNT
    FROM {previous} p
    JOIN installment_events i USING (SK_ID_CURR)
    WHERE p.NAME_CONTRACT_TYPE = 'Cash loans'
      AND p.NAME_CONTRACT_STATUS = 'Approved'
      AND (i.IS_LATE = 1 OR i.IS_SHORT = 1)
      AND p.DAYS_DECISION - i.DUE_DAY BETWEEN 0 AND 90
    GROUP BY p.SK_ID_CURR
),
pos_sequence AS (
    SELECT
        *,
        LAG(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END) OVER (
            PARTITION BY SK_ID_PREV ORDER BY MONTHS_BALANCE
        ) AS PRIOR_DPD_FLAG
    FROM {pos}
),
pos_customer_month AS (
    SELECT
        SK_ID_CURR,
        MONTHS_BALANCE,
        COUNT(DISTINCT CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN SK_ID_PREV END)
            AS ACTIVE_CONCURRENCY,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN CNT_INSTALMENT_FUTURE END)
            AS FUTURE_INSTALLMENTS,
        MAX(SK_DPD) AS MAX_DPD
    FROM pos_sequence
    GROUP BY SK_ID_CURR, MONTHS_BALANCE
),
pos_features AS (
    SELECT
        p.SK_ID_CURR,
        MAX(CASE WHEN p.MONTHS_BALANCE >= -12 THEN p.ACTIVE_CONCURRENCY END)
            AS TPOS_ACTIVE_CONCURRENCY_MAX_12M,
        AVG(CASE WHEN p.MONTHS_BALANCE >= -12 THEN p.ACTIVE_CONCURRENCY END)
            AS TPOS_ACTIVE_CONCURRENCY_MEAN_12M,
        MAX(CASE WHEN p.MONTHS_BALANCE >= -6 THEN p.FUTURE_INSTALLMENTS END)
            AS TPOS_FUTURE_INSTALLMENTS_MAX_6M,
        AVG(CASE WHEN p.MONTHS_BALANCE >= -6 THEN p.FUTURE_INSTALLMENTS END)
            AS TPOS_FUTURE_INSTALLMENTS_MEAN_6M,
        CASE WHEN isfinite(
            REGR_SLOPE(p.FUTURE_INSTALLMENTS, p.MONTHS_BALANCE)
                FILTER (WHERE p.MONTHS_BALANCE >= -12)
        ) THEN REGR_SLOPE(p.FUTURE_INSTALLMENTS, p.MONTHS_BALANCE)
                FILTER (WHERE p.MONTHS_BALANCE >= -12)
        END AS TPOS_FUTURE_INSTALLMENTS_SLOPE_12M,
        AVG(CASE WHEN p.MONTHS_BALANCE >= -6 AND p.MAX_DPD > 0 THEN 1 ELSE 0 END)
            AS TPOS_DPD_RATE_6M,
        MAX(CASE WHEN p.MONTHS_BALANCE >= -12 THEN p.MAX_DPD END) AS TPOS_DPD_MAX_12M,
        s.TPOS_DPD_CURE_COUNT,
        s.TPOS_DPD_RELAPSE_COUNT,
        s.TPOS_WEIGHTED_DPD_RATE_3M,
        s.TPOS_WEIGHTED_DPD_RATE_6M,
        s.TPOS_WEIGHTED_DPD_RATE_12M
    FROM pos_customer_month p
    JOIN (
        SELECT
            SK_ID_CURR,
            SUM(CASE WHEN PRIOR_DPD_FLAG = 1 AND SK_DPD = 0 THEN 1 ELSE 0 END)
                AS TPOS_DPD_CURE_COUNT,
            SUM(CASE WHEN PRIOR_DPD_FLAG = 0 AND SK_DPD > 0 THEN 1 ELSE 0 END)
                AS TPOS_DPD_RELAPSE_COUNT,
            SUM(EXP(-LN(2.0) * (-MONTHS_BALANCE) / 3.0) * CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)
                / NULLIF(SUM(EXP(-LN(2.0) * (-MONTHS_BALANCE) / 3.0)), 0)
                AS TPOS_WEIGHTED_DPD_RATE_3M,
            SUM(EXP(-LN(2.0) * (-MONTHS_BALANCE) / 6.0) * CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)
                / NULLIF(SUM(EXP(-LN(2.0) * (-MONTHS_BALANCE) / 6.0)), 0)
                AS TPOS_WEIGHTED_DPD_RATE_6M,
            SUM(EXP(-LN(2.0) * (-MONTHS_BALANCE) / 12.0) * CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)
                / NULLIF(SUM(EXP(-LN(2.0) * (-MONTHS_BALANCE) / 12.0)), 0)
                AS TPOS_WEIGHTED_DPD_RATE_12M
        FROM pos_sequence
        GROUP BY SK_ID_CURR
    ) s USING (SK_ID_CURR)
    GROUP BY ALL
),
credit_card_features AS (
    SELECT
        SK_ID_CURR,
        AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))
            FILTER (WHERE MONTHS_BALANCE >= -3) AS TCC_UTILIZATION_MEAN_3M,
        AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))
            FILTER (WHERE MONTHS_BALANCE >= -6) AS TCC_UTILIZATION_MEAN_6M,
        MAX(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))
            FILTER (WHERE MONTHS_BALANCE >= -12) AS TCC_UTILIZATION_MAX_12M,
        STDDEV_POP(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))
            FILTER (WHERE MONTHS_BALANCE >= -12) AS TCC_UTILIZATION_STD_12M,
        CASE WHEN isfinite(
            REGR_SLOPE(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0), MONTHS_BALANCE)
                FILTER (WHERE MONTHS_BALANCE >= -12)
        ) THEN REGR_SLOPE(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0), MONTHS_BALANCE)
                FILTER (WHERE MONTHS_BALANCE >= -12)
        END AS TCC_UTILIZATION_SLOPE_12M,
        AVG(AMT_PAYMENT_TOTAL_CURRENT / NULLIF(AMT_INST_MIN_REGULARITY, 0))
            FILTER (WHERE MONTHS_BALANCE >= -6) AS TCC_PAYMENT_TO_MINIMUM_MEAN_6M,
        CASE WHEN isfinite(
            REGR_SLOPE(
                AMT_PAYMENT_TOTAL_CURRENT / NULLIF(AMT_INST_MIN_REGULARITY, 0), MONTHS_BALANCE
            ) FILTER (WHERE MONTHS_BALANCE >= -12)
        ) THEN REGR_SLOPE(
                AMT_PAYMENT_TOTAL_CURRENT / NULLIF(AMT_INST_MIN_REGULARITY, 0), MONTHS_BALANCE
            ) FILTER (WHERE MONTHS_BALANCE >= -12)
        END AS TCC_PAYMENT_TO_MINIMUM_SLOPE_12M,
        SUM(AMT_DRAWINGS_ATM_CURRENT) FILTER (WHERE MONTHS_BALANCE >= -6)
            AS TCC_ATM_DRAWINGS_SUM_6M,
        SUM(AMT_DRAWINGS_ATM_CURRENT) FILTER (WHERE MONTHS_BALANCE >= -6)
            / NULLIF(SUM(AMT_PAYMENT_TOTAL_CURRENT) FILTER (WHERE MONTHS_BALANCE >= -6), 0)
            AS TCC_ATM_DRAWING_TO_PAYMENT_6M,
        AVG(CASE WHEN AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0) >= 0.90
                 THEN 1 ELSE 0 END) FILTER (WHERE MONTHS_BALANCE >= -6)
            AS TCC_MAXED_RATE_6M,
        AVG(CASE WHEN AMT_PAYMENT_TOTAL_CURRENT < AMT_INST_MIN_REGULARITY
                 THEN 1 ELSE 0 END) FILTER (WHERE MONTHS_BALANCE >= -6)
            AS TCC_BELOW_MINIMUM_RATE_6M,
        AVG(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END) FILTER (WHERE MONTHS_BALANCE >= -6)
            AS TCC_DPD_RATE_6M,
        MAX(SK_DPD) FILTER (WHERE MONTHS_BALANCE >= -12) AS TCC_DPD_MAX_12M
    FROM {credit_card}
    GROUP BY SK_ID_CURR
),
bureau_features AS (
    SELECT
        SK_ID_CURR,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN AMT_CREDIT_SUM_DEBT END)
            AS TBUREAU_ACTIVE_DEBT_SUM,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN AMT_CREDIT_SUM END)
            AS TBUREAU_ACTIVE_CREDIT_SUM,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN AMT_ANNUITY END)
            AS TBUREAU_ACTIVE_ANNUITY_SUM,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN AMT_CREDIT_SUM_OVERDUE END)
            AS TBUREAU_ACTIVE_OVERDUE_SUM,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN AMT_CREDIT_SUM_DEBT END)
            / NULLIF(SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN AMT_CREDIT_SUM END), 0)
            AS TBUREAU_ACTIVE_DEBT_TO_CREDIT,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active'
                      AND (AMT_CREDIT_SUM_DEBT < 0 OR AMT_CREDIT_SUM_DEBT > AMT_CREDIT_SUM)
                 THEN 1 ELSE 0 END) AS TBUREAU_ACTIVE_DEBT_ANOMALY_COUNT,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' AND DAYS_CREDIT >= -180 THEN 1 ELSE 0 END)
            AS TBUREAU_NEW_ACTIVE_COUNT_180D,
        MIN(CASE WHEN CREDIT_ACTIVE = 'Active' THEN -DAYS_CREDIT END)
            AS TBUREAU_DAYS_SINCE_NEWEST_ACTIVE
    FROM {bureau}
    GROUP BY SK_ID_CURR
),
joined AS (
    SELECT
        ids.SK_ID_CURR,
        CASE WHEN inst.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS HAS_TEMPORAL_INSTALLMENTS,
        CASE WHEN p.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS HAS_TEMPORAL_PREVIOUS,
        CASE WHEN pos.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS HAS_TEMPORAL_POS,
        CASE WHEN cc.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS HAS_TEMPORAL_CREDIT_CARD,
        CASE WHEN b.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS HAS_TEMPORAL_BUREAU,
        inst.* EXCLUDE (SK_ID_CURR),
        p.* EXCLUDE (SK_ID_CURR),
        COALESCE(cash.TPREV_CASH_AFTER_STRESS_90D_COUNT, 0)
            AS TPREV_CASH_AFTER_STRESS_90D_COUNT,
        pos.* EXCLUDE (SK_ID_CURR),
        cc.* EXCLUDE (SK_ID_CURR),
        b.* EXCLUDE (SK_ID_CURR)
    FROM base_ids ids
    LEFT JOIN installment_features inst USING (SK_ID_CURR)
    LEFT JOIN previous_features p USING (SK_ID_CURR)
    LEFT JOIN cash_after_stress cash USING (SK_ID_CURR)
    LEFT JOIN pos_features pos USING (SK_ID_CURR)
    LEFT JOIN credit_card_features cc USING (SK_ID_CURR)
    LEFT JOIN bureau_features b USING (SK_ID_CURR)
),
cycling AS (
    SELECT
        *,
        (CASE WHEN TPREV_APPLICATION_BURST_MAX_30D >= 3 OR TPREV_REFUSED_COUNT_90D >= 2
              THEN 1 ELSE 0 END)
        + (CASE WHEN TPOS_ACTIVE_CONCURRENCY_MAX_12M >= 2 THEN 1 ELSE 0 END)
        + (CASE WHEN TPREV_CASH_AFTER_STRESS_90D_COUNT > 0 THEN 1 ELSE 0 END)
        + (CASE WHEN TCC_MAXED_RATE_6M >= 0.50 OR TCC_BELOW_MINIMUM_RATE_6M >= 0.50
              THEN 1 ELSE 0 END)
        + (CASE WHEN TPREV_OTHER_LOAN_PURPOSE_COUNT > 0 THEN 1 ELSE 0 END)
            AS CYCLING_SIGNAL_FAMILY_COUNT
    FROM joined
)
SELECT
    *,
    CASE WHEN CYCLING_SIGNAL_FAMILY_COUNT >= 2 THEN 1 ELSE 0 END AS CYCLING_MULTI_SIGNAL_FLAG
FROM cycling
ORDER BY SK_ID_CURR
"""


def build_temporal_history_features(
    raw_dir: str | Path,
    *,
    output_path: str | Path,
    report_path: str | Path,
    memory_limit: str = "4GB",
    threads: int = 4,
) -> dict:
    """Build target-free additive temporal features at one row per application."""

    raw = Path(raw_dir)
    missing = [name for name in REQUIRED_FILES if not (raw / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required history inputs: {', '.join(missing)}")
    output = Path(output_path)
    report_output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output.parent.parent / "interim" / "duckdb_temporal_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.building{output.suffix}")
    temporary.unlink(missing_ok=True)

    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute(f"SET threads = {max(1, threads)}")
        connection.execute(f"SET temp_directory = '{_sql_path(temp_dir)}'")
        query = _temporal_feature_query(raw)
        connection.execute(
            f"COPY ({query}) TO '{_sql_path(temporary)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        summary = connection.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(DISTINCT SK_ID_CURR),
                COUNT(*) - COUNT(DISTINCT SK_ID_CURR),
                COUNT(*) FILTER (WHERE SK_ID_CURR IS NULL)
            FROM read_parquet('{_sql_path(temporary)}')
            """
        ).fetchone()
        schema = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(temporary)}')"
        ).fetchall()
    finally:
        connection.close()

    if summary is None or summary[0] != summary[1] or summary[2] or summary[3]:
        temporary.unlink(missing_ok=True)
        raise ValueError("temporal feature output violated the application-grain contract")
    temporary.replace(output)
    columns = [row[0] for row in schema]
    report = {
        "run_version": "2.0.0-research",
        "dataset": "home-credit-default-risk",
        "grain": "one additive temporal row per train or test SK_ID_CURR",
        "target_used": False,
        "holdout_policy": "features may be built for all IDs, but challenger training/evaluation must exclude the sealed holdout",
        "row_count": int(summary[0]),
        "unique_keys": int(summary[1]),
        "duplicate_keys": int(summary[2]),
        "feature_count_excluding_key": len(columns) - 1,
        "feature_groups": {
            "installment_temporal": sum(column.startswith("TINST_") for column in columns),
            "previous_velocity": sum(column.startswith("TPREV_") for column in columns),
            "pos_obligation": sum(column.startswith("TPOS_") for column in columns),
            "credit_card_temporal": sum(column.startswith("TCC_") for column in columns),
            "bureau_burden": sum(column.startswith("TBUREAU_") for column in columns),
            "credit_cycling": sum(column.startswith("CYCLING_") for column in columns),
            "availability": sum(column.startswith("HAS_TEMPORAL_") for column in columns),
        },
        "half_lives": {"installment_days": [90, 180, 365], "monthly_history": [3, 6, 12]},
        "artifact": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "leakage_controls": [
            "TARGET is never selected.",
            "Only non-positive relative event dates are expected and separately validated.",
            "Multiple payment rows are reconciled to installment grain before sequence features.",
            "Every source block is aggregated before the application-grain join.",
            "Credit cycling is labeled as a multi-source stress proxy, not verified EMI funding.",
        ],
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def validate_temporal_history_features(
    feature_path: str | Path,
    raw_dir: str | Path,
    *,
    output_path: str | Path,
) -> dict:
    """Validate temporal grain, bounded rates, dates, counts, and finite values."""

    feature_source = Path(feature_path)
    raw = Path(raw_dir)
    if not feature_source.is_file():
        raise FileNotFoundError(feature_source)
    missing = [name for name in REQUIRED_FILES if not (raw / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required history inputs: {', '.join(missing)}")

    feature_sql = f"read_parquet('{_sql_path(feature_source)}')"
    app_train = _source(raw / "application_train.csv")
    app_test = _source(raw / "application_test.csv")
    connection = duckdb.connect()
    try:
        schema = connection.execute(f"DESCRIBE SELECT * FROM {feature_sql}").fetchall()
        columns = [row[0] for row in schema]
        numeric = [
            row[0]
            for row in schema
            if any(token in row[1].upper() for token in ("INT", "DOUBLE", "FLOAT", "DECIMAL"))
        ]
        cardinality = connection.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(DISTINCT SK_ID_CURR),
                COUNT(*) FILTER (WHERE SK_ID_CURR IS NULL),
                (SELECT COUNT(*) FROM {app_train}) + (SELECT COUNT(*) FROM {app_test})
            FROM {feature_sql}
            """
        ).fetchone()
        rate_columns = [column for column in numeric if "_RATE_" in column or column.endswith("_RATE")]
        rate_violations = {}
        for column in rate_columns:
            minimum, maximum, invalid = connection.execute(
                f"""
                SELECT MIN(\"{column}\"), MAX(\"{column}\"),
                       COUNT(*) FILTER (WHERE \"{column}\" < 0 OR \"{column}\" > 1)
                FROM {feature_sql}
                """
            ).fetchone()
            if invalid:
                rate_violations[column] = {
                    "minimum": None if minimum is None else float(minimum),
                    "maximum": None if maximum is None else float(maximum),
                    "invalid_rows": int(invalid),
                }
        nonnegative_columns = [
            column
            for column in numeric
            if any(token in column for token in ("COUNT", "STREAK", "CONSECUTIVE"))
            or column.endswith("_FLAG")
        ]
        negative_violations = {}
        for column in nonnegative_columns:
            minimum, invalid = connection.execute(
                f"""
                SELECT MIN(\"{column}\"), COUNT(*) FILTER (WHERE \"{column}\" < 0)
                FROM {feature_sql}
                """
            ).fetchone()
            if invalid:
                negative_violations[column] = {
                    "minimum": float(minimum),
                    "invalid_rows": int(invalid),
                }
        finite_expressions = [
            f'SUM(CASE WHEN "{column}" IS NOT NULL AND NOT isfinite("{column}") '
            f"THEN 1 ELSE 0 END)"
            for column in numeric
        ]
        finite_counts = connection.execute(
            f"SELECT {', '.join(finite_expressions)} FROM {feature_sql}"
        ).fetchone()
        non_finite = {
            column: int(finite_counts[index])
            for index, column in enumerate(numeric)
            if finite_counts[index]
        }
        cycling_bounds = connection.execute(
            f"""
            SELECT MIN(CYCLING_SIGNAL_FAMILY_COUNT), MAX(CYCLING_SIGNAL_FAMILY_COUNT),
                   COUNT(*) FILTER (WHERE CYCLING_SIGNAL_FAMILY_COUNT < 0
                                          OR CYCLING_SIGNAL_FAMILY_COUNT > 5),
                   COUNT(*) FILTER (WHERE CYCLING_MULTI_SIGNAL_FLAG NOT IN (0, 1))
            FROM {feature_sql}
            """
        ).fetchone()
        temporal_validity = {
            "installment_due_max": connection.execute(
                f"SELECT MAX(DAYS_INSTALMENT) FROM {_source(raw / 'installments_payments.csv')}"
            ).fetchone()[0],
            "installment_payment_max": connection.execute(
                f"SELECT MAX(DAYS_ENTRY_PAYMENT) FROM {_source(raw / 'installments_payments.csv')}"
            ).fetchone()[0],
            "previous_decision_max": connection.execute(
                f"SELECT MAX(DAYS_DECISION) FROM {_source(raw / 'previous_application.csv')}"
            ).fetchone()[0],
            "pos_month_max": connection.execute(
                f"SELECT MAX(MONTHS_BALANCE) FROM {_source(raw / 'POS_CASH_balance.csv')}"
            ).fetchone()[0],
            "credit_card_month_max": connection.execute(
                f"SELECT MAX(MONTHS_BALANCE) FROM {_source(raw / 'credit_card_balance.csv')}"
            ).fetchone()[0],
            "bureau_credit_day_max": connection.execute(
                f"SELECT MAX(DAYS_CREDIT) FROM {_source(raw / 'bureau.csv')}"
            ).fetchone()[0],
        }
    finally:
        connection.close()

    future_sources = {
        name: value for name, value in temporal_validity.items() if value is not None and value > 0
    }
    findings = []
    if cardinality is None or cardinality[0] != cardinality[1] or cardinality[0] != cardinality[3]:
        findings.append(
            {
                "severity": "critical",
                "check": "application_grain_reconciliation",
                "evidence": cardinality,
            }
        )
    if "TARGET" in columns:
        findings.append({"severity": "critical", "check": "target_leakage"})
    if future_sources:
        findings.append(
            {"severity": "critical", "check": "future_relative_events", "evidence": future_sources}
        )
    if rate_violations:
        findings.append(
            {"severity": "high", "check": "bounded_rates", "evidence": rate_violations}
        )
    if negative_violations:
        findings.append(
            {"severity": "high", "check": "nonnegative_features", "evidence": negative_violations}
        )
    if non_finite:
        findings.append(
            {"severity": "high", "check": "finite_features", "evidence": non_finite}
        )
    if cycling_bounds[2] or cycling_bounds[3]:
        findings.append(
            {"severity": "high", "check": "cycling_bounds", "evidence": cycling_bounds}
        )
    report = {
        "run_version": "2.0.0-research",
        "dataset": feature_source.name,
        "grain": "one additive temporal row per train or test SK_ID_CURR",
        "summary": {
            "rows": int(cardinality[0]),
            "unique_keys": int(cardinality[1]),
            "expected_rows": int(cardinality[3]),
            "feature_count_excluding_key": len(columns) - 1,
            "ready_for_experiment": not findings,
        },
        "checks": {
            "target_absent": "TARGET" not in columns,
            "future_relative_events": future_sources,
            "rate_violations": rate_violations,
            "negative_feature_violations": negative_violations,
            "non_finite_features": non_finite,
            "cycling_bounds": {
                "minimum": int(cycling_bounds[0]),
                "maximum": int(cycling_bounds[1]),
                "invalid_count_rows": int(cycling_bounds[2]),
                "invalid_flag_rows": int(cycling_bounds[3]),
            },
        },
        "findings": findings,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
