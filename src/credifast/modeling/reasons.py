"""Controlled reason-code aggregation for LightGBM TreeSHAP contributions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReasonDefinition:
    code: str
    adverse: str
    favorable: str
    keywords: tuple[str, ...]


REASON_DEFINITIONS = (
    ReasonDefinition(
        "ML01",
        "Prior repayment performance increases estimated repayment risk",
        "Prior repayment performance reduces estimated repayment risk",
        (
            "DPD",
            "DELINQUENT",
            "OVERDUE",
            "BAD_DEBT",
            "DEFAULT_MONTH",
            "LATE_PAYMENT",
            "LATE_DAYS",
            "SHORT_PAYMENT",
            "SHORTFALL",
        ),
    ),
    ReasonDefinition(
        "ML02",
        "Credit utilization and revolving balances increase estimated repayment risk",
        "Credit utilization and revolving balances reduce estimated repayment risk",
        ("UTILIZATION", "CC_BALANCE", "CC_RECEIVABLE", "CC_DRAWING"),
    ),
    ReasonDefinition(
        "ML03",
        "Debt and scheduled repayment burden increase estimated repayment risk",
        "Debt and scheduled repayment burden reduce estimated repayment risk",
        (
            "DEBT",
            "ANNUITY_TO_INCOME",
            "AMT_ANNUITY",
            "PREV_ANNUITY",
            "CREDIT_TO_INCOME",
            "CREDIT_TO_ANNUITY",
            "BUREAU_ANNUITY",
            "INST_DUE",
            "INST_PAID",
            "PAYMENT_RATIO",
            "POS_INSTALLMENT",
            "FUTURE_INSTALLMENT",
        ),
    ),
    ReasonDefinition(
        "ML04",
        "Recent credit activity increases estimated repayment risk",
        "Recent credit activity reduces estimated repayment risk",
        (
            "AMT_REQ_CREDIT_BUREAU",
            "RECENT_12M",
            "RECENT_24M",
            "CREDIT_COUNT",
            "CONTRACT_COUNT",
            "APPLICATION_COUNT",
        ),
    ),
    ReasonDefinition(
        "ML05",
        "Prior credit-application outcomes increase estimated repayment risk",
        "Prior credit-application outcomes reduce estimated repayment risk",
        (
            "PREV_APPROV",
            "PREV_REFUS",
            "PREV_CANCEL",
            "PREV_UNUSED",
            "PREV_HIGH_YIELD",
            "PREV_MIDDLE_YIELD",
        ),
    ),
    ReasonDefinition(
        "ML06",
        "External credit signals increase estimated repayment risk",
        "External credit signals reduce estimated repayment risk",
        ("EXT_SOURCE",),
    ),
    ReasonDefinition(
        "ML07",
        "Limited or missing source information increases model uncertainty or risk",
        "Available source information reduces model uncertainty or risk",
        ("MISSINGINDICATOR", "MISSING_COUNT", "HAS_", "SENTINEL", "UNKNOWN_MONTH"),
    ),
    ReasonDefinition(
        "ML08",
        "Requested credit amount or loan terms increase estimated repayment risk",
        "Requested credit amount or loan terms reduce estimated repayment risk",
        (
            "AMT_CREDIT",
            "AMT_GOODS_PRICE",
            "DOWN_PAYMENT",
            "CREDIT_TO_GOODS",
            "PREV_CREDIT_TO_APPLICATION",
            "NAME_CONTRACT_TYPE",
            "PAYMENT_COUNT",
            "YIELD_GROUP",
        ),
    ),
    ReasonDefinition(
        "ML09",
        "Income and employment signals increase estimated repayment risk",
        "Income and employment signals reduce estimated repayment risk",
        ("AMT_INCOME", "EMPLOYMENT", "NAME_INCOME_TYPE"),
    ),
    ReasonDefinition(
        "ML10",
        "Account age and credit-history depth increase estimated repayment risk",
        "Account age and credit-history depth reduce estimated repayment risk",
        (
            "DAYS_SINCE",
            "MONTHS_OBSERVED",
            "OLDEST_MONTH",
            "LATEST_MONTH",
            "ID_AGE_YEARS",
            "REGISTRATION_YEARS",
            "PHONE_CHANGE_YEARS",
        ),
    ),
    ReasonDefinition(
        "ML11",
        "Verified application characteristics increase estimated repayment risk",
        "Verified application characteristics reduce estimated repayment risk",
        (
            "FLAG_DOCUMENT",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "NAME_EDUCATION_TYPE",
            "NAME_HOUSING_TYPE",
            "FLAG_OWN",
            "APARTMENTS",
            "LIVINGAREA",
            "TOTALAREA",
            "YEARS_BUILD",
            "HOUSETYPE",
            "WALLSMATERIAL",
            "EMERGENCYSTATE",
        ),
    ),
    ReasonDefinition(
        "ML99",
        "Other modeled financial inputs increase estimated repayment risk",
        "Other modeled financial inputs reduce estimated repayment risk",
        (),
    ),
)

REASON_BY_CODE = {definition.code: definition for definition in REASON_DEFINITIONS}


def normalized_feature_name(transformed_name: str) -> str:
    """Remove preprocessing prefixes while preserving missing-indicator identity."""

    return transformed_name.split("__", 1)[-1].upper()


def reason_code_for_feature(transformed_name: str) -> str:
    normalized = normalized_feature_name(transformed_name)
    if normalized.startswith("MISSINGINDICATOR_"):
        return "ML07"
    for definition in REASON_DEFINITIONS[:-1]:
        if any(keyword in normalized for keyword in definition.keywords):
            return definition.code
    return "ML99"


def aggregate_reason_contributions(
    feature_names: list[str] | np.ndarray,
    contributions: np.ndarray,
    *,
    top_k: int = 3,
) -> dict:
    """Aggregate feature-level log-odds contributions into controlled reasons."""

    names = list(feature_names)
    values = np.asarray(contributions, dtype=float)
    if values.ndim != 1 or len(values) != len(names):
        raise ValueError("contributions must be one-dimensional and match feature names")
    grouped_values: dict[str, float] = defaultdict(float)
    grouped_features: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for name, value in zip(names, values, strict=True):
        code = reason_code_for_feature(name)
        grouped_values[code] += float(value)
        grouped_features[code].append((normalized_feature_name(name), float(value)))

    def serialize(code: str, value: float, adverse: bool) -> dict:
        definition = REASON_BY_CODE[code]
        feature_values = sorted(
            grouped_features[code],
            key=lambda item: item[1],
            reverse=adverse,
        )[:3]
        return {
            "code": code,
            "description": definition.adverse if adverse else definition.favorable,
            "log_odds_contribution": round(value, 6),
            "leading_features": [
                {"feature": feature, "contribution": round(feature_value, 6)}
                for feature, feature_value in feature_values
                if (feature_value > 0 if adverse else feature_value < 0)
            ],
        }

    adverse_codes = sorted(
        ((code, value) for code, value in grouped_values.items() if value > 0),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
    favorable_codes = sorted(
        ((code, value) for code, value in grouped_values.items() if value < 0),
        key=lambda item: item[1],
    )[:top_k]
    return {
        "adverse": [serialize(code, value, True) for code, value in adverse_codes],
        "favorable": [serialize(code, value, False) for code, value in favorable_codes],
        "explanation_space": "LightGBM raw log odds before probability calibration",
    }
