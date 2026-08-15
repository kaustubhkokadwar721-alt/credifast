"""Transparent Phase 0 scorer and shared credit-score mapping.

The deterministic scorer validates the application flow. It is not trained ML and
must be replaced by the Phase 3/4 model adapter for evaluated lending results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .domain import ApplicantProfile, RiskAssessment

MODEL_NAME = "transparent-demo-baseline"
MODEL_VERSION = "demo-baseline-1.0.0"


REASON_CATALOGUE = {
    "HC01": "High total repayment obligations relative to income",
    "HC02": "Historical accounts currently show delinquency",
    "HC03": "High revolving-credit utilization",
    "HC04": "Multiple recent credit inquiries",
    "HC05": "Limited usable credit history",
    "HC06": "Requested credit is high relative to annual income",
    "HC07": "Existing outstanding credit is high relative to annual income",
    "HC08": "Employment history is missing or short",
    "HC09": "Critical source coverage is incomplete",
    "PF01": "Repayment obligations are moderate relative to income",
    "PF03": "Revolving-credit utilization is low",
    "PF05": "Credit history is established",
    "PF06": "Requested credit is moderate relative to annual income",
    "PF07": "Existing outstanding credit is moderate relative to income",
    "PF08": "Employment history is established",
    "PF09": "Critical source coverage is strong",
}


@dataclass(frozen=True)
class _Contribution:
    value: float
    adverse_code: str
    favorable_code: str | None = None


def probability_to_score(probability: float) -> int:
    """Map a probability to a 300-850 internal demo score."""

    if not 0 < probability < 1:
        raise ValueError("probability must be strictly between zero and one")
    odds = (1.0 - probability) / probability
    score = round(600.0 + 20.0 * math.log2(odds / 50.0))
    return max(300, min(850, score))


def risk_grade(probability: float) -> str:
    if probability < 0.02:
        return "A"
    if probability < 0.05:
        return "B"
    if probability < 0.10:
        return "C"
    if probability < 0.20:
        return "D"
    return "E"


def _employment_contribution(years: float | None) -> _Contribution:
    if years is None:
        return _Contribution(0.35, "HC08")
    if years < 1:
        return _Contribution(0.28, "HC08")
    return _Contribution(-min(years, 10.0) * 0.025, "HC08", "PF08")


def _history_contribution(months: int | None) -> _Contribution:
    if months is None or months == 0:
        return _Contribution(0.35, "HC05")
    if months < 12:
        return _Contribution(0.22, "HC05")
    if months >= 60:
        return _Contribution(-0.14, "HC05", "PF05")
    return _Contribution(-0.05, "HC05", "PF05")


def score_profile(profile: ApplicantProfile) -> RiskAssessment:
    """Return a transparent, deterministic risk estimate for integration testing."""

    monthly_income = profile.annual_income / 12.0
    proposed_monthly_payment = profile.annual_annuity / 12.0
    debt_service_ratio = (
        profile.existing_monthly_obligations + proposed_monthly_payment
    ) / monthly_income
    credit_to_income = profile.requested_amount / profile.annual_income
    outstanding_to_income = profile.total_outstanding / profile.annual_income
    utilization = profile.credit_utilization if profile.credit_utilization is not None else 0.65

    contributions = [
        _Contribution((debt_service_ratio - 0.30) * 3.2, "HC01", "PF01"),
        _Contribution((credit_to_income - 0.40) * 0.9, "HC06", "PF06"),
        _Contribution((outstanding_to_income - 0.30) * 0.55, "HC07", "PF07"),
        _Contribution((utilization - 0.35) * 1.1, "HC03", "PF03"),
        _Contribution(profile.delinquent_accounts * 0.72, "HC02"),
        _Contribution(max(profile.recent_inquiries - 1, 0) * 0.18, "HC04"),
        _employment_contribution(profile.employment_years),
        _history_contribution(profile.bureau_history_months),
        _Contribution(max(0.85 - profile.data_coverage, 0) * 1.5, "HC09", "PF09"),
    ]

    log_odds = -3.0 + sum(item.value for item in contributions)
    probability = 1.0 / (1.0 + math.exp(-log_odds))
    probability = max(0.005, min(0.80, probability))

    adverse = tuple(
        item.adverse_code
        for item in sorted(contributions, key=lambda item: item.value, reverse=True)
        if item.value >= 0.05
    )[:3]
    favorable = tuple(
        item.favorable_code
        for item in sorted(contributions, key=lambda item: item.value)
        if item.value <= -0.05 and item.favorable_code is not None
    )[:3]

    return RiskAssessment(
        probability=probability,
        credit_score=probability_to_score(probability),
        risk_grade=risk_grade(probability),
        adverse_reason_codes=adverse,
        favorable_reason_codes=favorable,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )
