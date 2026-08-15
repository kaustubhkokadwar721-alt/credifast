"""Affordability and decision policy, kept separate from risk scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .domain import ApplicantProfile, PolicyAssessment, RiskAssessment


@dataclass(frozen=True)
class DecisionPolicy:
    maximum_debt_service_ratio: float = 0.40
    minimum_data_coverage: float = 0.60
    policy_version: str = "demo-policy-1.0.0"

    @classmethod
    def from_json(cls, path: str | Path) -> DecisionPolicy:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            maximum_debt_service_ratio=float(values["maximum_debt_service_ratio"]),
            minimum_data_coverage=float(values["minimum_data_coverage"]),
            policy_version=str(values["policy_version"]),
        )


def _confidence(profile: ApplicantProfile, risk: RiskAssessment) -> str:
    history = profile.bureau_history_months or 0
    thresholds = (0.02, 0.05, 0.10, 0.20)
    distance_to_threshold = min(abs(risk.probability - value) for value in thresholds)
    if profile.data_coverage < 0.70 or history < 12:
        return "LOW"
    if profile.data_coverage >= 0.85 and history >= 24 and distance_to_threshold >= 0.01:
        return "HIGH"
    return "MEDIUM"


def apply_policy(
    profile: ApplicantProfile,
    risk: RiskAssessment,
    policy: DecisionPolicy | None = None,
) -> PolicyAssessment:
    active_policy = policy or DecisionPolicy()
    monthly_income = profile.annual_income / 12.0
    proposed_monthly_payment = profile.annual_annuity / 12.0
    total_monthly_obligations = profile.existing_monthly_obligations + proposed_monthly_payment
    debt_service_ratio = total_monthly_obligations / monthly_income
    affordable = debt_service_ratio <= active_policy.maximum_debt_service_ratio

    available_payment = max(
        active_policy.maximum_debt_service_ratio * monthly_income
        - profile.existing_monthly_obligations,
        0.0,
    )
    amount_per_annual_payment = profile.requested_amount / profile.annual_annuity
    estimated_maximum_amount = available_payment * 12.0 * amount_per_annual_payment

    reasons: list[str] = []
    if profile.data_coverage < active_policy.minimum_data_coverage:
        recommendation = "MANUAL_REVIEW"
        reasons.append("Critical application or bureau data is incomplete")
    elif risk.risk_grade == "E":
        recommendation = "DECLINE_RECOMMENDATION"
        reasons.append("Estimated repayment risk exceeds the demo policy threshold")
    elif not affordable:
        recommendation = "REDUCE_AMOUNT"
        reasons.append("Requested facility exceeds the configured affordability threshold")
    elif risk.risk_grade in {"A", "B"}:
        recommendation = "APPROVE"
        reasons.append("Risk and affordability checks pass the demo policy")
    else:
        recommendation = "MANUAL_REVIEW"
        reasons.append("Risk grade requires human review under the demo policy")

    confidence = _confidence(profile, risk)
    if confidence == "LOW":
        reasons.append("Confidence is low because history or source coverage is limited")

    return PolicyAssessment(
        recommendation=recommendation,
        affordability_status="PASS" if affordable else "FAIL",
        debt_service_ratio=debt_service_ratio,
        estimated_maximum_amount=max(0.0, estimated_maximum_amount),
        confidence=confidence,
        decision_reasons=tuple(reasons),
        policy_version=active_policy.policy_version,
    )
