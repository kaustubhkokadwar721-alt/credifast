"""Application service joining validation, risk scoring, and policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .domain import ApplicantProfile
from .policy import DecisionPolicy, apply_policy
from .scoring import REASON_CATALOGUE, score_profile


def evaluate_application(
    values: ApplicantProfile | Mapping[str, Any],
    policy: DecisionPolicy | None = None,
) -> dict[str, Any]:
    profile = (
        values if isinstance(values, ApplicantProfile) else ApplicantProfile.from_mapping(values)
    )
    risk = score_profile(profile)
    decision = apply_policy(profile, risk, policy)

    return {
        "application_id": profile.application_id,
        "risk": {
            "repayment_difficulty_probability": round(risk.probability, 6),
            "credit_score": risk.credit_score,
            "risk_grade": risk.risk_grade,
            "adverse_reasons": [
                {"code": code, "description": REASON_CATALOGUE[code]}
                for code in risk.adverse_reason_codes
            ],
            "favorable_reasons": [
                {"code": code, "description": REASON_CATALOGUE[code]}
                for code in risk.favorable_reason_codes
            ],
        },
        "decision": {
            "recommendation": decision.recommendation,
            "confidence": decision.confidence,
            "decision_reasons": list(decision.decision_reasons),
        },
        "affordability": {
            "status": decision.affordability_status,
            "debt_service_ratio": round(decision.debt_service_ratio, 4),
            "estimated_maximum_amount": round(decision.estimated_maximum_amount, 2),
        },
        "data_quality": {
            "coverage": profile.data_coverage,
            "scoring_path": (
                "FULL_FILE"
                if profile.data_coverage >= 0.85 and (profile.bureau_history_months or 0) >= 12
                else "THIN_FILE_DEMO"
            ),
        },
        "versions": {
            "model_name": risk.model_name,
            "model_version": risk.model_version,
            "policy_version": decision.policy_version,
        },
        "disclaimer": (
            "Hackathon decision-support output from a deterministic demo baseline; "
            "not a trained or production-approved lending model."
        ),
    }
