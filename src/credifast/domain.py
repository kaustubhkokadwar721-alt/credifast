"""Framework-independent domain contracts for CrediFast."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _optional_number(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _number(name, value)


@dataclass(frozen=True)
class ApplicantProfile:
    """Validated input used by both the demo scorer and future ML scorer."""

    application_id: str
    requested_amount: float
    annual_income: float
    annual_annuity: float
    family_size: int
    employment_years: float | None
    existing_monthly_obligations: float
    active_accounts: int
    total_outstanding: float
    credit_utilization: float | None
    recent_inquiries: int
    delinquent_accounts: int
    bureau_history_months: int | None
    data_coverage: float

    def __post_init__(self) -> None:
        if not isinstance(self.application_id, str) or not self.application_id.strip():
            raise ValueError("application_id must be a non-empty string")
        if self.requested_amount <= 0:
            raise ValueError("requested_amount must be greater than zero")
        if self.annual_income <= 0:
            raise ValueError("annual_income must be greater than zero")
        if self.annual_annuity <= 0:
            raise ValueError("annual_annuity must be greater than zero")
        if self.family_size < 1:
            raise ValueError("family_size must be at least one")
        if self.employment_years is not None and self.employment_years < 0:
            raise ValueError("employment_years cannot be negative")
        if self.existing_monthly_obligations < 0:
            raise ValueError("existing_monthly_obligations cannot be negative")
        if self.active_accounts < 0:
            raise ValueError("active_accounts cannot be negative")
        if self.total_outstanding < 0:
            raise ValueError("total_outstanding cannot be negative")
        if self.credit_utilization is not None and not 0 <= self.credit_utilization <= 5:
            raise ValueError("credit_utilization must be between 0 and 5")
        if self.recent_inquiries < 0:
            raise ValueError("recent_inquiries cannot be negative")
        if self.delinquent_accounts < 0:
            raise ValueError("delinquent_accounts cannot be negative")
        if self.bureau_history_months is not None and self.bureau_history_months < 0:
            raise ValueError("bureau_history_months cannot be negative")
        if not 0 <= self.data_coverage <= 1:
            raise ValueError("data_coverage must be between 0 and 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ApplicantProfile:
        """Build a validated profile from JSON-like input."""

        required = {
            "application_id",
            "requested_amount",
            "annual_income",
            "annual_annuity",
            "family_size",
            "employment_years",
            "existing_monthly_obligations",
            "active_accounts",
            "total_outstanding",
            "credit_utilization",
            "recent_inquiries",
            "delinquent_accounts",
            "bureau_history_months",
            "data_coverage",
        }
        missing = sorted(required - values.keys())
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")

        integer_fields = (
            "family_size",
            "active_accounts",
            "recent_inquiries",
            "delinquent_accounts",
        )
        converted_integers: dict[str, int] = {}
        for name in integer_fields:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            converted_integers[name] = value

        history = values["bureau_history_months"]
        if history is not None and (isinstance(history, bool) or not isinstance(history, int)):
            raise ValueError("bureau_history_months must be an integer or null")

        return cls(
            application_id=str(values["application_id"]),
            requested_amount=_number("requested_amount", values["requested_amount"]),
            annual_income=_number("annual_income", values["annual_income"]),
            annual_annuity=_number("annual_annuity", values["annual_annuity"]),
            family_size=converted_integers["family_size"],
            employment_years=_optional_number("employment_years", values["employment_years"]),
            existing_monthly_obligations=_number(
                "existing_monthly_obligations", values["existing_monthly_obligations"]
            ),
            active_accounts=converted_integers["active_accounts"],
            total_outstanding=_number("total_outstanding", values["total_outstanding"]),
            credit_utilization=_optional_number("credit_utilization", values["credit_utilization"]),
            recent_inquiries=converted_integers["recent_inquiries"],
            delinquent_accounts=converted_integers["delinquent_accounts"],
            bureau_history_months=history,
            data_coverage=_number("data_coverage", values["data_coverage"]),
        )


@dataclass(frozen=True)
class RiskAssessment:
    probability: float
    credit_score: int
    risk_grade: str
    adverse_reason_codes: tuple[str, ...]
    favorable_reason_codes: tuple[str, ...]
    model_name: str
    model_version: str


@dataclass(frozen=True)
class PolicyAssessment:
    recommendation: str
    affordability_status: str
    debt_service_ratio: float
    estimated_maximum_amount: float
    confidence: str
    decision_reasons: tuple[str, ...]
    policy_version: str
