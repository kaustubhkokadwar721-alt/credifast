"""Versioned provenance metadata for frozen-model intake and integrations."""

from __future__ import annotations

from collections import Counter
from typing import Any

INTAKE_SCHEMA_VERSION = "1.0.0"

CIBIL_EXACT = {"HAS_BUREAU_HISTORY"}
PARTNER_EXACT = {
    "HAS_PREVIOUS_APPLICATION",
    "HAS_POS_HISTORY",
    "HAS_CREDIT_CARD_HISTORY",
    "HAS_INSTALLMENT_HISTORY",
}
DERIVED_EXACT = {
    "ANNUITY_TO_INCOME",
    "APPLICATION_MISSING_COUNT",
    "CREDIT_TO_ANNUITY",
    "CREDIT_TO_GOODS",
    "CREDIT_TO_INCOME",
    "DOWN_PAYMENT_RATE",
    "EMPLOYMENT_SENTINEL",
    "EMPLOYMENT_YEARS",
    "EXT_SOURCE_COUNT",
    "EXT_SOURCE_MAX",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
    "EXT_SOURCE_STD",
    "ID_AGE_YEARS",
    "PHONE_CHANGE_YEARS",
    "REGISTRATION_YEARS",
}

SOURCE_DEFINITIONS = {
    "cibil_report_derived": {
        "label": "CIBIL report-derived",
        "short_label": "CIBIL",
        "integration_status": "not_connected",
        "description": (
            "Generated from bureau accounts, balances, payment history and enquiry records. "
            "The demo uses Kaggle bureau tables, not a live CIBIL response."
        ),
    },
    "partner_ledger_derived": {
        "label": "Lending-partner ledger-derived",
        "short_label": "PARTNER LEDGER",
        "integration_status": "not_connected",
        "description": (
            "Generated from previous applications, POS servicing, cards and installment events "
            "held by a lender or servicing partner."
        ),
    },
    "application_package": {
        "label": "Application and documents",
        "short_label": "APPLICATION",
        "integration_status": "local_record_only",
        "description": (
            "Applicant declarations, requested terms, KYC/contact flags and property or document "
            "fields. These require an application form, document extraction or partner LOS."
        ),
    },
    "computed": {
        "label": "Computed automatically",
        "short_label": "COMPUTED",
        "integration_status": "available",
        "description": (
            "Deterministic ratios, durations, source counts and missingness values calculated from "
            "other fields; never typed by a reviewer."
        ),
    },
    "external_score": {
        "label": "Separate external score",
        "short_label": "EXTERNAL MODEL",
        "integration_status": "provider_required",
        "description": (
            "Home Credit EXT_SOURCE inputs are unspecified external model scores and must not be "
            "renamed or presented as CIBIL scores."
        ),
    },
}

EDITABLE_FIELDS = {
    "annual_income": {
        "label": "Annual income",
        "model_field": "AMT_INCOME_TOTAL",
        "primary_source": "Applicant declaration",
        "source_badges": ["APPLICANT", "BANK / AA VERIFIABLE", "CIBIL REPORTED—CONFIRM"],
        "note": (
            "Use current applicant-declared income; verify recurring credits through a consented "
            "bank or Account Aggregator feed. CIBIL may contain income previously reported by a "
            "member, but it can be stale."
        ),
    },
    "requested_credit": {
        "label": "Requested credit",
        "model_field": "AMT_CREDIT",
        "primary_source": "Applicant request / lender origination system",
        "source_badges": ["APPLICANT REQUEST", "LENDER RECORD"],
        "note": "This is a current requested term, not a bureau or bank-statement field.",
    },
    "annual_annuity": {
        "label": "Annual proposed repayment",
        "model_field": "AMT_ANNUITY",
        "primary_source": "Lender calculation",
        "source_badges": ["LENDER CALCULATED"],
        "note": "Calculate from the proposed facility terms; do not import a historical EMI here.",
    },
    "goods_price": {
        "label": "Goods price",
        "model_field": "AMT_GOODS_PRICE",
        "primary_source": "Merchant quote / application document",
        "source_badges": ["MERCHANT / DOCUMENT", "APPLICANT CONFIRM"],
        "note": "Use the current invoice, quote or declared financed-asset price.",
    },
}

INTAKE_TEMPLATE = {
    "application_id": 176483,
    "annual_income": 157500.0,
    "requested_credit": 440784.0,
    "annual_annuity": 34956.0,
    "goods_price": 360000.0,
}


def classify_model_factor(feature: str) -> str:
    """Return the primary provenance family for a selected frozen-model factor."""

    if feature in CIBIL_EXACT or feature.startswith(
        ("BUREAU_", "BB_", "AMT_REQ_CREDIT_BUREAU_")
    ):
        return "cibil_report_derived"
    if feature in PARTNER_EXACT or feature.startswith(("PREV_", "POS_", "CC_", "INST_")):
        return "partner_ledger_derived"
    if feature in DERIVED_EXACT:
        return "computed"
    if feature.startswith("EXT_SOURCE_"):
        return "external_score"
    return "application_package"


def normalize_intake_package(payload: Any) -> dict[str, int | float]:
    """Validate the JSON intake shape supported by the frozen known-applicant runtime."""

    if not isinstance(payload, dict):
        raise TypeError("intake package must be a JSON object")
    allowed = {"application_id", *EDITABLE_FIELDS}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("unsupported intake fields: " + ", ".join(unknown))
    if "application_id" not in payload:
        raise ValueError("intake package requires application_id")
    application_id = payload["application_id"]
    if isinstance(application_id, bool):
        raise TypeError("application_id must be a positive integer")
    try:
        normalized_id = int(application_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("application_id must be a positive integer") from exc
    if normalized_id <= 0 or normalized_id != float(application_id):
        raise ValueError("application_id must be a positive integer")
    result: dict[str, int | float] = {"application_id": normalized_id}
    for field in EDITABLE_FIELDS:
        if field not in payload or payload[field] is None:
            continue
        value = payload[field]
        if isinstance(value, bool):
            raise TypeError(f"{field} must be a positive number")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a positive number") from exc
        if not 0.0 < numeric < 1_000_000_000.0:
            raise ValueError(f"{field} must be positive and below 1,000,000,000")
        result[field] = numeric
    return result


def build_input_source_schema(selected_features: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Describe how the frozen model's selected factors are assembled."""

    rows = [
        {
            "factor": feature,
            "source": classify_model_factor(feature),
            "source_label": SOURCE_DEFINITIONS[classify_model_factor(feature)]["label"],
        }
        for feature in selected_features
    ]
    counts = Counter(row["source"] for row in rows)
    display_order = (
        "cibil_report_derived",
        "partner_ledger_derived",
        "application_package",
        "computed",
        "external_score",
    )
    source_summary = []
    for source in display_order:
        definition = SOURCE_DEFINITIONS[source]
        source_summary.append(
            {
                "source": source,
                "label": definition["label"],
                "short_label": definition["short_label"],
                "factor_count": counts[source],
                "integration_status": definition["integration_status"],
                "description": definition["description"],
            }
        )
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "selected_factor_count": len(selected_features),
        "source_summary": source_summary,
        "factor_sources": rows,
        "editable_fields": EDITABLE_FIELDS,
        "json_template": INTAKE_TEMPLATE,
        "supported_intake_modes": [
            "curated_local_case",
            "known_local_application_id",
            "known_local_application_json",
        ],
        "current_runtime_limit": (
            "The frozen model can score only known local Home Credit application IDs. The JSON "
            "package changes declared current terms; all other factors are assembled from the "
            "local application row and historical feature store. Live CIBIL, Account Aggregator "
            "and lending-partner connectors are not implemented."
        ),
        "counting_note": (
            "Counts assign each model-ready factor to one primary provenance family and are not "
            "counts of raw API fields. Source overlap is intentionally not double-counted."
        ),
    }
