"""Collectible-input feature policy for the V3 workbook-scored model.

Every feature named here must be obtainable from an application form, a credit information
report, or arithmetic over those two. Features that exist only inside the Home Credit schema
are excluded regardless of predictive value. See ``docs/COLLECTIBLE_INPUT_SPEC.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

POLICY_VERSION = "1.0.0"

# --- Application form -------------------------------------------------------------------

APPLICATION_DECLARED = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "OWN_CAR_AGE",
    "EMPLOYMENT_YEARS",
    "EMPLOYMENT_SENTINEL",
    "REGISTRATION_YEARS",
    "ID_AGE_YEARS",
    "FLAG_EMAIL",
    "FLAG_WORK_PHONE",
)

# --- Deterministic arithmetic over declared terms and bureau totals ---------------------

COMPUTED_RATIOS = (
    "ANNUITY_TO_INCOME",
    "CREDIT_TO_INCOME",
    "CREDIT_TO_ANNUITY",
    "CREDIT_TO_GOODS",
    "DOWN_PAYMENT_RATE",
    "APPLICATION_MISSING_COUNT",
    "TOTAL_OBLIGATION_TO_INCOME",
    "EXTERNAL_DEBT_TO_INCOME",
)

# --- Credit information report ----------------------------------------------------------

CIR_PREFIXES = ("BUREAU_", "BB_", "AMT_REQ_CREDIT_BUREAU_")
CIR_EXACT = ("HAS_BUREAU_HISTORY",)

# Revolving and installment tradeline behaviour. A CIR reports limit, balance and monthly
# delinquency for every account at every lender, so these carry over in meaning. Home Credit's
# versions are own-ledger, which makes them a documented proxy rather than an equivalence.
EXTENDED_PREFIXES = ("CC_", "POS_")
EXTENDED_EXACT = ("HAS_CREDIT_CARD_HISTORY", "HAS_POS_HISTORY")

# Transaction-level detail a credit report does not carry.
EXTENDED_DENY = (
    "CC_DRAWINGS_MEAN",
    "CC_DRAWINGS_SUM",
    "CC_DRAWING_COUNT_SUM",
    "CC_PAYMENT_TOTAL_SUM",
    "CC_RECEIVABLE_MAX",
    "CC_RECEIVABLE_MEAN",
)

EXTERNAL_SCORE_PREFIX = "EXT_SOURCE"


def _matches(feature: str, prefixes: tuple[str, ...], exact: tuple[str, ...]) -> bool:
    return feature in exact or feature.startswith(prefixes)


def is_collectible_core(feature: str) -> bool:
    """Return True when the feature comes from an application form or a credit report."""

    if feature in APPLICATION_DECLARED or feature in COMPUTED_RATIOS:
        return True
    return _matches(feature, CIR_PREFIXES, CIR_EXACT)


def is_collectible_extended(feature: str) -> bool:
    """Return True for the core set plus proxied tradeline behaviour."""

    if is_collectible_core(feature):
        return True
    if feature in EXTENDED_DENY:
        return False
    return _matches(feature, EXTENDED_PREFIXES, EXTENDED_EXACT)


def is_external_score(feature: str) -> bool:
    return feature.startswith(EXTERNAL_SCORE_PREFIX)


VARIANT_FILTERS = {
    "v1_reference": lambda feature: True,
    "v1_no_external_score": lambda feature: not is_external_score(feature),
    "v3_core": is_collectible_core,
    "v3_extended": is_collectible_extended,
}

VARIANT_DESCRIPTIONS = {
    "v1_reference": (
        "Frozen V1 feature policy. Includes EXT_SOURCE and Home-Credit-only fields. "
        "Reference only; not obtainable for a submitted applicant."
    ),
    "v1_no_external_score": (
        "V1 minus the three unnamed external model scores and their aggregates. "
        "The honest baseline for what losing EXT_SOURCE costs."
    ),
    "v3_core": (
        "Application form plus credit information report only. Every input is collectible."
    ),
    "v3_extended": (
        "Core plus revolving and installment tradeline behaviour, proxied from "
        "Home Credit's own ledger."
    ),
}


@dataclass(frozen=True)
class VariantFeatures:
    variant: str
    selected: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    dropped: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "description": VARIANT_DESCRIPTIONS[self.variant],
            "selected_count": len(self.selected),
            "selected": list(self.selected),
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "dropped_count": len(self.dropped),
            "dropped": list(self.dropped),
        }


def restrict_selection(
    variant: str,
    selected: tuple[str, ...] | list[str],
    numeric: tuple[str, ...] | list[str],
    categorical: tuple[str, ...] | list[str],
) -> VariantFeatures:
    """Apply a variant filter to an existing responsible-feature selection."""

    if variant not in VARIANT_FILTERS:
        raise ValueError(f"unknown variant: {variant}")
    keep = VARIANT_FILTERS[variant]
    kept = tuple(feature for feature in selected if keep(feature))
    if not kept:
        raise ValueError(f"variant {variant} selected no features")
    dropped = tuple(feature for feature in selected if not keep(feature))
    kept_set = set(kept)
    return VariantFeatures(
        variant=variant,
        selected=kept,
        numeric=tuple(feature for feature in numeric if feature in kept_set),
        categorical=tuple(feature for feature in categorical if feature in kept_set),
        dropped=dropped,
    )


def add_obligation_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the two affordability ratios that need bureau totals.

    ``TOTAL_OBLIGATION_TO_INCOME`` is post-disbursal FOIR: proposed repayment plus every
    externally reported annuity, over declared income. ``EXTERNAL_DEBT_TO_INCOME`` is
    reported outstanding debt over declared income. Both stay missing when income is absent
    or non-positive; missing history is never read as zero obligation.
    """

    result = frame
    income = result.get("AMT_INCOME_TOTAL")
    if income is None:
        return result
    positive_income = income.astype(float).where(income > 0)

    annuity = result.get("AMT_ANNUITY")
    bureau_annuity = result.get("BUREAU_ANNUITY_SUM")
    if annuity is not None and bureau_annuity is not None:
        # An applicant with no reported external annuity has UNKNOWN external obligation, not
        # zero. FOIR is only meaningful when the external side is actually observed, so the
        # ratio stays missing whenever BUREAU_ANNUITY_SUM is absent. Treating absence as zero
        # would understate obligation for exactly the thin-file cases that need care most.
        total_obligation = annuity.astype(float).add(
            bureau_annuity.astype(float), fill_value=0.0
        )
        result["TOTAL_OBLIGATION_TO_INCOME"] = total_obligation.where(
            bureau_annuity.notna()
        ).div(positive_income)

    bureau_debt = result.get("BUREAU_DEBT_SUM")
    if bureau_debt is not None:
        result["EXTERNAL_DEBT_TO_INCOME"] = bureau_debt.astype(float).div(positive_income)
    return result


def recompute_missing_count(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Count missing values across the variant's own columns only.

    The V1 ``APPLICATION_MISSING_COUNT`` counts nulls across every merged column, which would
    leak the shape of dropped Home-Credit-only fields into a restricted model. Each variant
    recomputes it over the columns it actually consumes.
    """

    present = [column for column in columns if column in frame.columns]
    return frame[present].isna().sum(axis=1).astype("int16")
