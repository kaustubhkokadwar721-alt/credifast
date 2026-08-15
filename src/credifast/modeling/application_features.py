"""Application-table feature engineering and responsible feature policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

IDENTIFIER_AND_TARGET = {"SK_ID_CURR", "TARGET"}
AUDIT_ONLY = {
    "CODE_GENDER",
    "DAYS_BIRTH",
}
PROXY_EXCLUDED = {
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "NAME_FAMILY_STATUS",
    "NAME_TYPE_SUITE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
}
PROXY_PREFIXES = (
    "REGION_",
    "REG_",
    "LIVE_REGION_",
    "REG_CITY_",
    "LIVE_CITY_",
)


@dataclass(frozen=True)
class FeatureSelection:
    selected: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    excluded_by_policy: tuple[str, ...]
    excluded_high_missingness: tuple[str, ...]
    excluded_constant: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "selected": list(self.selected),
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "excluded_by_policy": list(self.excluded_by_policy),
            "excluded_high_missingness": list(self.excluded_high_missingness),
            "excluded_constant": list(self.excluded_constant),
        }


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).div(denominator.astype(float).where(denominator > 0))


def engineer_application_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, target-free transformations to application rows."""

    result = frame.copy()
    if "DAYS_EMPLOYED" in result:
        sentinel = result["DAYS_EMPLOYED"].eq(365243)
        result["EMPLOYMENT_SENTINEL"] = sentinel.astype("int8")
        cleaned_employment = result["DAYS_EMPLOYED"].mask(sentinel)
        result["EMPLOYMENT_YEARS"] = -cleaned_employment / 365.25
        result = result.drop(columns=["DAYS_EMPLOYED"])

    for source, output in (
        ("DAYS_REGISTRATION", "REGISTRATION_YEARS"),
        ("DAYS_ID_PUBLISH", "ID_AGE_YEARS"),
        ("DAYS_LAST_PHONE_CHANGE", "PHONE_CHANGE_YEARS"),
    ):
        if source in result:
            result[output] = -result[source] / 365.25
            result = result.drop(columns=[source])

    required_for_ratios = {
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    }
    if required_for_ratios.issubset(result.columns):
        result["CREDIT_TO_INCOME"] = _safe_ratio(result["AMT_CREDIT"], result["AMT_INCOME_TOTAL"])
        result["ANNUITY_TO_INCOME"] = _safe_ratio(result["AMT_ANNUITY"], result["AMT_INCOME_TOTAL"])
        result["CREDIT_TO_ANNUITY"] = _safe_ratio(result["AMT_CREDIT"], result["AMT_ANNUITY"])
        result["CREDIT_TO_GOODS"] = _safe_ratio(result["AMT_CREDIT"], result["AMT_GOODS_PRICE"])
        result["DOWN_PAYMENT_RATE"] = (result["AMT_GOODS_PRICE"] - result["AMT_CREDIT"]).div(
            result["AMT_GOODS_PRICE"].where(result["AMT_GOODS_PRICE"] > 0)
        )

    external_sources = [
        column for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if column in result
    ]
    if external_sources:
        external = result[external_sources]
        result["EXT_SOURCE_MEAN"] = external.mean(axis=1, skipna=True)
        result["EXT_SOURCE_MIN"] = external.min(axis=1, skipna=True)
        result["EXT_SOURCE_MAX"] = external.max(axis=1, skipna=True)
        result["EXT_SOURCE_STD"] = external.std(axis=1, skipna=True)
        result["EXT_SOURCE_COUNT"] = external.notna().sum(axis=1).astype("int8")

    result["APPLICATION_MISSING_COUNT"] = result.isna().sum(axis=1).astype("int16")
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def _policy_excluded(column: str) -> bool:
    return (
        column in IDENTIFIER_AND_TARGET
        or column in AUDIT_ONLY
        or column in PROXY_EXCLUDED
        or column.startswith(PROXY_PREFIXES)
    )


def select_responsible_features(
    training_frame: pd.DataFrame,
    *,
    maximum_missing_rate: float = 0.75,
) -> FeatureSelection:
    """Select candidate features using only the development partition."""

    excluded_by_policy = sorted(column for column in training_frame if _policy_excluded(column))
    candidates = [column for column in training_frame if column not in excluded_by_policy]
    missing_rates = training_frame[candidates].isna().mean()
    excluded_high_missingness = sorted(
        column for column in candidates if missing_rates[column] > maximum_missing_rate
    )
    candidates = [column for column in candidates if column not in excluded_high_missingness]
    excluded_constant = sorted(
        column for column in candidates if training_frame[column].nunique(dropna=False) <= 1
    )
    selected = sorted(column for column in candidates if column not in excluded_constant)
    categorical = sorted(
        column
        for column in selected
        if isinstance(training_frame[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(training_frame[column].dtype)
        or pd.api.types.is_string_dtype(training_frame[column].dtype)
    )
    numeric = sorted(column for column in selected if column not in categorical)
    return FeatureSelection(
        selected=tuple(selected),
        numeric=tuple(numeric),
        categorical=tuple(categorical),
        excluded_by_policy=tuple(excluded_by_policy),
        excluded_high_missingness=tuple(excluded_high_missingness),
        excluded_constant=tuple(excluded_constant),
    )
