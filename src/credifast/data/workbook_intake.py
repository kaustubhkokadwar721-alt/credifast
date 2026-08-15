"""Build model-ready features from a user-supplied intake workbook.

The aggregation here must match ``credifast.data.history_features`` exactly, feature by
feature, or the served row will not mean what the trained model learned. Every definition
below is a direct transcription of the corresponding DuckDB expression in that module.

Sheet schema is documented in ``docs/COLLECTIBLE_INPUT_SPEC.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

INTAKE_SCHEMA_VERSION = "1.1.0"

APPLICATION_SHEET = "application"
TRADELINES_SHEET = "tradelines"
DPD_SHEET = "dpd_history"
ACCOUNT_MONTHS_SHEET = "account_months"
ENQUIRIES_SHEET = "enquiries"
BANK_SHEET = "bank_statement"

REQUIRED_SHEETS = (APPLICATION_SHEET,)
OPTIONAL_SHEETS = (
    TRADELINES_SHEET,
    DPD_SHEET,
    ACCOUNT_MONTHS_SHEET,
    ENQUIRIES_SHEET,
    BANK_SHEET,
)

# ``dpd_history`` and ``account_months`` overlap in a real credit report, which reports one
# monthly section per account. They stay separate here because the model was trained on two
# distinct source families -- bureau monthly status codes (BB_*) and servicer monthly detail
# (CC_*/POS_*) -- and merging them at intake would change what the features mean.
REVOLVING_KIND = "revolving"
INSTALLMENT_KIND = "installment"

# Workbook column -> model column. Anything not listed is reviewer evidence, not a model input.
APPLICATION_COLUMN_MAP = {
    "annual_income": "AMT_INCOME_TOTAL",
    "requested_credit": "AMT_CREDIT",
    "annual_annuity": "AMT_ANNUITY",
    "goods_price": "AMT_GOODS_PRICE",
    "contract_type": "NAME_CONTRACT_TYPE",
    "income_type": "NAME_INCOME_TYPE",
    "education_type": "NAME_EDUCATION_TYPE",
    "housing_type": "NAME_HOUSING_TYPE",
    "owns_car": "FLAG_OWN_CAR",
    "owns_realty": "FLAG_OWN_REALTY",
    "car_age_years": "OWN_CAR_AGE",
    "employment_years": "EMPLOYMENT_YEARS",
    "address_vintage_years": "REGISTRATION_YEARS",
    "id_document_age_years": "ID_AGE_YEARS",
    "has_email": "FLAG_EMAIL",
    "has_work_phone": "FLAG_WORK_PHONE",
}

REQUIRED_APPLICATION_COLUMNS = (
    "annual_income",
    "requested_credit",
    "annual_annuity",
    "contract_type",
    "income_type",
    "education_type",
    "housing_type",
    "owns_car",
    "owns_realty",
    "employment_years",
)

YES_VALUES = {"y", "yes", "true", "1", "t"}
NO_VALUES = {"n", "no", "false", "0", "f"}

TRADELINE_STATUSES = ("Active", "Closed", "Sold", "Bad debt")
DELINQUENT_STATUSES = ("1", "2", "3", "4", "5")
SEVERE_STATUSES = ("3", "4", "5")

# Enquiry bands in days, as (exclusive lower, inclusive upper].
#
# These are DISJOINT, not nested. Verified against application_train.csv: applicant 100059
# carries WEEK=1 with MON=0 and QRT=0, which is impossible under cumulative windows, and
# AMT_REQ_CREDIT_BUREAU_QRT >= AMT_REQ_CREDIT_BUREAU_MON holds for only 227,480 of 265,992
# non-null rows. Each field counts enquiries falling in its own period alone.
ENQUIRY_BANDS = {
    "AMT_REQ_CREDIT_BUREAU_HOUR": (-0.5, 0.0),
    "AMT_REQ_CREDIT_BUREAU_DAY": (0.0, 1.0),
    "AMT_REQ_CREDIT_BUREAU_WEEK": (1.0, 7.0),
    "AMT_REQ_CREDIT_BUREAU_MON": (7.0, 30.0),
    "AMT_REQ_CREDIT_BUREAU_QRT": (30.0, 90.0),
    "AMT_REQ_CREDIT_BUREAU_YEAR": (90.0, 365.0),
}


class IntakeError(ValueError):
    """Raised when a workbook cannot be read as a valid application package."""


@dataclass
class IntakeResult:
    features: pd.DataFrame
    findings: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


def _normalise_flag(value) -> float:
    """Map Y/N style input to 1.0/0.0, preserving missing as NaN."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in YES_VALUES:
        return 1.0
    if text in NO_VALUES:
        return 0.0
    if text in {"", "na", "nan", "none"}:
        return np.nan
    raise IntakeError(f"expected Y or N, got {value!r}")


def _numeric(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text == "" or text.lower() in {"na", "nan", "none"}:
            return np.nan
        value = text
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IntakeError(f"expected a number, got {value!r}") from exc
    if np.isinf(result):
        return np.nan
    return result


def load_workbook(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read an .xlsx workbook or a directory of same-named CSV files."""

    source = Path(path)
    sheets: dict[str, pd.DataFrame] = {}
    if source.is_dir():
        for name in (*REQUIRED_SHEETS, *OPTIONAL_SHEETS):
            candidate = source / f"{name}.csv"
            if candidate.exists():
                sheets[name] = pd.read_csv(candidate)
    elif source.suffix.lower() in {".xlsx", ".xlsm"}:
        book = pd.read_excel(source, sheet_name=None)
        sheets = {str(name).strip().lower(): frame for name, frame in book.items()}
    else:
        raise IntakeError(f"unsupported intake source: {source}")

    missing = [name for name in REQUIRED_SHEETS if name not in sheets]
    if missing:
        raise IntakeError(f"missing required sheet(s): {', '.join(missing)}")
    return sheets


def _application_features(frame: pd.DataFrame, findings: list[str]) -> dict[str, float | str]:
    if len(frame) != 1:
        raise IntakeError(f"'{APPLICATION_SHEET}' must hold exactly one row, found {len(frame)}")
    row = frame.iloc[0]

    absent = [
        column
        for column in REQUIRED_APPLICATION_COLUMNS
        if column not in frame.columns or pd.isna(row.get(column))
    ]
    if absent:
        raise IntakeError(f"missing required application field(s): {', '.join(absent)}")

    features: dict[str, float | str] = {}
    for source_column, model_column in APPLICATION_COLUMN_MAP.items():
        if source_column not in frame.columns:
            features[model_column] = np.nan
            continue
        value = row[source_column]
        if model_column in {"FLAG_OWN_CAR", "FLAG_OWN_REALTY"}:
            # These stay categorical Y/N in the trained schema.
            features[model_column] = np.nan if pd.isna(value) else str(value).strip().upper()[:1]
        elif model_column in {"FLAG_EMAIL", "FLAG_WORK_PHONE"}:
            features[model_column] = _normalise_flag(value)
        elif model_column.startswith("NAME_"):
            features[model_column] = np.nan if pd.isna(value) else str(value).strip()
        else:
            features[model_column] = _numeric(value)

    income = features.get("AMT_INCOME_TOTAL")
    if income is not None and not pd.isna(income) and income <= 0:
        raise IntakeError("annual_income must be positive")

    if features.get("FLAG_OWN_CAR") == "Y" and pd.isna(features.get("OWN_CAR_AGE")):
        findings.append("owns_car is Y but car_age_years is missing; OWN_CAR_AGE stays missing")

    # Home Credit encodes 'not employed' as a sentinel rather than a duration.
    employment = features.get("EMPLOYMENT_YEARS")
    if employment is None or pd.isna(employment) or employment < 0:
        features["EMPLOYMENT_SENTINEL"] = 1
        features["EMPLOYMENT_YEARS"] = np.nan
    else:
        features["EMPLOYMENT_SENTINEL"] = 0
    return features


def _per_account_dpd(dpd: pd.DataFrame) -> pd.DataFrame:
    """Reproduce bureau_balance_by_loan from history_features.py."""

    status = dpd["status"].astype(str).str.strip().str.upper()
    months_balance = -dpd["months_ago"].astype(float)
    numeric_status = pd.to_numeric(status, errors="coerce").fillna(0)
    work = pd.DataFrame(
        {
            "account_ref": dpd["account_ref"].astype(str),
            "months_balance": months_balance,
            "delinquent": status.isin(DELINQUENT_STATUSES).astype(int),
            "severe": status.isin(SEVERE_STATUSES).astype(int),
            "level": numeric_status,
            "default": status.eq("5").astype(int),
            "closed": status.eq("C").astype(int),
            "unknown": status.eq("X").astype(int),
            "recent_delinquent": (status.isin(DELINQUENT_STATUSES) & (months_balance >= -12)).astype(
                int
            ),
        }
    )
    grouped = work.groupby("account_ref", sort=False)
    return pd.DataFrame(
        {
            "BB_MONTHS_OBSERVED": grouped.size(),
            "BB_DELINQUENT_MONTHS": grouped["delinquent"].sum(),
            "BB_SEVERE_DELINQUENT_MONTHS": grouped["severe"].sum(),
            "BB_MAX_DELINQUENCY_LEVEL": grouped["level"].max(),
            "BB_DEFAULT_MONTHS": grouped["default"].sum(),
            "BB_CLOSED_MONTHS": grouped["closed"].sum(),
            "BB_UNKNOWN_MONTHS": grouped["unknown"].sum(),
            "BB_RECENT_12M_DELINQUENT_MONTHS": grouped["recent_delinquent"].sum(),
        }
    )


def expand_dpd_strips(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert one-row-per-account ``dpd_strip`` form into account-month rows."""

    if "dpd_strip" not in frame.columns:
        return frame
    rows = []
    for _, row in frame.iterrows():
        strip = row.get("dpd_strip")
        if pd.isna(strip):
            continue
        for offset, character in enumerate(str(strip).strip()):
            if character in {"-", " "}:
                continue
            rows.append(
                {
                    "account_ref": row["account_ref"],
                    "months_ago": offset,
                    "status": character,
                }
            )
    return pd.DataFrame(rows, columns=["account_ref", "months_ago", "status"])


def _bureau_features(
    tradelines: pd.DataFrame | None,
    dpd: pd.DataFrame | None,
    findings: list[str],
) -> dict[str, float]:
    """Reproduce bureau_features from history_features.py, including the BB roll-up."""

    if tradelines is None or tradelines.empty:
        findings.append("No tradelines supplied; treated as no bureau record, not as zero debt.")
        return {"HAS_BUREAU_HISTORY": 0}

    frame = tradelines.copy()
    for column in (
        "sanctioned_amount",
        "credit_limit",
        "current_balance",
        "overdue_amount",
        "days_overdue",
        "annuity",
        "times_prolonged",
        "opened_days_ago",
    ):
        frame[column] = (
            frame[column].map(_numeric) if column in frame.columns else np.nan
        )
    status = frame.get("status", pd.Series(dtype=object)).astype(str).str.strip()
    unknown_status = sorted(set(status) - set(TRADELINE_STATUSES))
    if unknown_status:
        findings.append(
            f"Unrecognised tradeline status value(s) treated as neither active nor closed: "
            f"{', '.join(unknown_status)}"
        )

    count = float(len(frame))
    credit_sum = frame["sanctioned_amount"].sum(min_count=1)
    active_count = float(status.eq("Active").sum())
    overdue_count = float((frame["days_overdue"] > 0).sum())

    features: dict[str, float] = {
        "HAS_BUREAU_HISTORY": 1,
        "BUREAU_CREDIT_COUNT": count,
        "BUREAU_UNIQUE_CREDIT_COUNT": float(frame["account_ref"].nunique())
        if "account_ref" in frame.columns
        else count,
        "BUREAU_CREDIT_TYPE_COUNT": float(frame["credit_type"].nunique())
        if "credit_type" in frame.columns
        else np.nan,
        "BUREAU_ACTIVE_COUNT": active_count,
        "BUREAU_CLOSED_COUNT": float(status.eq("Closed").sum()),
        "BUREAU_SOLD_COUNT": float(status.eq("Sold").sum()),
        "BUREAU_BAD_DEBT_COUNT": float(status.eq("Bad debt").sum()),
        "BUREAU_RECENT_12M_COUNT": float((frame["opened_days_ago"] <= 365).sum()),
        "BUREAU_RECENT_24M_COUNT": float((frame["opened_days_ago"] <= 730).sum()),
        "BUREAU_DAYS_SINCE_CREDIT_MEAN": frame["opened_days_ago"].mean(),
        "BUREAU_DAYS_SINCE_CREDIT_MIN": frame["opened_days_ago"].min(),
        "BUREAU_DAYS_SINCE_CREDIT_MAX": frame["opened_days_ago"].max(),
        "BUREAU_DAYS_OVERDUE_MEAN": frame["days_overdue"].mean(),
        "BUREAU_DAYS_OVERDUE_MAX": frame["days_overdue"].max(),
        "BUREAU_OVERDUE_CREDIT_COUNT": overdue_count,
        "BUREAU_PROLONG_SUM": frame["times_prolonged"].sum(min_count=1),
        "BUREAU_PROLONG_MAX": frame["times_prolonged"].max(),
        "BUREAU_CREDIT_SUM": credit_sum,
        "BUREAU_CREDIT_MEAN": frame["sanctioned_amount"].mean(),
        "BUREAU_CREDIT_MAX": frame["sanctioned_amount"].max(),
        "BUREAU_DEBT_SUM": frame["current_balance"].sum(min_count=1),
        "BUREAU_DEBT_MEAN": frame["current_balance"].mean(),
        "BUREAU_DEBT_MAX": frame["current_balance"].max(),
        "BUREAU_OVERDUE_SUM": frame["overdue_amount"].sum(min_count=1),
        "BUREAU_OVERDUE_MAX": frame["overdue_amount"].max(),
        "BUREAU_LIMIT_SUM": frame["credit_limit"].sum(min_count=1),
        "BUREAU_ANNUITY_SUM": frame["annuity"].sum(min_count=1),
        "BUREAU_ACTIVE_RATE": active_count / count,
        "BUREAU_OVERDUE_RATE": overdue_count / count,
    }
    debt_sum = features["BUREAU_DEBT_SUM"]
    overdue_sum = features["BUREAU_OVERDUE_SUM"]
    divisor = credit_sum if credit_sum not in (0, None) and not pd.isna(credit_sum) else np.nan
    features["BUREAU_DEBT_TO_CREDIT"] = debt_sum / divisor
    features["BUREAU_OVERDUE_TO_CREDIT"] = overdue_sum / divisor

    if dpd is None or dpd.empty:
        findings.append("No monthly delinquency history supplied; BB_* features stay missing.")
        return features

    per_account = _per_account_dpd(dpd)
    known = set(frame["account_ref"].astype(str)) if "account_ref" in frame.columns else set()
    orphans = sorted(set(per_account.index) - known) if known else []
    if orphans:
        findings.append(
            f"dpd_history references {len(orphans)} account_ref value(s) absent from tradelines; "
            "they are excluded"
        )
        per_account = per_account.drop(index=orphans, errors="ignore")
    if per_account.empty:
        return features

    months_observed = float(per_account["BB_MONTHS_OBSERVED"].sum())
    delinquent = float(per_account["BB_DELINQUENT_MONTHS"].sum())
    severe = float(per_account["BB_SEVERE_DELINQUENT_MONTHS"].sum())
    features.update(
        {
            "BB_MONTHS_OBSERVED": months_observed,
            "BB_DELINQUENT_MONTHS": delinquent,
            "BB_SEVERE_DELINQUENT_MONTHS": severe,
            "BB_MAX_DELINQUENCY_LEVEL": float(per_account["BB_MAX_DELINQUENCY_LEVEL"].max()),
            "BB_DEFAULT_MONTHS": float(per_account["BB_DEFAULT_MONTHS"].sum()),
            "BB_CLOSED_MONTHS": float(per_account["BB_CLOSED_MONTHS"].sum()),
            "BB_UNKNOWN_MONTHS": float(per_account["BB_UNKNOWN_MONTHS"].sum()),
            "BB_RECENT_12M_DELINQUENT_MONTHS": float(
                per_account["BB_RECENT_12M_DELINQUENT_MONTHS"].sum()
            ),
            "BB_DELINQUENT_MONTH_RATE": delinquent / months_observed
            if months_observed
            else np.nan,
            "BB_SEVERE_DELINQUENT_MONTH_RATE": severe / months_observed
            if months_observed
            else np.nan,
        }
    )
    return features


def _monthly_frame(months: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Select one account class and normalise its columns to the trained semantics."""

    if "account_kind" not in months.columns:
        raise IntakeError(f"'{ACCOUNT_MONTHS_SHEET}' requires an account_kind column")
    subset = months.loc[months["account_kind"].astype(str).str.strip().str.lower() == kind]
    if subset.empty:
        return subset
    frame = pd.DataFrame(
        {
            "account_ref": subset["account_ref"].astype(str),
            # MONTHS_BALANCE counts backwards from the application date.
            "months_balance": -subset["months_ago"].map(_numeric),
        }
    )
    for source, name in (
        ("dpd_days", "dpd"),
        ("dpd_days_tolerant", "dpd_tolerant"),
        ("balance", "balance"),
        ("credit_limit", "limit"),
        ("installments_total", "installments_total"),
        ("installments_remaining", "installments_remaining"),
    ):
        frame[name] = subset[source].map(_numeric) if source in subset.columns else np.nan
    frame["status"] = (
        subset["contract_status"].astype(str).str.strip()
        if "contract_status" in subset.columns
        else ""
    )
    return frame


def _revolving_features(months: pd.DataFrame) -> dict[str, float]:
    """Reproduce credit_card_features from history_features.py."""

    count = float(len(months))
    dpd_months = float((months["dpd"] > 0).sum())
    dpd_default_months = float((months["dpd_tolerant"] > 0).sum())
    # AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)): per-month ratio, then mean.
    utilisation = months["balance"].div(months["limit"].where(months["limit"] != 0))
    return {
        "HAS_CREDIT_CARD_HISTORY": 1,
        "CC_MONTH_COUNT": count,
        "CC_CONTRACT_COUNT": float(months["account_ref"].nunique()),
        "CC_OLDEST_MONTH": months["months_balance"].min(),
        "CC_LATEST_MONTH": months["months_balance"].max(),
        "CC_BALANCE_MEAN": months["balance"].mean(),
        "CC_BALANCE_MAX": months["balance"].max(),
        "CC_BALANCE_SUM": months["balance"].sum(min_count=1),
        "CC_LIMIT_MEAN": months["limit"].mean(),
        "CC_LIMIT_MAX": months["limit"].max(),
        "CC_UTILIZATION_MEAN": utilisation.mean(),
        "CC_UTILIZATION_MAX": utilisation.max(),
        "CC_DPD_MEAN": months["dpd"].mean(),
        "CC_DPD_MAX": months["dpd"].max(),
        "CC_DPD_MONTH_COUNT": dpd_months,
        "CC_DPD_DEF_MONTH_COUNT": dpd_default_months,
        "CC_RECENT_12M_DPD_COUNT": float(
            ((months["months_balance"] >= -12) & (months["dpd"] > 0)).sum()
        ),
        "CC_ACTIVE_MONTH_COUNT": float(months["status"].eq("Active").sum()),
        "CC_DPD_MONTH_RATE": dpd_months / count,
        "CC_DPD_DEF_MONTH_RATE": dpd_default_months / count,
    }


def _installment_account_features(months: pd.DataFrame) -> dict[str, float]:
    """Reproduce pos_features from history_features.py."""

    count = float(len(months))
    dpd_months = float((months["dpd"] > 0).sum())
    dpd_default_months = float((months["dpd_tolerant"] > 0).sum())
    return {
        "HAS_POS_HISTORY": 1,
        "POS_MONTH_COUNT": count,
        "POS_CONTRACT_COUNT": float(months["account_ref"].nunique()),
        "POS_OLDEST_MONTH": months["months_balance"].min(),
        "POS_LATEST_MONTH": months["months_balance"].max(),
        "POS_INSTALLMENT_COUNT_MEAN": months["installments_total"].mean(),
        "POS_INSTALLMENT_COUNT_MAX": months["installments_total"].max(),
        "POS_FUTURE_INSTALLMENTS_MEAN": months["installments_remaining"].mean(),
        "POS_FUTURE_INSTALLMENTS_MIN": months["installments_remaining"].min(),
        "POS_DPD_MEAN": months["dpd"].mean(),
        "POS_DPD_MAX": months["dpd"].max(),
        "POS_DPD_SUM": months["dpd"].sum(min_count=1),
        "POS_DPD_DEF_MEAN": months["dpd_tolerant"].mean(),
        "POS_DPD_DEF_MAX": months["dpd_tolerant"].max(),
        "POS_DPD_MONTH_COUNT": dpd_months,
        "POS_DPD_DEF_MONTH_COUNT": dpd_default_months,
        "POS_RECENT_12M_DPD_COUNT": float(
            ((months["months_balance"] >= -12) & (months["dpd"] > 0)).sum()
        ),
        "POS_ACTIVE_MONTH_COUNT": float(months["status"].eq("Active").sum()),
        "POS_COMPLETED_MONTH_COUNT": float(months["status"].eq("Completed").sum()),
        "POS_DPD_MONTH_RATE": dpd_months / count,
        "POS_DPD_DEF_MONTH_RATE": dpd_default_months / count,
    }


def _account_month_features(
    months: pd.DataFrame | None, findings: list[str]
) -> dict[str, float]:
    """Build the tradeline-behaviour families the extended model consumes."""

    if months is None or months.empty:
        findings.append(
            "No per-account monthly detail supplied; revolving and installment behaviour "
            "features stay missing. The extended model cannot be served without them."
        )
        return {"HAS_CREDIT_CARD_HISTORY": 0, "HAS_POS_HISTORY": 0}

    features: dict[str, float] = {}
    revolving = _monthly_frame(months, REVOLVING_KIND)
    if revolving.empty:
        features["HAS_CREDIT_CARD_HISTORY"] = 0
        findings.append("No revolving-account months supplied; CC_* features stay missing.")
    else:
        features.update(_revolving_features(revolving))

    installment = _monthly_frame(months, INSTALLMENT_KIND)
    if installment.empty:
        features["HAS_POS_HISTORY"] = 0
        findings.append("No installment-account months supplied; POS_* features stay missing.")
    else:
        features.update(_installment_account_features(installment))
    return features


def _enquiry_features(enquiries: pd.DataFrame | None, findings: list[str]) -> dict[str, float]:
    if enquiries is None or enquiries.empty:
        findings.append("No enquiry records supplied; enquiry-velocity features stay missing.")
        return {}
    days_ago = enquiries["days_ago"].map(_numeric)
    features = {
        column: float(((days_ago > lower) & (days_ago <= upper)).sum())
        for column, (lower, upper) in ENQUIRY_BANDS.items()
    }
    beyond = int((days_ago > 365).sum())
    if beyond:
        findings.append(
            f"{beyond} enquiry record(s) older than 365 days fall outside every reported band "
            "and are not counted"
        )
    return features


def _computed_ratios(features: dict) -> dict[str, float]:
    """Match engineer_application_features plus the two obligation ratios."""

    def ratio(numerator, denominator) -> float:
        if numerator is None or denominator is None:
            return np.nan
        if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
            return np.nan
        return float(numerator) / float(denominator)

    income = features.get("AMT_INCOME_TOTAL")
    credit = features.get("AMT_CREDIT")
    annuity = features.get("AMT_ANNUITY")
    goods = features.get("AMT_GOODS_PRICE")
    bureau_annuity = features.get("BUREAU_ANNUITY_SUM")
    bureau_debt = features.get("BUREAU_DEBT_SUM")

    computed = {
        "CREDIT_TO_INCOME": ratio(credit, income),
        "ANNUITY_TO_INCOME": ratio(annuity, income),
        "CREDIT_TO_ANNUITY": ratio(credit, annuity),
        "CREDIT_TO_GOODS": ratio(credit, goods),
        "DOWN_PAYMENT_RATE": ratio(
            (goods - credit) if goods is not None and credit is not None else None, goods
        ),
        "EXTERNAL_DEBT_TO_INCOME": ratio(bureau_debt, income),
    }
    # Post-disbursal FOIR. Unknown external obligation stays unknown; it is never read as zero.
    if bureau_annuity is None or pd.isna(bureau_annuity):
        computed["TOTAL_OBLIGATION_TO_INCOME"] = np.nan
    else:
        total = (0.0 if annuity is None or pd.isna(annuity) else float(annuity)) + float(
            bureau_annuity
        )
        computed["TOTAL_OBLIGATION_TO_INCOME"] = ratio(total, income)
    return computed


def build_features(sheets: dict[str, pd.DataFrame]) -> IntakeResult:
    """Assemble one model-ready row from a validated workbook."""

    findings: list[str] = []
    features: dict = {}
    features.update(_application_features(sheets[APPLICATION_SHEET], findings))

    tradelines = sheets.get(TRADELINES_SHEET)
    dpd = sheets.get(DPD_SHEET)
    if dpd is not None and not dpd.empty:
        dpd = expand_dpd_strips(dpd)
    features.update(_bureau_features(tradelines, dpd, findings))
    account_months = sheets.get(ACCOUNT_MONTHS_SHEET)
    features.update(_account_month_features(account_months, findings))
    features.update(_enquiry_features(sheets.get(ENQUIRIES_SHEET), findings))
    features.update(_computed_ratios(features))

    row = pd.DataFrame([features])
    # APPLICATION_MISSING_COUNT is deliberately NOT set here. It must be counted over the
    # serving model's own column list, which intake does not know; computing it over whatever
    # the workbook happened to produce would disagree with training. It is filled in during
    # feature alignment -- see credifast.modeling.collectible_inference.align_features.

    evidence = {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "tradeline_count": 0 if tradelines is None else len(tradelines),
        "dpd_month_count": 0 if dpd is None else len(dpd),
        "account_month_count": 0 if account_months is None else len(account_months),
        "enquiry_count": len(sheets.get(ENQUIRIES_SHEET, pd.DataFrame())),
        "has_bureau_history": int(features.get("HAS_BUREAU_HISTORY", 0)),
        "total_obligation_to_income": features.get("TOTAL_OBLIGATION_TO_INCOME"),
        "external_debt_to_income": features.get("EXTERNAL_DEBT_TO_INCOME"),
        "reported_emi_total": features.get("BUREAU_ANNUITY_SUM"),
    }
    return IntakeResult(features=row, findings=findings, evidence=evidence)


def read_intake(path: str | Path) -> IntakeResult:
    """Load a workbook and return model-ready features plus reviewer findings."""

    return build_features(load_workbook(path))
