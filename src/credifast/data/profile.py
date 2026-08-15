"""Streaming quality profile for the labelled application table."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

NULL_MARKERS = {"", "NA", "N/A", "NAN", "NULL", "NONE"}
CRITICAL_COLUMNS = {
    "SK_ID_CURR",
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
}
NUMERIC_PROFILE_COLUMNS = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
)
CATEGORICAL_PROFILE_COLUMNS = (
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
)


def is_null(value: str | None) -> bool:
    return value is None or value.strip().upper() in NULL_MARKERS


@dataclass
class RunningStats:
    count: int = 0
    invalid_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float = 0.0
    _m2: float = 0.0

    def add(self, raw_value: str | None) -> None:
        if is_null(raw_value):
            return
        try:
            value = float(raw_value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            self.invalid_count += 1
            return
        if not math.isfinite(value):
            self.invalid_count += 1
            return

        self.count += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)

    def to_dict(self) -> dict:
        variance = self._m2 / (self.count - 1) if self.count > 1 else 0.0
        return {
            "count": self.count,
            "invalid_count": self.invalid_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean if self.count else None,
            "standard_deviation": math.sqrt(variance) if self.count else None,
        }


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    count: int
    rate: float
    message: str
    analytical_risk: str
    remediation: str

    def to_dict(self) -> dict:
        return asdict(self)


def _finding(
    code: str,
    severity: str,
    count: int,
    row_count: int,
    message: str,
    analytical_risk: str,
    remediation: str,
) -> Finding:
    return Finding(
        code=code,
        severity=severity,
        count=count,
        rate=count / row_count if row_count else 0.0,
        message=message,
        analytical_risk=analytical_risk,
        remediation=remediation,
    )


def profile_application_table(input_path: str | Path) -> dict:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Application table does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing_critical = sorted(CRITICAL_COLUMNS - set(columns))
        if missing_critical:
            raise ValueError(
                "Application table is missing critical columns: " + ", ".join(missing_critical)
            )

        missing_counts = Counter({column: 0 for column in columns})
        target_counts: Counter[str] = Counter()
        seen_keys: set[str] = set()
        duplicate_key_count = 0
        missing_key_count = 0
        employment_sentinel_count = 0
        gender_xna_count = 0
        nonpositive_income_count = 0
        nonpositive_credit_count = 0
        nonpositive_annuity_count = 0
        numeric_stats = {
            column: RunningStats() for column in NUMERIC_PROFILE_COLUMNS if column in columns
        }
        category_values: dict[str, Counter[str]] = {
            column: Counter() for column in CATEGORICAL_PROFILE_COLUMNS if column in columns
        }
        row_count = 0

        for row in reader:
            row_count += 1
            for column in columns:
                if is_null(row.get(column)):
                    missing_counts[column] += 1

            key = (row.get("SK_ID_CURR") or "").strip()
            if not key:
                missing_key_count += 1
            elif key in seen_keys:
                duplicate_key_count += 1
            else:
                seen_keys.add(key)

            target = (row.get("TARGET") or "").strip()
            target_counts[target if target in {"0", "1"} else "other"] += 1

            for column, stats in numeric_stats.items():
                stats.add(row.get(column))
            for column, counts in category_values.items():
                value = row.get(column)
                counts["<missing>" if is_null(value) else value.strip()] += 1  # type: ignore[union-attr]

            if (row.get("DAYS_EMPLOYED") or "").strip() in {"365243", "365243.0"}:
                employment_sentinel_count += 1
            if (row.get("CODE_GENDER") or "").strip().upper() == "XNA":
                gender_xna_count += 1

            for column, counter_name in (
                ("AMT_INCOME_TOTAL", "income"),
                ("AMT_CREDIT", "credit"),
                ("AMT_ANNUITY", "annuity"),
            ):
                raw_value = row.get(column)
                if is_null(raw_value):
                    continue
                try:
                    nonpositive = float(raw_value) <= 0  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if nonpositive and counter_name == "income":
                    nonpositive_income_count += 1
                elif nonpositive and counter_name == "credit":
                    nonpositive_credit_count += 1
                elif nonpositive and counter_name == "annuity":
                    nonpositive_annuity_count += 1

    findings: list[Finding] = []
    invalid_target_count = target_counts["other"]
    finding_specs = (
        (
            "DQ-APP-001",
            "critical",
            duplicate_key_count,
            "Duplicate applicant keys violate the expected one-row-per-application grain",
            "Duplicate applicants can leak across splits and bias model training",
            "Resolve duplicate records before assigning train/calibration/holdout partitions",
        ),
        (
            "DQ-APP-002",
            "critical",
            missing_key_count,
            "Applicant rows have missing SK_ID_CURR values",
            "Rows cannot be joined reliably to historical sources",
            "Reject or repair missing identifiers before feature generation",
        ),
        (
            "DQ-APP-003",
            "high",
            invalid_target_count,
            "Target values are missing or outside the accepted 0/1 set",
            "Invalid labels make supervised evaluation unreliable",
            "Quarantine invalid labels and reconcile the source extract",
        ),
        (
            "DQ-APP-004",
            "medium",
            employment_sentinel_count,
            "DAYS_EMPLOYED contains the known 365243 sentinel value",
            "Treating the sentinel as a duration creates extreme and misleading features",
            "Replace with null and add an employment-sentinel indicator",
        ),
        (
            "DQ-APP-005",
            "medium",
            gender_xna_count,
            "CODE_GENDER contains XNA values",
            "Unmodelled categories can break transforms and distort audit segments",
            "Normalize to an explicit unknown category and keep gender audit-only",
        ),
        (
            "DQ-APP-006",
            "high",
            nonpositive_income_count,
            "Annual income is zero or negative",
            "Affordability ratios become invalid or misleading",
            "Quarantine impossible values and confirm source rules",
        ),
        (
            "DQ-APP-007",
            "high",
            nonpositive_credit_count,
            "Requested credit is zero or negative",
            "The current application amount is not valid for risk or affordability modeling",
            "Quarantine impossible values and confirm source rules",
        ),
        (
            "DQ-APP-008",
            "high",
            nonpositive_annuity_count,
            "Annuity is zero or negative",
            "Debt-service and affordability features become invalid",
            "Treat missing separately and quarantine nonpositive observed values",
        ),
    )
    for code, severity, count, message, risk, remediation in finding_specs:
        if count:
            findings.append(_finding(code, severity, count, row_count, message, risk, remediation))

    high_missingness = sorted(
        (
            {
                "column": column,
                "missing_count": count,
                "missing_rate": count / row_count if row_count else 0.0,
            }
            for column, count in missing_counts.items()
            if row_count and count / row_count >= 0.50
        ),
        key=lambda item: item["missing_rate"],
        reverse=True,
    )

    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    maximum_severity = max(
        (findings_item.severity for findings_item in findings),
        key=lambda value: severity_rank[value],
        default="none",
    )
    ready_for_modeling = not any(item.severity in {"critical", "high"} for item in findings)

    return {
        "profile_version": "1.0.0",
        "table": "application_train",
        "grain": "one labelled current loan application",
        "source_file": path.name,
        "summary": {
            "row_count": row_count,
            "column_count": len(columns),
            "unique_applicant_keys": len(seen_keys),
            "duplicate_key_count": duplicate_key_count,
            "target_counts": dict(target_counts),
            "target_prevalence": target_counts["1"] / row_count if row_count else None,
            "maximum_finding_severity": maximum_severity,
            "ready_for_modeling": ready_for_modeling,
        },
        "missingness": {
            "by_column": {
                column: {
                    "count": missing_counts[column],
                    "rate": missing_counts[column] / row_count if row_count else 0.0,
                }
                for column in columns
            },
            "columns_at_or_above_50_percent": high_missingness,
        },
        "numeric_profile": {column: stats.to_dict() for column, stats in numeric_stats.items()},
        "categorical_profile": {
            column: dict(counts.most_common()) for column, counts in category_values.items()
        },
        "findings": [item.to_dict() for item in findings],
        "assumptions": [
            "SK_ID_CURR is expected to be unique in application_train",
            "TARGET accepts only 0 and 1",
            "Empty strings and common textual null markers are treated as missing",
            "DAYS_EMPLOYED=365243 is treated as a sentinel rather than a real duration",
        ],
    }


def write_profile(profile: dict, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return destination
