"""Generate synthetic demo intake workbooks for the hackathon demonstration.

Every value here is invented. No Home Credit applicant record is exported, because
row-level dataset records are Git-ignored and could not ship with a demo. The amounts are
plausible Indian retail-lending figures that also sit inside the distribution the model was
trained on (training medians: income 147,150, credit 513,531, annuity 24,903).

Categorical values use the exact domains the model was trained on. Anything outside them
would encode to "unknown" and score out of distribution.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\generate_demo_workbooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credifast.data.workbook_intake import read_intake
from credifast.modeling.collectible_inference import (
    CollectibleModelUnavailable,
    score_workbook,
)

OUTPUT_DIR = ROOT / "examples" / "demo"

APPLICATION_COLUMNS = [
    "applicant_ref",
    "annual_income",
    "requested_credit",
    "annual_annuity",
    "goods_price",
    "contract_type",
    "income_type",
    "education_type",
    "housing_type",
    "owns_car",
    "car_age_years",
    "owns_realty",
    "employment_years",
    "address_vintage_years",
    "id_document_age_years",
    "has_email",
    "has_work_phone",
]
TRADELINE_COLUMNS = [
    "account_ref",
    "credit_type",
    "status",
    "opened_days_ago",
    "sanctioned_amount",
    "credit_limit",
    "current_balance",
    "overdue_amount",
    "days_overdue",
    "annuity",
    "times_prolonged",
]
DPD_COLUMNS = ["account_ref", "months_ago", "status"]
ACCOUNT_MONTH_COLUMNS = [
    "account_ref",
    "account_kind",
    "months_ago",
    "dpd_days",
    "dpd_days_tolerant",
    "contract_status",
    "balance",
    "credit_limit",
    "installments_total",
    "installments_remaining",
]
ENQUIRY_COLUMNS = ["days_ago", "purpose"]


def _clean_months(account_ref: str, count: int, closed_after: int | None = None) -> list[dict]:
    """A run of on-time months, optionally reported closed beyond a point."""

    rows = []
    for month in range(count):
        status = "C" if closed_after is not None and month >= closed_after else "0"
        rows.append({"account_ref": account_ref, "months_ago": month, "status": status})
    return rows


def _installment_months(
    account_ref: str,
    count: int,
    total_installments: int,
    *,
    dpd_schedule: dict[int, int] | None = None,
) -> list[dict]:
    dpd_schedule = dpd_schedule or {}
    rows = []
    for month in range(count):
        remaining = max(total_installments - (count - month), 0)
        rows.append(
            {
                "account_ref": account_ref,
                "account_kind": "installment",
                "months_ago": month,
                "dpd_days": dpd_schedule.get(month, 0),
                "dpd_days_tolerant": max(dpd_schedule.get(month, 0) - 30, 0),
                "contract_status": "Active" if remaining > 0 else "Completed",
                "balance": None,
                "credit_limit": None,
                "installments_total": total_installments,
                "installments_remaining": remaining,
            }
        )
    return rows


def _revolving_months(
    account_ref: str,
    count: int,
    limit: float,
    utilisation: list[float],
    *,
    dpd_schedule: dict[int, int] | None = None,
) -> list[dict]:
    dpd_schedule = dpd_schedule or {}
    rows = []
    for month in range(count):
        used = utilisation[month % len(utilisation)]
        rows.append(
            {
                "account_ref": account_ref,
                "account_kind": "revolving",
                "months_ago": month,
                "dpd_days": dpd_schedule.get(month, 0),
                "dpd_days_tolerant": max(dpd_schedule.get(month, 0) - 30, 0),
                "contract_status": "Active",
                "balance": round(limit * used, 2),
                "credit_limit": limit,
                "installments_total": None,
                "installments_remaining": None,
            }
        )
    return rows


def clean_full_file() -> dict[str, list[dict]]:
    """Long clean history across all three source families, comfortable obligation."""

    return {
        "application": [
            {
                "applicant_ref": "DEMO-01-CLEAN",
                "annual_income": 1_200_000,
                "requested_credit": 900_000,
                "annual_annuity": 216_000,
                "goods_price": 950_000,
                "contract_type": "Cash loans",
                "income_type": "Commercial associate",
                "education_type": "Higher education",
                "housing_type": "House / apartment",
                "owns_car": "Y",
                "car_age_years": 5,
                "owns_realty": "Y",
                "employment_years": 8.5,
                "address_vintage_years": 11.0,
                "id_document_age_years": 6.2,
                "has_email": "Y",
                "has_work_phone": "Y",
            }
        ],
        "tradelines": [
            {
                "account_ref": "TL-1001",
                "credit_type": "Consumer credit",
                "status": "Closed",
                "opened_days_ago": 2100,
                "sanctioned_amount": 250_000,
                "credit_limit": 0,
                "current_balance": 0,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": None,
                "times_prolonged": 0,
            },
            {
                "account_ref": "TL-1002",
                "credit_type": "Car loan",
                "status": "Closed",
                "opened_days_ago": 1650,
                "sanctioned_amount": 620_000,
                "credit_limit": 0,
                "current_balance": 0,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": None,
                "times_prolonged": 0,
            },
            {
                "account_ref": "TL-1003",
                "credit_type": "Mortgage",
                "status": "Active",
                "opened_days_ago": 980,
                "sanctioned_amount": 2_400_000,
                "credit_limit": 0,
                "current_balance": 1_760_000,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": 48_000,
                "times_prolonged": 0,
            },
            {
                "account_ref": "TL-1004",
                "credit_type": "Credit card",
                "status": "Active",
                "opened_days_ago": 1450,
                "sanctioned_amount": 200_000,
                "credit_limit": 200_000,
                "current_balance": 34_000,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": 24_000,
                "times_prolonged": 0,
            },
        ],
        "dpd_history": (
            _clean_months("TL-1001", 40, closed_after=18)
            + _clean_months("TL-1002", 40, closed_after=12)
            + _clean_months("TL-1003", 32)
            + _clean_months("TL-1004", 40)
        ),
        "account_months": (
            _installment_months("POS-2001", 30, 48)
            + _revolving_months("CC-3001", 30, 200_000, [0.12, 0.17, 0.15, 0.19])
        ),
        "enquiries": [{"days_ago": 210, "purpose": "Credit application"}],
    }


def thin_new_to_credit() -> dict[str, list[dict]]:
    """No credit record at all. The hardest and most common real case."""

    return {
        "application": [
            {
                "applicant_ref": "DEMO-02-THIN",
                "annual_income": 360_000,
                "requested_credit": 150_000,
                "annual_annuity": 60_000,
                "goods_price": 145_000,
                "contract_type": "Cash loans",
                "income_type": "Working",
                "education_type": "Secondary / secondary special",
                "housing_type": "With parents",
                "owns_car": "N",
                "car_age_years": None,
                "owns_realty": "N",
                "employment_years": 0.8,
                "address_vintage_years": 1.5,
                "id_document_age_years": 0.9,
                "has_email": "Y",
                "has_work_phone": "N",
            }
        ],
        "tradelines": [],
        "dpd_history": [],
        "account_months": [],
        "enquiries": [],
    }


def recent_delinquency() -> dict[str, list[dict]]:
    """Repayment stress inside the last few reported months."""

    return {
        "application": [
            {
                "applicant_ref": "DEMO-03-DELINQ",
                "annual_income": 480_000,
                "requested_credit": 600_000,
                "annual_annuity": 168_000,
                "goods_price": 580_000,
                "contract_type": "Cash loans",
                "income_type": "Working",
                "education_type": "Secondary / secondary special",
                "housing_type": "Rented apartment",
                "owns_car": "N",
                "car_age_years": None,
                "owns_realty": "N",
                "employment_years": 2.4,
                "address_vintage_years": 3.1,
                "id_document_age_years": 4.0,
                "has_email": "N",
                "has_work_phone": "Y",
            }
        ],
        "tradelines": [
            {
                "account_ref": "TL-2001",
                "credit_type": "Consumer credit",
                "status": "Active",
                "opened_days_ago": 640,
                "sanctioned_amount": 300_000,
                "credit_limit": 0,
                "current_balance": 214_000,
                "overdue_amount": 18_400,
                "days_overdue": 46,
                "annuity": 30_000,
                "times_prolonged": 1,
            },
            {
                "account_ref": "TL-2002",
                "credit_type": "Credit card",
                "status": "Active",
                "opened_days_ago": 900,
                "sanctioned_amount": 120_000,
                "credit_limit": 120_000,
                "current_balance": 111_500,
                "overdue_amount": 6_200,
                "days_overdue": 22,
                "annuity": 18_000,
                "times_prolonged": 0,
            },
            {
                "account_ref": "TL-2003",
                "credit_type": "Microloan",
                "status": "Closed",
                "opened_days_ago": 1300,
                "sanctioned_amount": 45_000,
                "credit_limit": 0,
                "current_balance": 0,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": None,
                "times_prolonged": 0,
            },
        ],
        "dpd_history": (
            [
                {"account_ref": "TL-2001", "months_ago": 0, "status": "2"},
                {"account_ref": "TL-2001", "months_ago": 1, "status": "1"},
                {"account_ref": "TL-2001", "months_ago": 2, "status": "1"},
                {"account_ref": "TL-2001", "months_ago": 3, "status": "0"},
            ]
            + _clean_months("TL-2001", 20)[4:]
            + [
                {"account_ref": "TL-2002", "months_ago": 0, "status": "1"},
                {"account_ref": "TL-2002", "months_ago": 1, "status": "0"},
                {"account_ref": "TL-2002", "months_ago": 2, "status": "1"},
            ]
            + _clean_months("TL-2002", 28)[3:]
            + _clean_months("TL-2003", 30, closed_after=8)
        ),
        "account_months": (
            _installment_months("POS-4001", 22, 36, dpd_schedule={0: 46, 1: 31, 2: 12})
            + _revolving_months(
                "CC-5001",
                24,
                120_000,
                [0.93, 0.88, 0.95, 0.91],
                dpd_schedule={0: 22, 2: 8},
            )
        ),
        "enquiries": [
            {"days_ago": 40, "purpose": "Credit application"},
            {"days_ago": 150, "purpose": "Credit application"},
        ],
    }


def high_obligation() -> dict[str, list[dict]]:
    """Clean repayment record, but obligations already consume most of income."""

    return {
        "application": [
            {
                "applicant_ref": "DEMO-04-FOIR",
                "annual_income": 600_000,
                "requested_credit": 750_000,
                "annual_annuity": 210_000,
                "goods_price": 720_000,
                "contract_type": "Cash loans",
                "income_type": "State servant",
                "education_type": "Higher education",
                "housing_type": "House / apartment",
                "owns_car": "Y",
                "car_age_years": 9,
                "owns_realty": "Y",
                "employment_years": 12.0,
                "address_vintage_years": 14.5,
                "id_document_age_years": 8.1,
                "has_email": "Y",
                "has_work_phone": "Y",
            }
        ],
        "tradelines": [
            {
                "account_ref": "TL-3001",
                "credit_type": "Mortgage",
                "status": "Active",
                "opened_days_ago": 1900,
                "sanctioned_amount": 3_200_000,
                "credit_limit": 0,
                "current_balance": 2_050_000,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": 96_000,
                "times_prolonged": 0,
            },
            {
                "account_ref": "TL-3002",
                "credit_type": "Car loan",
                "status": "Active",
                "opened_days_ago": 820,
                "sanctioned_amount": 900_000,
                "credit_limit": 0,
                "current_balance": 540_000,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": 45_000,
                "times_prolonged": 0,
            },
            {
                "account_ref": "TL-3003",
                "credit_type": "Consumer credit",
                "status": "Active",
                "opened_days_ago": 420,
                "sanctioned_amount": 180_000,
                "credit_limit": 0,
                "current_balance": 96_000,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": 24_000,
                "times_prolonged": 0,
            },
        ],
        "dpd_history": (
            _clean_months("TL-3001", 40)
            + _clean_months("TL-3002", 27)
            + _clean_months("TL-3003", 14)
        ),
        "account_months": (
            _installment_months("POS-6001", 26, 60)
            + _revolving_months("CC-7001", 20, 90_000, [0.35, 0.41, 0.38])
        ),
        "enquiries": [{"days_ago": 95, "purpose": "Credit application"}],
    }


def credit_hungry() -> dict[str, list[dict]]:
    """Enquiry burst and several accounts opened recently."""

    return {
        "application": [
            {
                "applicant_ref": "DEMO-05-HUNGRY",
                "annual_income": 540_000,
                "requested_credit": 480_000,
                "annual_annuity": 144_000,
                "goods_price": 460_000,
                "contract_type": "Cash loans",
                "income_type": "Working",
                "education_type": "Secondary / secondary special",
                "housing_type": "Rented apartment",
                "owns_car": "N",
                "car_age_years": None,
                "owns_realty": "N",
                "employment_years": 1.6,
                "address_vintage_years": 2.0,
                "id_document_age_years": 3.4,
                "has_email": "Y",
                "has_work_phone": "N",
            }
        ],
        "tradelines": [
            {
                "account_ref": "TL-4001",
                "credit_type": "Consumer credit",
                "status": "Active",
                "opened_days_ago": 38,
                "sanctioned_amount": 90_000,
                "credit_limit": 0,
                "current_balance": 88_500,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": 15_000,
                "times_prolonged": 0,
            },
            {
                "account_ref": "TL-4002",
                "credit_type": "Microloan",
                "status": "Active",
                "opened_days_ago": 96,
                "sanctioned_amount": 60_000,
                "credit_limit": 0,
                "current_balance": 57_800,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": 12_000,
                "times_prolonged": 0,
            },
            {
                "account_ref": "TL-4003",
                "credit_type": "Credit card",
                "status": "Active",
                "opened_days_ago": 160,
                "sanctioned_amount": 80_000,
                "credit_limit": 80_000,
                "current_balance": 74_200,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": 9_600,
                "times_prolonged": 0,
            },
            {
                "account_ref": "TL-4004",
                "credit_type": "Consumer credit",
                "status": "Active",
                "opened_days_ago": 300,
                "sanctioned_amount": 120_000,
                "credit_limit": 0,
                "current_balance": 71_000,
                "overdue_amount": 0,
                "days_overdue": 0,
                "annuity": 18_000,
                "times_prolonged": 0,
            },
        ],
        "dpd_history": (
            _clean_months("TL-4001", 2)
            + _clean_months("TL-4002", 4)
            + _clean_months("TL-4003", 6)
            + _clean_months("TL-4004", 10)
        ),
        "account_months": (
            _installment_months("POS-8001", 9, 24)
            + _revolving_months("CC-9001", 6, 80_000, [0.88, 0.92, 0.94])
        ),
        # Disjoint enquiry bands: 0 -> HOUR, (0,1] -> DAY, (1,7] -> WEEK,
        # (7,30] -> MON, (30,90] -> QRT, (90,365] -> YEAR.
        "enquiries": [
            {"days_ago": 3, "purpose": "Credit application"},
            {"days_ago": 6, "purpose": "Credit application"},
            {"days_ago": 14, "purpose": "Credit application"},
            {"days_ago": 26, "purpose": "Credit application"},
            {"days_ago": 58, "purpose": "Credit application"},
            {"days_ago": 190, "purpose": "Credit application"},
        ],
    }


PROFILES = {
    "clean_full_file": clean_full_file,
    "thin_new_to_credit": thin_new_to_credit,
    "recent_delinquency": recent_delinquency,
    "high_obligation": high_obligation,
    "credit_hungry": credit_hungry,
}

SHEET_COLUMNS = {
    "application": APPLICATION_COLUMNS,
    "tradelines": TRADELINE_COLUMNS,
    "dpd_history": DPD_COLUMNS,
    "account_months": ACCOUNT_MONTH_COLUMNS,
    "enquiries": ENQUIRY_COLUMNS,
}


def _frames(profile: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
    """Build fixed-column frames so empty sheets still carry their headers."""

    return {
        name: pd.DataFrame(rows, columns=SHEET_COLUMNS[name])
        for name, rows in profile.items()
    }


def write_profile(name: str, frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    directory = OUTPUT_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    for sheet, frame in frames.items():
        frame.to_csv(directory / f"{sheet}.csv", index=False)

    workbook = OUTPUT_DIR / f"{name}.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        for sheet, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    return workbook, directory


def _format_ratio(value) -> str:
    if value is None or pd.isna(value):
        return "not available"
    return f"{value * 100:.1f}%"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, builder in PROFILES.items():
        frames = _frames(builder())
        workbook, directory = write_profile(name, frames)

        intake = read_intake(workbook)
        try:
            score = score_workbook(workbook)
            probability = f"{score.probability * 100:.2f}%"
            sources = ", ".join(score.sources_present) or "none"
        except CollectibleModelUnavailable as error:
            probability = "model unavailable"
            sources = "-"
            print(f"  {name}: {error}")

        evidence = intake.evidence
        rows.append(
            {
                "profile": name,
                "tradelines": evidence["tradeline_count"],
                "dpd_months": evidence["dpd_month_count"],
                "account_months": evidence["account_month_count"],
                "enquiries": evidence["enquiry_count"],
                "sources_present": sources,
                "foir": _format_ratio(evidence["total_obligation_to_income"]),
                "probability": probability,
            }
        )
        # Verify the CSV directory parses identically to the workbook.
        if len(read_intake(directory).features.columns) != len(intake.features.columns):
            raise SystemExit(f"{name}: csv directory and xlsx disagree on feature count")

    summary = pd.DataFrame(rows)
    print(f"\nWrote {len(rows)} demo workbooks to {OUTPUT_DIR}\n")
    print(summary.to_string(index=False))
    print(
        "\nInternal research scores from an unvalidated candidate model. "
        "Decision support only; not an approval decision."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
