"""Intake screen for the collectible-input candidate.

Scores an applicant from data they supply, rather than by looking up an identifier that
already exists in the training population. Separate from the frozen V1 review workbench,
which keeps its own model, routing and evidence.

Layout follows the content, not one template. A demonstration case or an upload is a small
selection, so it keeps the two-column docket. Direct entry is a full record plus four
ledgers, so it takes the full measure and sends its result below.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from field_policy import (
    affordability_gaps,
    load_policy,
    missing_expected,
    required_application_fields,
)
from intake_style import inject as inject_style

from credifast.data.workbook_intake import IntakeError, build_features, read_intake
from credifast.modeling.collectible_inference import (
    CollectibleModelUnavailable,
    score_intake,
)

DEMO_DIR = Path(__file__).resolve().parents[1] / "examples" / "demo"

DEMO_WORKBOOKS = {
    "Clean full file — three sources, long clean record": "clean_full_file",
    "New to credit — no credit report supplied": "thin_new_to_credit",
    "Recent delinquency — arrears inside the last three months": "recent_delinquency",
    "High obligation — clean record, obligations near income": "high_obligation",
    "Credit hungry — enquiry burst and several new accounts": "credit_hungry",
}

# Trained categorical domains. Values outside these encode as unknown and score out of
# distribution, so the form offers only what the model has actually seen.
CONTRACT_TYPES = ["Cash loans", "Revolving loans"]
INCOME_TYPES = [
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Unemployed",
    "Student",
    "Businessman",
    "Maternity leave",
]
EDUCATION_TYPES = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
HOUSING_TYPES = [
    "House / apartment",
    "With parents",
    "Municipal apartment",
    "Rented apartment",
    "Office apartment",
    "Co-op apartment",
]
CREDIT_TYPES = [
    "Consumer credit",
    "Credit card",
    "Car loan",
    "Mortgage",
    "Microloan",
    "Loan for business development",
    "Another type of loan",
]
TRADELINE_STATUSES = ["Active", "Closed", "Sold", "Bad debt"]
ACCOUNT_KINDS = ["revolving", "installment"]
CONTRACT_STATUSES = ["Active", "Completed"]

SOURCE_FAMILIES = (
    ("credit report", "Credit report"),
    ("revolving account history", "Revolving history"),
    ("installment account history", "Installment history"),
)

# Research routing boundary for the demonstration, taken from the 20% review-capacity
# operating point in the fairness audit. It is an audit-derived reference, not a lending
# decision threshold, and it never produces an approval or a decline.
ENHANCED_REVIEW_BOUNDARY = 0.114421


# --- worked example rows -----------------------------------------------------------------
# Prefilled so a reviewer sees the shape of each ledger immediately. Every row is editable
# and deletable; none of it is precomputed evidence.

def _example_tradelines() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "account_ref": "TL-01",
                "credit_type": "Consumer credit",
                "status": "Active",
                "opened_days_ago": 640,
                "sanctioned_amount": 300_000.0,
                "credit_limit": 0.0,
                "current_balance": 186_000.0,
                "overdue_amount": 0.0,
                "days_overdue": 0,
                "annuity": 36_000.0,
                "times_prolonged": 0,
            },
            {
                "account_ref": "TL-02",
                "credit_type": "Credit card",
                "status": "Active",
                "opened_days_ago": 1180,
                "sanctioned_amount": 150_000.0,
                "credit_limit": 150_000.0,
                "current_balance": 42_500.0,
                "overdue_amount": 0.0,
                "days_overdue": 0,
                "annuity": 18_000.0,
                "times_prolonged": 0,
            },
        ]
    )


def _example_delay_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"account_ref": "TL-01", "dpd_strip": "000000000000000000000"},
            {"account_ref": "TL-02", "dpd_strip": "000010000000000000000000000000000000"},
        ]
    )


def _example_account_months() -> pd.DataFrame:
    rows = []
    for month in range(6):
        rows.append(
            {
                "account_ref": "TL-02",
                "account_kind": "revolving",
                "months_ago": month,
                "dpd_days": 0,
                "dpd_days_tolerant": 0,
                "contract_status": "Active",
                "balance": 42_500.0 + month * 1_500,
                "credit_limit": 150_000.0,
                "installments_total": None,
                "installments_remaining": None,
            }
        )
    for month in range(6):
        rows.append(
            {
                "account_ref": "TL-01",
                "account_kind": "installment",
                "months_ago": month,
                "dpd_days": 0,
                "dpd_days_tolerant": 0,
                "contract_status": "Active",
                "balance": None,
                "credit_limit": None,
                "installments_total": 36,
                "installments_remaining": 15 + month,
            }
        )
    return pd.DataFrame(rows)


def _example_enquiries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"days_ago": 45, "purpose": "Credit application"},
            {"days_ago": 210, "purpose": "Credit application"},
        ]
    )


# --- formatting --------------------------------------------------------------------------


def _ratio(value) -> str:
    """Render a ratio, or say so plainly when it is unavailable. Never zero."""

    if value is None or pd.isna(value):
        return "not available"
    return f"{value * 100:.1f}%"


def _amount(value) -> str:
    if value is None or pd.isna(value):
        return "not available"
    return f"{value:,.0f}"


def _route(
    sources_present: int,
    probability: float,
    missing_expected_count: int = 0,
) -> tuple[str, str, str]:
    """Return (route label, plain-language note, notice class).

    Thin and partial files stay in data-limited manual review whatever the point estimate
    says, because their evidence base cannot support a narrower route.

    An absent EXPECTED field does the same. Those fields were measured, not assumed: each
    one moved at least 1.00% of calibration cases between review queues when withheld.
    See docs/INPUT_COVERAGE_AUDIT.md.
    """

    if sources_present >= 3 and missing_expected_count:
        return (
            "DATA-LIMITED MANUAL REVIEW",
            (
                "All three source families were supplied, but evidence the model measurably "
                "depends on is missing. A reviewer must close the gap before this estimate "
                "carries its usual weight."
            ),
            "",
        )
    if sources_present <= 1:
        return (
            "DATA-LIMITED MANUAL REVIEW",
            (
                "Little or no credit history was supplied. A reviewer must gather evidence "
                "before this case can be assessed further."
            ),
            "",
        )
    if sources_present == 2:
        return (
            "DATA-LIMITED MANUAL REVIEW",
            (
                "Part of the credit history is missing. The estimate stands on an incomplete "
                "evidence base and needs a reviewer."
            ),
            "",
        )
    if probability >= ENHANCED_REVIEW_BOUNDARY:
        return (
            "ENHANCED MANUAL REVIEW",
            (
                "All three source families were supplied and the estimate sits above the "
                "research review boundary. Send to the enhanced review queue."
            ),
            "red",
        )
    return (
        "STANDARD MANUAL REVIEW",
        (
            "All three source families were supplied and the estimate sits below the research "
            "review boundary. Send to the standard review queue."
        ),
        "green",
    )


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows the reviewer left blank without discarding legitimate zeros."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=frame.columns if frame is not None else [])
    return frame.dropna(how="all").reset_index(drop=True)


def _render_tally(present: list[str], absent: list[str]) -> None:
    st.markdown(
        '<div class="tally">'
        + "".join(
            f'<div class="tally-cell"><span class="tally-name">{label}</span>'
            f'<span class="tally-state {"have" if key in present else "gap"}">'
            f'{"supplied" if key in present else "not supplied"}</span></div>'
            for key, label in SOURCE_FAMILIES
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    if absent:
        st.markdown(
            '<div class="notice">Features from the families marked not supplied stay missing. '
            "They are never read as zero, and the case routes on the evidence that exists.</div>",
            unsafe_allow_html=True,
        )


# --- manual entry ------------------------------------------------------------------------


def _manual_form() -> dict[str, pd.DataFrame] | None:
    """Full-width record plus four ledgers. Returns sheets on submit."""

    policy = load_policy()
    required = set(required_application_fields(policy))

    def label(field: str, text: str) -> str:
        """Mark required fields in the label itself, not by colour alone."""

        return f"{text} · required" if field in required else text

    with st.form("collectible_manual", border=False):
        st.markdown(
            '<div class="section-rule"><strong>Requested facility</strong>'
            "<span>Declared current terms</span></div>",
            unsafe_allow_html=True,
        )
        if required:
            st.markdown(
                '<div class="ledger-note">Fields marked <strong>required</strong> are rejected '
                "if left blank, because the record cannot be built without them. Everything "
                "else is accepted missing and reported as a gap — nothing is filled in with "
                "zero on your behalf.</div>",
                unsafe_allow_html=True,
            )
        st.markdown('<div class="field-row">', unsafe_allow_html=True)
        request = st.columns(4, gap="medium")
        with request[0]:
            reference = st.text_input("Applicant reference", value="MANUAL-001")
        with request[1]:
            requested_credit = st.number_input(
                label("requested_credit", "Requested credit"), min_value=1.0, value=600_000.0, step=10_000.0
            )
        with request[2]:
            annual_annuity = st.number_input(
                label("annual_annuity", "Annual repayment"), min_value=1.0, value=168_000.0, step=5_000.0
            )
        with request[3]:
            goods_price = st.number_input(
                "Asset value", min_value=0.0, value=580_000.0, step=10_000.0
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-rule"><strong>Applicant</strong>'
            "<span>Declared profile and income</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="field-row">', unsafe_allow_html=True)
        profile = st.columns(4, gap="medium")
        with profile[0]:
            annual_income = st.number_input(
                label("annual_income", "Annual income"), min_value=1.0, value=720_000.0, step=10_000.0
            )
        with profile[1]:
            income_type = st.selectbox(label("income_type", "Income type"), INCOME_TYPES)
        with profile[2]:
            contract_type = st.selectbox(label("contract_type", "Contract type"), CONTRACT_TYPES)
        with profile[3]:
            education_type = st.selectbox(label("education_type", "Education"), EDUCATION_TYPES)

        holdings = st.columns(4, gap="medium")
        with holdings[0]:
            housing_type = st.selectbox(label("housing_type", "Housing"), HOUSING_TYPES)
        with holdings[1]:
            owns_realty = st.selectbox(label("owns_realty", "Owns property"), ["N", "Y"])
        with holdings[2]:
            owns_car = st.selectbox(label("owns_car", "Owns a car"), ["N", "Y"])
        with holdings[3]:
            car_age_years = st.number_input(
                "Car age, years", min_value=0.0, value=0.0, step=1.0,
                help="Left at zero and ignored when no car is owned.",
            )

        stability = st.columns(4, gap="medium")
        with stability[0]:
            employment_years = st.number_input(
                label("employment_years", "Years in current job"), min_value=0.0, value=3.5, step=0.5
            )
        with stability[1]:
            address_vintage_years = st.number_input(
                "Years at address", min_value=0.0, value=4.0, step=0.5
            )
        with stability[2]:
            id_document_age_years = st.number_input(
                "ID document age, years", min_value=0.0, value=3.0, step=0.5
            )
        with stability[3]:
            contact = st.selectbox(
                "Contact on record",
                ["Email and work phone", "Email only", "Work phone only", "Neither"],
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-rule"><strong>Credit history</strong>'
            "<span>Four ledgers, as a credit report presents them</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ledger-note">Rows are prefilled with a worked example so the shape of '
            "each ledger is visible. Edit them, add rows, or delete every row to submit a "
            "<strong>thin file</strong> with no credit history at all.</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ledger-tabs">', unsafe_allow_html=True)
        accounts_tab, delay_tab, behaviour_tab, enquiry_tab = st.tabs(
            ["Accounts", "Delay history", "Account behaviour", "Enquiries"]
        )

        with accounts_tab:
            st.markdown(
                '<div class="ledger-note">One row per credit account on the report. '
                "<strong>Opened days ago</strong> drives recency; <strong>annuity</strong> is the "
                "account instalment and feeds the obligation ratio.</div>",
                unsafe_allow_html=True,
            )
            tradelines = st.data_editor(
                _example_tradelines(),
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key="manual_tradelines",
                column_config={
                    "account_ref": st.column_config.TextColumn("Account", width="small"),
                    "credit_type": st.column_config.SelectboxColumn(
                        "Type", options=CREDIT_TYPES, width="medium"
                    ),
                    "status": st.column_config.SelectboxColumn(
                        "Status", options=TRADELINE_STATUSES, width="small"
                    ),
                    "opened_days_ago": st.column_config.NumberColumn(
                        "Opened days ago", min_value=0, step=1
                    ),
                    "sanctioned_amount": st.column_config.NumberColumn(
                        "Sanctioned", format="%.0f"
                    ),
                    "credit_limit": st.column_config.NumberColumn("Limit", format="%.0f"),
                    "current_balance": st.column_config.NumberColumn("Balance", format="%.0f"),
                    "overdue_amount": st.column_config.NumberColumn("Overdue", format="%.0f"),
                    "days_overdue": st.column_config.NumberColumn("Days overdue", min_value=0),
                    "annuity": st.column_config.NumberColumn("Instalment", format="%.0f"),
                    "times_prolonged": st.column_config.NumberColumn("Prolonged", min_value=0),
                },
            )

        with delay_tab:
            st.markdown(
                '<div class="ledger-note">Monthly repayment status per account, most recent '
                "month first, exactly as a report prints it. <strong>0</strong> paid on time, "
                "<strong>1</strong>–<strong>5</strong> increasing delinquency, <strong>C</strong> "
                "closed, <strong>X</strong> not reported. The account reference must match the "
                "Accounts ledger.</div>",
                unsafe_allow_html=True,
            )
            delay_history = st.data_editor(
                _example_delay_history(),
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key="manual_delay",
                column_config={
                    "account_ref": st.column_config.TextColumn("Account", width="small"),
                    "dpd_strip": st.column_config.TextColumn(
                        "Monthly status, newest first", width="large"
                    ),
                },
            )

        with behaviour_tab:
            st.markdown(
                '<div class="ledger-note">Month-by-month balance, limit and days past due for '
                "revolving accounts, and remaining instalments for loans. This ledger is what "
                "separates a partial file from a complete one.</div>",
                unsafe_allow_html=True,
            )
            account_months = st.data_editor(
                _example_account_months(),
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key="manual_behaviour",
                column_config={
                    "account_ref": st.column_config.TextColumn("Account", width="small"),
                    "account_kind": st.column_config.SelectboxColumn(
                        "Kind", options=ACCOUNT_KINDS, width="small"
                    ),
                    "months_ago": st.column_config.NumberColumn("Months ago", min_value=0),
                    "dpd_days": st.column_config.NumberColumn("Days past due", min_value=0),
                    "dpd_days_tolerant": st.column_config.NumberColumn(
                        "Days past due, tolerant", min_value=0
                    ),
                    "contract_status": st.column_config.SelectboxColumn(
                        "Status", options=CONTRACT_STATUSES, width="small"
                    ),
                    "balance": st.column_config.NumberColumn("Balance", format="%.0f"),
                    "credit_limit": st.column_config.NumberColumn("Limit", format="%.0f"),
                    "installments_total": st.column_config.NumberColumn("Instalments"),
                    "installments_remaining": st.column_config.NumberColumn("Remaining"),
                },
            )

        with enquiry_tab:
            st.markdown(
                '<div class="ledger-note">One row per credit enquiry. Bands are separate '
                "periods, not nested totals: a search 45 days ago counts in the quarter band "
                "alone.</div>",
                unsafe_allow_html=True,
            )
            enquiries = st.data_editor(
                _example_enquiries(),
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key="manual_enquiries",
                column_config={
                    "days_ago": st.column_config.NumberColumn("Days ago", min_value=0),
                    "purpose": st.column_config.TextColumn("Purpose", width="medium"),
                },
            )
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Evaluate case", type="primary", width="stretch")

    if not submitted:
        return None

    application = {
        "applicant_ref": reference,
        "annual_income": annual_income,
        "requested_credit": requested_credit,
        "annual_annuity": annual_annuity,
        # Optional fields stay missing rather than becoming zero.
        "goods_price": goods_price or None,
        "contract_type": contract_type,
        "income_type": income_type,
        "education_type": education_type,
        "housing_type": housing_type,
        "owns_car": owns_car,
        "car_age_years": car_age_years if owns_car == "Y" else None,
        "owns_realty": owns_realty,
        "employment_years": employment_years,
        "address_vintage_years": address_vintage_years,
        "id_document_age_years": id_document_age_years,
        "has_email": "Y" if contact in {"Email and work phone", "Email only"} else "N",
        "has_work_phone": "Y" if contact in {"Email and work phone", "Work phone only"} else "N",
    }
    return {
        "application": pd.DataFrame([application]),
        "tradelines": _clean(tradelines),
        "dpd_history": _clean(delay_history),
        "account_months": _clean(account_months),
        "enquiries": _clean(enquiries),
    }


# --- result ------------------------------------------------------------------------------


def _render_parse_summary(intake) -> None:
    evidence = intake.evidence
    st.markdown(
        '<div class="section-rule"><strong>What was supplied</strong>'
        "<span>Read from the intake package before scoring</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="score-line">
          <div class="score-cell"><span class="score-label">Accounts</span>
            <span class="score-value">{evidence['tradeline_count']}</span></div>
          <div class="score-cell"><span class="score-label">Monthly delay records</span>
            <span class="score-value">{evidence['dpd_month_count']}</span></div>
          <div class="score-cell"><span class="score-label">Account months</span>
            <span class="score-value">{evidence['account_month_count']}</span></div>
          <div class="score-cell"><span class="score-label">Enquiries</span>
            <span class="score-value">{evidence['enquiry_count']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for finding in intake.findings:
        st.markdown(f'<div class="notice">Data note: {finding}</div>', unsafe_allow_html=True)


def _render_result(intake, score) -> None:
    evidence = intake.evidence
    present = list(score.sources_present)
    absent = list(score.sources_absent)
    gaps = missing_expected(intake)
    route, note, tone = _route(len(present), score.probability, len(gaps))

    st.markdown(
        f"""
        <div class="route-band">
          <span class="route-label">Human review route</span>
          <span class="route-value">{route}</span>
          <span class="route-note">{note}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if tone == "red":
        st.markdown(
            '<div class="notice red">Enhanced review is required. This is not an approval, '
            "decline, price, or adverse-action reason.</div>",
            unsafe_allow_html=True,
        )
    elif tone == "green":
        st.markdown(
            '<div class="notice green">The evidence package is ready for its designated human '
            "review queue.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="notice">Data-gap guardrail active: the case stays in human review '
            "even when the point estimate is low.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-rule"><strong>Internal research score</strong>'
        "<span>Decision support from an unvalidated candidate model</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="score-line">
          <div class="score-cell"><span class="score-label">Repayment-difficulty probability</span>
            <span class="score-value">{score.probability:.2%}</span></div>
          <div class="score-cell"><span class="score-label">Raw probability</span>
            <span class="score-value">{score.probability:.6f}</span></div>
          <div class="score-cell"><span class="score-label">Source families supplied</span>
            <span class="score-value">{len(present)} of 3</span></div>
          <div class="score-cell"><span class="score-label">Features supplied</span>
            <span class="score-value">{score.features_supplied} of {score.feature_count}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-rule"><strong>Evidence supplied</strong>'
        "<span>Named sources, not a coverage score</span></div>",
        unsafe_allow_html=True,
    )
    _render_tally(present, absent)

    if gaps:
        st.markdown(
            '<div class="section-rule"><strong>Evidence the model depends on</strong>'
            "<span>Measured gaps, and what closes them</span></div>",
            unsafe_allow_html=True,
        )
        for gap in gaps:
            st.markdown(
                f'<div class="evidence-row"><span class="evidence-code">NEED</span>'
                f'<span class="evidence-copy">{gap["copy"]} <strong>To resolve:</strong> '
                f'{gap["resolve"]}</span>'
                f'<span class="evidence-weight">{gap["label"]}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="notice">Each gap above was measured, not assumed: withholding it '
            "moved at least 1.00% of calibration cases between review queues. See "
            "docs/INPUT_COVERAGE_AUDIT.md.</div>",
            unsafe_allow_html=True,
        )

    for note_copy in affordability_gaps(intake):
        st.markdown(f'<div class="notice">{note_copy}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-rule"><strong>Separate affordability screen</strong>'
        "<span>Obligation ratios, independent of the risk estimate</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="score-line">
          <div class="score-cell"><span class="score-label">Obligation to income</span>
            <span class="score-value">{_ratio(evidence['total_obligation_to_income'])}</span></div>
          <div class="score-cell"><span class="score-label">External debt to income</span>
            <span class="score-value">{_ratio(evidence['external_debt_to_income'])}</span></div>
          <div class="score-cell"><span class="score-label">Reported instalments</span>
            <span class="score-value">{_amount(evidence['reported_emi_total'])}</span></div>
          <div class="score-cell"><span class="score-label">Credit report</span>
            <span class="score-value">{'supplied' if evidence['has_bureau_history'] else 'absent'}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice">Affordability is measured separately from repayment risk. A clean '
        "repayment record with obligations near income can produce a low risk estimate and still "
        "fail affordability; both must be read together.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-rule"><strong>Model disclosure</strong>'
        "<span>Required context for every score on this screen</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="notice">{score.model_name} v{score.model_version}, '
        f"{score.calibrator_method} calibration. <strong>Not validated.</strong> Selected on a "
        "calibration partition only, with no unbiased final evaluation. Overall ROC-AUC is 0.7404, "
        "but for applicants aged 18-24 it is <strong>0.6637</strong> — that band is ranked less "
        "reliably and is routed into review more often. The overall figure does not describe it. "
        "See docs/PHASE7_COLLECTIBLE_FAIRNESS.md.</div>",
        unsafe_allow_html=True,
    )


def _score_and_render(intake) -> None:
    _render_parse_summary(intake)
    try:
        score = score_intake(intake)
    except CollectibleModelUnavailable as error:
        st.markdown(
            f'<div class="notice red">The candidate model is not available locally: {error}</div>',
            unsafe_allow_html=True,
        )
        return
    _render_result(intake, score)


def _awaiting() -> None:
    st.markdown(
        '<div class="section-rule"><strong>Assessment</strong>'
        "<span>Awaiting an intake package</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice">Choose an input method and select Evaluate case. Results are '
        "internal research scores for decision support, never approval decisions.</div>",
        unsafe_allow_html=True,
    )


def render() -> None:
    inject_style()
    st.title("Score an application from the data you hold.")
    st.write(
        "This screen scores an applicant from a supplied intake package rather than a known "
        "identifier. Load a demonstration case, upload a workbook, or enter the record and its "
        "credit history directly. Missing evidence stays missing and is never read as zero."
    )

    method = st.radio(
        "Input method",
        ["Demonstration case", "Upload workbook", "Direct entry"],
        horizontal=True,
        key="collectible_method",
        label_visibility="collapsed",
    )

    parsed = None
    error_message = None

    if method == "Direct entry":
        sheets = _manual_form()
        if sheets is not None:
            try:
                parsed = build_features(sheets)
            except IntakeError as error:
                error_message = str(error)
        if error_message:
            st.markdown(
                f'<div class="notice red">Intake rejected: {error_message}</div>',
                unsafe_allow_html=True,
            )
        elif parsed is not None:
            _score_and_render(parsed)
        return

    intake_column, outcome_column = st.columns([0.82, 1.18], gap="large")

    with intake_column:
        st.markdown(
            '<div class="section-rule"><strong>Application intake</strong>'
            "<span>Prepared package</span></div>",
            unsafe_allow_html=True,
        )
        if method == "Demonstration case":
            label = st.selectbox("Demonstration case", list(DEMO_WORKBOOKS), key="demo_choice")
            workbook = DEMO_DIR / f"{DEMO_WORKBOOKS[label]}.xlsx"
            st.markdown(
                f'<div class="smallprint">Reading {workbook.name}. '
                "Every value in these files is synthetic.</div>",
                unsafe_allow_html=True,
            )
            if not workbook.exists():
                error_message = (
                    f"{workbook.name} is missing. Run scripts/generate_demo_workbooks.py to "
                    "create the demonstration files."
                )
            elif st.button("Evaluate case", key="demo_submit", type="primary", width="stretch"):
                try:
                    parsed = read_intake(workbook)
                except IntakeError as error:
                    error_message = str(error)
        else:
            uploaded = st.file_uploader(
                "Intake workbook (.xlsx)",
                type=["xlsx"],
                key="collectible_upload",
                help=(
                    "Sheets: application, tradelines, dpd_history, account_months, enquiries. "
                    "The schema is documented in docs/COLLECTIBLE_INPUT_SPEC.md."
                ),
            )
            if uploaded is None:
                st.markdown(
                    '<div class="ledger-note">No workbook selected yet. Direct entry accepts the '
                    "same evidence without a file.</div>",
                    unsafe_allow_html=True,
                )
            elif st.button("Evaluate case", key="upload_submit", type="primary", width="stretch"):
                with tempfile.TemporaryDirectory() as workspace:
                    temporary = Path(workspace) / uploaded.name
                    temporary.write_bytes(uploaded.getvalue())
                    try:
                        parsed = read_intake(temporary)
                    except (IntakeError, ValueError) as error:
                        error_message = str(error)

    with outcome_column:
        if error_message:
            st.markdown(
                f'<div class="notice red">Intake rejected: {error_message}</div>',
                unsafe_allow_html=True,
            )
        elif parsed is None:
            _awaiting()
        else:
            _score_and_render(parsed)
