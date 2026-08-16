"""Scenario analysis for the collectible-input candidate.

Only declared current terms move. Historical evidence -- the credit report, monthly
delinquency and account behaviour -- is immutable: a reviewer may ask what happens if the
applicant borrows less, not what happens if their arrears disappear.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from new_application import (
    DEMO_DIR,
    DEMO_WORKBOOKS,
    _amount,
    _ratio,
    _route,
)

from credifast.data.workbook_intake import build_features, load_workbook
from credifast.modeling.collectible_inference import (
    CollectibleModelUnavailable,
    score_intake,
)

# Multipliers applied to requested credit and proposed repayment together, so the implied
# tenor stays fixed. Varying credit alone would silently change the repayment schedule.
SWEEP_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)

EDITABLE_TERMS = ("annual_income", "requested_credit", "annual_annuity", "goods_price")


def _apply_terms(sheets: dict[str, pd.DataFrame], terms: dict[str, float]):
    """Rebuild features with new declared terms and untouched history sheets."""

    revised = dict(sheets)
    application = sheets["application"].copy()
    for column, value in terms.items():
        if value is not None:
            application.loc[application.index[0], column] = value
    revised["application"] = application
    return build_features(revised)


def _evaluate(sheets: dict[str, pd.DataFrame], terms: dict[str, float]) -> dict | None:
    intake = _apply_terms(sheets, terms)
    try:
        score = score_intake(intake)
    except CollectibleModelUnavailable:
        return None
    route, _, _ = _route(len(score.sources_present), score.probability)
    return {
        "probability": score.probability,
        "foir": intake.evidence["total_obligation_to_income"],
        "debt_to_income": intake.evidence["external_debt_to_income"],
        "route": route,
        "sources": len(score.sources_present),
    }


def _delta_words(scenario: float, baseline: float) -> str:
    change = scenario - baseline
    if abs(change) < 0.0001:
        return "unchanged from baseline"
    direction = "higher" if change > 0 else "lower"
    return f"{abs(change) * 100:.2f} percentage points {direction} than baseline"


def render() -> None:
    st.title("Change the terms, not the history.")
    st.write(
        "Scenario analysis moves only the declared terms of the current request. The credit "
        "report, monthly delinquency record and account behaviour stay exactly as supplied, "
        "because a reviewer cannot rewrite an applicant's past."
    )

    label = st.selectbox("Case", list(DEMO_WORKBOOKS), key="scenario_case")
    workbook = DEMO_DIR / f"{DEMO_WORKBOOKS[label]}.xlsx"
    if not workbook.exists():
        st.markdown(
            '<div class="notice red">Demonstration files are missing. Run '
            "scripts/generate_demo_workbooks.py first.</div>",
            unsafe_allow_html=True,
        )
        return

    sheets = load_workbook(workbook)
    baseline_terms = {
        column: sheets["application"].iloc[0].get(column) for column in EDITABLE_TERMS
    }
    baseline = _evaluate(sheets, {})
    if baseline is None:
        st.markdown(
            '<div class="notice red">The candidate model is not available locally.</div>',
            unsafe_allow_html=True,
        )
        return

    controls, outcome = st.columns([0.82, 1.18], gap="large")

    with controls:
        st.markdown(
            '<div class="section-rule"><strong>Declared terms</strong>'
            "<span>The only editable inputs</span></div>",
            unsafe_allow_html=True,
        )
        income = st.number_input(
            "Annual income",
            min_value=1.0,
            value=float(baseline_terms["annual_income"]),
            step=10_000.0,
        )
        credit = st.number_input(
            "Requested credit",
            min_value=1.0,
            value=float(baseline_terms["requested_credit"]),
            step=10_000.0,
        )
        annuity = st.number_input(
            "Annual proposed repayment",
            min_value=1.0,
            value=float(baseline_terms["annual_annuity"]),
            step=5_000.0,
        )
        goods_raw = baseline_terms["goods_price"]
        goods = st.number_input(
            "Financed asset value",
            min_value=0.0,
            value=float(goods_raw) if pd.notna(goods_raw) else 0.0,
            step=10_000.0,
        )
        st.markdown(
            '<div class="smallprint">Historical evidence is locked. Source families, tradelines '
            "and delinquency months are read from the case as supplied and cannot be edited "
            "here.</div>",
            unsafe_allow_html=True,
        )

    scenario = _evaluate(
        sheets,
        {
            "annual_income": income,
            "requested_credit": credit,
            "annual_annuity": annuity,
            "goods_price": goods or None,
        },
    )

    with outcome:
        route, note, tone = _route(scenario["sources"], scenario["probability"])
        st.markdown(
            f"""
            <div class="route-band">
              <span class="route-label">Human review route under these terms</span>
              <span class="route-value">{route}</span>
              <span class="route-note">{note}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-rule"><strong>Baseline versus scenario</strong>'
            "<span>Same applicant, same history, different request</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="score-line">
              <div class="score-cell"><span class="score-label">Baseline probability</span>
                <span class="score-value">{baseline['probability']:.2%}</span></div>
              <div class="score-cell"><span class="score-label">Scenario probability</span>
                <span class="score-value">{scenario['probability']:.2%}</span></div>
              <div class="score-cell"><span class="score-label">Baseline FOIR</span>
                <span class="score-value">{_ratio(baseline['foir'])}</span></div>
              <div class="score-cell"><span class="score-label">Scenario FOIR</span>
                <span class="score-value">{_ratio(scenario['foir'])}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="notice">Scenario estimate is '
            f"{_delta_words(scenario['probability'], baseline['probability'])}. Movement in the "
            "estimate reflects the changed request only; no historical evidence was altered."
            "</div>",
            unsafe_allow_html=True,
        )
        if tone == "red":
            st.markdown(
                '<div class="notice red">These terms place the case in the enhanced review '
                "queue. That is a routing outcome, not a decline.</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="section-rule"><strong>Request sweep</strong>'
            "<span>Credit and repayment scaled together, tenor held constant</span></div>",
            unsafe_allow_html=True,
        )
        rows = []
        for multiplier in SWEEP_MULTIPLIERS:
            swept = _evaluate(
                sheets,
                {
                    "annual_income": income,
                    "requested_credit": credit * multiplier,
                    "annual_annuity": annuity * multiplier,
                    "goods_price": (goods * multiplier) if goods else None,
                },
            )
            rows.append(
                {
                    "Requested credit": _amount(credit * multiplier),
                    "Annual repayment": _amount(annuity * multiplier),
                    "Obligation to income": _ratio(swept["foir"]),
                    "Repayment-difficulty probability": f"{swept['probability']:.2%}",
                    "Review route": swept["route"],
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown(
            '<div class="notice">The sweep shows how the estimate and the obligation ratio move '
            "with the size of the request. It is not an offer, a sanctioned amount, a limit, or "
            "a price. Every row still requires human review.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="notice">Internal research score from an unvalidated candidate model. '
            "Overall ROC-AUC is 0.7404, but for applicants aged 18-24 it is <strong>0.6637</strong>; "
            "the overall figure does not describe that band. See "
            "docs/PHASE7_COLLECTIBLE_FAIRNESS.md.</div>",
            unsafe_allow_html=True,
        )
