"""Portfolio screen: score a folder of intake packages into a ranked review queue.

This is a work-allocation view, not a book of decisions. It orders cases by estimated
repayment difficulty so a reviewer knows where to start, and it never states an outcome.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from new_application import DEMO_DIR, _ratio, _route

from credifast.data.workbook_intake import IntakeError, build_features, load_workbook
from credifast.modeling.collectible_inference import (
    CollectibleModelUnavailable,
    score_intake,
)

ROUTE_ORDER = [
    "ENHANCED MANUAL REVIEW",
    "DATA-LIMITED MANUAL REVIEW",
    "STANDARD MANUAL REVIEW",
]


@st.cache_data(show_spinner=False)
def score_folder(folder: str) -> pd.DataFrame:
    """Score every .xlsx intake package in a folder."""

    rows = []
    for workbook in sorted(Path(folder).glob("*.xlsx")):
        try:
            # Load the sheets directly so the applicant's own reference is available.
            # It is deliberately not a model input, so it never reaches the feature row.
            sheets = load_workbook(workbook)
            reference = sheets["application"].iloc[0].get("applicant_ref")
            intake = build_features(sheets)
            score = score_intake(intake)
        except (IntakeError, ValueError, KeyError, CollectibleModelUnavailable) as error:
            rows.append(
                {
                    "case": workbook.stem,
                    "reference": "-",
                    "probability": None,
                    "foir": None,
                    "sources": None,
                    "route": "NOT SCORED",
                    "note": str(error)[:120],
                }
            )
            continue
        route, _, _ = _route(len(score.sources_present), score.probability)
        rows.append(
            {
                "case": workbook.stem,
                "reference": workbook.stem if pd.isna(reference) else str(reference),
                "probability": score.probability,
                "foir": intake.evidence["total_obligation_to_income"],
                "sources": len(score.sources_present),
                "route": route,
                "note": "; ".join(intake.findings) if intake.findings else "",
            }
        )
    return pd.DataFrame(rows)


def render() -> None:
    st.title("Work the queue in the order the evidence suggests.")
    st.write(
        "Every intake package in a folder, scored and ordered by estimated repayment "
        "difficulty. This allocates reviewer attention. It records no decisions, and nothing "
        "here is an approval, a decline, or a limit."
    )

    folder = st.text_input(
        "Intake folder",
        value=str(DEMO_DIR),
        help="Any folder containing .xlsx intake workbooks in the documented schema.",
    )
    if not Path(folder).is_dir():
        st.markdown(
            f'<div class="notice red">No folder at {folder}.</div>', unsafe_allow_html=True
        )
        return

    frame = score_folder(folder)
    if frame.empty:
        st.markdown(
            '<div class="notice">No .xlsx intake packages found in that folder. Run '
            "scripts/generate_demo_workbooks.py to create the demonstration set.</div>",
            unsafe_allow_html=True,
        )
        return

    scored = frame.loc[frame["probability"].notna()]
    counts = frame["route"].value_counts()

    st.markdown(
        '<div class="section-rule"><strong>Queue composition</strong>'
        "<span>Where reviewer attention is required</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="score-line">
          <div class="score-cell"><span class="score-label">Cases in folder</span>
            <span class="score-value">{len(frame)}</span></div>
          <div class="score-cell"><span class="score-label">Enhanced review</span>
            <span class="score-value">{int(counts.get('ENHANCED MANUAL REVIEW', 0))}</span></div>
          <div class="score-cell"><span class="score-label">Data-limited review</span>
            <span class="score-value">{int(counts.get('DATA-LIMITED MANUAL REVIEW', 0))}</span></div>
          <div class="score-cell"><span class="score-label">Standard review</span>
            <span class="score-value">{int(counts.get('STANDARD MANUAL REVIEW', 0))}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice">Every case in every column goes to a human. The route names which '
        "queue, and nothing more.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-rule"><strong>Review queue</strong>'
        "<span>Ordered by estimated repayment difficulty</span></div>",
        unsafe_allow_html=True,
    )
    ordered = frame.sort_values(
        "probability", ascending=False, na_position="last"
    ).reset_index(drop=True)
    display = pd.DataFrame(
        {
            "Case": ordered["case"],
            "Reference": ordered["reference"],
            "Sources supplied": ordered["sources"].map(
                lambda value: "-" if pd.isna(value) else f"{int(value)} of 3"
            ),
            "Repayment-difficulty probability": ordered["probability"].map(
                lambda value: "not scored" if pd.isna(value) else f"{value:.2%}"
            ),
            "Obligation to income": ordered["foir"].map(_ratio),
            "Review route": ordered["route"],
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)

    gaps = ordered.loc[ordered["note"].astype(bool)]
    if not gaps.empty:
        st.markdown(
            '<div class="section-rule"><strong>Evidence gaps in this queue</strong>'
            "<span>What is missing, case by case</span></div>",
            unsafe_allow_html=True,
        )
        for _, row in gaps.iterrows():
            st.markdown(
                f'<div class="evidence-row"><span class="evidence-code">GAP</span>'
                f'<span class="evidence-copy"><strong>{row["case"]}</strong> — {row["note"]}</span>'
                f'<span class="evidence-weight">{row["route"].split()[0].lower()}</span></div>',
                unsafe_allow_html=True,
            )

    if not scored.empty:
        st.markdown(
            '<div class="section-rule"><strong>Distribution</strong>'
            "<span>Estimated repayment difficulty across the folder</span></div>",
            unsafe_allow_html=True,
        )
        chart = scored.set_index("case")["probability"]
        st.bar_chart(chart, height=220)

    st.markdown(
        '<div class="notice">Internal research scores from an unvalidated candidate model, '
        "selected on a calibration partition with no unbiased final evaluation. Overall ROC-AUC "
        "is 0.7404, but for applicants aged 18-24 it is <strong>0.6637</strong>; ranking within "
        "that band is weaker and the overall figure does not describe it. Ordering this queue by "
        "score will therefore surface younger applicants less reliably. See "
        "docs/PHASE7_COLLECTIBLE_FAIRNESS.md.</div>",
        unsafe_allow_html=True,
    )
