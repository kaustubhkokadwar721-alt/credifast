"""CrediFast reviewer workbench backed by the frozen local model runtime."""

from __future__ import annotations

import hashlib
import json
from base64 import b64encode
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from new_application import render as render_new_application
from portfolio import render as render_portfolio
from scenario import render as render_scenario

from credifast.input_sources import EDITABLE_FIELDS, normalize_intake_package
from credifast.model_runtime import LocalModelRuntime, get_local_model_runtime

DOCKET_FONT_PATH = Path(__file__).parent / "assets/SourceSerif4-Variable.woff2"
DOCKET_FONT_DATA = b64encode(DOCKET_FONT_PATH.read_bytes()).decode("ascii")

st.set_page_config(page_title="CrediFast Review Docket", page_icon="C", layout="wide")

st.markdown(
    """
<!--
THESIS: Credit risk is an evidence docket, not a verdict machine; the surface refuses generic KPI tiles and autonomous approval language.
OWN-WORLD: Ink navy, paper white, cobalt actions, ruled evidence rows, tabular numerals, square docket labels, and restrained amber exceptions.
STORY: A reviewer selects a real case, adjusts declared terms, evaluates it, understands the evidence and gaps, then routes it for human review.
FIRST VIEWPORT: A compact masthead sits above a two-column docket: editable case intake at left and a persistent decision band with evidence summary at right; Evaluate case is visible without scrolling.
FORM: Evidence Docket, position 4 in the grounded directions, seed 8b27f766.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md
-->
<style>
@font-face {
  font-family: "CrediFast Docket";
  src: url("data:font/woff2;base64,__DOCKET_FONT_DATA__") format("woff2");
  font-style: normal;
  font-weight: 200 900;
  font-display: swap;
}
:root {
  --ink: #13243a;
  --ink-soft: #415268;
  --paper: #f7f5ef;
  --surface: #ffffff;
  --line: #d6dbe2;
  --cobalt: #2457d6;
  --cobalt-dark: #173b94;
  --amber: #a45b08;
  --amber-bg: #fff4dc;
  --red: #a82d2d;
  --red-bg: #fff0ef;
  --green: #16614a;
  --green-bg: #eaf7f0;
}
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"], [data-testid="stToolbar"], .stDeployButton, #MainMenu { display: none !important; }
[data-testid="stSidebar"] { background: #eef1f5; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--ink-soft); }
.block-container { max-width: 1440px; padding-top: 1.4rem; padding-bottom: 4rem; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
h1 { font-family: "CrediFast Docket", Georgia, serif !important; font-size: clamp(2.2rem, 4.4vw, 3.8rem) !important; font-weight: 650 !important; line-height: 0.94 !important; letter-spacing: -.04em !important; }
h2 { margin-top: 2.4rem !important; }
p, label, li { color: var(--ink-soft); }
.masthead { display: flex; justify-content: space-between; align-items: end; border-bottom: 3px solid var(--ink); padding: 0 0 .9rem; margin-bottom: 1.2rem; }
.wordmark { font-size: .82rem; letter-spacing: .16em; font-weight: 800; color: var(--ink); }
.mast-meta { font-size: .76rem; text-align: right; color: var(--ink-soft); }
.status-dot { display: inline-block; width: .48rem; height: .48rem; background: var(--green); border-radius: 50%; margin-right: .35rem; }
.route-band { border-top: 6px solid var(--ink); border-bottom: 1px solid var(--ink); padding: 1.15rem 0 1rem; margin: .3rem 0 1rem; }
.route-label { font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 800; color: var(--ink-soft); }
.route-value { font-size: clamp(1.5rem, 3vw, 2.7rem); line-height: 1; font-weight: 760; letter-spacing: -.035em; color: var(--ink); margin: .35rem 0 .45rem; }
.route-note { font-size: .88rem; max-width: 70ch; color: var(--ink-soft); }
.docket-id { display: inline-block; background: var(--ink); color: white; font-size: .72rem; letter-spacing: .08em; padding: .32rem .52rem; font-weight: 750; }
.score-line { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid var(--line); background: var(--surface); margin: .8rem 0 1.1rem; }
.score-cell { padding: .82rem .9rem; border-right: 1px solid var(--line); min-width: 0; }
.score-cell:last-child { border-right: 0; }
.score-label { display: block; color: var(--ink-soft); font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 750; }
.score-value { display: block; color: var(--ink); font-size: 1.25rem; font-weight: 760; margin-top: .2rem; font-variant-numeric: tabular-nums; }
.section-rule { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid var(--ink); padding-bottom: .45rem; margin: 1.7rem 0 .85rem; }
.section-rule strong { color: var(--ink); font-size: 1rem; }
.section-rule span { color: var(--ink-soft); font-size: .74rem; }
.evidence-row { display: grid; grid-template-columns: 4rem 1fr auto; gap: .8rem; align-items: start; border-bottom: 1px solid var(--line); padding: .78rem .1rem; }
.evidence-row.provenance { grid-template-columns: 1fr auto; }
.evidence-code { color: var(--cobalt-dark); font-weight: 800; font-size: .76rem; }
.evidence-copy { color: var(--ink); font-size: .9rem; line-height: 1.4; }
.evidence-weight { color: var(--ink-soft); font-size: .76rem; font-variant-numeric: tabular-nums; }
.notice { padding: .85rem 1rem; border: 1px solid #e7c98a; border-top: 3px solid var(--amber); background: var(--amber-bg); color: #6b430e; font-size: .86rem; margin: .7rem 0; }
.notice.red { border-color: #e0afaa; border-top-color: var(--red); background: var(--red-bg); color: #722323; }
.notice.green { border-color: #a9d2bf; border-top-color: var(--green); background: var(--green-bg); color: #174e3d; }
.metric-ledger { width: 100%; border-collapse: collapse; background: white; }
.metric-ledger td { border-bottom: 1px solid var(--line); padding: .72rem .2rem; }
.metric-ledger td:last-child { text-align: right; font-weight: 760; color: var(--ink); font-variant-numeric: tabular-nums; }
.smallprint { color: var(--ink-soft); font-size: .76rem; line-height: 1.5; }
.source-badges { display: flex; flex-wrap: wrap; gap: .28rem; margin: -.1rem 0 .32rem; }
.source-chip { display: inline-block; border: 1px solid var(--line); background: var(--surface); color: var(--ink-soft); font-size: .72rem; line-height: 1; letter-spacing: .075em; padding: .26rem .34rem; font-weight: 800; }
.source-chip.bank { border-color: var(--cobalt); color: var(--cobalt-dark); }
.source-chip.cibil { border-color: var(--cobalt); background: var(--paper); color: var(--cobalt-dark); }
.source-chip.lender { border-color: var(--ink); background: var(--ink); color: var(--surface); }
.stButton > button { border-radius: 2px; min-height: 2.75rem; font-weight: 750; }
.stButton > button[kind="primary"] { background: var(--cobalt); border-color: var(--cobalt); }
.stButton > button[kind="primary"], .stButton > button[kind="primary"] p { color: white !important; }
.stButton > button[kind="primary"]:hover { background: var(--cobalt-dark); border-color: var(--cobalt-dark); }
div[data-baseweb="select"] > div, [data-testid="stNumberInput"] input { border-radius: 2px !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.2rem; border-bottom: 1px solid var(--line); }
[data-testid="stTabs"] button { font-weight: 700; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); }
@keyframes docket-settle {
  from { transform: translateY(8px); filter: blur(2px); box-shadow: inset 0 6px 0 var(--cobalt); }
  to { transform: translateY(0); filter: blur(0); box-shadow: inset 0 0 0 transparent; }
}
.route-band { animation: docket-settle 520ms cubic-bezier(.16, 1, .3, 1) both; }
@media (prefers-reduced-motion: reduce) {
  .route-band { animation: none; }
}
@media (max-width: 760px) {
  .block-container { padding-top: .8rem; }
  .masthead { align-items: start; }
  .mast-meta { display: none; }
  .score-line { grid-template-columns: repeat(2, 1fr); }
  .score-cell:nth-child(2) { border-right: 0; }
  .score-cell:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
  .evidence-row { grid-template-columns: 3.2rem 1fr; }
  .evidence-weight { grid-column: 2; }
}
</style>
""".replace("__DOCKET_FONT_DATA__", DOCKET_FONT_DATA),
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_runtime() -> LocalModelRuntime:
    return get_local_model_runtime()


def money(value: float | None) -> str:
    return "Not available" if value is None else f"{value:,.0f}"


FIELD_PLAIN_NAMES = {
    "annual_income": "Yearly income",
    "requested_credit": "Amount being borrowed",
    "annual_annuity": "Yearly repayment on this loan",
    "goods_price": "Price of the item being bought",
}


def render_field_provenance() -> None:
    """Where each editable number comes from.

    This used to sit as a stack of uppercase chips above every input, where it was louder
    than the field labels and unreadable to anyone who does not already know what an
    Account Aggregator is. It belongs with the rest of the provenance, not on the form.
    """

    for field, plain_name in FIELD_PLAIN_NAMES.items():
        metadata = EDITABLE_FIELDS[field]
        chips = []
        for badge in metadata["source_badges"]:
            css_class = "source-chip"
            if "BANK" in badge or "AA" in badge:
                css_class += " bank"
            elif "CIBIL" in badge:
                css_class += " cibil"
            elif "LENDER" in badge:
                css_class += " lender"
            chips.append(f'<span class="{css_class}">{badge}</span>')
        st.markdown(
            f"""<div class="evidence-row provenance">
              <div class="evidence-copy"><strong>{plain_name}</strong><br>
                <span class="smallprint">{metadata['note']}</span></div>
              <div class="source-badges">{''.join(chips)}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def intake_fingerprint(payload: dict[str, int | float]) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def route_copy(route: str) -> str:
    """What happens next, and who does it. Never an outcome."""

    return {
        "STANDARD_REVIEW": (
            "What happens next: a credit reviewer picks this up in the normal manual review "
            "queue. Nothing has been approved or declined."
        ),
        "BOUNDARY_REVIEW": (
            "What happens next: a credit reviewer looks at this one closely, because the "
            "estimate sits close to a policy boundary and could fall either side of it."
        ),
        "DATA_LIMITED_REVIEW": (
            "What happens next: a credit reviewer must gather the missing records before this "
            "case can go further. It stays in manual review no matter how low the estimate is."
        ),
        "HIGH_RISK_REVIEW": (
            "What happens next: a senior credit reviewer takes this one. That is a queue, not "
            "a decline — no decision has been made."
        ),
        "AFFORDABILITY_REVIEW": (
            "What happens next: a credit reviewer checks affordability separately, because the "
            "proposed repayment is large relative to declared income."
        ),
    }.get(route, "What happens next: a credit reviewer handles this case in manual review.")


def natural_frequency(probability: float) -> str:
    """State a probability as a count of people, scaling the denominator to stay honest.

    A 0.31% estimate rendered as "about 0 out of 100" reads as certainty of repayment. The
    denominator therefore grows until the numerator is a number a reader can picture.
    """

    if probability >= 0.01:
        return f"about <strong>{probability * 100:.0f}</strong> out of every 100"
    if probability >= 0.001:
        return f"about <strong>{probability * 1000:.0f}</strong> out of every 1,000"
    if probability > 0:
        return "<strong>fewer than 1</strong> in every 1,000"
    return "<strong>none</strong> of"


def effect_size(log_odds: float) -> str:
    """Describe a contribution in words a non-specialist can act on.

    The exact log-odds figure stays available under Technical detail for auditors; it is
    not a number a reviewer can reason about at a glance.
    """

    magnitude = abs(log_odds)
    if magnitude >= 0.5:
        return "large effect"
    if magnitude >= 0.2:
        return "moderate effect"
    return "small effect"


def render_reasons(reasons: dict[str, Any] | None, side: str) -> None:
    items = (reasons or {}).get(side, [])
    if not items:
        st.caption("Nothing in this applicant's record pushed the estimate this way.")
        return
    direction = "raised" if side == "adverse" else "lowered"
    for item in items:
        st.markdown(
            f"""<div class="evidence-row">
              <div class="evidence-copy">{item['description']}</div>
              <div class="evidence-weight">{effect_size(item['log_odds_contribution'])}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with st.expander("Technical detail"):
        st.caption(
            f"Contributions that {direction} the estimate, in pre-calibration log odds. "
            "These describe how the model behaved, not why the applicant behaved as they did."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Code": item["code"],
                        "Factor group": item["description"],
                        "Log odds": round(item["log_odds_contribution"], 3),
                    }
                    for item in items
                ]
            ),
            hide_index=True,
            width="stretch",
        )


try:
    with st.spinner("Opening the frozen model docket..."):
        runtime = load_runtime()
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    st.error(f"The local scoring runtime could not open: {exc}")
    st.stop()

status = runtime.status()
st.markdown(
    f"""<div class="masthead">
      <div><div class="wordmark">CREDIFAST / REVIEW DOCKET</div></div>
      <div class="mast-meta"><span class="status-dot"></span>Frozen runtime ready<br>
      Model {status['model']['version']} / research use only</div>
    </div>""",
    unsafe_allow_html=True,
)

(
    review_tab,
    new_application_tab,
    scenario_tab,
    portfolio_tab,
    evidence_tab,
    governance_tab,
) = st.tabs(
    [
        "Review workbench",
        "New application",
        "Scenario analysis",
        "Portfolio",
        "Model evidence",
        "Responsible AI",
    ]
)

with new_application_tab:
    render_new_application()

with scenario_tab:
    render_scenario()

with portfolio_tab:
    render_portfolio()

with review_tab:
    profiles = runtime.profiles()
    profile_by_label = {profile["label"]: profile for profile in profiles}
    input_schema = runtime.input_schema()
    st.title("Review the evidence, not just the score.")
    st.write(
        "Pick an example applicant, or look one up by their application number. You can change "
        "the loan being asked for and see how the assessment moves. Their past borrowing record "
        "stays as it is."
    )

    intake, outcome = st.columns([0.82, 1.18], gap="large")
    with intake:
        st.markdown(
            '<div class="section-rule"><strong>The case</strong>'
            "<span>Pick an applicant, adjust the four editable numbers</span></div>",
            unsafe_allow_html=True,
        )
        intake_method = st.radio(
            "How to pick the case",
            ["Example applicant", "Look up by number", "Upload a file"],
            horizontal=True,
            help=(
                "The JSON package is the fastest repeatable entry method. It still requires an "
                "applicant ID present in the local demo population."
            ),
        )
        selected_profile = profiles[0]
        uploaded_package: dict[str, int | float] | None = None
        package_error = False
        if intake_method == "Example applicant":
            selected_label = st.selectbox("Choose one", list(profile_by_label))
            selected_profile = profile_by_label[selected_label]
            application_id = int(selected_profile["application_id"])
            st.markdown(
                f'<span class="docket-id">APPLICATION {application_id}</span>',
                unsafe_allow_html=True,
            )
            st.caption(selected_profile["description"])
        elif intake_method == "Look up by number":
            application_id = int(
                st.number_input(
                    "Known local applicant ID",
                    min_value=1,
                    value=int(selected_profile["application_id"]),
                    step=1,
                )
            )
            st.caption(
                "The frozen prototype can look up any ID in its 48,744-row local test population."
            )
        else:
            template_json = json.dumps(input_schema["json_template"], indent=2) + "\n"
            st.download_button(
                "Download JSON template",
                data=template_json,
                file_name="credifast_intake.json",
                mime="application/json",
                use_container_width=True,
            )
            uploaded = st.file_uploader(
                "Upload completed JSON package",
                type=["json"],
                help="Accepted keys are application_id and the four editable current terms.",
            )
            if uploaded is not None:
                try:
                    uploaded_package = normalize_intake_package(
                        json.loads(uploaded.getvalue().decode("utf-8"))
                    )
                    st.success(
                        f"Package loaded for application {uploaded_package['application_id']}."
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    st.error(f"The JSON package could not be loaded: {exc}")
                    package_error = True
            if uploaded_package is None:
                application_id = int(selected_profile["application_id"])
                if not package_error:
                    st.info("Upload the template to populate and evaluate a case.")
            else:
                application_id = int(uploaded_package["application_id"])

        case_lookup_valid = True
        try:
            base_inputs = runtime.application_inputs(application_id)
        except ValueError as exc:
            st.error(str(exc))
            base_inputs = runtime.application_inputs(int(selected_profile["application_id"]))
            application_id = int(selected_profile["application_id"])
            case_lookup_valid = False

        defaults = dict(base_inputs)
        if intake_method == "Example applicant":
            defaults.update(selected_profile.get("overrides", {}))
        elif uploaded_package is not None:
            defaults.update(
                {
                    key: value
                    for key, value in uploaded_package.items()
                    if key in EDITABLE_FIELDS
                }
            )
        fingerprint_payload = {"application_id": application_id, **defaults}
        field_key = f"{intake_method}-{intake_fingerprint(fingerprint_payload)}"
        input_left, input_right = st.columns(2)
        with input_left:
            annual_income = st.number_input(
                "Yearly income",
                min_value=1.0,
                value=float(defaults["annual_income"] or 1.0),
                step=5_000.0,
                key=f"income-{field_key}",
                help=EDITABLE_FIELDS["annual_income"]["note"],
            )
            annual_annuity = st.number_input(
                "Yearly repayment on this loan",
                min_value=1.0,
                value=float(defaults["annual_annuity"] or 1.0),
                step=1_000.0,
                key=f"annuity-{field_key}",
                help=EDITABLE_FIELDS["annual_annuity"]["note"],
            )
        with input_right:
            requested_credit = st.number_input(
                "Amount being borrowed",
                min_value=1.0,
                value=float(defaults["requested_credit"] or 1.0),
                step=5_000.0,
                key=f"credit-{field_key}",
                help=EDITABLE_FIELDS["requested_credit"]["note"],
            )
            goods_price = st.number_input(
                "Price of the item being bought",
                min_value=1.0,
                value=float(defaults["goods_price"] or 1.0),
                step=5_000.0,
                key=f"goods-{field_key}",
                help=EDITABLE_FIELDS["goods_price"]["note"],
            )
        st.markdown(
            '<div class="smallprint">These four are the only numbers you can change. '
            "Everything else about this applicant &mdash; their past borrowing, repayment "
            "record and account history &mdash; is read from the records on file and cannot "
            "be edited here.</div>",
            unsafe_allow_html=True,
        )
        default_outage = bool(selected_profile.get("simulate_history_unavailable")) and (
            intake_method == "Example applicant"
        )
        simulate_outage = st.checkbox(
            "Test what happens if this applicant's past records are missing",
            value=default_outage,
            help="A controlled failure-mode demonstration; it does not delete or alter local data.",
            key=f"outage-{field_key}",
        )
        can_evaluate = case_lookup_valid and not (
            intake_method == "Upload a file" and uploaded_package is None
        )
        evaluate = st.button(
            "Evaluate case",
            type="primary",
            use_container_width=True,
            disabled=not can_evaluate,
        )
        with st.expander(
            "Where these numbers come from"
        ):
            st.caption("The four numbers you can edit, and what each is normally taken from.")
            render_field_provenance()
            st.caption(
                f"The other {input_schema['selected_factor_count'] - 4} things the model reads "
                "are assembled from records on file or calculated automatically."
            )
            source_rows = pd.DataFrame(input_schema["source_summary"])[
                ["label", "factor_count", "integration_status"]
            ].rename(
                columns={
                    "label": "Primary source",
                    "factor_count": "Model-ready factors",
                    "integration_status": "Current status",
                }
            )
            source_rows["Current status"] = source_rows["Current status"].map(
                {
                    "not_connected": "Connector required",
                    "local_record_only": "Local record only",
                    "available": "Calculated now",
                    "provider_required": "Provider contract required",
                }
            )
            st.dataframe(source_rows, hide_index=True, use_container_width=True)
            factor_source_rows = pd.DataFrame(input_schema["factor_sources"])[
                ["factor", "source_label"]
            ].rename(columns={"factor": "Model factor", "source_label": "Primary source"})
            st.download_button(
                "Download complete factor-source map",
                data=factor_source_rows.to_csv(index=False),
                file_name="credifast_factor_sources.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(input_schema["counting_note"])
            st.warning(input_schema["current_runtime_limit"])

    if (evaluate or "latest_result" not in st.session_state) and can_evaluate:
        overrides = {
            "annual_income": annual_income,
            "requested_credit": requested_credit,
            "annual_annuity": annual_annuity,
            "goods_price": goods_price,
        }
        try:
            with st.spinner("Assembling evidence and calculating explanations..."):
                st.session_state.latest_result = runtime.score(
                    application_id,
                    overrides=overrides,
                    explain=True,
                    simulate_history_unavailable=simulate_outage,
                )
        except (ValueError, RuntimeError) as exc:
            st.error(f"The case could not be evaluated: {exc}")
            st.session_state.pop("latest_result", None)

    result = st.session_state.get("latest_result")
    with outcome:
        st.markdown(
            '<div class="section-rule"><strong>The assessment</strong>'
            "<span>Decision support for a human reviewer, never an automatic decision</span></div>",
            unsafe_allow_html=True,
        )
        if result:
            route = result["review_route"]
            st.markdown(
                f"""<div class="route-band">
                  <div class="route-label">Human review route</div>
                  <div class="route-value">{route.replace('_', ' ')}</div>
                  <div class="route-note">{route_copy(route)}</div>
                </div>
                <div class="score-line">
                  <div class="score-cell"><span class="score-label">Repayment-difficulty probability</span><span class="score-value">{result['repayment_difficulty_probability']:.2%}</span></div>
                  <div class="score-cell"><span class="score-label">Internal research score</span><span class="score-value">{result['internal_credit_score']}</span></div>
                  <div class="score-cell"><span class="score-label">Risk grade</span><span class="score-value">{result['risk_grade']}</span></div>
                  <div class="score-cell"><span class="score-label">Data confidence</span><span class="score-value">{result['confidence']}</span></div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<div class="smallprint">
                <strong>Reading these four:</strong>
                of applicants whose records look like this one, the model estimates
                {natural_frequency(result['repayment_difficulty_probability'])}
                would run into difficulty repaying.
                The internal research score ({result['internal_credit_score']}) and grade
                ({result['risk_grade']}) are the same estimate on a 300&ndash;900 scale and an
                A&ndash;E band, for easier comparison between cases &mdash; they are not a bureau
                score. Data confidence ({str(result['confidence']).lower()}) describes how complete
                this applicant's records were, not how likely they are to repay.
                </div>""",
                unsafe_allow_html=True,
            )

            if route == "DATA_LIMITED_REVIEW":
                st.markdown(
                    '<div class="notice">Data-gap guardrail active: the case remains in human review even when the point estimate is low.</div>',
                    unsafe_allow_html=True,
                )
            elif route in {"HIGH_RISK_REVIEW", "AFFORDABILITY_REVIEW"}:
                st.markdown(
                    '<div class="notice red">Enhanced review is required. Do not interpret this route as an approval, decline, price, or adverse-action reason.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="notice green">The evidence package is ready for its designated human review queue.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="section-rule"><strong>Records we had</strong>'
                "<span>What this estimate was built from</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<div class="smallprint">
                We found <strong>{result['available_history_sources']} of 5</strong> kinds of past
                borrowing record for this applicant. The more of these exist, the more the estimate
                rests on how they have actually repaid before, rather than on the application form
                alone.
                </div>""",
                unsafe_allow_html=True,
            )
            for reason in result["confidence_reasons"]:
                st.caption(reason)
            with st.expander("Technical detail"):
                coverage = pd.DataFrame(
                    [
                        {
                            "Historical source families": result["available_history_sources"],
                            "Availability segment": result["history_segment"],
                            "History model weight": result["history_weight"],
                            "History probability": result["history_component_probability"],
                            "Application probability": result["application_component_probability"],
                        }
                    ]
                )
                st.dataframe(
                    coverage,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "History model weight": st.column_config.NumberColumn(format="percent"),
                        "History probability": st.column_config.NumberColumn(format="percent"),
                        "Application probability": st.column_config.NumberColumn(
                            format="percent"
                        ),
                    },
                )
                st.caption(
                    "Two models score the case: one using past borrowing records, one using the "
                    "application form alone. The weight shows how much each contributed."
                )
        else:
            st.info("Evaluate a valid case to assemble the routing docket.")

    if result:
        st.markdown(
            '<div class="section-rule"><strong>What moved this estimate</strong>'
            "<span>The applicant's records, grouped</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="smallprint">These describe how the model weighed this applicant\'s '
            "records. They explain the estimate; they are not the reasons a lender would give "
            "an applicant, and they do not prove what caused what.</div>",
            unsafe_allow_html=True,
        )
        adverse, favorable = st.columns(2, gap="large")
        with adverse:
            st.subheader("Pushed the estimate up")
            render_reasons(result["reasons"], "adverse")
        with favorable:
            st.subheader("Pushed the estimate down")
            render_reasons(result["reasons"], "favorable")

        terms, affordability = st.columns(2, gap="large")
        with terms:
            st.markdown(
                '<div class="section-rule"><strong>Declared-term audit</strong><span>Original versus evaluated values</span></div>',
                unsafe_allow_html=True,
            )
            snapshots = []
            for name, original in result["input_snapshot"]["original"].items():
                snapshots.append(
                    {
                        "Field": name.replace("_", " ").title(),
                        "Original": money(original),
                        "Evaluated": money(result["input_snapshot"]["effective"][name]),
                        "Changed": name in result["input_snapshot"]["overrides"]
                        and result["input_snapshot"]["overrides"][name] != original,
                    }
                )
            st.dataframe(snapshots, hide_index=True, use_container_width=True)
        with affordability:
            aff = result["affordability"]
            st.markdown(
                '<div class="section-rule"><strong>Can they afford it?</strong>'
                "<span>Checked separately from repayment risk</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<table class="metric-ledger">
                  <tr><td>Affordability check</td><td>{aff['status'].replace('_', ' ')}</td></tr>
                  <tr><td>Repayment takes this share of income</td><td>{aff['proposed_repayment_to_income']:.1%}</td></tr>
                  <tr><td>Most this research rule allows</td><td>{aff['maximum_research_ratio']:.0%} of income</td></tr>
                  <tr><td>Borrowing that would reach that share</td><td>{money(aff['estimated_credit_at_maximum_ratio'])}</td></tr>
                </table>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="smallprint">This is a separate question from whether they repay on '
                "time. Someone with a spotless record can still be taking on more than their "
                "income supports, and this check is what catches that. It counts only the "
                "repayment proposed here &mdash; it cannot see loans held elsewhere.</div>",
                unsafe_allow_html=True,
            )
            st.caption(aff["note"])

        if result["data_quality_flags"]:
            st.warning("Data quality flags: " + ", ".join(result["data_quality_flags"]))
        with st.expander("Audit payload and version trace"):
            st.json(result)
        st.markdown(f'<p class="smallprint">{result["disclaimer"]}</p>', unsafe_allow_html=True)

with evidence_tab:
    st.title("Performance with the holdout still sealed.")
    st.write(
        "These are frozen results from 46,127 applicants. The holdout was evaluated once after model, "
        "calibration, and data-gap policy choices were locked."
    )
    final = runtime.final_metrics
    candidate = final["model_metrics"]["frozen_operational_candidate"]["ranking"]
    calibration = final["model_metrics"]["frozen_operational_candidate"]["calibration"]
    st.markdown(
        f"""<div class="score-line">
          <div class="score-cell"><span class="score-label">ROC AUC</span><span class="score-value">{candidate['roc_auc']:.3f}</span></div>
          <div class="score-cell"><span class="score-label">Average precision</span><span class="score-value">{candidate['average_precision']:.3f}</span></div>
          <div class="score-cell"><span class="score-label">Brier score</span><span class="score-value">{candidate['brier_score']:.3f}</span></div>
          <div class="score-cell"><span class="score-label">Calibration slope</span><span class="score-value">{calibration['calibration_slope']:.3f}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice green">All four predeclared acceptance gates passed. These results support a research demonstration, not production approval.</div>',
        unsafe_allow_html=True,
    )

    rows = []
    names = {
        "application_only": "Application-only baseline",
        "history_only": "History-enriched challenger",
        "frozen_operational_candidate": "Frozen gap-aware candidate",
    }
    for key, bundle in final["model_metrics"].items():
        ranking = bundle["ranking"]
        rows.append(
            {
                "Model": names[key],
                "ROC AUC": ranking["roc_auc"],
                "Average precision": ranking["average_precision"],
                "Top 20% event capture": ranking["top_20_percent"]["event_capture_rate"],
                "Brier score": ranking["brier_score"],
            }
        )
    comparison = pd.DataFrame(rows).set_index("Model")
    st.subheader("Frozen model comparison")
    st.dataframe(
        comparison,
        use_container_width=True,
        column_config={column: st.column_config.NumberColumn(format="%.3f") for column in comparison},
    )
    chart_data = comparison[["ROC AUC", "Average precision", "Top 20% event capture"]]
    st.bar_chart(chart_data, horizontal=True, color=["#2457d6", "#7092e8", "#13243a"])

    st.subheader("What the numbers mean")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            f"""<table class="metric-ledger">
              <tr><td>Holdout applicants</td><td>{final['evaluation_protocol']['rows']:,}</td></tr>
              <tr><td>Repayment-difficulty events</td><td>{final['evaluation_protocol']['events']:,}</td></tr>
              <tr><td>Top 20% event capture</td><td>{candidate['top_20_percent']['event_capture_rate']:.1%}</td></tr>
              <tr><td>Average precision / prevalence</td><td>{final['acceptance_gates']['observed']['average_precision_multiple_of_prevalence']:.2f}x</td></tr>
            </table>""",
            unsafe_allow_html=True,
        )
    with right:
        st.write(
            "Ranking quality improves materially over the application-only baseline. The operational candidate "
            "preserves that gain while routing incomplete-history cases for human review."
        )
        st.write(
            "Probability calibration is close to the observed rate overall, but thin-file segments remain a "
            "known limitation and are never presented as high-confidence cases."
        )

with governance_tab:
    st.title("The guardrails are part of the product.")
    fairness = runtime.fairness_audit
    st.write(
        "Sensitive fields were retained only for post-model auditing. Gender and age were excluded by policy "
        "from selected prediction features, while outcome and review-rate gaps remain visible below."
    )
    st.markdown(
        '<div class="notice">Fairness disposition: share with caveats. The audit is descriptive, not proof of legal or production fairness.</div>',
        unsafe_allow_html=True,
    )

    gender = fairness["dimensions"]["gender"]
    gender_rows = []
    for group, metrics in gender["groups"].items():
        if metrics.get("eligible_for_comparison"):
            gender_rows.append(
                {
                    "Audit group": group,
                    "Rows": metrics["rows"],
                    "Observed event rate": metrics["event_rate"],
                    "Mean probability": metrics["mean_probability"],
                    "Review-queue rate": metrics["review_rate"],
                    "ROC AUC": metrics["roc_auc"],
                }
            )
    st.subheader("Gender audit at a fixed 20% review capacity")
    st.dataframe(
        gender_rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Observed event rate": st.column_config.NumberColumn(format="percent"),
            "Mean probability": st.column_config.NumberColumn(format="percent"),
            "Review-queue rate": st.column_config.NumberColumn(format="percent"),
            "ROC AUC": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    gap = gender["observed_gaps"]["review_rate"]
    st.caption(
        f"Observed review-rate gap: {gap['absolute_gap']:.1%} between {gap['minimum_group']} and "
        f"{gap['maximum_group']}. It reflects both population outcome differences and model behavior."
    )

    prohibited, required = st.columns(2, gap="large")
    with prohibited:
        st.subheader("Not permitted")
        st.markdown(
            "- Autonomous approval or decline\n"
            "- Pricing, limits, or adverse-action notices\n"
            "- Production use without independent validation\n"
            "- Treating missing history as evidence of low risk"
        )
    with required:
        st.subheader("Required before production")
        st.markdown(
            "- Jurisdiction-specific compliance and explainability review\n"
            "- Temporal and out-of-distribution validation\n"
            "- Data contracts, monitoring, and incident controls\n"
            "- Fair-lending assessment with business policy and outcomes"
        )

    st.subheader("Artifact trace")
    trace = {
        "Model": status["model"]["name"],
        "Model version": status["model"]["version"],
        "Calibration": status["model"]["calibration"],
        "Gap strategy": status["model"]["gap_strategy"],
        "Fairness assessment": status["fairness_assessment"],
        "Production approved": status["production_approved"],
    }
    st.json(trace)
