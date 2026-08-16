"""Shared docket styling for the collectible-input screens.

Extends the Evidence Docket system in DESIGN.md to the ledger grids these screens
introduce. Square corners, one-pixel rules, tabular numerals, no shadows, no fills that
carry meaning without words.
"""

from __future__ import annotations

import streamlit as st

_STYLE = """
<style>
/* Ledger grids: square, ruled, tabular. The editor is a document, not a widget. */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  border: 1px solid var(--line);
  border-radius: 0 !important;
  background: var(--surface);
}
[data-testid="stDataFrame"] *, [data-testid="stDataEditor"] * {
  border-radius: 0 !important;
  font-variant-numeric: tabular-nums;
}
[data-testid="stDataEditor"] [data-testid="stTooltipHoverTarget"] { color: var(--ink-soft); }

/* Field rows sit tight; the section rule above them carries the separation. */
.field-row [data-testid="stNumberInput"] label,
.field-row [data-testid="stSelectbox"] label,
.field-row [data-testid="stTextInput"] label {
  font-size: .72rem;
  letter-spacing: .06em;
  text-transform: uppercase;
  font-weight: 750;
  color: var(--ink-soft);
}

/* Sub-tabs inside a page read as ledger sections, quieter than the page nav. */
.ledger-tabs [data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: .9rem;
  border-bottom: 1px solid var(--line);
}
.ledger-tabs [data-testid="stTabs"] button { font-weight: 700; font-size: .9rem; }

/* Ledger caption: what this grid is and how to fill it. Body copy, held back by colour
   rather than by size, so instructions stay readable while entering data. */
.ledger-note {
  color: var(--ink-soft);
  font-size: .9rem;
  line-height: 1.5;
  border-left: 1px solid var(--line);
  padding-left: .7rem;
  margin: .1rem 0 .6rem;
}
.ledger-note strong { color: var(--ink); font-weight: 750; }

/* Source tally: named families, present or absent, never a bare score. */
.tally { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border: 1px solid var(--line); background: var(--surface); margin: .5rem 0 .9rem; }
.tally-cell { padding: .7rem .85rem; border-right: 1px solid var(--line); }
.tally-cell:last-child { border-right: 0; }
.tally-name { display: block; font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 750; color: var(--ink-soft); }
.tally-state { display: block; font-size: 1rem; font-weight: 760; margin-top: .18rem; }
.tally-state.have { color: var(--ink); }
.tally-state.gap { color: var(--amber); }

@media (max-width: 760px) {
  .tally { grid-template-columns: 1fr; }
  .tally-cell { border-right: 0; border-bottom: 1px solid var(--line); }
  .tally-cell:last-child { border-bottom: 0; }
}
</style>
"""


def inject() -> None:
    """Emit the shared grid and field styling once per rerun."""

    st.markdown(_STYLE, unsafe_allow_html=True)
