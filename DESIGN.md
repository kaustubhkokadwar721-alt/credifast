---
name: CrediFast
description: An evidence-led review docket for explainable credit decision support.
colors:
  ink-navy: "#13243a"
  ink-soft: "#415268"
  paper: "#f7f5ef"
  surface: "#ffffff"
  rule: "#d6dbe2"
  action-cobalt: "#2457d6"
  action-cobalt-dark: "#173b94"
  exception-amber: "#a45b08"
  exception-amber-bg: "#fff4dc"
  alert-red: "#a82d2d"
  alert-red-bg: "#fff0ef"
  ready-green: "#16614a"
  ready-green-bg: "#eaf7f0"
  exception-amber-border: "#e7c98a"
  exception-amber-ink: "#6b430e"
  alert-red-border: "#e0afaa"
  alert-red-ink: "#722323"
  ready-green-border: "#a9d2bf"
  ready-green-ink: "#174e3d"
typography:
  display:
    fontFamily: '"CrediFast Docket", "Source Serif 4", Georgia, serif'
    fontSize: "clamp(2.2rem, 4.4vw, 3.8rem)"
    fontWeight: 650
    lineHeight: 0.94
    letterSpacing: "-0.04em"
  headline:
    fontFamily: "sans-serif"
    fontSize: "clamp(1.5rem, 3vw, 2.7rem)"
    fontWeight: 760
    lineHeight: 1
    letterSpacing: "-0.035em"
  title:
    fontFamily: "sans-serif"
    fontSize: "1rem"
    fontWeight: 700
  body:
    fontFamily: "sans-serif"
    fontSize: "0.9rem"
    fontWeight: 400
    lineHeight: 1.4
  label:
    fontFamily: "sans-serif"
    fontSize: "0.72rem"
    fontWeight: 800
    letterSpacing: "0.12em"
  label-lead:
    fontFamily: "sans-serif"
    fontSize: "0.82rem"
    fontWeight: 800
    letterSpacing: "0.16em"
  label-meta:
    fontFamily: "sans-serif"
    fontSize: "0.76rem"
    fontWeight: 700
  metric:
    fontFamily: "sans-serif"
    fontSize: "1.25rem"
    fontWeight: 760
  notice:
    fontFamily: "sans-serif"
    fontSize: "0.86rem"
    fontWeight: 400
    lineHeight: 1.45
rounded:
  square: "0"
  control: "2px"
  circle: "50%"
spacing:
  hairline: "0.2rem"
  compact: "0.45rem"
  row: "0.8rem"
  inset: "1rem"
  section: "1.2rem"
  section-large: "1.7rem"
components:
  button-primary:
    backgroundColor: "{colors.action-cobalt}"
    textColor: "{colors.surface}"
    rounded: "{rounded.control}"
  button-primary-hover:
    backgroundColor: "{colors.action-cobalt-dark}"
    textColor: "{colors.surface}"
    rounded: "{rounded.control}"
  docket-label:
    backgroundColor: "{colors.ink-navy}"
    textColor: "{colors.surface}"
    typography: "{typography.label}"
    rounded: "{rounded.square}"
    padding: "0.32rem 0.52rem"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink-navy}"
    rounded: "{rounded.control}"
  score-strip:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink-navy}"
    rounded: "{rounded.square}"
---

# Design System: CrediFast

## Overview

**Creative North Star: "The Evidence Docket"**

CrediFast presents credit review as a composed evidence file, not a verdict machine. Its visual authority comes from ink-on-paper contrast, editorial hierarchy, ruled ledgers, square docket marks, and explicit source coverage. The interface feels calm, exacting, and reviewable: it gives the human operator room to inspect what is known, what is missing, and what must happen next.

The system pairs one locally bundled Source Serif 4 display face with the Streamlit runtime's practical sans-serif body. Cobalt is reserved for action and navigation; amber, red, and green are restrained operational signals that always arrive with words. Dense information is divided by rules and tonal surfaces rather than decorative imagery, gradients, or floating cards.

**Key Characteristics:**

- Editorial thesis headlines paired with operational sans-serif copy.
- Ink navy on warm paper, with white working surfaces and cobalt actions.
- Ruled evidence rows, ledger tables, tabular numerals, and square docket labels.
- Explicit data gaps and human-review routes instead of autonomous verdict language.
- Flat, border-led depth with no shadows or gradients at rest.
- A desktop two-column workbench that becomes a single reading column on mobile.

## Colors

The palette reads like a marked case file: deep navy ink, warm paper, crisp white inserts, one cobalt action color, and restrained exception fills.

### Primary

- **Action Cobalt** (`action-cobalt`): The sole high-energy color for the main action and active navigation state; its darker partner is reserved for hover.
- **Deep Action Cobalt** (`action-cobalt-dark`): Hover emphasis and evidence codes that need stronger contrast without becoming alarmist.

### Tertiary

- **Exception Amber** (`exception-amber`, `exception-amber-bg`): Data limitations, caveats, and guardrails that require attention without implying failure.
- **Review Red** (`alert-red`, `alert-red-bg`): Enhanced-review conditions and explicit prohibitions; never a standalone risk verdict.
- **Ready Green** (`ready-green`, `ready-green-bg`): Runtime readiness and evidence-package readiness, always paired with descriptive copy.

### Neutral

- **Ink Navy** (`ink-navy`): Primary text, heavy rules, routing emphasis, and inverted docket labels.
- **Soft Ink** (`ink-soft`): Supporting copy, metadata, captions, and secondary labels.
- **Warm Paper** (`paper`): The application canvas and dominant visual atmosphere.
- **Working White** (`surface`): Inputs, score strips, tables, and contained evidence surfaces.
- **Ledger Rule** (`rule`): Dividers, table borders, field edges, and structural separation.

### Named Rules

**The Cobalt Action Rule.** Reserve Action Cobalt for the current navigation state and consequential reviewer actions; evidence itself remains ink-led.

**The Exception Color Rule.** Amber, red, and green always accompany explicit text and never carry risk, confidence, or status alone.

## Typography

**Display Font:** Source Serif 4, locally bundled as CrediFast Docket (with Georgia and serif fallbacks)
**Body Font:** Streamlit runtime sans-serif
**Label/Mono Font:** Streamlit runtime sans-serif with tracked uppercase styling and tabular numerals where values align

**Character:** The serif display voice frames the reviewer’s thesis with editorial gravity. The inherited sans-serif body keeps forms, evidence, metadata, and governance language fast to scan and operationally neutral.

### Hierarchy

- **Display** (650, `clamp(2.2rem, 4.4vw, 3.8rem)`, 0.94): Page-level thesis headlines only, with tight negative tracking.
- **Headline** (760, `clamp(1.5rem, 3vw, 2.7rem)`, 1): The active human-review route inside the persistent decision band.
- **Title** (700, `1rem`): Section rules, subheads, and ledger headings that organize the docket.
- **Body** (400, approximately `0.9rem`, 1.4): Explanations, evidence descriptions, field content, and plain-language guidance; route notes stop at about 70 characters.
- **Label** (750–800, `0.7rem–0.82rem`, `0.08em–0.16em`, uppercase when structural): Wordmarks, route labels, score labels, evidence codes, and docket identifiers.

### Named Rules

**The One Serif Move Rule.** Source Serif 4 belongs to page-level thesis headlines; operational controls and evidence remain sans-serif.

**The Ledger Numeral Rule.** Scores, percentages, weights, and right-aligned metrics use tabular numerals so comparisons hold their columns.

## Layout

The workbench sits in a wide container capped at 1440px, with compact top padding and generous bottom breathing room. A masthead and three-tab navigation establish the file before each page-level thesis. On the review surface, the first viewport is a two-column docket: editable case intake occupies the narrower `0.82` share and the routing band with evidence coverage occupies the wider `1.18` share. The full-width reason ledger and follow-on audits continue below.

Section rules carry paired title and metadata across the available width. Score summaries form a four-column strip; evidence rows use a compact code column, flexible explanation column, and right-aligned contribution column. Spacing is dense but not cramped, with roughly `0.8rem` row rhythm, `1rem` notice insets, and `1.7rem` section separation.

At 760px and below, the Streamlit columns resolve to a single reading column, masthead metadata disappears, the score strip becomes a two-by-two grid, and evidence weights move beneath their descriptions. The mobile sequence preserves the reviewer story: choose the case, adjust terms, evaluate, then read the route and evidence.

### Named Rules

**The First Viewport Rule.** On desktop, keep case intake, Evaluate case, the current human-review route, and the evidence summary together before the long-form ledgers.

## Elevation & Depth

The system is flat at rest. It uses warm-versus-white tonal separation, one-pixel ledger rules, and heavier navy bands to establish hierarchy; cards do not float and surfaces do not cast shadows. The routing band briefly settles into place after evaluation with an inset cobalt trace, translation, and blur, then returns to the same flat plane.

### Named Rules

**The Flat Docket Rule.** Use borders, rule weight, and surface tone for depth; do not add persistent shadows, gradients, glass effects, or floating card stacks.

## Shapes

The dominant geometry is near-square. Buttons and form controls use a restrained 2px corner, while routing bands, score strips, notices, ledger tables, evidence rows, and docket labels remain square. Borders are structural, usually one pixel, with three- or six-pixel top rules only when a band must carry document-level emphasis. The small circular runtime dot is the sole recurring round exception.

### Named Rules

**The No Pill Rule.** Buttons, labels, filters, and status treatments stay square or nearly square; reserve circles for point indicators, not text containers.

## Components

### Buttons

- **Shape:** Near-square controls with a 2px radius and a minimum height of `2.75rem`.
- **Primary:** Full-width Action Cobalt with white copy and a heavy sans-serif label; the review workbench uses one visible primary action, “Evaluate case.”
- **Hover / Focus:** Hover darkens to Deep Action Cobalt. Preserve Streamlit’s keyboard-visible focus affordance; any future custom focus treatment must remain high contrast and must not rely on color alone.
- **Secondary:** Native restrained Streamlit treatment when required; do not create competing filled actions beside Evaluate case.

### Chips

- **Style:** The docket identifier is a square Ink Navy label with white, tracked uppercase text and compact `0.32rem 0.52rem` padding.
- **State:** Informational only. It names the selected application and does not behave like a removable or selectable pill.

### Cards / Containers

- **Corner Style:** Square at `0` radius.
- **Background:** Working White for score strips, tables, inputs, and evidence surfaces; Warm Paper remains visible around them.
- **Shadow Strategy:** None at rest; use rules and tonal contrast per the Flat Docket Rule.
- **Border:** One-pixel Ledger Rule dividers, with heavier Ink Navy rules for mastheads and routing bands.
- **Internal Padding:** Compact, typically `0.72rem–1rem`.

### Inputs / Fields

- **Style:** White fields with inherited Streamlit borders, Ink Navy values, explicit labels and units, and a 2px radius.
- **Focus:** Keep the framework’s keyboard focus indicator visible and unobscured.
- **Error / Disabled:** Use plain-language recovery guidance. Missing history is an operational condition shown through copy and the amber exception treatment, never a silent disabled state.

### Navigation

The three-tab navigation is a quiet text row with generous `1.2rem` gaps, a shared Ledger Rule baseline, and a cobalt underline on the active tab. Labels use the body sans at a firm weight. On mobile, the same row stays readable and scrolls or compresses within the viewport rather than becoming an unrelated navigation pattern.

### Routing Band

The signature decision-support component uses a six-pixel Ink Navy top rule, one-pixel bottom rule, tracked uppercase label, large sans-serif route name, and a plain-language note. Its docket-settle animation lasts 520ms with a `cubic-bezier(.16, 1, .3, 1)` easing curve; it is disabled under `prefers-reduced-motion: reduce`.

### Score Strip

Four equal ruled cells report risk estimate, internal score, risk grade, and confidence. Labels are small, tracked, and uppercase; values are bold and use tabular numerals. The strip becomes two columns on mobile without changing reading order.

### Evidence Rows and Notices

Evidence rows are ruled ledgers: cobalt-dark codes, Ink Navy explanations, and right-aligned Soft Ink weights. Notices use a one-pixel tinted border plus a three-pixel semantic top rule. Amber marks data gaps, red marks enhanced review or prohibition, and green marks readiness; each treatment includes direct explanatory text.

### Named Rules

**The Human Route Rule.** The routing band names a human queue and explains why; it never presents approval, decline, pricing, or adverse-action language.

## Do's and Don'ts

### Do:

- **Do** lead with evidence, source coverage, limitations, and the next human-review action.
- **Do** keep the ink-and-paper hierarchy, restrained cobalt actions, ruled rows, square labels, and tabular quantitative outputs.
- **Do** pair every semantic color with a label or sentence that explains the state.
- **Do** preserve the desktop two-column first viewport and the single-column mobile reading order.
- **Do** disable the docket-settle animation when reduced motion is requested.

### Don't:

- **Don't** turn the score or route into an autonomous approval, decline, price, or adverse-action verdict.
- **Don't** add gradients, persistent shadows, glass surfaces, decorative banking imagery, or generic KPI-card grids.
- **Don't** use pills or excessive rounding; the docket’s authority comes from near-square geometry and rules.
- **Don't** use red, amber, or green without explicit supporting copy.
- **Don't** introduce another display family or spread the serif into controls, tables, and routine evidence copy.
