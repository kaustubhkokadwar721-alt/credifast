# Phase 3 report source notes

Audience: technical. Delivery surface: native Data Analytics report artifact.

Required technical-report roles map as follows:

- Title: report title block.
- Technical summary: first narrative section.
- Key findings with visual evidence: metric strip, calibration-improvement chart, and exact model table.
- Scope, data, and definitions: feature-store and metric-definition section.
- Methodology and model specification: dedicated modeling section.
- Limitations, uncertainty, and robustness: limitations section plus sealed-holdout statement.
- Recommended next steps: calibration and model-freeze section.
- Further questions: final report section.

Chart map:

| Segment | Question | Chart | Fields | Claim |
|---|---|---|---|---|
| Model improvement | How much does history add over application-only LightGBM? | Vertical bar | Metric, percentage-point improvement | History improves ranking, rare-event precision, capture, and Brier skill on the shared calibration partition. |
| Source coverage | How sparse is each history family? | Vertical bar | Source, coverage rate | Credit-card history is structurally much sparser than the other histories. |

The charts use separate datasets because model-quality improvements and source coverage answer different questions. All model values come from saved metric artifacts; all coverage values come from the history quality report. No applicant-level rows are exposed in the report snapshot.
