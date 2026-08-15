# Final frozen-candidate validation

## Protocol

The model, feature schema, identity calibrator, segment weights, confidence routing, and promotion criteria were frozen before the final random holdout was opened. The holdout contains 46,127 applicants and 3,724 repayment-difficulty events. The evaluation code records input hashes, writes row-level predictions only to a Git-ignored Parquet file, and refuses to overwrite the versioned final artifact.

These results close the competition-model tuning loop. They must not be used to tune another candidate against the same holdout.

## Acceptance result

All targets documented before final evaluation passed.

| Gate | Target | Observed | Result |
|---|---:|---:|---|
| ROC-AUC | at least 0.75 | 0.78493 | Pass |
| Average precision | at least 2.5x prevalence | 3.47x | Pass |
| Brier | better than base-rate predictor | 0.066071 vs 0.074216 | Pass |
| Calibration slope | 0.90-1.10 | 1.023 | Pass |

Additional evidence:

- average precision: 0.27988 at 8.07% prevalence;
- log loss: 0.23785;
- equal-count ECE: 0.00342;
- calibration intercept: 0.0570;
- top-decile lift: 3.68 and event capture: 36.82%;
- top-20% lift: 2.82 and event capture: 56.50%.

## Component comparison

| Model | ROC-AUC | Average precision | Brier | Log loss |
|---|---:|---:|---:|---:|
| Application only | 0.76234 | 0.24974 | 0.067617 | 0.24500 |
| History only | 0.78506 | 0.27921 | 0.066083 | 0.23786 |
| Frozen segment candidate | 0.78493 | 0.27988 | 0.066071 | 0.23785 |

The operational candidate slightly improves average precision, Brier, and log loss over history-only, while ROC-AUC is lower by 0.00013. The paired 95% intervals for both Brier and ROC-AUC differences cross zero. The proper conclusion is that the segment strategy is approximately performance-neutral overall while improving the targeted partial-history population.

## Coverage segments

| Segment | Rows | Events | History weight | ROC-AUC | Average precision | Brier | ECE | Slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full, 4-5 sources | 38,941 | 3,114 | 1.00 | 0.78325 | 0.27872 | 0.065578 | 0.00402 | 1.010 |
| Partial, 2-3 sources | 4,915 | 486 | 0.75 | 0.77383 | 0.28485 | 0.079688 | 0.00641 | 1.054 |
| Thin, 0-1 sources | 2,271 | 124 | 1.00 | 0.82438 | 0.29320 | 0.045053 | 0.01448 | 1.290 |

The 75/25 partial-history blend replicates its pre-holdout direction: it improves ROC-AUC, average precision, Brier, and log loss versus history-only in that segment. This is the strongest evidence for retaining the blend.

Thin-file discrimination is good, but the calibration slope of 1.290 is outside the documented acceptable band. Exact zero-source results are exploratory: 339 rows, 21 events, ECE 0.0369, and slope 1.350. Two-source results have only 145 rows and 5 events. These are data-gap warnings, not an invitation to optimize on the final holdout.

## Decision

- Retain the frozen 75% history / 25% application blend for partial histories.
- Retain history-only probability for full and thin histories.
- Keep every zero-to-three-source applicant on `MANUAL_REVIEW_DATA_LIMITED`.
- Keep missing feature-store rows explicitly flagged and fail closed to manual review.
- Do not retune or recalibrate using this holdout.
- Treat the prototype as technically validated for the hackathon only, not approved for real lending.

## Reproducibility

```powershell
credifast-train final-holdout
```

The command is intentionally one-shot after `artifacts/final_holdout_evaluation.json` exists. The versioned artifact contains metrics, uncertainty, exact-source diagnostics, controls, environment versions, and hashes for both models, the calibrator, policy, and feature store.

Further model development requires a new untouched dataset: preferably time-separated, lender-owned, geographically relevant, and governed for the intended target and decision population.
