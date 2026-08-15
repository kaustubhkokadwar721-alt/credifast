# Accuracy and data-gap strategy milestone

## Outcome

CrediFast now has an evidence-based, segment-specific scoring candidate. The experiment compares the application-only LightGBM, history-enriched LightGBM, and coarse convex blends. Blend weights are selected only on `calibrator_fit` rows and evaluated on the disjoint `calibrator_evaluation` rows. The final holdout remains sealed.

The operational research candidate is:

| History coverage | History weight | Application weight | Route |
|---|---:|---:|---|
| Full, 4-5 sources | 100% | 0% | Standard or boundary review |
| Partial, 2-3 sources | 75% | 25% | Data-limited manual review |
| Thin, 0-1 sources | 100% | 0% | Data-limited manual review |

The application-only model is not automatically superior when history is sparse. Missing-value patterns, application variables, and the available history fields are learned jointly by the history model, so the correct fallback must be measured rather than assumed.

## Selection and promotion design

Each segment compares history weights `0.00`, `0.25`, `0.50`, `0.75`, and `1.00`. The fit-only selection rule minimizes Brier score, then log loss, with a final tie-break toward less feature-store dependence.

A selected blend is promoted only if the disjoint evaluation rows satisfy all four gates:

1. Brier score improves over the history champion.
2. Log loss improves.
3. Average precision does not worsen.
4. A paired, class-stratified bootstrap gives at least 90% support for a Brier improvement.

This gate rejected the 75/25 full-history blend, promoted the 75/25 partial-history blend, and retained the history-only thin-file path.

## Evaluation evidence

| Strategy on 23,064 evaluation rows | ROC-AUC | Average precision | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Application only | 0.7562 | 0.2575 | 0.067381 | 0.2455 | 0.00471 |
| History only | 0.7794 | 0.2853 | 0.065831 | 0.2383 | 0.00321 |
| Operational segment strategy | 0.7794 | 0.2869 | 0.065798 | 0.2383 | 0.00370 |

Versus history-only, the operational strategy changes Brier by -0.0000323, where negative is better. Its paired 95% bootstrap interval is approximately [-0.0000767, 0.0000161], with 88.8% bootstrap support for improvement. The overall interval crosses zero, so the gain is promising but not conclusive. The candidate must be evaluated once on the sealed holdout before any stronger claim.

Within partial histories, the 75/25 blend improves Brier from 0.084682 to 0.084376, log loss from 0.299376 to 0.298811, ECE from 0.01979 to 0.01712, and average precision from 0.24297 to 0.24840. Paired-bootstrap support for lower Brier is 90.8%; the 95% interval still narrowly crosses zero, which is retained as an uncertainty caveat.

## Exact source-count evidence

The broad segments are also reported by exact source count. Zero-source evaluation contains only 164 applicants and 16 events. Application-only has higher ROC-AUC there (0.8091 versus 0.7741), while history-only has slightly better average precision, Brier, and ECE. Because the sample is small and the metrics conflict, zero-source evidence is labelled exploratory and no separate automatic weight is selected.

Two-source evaluation is also exploratory: 61 rows and 7 events. Operational decisions remain at the broader segment level to avoid overfitting sparse strata.

## Missing feature-store behavior

A missing feature-store row is an operational data-quality fault, not evidence of good or bad repayment behavior. Inference therefore:

- preserves numeric historical fields as missing, allowing the trained preprocessing/model path to handle them;
- sets only explicit `HAS_*` availability flags to zero;
- assigns the thin-file segment and low confidence;
- emits `FEATURE_STORE_ROW_MISSING` and `feature_store_row_missing=true`;
- forces `MANUAL_REVIEW_DATA_LIMITED`;
- records application and history component probabilities and the applied blend weight.

The application model remains loaded as a validated fallback component. Under the current evidence it receives non-zero weight only for partial histories.

## Reproduce

```powershell
credifast-train history-gap-strategy
credifast-score --limit 100 --no-explanations
```

Versioned outputs:

- `artifacts/history_gap_strategy_metrics.json`
- `configs/model_gap_strategy.json`
- `src/credifast/modeling/gap_strategy.py`
- `src/credifast/modeling/inference.py`

Local model binaries and row-level predictions remain Git-ignored.

## Subsequent final gate

The frozen policy was subsequently evaluated once on the sealed holdout. The partial-history blend replicated its Brier, log-loss, average-precision, and ROC-AUC improvements within that segment. Overall technical targets passed. See `PHASE6_FINAL_VALIDATION_STATUS.md`.

## Remaining adaptation gates

- Add out-of-time and lender-population validation before any real deployment.
- Monitor source coverage, store-miss rates, segment volumes, calibration, and outcome drift.
- Define escalation behavior for schema mismatch and sustained feature-store outages.
- Treat all outputs as research-only decision support; never use them for autonomous adverse action.
