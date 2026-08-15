# History-enriched model milestone

## Outcome

The target-free relational feature store and history-enriched LightGBM challenger are complete. The feature store aggregates bureau, bureau-balance, previous-application, POS, credit-card, and installment-payment behavior to exactly one row per current application.

The enriched model is the current pre-calibration champion. It materially improves every reported calibration-partition ranking metric over the application-only challenger. The final 46,127-row holdout remains unscored.

## Feature-store contract and quality gate

| Check | Result |
|---|---:|
| Applications reconciled | 356,255 / 356,255 |
| Feature columns excluding key | 150 |
| Unique application keys | 356,255 |
| Duplicate keys | 0 |
| Null keys | 0 |
| Target included | No |
| Invalid bounded rates | 0 |
| Negative count features | 0 |
| Non-finite numeric values | 0 |
| Ready for modeling | Yes |

All sources are aggregated independently before the application-level join. Missing source histories remain null and are paired with explicit `HAS_*` flags; they are not converted to observed zero balances or zero delinquencies.

## Source coverage

| Historical source | Applications with history | Coverage |
|---|---:|---:|
| Bureau | 305,811 | 85.84% |
| Previous applications | 338,857 | 95.12% |
| POS cash balances | 337,252 | 94.67% |
| Credit-card balances | 103,558 | 29.07% |
| Installment payments | 339,587 | 95.32% |

Credit-card sparsity is structural in this dataset. High-missingness selection excludes the sparsest derived fields using only the development partition, while availability flags preserve the distinction between no source and observed zero.

## Calibration-partition comparison

| Metric | Logistic | Application LightGBM | History LightGBM |
|---|---:|---:|---:|
| ROC-AUC | 0.7459 | 0.7588 | **0.7814** |
| Average precision | 0.2326 | 0.2492 | **0.2739** |
| Brier score | 0.06845 | 0.06762 | **0.06629** |
| Brier skill vs base rate | 7.76% | 8.89% | **10.68%** |
| Top-10% event capture | 32.41% | 34.16% | **36.73%** |
| Top-20% event capture | 50.19% | 51.93% | **55.88%** |

Relative to the application-only LightGBM, history adds 2.26 percentage points of ROC-AUC, 2.47 points of average precision, and 3.95 points of top-20% default capture. Raw probability quality also improves: Brier score falls by 0.00133.

## Model specification

- Deterministic seed 42 and the existing 70/15/15 stratified applicant split.
- Responsible application feature policy remains unchanged.
- 150 target-free history candidates are joined one-to-one by application ID.
- Feature selection is fit on development rows only; 257 total features survive policy, sparsity, and constant checks.
- An internal development-only sub-split selects 825 boosting iterations by ROC-AUC.
- The final model refits on all development rows and scores only calibration rows.
- The sealed holdout is not used for feature selection, early stopping, calibration comparison, or reporting.

## Limitations and next gate

- This is a random stratified competition split, not lender-owned out-of-time validation.
- History timestamps are relative to application time; the dataset does not provide a production event-time replay contract.
- Availability differences are predictive associations, not causal or fairness conclusions.
- The probabilities are not yet formally calibrated, and no approval threshold is authorized.
- Competition data cannot validate Indian lending policy, affordability, pricing, adverse action, or legal compliance.

Next compare Platt and isotonic calibration on the calibration partition, define thin-file confidence behavior, freeze the model bundle, and only then evaluate the holdout once.

## Reproducibility

```powershell
credifast-data aggregate-history
credifast-data validate-history
credifast-train history-lightgbm
python scripts/create_history_quality_notebook.py
```

Evidence is stored in `artifacts/history_features_report.json`, `artifacts/history_quality_report.json`, `artifacts/history_lightgbm_metrics.json`, `artifacts/history_lightgbm_features.json`, and `artifacts/phase3_model_comparison.json`. The executed audit notebook is `notebooks/phase3_history_quality.ipynb`.
