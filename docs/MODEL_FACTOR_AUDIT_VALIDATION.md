# Model factor audit validation

## Overall assessment: share with caveats

The factor audit is internally consistent and reproducible enough for hackathon model review. It is not sufficient for production credit policy, causal interpretation, feature removal, or regulatory adverse-action use.

## Question and scope

The audit answers five questions about the frozen history-enriched LightGBM model:

1. Which raw and transformed inputs are used?
2. How is each input preprocessed?
3. What risk direction is intuitively expected, and what direction does the fitted model exhibit?
4. How large is each factor's global contribution?
5. Which inputs are correlated enough to be possible substitutes?

The population is the once-opened 46,127-row final random holdout. The model, calibration, gap policy, and thresholds were not refit or selected from this audit.

## Methodology review

- The 257 selected raw factors come from the frozen model bundle's persisted feature selection.
- The fitted preprocessing pipeline expands them to 465 columns: direct numeric/categorical inputs plus 208 numeric missingness indicators.
- Exact LightGBM TreeSHAP contributions are calculated for every holdout row in the history model's uncalibrated log-odds space.
- Raw-factor magnitude combines the direct factor and its automatic missingness indicator. `shap_share` divides each factor's mean absolute contribution by the sum across all raw factors.
- Observed numeric direction uses Spearman correlation between a factor value and that factor's own SHAP contribution. Upper-versus-lower observed event-rate differences are reported separately.
- Numeric redundancy uses pairwise Spearman correlation on a deterministic 25,000-row sample. Categorical redundancy uses Cramer's V. Only associations of at least 0.50 are written to the pair file.
- Categorical factor direction is reported by category rather than assigning meaning to the ordinal-encoder numbers.

## Calculation spot checks

| Check | Result |
|---|---|
| Holdout IDs are unique | Passed |
| Holdout targets match `application_train.csv` | Passed |
| 257 raw factors are unique | Passed |
| 465 transformed inputs are unique | Passed |
| Every transformed input maps to a raw factor | Passed |
| Raw-factor SHAP shares sum to 1 | Passed |
| Raw-factor gain shares sum to 1 | Passed |
| Signal-family SHAP and gain shares sum to 1 | Passed |
| Correlation-pair threshold is respected | Passed |
| `TARGET` and `SK_ID_CURR` are absent from model factors | Passed |

## Findings that materially affect interpretation

1. **External-score concentration — medium risk, high confidence.** Eight external-score factors represent 22.38% of global SHAP magnitude and 37.83% of split gain. The family contains correlated originals and derived mean/min/max features, so individual importance cannot be read as independent information.
2. **Substantial duplicate representation — medium risk, high confidence.** Several pairs are algebraic mirrors or near duplicates. `CREDIT_TO_GOODS` and `DOWN_PAYMENT_RATE` have Spearman correlation -1.00; multiple amount/count/max/rate aggregates exceed 0.99 absolute correlation. Feature removal still requires grouped ablation because missingness and interactions differ.
3. **Sparse source families — high risk for generalization, high confidence.** Median holdout missingness is 72.23% for credit-card factors, 69.79% for bureau-monthly factors, and 55.53% for property factors. Importance in available-file applicants does not imply dependable population-wide availability.
4. **Direction is conditional, not causal — high risk if misused, high confidence.** A factor may have a model-SHAP direction that differs from its raw event-rate association because LightGBM conditions on correlated inputs and interactions. For example, income has a favorable raw upper-versus-lower event-rate difference but a mixed conditional SHAP pattern.
5. **The current history representation is mostly aggregate — high accuracy opportunity, high confidence.** Recent late-payment counts exist, but exponential recency weighting, rolling trends, streaks, cures, and event sequences do not. Loan/application counts exist, but verified EMI funding and account-level cash-flow tracing do not.

## Required caveats

- Average precision is prevalence-sensitive; compare 0.2799 with the 0.0807 random baseline and with capacity-specific precision/recall.
- Global SHAP magnitude describes the fitted model, not real-world causality, approval policy, or customer-level reason sufficiency.
- Correlation is a redundancy screen, not proof of safe substitution.
- The external scores are anonymized, limiting semantic and operational validation.
- The holdout is a random split from one Kaggle competition dataset, not lender-owned out-of-time evidence.
- This audit intentionally diagnoses the sealed holdout but does not authorize any further tuning against it.

## Reproducible outputs

- `artifacts/average_precision_diagnostic.json`
- `artifacts/factor_audit/model_factor_audit.csv`
- `artifacts/factor_audit/transformed_factor_audit.csv`
- `artifacts/factor_audit/factor_correlation_pairs.csv`
- `artifacts/factor_audit/factor_group_summary.csv`
- `scripts/diagnose_average_precision.py`
- `scripts/audit_model_factors.py`
