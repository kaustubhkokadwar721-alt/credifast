# Model plan

## Modeling contract

The hackathon model estimates the probability of the Home Credit repayment-difficulty target using only information that is defensibly available before the current lending decision.

A real implementation must replace this with a lender-owned outcome such as 90 or more days past due within 12 months.

## Model suite

| Model | Role | Key value |
|---|---|---|
| Base-rate predictor | Sanity baseline | Establishes no-skill calibration and PR baseline |
| Regularized logistic regression | Explainable baseline | Stable coefficients and simple deployment |
| WoE logistic scorecard | Governance challenger | Monotonic bins and traditional reasonability review |
| LightGBM | Planned champion | Strong nonlinear tabular performance and fast SHAP |
| CatBoost | Challenger | Native categorical handling and robust missing categories |

Deep learning is outside the MVP unless the tabular baselines reveal a justified advantage.

## Splitting

Home Credit Default Risk lacks a dependable complete decision date, so the initial validation is stratified rather than genuinely out-of-time:

- 70% development.
- 15% calibration and threshold tuning.
- 15% untouched final holdout.
- Five folds within development.
- Persisted applicant-level assignments.

The newer Home Credit Model Stability dataset is the proposed future source for genuine time-based validation.

## Preprocessing

### Logistic models

- Median numerical imputation.
- Explicit missing indicators.
- Normalized categorical values and unknown bucket.
- One-hot or fold-safe Weight of Evidence encoding.
- Winsorization only when justified and documented.
- Regularization selected inside development cross-validation.

### Tree models

- Native missing-value handling where supported.
- Stable category encoding.
- No scaling required.
- Minimum leaf sizes and regularization to prevent sparse segment memorization.

## Imbalance strategy

- Preserve natural class distribution in calibration and holdout sets.
- Evaluate class weighting.
- Tune decision thresholds by operational cost and review capacity.
- Do not use accuracy as a primary measure.
- Do not use SMOTE by default because it can distort probability calibration and generate implausible applicants.

## Calibration

Compare raw predictions with Platt scaling and isotonic regression using the dedicated calibration partition.

Select using:

- Brier score.
- Reliability diagram.
- Calibration intercept and slope.
- Expected calibration error.
- Stability across risk bands and important audit groups.

## Score mapping

Use calibrated probability `PD`:

```text
odds  = (1 - PD) / PD
score = 600 + 20 * log2(odds / 50)
```

This gives 600 points at approximately 50:1 good-to-bad odds and 20 points for each doubling of odds. Clamp only for display; preserve the original calibrated probability for decisions and monitoring.

## Model selection

Select the champion by a balanced review of:

1. Holdout discrimination.
2. Probability calibration.
3. Risk-band stability.
4. Fairness and segment errors.
5. Missing-source robustness.
6. Explanation quality.
7. Latency and artifact size.

The largest ROC-AUC does not automatically win.

## Explainability

- Global SHAP for model behavior.
- Local SHAP for applicant contributions.
- Controlled mapping from features to reason codes.
- Top adverse and favorable factors.
- No sensitive characteristic in customer-facing reasons.
- Stability test under small, non-material input changes.

Counterfactual simulations are labelled illustrative and non-causal.

## Fairness analysis

Where lawful and sufficiently sampled, compare:

- score and probability distribution;
- approval/manual-review rates;
- true-positive and false-positive rates;
- false-negative rates;
- calibration curves and Brier score;
- reason-code prevalence;
- missing-source and thin-file routing.

Report sample sizes and uncertainty. Do not infer fairness from approval-rate parity alone.

## Monitoring

Before outcomes mature:

- Feature drift.
- Missingness and category drift.
- Score and grade distribution.
- Out-of-distribution rate.
- Full-file versus thin-file share.
- Manual-review and override rate.

After outcomes mature:

- ROC-AUC, PR-AUC, lift, and capture.
- Calibration and bad rate by grade.
- Fairness metrics.
- Vintage performance.
- Threshold and policy effectiveness.
