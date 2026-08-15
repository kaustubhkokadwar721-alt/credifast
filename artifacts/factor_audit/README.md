# CrediFast frozen-model factor audit

This audit covers all **257 raw factors** and **465 transformed inputs** used by the frozen history LightGBM model on **46,127 untouched holdout rows**. It is diagnostic only; the holdout was not used to refit, tune, calibrate, or select features.

## How to read magnitude and direction

- `total_mean_abs_shap_log_odds` is the average absolute raw-score contribution, including an automatically generated missingness indicator when present. It is the best global measure here of how strongly the fitted model moves because of a factor.
- `gain_share` is LightGBM's training-time split-gain allocation. Correlated variables divide gain unpredictably, so use it alongside SHAP rather than alone.
- `observed_model_direction` comes from the Spearman relationship between the factor and its own SHAP contribution. `upper_minus_lower_shap` and the event-rate difference show the marginal upper-versus-lower contrast.
- SHAP values are in the history model's uncalibrated log-odds space. They explain model behavior, not causality or a legally sufficient lending reason.
- `substitutability_assessment` is a redundancy screen. It is not permission to remove a variable; correlated factors can have distinct interactions, missingness, and stability.

## Contribution by signal family

| source_group | factor_count | shap_share | gain_share | median_missing_rate |
| --- | --- | --- | --- | --- |
| External scores | 8 | 22.38% | 37.83% | 0.1% |
| Installment repayments | 23 | 12.85% | 10.38% | 5.3% |
| Application derived | 11 | 12.80% | 11.42% | 0.0% |
| Previous applications | 30 | 12.04% | 10.46% | 5.4% |
| POS/cash loan history | 20 | 8.33% | 5.62% | 6.0% |
| Credit bureau | 31 | 8.16% | 8.63% | 14.2% |
| Application financial | 9 | 7.51% | 4.60% | 0.3% |
| Application categorical | 4 | 4.24% | 1.67% | 0.0% |
| Credit card history | 25 | 4.21% | 4.15% | 72.2% |
| Application property | 47 | 3.17% | 3.39% | 55.5% |
| Application verified flags | 8 | 1.56% | 0.46% | 0.0% |
| Bureau monthly status | 10 | 1.05% | 0.65% | 69.8% |
| Application documents | 20 | 0.93% | 0.43% | 0.0% |
| Application credit enquiries | 6 | 0.66% | 0.25% | 13.4% |
| Source availability | 5 | 0.11% | 0.04% | 0.0% |

## Top 30 individual factors

| Factor | Signal family | SHAP share | Gain share | Observed direction | Missing | Strongest related factor |
| --- | --- | --- | --- | --- | --- | --- |
| EXT_SOURCE_MEAN | External scores | 12.57% | 23.78% | higher -> less risk | 0.0% | EXT_SOURCE_MIN |
| EMPLOYMENT_YEARS | Application derived | 3.09% | 1.94% | higher -> less risk | 18.1% |  |
| AMT_ANNUITY | Application financial | 3.06% | 1.62% | higher -> more risk | 0.0% | AMT_CREDIT |
| EXT_SOURCE_MAX | External scores | 3.05% | 4.44% | higher -> less risk | 0.0% | EXT_SOURCE_MEAN |
| CREDIT_TO_ANNUITY | Application derived | 2.92% | 3.01% | higher -> less risk | 0.0% | AMT_CREDIT |
| NAME_EDUCATION_TYPE | Application categorical | 2.70% | 1.07% | category-dependent | 0.0% |  |
| PREV_CREDIT_TO_APPLICATION_MEAN | Previous applications | 2.19% | 1.65% | higher -> more risk | 5.7% | PREV_DOWN_PAYMENT_RATE_MEAN |
| INST_PAID_SUM | Installment repayments | 2.15% | 1.24% | higher -> less risk | 5.3% | INST_DUE_SUM |
| PREV_HIGH_YIELD_COUNT | Previous applications | 1.77% | 0.49% | higher -> more risk | 5.4% | POS_CONTRACT_COUNT |
| CREDIT_TO_GOODS | Application derived | 1.75% | 1.31% | higher -> more risk | 0.1% | DOWN_PAYMENT_RATE |
| EXT_SOURCE_1 | External scores | 1.67% | 1.01% | higher -> less risk | 56.3% | EXT_SOURCE_MEAN |
| EXT_SOURCE_MIN | External scores | 1.67% | 3.73% | higher -> less risk | 0.0% | EXT_SOURCE_MEAN |
| POS_INSTALLMENT_COUNT_MAX | POS/cash loan history | 1.58% | 0.56% | higher -> more risk | 6.0% | POS_INSTALLMENT_COUNT_MEAN |
| POS_FUTURE_INSTALLMENTS_MEAN | POS/cash loan history | 1.56% | 1.06% | higher -> more risk | 6.0% | POS_INSTALLMENT_COUNT_MEAN |
| DOWN_PAYMENT_RATE | Application derived | 1.47% | 0.99% | higher -> less risk | 0.1% | CREDIT_TO_GOODS |
| INST_RECENT_12M_LATE_COUNT | Installment repayments | 1.47% | 0.73% | higher -> more risk | 5.3% | INST_RECENT_24M_LATE_COUNT |
| POS_LATEST_MONTH | POS/cash loan history | 1.43% | 0.54% | higher -> less risk | 6.0% | INST_DAYS_SINCE_DUE_MIN |
| BUREAU_DEBT_TO_CREDIT | Credit bureau | 1.41% | 1.27% | higher -> more risk | 17.0% | BUREAU_DEBT_MEAN |
| EXT_SOURCE_3 | External scores | 1.37% | 2.55% | higher -> less risk | 19.8% | EXT_SOURCE_MEAN |
| INST_RECENT_24M_LATE_COUNT | Installment repayments | 1.33% | 0.89% | higher -> more risk | 5.3% | INST_RECENT_12M_LATE_COUNT |
| OWN_CAR_AGE | Application financial | 1.30% | 0.52% | higher -> more risk | 65.9% |  |
| PREV_REFUSAL_RATE | Previous applications | 1.24% | 1.26% | higher -> more risk | 5.4% | PREV_REFUSED_COUNT |
| POS_ACTIVE_MONTH_COUNT | POS/cash loan history | 1.20% | 0.63% | higher -> less risk | 6.0% | POS_MONTH_COUNT |
| PREV_ANNUITY_MEAN | Previous applications | 1.19% | 0.56% | higher -> less risk | 5.6% | PREV_ANNUITY_MAX |
| NAME_INCOME_TYPE | Application categorical | 1.14% | 0.45% | category-dependent | 0.0% |  |
| AMT_GOODS_PRICE | Application financial | 1.04% | 0.85% | higher -> less risk | 0.1% | AMT_CREDIT |
| INST_LATE_PAYMENT_RATE | Installment repayments | 1.03% | 0.84% | higher -> more risk | 5.3% | INST_LATE_DAYS_MEAN |
| INST_LATE_DAYS_MEAN | Installment repayments | 0.95% | 0.77% | higher -> more risk | 5.3% | INST_LATE_DAYS_SUM |
| EXT_SOURCE_2 | External scores | 0.95% | 1.01% | higher -> less risk | 0.2% | EXT_SOURCE_MEAN |
| BUREAU_DAYS_SINCE_CREDIT_MIN | Credit bureau | 0.94% | 1.17% | higher -> less risk | 14.2% | BUREAU_RECENT_12M_COUNT |

## Complete files

- `model_factor_audit.csv`: all raw business factors, definitions, intuitive and observed directions, magnitudes, missingness, correlations, and substitute warnings.
- `transformed_factor_audit.csv`: all direct model columns plus automatic missingness indicators.
- `factor_correlation_pairs.csv`: all numeric Spearman or categorical Cramer's V associations at absolute association >= 0.50.
- `factor_group_summary.csv`: contribution and missingness summarized by signal family.

## Important limitations

This is an observational audit of one fitted model on one random holdout from Home Credit. Marginal direction can differ from local direction because LightGBM is nonlinear and interactive. External-score definitions are anonymized. Categorical-to-numeric dependence is not summarized in the pair file. A factor-removal decision requires grouped ablation on a fresh development or out-of-time split.
