# Application-only model milestone

## Outcome

Two real models have been trained on the Home Credit application table using a deterministic applicant split and a responsible feature policy. LightGBM is the current application-only challenger; the logistic model remains the explainable baseline.

The final 46,127-row holdout has not been scored. All reported metrics come from the 46,127-row calibration partition.

## Data split

| Partition | Rows | Positive target | Target rate |
|---|---:|---:|---:|
| Development | 215,257 | 17,377 | 8.073% |
| Calibration | 46,127 | 3,724 | 8.073% |
| Sealed holdout | 46,127 | 3,724 | 8.073% |

Assignments are stratified, deterministic with seed 42, saved locally as Parquet, and ignored by Git because they contain applicant identifiers.

## Responsible feature policy

- 113 selected inputs: 103 numerical and 10 categorical.
- Gender and age are audit-only and excluded from prediction.
- Family composition, occupation, organization, granular region/city, and application-time fields are excluded as sensitive, proxy-prone, or operationally unstable.
- Applicant ID and target are excluded.
- Employment sentinel values are converted to missing plus an indicator.
- Affordability, external-source summary, application missingness, and duration features are engineered without target information.

## Calibration-partition performance

| Metric | Logistic baseline | LightGBM challenger | Change |
|---|---:|---:|---:|
| ROC-AUC | 0.7459 | **0.7588** | +0.0129 |
| Average precision | 0.2326 | **0.2492** | +0.0166 |
| Brier score | 0.06845 | **0.06762** | -0.00083 |
| Brier skill vs base rate | 7.76% | **8.89%** | +1.13 pp |
| Top-10% lift | 3.241x | **3.415x** | +0.174x |
| Top-10% event capture | 32.41% | **34.16%** | +1.75 pp |
| Top-20% lift | 2.509x | **2.597x** | +0.087x |
| Top-20% event capture | 50.19% | **51.93%** | +1.75 pp |

The 0.5 threshold is not an operating threshold: severe class imbalance makes its recall extremely low. Policy thresholds must be selected after calibration using review capacity and loss assumptions.

## Current decision

- Keep logistic regression as the governance and explainability baseline.
- Advance LightGBM as the application-only challenger.
- Do not evaluate the sealed holdout yet.
- Do not connect this trained model to the live demo API yet.
- Next add bureau and previous-application aggregates, measure incremental value, then calibrate the selected champion.

## Limitations

- Application-only inputs omit detailed external credit histories and repayment behavior.
- External source scores remain opaque and require ablation/explanation review.
- The random stratified split is not a true out-of-time validation.
- Competition outcomes and population do not represent production Indian lending.
- The current probabilities are raw model outputs; formal Platt/isotonic selection is pending.
- No real lending approval, decline, pricing, LGD, or EAD policy has been validated.

## Reproducibility artifacts

- `artifacts/application_baseline_metrics.json`
- `artifacts/application_baseline_features.json`
- `artifacts/application_lightgbm_metrics.json`
- `artifacts/application_lightgbm_features.json`
- Local model binaries under `artifacts/models/`
- Local split assignments under `data/processed/`
