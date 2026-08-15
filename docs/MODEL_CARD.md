# CrediFast research model card

This card covers **two models**. They are not interchangeable and must not be quoted as one.

| | Frozen V1 champion | Collectible-input candidate (V3) |
|---|---|---|
| Scores | A known Home Credit application ID | An applicant-supplied intake workbook |
| Features | 257 | 114 |
| Final evidence | One-time sealed holdout, all gates passed | **None.** Calibration partition only |
| Powers | The current API and dashboard | Nothing yet; library only |
| Status | Research prototype, not production approved | Research candidate, not validated |

Sections up to "Monitoring requirements" describe **V1**. The V3 section is at the end and is
self-contained.

## Model identity

- **System:** CrediFast credit-rating decision-support prototype
- **Base model:** history-enriched LightGBM challenger, version 0.2.0
- **Calibration:** identity calibrator, version 0.1.0
- **Gap policy:** history-gap-strategy-0.1.0
- **Confidence policy:** history-confidence-0.1.0
- **Status:** random-holdout technical gates passed; not production approved

## Intended use

The model supports a hackathon demonstration of faster credit-risk review. It estimates the probability represented by the Home Credit `TARGET`: repayment difficulty in the competition population. It can rank applications, provide an internal research score/grade, expose controlled model reasons, and route incomplete histories to manual review.

It is not intended for autonomous approval, decline, pricing, credit limits, adverse-action notices, or regulatory decisions. It must not be transferred directly to an Indian lender population or any other real applicant population.

## Data

The development dataset is Kaggle Home Credit Default Risk. The local manifest records 2.684 GB of extracted data and approximately 58.5 million relational rows across application, bureau, monthly bureau, previous application, POS, credit-card, and installment sources. The applicant feature store has 356,255 unique rows and 150 target-free historical features plus `SK_ID_CURR`.

Historical source coverage in the feature store is approximately 85.84% bureau, 95.12% previous applications, 94.67% POS, 29.07% credit card, and 95.32% installments. Coverage is not random and can encode product access or prior selection.

Raw data, model binaries, row-level predictions, and credentials are Git-ignored.

## Split and leakage controls

- 70% development, 15% calibration, 15% sealed holdout, stratified with seed 42.
- Models are trained only on development rows.
- Calibration method and gap weights use internal splits of calibration rows.
- Gap weights are selected on `calibrator_fit`; promotion metrics use `calibrator_evaluation`.
- The sealed random holdout was scored exactly once after the model, calibration, and gap policy were frozen. The evaluator refuses to overwrite its artifact.
- Historical aggregation is target-free and produces one row per applicant.
- Protected audit fields `CODE_GENDER` and `DAYS_BIRTH` are excluded from prediction features.

## Model inputs

Inputs include current application attributes, requested amount and annuity, income and affordability ratios, external score fields, bureau account summaries, previous-application outcomes, POS behavior, credit-card utilization/payment behavior, installment timeliness, recency, counts, trends, and explicit source-availability/missingness indicators.

Missing numeric history remains missing and is handled through the fitted preprocessing and LightGBM path. Availability flags are binary and are never used to pretend that an unknown monetary or behavioral value is zero.

## Validation performance

On the full calibration partition, the history LightGBM reports ROC-AUC 0.7814, average precision 0.2739, Brier 0.06629, and top-20% event capture 55.88%. It improves over the application-only LightGBM (ROC-AUC 0.7588, average precision 0.2492, Brier 0.06762) and the logistic baseline (ROC-AUC 0.7459).

On the disjoint 23,064-row calibration-evaluation subset, the operational gap candidate reports ROC-AUC 0.7794, average precision 0.2869, Brier 0.065798, log loss 0.23827, and ECE 0.00370. The improvement over history-only is small and statistically uncertain; see `PHASE5_GAP_STRATEGY_STATUS.md`.

On the one-time 46,127-row sealed random holdout, the frozen candidate reports ROC-AUC 0.78493, average precision 0.27988, Brier 0.066071, log loss 0.23785, ECE 0.00342, calibration slope 1.023, top-decile lift 3.68, and top-20% event capture 56.50%. Average precision is 3.47 times the 8.07% event prevalence, and Brier improves on the base-rate predictor's 0.07422. Every predeclared technical gate passed.

The partial-history blend replicated its validation improvement on holdout: versus history-only, ROC-AUC changes from 0.77375 to 0.77383, average precision from 0.28373 to 0.28485, Brier from 0.079807 to 0.079688, and log loss from 0.277627 to 0.277582. The overall candidate/history difference remains small and uncertain; its paired Brier-delta 95% interval crosses zero.

These random-split competition metrics are not estimates of real lender performance.

## Calibration and data gaps

Identity calibration beat Platt and isotonic alternatives on disjoint calibration-evaluation rows. Full histories are best served by the history model. Partial histories use a validated 75% history / 25% application blend and remain manual-review cases. Thin histories retain the history model and remain manual-review cases.

A missing feature-store row produces an explicit quality flag, zero availability sources, low confidence, and `MANUAL_REVIEW_DATA_LIMITED`. Numeric history is not zero-filled. Holdout thin-file ranking is strong, but its calibration slope is 1.290 and ECE is 0.0145. Exact zero-source holdout evidence remains exploratory: 339 rows, 21 events, calibration slope 1.350, and ECE 0.0369. These findings prohibit an automated thin-file route.

## Explanations

TreeSHAP contributions are aggregated into a controlled reason-code catalogue. The scorer can return adverse and favorable reasons for each component. For blended scores, the primary reason list represents the dominant component and is explicitly labelled `dominant_component_not_full_blend`; component-level reasons are also retained. These research reasons are not compliant adverse-action notices.

## Fairness audit

The research audit uses protected fields only after prediction. At a fixed top-20% review threshold, observed female/male true-positive-rate and review-rate gaps are approximately 7.78 and 5.95 percentage points. Age bands show larger variation, including much higher review and capture rates for younger applicants than for the 65+ band. ROC-AUC by gender is similar, but similar ranking performance does not remove calibration, error-rate, representation, or proxy concerns.

The audit is descriptive, not causal, and does not establish legal fairness. Before any real use, conduct lender-population validation, proxy/ablation analysis, uncertainty-aware mitigation review, and independent legal/compliance assessment.

## Key limitations

- Competition target and population do not match a lender-owned production definition.
- Random splits do not test economic, temporal, or geographic drift.
- Observed borrowers create selection bias; rejected applicants are not represented.
- Coverage gaps can correlate with access, age, product history, or other protected/proxy factors.
- Exact two-source and zero-source strata have low event counts; thin-file calibration is not acceptable for autonomous use.
- Current risk grades and policy boundaries are research defaults, not cost-optimized lending policy.
- Explanation stability code exists, but the final stability experiment and governance sign-off remain pending.
- No independent validation, security approval, privacy assessment, or compliance approval has occurred.

## Monitoring requirements for any adaptation

Track schema validity, feature-store row misses, source coverage by segment, feature missingness, score and grade distributions, component disagreement, calibration and discrimination after outcomes mature, review rates and errors by approved audit groups, data drift, overrides, and model/policy versions. A sustained store outage or schema mismatch must fail closed to manual review.

---

# Second model: collectible-input candidate (V3)

Everything below describes a **different model** from the one above. It is not the model
serving the current API or dashboard.

## Identity

- **Model:** `collectible-input-lightgbm-v3_extended`, version 0.3.0
- **Calibration:** identity calibrator (selected on the same protocol V1 used)
- **Feature policy:** collectible-features 1.0.0, 114 features
- **Intake schema:** workbook intake 1.1.0
- **Status:** research candidate. **Not validated.** No unbiased final evaluation exists.

## Why it exists

V1 cannot score an applicant who is not already a row in the training population. Of its 88
application-side factors, 47 are Home Credit's anonymised building-block statistics, 20 are
unnamed document flags, and 4 are social-circle default counts — none obtainable from a real
applicant. Three more inputs are unnamed third-party scores with no published provider.

V3 restricts the model to inputs obtainable from an application form, a credit information
report, and arithmetic over those two. See `docs/COLLECTIBLE_INPUT_SPEC.md`.

## Inputs

17 declared application fields, 8 computed ratios, 31 bureau tradeline aggregates, 10 monthly
delinquency features, 6 enquiry-band counts, 3 source-availability flags, and 39 revolving and
installment tradeline-behaviour features.

Two ratios are new and are the ones an underwriter reads directly:
`TOTAL_OBLIGATION_TO_INCOME` (post-disbursal FOIR, proposed repayment plus reported external
annuities over declared income) and `EXTERNAL_DEBT_TO_INCOME`. Both stay **missing** when the
external side is unobserved; absence is never read as zero obligation.

`CC_*` and `POS_*` are trained on Home Credit's own-ledger monthly records as a **documented
proxy** for credit-report tradeline behaviour. Their semantics must be reconciled against a
real report payload before any deployment claim.

Protected fields `CODE_GENDER`, `DAYS_BIRTH`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, family
counts and region ratings remain excluded from prediction, as in V1.

## Validation performance

**Calibration partition only, 46,127 rows, 8.07% prevalence. The V1 final holdout is spent and
was not read. These numbers are not unbiased final performance.**

| Variant | Features | ROC-AUC | Avg precision | Brier | Top-20 capture |
|---|---:|---:|---:|---:|---:|
| V1 feature policy (reference) | 259 | 0.7822 | 0.2775 | 0.06616 | 56.10% |
| V1 minus external scores | 251 | 0.7547 | 0.2373 | 0.06811 | 52.20% |
| V3 core (form + report) | 73 | 0.7248 | 0.2099 | 0.06955 | 47.13% |
| **V3 extended (shipped candidate)** | 114 | **0.7413** | **0.2251** | 0.06879 | 50.00% |

Paired bootstrap, 300 iterations, against the no-external-score baseline: V3 extended ΔAP
−0.0123 [−0.0194, −0.0056], ΔROC-AUC −0.0132 [−0.0177, −0.0086].

Removing the unobtainable external scores accounts for most of the loss (−0.0276 ROC-AUC).
Dropping the remaining 137 uncollectible fields costs a further −0.0132 only.

Identity calibration was selected again over Platt and isotonic, fitted on one half of the
calibration partition and chosen on the disjoint half.

## Train/serve consistency

`scripts/verify_collectible_serving.py` scores calibration applicants twice — from the
engineered training frame, and through the workbook export/parse/score path — and reports the
maximum absolute probability difference. Current result: **0.000e+00 across 8 applicants**.

That check found and corrected three defects, including one where an absent external annuity
was being read as zero on the training side.

## Fairness audit

Full result: `docs/PHASE7_COLLECTIBLE_FAIRNESS.md`. Audited on 23,064 calibrator-evaluation
rows using the identical method to the V1 audit. Assessment: `share_with_caveats`.

**The 18-24 band degraded materially and must not be represented by the overall figure.**

| 18-24 band | V1 | V3 |
|---|---:|---:|
| ROC-AUC | 0.7345 | **0.6637** |
| Review rate at 20% capacity | 46.2% | **51.5%** |
| True positive rate | 0.8205 | **0.6923** |
| ECE | — | 0.0394 |

More of that band is routed into review and fewer of its actual events are caught. Overall
ROC-AUC of 0.7404 does not describe this band and must never be quoted as if it did.

Age-band spread widened overall: review-rate gap 0.4056 → 0.4874, false-positive-rate gap
0.3639 → 0.4668, ROC-AUC gap 0.0701 → 0.0857 (all 18-24 versus 65+, except ROC-AUC).

Gender routing gaps narrowed: review rate 0.0595 → 0.0462, true positive rate 0.0778 → 0.0634,
false positive rate 0.0465 → 0.0353. Gender ranking gaps widened slightly.

Thin-file calibration improved markedly: slope 1.000 and ECE 0.0113, against V1's holdout thin
slope of 1.290 and zero-source slope of 1.350. All source-availability gaps fell below the
attention thresholds. **This does not license relaxing the thin-file manual-review rule** —
that would be a policy change resting on calibration-partition evidence for an unvalidated
candidate.

## Key limitations

- **No unbiased final evaluation.** Selection used the calibration partition only. The V1
  holdout is spent, so this candidate cannot be described as validated, and a new untouched or
  out-of-time dataset is required before any promotion claim.
- The 18-24 accuracy and routing regression above.
- `CC_*`/`POS_*` are own-ledger proxies, not verified equivalents of report tradeline data.
- Enquiry bands are implemented as disjoint periods. This was verified against the source data
  rather than published documentation.
- Gap strategy, confidence routing and reason codes have not been rebuilt for this feature set;
  V1's versions are not transferable without re-testing.
- Not wired to any interface. No endpoint or dashboard serves it.
- Every V1 limitation about competition data, random splits, selection bias and absent
  regulatory review applies unchanged.

## Required disclosures before any interface ships on this model

1. State the 18-24 result explicitly wherever performance is shown.
2. Keep the fixed-capacity review threshold labelled an audit device, never a decision
   threshold.
3. Keep thin and partial files on data-limited manual review despite improved calibration.
4. Label every output a research score with no final validation.
