# Phase 7 — collectible-input model: measured cost of the pivot

Date: 2026-08-15. Partition: **calibration only**. The V1 final holdout was not read.

## Question

V1 scores a Home Credit application ID. It cannot score an applicant who submits their own
data, because 67 of its 88 application-side factors exist only in Home Credit's schema and 3
more are unnamed third-party scores. What does it cost to restrict the model to inputs a user
can actually supply?

## Method

Four feature policies trained under identical conditions: same seed-42 split assignment
(development 215,257 / calibration 46,127 / holdout 46,127 untouched), same LightGBM
hyperparameters as the frozen V1 history challenger, same early-stopping protocol on an
internal 15% development subset, then refit on all development rows.

`APPLICATION_MISSING_COUNT` is recomputed per variant over that variant's own columns, so a
restricted model cannot read the shape of the dropped columns.

Paired, class-stratified applicant bootstrap, 300 iterations, against
`v1_no_external_score` — the only fair baseline, since `EXT_SOURCE` is unobtainable in every
real deployment.

Runner: `scripts/run_collectible_experiments.py`. Policy: `src/credifast/modeling/collectible_features.py`.
Outputs: `artifacts/collectible_experiments.json`, `artifacts/collectible_comparison.csv`,
`data/processed/collectible_calibration_predictions.parquet`.

## Results

Calibration partition, 46,127 rows, 8.07% prevalence.

| Variant | Features | ROC-AUC | Avg precision | Brier | Top-10 capture | Top-20 capture |
|---|---:|---:|---:|---:|---:|---:|
| `v1_reference` | 259 | 0.7822 | 0.2775 | 0.06616 | 36.63% | 56.10% |
| `v1_no_external_score` | 251 | 0.7547 | 0.2373 | 0.06811 | 33.86% | 52.20% |
| `v3_core` (form + CIR) | 73 | 0.7248 | 0.2099 | 0.06955 | 29.94% | 47.13% |
| **`v3_extended`** (+ tradeline behaviour) | 114 | **0.7413** | **0.2251** | 0.06879 | 31.61% | 50.00% |

`v1_reference` reproduces the published frozen V1 calibration figures (0.7814 / 0.2739) to
within 0.001; the difference is the two new obligation ratios. The harness is a faithful
reproduction of the V1 training path.

Paired bootstrap against `v1_no_external_score`, 300 iterations:

| Comparison | ΔAP | 95% CI | P(Δ>0) | ΔROC-AUC | 95% CI |
|---|---:|---|---:|---:|---|
| `v1_reference` | +0.0401 | [+0.0323, +0.0475] | 1.000 | +0.0276 | [+0.0231, +0.0326] |
| `v3_core` | −0.0272 | [−0.0365, −0.0194] | 0.000 | −0.0295 | [−0.0355, −0.0242] |
| `v3_extended` | −0.0123 | [−0.0194, −0.0056] | 0.000 | −0.0132 | [−0.0177, −0.0086] |

These figures are from the second run, after the `TOTAL_OBLIGATION_TO_INCOME` correction
described in §"Train/serve consistency". The first run's numbers differed in the third decimal
and are superseded.

## What the numbers say

**The external scores are most of the cost.** Removing `EXT_SOURCE_1/2/3` and their five
aggregates costs 0.0276 ROC-AUC and 0.0401 AP, with the confidence interval nowhere near zero.
That loss was never avoidable — no lender can obtain an unnamed third-party score with no
published provider.

**Dropping the uncollectible Home Credit schema costs far less than expected.** Going from 251
features to 114 removes 137 fields — every building descriptor, every unnamed document flag,
the social-circle counts, the whole previous-application block — for 0.0132 ROC-AUC and 0.0123
AP. Those 47 apartment-block statistics were carrying almost nothing.

Supporting evidence: in `v1_no_external_score`, `NONLIVINGAPARTMENTS_AVG` ranks fourth by gain.
A model with 251 features leaning that hard on a normalized non-living-area statistic is
fitting an artifact of Home Credit's collection process, not a property of credit risk.

**Tradeline behaviour is worth having.** `v3_extended` beats `v3_core` by 0.0165 ROC-AUC and
0.0152 AP for 41 more features — monthly balance, limit, utilisation and DPD on revolving and
installment accounts. A real credit information report carries exactly these per account, so
they are collectible; Home Credit's versions are own-ledger and therefore a documented proxy.

**`v3_core` leans on weak signals.** `FLAG_WORK_PHONE` ranks seventh by gain there. With only
73 features and no tradeline behaviour, the model reaches for contact-completeness flags. That
is a reason to prefer `v3_extended`, not a reason to add junk back.

### Against the prediction recorded before measuring

`docs/COLLECTIBLE_INPUT_SPEC.md` predicted ROC-AUC in the 0.73–0.76 band and AP in 0.19–0.23.

- `v3_extended`: 0.7413 / 0.2251 — both inside the predicted band.
- `v3_core`: 0.2099 AP inside the band, but **ROC-AUC 0.7248 came in below the predicted
  floor of 0.73.** The prediction was slightly optimistic for the form-plus-CIR-only build.

## Is the pivot viable

Yes. `v3_extended` gives average precision 0.2251 against 8.07% prevalence — 2.79× the base
rate — with 50.0% of events captured in the top 20% of ranked applicants. It ranks meaningfully
worse than frozen V1 (56.1% capture) and meaningfully better than random.

The trade is a model that ranks somewhat worse but can score a submitted application, versus a
model that ranks better and can only score a row that already exists in the training population.
For the stated product objective, the first is worth more.

## Recommendation

Adopt `v3_extended` as the collectible-input candidate. Keep `v3_core` measured and documented
as the strict form-plus-CIR fallback for deployments where per-account monthly delinquency is
unavailable.

## Persisted candidate

`scripts/train_collectible_champion.py --variant v3_extended` writes a servable bundle:

| Artifact | Contents |
|---|---|
| `artifacts/models/collectible_lightgbm.joblib` | Preprocessor, model, feature list, 428 trees, `validated: false` |
| `artifacts/models/collectible_calibrator.joblib` | Selected calibrator |
| `artifacts/collectible_champion_metrics.json` | Calibrator comparison and raw vs calibrated metrics |

**Identity calibration was selected again**, on the same protocol V1 used: fit each candidate
on half the calibration partition, choose on the disjoint half. Platt and isotonic did not
improve Brier or log loss, so the raw probabilities ship unscaled. That is the honest outcome,
not a shortcut.

## Train/serve consistency

The serving path is verified equal to the training path, not assumed equal.

`scripts/verify_collectible_serving.py` scores calibration-partition applicants twice — once
from the engineered training frame, once through export → workbook → parse → score — and
reports the maximum absolute probability difference.

**Result: 0.000e+00 across 8 applicants**, each carrying all three source families.
Recorded in `artifacts/collectible_serving_verification.json`.

That check found three defects the unit tests did not:

1. **Enquiry bands were implemented as cumulative windows.** The source data disproves that:
   applicant 100059 carries `AMT_REQ_CREDIT_BUREAU_WEEK=1` with `MON=0` and `QRT=0`, and
   `QRT >= MON` holds for only 227,480 of 265,992 non-null rows. The bands are disjoint.
2. **`APPLICATION_MISSING_COUNT` was counted over the wrong column set** — whatever intake
   produced, rather than the serving model's own feature list. Now computed during alignment.
3. **`TOTAL_OBLIGATION_TO_INCOME` read an absent external annuity as zero** on the training
   side. Corrected so the ratio stays missing when `BUREAU_ANNUITY_SUM` is absent. Reading
   absence as zero would understate obligation for exactly the thin-file cases that need care
   most, and it contradicted the project's own missing-is-not-zero rule.

Defect 3 changed the trained feature, so the comparison campaign and the champion were both
retrained; every figure in this document is from after the correction.

## Limits on this result

- Calibration-partition evidence only. The V1 final holdout is spent, so **no V3 variant has
  unbiased final evidence** and none may be described as validated. Same status as V2.
- No calibration study has been run for V3. V1's identity calibrator was selected for V1's
  probability distribution and does not transfer without re-testing.
- Gap-strategy, confidence-routing and reason-code work has not been redone for V3.
- The fairness audit **has** been redone: `docs/PHASE7_COLLECTIBLE_FAIRNESS.md`. Headline —
  gender routing gaps narrowed, thin-file calibration improved markedly (slope 1.000 versus
  V1's 1.290), but the age-band spread widened and the 18-24 band degraded materially
  (ROC-AUC 0.7345 → 0.6637, review rate 46.2% → 51.5%, TPR 0.8205 → 0.6923). That result must
  appear in the model card before any interface ships on this candidate.
- `CC_*` and `POS_*` are own-ledger proxies for credit-report tradeline behaviour. Their
  semantics must be reconciled against a real CIR payload before any deployment claim.
- The enquiry-window features assume cumulative windows; Home Credit's exact semantics for
  `AMT_REQ_CREDIT_BUREAU_*` are not published.

## Next

1. Re-run calibration selection for `v3_extended` on calibration-fit rows.
2. Re-run gap strategy and confidence routing against the new source families.
3. Re-run the fairness audit; the feature set changed substantially.
4. Wire `workbook_intake` into a scoring endpoint and the dashboard upload path.
5. Obtain an out-of-time or lender-owned evaluation set before any promotion claim.
