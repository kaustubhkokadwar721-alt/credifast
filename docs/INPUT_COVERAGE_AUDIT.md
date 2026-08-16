# Input coverage audit and field policy

Date: 2026-08-16. Model: `collectible-input-lightgbm-v3_extended` v0.3.0, 114 features.
Partition: **calibration only**, 46,127 rows. The V1 final holdout is spent and was not read.

Runner: `scripts/audit_input_coverage.py`.
Outputs: `artifacts/input_coverage_audit.json`, `configs/intake_requirements.json`.

## 1. Coverage — every feature is reachable

| Classification | Count | Meaning |
|---|---:|---|
| DIRECT | 16 | A named intake column maps straight to the feature |
| AGGREGATED | 86 | Computed from repeating rows in tradelines, dpd_history, account_months or enquiries |
| DERIVED | 12 | Arithmetic over other supplied values |
| **UNREACHABLE** | **0** | — |

**No feature is unreachable.** Every one of the 114 can be produced from the documented
intake schema.

### The feature the direct-entry form does not supply

`APPLICATION_MISSING_COUNT`, and it is not a defect.

The parser deliberately does not emit it. It must be counted over the **serving model's**
column list, not over whatever columns intake happened to produce, or a restricted model
would read the shape of columns it does not consume. `align_features` fills it at scoring
time. This was the fix for defect 2 in the train/serve verification work; the comment in
`workbook_intake.build_features` records it.

So "113 of 114 supplied" is correct behaviour, and 114 of 114 reach the model.

## 2. What the model actually depends on

Two measurements. **Gain** says what the model used when data was present. **Ablation** says
what breaks when a field is absent. Only the second justifies making a field mandatory, so
the policy is built on ablation.

Ablation masks by intake field, not by model feature, because a field is the only thing a
reviewer can be required to supply. Features fed by one field are masked together, missing
stays missing (never zero), availability flags drop to zero exactly as the parser sets them,
`APPLICATION_MISSING_COUNT` is recomputed, and no model is refitted.

Baseline: average precision 0.2251, ROC-AUC 0.7413.

| Field | Gain share | ΔAP | ΔROC-AUC | Mean prob shift | **Route change** |
|---|---:|---:|---:|---:|---:|
| `sheet.tradelines` | 28.83% | −0.02528 | −0.02250 | 0.02003 | **23.568%** |
| `sheet.account_months` | 23.94% | −0.01955 | −0.01995 | 0.01666 | **23.568%** |
| `account_months.balance_and_limit` | 5.11% | −0.00509 | −0.00270 | 0.00516 | 1.771% |
| `application.annual_annuity` | 14.91% | −0.01615 | −0.01392 | 0.01505 | 1.450% |
| `application.requested_credit` | 18.28% | −0.01714 | −0.01743 | 0.01567 | 1.442% |
| `tradelines.opened_days_ago` | 11.69% | −0.00616 | −0.00429 | 0.01044 | 1.340% |
| `application.employment_years` | 7.51% | −0.00465 | −0.00559 | 0.01187 | 1.095% |
| `application.goods_price` | 7.73% | −0.00673 | −0.00622 | 0.01122 | 1.036% |
| `tradelines.current_balance` | 6.20% | −0.00521 | −0.00377 | 0.00781 | 0.967% |
| `tradelines.sanctioned_amount` | 7.94% | −0.00508 | −0.00430 | 0.00735 | 0.843% |
| `account_months.installments_total` | 1.74% | −0.00147 | −0.00180 | 0.00439 | 0.644% |
| `application.education_type` | 3.82% | −0.00457 | −0.00462 | 0.00689 | 0.614% |
| `account_months.contract_status` | 1.99% | −0.00176 | −0.00161 | 0.00362 | 0.603% |
| `tradelines.status` | 3.90% | −0.00310 | −0.00182 | 0.00477 | 0.596% |
| `tradelines.credit_limit` | 1.31% | −0.00203 | −0.00156 | 0.00480 | 0.570% |
| `account_months.installments_remaining` | 1.94% | −0.00144 | −0.00121 | 0.00317 | 0.434% |
| `account_months.balance` | 1.18% | −0.00055 | −0.00078 | 0.00104 | 0.401% |
| `application.income_type` | 1.38% | −0.00192 | −0.00206 | 0.00438 | 0.399% |
| `application.address_vintage_years` | 1.88% | −0.00158 | −0.00063 | 0.00339 | 0.358% |
| `application.annual_income` | 3.46% | −0.00241 | −0.00056 | 0.00399 | 0.347% |
| `sheet.dpd_history` | 2.22% | −0.00166 | −0.00133 | 0.00252 | 0.334% |
| `application.car_age_years` | 1.34% | −0.00091 | −0.00086 | 0.00250 | 0.256% |
| `application.id_document_age_years` | 1.66% | −0.00134 | −0.00120 | 0.00322 | 0.245% |
| `account_months.dpd_days` | 2.84% | −0.00076 | −0.00168 | 0.00223 | 0.219% |
| `account_months.dpd_days_tolerant` | 2.95% | −0.00164 | −0.00150 | 0.00259 | 0.210% |
| `account_months.credit_limit` | 0.51% | −0.00003 | +0.00019 | 0.00062 | 0.208% |
| `sheet.enquiries` | 0.50% | +0.00003 | −0.00008 | 0.00114 | 0.126% |
| `application.has_work_phone` | 0.69% | −0.00117 | −0.00053 | 0.00226 | 0.124% |
| `tradelines.annuity` | 0.57% | −0.00009 | +0.00019 | 0.00047 | 0.072% |
| `tradelines.credit_type` | 0.08% | +0.00007 | +0.00000 | 0.00044 | 0.065% |
| `tradelines.overdue_amount` | 1.13% | −0.00171 | −0.00040 | 0.00046 | 0.041% |
| `application.has_email` | 0.04% | +0.00001 | +0.00002 | 0.00046 | 0.039% |
| `application.housing_type` | 0.31% | −0.00043 | −0.00030 | 0.00073 | 0.035% |
| `application.contract_type` | 0.64% | −0.00042 | −0.00054 | 0.00106 | 0.026% |
| `tradelines.days_overdue` | 0.32% | −0.00030 | −0.00007 | 0.00030 | 0.026% |
| `tradelines.times_prolonged` | 0.03% | +0.00000 | −0.00001 | 0.00013 | 0.013% |
| `application.owns_car` | 0.01% | +0.00000 | +0.00000 | 0.00035 | 0.007% |
| `application.owns_realty` | 0.02% | −0.00002 | −0.00001 | 0.00027 | 0.002%
 |

### What this says

**Two sheets carry the system.** Losing `tradelines` or `account_months` re-routes 23.6% of
cases — more than thirteen times any individual field. Both drop the source count below
three, which forces data-limited manual review by policy. Everything else is a rounding
error by comparison.

**Gain and absence-impact disagree, exactly as expected.** `application.annual_income`
carries 3.46% of gain but changes only 0.347% of routes. `tradelines.overdue_amount` carries
1.13% of gain and changes 0.041%. Ranking fields by gain would have produced a different and
wrong policy. This is why the tiering is built on ablation.

**Several fields do essentially nothing.** `owns_realty` (0.002%), `owns_car` (0.007%),
`times_prolonged` (0.013%), `days_overdue` (0.026%), `contract_type` (0.026%),
`housing_type` (0.035%), `has_email` (0.039%). Their combined gain share is under 1.5%.

## 3. Field policy

### The threshold

**1.00% route-change share** separates EXPECTED from OPTIONAL.

Chosen because the route is the operational output. Below 1%, a field's absence redirects
fewer than one case in a hundred — inside the variation a review queue already absorbs, and
not worth blocking or warning a reviewer over. At or above it, absence systematically moves
work between queues, which a reviewer needs told.

The distribution supports the cut rather than merely permitting it: there is a clear gap
between 1.036% and 0.967%, and the two sheets sit an order of magnitude above everything
else. No field lands ambiguously near the line.

REQUIRED is assigned by a different criterion: the parser cannot construct a row without the
field and raises `IntakeError`. That is a structural fact, not a measurement.

| Tier | Count | Behaviour on absence |
|---|---:|---|
| REQUIRED | 10 | Intake rejected |
| EXPECTED | 5 | Accepted, forced to data-limited manual review, named in findings |
| OPTIONAL | 23 | Accepted, recorded as a finding, routing unchanged |

### REQUIRED (10)

`application.annual_income`, `requested_credit`, `annual_annuity`, `contract_type`,
`income_type`, `education_type`, `housing_type`, `owns_car`, `owns_realty`,
`employment_years`.

**Plan on absence:** the parser raises and names the missing fields. The reviewer is told
which fields are absent and that the package cannot be scored until they are supplied.

**Finding worth acting on:** four of these have near-zero model dependence —
`contract_type` 0.026%, `housing_type` 0.035%, `owns_car` 0.007%, `owns_realty` 0.002%.
They are required only because `REQUIRED_APPLICATION_COLUMNS` says so, not because the model
needs them. Relaxing them to OPTIONAL would shorten the form with no measurable effect on
routing. That change belongs to the parser, which this audit is forbidden to modify, so it
is recorded here as a recommendation.

`annual_income` stays REQUIRED on different grounds: the risk model barely uses it (0.347%),
but every affordability ratio divides by it. Removing it would silently disable the
affordability screen.

### EXPECTED (5)

| Field | Route change | Plan on absence |
|---|---:|---|
| `sheet.tradelines` | 23.568% | Case becomes a thin file. Reviewer told no credit report was supplied and that bureau features stay missing, not zero. Resolved by a credit information report. |
| `sheet.account_months` | 23.568% | Revolving and installment behaviour stay missing; source count drops below three. Reviewer told which families are absent. Resolved by per-account monthly detail from the report. |
| `account_months.balance_and_limit` | 1.771% | Utilisation cannot be computed. Reviewer told utilisation is unavailable rather than shown as zero. Resolved by monthly balance and limit for revolving accounts. |
| `tradelines.opened_days_ago` | 1.340% | Account recency and 12/24-month opening counts stay missing. Resolved by the account open date on each tradeline. |
| `application.goods_price` | 1.036% | Down-payment rate and credit-to-goods stay missing. Resolved by the invoice, quote or declared asset value. |

### OPTIONAL (23)

Everything below 1.00%. Absence is recorded as a finding, shown to the reviewer, and does
not change the route. The full list with per-field numbers is in
`configs/intake_requirements.json`.

**Plan on absence, uniformly:** the feature stays missing, never zero; the parser emits a
finding naming what was not supplied; the estimate proceeds on the evidence that exists.

### One field where the tier is misleading

`tradelines.annuity` measures OPTIONAL — 0.072% route change, 0.57% gain. The **risk model**
barely uses it.

It is nonetheless the only source of `BUREAU_ANNUITY_SUM`, which is the entire external side
of `TOTAL_OBLIGATION_TO_INCOME`. Without it the affordability screen reports "not available"
and the obligation ratio cannot be computed at all.

**The tiering in this document measures risk-model dependence only.** A field can be
optional for the model and mandatory for affordability. `tradelines.annuity` should be
treated as expected by the affordability screen even though the model does not need it, and
the intake form should say so.

## 4. Limitations

- Measured on the calibration partition of a research candidate with **no unbiased final
  evaluation**. The V1 holdout is spent. Every number inherits that.
- Ablation masks at inference time. It measures how much **this fitted model** depends on a
  field, not how accurate a model **trained without** that field would be. A field that
  ablates to near-zero might still be recoverable signal for a retrained model.
- Route-change share depends on the research routing boundary (0.114421) and the
  thin/partial rule. Change either and the tiering changes. The two sheets score 23.568%
  largely *because* of the thin/partial rule, not purely because of model behaviour.
- Fields are ablated one at a time. Combined absence is not measured and is not the sum of
  the parts; correlated fields will partly cover for each other.
- `account_months.balance_and_limit` is ablated as a pair because utilisation needs both;
  its individual columns are also measured separately and score lower.
