# Collectible-input model (V3) — intake specification

Status: **proposed, not implemented.** This document defines the feature set and workbook
schema for a model that scores an applicant from data a user supplies, rather than by looking
up a Home Credit application ID.

Supersedes nothing. V1 stays frozen and keeps powering the existing demo path until V3 has
its own measured evidence.

## 1. Design rule

Every model input must be obtainable from one of exactly three artifacts:

| Artifact | Real-world source | Demo source |
|---|---|---|
| Application form | Applicant declaration + lender origination system | User-typed workbook sheet |
| Credit information report (CIR) | TransUnion CIBIL member pull | User-supplied tradeline/DPD/enquiry sheets, synthesized from `bureau.csv` + `bureau_balance.csv` |
| Bank statement (optional) | Consented bank / Account Aggregator feed | Optional sheet, synthesized from `installments_payments.csv` |

If a field cannot come from one of those three, it is not a model input. No exceptions for
predictive power.

## 2. What is dropped from V1 and why

V1 selected 257 factors. The following are removed:

| Dropped block | Count | Reason |
|---|---:|---|
| Apartment/building descriptors (`APARTMENTS_*`, `BASEMENTAREA_*`, `ELEVATORS_*`, `ENTRANCES_*`, `FLOORSMAX/MIN_*`, `LANDAREA_*`, `LIVINGAREA_*`, `NONLIVING*`, `COMMONAREA_*`, `YEARS_BUILD_*`, `YEARS_BEGINEXPLUATATION_*`, `TOTALAREA_MODE`, `WALLSMATERIAL_MODE`, `FONDKAPREMONT_MODE`, `HOUSETYPE_MODE`, `EMERGENCYSTATE_MODE`) | 47 | Home Credit's anonymized normalized building statistics. No origination system captures them. Not collectible. |
| `FLAG_DOCUMENT_2` … `FLAG_DOCUMENT_21` | 20 | Unnamed document flags with no published meaning. Cannot be requested from an applicant. |
| `OBS_30/60_CNT_SOCIAL_CIRCLE`, `DEF_30/60_CNT_SOCIAL_CIRCLE` | 4 | Default counts among the applicant's social surroundings. Collectible in principle; excluded on guilt-by-association grounds. |
| `EXT_SOURCE_1/2/3` and their five derived aggregates | 8 | Unnamed third-party model scores. No provider identity, no way to obtain them. **This is the largest accuracy loss in the pivot.** |
| `PREV_*` (previous applications to this lender) | 30 | Application outcomes at one lender. A new applicant has none, and a CIR reports enquiries, not outcomes. Only enquiry counts survive, and those already exist as `AMT_REQ_CREDIT_BUREAU_*`. |
| `FLAG_MOBIL`, `FLAG_CONT_MOBILE`, `FLAG_PHONE`, `FLAG_EMP_PHONE`, `PHONE_CHANGE_YEARS` | 5 | Near-zero signal; lengthens the form for nothing. |

Retained exclusions from V1 policy, unchanged: `CODE_GENDER`, `DAYS_BIRTH`, `OCCUPATION_TYPE`,
`ORGANIZATION_TYPE`, `NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`, region ratings,
and process-timing fields. All are collectible; all stay out by fairness policy. Revisiting that
is a product decision, not a schema one.

## 3. Feature set

### 3.1 Core build — CIR + application only (73 features)

The clean story: everything in the model comes from a credit report and an application form.
Count verified against the real V1 selection list, not estimated.

| Family | Count | Origin |
|---|---:|---|
| Application declared | 17 | Application sheet |
| Computed ratios | 8 | Derived, never typed |
| Bureau tradeline aggregates (`BUREAU_*`) | 31 | Tradelines sheet |
| Bureau monthly delinquency (`BB_*`) | 10 | Monthly DPD sheet |
| Enquiry velocity (`AMT_REQ_CREDIT_BUREAU_*`) | 6 | Enquiries sheet |
| Availability (`HAS_BUREAU_HISTORY`) | 1 | Derived |

**Application declared (17)**

`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `NAME_CONTRACT_TYPE`,
`NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_HOUSING_TYPE`, `FLAG_OWN_CAR`,
`FLAG_OWN_REALTY`, `OWN_CAR_AGE`, `EMPLOYMENT_YEARS`, `EMPLOYMENT_SENTINEL`,
`REGISTRATION_YEARS`, `ID_AGE_YEARS`, `FLAG_EMAIL`, `FLAG_WORK_PHONE`.

**Computed ratios (8)**

Six carried over: `ANNUITY_TO_INCOME`, `CREDIT_TO_INCOME`, `CREDIT_TO_ANNUITY`,
`CREDIT_TO_GOODS`, `DOWN_PAYMENT_RATE`, `APPLICATION_MISSING_COUNT`.

`APPLICATION_MISSING_COUNT` is **recomputed over each variant's own columns**. V1 counts nulls
across every merged column, which would leak the shape of the dropped Home-Credit-only fields
into a restricted model.

Two new, and these are the ones an underwriter actually reads:

- `TOTAL_OBLIGATION_TO_INCOME` = (`AMT_ANNUITY` + `BUREAU_ANNUITY_SUM`) ÷ `AMT_INCOME_TOTAL`
  — post-disbursal FOIR/DBR, the standard Indian retail affordability measure.
- `EXTERNAL_DEBT_TO_INCOME` = `BUREAU_DEBT_SUM` ÷ `AMT_INCOME_TOTAL`.

Both are currently unavailable to V1's affordability screen, which is a named limitation in
`codex.md` §12. They arrive for free here.

### 3.2 Extended build — adds revolving/installment tradeline behavior (+~55)

`CC_*` and `POS_*` describe monthly balance, limit, utilization, and DPD on revolving and
installment accounts. A CIR carries the same semantics for every account at every lender.
Home Credit's versions are own-ledger, so using them is a **documented proxy**, not an
equivalence.

Kept from `CC_*`: balance, limit, utilization, DPD, month counts, contract counts.
Dropped from `CC_*`: `CC_DRAWINGS_*`, `CC_DRAWING_COUNT_SUM`, `CC_PAYMENT_TOTAL_SUM`,
`CC_RECEIVABLE_*` — transaction-level detail a CIR does not report.

Kept from `POS_*`: DPD max/mean/counts/rates, month counts, contract counts, remaining
installments.

`INST_*` (23 features: scheduled vs actual payment amounts and dates) requires the optional
bank-statement sheet. Excluded from the core and extended builds; gated behind Tier 5 below.

Both builds get trained and compared. The core build is the headline if the gap is small.

## 4. Workbook schema

One `.xlsx` workbook or a directory of CSVs. Sheet names fixed.

### Sheet 1 — `application` (one row)

| Column | Type | Required | Notes |
|---|---|---|---|
| `applicant_ref` | string | yes | Free-text reference. Not a model input. |
| `annual_income` | number | yes | |
| `requested_credit` | number | yes | |
| `annual_annuity` | number | yes | Proposed annual repayment |
| `goods_price` | number | no | Financed asset value |
| `contract_type` | enum | yes | `Cash loans` \| `Revolving loans` |
| `income_type` | enum | yes | Home Credit `NAME_INCOME_TYPE` domain |
| `education_type` | enum | yes | |
| `housing_type` | enum | yes | |
| `owns_car` | Y/N | yes | |
| `car_age_years` | number | no | Required when `owns_car = Y` |
| `owns_realty` | Y/N | yes | |
| `employment_years` | number | yes | Negative/sentinel handled as `EMPLOYMENT_SENTINEL` |
| `address_vintage_years` | number | no | → `REGISTRATION_YEARS` |
| `id_document_age_years` | number | no | → `ID_AGE_YEARS` |
| `has_email` | Y/N | no | |
| `has_work_phone` | Y/N | no | |

Missing optional fields stay **missing**, never zero. That rule is inherited from V1 and is
not negotiable.

### Sheet 2 — `tradelines` (one row per credit account)

| Column | Type | Notes |
|---|---|---|
| `account_ref` | string | Join key to `dpd_history` |
| `credit_type` | string | Consumer loan, credit card, mortgage, … |
| `status` | enum | `Active` \| `Closed` \| `Sold` \| `Bad debt` |
| `opened_days_ago` | integer | → `BUREAU_DAYS_SINCE_CREDIT_*`, recency counts |
| `sanctioned_amount` | number | |
| `credit_limit` | number | Revolving accounts |
| `current_balance` | number | |
| `overdue_amount` | number | |
| `days_overdue` | integer | |
| `annuity` | number | Account EMI — feeds `BUREAU_ANNUITY_SUM` and therefore FOIR |
| `times_prolonged` | integer | |

An applicant with no credit history submits this sheet empty. That produces
`HAS_BUREAU_HISTORY = 0`, all bureau numerics missing, and the existing thin-file route.
Fail-closed behavior is unchanged.

### Sheet 3 — `dpd_history` (one row per account-month, or one row per account with a strip)

| Column | Type | Notes |
|---|---|---|
| `account_ref` | string | |
| `months_ago` | integer | 0 = most recent reported month |
| `status` | enum | `C` closed, `X` unknown, `0`–`5` delinquency buckets |

Alternate accepted form: one row per account with `dpd_strip` = a 36-character string, most
recent month first. The parser expands it.

### Sheet 4 — `account_months` (one row per account-month)

Per-account monthly detail. Feeds the revolving (`CC_*`) and installment (`POS_*`) behaviour
families the extended model consumes. A credit report carries this per account; it is kept
separate from `dpd_history` because the model was trained on two distinct source families and
merging them at intake would change what the features mean.

| Column | Type | Notes |
|---|---|---|
| `account_ref` | string | |
| `account_kind` | enum | `revolving` \| `installment` |
| `months_ago` | integer | 0 = most recent reported month |
| `dpd_days` | number | Days past due that month |
| `dpd_days_tolerant` | number | Days past due excluding tolerated amounts |
| `contract_status` | string | `Active`, `Completed`, … |
| `balance` | number | Revolving accounts |
| `credit_limit` | number | Revolving accounts |
| `installments_total` | number | Installment accounts |
| `installments_remaining` | number | Installment accounts |

### Sheet 5 — `enquiries` (one row per enquiry)

| Column | Type |
|---|---|
| `days_ago` | integer |
| `purpose` | string (not a model input; reviewer evidence) |

Aggregated into the six `AMT_REQ_CREDIT_BUREAU_*` fields. **These are disjoint bands, not
cumulative windows** — verified against `application_train.csv`, where applicant 100059 carries
`WEEK=1` with `MON=0` and `QRT=0`, impossible under nesting, and `QRT >= MON` holds for only
227,480 of 265,992 non-null rows. Each field counts its own period alone. Enquiries older than
365 days fall outside every band and are reported as a finding rather than silently dropped.

### Sheet 6 — `bank_statement` (optional, Tier 5)

Enables the `INST_*` family and bounce evidence. Not in the core or extended builds.

| Column | Type |
|---|---|
| `days_ago` | integer |
| `amount` | number |
| `direction` | `credit` \| `debit` |
| `category` | `salary` \| `emi` \| `other` |
| `returned` | Y/N |

## 5. Training plan

Train on Home Credit by restricting to the mapped columns. Split assignment stays deterministic
at seed 42 and reuses the existing development/calibration partitions.

**The V1 final holdout is spent and stays spent.** V3 is selected on calibration rows only.
V3 carries the same "research candidate pending new untouched evaluation" label as V2.

Comparison set:

1. V1 frozen — reference only, has `EXT_SOURCE`, not a fair comparator.
2. V1 minus `EXT_SOURCE` — the honest baseline for what the pivot costs.
3. V3 core (75 features).
4. V3 extended (core + `CC_*`/`POS_*` subset).

Reported for each: ROC-AUC, average precision, Brier, log loss, calibration slope/intercept,
top-10/top-20 capture, paired bootstrap CI against baseline 2, and full/partial/thin segment
breakdown.

### Measured result

Run on 2026-08-15. Full evidence in `docs/PHASE7_COLLECTIBLE_MODEL_STATUS.md`.

| Variant | Features | ROC-AUC | Avg precision |
|---|---:|---:|---:|
| `v1_reference` | 259 | 0.7815 | 0.2741 |
| `v1_no_external_score` | 251 | 0.7553 | 0.2384 |
| `v3_core` | 73 | 0.7254 | 0.2096 |
| `v3_extended` | 114 | 0.7415 | 0.2238 |

`EXT_SOURCE` removal accounts for most of the loss (−0.0263 ROC-AUC). Dropping the remaining
137 uncollectible fields costs only −0.0137 further. `v3_extended` is the recommended candidate.
The pre-registered prediction below was correct for `v3_extended` and slightly optimistic for
`v3_core`, whose ROC-AUC landed below the predicted floor.

### Expected cost, stated before measuring

Removing `EXT_SOURCE` alone is expected to be the dominant loss — those three features are
among the strongest signals in this dataset. A drop from V1's calibration ROC-AUC 0.7814 /
AP 0.2739 into roughly the 0.73–0.76 ROC-AUC band is the honest expectation, not a failure.
Recording this prediction now so the result cannot be retrofitted into a success story.

The product gains a model that can actually score a submitted applicant. That is worth more
than the AP.

## 6. What this changes downstream

- `POST /v1/model/score` gains a sibling that accepts a workbook/JSON payload instead of an
  application ID. The ID-based endpoint stays for V1.
- The dashboard gains an upload path and a template download.
- Affordability screen consumes `TOTAL_OBLIGATION_TO_INCOME` instead of proposed annuity alone.
- Source-coverage language on the primary surface is replaced by named facts ("no bureau record
  supplied", "6 tradelines, oldest 71 months"). Coverage stays as the routing mechanism.

## 7. Train/serve consistency

The workbook path is verified against the training path, not assumed equal to it.

- `tests/test_workbook_intake.py` reconstructs a real applicant's bureau and account-month
  records from the raw CSVs, runs them through the workbook parser, and asserts every
  `BUREAU_*`, `BB_*`, `CC_*` and `POS_*` value equals the built V1 feature store exactly.
- `scripts/verify_collectible_serving.py` scores calibration-partition applicants twice —
  once from the engineered training frame, once through export → parse → score — and reports
  the maximum absolute probability difference. Current result: **0.000e+00 across 8
  applicants**, recorded in `artifacts/collectible_serving_verification.json`.

Three defects were found and fixed by that check, none of which the unit tests caught:

1. Enquiry bands were implemented as cumulative windows; the source data proves they are
   disjoint.
2. `APPLICATION_MISSING_COUNT` was counted over whatever columns intake produced rather than
   over the serving model's column list. It is now computed during feature alignment.
3. `TOTAL_OBLIGATION_TO_INCOME` read an absent external annuity as zero on the training side.
   Corrected so the ratio stays missing when `BUREAU_ANNUITY_SUM` is absent — treating absence
   as zero would understate obligation for exactly the thin-file cases that need care most.

## 8. Open items

- Whether to re-include `OCCUPATION_TYPE` / `ORGANIZATION_TYPE` / dependants. Collectible and
  standard in underwriting; currently excluded by fairness policy. Product decision.
- V3 needs a new untouched evaluation set before any promotion claim, same as V2.
- Home Credit's `CC_*`/`POS_*` are own-ledger proxies for CIR tradeline behavior. Documented,
  not equivalent.
- Gap strategy, confidence routing, reason codes and the fairness audit have not been redone
  for the V3 feature set.
