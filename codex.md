# CrediFast project handoff

Last updated: 2026-08-15 (Asia/Calcutta)

This is the first file a new coding agent should read. It records the product intent,
technical architecture, evidence, local-only dependencies, non-negotiable modeling rules,
current repository state, and the exact next work. Read it end to end before changing code or
running another model experiment.

## 1. Executive truth

CrediFast is a local, explainable credit-risk decision-support prototype for Cognizant NPN
Hackathon line item 7, **Credit Rating Checks for Faster Lending**. It uses the Kaggle Home
Credit Default Risk dataset to estimate the competition target: probability of repayment
difficulty.

Two generations now coexist and must not be confused:

1. **V1 frozen demo candidate is complete.** It has a working FastAPI API, Streamlit reviewer
   dashboard, application-only and history-enriched LightGBM models, identity calibration,
   segment-specific data-gap handling, controlled reason codes, fairness diagnostics, and a
   one-time final holdout evaluation. This is the model used by the API and UI.
2. **V2 temporal challenger is research in progress.** Its 111-feature additive temporal store
   is built and passes its data contracts. The full feature/model ablation campaign was started
   but interrupted after three research model binaries were produced. No V2 comparison report
   exists yet, no V2 model has passed a promotion gate, and V2 is not integrated into inference,
   the API, or the dashboard.

The user's explicit priority is **accuracy, evidence quality, and principled data-gap handling
over latency**. Do not downsample merely to make a run quicker. Latency is only a local demo
smoke gate.

This remains a hackathon research system. Never describe it as an official bureau score, an
autonomous approval/decline engine, production lending software, or a compliant adverse-action
system.

## 2. Absolute locations and local environment

Project root:

```text
C:\Users\hp\Documents\Codex\2026-08-14\thoi\outputs\credifast
```

Original downloaded archive:

```text
C:\Users\hp\Documents\Codex\2026-08-14\thoi\home-credit-default-risk.zip
```

The archive has already been extracted under `data/raw/`. Do not redownload it and do not
commit it.

The working virtual environment is `.venv`. Versions observed at handoff:

| Component | Version |
|---|---:|
| Python | 3.13.3 |
| pandas | 3.0.5 |
| NumPy | 2.5.2 |
| scikit-learn | 1.9.0 |
| LightGBM | 4.7.0 |
| DuckDB | 1.5.5 |
| PyArrow | 24.0.0 |
| CatBoost | 1.2.10 |
| FastAPI | 0.141.1 |
| Streamlit | 1.61.1 |

Bootstrap from a clean environment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

The package is installed editable with optional groups `api`, `ui`, `data`, `ml`, and `dev`.
CatBoost was added to the `ml` group specifically for the V2 challenger campaign.

Approximate local footprint at handoff:

| Path | Files | Bytes |
|---|---:|---:|
| `data/raw/` | 10 | 2,684,261,617 |
| `data/processed/` | 8 | 109,640,728 |
| `artifacts/models/` | 7 | 13,556,464 |
| `artifacts/` including models | 39 | 14,615,576 |

## 3. Non-negotiable rules

### 3.1 The old final holdout is spent

The V1 candidate used a deterministic, stratified split with seed 42:

- 70% development;
- 15% calibration;
- 15% final holdout.

The 46,127-row final holdout was scored once after V1 was frozen. Its results are in
`artifacts/final_holdout_evaluation.json`. It is now **spent**.

For V2 or any later model:

- do not use those holdout rows for feature selection;
- do not use them for hyperparameter selection;
- do not use them for threshold, calibration, blend, or policy selection;
- do not report V2 performance on them as if it were unbiased;
- do not rerun `credifast-train final-holdout`;
- obtain a new untouched, preferably time-separated or lender-owned dataset for a true final
  evaluation.

The V2 experiment runner explicitly removes the old holdout by applicant ID and compares models
only on the existing calibration partition. Those results can promote a model only to further
research, never to production or to a claim of final validation.

### 3.2 Prevent leakage and grain errors

- `TARGET` must never enter a feature store or prediction matrix.
- `SK_ID_CURR`, child record IDs, and other identifiers are join keys, never predictors.
- Every child table must be aggregated to one row per applicant before joining.
- Multiple payment rows for one scheduled installment must be reconciled at installment grain
  before sequence or late/short calculations.
- Relative event dates after the current application cutoff are invalid.
- Fit selectors, imputers, encoders, calibrators, and learned blends on allowed training data
  only.
- Missing history is not observed zero. Preserve numeric nulls and use explicit source
  availability flags.
- Protected audit fields `CODE_GENDER` and `DAYS_BIRTH` are excluded from prediction and may be
  used only for post-prediction audit slices.

### 3.3 Keep responsibilities separate

The ML model estimates repayment-difficulty probability. It does not decide approval, price,
credit limit, LGD, EAD, fraud, KYC, AML, or collections action.

The application deliberately separates:

1. data validation and data confidence;
2. repayment-risk probability;
3. affordability screening;
4. decision/review policy;
5. explanations and reason codes.

Use these exact product phrases where practical:

- “repayment-difficulty probability”;
- “internal research score”;
- “decision support”;
- “manual review.”

## 4. Current Git and workspace state

There is currently **no Git commit**. The original repository contents are staged as additions.
Two earlier files have additional unstaged modifications (`pyproject.toml` and
`src/credifast/data_cli.py`), and the V2 files are untracked at handoff. Preserve everything.

Important V2 untracked files before this handoff file was added:

```text
artifacts/history_temporal_features_report.json
artifacts/history_temporal_quality_report.json
configs/temporal_challenger_experiment.json
scripts/run_temporal_experiments.py
src/credifast/data/history_features_v2.py
src/credifast/modeling/temporal_experiments.py
tests/test_history_features_v2.py
tests/test_temporal_experiments.py
```

Do not use `git reset --hard`, `git checkout --`, broad cleanup commands, or deletion of ignored
data/model directories. Treat all existing changes as user-owned. Before a first commit, inspect:

```powershell
git status --short
git diff
git diff --cached
```

Raw data, processed row-level data, model binaries, environment files, and logs are intentionally
ignored by `.gitignore`. Small non-sensitive aggregate metrics, manifests, schemas, and reports
under `artifacts/` are intended to be versioned.

## 5. Product and system architecture

The code has two scoring paths:

```text
Synthetic Phase 0 path
JSON input -> domain validation -> deterministic scorer -> policy -> CLI or POST /v1/score

Frozen trained-model path used by the demo
local application row + V1 history row
    -> application feature engineering
    -> frozen history LightGBM probability
    -> frozen application LightGBM probability
    -> identity calibration
    -> source-coverage segment blend
    -> confidence and gap routing
    -> reason-code aggregation
    -> separate affordability screen
    -> API / dashboard response
```

V2 is not part of this runtime. It is currently an offline challenger only.

### Main modules

| Path | Responsibility |
|---|---|
| `src/credifast/domain.py` | Typed Phase 0 input/output contracts. |
| `src/credifast/input_sources.py` | Versioned 257-factor provenance registry and compact JSON intake validation. |
| `src/credifast/scoring.py` | Deterministic synthetic risk scorer and score/grade mapping. |
| `src/credifast/policy.py` | Phase 0 policy and confidence rules. |
| `src/credifast/service.py` | Phase 0 orchestration. |
| `src/credifast/data/contracts.py` | Kaggle file, schema, key, and grain contracts. |
| `src/credifast/data/history_features.py` | Frozen V1 DuckDB relational history store. |
| `src/credifast/data/history_quality.py` | V1 modeling-readiness checks. |
| `src/credifast/data/history_features_v2.py` | Additive V2 temporal/burden/cycling store and validator. |
| `src/credifast/modeling/application_features.py` | Target-free application engineering and responsible feature selection. |
| `src/credifast/modeling/application_baseline.py` | Logistic baseline and deterministic split assignment. |
| `src/credifast/modeling/application_lightgbm.py` | Application-only LightGBM. |
| `src/credifast/modeling/history_lightgbm.py` | Frozen V1 history-enriched LightGBM. |
| `src/credifast/modeling/calibration.py` | Identity/Platt/isotonic comparison and frozen calibrator. |
| `src/credifast/modeling/gap_strategy.py` | Full/partial/thin source-segment blend experiments. |
| `src/credifast/modeling/confidence.py` | Data confidence and boundary routing. |
| `src/credifast/modeling/reasons.py` | TreeSHAP-to-controlled-reason mapping. |
| `src/credifast/modeling/inference.py` | Frozen batch and applicant scoring, gap-safe merge behavior. |
| `src/credifast/modeling/final_evaluation.py` | One-shot V1 holdout evaluator; do not reuse for V2 selection. |
| `src/credifast/modeling/temporal_experiments.py` | V2 ablations, LightGBM/CatBoost challengers, bootstrap gates. |
| `src/credifast/model_runtime.py` | Cached local runtime used by API and UI. |
| `src/credifast/api.py` | FastAPI endpoints. |
| `ui/app.py` | Streamlit reviewer workbench and evidence views. |

### Entrypoints

Installed console scripts:

- `credifast`: deterministic Phase 0 scorer;
- `credifast-data`: data contracts, manifests, profiling, V1 and V2 feature stores;
- `credifast-train`: V1 training, calibration, audits, gap strategy, and final evaluation;
- `credifast-score`: frozen V1 local batch/single-applicant scoring.

The V2 campaign currently uses `scripts/run_temporal_experiments.py`; it has not been added to
`credifast-train`.

## 6. Data inventory and contracts

Primary dataset: Kaggle Home Credit Default Risk.

Raw sources under `data/raw/`:

- `application_train.csv`;
- `application_test.csv`;
- `bureau.csv`;
- `bureau_balance.csv`;
- `previous_application.csv`;
- `POS_CASH_balance.csv`;
- `credit_card_balance.csv`;
- `installments_payments.csv`;
- `HomeCredit_columns_description.csv`;
- `sample_submission.csv`.

The manifest records approximately 2.684 GB and 58.5 million relational rows. Exact file hashes,
row counts, columns, and contract results are in `artifacts/data_manifest.json` and
`artifacts/application_profile.json`.

Data zones:

```text
data/raw/        immutable source CSVs; never commit
data/interim/    optional cleaned/table-grain intermediates; never commit
data/processed/  feature stores, splits, predictions, scores; never commit
artifacts/       safe aggregate evidence and manifests; model binaries still ignored
```

V1 feature store:

```text
data/processed/history_features.parquet
```

- 356,255 unique applicant rows;
- 150 target-free historical features plus key;
- all source tables aggregated independently before join;
- source coverage: bureau 85.84%, previous applications 95.12%, POS 94.67%,
  credit cards 29.07%, installments 95.32%;
- passes key, reconciliation, bounded-rate, nonnegative-count, finite-value, and coverage gates.

V2 additive feature store:

```text
data/processed/history_temporal_features.parquet
```

- 356,255 rows and 356,255 unique applicant keys;
- 111 target-free features plus key;
- 31,743,129 bytes at handoff;
- SHA-256 `d5dd527234fdef457a1318892e54286c538e898a89c21076cf6cc4cd1e7f8511`;
- validation status `ready_for_experiment=true` with zero findings;
- built for all train/test IDs, but V2 training and evaluation code excludes old holdout IDs.

V2 feature-family counts:

| Family | Count | Main meaning |
|---|---:|---|
| Installment temporal | 57 | Recency weighting, rolling late/short behavior, most-recent event sequence, streaks, cures, relapses. |
| Previous-application velocity | 14 | 30/90/180/365-day demand, bursts, gaps, refusals, cash-loan behavior, stress-following applications. |
| POS obligation | 12 | Active-loan concurrency, remaining installments, DPD recency/trajectory, cures and relapses. |
| Credit-card temporal | 13 | 3/6/12-month utilization, minimum-payment behavior, ATM usage, DPD, and slopes. |
| Bureau burden | 8 | Active credit, debt, annuity, overdue amount, recency, and anomaly checks. |
| Credit-cycling proxy | 2 | Count and flag requiring multiple independent stress families. |
| Availability | 5 | Explicit source-presence flags. |

The temporal store also feeds six transparent application/history cross features inside the V2
runner: active annuity/income, active debt/income, proposed-plus-active annuity/income,
requested-plus-active debt/income, temporal source count, and cycling-by-burden interaction.

### How time weighting works

Recent repayment evidence receives more weight than old evidence using fixed exponential decay.
Conceptually:

```text
weight = 0.5 ^ (age_in_days / half_life_in_days)
```

Installment half-lives are fixed at 90, 180, and 365 days. Monthly-history horizons are 3, 6,
and 12 months. Rolling event windows are 30, 90, 180, 365, and 730 days. The last six reconciled
installment events are also represented positionally.

These values were predeclared before model outcomes were inspected. Do not tune them against the
spent holdout.

### Credit-cycling interpretation

The data cannot prove that one loan funded another loan's EMI. V2 therefore implements only a
multi-source **stress proxy** based on timing patterns such as application bursts, recent refusals,
cash-loan applications after repayment stress, concurrent POS obligations, revolving-credit
stress, and burden. It requires at least two independent signal families for its flag. Never turn
this proxy alone into an adverse route or describe it as verified refinancing behavior.

### Quality hierarchy for inputs

When extending the model, prefer evidence in this order:

1. verified, point-in-time transaction or payment behavior;
2. independently reported active obligations and delinquency;
3. internally consistent application financial terms;
4. repeated historical behavior and cross-source agreement;
5. model-derived or inferred proxies;
6. missingness/availability patterns, used cautiously because access and selection effects may be
   embedded in them.

Important unavailable or weak evidence includes verified payroll/income, bank cash-flow data,
complete cross-lender EMI schedules, mandate failures/bounces, source freshness timestamps,
lender policy/product context, and a true lender-owned out-of-time repayment outcome. These gaps
will eventually limit accuracy more than another round of tree tuning.

## 7. Frozen V1 model and measured evidence

### Calibration-partition baselines

| Model | ROC-AUC | Average precision | Brier | Top-20% event capture |
|---|---:|---:|---:|---:|
| Logistic application baseline | 0.7459 | 0.2326 | 0.06845 | 50.19% |
| Application LightGBM | 0.7588 | 0.2492 | 0.06762 | 51.93% |
| History LightGBM | 0.7814 | 0.2739 | 0.06629 | 55.88% |

Identity calibration beat Platt and isotonic on a held-back half of calibration data. The frozen
gap policy is:

| History segment | History probability | Application probability | Route |
|---|---:|---:|---|
| Full, 4-5 sources | 100% | 0% | Standard or boundary review |
| Partial, 2-3 sources | 75% | 25% | Data-limited manual review |
| Thin, 0-1 sources | 100% | 0% | Data-limited manual review |

A missing feature-store row is a data-quality fault: numeric history stays missing, availability
flags become zero, `FEATURE_STORE_ROW_MISSING` is emitted, confidence is low, and routing fails
closed to data-limited manual review.

### One-time final holdout result

Population: 46,127 applicants, 3,724 events, prevalence 8.07%.

| Metric | Frozen operational candidate |
|---|---:|
| ROC-AUC | 0.78493 |
| Average precision | 0.27988 |
| AP / prevalence | 3.47x |
| Brier score | 0.066071 |
| Base-rate Brier | 0.074216 |
| Log loss | 0.23785 |
| Equal-count ECE | 0.00342 |
| Calibration slope | 1.023 |
| Calibration intercept | 0.0570 |
| Top-decile lift | 3.68x |
| Top-10% event capture | 36.82% |
| Top-20% event capture | 56.50% |

All predeclared V1 technical gates passed. Average precision of 0.2799 is not “28% model
accuracy”; it is the area under the precision-recall curve. Against only 8.07% prevalence it is
substantially above random, though still leaves meaningful false-positive/false-negative tradeoffs.
Use `artifacts/average_precision_diagnostic.json` and
`scripts/diagnose_average_precision.py` for operating-point detail.

Thin-file calibration is not good enough for autonomous use even though ranking is strong. On
holdout, the thin segment has slope 1.290 and exact zero-source cases have slope 1.350 and ECE
0.0369. Keep zero-to-three-source cases in manual review.

### Factors and explanations

The frozen V1 model selected 257 raw factors, expanded to 465 transformed features. Exact TreeSHAP
was calculated across all 46,127 old holdout rows. The complete factor dictionary is:

```text
artifacts/factor_audit/model_factor_audit.csv
```

Supporting files:

- `artifacts/factor_audit/README.md`;
- `artifacts/factor_audit/factor_group_summary.csv`;
- `artifacts/factor_audit/factor_correlation_pairs.csv`;
- `artifacts/factor_audit/transformed_factor_audit.csv`;
- `docs/MODEL_FACTOR_AUDIT_VALIDATION.md`.

The audit includes definitions, preprocessing, intuitive direction, observed direction and
magnitude, missingness effect, strongest correlations, and substitutability warnings. SHAP values
describe model behavior, not causal effects. Reviewer-facing reasons are a controlled subset.

A 250-row 0.5% input perturbation probe found a 98.0% top-adverse-reason match, mean top-three
Jaccard 0.990, and 95th-percentile absolute probability movement 0.00716. This is local robustness
evidence, not a global guarantee.

### Fairness evidence

The post-prediction research audit reports gender gaps at a fixed top-20% review capacity and
larger age-band variation. It is descriptive, not causal or legally sufficient. Do not use
protected fields for scoring, do not hide the caveat, and do not claim that similar ROC-AUC proves
fairness. See `artifacts/history_fairness_audit.json` and `docs/MODEL_CARD.md`.

## 8. V2 temporal challenger: exact state

Predeclared configuration:

```text
configs/temporal_challenger_experiment.json
```

Primary metric: average precision. Secondary metrics: ROC-AUC, Brier score, log loss, top-10%
capture, and top-20% capture.

The runner compares the frozen V1 baseline with:

1. repayment-recency additions;
2. repayment-sequence additions;
3. all installment-temporal additions;
4. previous-application velocity;
5. POS and bureau burden;
6. credit-cycling proxy;
7. credit-card temporal behavior;
8. all temporal features with standard LightGBM;
9. all temporal features with predeclared structural pruning;
10. all temporal features without external scores;
11. all temporal features with a higher-capacity LightGBM;
12. all temporal features with native-categorical CatBoost;
13. a fixed 50/50 probability ensemble of the two all-temporal LightGBMs.

Structural pruning removes only the predeclared algebraic/source-key equivalents:
`PREV_APPLICATION_COUNT`, `BUREAU_CREDIT_COUNT`, and `CREDIT_TO_GOODS`.

Each tree challenger uses development rows, with an internal 15% stratified development subset for
early stopping, then refits on all development rows and scores calibration rows. Paired,
class-stratified applicant bootstrap uses 300 iterations.

Promotion to **further research** requires every check:

- absolute AP gain at least 0.003 over frozen V1;
- bootstrap probability of positive AP gain at least 0.90;
- Brier regression no worse than 0.0002;
- ROC-AUC regression no worse than 0.001;
- top-20% event-capture regression no worse than 0.005;
- clean V2 data-quality gate.

### Interrupted run

The full runner was started on 2026-08-15 and then interrupted when this handoff was requested.
The process is no longer running. It had completed three local research binaries:

```text
artifacts/models/research/v1_plus_repayment_recency.joblib
artifacts/models/research/v1_plus_repayment_sequence.joblib
artifacts/models/research/v1_plus_installment_temporal.joblib
```

These files alone are **not results**. Metrics were held in memory and the runner writes the JSON,
CSV, and prediction Parquet only after all variants complete. At handoff, these expected outputs do
not exist:

```text
artifacts/temporal_challenger_experiments.json
artifacts/temporal_challenger_comparison.csv
data/processed/temporal_challenger_calibration_predictions.parquet
```

Do not infer model quality from the presence, timestamp, or size of the three partial binaries.
The current script does not resume; rerunning it overwrites/recreates research models and starts
the campaign from the beginning.

The preparation stage used roughly 5-6 GB of memory and substantial multi-core CPU. The first
three models took roughly eight minutes in total on this machine, but a full CatBoost-inclusive
run may take considerably longer. Accuracy is the priority.

Recommended reliability improvement before rerunning: add atomic per-variant checkpoints and a
validated resume mode to `temporal_experiments.py`, while preserving the predeclared variants,
split, and gates. A checkpoint must include variant config, data/model/config hashes, calibration
IDs hash, probability vector, metrics, and model hash. Resume only when every hash matches. Never
silently mix results from different stores or code/config versions.

If restarting without adding checkpoints:

```powershell
.\.venv\Scripts\python.exe scripts\run_temporal_experiments.py
```

Expected final outputs are the three paths listed above plus all research model artifacts under
`artifacts/models/research/`.

## 9. API and dashboard

Start both services:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1
```

Then open:

- dashboard: `http://127.0.0.1:8501`;
- OpenAPI: `http://127.0.0.1:8000/docs`.

Or run separately:

```powershell
.\.venv\Scripts\python.exe -m uvicorn credifast.api:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

API endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process and frozen-runtime readiness. |
| GET | `/v1/model/status` | Frozen model, holdout, and approval status. |
| GET | `/v1/model/applicants` | Seven curated offline demo profiles. |
| GET | `/v1/model/input-schema` | Editable terms, JSON template, and provenance for all 257 selected factors. |
| POST | `/v1/model/score` | Score a known local test applicant with safe overrides. |
| POST | `/v1/score` | Legacy deterministic Phase 0 contract demo. |

The trained endpoint accepts an application ID and optional overrides for only annual income,
requested credit, annual annuity, and goods price. The dashboard now offers curated-case,
known-applicant-ID, and compact JSON-package modes using `configs/intake_template.json`. It marks
each editable field by source/verification route and exposes the full factor-source map as CSV.
Historical evidence is immutable. The `simulate_history_unavailable` flag deliberately tests
fail-closed feature-store behavior. Live CIBIL, bank/Account Aggregator, and partner-ledger
connectors remain unimplemented; see `docs/INPUT_ACQUISITION.md`.

The reviewer dashboard is an “Evidence Docket”: warm paper, navy ink, cobalt actions, explicit
source coverage, flat ruled sections, no celebratory approval language, no risk status conveyed by
color alone. Product/design guidance is in `PRODUCT.md`, `DESIGN.md`, and
`.impeccable/design.json`. Desktop and mobile screenshots are in `artifacts/ui_desktop.png` and
`artifacts/ui_mobile.png`.

Curated demo flow and mandatory language are in `docs/DEMO_RUNBOOK.md`.

## 10. Commands and reproducibility

Run all project verification:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

Direct pytest command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

At the latest verification, all 49 collected pytest tests pass. The five V2-focused tests and
three input-provenance/intake tests are included in that pass. Re-run the full suite after any
experiment-checkpoint changes. Known dependency warnings are
Starlette TestClient's `httpx` deprecation and joblib assigning NumPy array shape during frozen
model loading; neither currently fails a test.

Lint:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests ui scripts
```

Rebuild and validate V1 history:

```powershell
credifast-data aggregate-history
credifast-data validate-history
```

Rebuild and validate V2 temporal history:

```powershell
credifast-data aggregate-history-v2
credifast-data validate-history-v2
```

The V2 build uses DuckDB defaults of a 4 GB memory limit and four threads. It took about 49
seconds on this machine after query corrections.

Reproduce the frozen V1 modeling stages only when necessary:

```powershell
credifast-train application-baseline
credifast-train application-lightgbm
credifast-train history-lightgbm
credifast-train history-calibrate
credifast-train history-gap-strategy
```

Do not run `credifast-train final-holdout` for new selection. The existing final artifact is
one-shot and should remain unchanged.

Score local test rows:

```powershell
credifast-score --limit 100 --no-explanations
credifast-score --application-id 176483
```

Diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_average_precision.py
.\.venv\Scripts\python.exe scripts\audit_model_factors.py
```

The factor audit is expensive and uses the spent V1 holdout only to explain the already frozen
model. Do not use its observed directions or magnitudes to tune V2 against that holdout.

## 11. Artifact map

### Canonical small evidence

| Artifact | Meaning |
|---|---|
| `artifacts/data_manifest.json` | Raw source identity, schemas, rows, hashes. |
| `artifacts/application_profile.json` | Application table and target profile. |
| `artifacts/history_features_report.json` | V1 feature-store build report. |
| `artifacts/history_quality_report.json` | V1 quality gate. |
| `artifacts/history_lightgbm_metrics.json` | V1 calibration metrics. |
| `artifacts/history_calibration_metrics.json` | Calibrator comparison and selection. |
| `artifacts/history_gap_strategy_metrics.json` | Source-segment blend experiment. |
| `artifacts/final_holdout_evaluation.json` | One-time frozen V1 final evidence. |
| `artifacts/history_fairness_audit.json` | Post-prediction group audit. |
| `artifacts/explanation_stability.json` | Local perturbation stability. |
| `artifacts/inference_benchmark.json` | Warm local inference smoke benchmark. |
| `artifacts/history_temporal_features_report.json` | V2 store identity and family counts. |
| `artifacts/history_temporal_quality_report.json` | V2 readiness gate; currently clean. |

### Local-only required V1 runtime artifacts

```text
artifacts/models/application_lightgbm.joblib
artifacts/models/history_lightgbm.joblib
artifacts/models/history_calibrator.joblib
data/raw/application_test.csv
data/processed/history_features.parquet
configs/model_gap_strategy.json
configs/demo_profiles.json
artifacts/final_holdout_evaluation.json
artifacts/history_fairness_audit.json
```

`LocalModelRuntime` checks these at startup. If any are missing, `/health` returns degraded and
trained scoring returns HTTP 503.

## 12. Known limitations and open technical risks

- Competition target/population are not a real lender's governed target or applicant population.
- Existing validation is random, not out-of-time, geographic, or economic-regime validation.
- Rejected applicants are absent, creating selection bias.
- Source coverage itself may encode access, product history, and protected/proxy effects.
- V1 final holdout is spent; V2 needs new final evidence.
- V2 campaign has no checkpoint/resume and writes its result report only at the end.
- V2 experiment preprocessing repeats feature eligibility and transformation work per ablation,
  making it accurate but memory/CPU intensive.
- API/UI still use V1 and cannot expose V2 recency, freshness, trajectory, or cycling evidence.
- The API scores known local Home Credit test applicants, not arbitrary real applications with a
  minimal schema.
- Affordability uses proposed annuity/income and cannot see complete external obligations.
- No authentication, authorization, encrypted audit log, container, production feature service,
  drift service, privacy assessment, security approval, legal review, or independent model
  validation is present.
- No lender-owned outcome, true application timestamp split, verified income, or bank cash-flow
  data is present.
- The current environment is not dependency-locked to exact versions beyond ranges in
  `pyproject.toml`.
- The repository has never been committed; establish a clean baseline commit only after user
  review.

## 13. Exact next sequence for Claude

Follow this order unless the user explicitly changes priorities.

### Step 1: establish a safe baseline

1. Read `README.md`, `PRODUCT.md`, `DESIGN.md`, `docs/MODEL_CARD.md`,
   `docs/INPUT_SIGNAL_STRATEGY.md`, and this file.
2. Inspect `git status --short` and preserve staged, unstaged, untracked, ignored, and local-only
   work.
3. Confirm no experiment process is running.
4. Run the 46-test suite and Ruff. Record exact results; do not claim a pass without running it.
5. Re-run `credifast-data validate-history-v2` before any V2 model work.

### Step 2: harden the V2 runner

Add crash-safe atomic checkpoints and hash-validated resume support. Keep these immutable:

- seed 42;
- development/calibration/old-holdout assignment;
- holdout exclusion;
- feature families and fixed horizons;
- model variants;
- structural-pruning list;
- paired bootstrap method and 300 iterations;
- promotion gates.

Add tests proving that:

- old holdout IDs cannot enter training, calibration, predictions, or checkpoint files;
- incompatible data/config/code hashes reject resume;
- an interrupted run can resume without changing completed probabilities/metrics;
- results are atomically written and partial files are labeled incomplete;
- protected fields and target never enter predictors.

### Step 3: complete the calibration-only V2 campaign

Run every predeclared variant on the full development/calibration population. Do not downsample.
Produce:

- comparison CSV;
- complete metrics JSON;
- calibration prediction Parquet;
- per-model hashes and versions;
- paired bootstrap confidence intervals;
- calibration slope/intercept and deciles;
- top-10/top-20 capture;
- full/partial/thin, exact-source-count, gender, and age-band diagnostics;
- feature-family gain importance;
- explicit failure reasons for every non-passing gate.

Do not pick a winner on AP alone. A candidate must pass every predeclared gate.

### Step 4: independently validate and document the result

Create and execute a reproducible notebook, for example
`notebooks/temporal_challenger_experiments.ipynb`, that reads saved outputs rather than rerunning
training. It should show baseline-relative AP, Brier, ROC-AUC, capture, bootstrap intervals,
calibration, segments, and feature-family contribution. Add a durable status report under `docs/`.

Check row counts, ID hashes, prediction bounds, target alignment, metric recomputation, model hashes,
and report consistency independently of the training runner.

### Step 5: decide honestly

- If no V2 candidate passes, keep V1 and document what failed.
- If a V2 candidate passes, label it a research candidate pending a **new untouched** evaluation.
- Do not integrate V2 into the demo champion merely because its calibration AP is numerically best.
- Acquire an out-of-time or lender-owned final evaluation set before any strong promotion claim.

### Step 6: integrate only after evidence supports it

If V2 eventually earns promotion, version a new runtime bundle and extend API/dashboard contracts
with:

- evidence freshness and horizons;
- recent-versus-lifetime trends;
- repayment streak/cure/relapse evidence;
- active obligation and concurrency evidence;
- cycling-proxy contributing families with cautious language;
- source quality, availability, disagreement, and out-of-distribution indicators;
- model and feature-store versions.

Historical events must remain immutable in what-if mode. Only declared current terms may be edited.
Keep V1 available for rollback and comparison.

## 14. Definition of done for the next milestone

The temporal-improvement milestone is done only when all of the following are true:

- V2 feature store still passes every data-quality and leakage gate;
- the entire predeclared campaign completes reproducibly;
- saved metrics independently recompute from saved IDs, targets, and probabilities;
- interrupted runs can resume safely or the complete run is otherwise durable;
- paired uncertainty and calibration are reported, not just point AP;
- all relevant history-coverage and audit segments are checked;
- no result uses the spent V1 holdout for selection;
- the selected outcome is either “no promotion” or “research candidate pending new test data”;
- an executed notebook and status document explain the evidence and limitations;
- all tests, lint, artifact checks, API imports, and local demo smoke tests pass;
- README/model-card wording is updated without overstating production readiness;
- the user reviews the complete working tree before the first commit.

## 15. Best supporting documents

Read these for deeper detail instead of duplicating or guessing:

- `README.md`: runnable project overview and current V1 evidence;
- `PRODUCT.md`: product scope, language, audience, and design commitments;
- `DESIGN.md`: complete dashboard visual/interaction system;
- `docs/ARCHITECTURE.md`: component boundaries;
- `docs/DATA_PLAN.md`: source grains, joins, leakage, quality, and data policy;
- `docs/MODEL_PLAN.md`: modeling and validation design;
- `docs/INPUT_SIGNAL_STRATEGY.md`: factor priority, temporal weighting, cycling logic, and future data;
- `docs/FINISH_ROADMAP.md`: broad delivery roadmap;
- `docs/MODEL_CARD.md`: frozen V1 intended use, evidence, gaps, fairness, and limitations;
- `docs/API.md`: trained endpoint contract;
- `docs/DEMO_RUNBOOK.md`: five-minute demo and mandatory wording;
- `docs/PHASE3_HISTORY_STATUS.md` through `docs/PHASE6_FINAL_VALIDATION_STATUS.md`: evidence trail;
- `docs/MODEL_FACTOR_AUDIT_VALIDATION.md`: how to interpret the full factor audit;
- `artifacts/factor_audit/README.md`: factor-audit column guide;
- `configs/temporal_challenger_experiment.json`: V2 experiment constitution.

## 16. Suggested first prompt when handing this to Claude

```text
Read codex.md end to end, then inspect the current Git state without changing or deleting
anything. Verify the stated V1/V2 boundary, run the full test and lint baseline, and report any
drift. Do not touch or evaluate against the spent final holdout. Next, make the temporal challenger
runner checkpointable and safely resumable without changing its predeclared experiment, then run
the full calibration-only campaign and document the evidence honestly.
```
