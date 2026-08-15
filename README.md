# CrediFast

CrediFast is an explainable credit-risk decision-support prototype for Cognizant NPN Hackathon line item 7: **Credit Rating Checks for Faster Lending**.

The platform is designed to combine current application data, affordability, credit-bureau history, previous applications, and repayment behavior to produce:

- a calibrated repayment-difficulty probability;
- an internal credit score and A-E risk grade;
- an affordability assessment;
- a standard, boundary, data-limited, high-risk, or affordability review route;
- controlled adverse and favorable reason codes;
- data-coverage and model-confidence indicators.

The repository now contains the deterministic vertical slice plus a trained, calibrated, history-enriched research model and a gap-aware scoring CLI. The original deterministic scorer remains available for contract and policy demonstrations. Neither path is a production lending model.

## Current status

| Area | Status |
|---|---|
| Repository and project contracts | Complete |
| Deterministic demo scorer | Complete |
| Decision policy and reason codes | Complete |
| CLI and example applicant | Complete |
| Shared frozen-model runtime | Complete: local artifacts, controlled what-if inputs, gap handling |
| Functional FastAPI trained-model endpoints | Complete |
| Reviewer dashboard and evidence views | Complete: desktop and narrow-screen verified |
| Unit tests | Complete |
| Dataset contracts, manifesting, and application profiling | Complete |
| Kaggle data acquisition and immutable real-data manifest | Complete |
| Application-only logistic baseline | Complete |
| Application-only LightGBM challenger | Complete |
| Relational history feature store and quality gate | Complete |
| History-enriched LightGBM challenger | Complete |
| Champion probability calibration | Complete: identity selected honestly |
| Thin-file confidence routing | Complete: research policy |
| Segment-specific application/history gap strategy | Complete: partial-history blend promoted for holdout evaluation |
| Missing feature-store row handling | Complete: explicit quality flag and data-limited route |
| Reason-code integration, stability probe, and fairness audit | Complete: research audit with caveats |
| Sealed holdout evaluation | Complete: all predeclared technical quality gates passed |

The finish plan and current handoff are in [docs/FINISH_ROADMAP.md](docs/FINISH_ROADMAP.md). The next-model input, recency, repayment-pattern, and credit-cycling strategy is in [docs/INPUT_SIGNAL_STRATEGY.md](docs/INPUT_SIGNAL_STRATEGY.md). The complete frozen-model factor dictionary and validation notes are in [artifacts/factor_audit/README.md](artifacts/factor_audit/README.md) and [docs/MODEL_FACTOR_AUDIT_VALIDATION.md](docs/MODEL_FACTOR_AUDIT_VALIDATION.md). Data and modeling details live in [docs/DATA_PLAN.md](docs/DATA_PLAN.md) and [docs/MODEL_PLAN.md](docs/MODEL_PLAN.md).

## Start the complete local demo

After bootstrapping and restoring the ignored local data/model artifacts, start the API and reviewer dashboard together:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1
```

Open `http://127.0.0.1:8501` for the dashboard and `http://127.0.0.1:8000/docs` for OpenAPI. The demo runs offline after local artifacts exist.

The review workbench supports a curated case, any known local application ID, or a compact JSON
intake package. Source badges distinguish applicant declarations, lender-calculated terms,
bank/Account-Aggregator-verifiable income, and CIBIL-reported-but-confirmable income. The other
model factors are assembled from local stores or computed; reviewers do not type them manually.
See [docs/INPUT_ACQUISITION.md](docs/INPUT_ACQUISITION.md) for the 257-factor source map and live
integration boundaries.

## Quick start: core demo

The core scorer has no third-party runtime dependencies.

```powershell
.\scripts\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
credifast examples/applicant.json
python -m unittest discover -s tests -v
```

Alternatively, without installing the package:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m credifast examples/applicant.json
```

## Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn credifast.api:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

## Run the UI

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

If local PowerShell script execution is restricted, run the bootstrap explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

## Kaggle data

The selected primary dataset is Home Credit Default Risk. Do not commit raw Kaggle files to Git.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_data.ps1
```

The download script refuses to overwrite existing raw CSV files. It downloads and extracts the competition data, then creates `artifacts/data_manifest.json` and `artifacts/application_profile.json`.

Access requires a Kaggle account, API credentials, and acceptance of the competition rules. See [docs/DATA_PLAN.md](docs/DATA_PLAN.md) before downloading or transforming the data.

Phase 1 utilities can also be run independently:

```powershell
credifast-data contracts
credifast-data manifest --raw-dir data/raw --output artifacts/data_manifest.json
credifast-data profile-application --input data/raw/application_train.csv
```

## Train the application-only models

The split assignment is deterministic: 70% development, 15% calibration, and 15% sealed holdout.

```powershell
credifast-train application-baseline
credifast-train application-lightgbm
```

Measured results and limitations are documented in [docs/PHASE2_BASELINE_STATUS.md](docs/PHASE2_BASELINE_STATUS.md). The trained binaries are local and Git-ignored; non-sensitive metrics and feature inventories are versioned under `artifacts/`.

## Build and train the history-enriched model

The history pipeline aggregates every child table before joining the application grain. It never selects `TARGET`, and its automated gate checks row reconciliation, key uniqueness, bounded rates, nonnegative counts, finite values, and source coverage.

```powershell
credifast-data aggregate-history
credifast-data validate-history
credifast-train history-lightgbm
credifast-train history-calibrate
python scripts/create_history_quality_notebook.py
```

The current history-enriched challenger reaches 0.7814 ROC-AUC and 0.2739 average precision on the calibration partition. See [docs/PHASE3_HISTORY_STATUS.md](docs/PHASE3_HISTORY_STATUS.md) for the complete comparison and limitations.

An honest calibration sub-experiment selects identity scaling: the raw probabilities already outperform Platt and isotonic calibration on held-back calibration rows. Thin and partial histories are routed to research-only manual review. See [docs/PHASE4_CALIBRATION_STATUS.md](docs/PHASE4_CALIBRATION_STATUS.md).

## Evaluate and use the data-gap strategy

Segment blend weights are selected on `calibrator_fit` rows and assessed on disjoint `calibrator_evaluation` rows. A conservative promotion gate retains the history champion unless probability and ranking evidence supports the blend.

```powershell
credifast-train history-gap-strategy
credifast-score --limit 100 --no-explanations
```

The frozen candidate uses 100% history probability for full and thin histories, and a 75% history / 25% application blend for partial histories. Missing feature-store rows retain numeric history as missing, set all source-availability flags to zero, emit `FEATURE_STORE_ROW_MISSING`, and force data-limited review. See [docs/PHASE5_GAP_STRATEGY_STATUS.md](docs/PHASE5_GAP_STRATEGY_STATUS.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Final frozen-candidate validation

The candidate was evaluated once on the previously sealed 46,127-row random holdout. It passed every predeclared technical gate: ROC-AUC 0.7849, average precision 0.2799 (3.47 times prevalence), Brier 0.06607 versus 0.07422 for the base-rate predictor, calibration slope 1.023, and top-20% event capture 56.50%. The evaluator now refuses to overwrite the final artifact.

Thin-file calibration remains a material limitation despite good ranking, so zero-to-three-source cases stay on the data-limited manual-review route. See [docs/PHASE6_FINAL_VALIDATION_STATUS.md](docs/PHASE6_FINAL_VALIDATION_STATUS.md).

## Diagnose ranking and model factors

The average-precision diagnostic reads the saved holdout predictions and reports precision/recall tradeoffs without fitting or tuning anything:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_average_precision.py
```

The complete factor audit calculates exact TreeSHAP contributions across all 46,127 frozen holdout rows, aggregates 465 transformed columns to 257 raw factors, and screens numeric and categorical redundancy:

```powershell
.\.venv\Scripts\python.exe scripts\audit_model_factors.py
```

The output directory is `artifacts/factor_audit/`. `model_factor_audit.csv` contains every factor's definition, preprocessing, intuitive direction, observed model direction and magnitude, missingness effect, strongest correlations, and substitutability warning. These are diagnostic explanations of model behavior, not causal effects or production adverse-action reasons.

A 250-row local robustness probe also perturbed the four editable financial amounts by 0.5%. The top adverse reason remained unchanged for 98.0% of rows, mean top-three Jaccard similarity was 0.990, and the 95th-percentile absolute probability change was 0.00716. This is a useful local check, not proof of global explanation robustness.

## Important limitation

This is a hackathon decision-support prototype trained or demonstrated on competition data. It is not approved for autonomous real-world lending, pricing, adverse-action, or regulatory use. Any production adaptation requires lender-owned outcome definitions, local data, independent model validation, fairness review, security review, and legal/compliance approval.
