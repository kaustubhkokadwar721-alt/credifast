# CrediFast delivery plan

## 1. Product objective

Build an explainable, calibrated credit-risk decision-support system that transforms application, affordability, bureau, and repayment-history inputs into a risk probability, internal credit score, risk grade, lending recommendation, reason codes, and manual-review route. Predictive quality, calibration, and safe handling of data gaps take precedence over latency optimization; latency remains a smoke gate for demo usability.

The project deliberately separates five responsibilities:

1. **Data validation** decides whether an application is complete and internally consistent.
2. **Risk model** estimates probability of repayment difficulty.
3. **Affordability engine** checks whether the requested facility is serviceable.
4. **Policy engine** applies lender-configured rules and selects a review route.
5. **Explanation layer** returns controlled, auditable reasons for the result.

Fraud, KYC, AML, pricing, LGD, EAD, and collections are integration boundaries, not hidden responsibilities of the first risk model.

## 2. Success definition

### Hackathon definition of done

- Reproducible repository with one-command tests.
- Documented Kaggle data acquisition and relational data contract.
- Applicant-level feature mart generated from selected source tables.
- Logistic-regression baseline and LightGBM champion.
- Untouched holdout evaluation with discrimination and calibration metrics.
- Calibrated probabilities mapped to an internal score and A-E grade.
- Applicant-level reason codes backed by model evidence.
- Explicit thin-file and low-confidence handling.
- FastAPI scoring endpoint and Streamlit reviewer experience.
- Demonstration using strong, borderline, thin-file, and high-risk profiles.
- Model card, limitations, fairness analysis, and audit fields.
- No raw Kaggle data, credentials, or personal data committed to Git.

### Initial engineering targets

| Dimension | Target |
|---|---|
| ROC-AUC | At least 0.75 on the untouched holdout |
| PR-AUC | At least 2.5 times target prevalence |
| Calibration | Better Brier score than the base-rate predictor |
| Calibration slope | 0.90-1.10 where achievable |
| Top-decile lift | At least 3 times population risk |
| Model-only latency | p95 below 300 ms |
| End-to-end demo latency | Below 2 seconds |
| Reproducibility | Same inputs and model version produce the same result |
| Auditability | Every decision records data, model, and policy versions |

Targets are acceptance goals, not promised results. Final thresholds must be evidence driven.

## 3. Phase roadmap

### Phase 0 - Repository foundation and vertical slice

**Status:** In progress in version 0.1.0.

**Objectives**

- Establish package boundaries and stable input/output contracts.
- Prove the end-to-end scoring and decision flow before adding data complexity.
- Make core logic runnable without third-party dependencies.

**Work items**

- [x] Initialize standalone Git repository.
- [x] Add Python package and optional dependency groups.
- [x] Define validated applicant input contract.
- [x] Add deterministic demo scorer and score mapping.
- [x] Add affordability and policy engine.
- [x] Add controlled reason-code catalogue.
- [x] Add CLI, example input, FastAPI adapter, and Streamlit adapter.
- [x] Add unit tests.
- [ ] Create initial Git commit after user review.

**Exit gate**

- Example applicant scores successfully.
- All core unit tests pass.
- API and UI import successfully when optional dependencies are installed.
- Output explicitly identifies the baseline as non-production.

### Phase 1 - Data acquisition, inventory, and governance

**Status:** Tooling complete; real-data execution is waiting for Kaggle authentication and competition-rule acceptance.

**Objectives**

- Acquire the Home Credit Default Risk dataset reproducibly.
- Inventory schemas, files, grains, keys, targets, and terms of use.
- Establish raw-data immutability and dataset versioning.

**Work items**

1. Configure Kaggle credentials outside the repository.
2. Accept competition terms and download the archive to `data/raw`.
3. Record file names, bytes, hashes, row counts, and column counts in a manifest.
4. Read the official column-description file using the documented encoding.
5. Confirm table grains and relationship keys.
6. Profile target prevalence and source-table coverage.
7. Create a source-to-canonical-field mapping.
8. Document prohibited, audit-only, policy-only, and candidate predictor fields.
9. Add data validation tests for schemas and keys.

**Deliverables**

- `data/raw/` local files, ignored by Git.
- `artifacts/data_manifest.json` without personal rows.
- `src/credifast/data/ingest.py`.
- `src/credifast/data/contracts.py`.
- Data-profile notebook or reproducible script.
- Dataset limitations and usage notes.

**Exit gate**

- Raw manifest is reproducible.
- Expected files and columns are present.
- Application key uniqueness and relationship coverage are quantified.
- Target meaning and modeling population are documented.
- No raw data is committed.

### Phase 2 - Data quality and applicant feature mart

**Status:** Complete. Application and relational historical feature stores pass their automated data-quality gates.

**Objectives**

- Transform approximately 58.5 million relational records into one row per applicant.
- Prevent leakage and time-travel features.
- Produce a reusable feature mart with stable definitions.

**Work streams**

#### 2A. Application features

- Convert relative-day columns into durations.
- Replace known sentinel values with null plus anomaly indicators.
- Calculate affordability ratios.
- Normalize categories and create explicit unknown levels.
- Remove identifiers from candidate predictors.

#### 2B. Bureau features

- Aggregate account counts, status, debt, limits, overdue amounts, duration, and recency.
- Add active/closed/bad-account proportions.
- Aggregate monthly status histories into delinquency frequency, severity, recency, and streaks.

#### 2C. Previous-application features

- Calculate application frequency, approval/refusal rates, amount differences, recency, and product mix.
- Exclude fields that would not have been available at the current decision point.

#### 2D. Installment features

- Derive lateness, underpayment, payment ratio, streak, volatility, and recent-window measures.
- Verify date signs and handling of early payments.

#### 2E. Credit-card and POS features

- Create utilization, payment coverage, balance trend, cash-advance, delinquency, and completion features.

#### 2F. Cross-source features

- Debt-to-income and obligation-to-income ratios.
- Internal-versus-bureau history consistency.
- Recent-versus-lifetime behavior differences.
- Source availability, missingness, and thin-file indicators.

**Technical approach**

- Use DuckDB and/or Polars for source scanning and grouped aggregation.
- Persist intermediate aggregates as Parquet.
- Use configuration-driven feature naming and recency windows.
- Keep target-free feature generation separate from model fitting.
- Validate one final row per `SK_ID_CURR`.

**Exit gate**

- Feature mart has one row per training applicant.
- No duplicate applicant keys.
- Join expansion and orphan rates are documented.
- Nulls, infinities, invalid ranges, and sentinels are handled deterministically.
- Leakage review is signed off.
- Feature-generation tests pass on a small fixture.

### Phase 3 - Baseline models and experimental contract

**Status:** Complete. Logistic, application-only LightGBM, and history-enriched LightGBM challengers are compared on calibration data; final holdout remains sealed.

**Objectives**

- Establish credible reference performance.
- Freeze split, metrics, and experiment logging before tuning.

**Data split**

- 70% development.
- 15% calibration and policy tuning.
- 15% untouched final holdout.
- Stratify all partitions by target.
- Use five-fold cross-validation inside the development partition.
- Persist split assignments by applicant ID.

**Models**

1. Base-rate predictor.
2. Regularized logistic regression.
3. Logistic regression with Weight of Evidence as a governance challenger.
4. Application-only tree model as a thin-file baseline.

**Metrics**

- ROC-AUC and Gini.
- Average precision and PR curve.
- Brier score and log loss.
- Calibration intercept and slope.
- Lift and capture by risk decile.
- Recall and precision at review-capacity thresholds.
- Metrics by selected audit groups.

**Exit gate**

- Split assignments and metric calculations are reproducible.
- Baselines outperform the base-rate model on meaningful metrics.
- Error analysis identifies false-negative and false-positive patterns.
- No holdout-based feature selection or hyperparameter tuning occurs.

### Phase 4 - Champion model, calibration, and thin-file strategy

**Status:** Complete for the research build. The frozen history LightGBM, identity calibration, confidence routing, and partial-history blend passed all predeclared gates on the one-time random holdout. Thin-file calibration limitations remain routed to manual review.

**Objectives**

- Train the strongest responsible tabular model.
- Produce reliable probabilities rather than rankings alone.
- Route applicants according to available information.

**Work items**

- Train LightGBM champion with early stopping.
- Train CatBoost challenger.
- Compare full-file and application-only feature sets.
- Run missing-source stress tests.
- Tune with cross-validation; reserve calibration partition.
- Compare Platt and isotonic calibration.
- Choose calibration using Brier score, reliability curves, and stability.
- Add out-of-distribution and model-disagreement indicators.
- Freeze model, feature schema, preprocessing, and calibration artifacts together.
- Select segment blends on calibration-fit rows and promote them only through a disjoint evaluation gate.
- Preserve missing numeric history as missing; use zero only for explicit availability flags.

**Exit gate**

- Champion materially beats baseline without unacceptable calibration or fairness regression.
- Full-file and thin-file scoring paths are explicit.
- Predictions remain deterministic under frozen versions.
- Missing-source behavior is tested.

### Phase 5 - Explainability, fairness, and model governance

**Status:** In progress. Controlled local explanations, a research fairness audit, gap diagnostics, and a model card are present. Explanation-stability execution, mitigation decisions, and independent review remain.

**Objectives**

- Make every result reviewable and auditable.
- Identify performance disparities and proxy dependence.

**Work items**

- Global feature importance and SHAP analysis.
- Local reason-code generation from validated contributions.
- Map technical features to controlled customer/reviewer language.
- Separate adverse and favorable drivers.
- Run sensitive/proxy feature ablations.
- Measure group calibration, error rates, approval routing, and review rates.
- Add sample-size and uncertainty context to group metrics.
- Test explanations for stability under small input perturbations.
- Produce model card and limitation register.

**Exit gate**

- No sensitive characteristic appears in customer-facing reasons.
- Explanations trace back to model evidence and version.
- Material fairness gaps are documented and mitigated or escalated.
- Model card is complete.

### Phase 6 - Decision policy, affordability, and simulations

**Objectives**

- Convert calibrated risk into an operational recommendation without confusing policy with ML.

**Work items**

- Configure A-E risk bands from calibrated outcomes.
- Add eligibility and data-completeness rules.
- Add debt-service affordability calculation.
- Add manual-review capacity scenarios.
- Add expected-loss scenarios using explicitly synthetic LGD/EAD assumptions.
- Add actionable what-if simulation for amount, tenor, income verification, obligations, and utilization.
- Version policy separately from model.
- Add override reasons and reviewer audit records.

**Exit gate**

- Every decision records model version, policy version, and critical inputs.
- Policy changes do not require retraining the model.
- What-if outputs are labelled non-causal and non-binding.

### Phase 7 - Product application and integrations

**Objectives**

- Deliver a fast reviewer experience and stable integration surface.

**API features**

- Single application score.
- Batch score.
- What-if simulation.
- Explanation retrieval.
- Model-card endpoint.
- Health/readiness endpoints.

**UI features**

- Guided application form.
- Preset strong, borderline, thin-file, and high-risk profiles.
- Score, grade, probability, affordability, reasons, and confidence.
- Reviewer queue and override capture.
- Model performance, calibration, fairness, and drift page.

**Exit gate**

- API schema tests pass.
- End-to-end demo completes within two seconds on target hardware.
- Errors are safe, actionable, and do not expose sensitive data.

### Phase 8 - MLOps, security, and monitoring

**Objectives**

- Make the prototype reproducible and production-shaped.

**Work items**

- Dataset, feature, model, calibration, and policy versioning.
- Container build and dependency locking.
- Structured logging without raw personal fields.
- Authentication/authorization boundary design.
- Secrets handling and environment separation.
- Unit, integration, data-contract, and regression tests.
- Feature, score, missingness, and outcome drift monitoring.
- Model performance and calibration monitoring after outcome maturity.
- Rollback and champion/challenger plan.

**Exit gate**

- Container and tests build from a clean checkout.
- No secrets or raw personal data appear in the repository or logs.
- Rollback artifacts and runbook are documented.

### Phase 9 - Hackathon validation and pitch

**Objectives**

- Demonstrate product value honestly and repeatably.

**Demo cases**

1. Strong full-file applicant: fast approval route.
2. Borderline applicant: manual review with clear reasons.
3. Thin-file applicant: fallback model and confidence warning.
4. High-risk applicant: adverse drivers and responsible review route.
5. Affordability failure: lower-amount simulation without changing risk facts.

**Pitch evidence**

- Dataset scale and relational complexity.
- Model performance and calibration.
- Processing time versus manual workflow scenario.
- Explainability and reason codes.
- Fairness and governance controls.
- Architecture and production roadmap.

**Exit gate**

- Five-minute demo rehearsed offline.
- Backup screenshots and recorded outputs available.
- Claims distinguish measured results from assumptions.
- Raw Kaggle data never appears in distributed demo assets.

### Phase 10 - Production adaptation after the hackathon

The competition model must not be promoted directly into lending. Production work requires:

- Lender-owned target: e.g. 90+ days past due within 12 months.
- Local, representative, consented development data.
- Reject-inference and selection-bias strategy.
- True out-of-time and out-of-population validation.
- LGD, EAD, pricing, and capital models where applicable.
- Independent model validation and approval.
- Security, privacy, compliance, and legal review.
- Controlled pilot with human decisions and conservative limits.
- Outcome monitoring and scheduled recalibration/redevelopment.

## 4. Delivery sequence for the current build

The immediate next work should follow this order:

1. Verify Phase 0 tests and CLI.
2. Review and commit the repository baseline.
3. Configure Kaggle access locally.
4. Implement a manifest-only data acquisition command.
5. Implement application-table ingestion and profiling.
6. Create deterministic train/calibration/holdout assignments.
7. Train the first logistic and LightGBM application-only baselines.
8. Add bureau and previous-application aggregates.
9. Measure incremental value before adding the largest monthly tables.

This sequence produces measurable model progress early and avoids spending the entire hackathon building a large data pipeline before a first validated model exists.

## 5. Ownership model

| Role | Primary ownership |
|---|---|
| Data/feature engineer | Ingestion, keys, aggregations, feature mart, data tests |
| ML engineer | Splits, models, calibration, evaluation, explanations |
| Backend/MLOps engineer | API, artifacts, versioning, container, monitoring |
| Product/frontend lead | Reviewer UX, demo profiles, narrative, documentation |

Leakage review, fairness review, and demo claims are shared responsibilities.

## 6. Risk register

| Risk | Impact | Primary mitigation |
|---|---|---|
| Selection bias from observed loans | Model may not represent all applicants | State limitation; local pilot data and reject-inference analysis later |
| Geographic and economic domain shift | Kaggle results may not transfer to India | Treat as prototype; retrain and validate on lender population |
| Target mismatch | Repayment difficulty is not the final production definition | Replace target before production and revalidate |
| Leakage | Inflated performance | Availability inventory, fold-safe transforms, suspicious-feature review |
| Class imbalance | Accuracy masks default errors | PR-AUC, lift, cost-sensitive thresholds, calibration |
| Poor calibration | Percentages and grades mislead | Dedicated calibration data and reliability monitoring |
| Sensitive proxies | Unequal decisions | Exclusion, ablations, segment metrics, human governance |
| Thin files | Unsupported confident predictions | Separate model, confidence gate, manual review |
| Dataset terms | Redistribution risk | Review terms; commit code/manifests, not raw data |
| Compute/time overrun | No integrated demo | Incremental tables, cached Parquet, vertical slices |
| Pricing overclaim | Misleading commercial result | Keep pricing out of MVP; label financial scenarios |

## 7. Decision log

| Decision | Rationale |
|---|---|
| Home Credit Default Risk is the primary dataset | Best balance of depth, scale, relevance, and hackathon feasibility |
| LightGBM is the planned champion | Strong tabular performance and efficient SHAP explanations |
| Logistic regression remains a challenger | Governance, interpretability, and fallback value |
| No SMOTE by default | Synthetic oversampling can distort probability calibration |
| ML and policy are separate | Thresholds and affordability can change without retraining |
| Full-file and thin-file paths are distinct | Missing histories require different confidence and behavior |
| Phase 0 uses a deterministic scorer | Validates contracts before data/model work; never represented as trained ML |
