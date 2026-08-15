# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- Primary: credit reviewer or lending-operations analyst assessing an application.
- Demonstration audience: Cognizant NPN Hackathon judges evaluating usefulness, technical depth, responsible AI, and completeness.
- Secondary: model-risk, compliance, and engineering reviewers inspecting evidence and limitations.

## Product Purpose

CrediFast turns current application attributes and fragmented credit histories into an explainable repayment-difficulty assessment, affordability context, data-confidence classification, and human-review route. Hackathon success means a reviewer can evaluate representative cases offline, understand why the result was produced, see when information is insufficient, and inspect the evidence behind the model.

## Positioning

CrediFast treats missing credit history as a first-class decision-support condition. It combines application and history models only where validation supports the blend, preserves unknown values as missing, exposes source coverage and component weights, and routes uncertain cases to manual review instead of presenting false precision.

## Operating Context

- Local, offline-first hackathon demonstration using the Home Credit competition dataset and frozen local artifacts.
- A reviewer selects a curated or known applicant record, optionally changes a small set of financial what-if inputs, and receives a versioned assessment.
- Reviewers can load those current terms from a compact JSON package, while source badges distinguish applicant, lender, CIBIL-report-derived, partner-ledger, and computed evidence.
- Completely new real applications are outside the honest scope of this trained prototype because the model expects a broad application schema and historical feature store.
- The dashboard must remain usable during a live five-minute demonstration without Kaggle or network access.

## Capabilities and Constraints

- Frozen history-enriched LightGBM, application-only LightGBM component, identity calibration, and segment-specific gap policy.
- Functional applicant lookup, safe financial overrides, model scoring, reason codes, confidence routing, affordability context, final validation evidence, and governance views.
- Versioned input-provenance schema covering all 257 selected model factors; live CIBIL and banking connectors remain explicit future integrations.
- Protected audit fields are excluded from prediction.
- Thin and partial histories require data-limited manual review.
- Outputs are research-only and cannot be described as autonomous approval, official bureau scoring, pricing, or compliant adverse action.
- Final random holdout is spent and cannot be used for additional tuning.
- Latency is a usability smoke check, not an optimization objective.

## Brand Commitments

- Product name: CrediFast.
- Voice: direct, evidence-led, calm, precise, and explicit about uncertainty.
- Required terms: “repayment-difficulty probability,” “internal research score,” “decision support,” and “manual review.”
- Avoid celebratory approval language, fear-based high-risk language, and decorative banking imagery.

## Evidence on Hand

- Final holdout evaluation: `artifacts/final_holdout_evaluation.json`.
- Model card: `docs/MODEL_CARD.md`.
- Fairness audit: `artifacts/history_fairness_audit.json`.
- Gap strategy: `artifacts/history_gap_strategy_metrics.json` and `configs/model_gap_strategy.json`.
- Input acquisition and source mapping: `docs/INPUT_ACQUISITION.md` and `src/credifast/input_sources.py`.
- Local frozen model binaries under ignored `artifacts/models/`.
- Local application and feature-store data under ignored `data/raw/` and `data/processed/`.
- No lender-owned outcome data, customer testimonials, production deployment evidence, or regulatory approval exists; future work must not fabricate them.

## Product Principles

1. Show evidence with every risk result.
2. Make missing information visible and operationally actionable.
3. Keep prediction, affordability, and policy responsibilities separate.
4. Prefer a complete, reproducible reviewer workflow over another model experiment.
5. Never overstate what competition data can establish.

## Accessibility & Inclusion

- Keyboard-operable inputs and navigation.
- Do not rely on color alone for risk, confidence, or status.
- Use readable labels, explicit units, sufficient contrast, and plain-language recovery guidance.
- Sensitive fields appear only in aggregate audit evidence, never in scoring inputs or individual explanations.
