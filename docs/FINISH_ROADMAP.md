# CrediFast finish roadmap

## Delivery snapshot - 14 August 2026

- Phases A-C are implemented: one frozen-model runtime now powers trained-model API endpoints and the reviewer dashboard.
- Phase D is substantially complete: data-gap regression behavior is tested and the explanation-stability probe passed with 98.0% top-reason agreement on 250 local perturbations.
- Phase E is in release verification: desktop/mobile captures, operational commands, API documentation, and the demo narrative are present.
- Remaining production work is intentionally outside the hackathon scope: external validation data, lender-specific outcomes, compliance approval, security hardening, authentication, and monitoring infrastructure.
- The next research cycle is specified separately in `INPUT_SIGNAL_STRATEGY.md`: recency-weighted repayment, trajectories, active-loan overlap, credit-seeking velocity, and a guarded credit-cycling stress proxy, all requiring a new untouched evaluation set.

## Finish objective

Deliver an offline, judge-ready decision-support product around the frozen CrediFast model. A reviewer must be able to select a real local applicant, change a controlled set of financial inputs, score the application, understand risk and affordability, inspect data gaps and explanations, and review model evidence without leaving the application.

Model development is frozen. Any future accuracy work requires a new untouched dataset rather than reuse of the final holdout.

## Phase A — Runtime integration

**Outcome:** one reusable runtime powers CLI, API, and dashboard.

- Add a local model runtime that lazily loads both model bundles, calibrator, gap policy, application data, and feature store.
- Validate required files and expose readiness diagnostics.
- Support applicant lookup by `SK_ID_CURR`.
- Permit only controlled what-if overrides: annual income, requested credit, annuity, and goods price.
- Preserve the source record and historical snapshot for auditability.
- Return model probability, component probabilities, blend weight, internal score, grade, segment, confidence, route, explanations, quality flags, artifact versions, and affordability context.
- Fail closed with actionable errors for missing artifacts, schema drift, unknown applicants, invalid overrides, and feature-store gaps.

**Exit gate:** the same applicant and overrides return deterministic results through runtime, CLI, API, and dashboard.

## Phase B — Functional API

**Outcome:** stable local integration surface for the trained model.

- `GET /health`: process health plus model readiness.
- `GET /v1/model/status`: artifact availability, frozen versions, data counts, and limitations.
- `GET /v1/model/applicants`: curated demo profiles and safe applicant summaries.
- `POST /v1/model/score`: applicant ID, optional overrides, and explanation toggle.
- Retain `/v1/score` as a clearly labelled synthetic Phase 0 compatibility endpoint.
- Add Pydantic contracts, response examples, error mapping, and API tests.

**Exit gate:** OpenAPI loads, tests cover success and failure paths, and no endpoint implies production lending approval.

## Phase C — Reviewer dashboard

**Outcome:** a five-minute, offline, end-to-end demonstration.

- Build an application workspace with preset selector, applicant lookup, and financial what-if inputs.
- Present the primary result as repayment-difficulty probability plus internal score and grade.
- Keep recommendation language to standard review, boundary review, data-limited review, affordability review, or high-risk review.
- Show source coverage, blend composition, quality flags, and confidence beside the result.
- Present adverse and favorable component reasons with explanation-scope caveats.
- Add model evidence: holdout comparison, calibration, lift/capture, and segment diagnostics.
- Add responsible-AI evidence: fairness audit, thin-file caveat, prohibited uses, and model versions.
- Provide explicit loading, missing-artifact, empty, and invalid-input states.
- Ensure desktop and narrow-screen layouts remain usable.

**Exit gate:** five curated cases can be completed offline without editing files or using the terminal.

## Phase D — Accuracy and robustness closure

**Outcome:** preserve validated accuracy and test the demonstrated failure modes.

- Do not retrain or tune against the final holdout.
- Execute explanation-stability analysis and fix only material explanation defects.
- Add regression fixtures for representative full, partial, thin, and missing-store cases.
- Test schema mismatch, missing model files, absent history rows, unknown categories, null financial inputs, and extreme but valid amounts.
- Confirm protected audit fields remain excluded.
- Confirm segment weights match the frozen policy.
- Record new model work as a future out-of-time/lender-data program.

**Exit gate:** scoring invariants and failure behavior are automated and documented; model hashes remain unchanged.

## Phase E — Demo and release

**Outcome:** reproducible submission package.

- Curate five profiles: strong/full, borderline, partial, thin-or-missing, and affordability-limited.
- Rehearse a five-minute narrative with measured claims and mandatory caveats.
- Save offline screenshots and representative JSON responses.
- Reconcile README, architecture, API, dashboard, runbook, and model card.
- Lock runtime dependencies and document one-command startup.
- Create the initial Git commit, tag the hackathon candidate, and verify from a clean checkout with local ignored artifacts restored.
- Prepare pitch slides only after the working application is complete.

**Exit gate:** a reviewer can start the system, run all five cases, inspect evidence, and recover from a simulated data-gap error using only the documented runbook.

## Priority order

1. Runtime integration.
2. API contracts.
3. Dashboard scoring workspace.
4. Five demo profiles.
5. Evidence/governance dashboard.
6. Explanation stability and integration tests.
7. Release, screenshots, and pitch.

## Explicit non-goals for the hackathon finish

- Further tuning on the final holdout.
- Autonomous approve/decline decisions.
- Production authentication, cloud-scale infrastructure, pricing, LGD/EAD, AML/KYC, or regulatory approval.
- A blank-form claim that the model can score arbitrary real applicants without the full required schema and history.
