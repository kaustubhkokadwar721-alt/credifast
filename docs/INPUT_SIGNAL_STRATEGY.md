# Input signal and temporal modeling strategy

## Decision

The next CrediFast model should make repayment history **time-aware and sequence-aware**, while keeping the frozen 0.2 candidate unchanged. Recent behavior should usually count more than old behavior, but permanent severity anchors must remain so an old default is not erased by decay.

The recommended next challenger is still a high-quality tabular model first: LightGBM/CatBoost over carefully engineered temporal, loan-level, and customer-level signals. A sequence encoder can be tested as a separate challenger and blended only if it improves out-of-time ranking, calibration, stability, and data-gap behavior.

“A new loan funded an old EMI” must not be presented as a known fact with this dataset. Home Credit does not contain bank-account fund flows or payment-purpose linkage. We can build a **credit-cycling stress proxy** from timing, overlapping obligations, cash borrowing, ATM drawings, and payment shortfalls. It is a risk signal for review, not a causal accusation.

## What the current model actually uses

The frozen model combines current application fields with 150 aggregated history features from bureau, bureau-balance, previous-application, POS/CASH, credit-card, and installment tables. It then uses a gap policy: full and thin histories retain the history model; partial histories use a frozen 75% history / 25% application blend; missing-store rows fail closed to data-limited review.

Gain importance is descriptive, not causal, and correlated features share importance. It nevertheless shows where the current candidate obtains most of its split gain:

| Current feature family | Share of model gain | Current support |
|---|---:|---|
| External scores | 37.30% | Mean/min/max/std and raw external score fields; strong but opaque and dataset-specific. |
| Application and derived fields | 23.10% | Income, credit, annuity, goods price, employment, application missingness, and affordability ratios. |
| Previous loans/applications | 10.45% | Counts, approval/refusal, amounts, yield group, loan type, and 12/24-month activity counts. |
| Installment repayment | 10.38% | Lifetime payment ratio, late/short counts and rates, late days, shortfalls, and 12/24-month late counts. |
| Bureau credit | 8.36% | Active/closed counts, debt, overdue amounts, recency, annuity, and debt-to-credit ratios. |
| POS/CASH history | 5.62% | Future installments, DPD severity, active/completed months, and recent 12-month DPD. |
| Credit card | 4.15% | Utilization, drawing, payment-to-minimum, receivable, DPD, and activity. |
| Bureau monthly status | 0.59% | Delinquent/severe/default month counts and rates, including a recent 12-month count. |

The current implementation therefore has **coarse recency windows**, not true recency weighting. It has customer-lifetime summaries, not repayment trajectories, streaks, delinquency transitions, loan overlap, or event-to-event timing.

## Source quality evidence

The feature store is structurally sound at one row per applicant: 356,255 unique applications, 150 history features, no duplicate applicant keys, no invalid rate features, and no target field selected. Source-event checks also found no positive relative dates or months in installments, prior applications, POS/CASH, credit-card, or bureau data, which supports an as-of-application interpretation.

| Source | Raw rows | Applicant coverage | Important quality finding | Modeling consequence |
|---|---:|---:|---|---|
| Installments | 13,605,401 | 95.32% | 1,146,669 late rows; 1,295,493 short-payment rows; only 2,905 missing payment amounts; no future due/payment dates. | Highest-priority temporal source. Preserve multiple-payment behavior and aggregate at installment, loan, then applicant grain. |
| Previous applications | 1,670,214 | 95.12% | Unique prior-loan IDs; no future decisions; purpose is unknown/XAP/XNA for 1,600,579 rows (95.83%). | Strong for credit-seeking and loan-sequence behavior; weak for declared purpose. |
| POS/CASH monthly | 10,001,358 | 94.67% | Unique loan-month grain and no future months. | Strong for active-loan overlap, remaining-term trajectory, and DPD transitions. |
| Bureau | 1,716,428 | 85.84% | Unique bureau IDs; debt is missing on 257,669 rows and annuity on 1,226,791 rows. | Use availability/denominator flags; never turn missing debt or annuity into zero. |
| Credit card monthly | 3,840,312 | 29.07% | Unique loan-month grain; minimum-payment amount missing on 305,236 rows; customer coverage is structurally low. | Valuable conditional signal, never mandatory evidence; use a specialist block plus missing-source confidence. |

Only 1,931 previous applications (0.12%) state “Payments on other loans,” while 95.83% have no usable purpose. That field cannot carry the credit-cycling inference.

## What quality and priority mean

**Input quality** and **model priority** are separate. A predictive field can be low quality or legally risky; a high-quality field can add little incremental information.

### Quality dimensions

1. **Point-in-time validity:** the value existed before the scoring cutoff and cannot look into the outcome window.
2. **Provenance:** verified bureau/lender/transaction data ranks above self-reported or inferred data.
3. **Completeness:** coverage, null mechanism, and denominator are explicit by source and segment.
4. **Grain integrity:** event, installment, loan, and applicant grains are not mixed or duplicated by joins.
5. **Freshness:** source age is known; stale evidence is not treated as current evidence.
6. **Semantic validity:** the field means what its name suggests and is not a weak proxy presented as a fact.
7. **Stability:** definitions, distributions, units, and categories remain stable across time and lender systems.
8. **Bias and compliance risk:** protected attributes and unjustified proxies are excluded from prediction and retained only for controlled audits.
9. **Explainability:** the signal can be described truthfully to a reviewer without causal overclaiming.
10. **Failure behavior:** missing, delayed, conflicting, or out-of-range values trigger flags and confidence changes rather than silent imputation.

### Priority rule

Prioritize a signal when it has strong business relevance, point-in-time validity, coverage, incremental out-of-time value, stability, and explainability. Penalize leakage risk, source ambiguity, bias risk, brittleness, and operational cost. Feature importance alone never determines priority.

| Priority | Meaning | Release rule |
|---|---|---|
| P0 | Core signal required for a trustworthy credit review. | Must have data contract, freshness, missingness behavior, ablation evidence, stability tests, and reviewer explanation. |
| P1 | Material enhancer with good value but partial coverage or greater ambiguity. | May improve a specialist/modality path; absence must not count as good behavior. |
| P2 | Experimental corroborating signal. | Cannot independently trigger an adverse route; promote only after out-of-time and fairness evidence. |
| Exclude | Leakage, protected attribute, identifier, weak unreviewable proxy, or unavailable-at-score data. | Audit-only or removed. |

## Proposed signal portfolio

| Signal block | Priority | Current model | Next implementation |
|---|---|---|---|
| Recency-weighted repayment timeliness | P0 | Partial: lifetime late rate/days plus 12/24-month counts. | Exponential weighted late rate, weighted late days, shortfall ratio, and on-time ratio with multiple half-lives. |
| Repayment trajectory and recovery | P0 | Not explicit. | 3/6/12-month slopes, recent-versus-prior deltas, consecutive late/on-time streaks, DPD roll rates, cure rate, and recovery time. |
| Current obligation and affordability burden | P0 | Partial: application ratios, bureau debt/annuity, future installment averages. | As-of active monthly obligation sum, obligation-to-verified-income, remaining principal/term, stress scenario, and source confidence. |
| Loan stacking and credit-seeking velocity | P0 | Partial: counts and recent 12/24-month prior/bureau activity. | Active-loan concurrency by month, peak/mean overlap, applications in 30/90/180 days, time between loans, new loan during high burden, and refusal-to-approval sequences. |
| Delinquency severity | P0 | Good aggregates, weak sequence. | Preserve lifetime max/default anchor; add weighted severity, severe-event recency, transition matrix, and repeated-episode count. |
| Credit-cycling stress proxy | P1 | Not explicit. | Combine new cash borrowing near installment stress, overlapping active loans, rising obligation burden, cash/ATM draw spikes, minimum-payment behavior, and explicit other-loan purpose when present. |
| Revolving-credit utilization and payment behavior | P1 | Partial lifetime aggregates; 29.07% customer coverage. | Recent utilization level/slope/volatility, payment-to-minimum trend, ATM-draw-to-payment ratio, maxed-out months, and modality-specific confidence. |
| Income and cash-flow resilience | P0 with lender data; P1 here | Self-reported current income and application ratios only. | With bank/payroll data: verified income, stability, income shocks, balance buffer, essential spending, and existing EMI cash outflow. Without it, keep a low-confidence declared-income screen. |
| External credit scores | P1 | Very strong and dominant. | Retain as a benchmark signal, add freshness/provenance, monitor dominance and portability, and train a no-external-score challenger. |
| Previous loan purpose | P2 | Not used explicitly. | Use only as a sparse corroborating flag; unknown is not “no refinancing.” Never use alone. |
| Demographics/protected attributes | Exclude from prediction | Gender and age excluded by policy; retained for fairness audit. | Keep audit-only. Evaluate proxy leakage in geography, occupation, education, device, and contact fields before use. |

## Recency weighting design

For an event `i` occurring `t_i` days before the application cutoff, use an exponential weight:

`w_i = exp(-ln(2) * t_i / h)`

where `h` is the half-life. The newest event receives the highest weight; an event exactly one half-life old receives half weight.

Do not select one half-life by intuition and hard-code it. Predeclare a small set and compare it only on development/time-validation data:

- **90 days:** acute cash-flow stress, utilization, and credit-seeking velocity.
- **180 days:** late/short repayments and emerging delinquency.
- **365 days:** repayment consistency, recovery, and positive behavior.
- **Lifetime anchor:** maximum DPD, default/bad-debt history, total severe episodes, and oldest account age.

Core features:

- `weighted_late_rate_h = sum(w_i * late_i) / sum(w_i)`;
- `weighted_shortfall_ratio_h = sum(w_i * max(due_i - paid_i, 0)) / sum(w_i * due_i)`;
- `weighted_days_late_h = sum(w_i * max(payment_day_i - due_day_i, 0)) / sum(w_i)`;
- `recent_vs_prior_late_delta = late_rate_0_90d - late_rate_91_365d`;
- `consecutive_late_max`, `current_on_time_streak`, `days_since_last_late`, and `days_to_cure`;
- per-loan versions first, then customer aggregation, so a long loan with many installments does not automatically dominate every other loan.

Positive and negative evidence should not decay identically. A recent missed payment can deteriorate risk quickly; reliable on-time behavior should improve confidence gradually. Severe default/bad-debt markers retain a lifetime anchor and are never erased by recent clean months.

## Detecting possible “loan-funded EMI” behavior

Name the feature group **credit-cycling stress**, not “loan-funded EMI.” Use an evidence ladder:

### Stronger proxies available in Home Credit

- A new cash or revolving loan starts while another loan is active and has future installments.
- New borrowing occurs within 30/60/90 days after a late or short installment.
- Active-loan concurrency and total scheduled obligations rise across recent months.
- Credit-card ATM drawings rise while payment-to-minimum falls or delinquency appears.
- Several applications/refusals occur in a compressed period before a new approval.
- The sparse “Payments on other loans” purpose is present as corroboration.

### Signals that require additional lender/bank data

- Loan disbursement credit followed by an EMI debit to another lender.
- Round-amount transfers to lenders or loan wallets shortly after disbursement.
- Salary-to-EMI timing gaps, declining cash buffer, bounced mandates, and recurring overdraft use.
- Cross-lender refinance identifiers or a verified balance-transfer product.

### Guardrails

- Require at least two independent proxy families before surfacing the pattern to a reviewer.
- Present “possible credit-cycling stress” with supporting dates and amounts, never a causal statement.
- Missing transaction or credit-card history lowers confidence; it does not indicate absence.
- Do not let the proxy independently produce a decline/adverse-action reason.
- Audit by income, age, gender, geography, and thin-file segment because cash usage and credit access differ structurally.

## Next model architecture

### Champion path: temporal feature GBDT

1. Build point-in-time event features at **installment -> loan -> applicant** grain.
2. Add multi-horizon recency, trajectory, overlap, and credit-cycling blocks.
3. Train LightGBM and CatBoost challengers on development data only.
4. Calibrate on a disjoint calibration period.
5. Use modality dropout during training and explicit source-age/availability fields so missing sources are modeled deliberately.
6. Preserve component probabilities: application, bureau, installment/POS, and revolving-credit specialists.
7. Learn or validate segment-specific blends without using the sealed 0.2 holdout.

### Research challenger: event-sequence encoder

Encode repayment events with event type, amount ratios, days late, loan age, and time gaps using a small temporal convolution/GRU/Transformer. Feed its out-of-fold embedding or probability into the tabular model. Promote only if it materially improves out-of-time average precision and Brier score, remains calibrated across history-availability segments, and produces stable reviewer-level explanations. Complexity alone is not value.

### Uncertainty and gaps

- Produce probability plus data-confidence dimensions: coverage, freshness, source agreement, and out-of-distribution score.
- Train source-specialist models and a missing-modality gate rather than relying on one universal imputation.
- Keep fail-closed human routing for thin files, stale sources, schema gaps, and store outages.
- Add confidence intervals or bootstrap stability bands for evaluation; do not imply individual probability certainty.

## Validation plan without contaminating the frozen holdout

The existing 46,127-row final holdout is sealed and cannot select any new feature, half-life, hyperparameter, blend, or threshold.

1. Create an as-of development table with strict cutoff enforcement.
2. Use grouped, temporal folds when timestamps permit; otherwise create a new untouched external/lender or competition split.
3. Run additive ablations: current features -> recency -> trajectories -> overlap/velocity -> cycling proxy -> sequence challenger.
4. Require incremental improvement in average precision, ROC-AUC, Brier score, calibration slope/intercept, top-10/20% capture, and net benefit at review capacity.
5. Require no material regression for full, partial, thin, source-missing, gender, and age-band segments.
6. Check temporal leakage, duplicate grain, late-arriving data, drift, reason stability, and missing-source perturbations.
7. Select half-lives and feature blocks on development folds; calibrate on a disjoint set; evaluate once on a new untouched test set.

Minimum promotion gates should be written before training. A useful starting standard is statistically credible improvement in average precision and Brier score over 0.2, no worse calibration, stable top-reason behavior under small perturbations, and no new fairness or thin-file regression.

## Implementation phases

### Phase 1 - point-in-time signal store

- Add a versioned V2 feature builder rather than editing the frozen V1 query.
- Define event, installment, loan, and applicant grains plus cutoff semantics.
- Add uniqueness, no-future-event, missingness, source-age, and join-expansion gates.
- Materialize loan-level intermediate Parquet files for inspectability.

### Phase 2 - P0 temporal blocks

- Implement recency-weighted repayment, delinquency transitions, recovery/streaks, active obligation, concurrency, and credit-seeking velocity.
- Validate each feature using hand-constructed event sequences and edge cases.
- Add per-feature provenance, freshness, denominator, and missingness metadata.

### Phase 3 - P1 cycling and revolving blocks

- Implement the multi-signal credit-cycling proxy and credit-card trends.
- Keep sparse purpose and low-coverage credit-card data conditional.
- Add a reviewer evidence payload listing the contributing timing patterns.

### Phase 4 - challenger experiments

- Train temporal LightGBM and CatBoost ablations first.
- Test an event-sequence encoder only after the tabular temporal baseline is sound.
- Calibrate, audit segments, test explanation stability, and freeze the candidate.

### Phase 5 - product integration

- Extend API/dashboard output with signal freshness, recency horizon, source quality, current-vs-prior trend, and credit-cycling evidence.
- Add what-if controls only for declared current terms; historical events remain immutable evidence.
- Add data-quality alerts for stale or conflicting obligations/income.

### Phase 6 - new untouched evaluation

- Acquire an out-of-time or lender-owned dataset with verified income, existing EMI obligations, and repayment outcomes.
- Lock features, calibration, blends, routes, and gates before evaluation.
- Evaluate once, document failures honestly, and retain 0.2 as the reproducible benchmark.

## Automated checks to add

- No event date after application cutoff; no future-relative day/month values.
- Unique bureau-loan and prior-loan IDs; unique POS/credit-card loan-month keys.
- Installment multi-payment rows reconciled before loan aggregation.
- Weighted rates bounded in `[0, 1]`; weights finite and monotonically decrease with age.
- Per-loan aggregation prevents row-count/term-length dominance.
- Missing source is distinct from measured zero; every ratio has denominator and availability flags.
- Active-loan concurrency and scheduled obligation totals reconcile to source contracts.
- Credit-cycling evidence requires multiple families and never independently triggers an adverse route.
- Feature freshness and distribution drift monitored by source and availability segment.
- Protected audit fields remain excluded from every prediction matrix.

## Open data gaps

Accuracy will eventually be limited less by model choice than by unavailable evidence. Highest-value additions are verified payroll/income, bank transaction cash flow, complete cross-lender open obligations and EMI schedules, mandate failures/bounces, source freshness timestamps, lender product/policy context, and a true out-of-time repayment outcome. Until those exist, the dashboard must distinguish verified, reported, inferred, stale, and unavailable evidence.
