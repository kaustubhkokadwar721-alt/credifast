# Data plan

## Selected source

Primary source: Kaggle Home Credit Default Risk.

The modeling population is the labelled `application_train` table. Historical tables are aggregated using `SK_ID_CURR`, `SK_ID_BUREAU`, and `SK_ID_PREV` into one feature row per current application.

## Source inventory

| Source | Grain | Key | Planned feature families |
|---|---|---|---|
| Application | One current loan application | `SK_ID_CURR` | Amounts, income, annuity, household, employment, external scores, missingness |
| Bureau | One previous external credit | `SK_ID_BUREAU` and `SK_ID_CURR` | Debt, limits, status, duration, overdue, recency, credit type |
| Bureau balance | One month per bureau credit | `SK_ID_BUREAU`, relative month | Delinquency frequency, severity, streak, recency, trend |
| Previous application | One historical Home Credit application | `SK_ID_PREV`, `SK_ID_CURR` | Approval/refusal, amounts, recency, channel, product mix |
| Installments | One scheduled/actual historical payment | `SK_ID_PREV`, installment identifiers | Late days, shortfall, payment ratio, streak, volatility |
| Credit-card balance | One month per historical card | `SK_ID_PREV`, relative month | Utilization, payment coverage, balances, drawings, delinquency |
| POS/cash balance | One month per historical loan | `SK_ID_PREV`, relative month | Days past due, installments remaining, completion, status trend |

## Canonical depth model

- **Depth 0:** current application.
- **Depth 1:** previous accounts and applications.
- **Depth 2:** monthly or payment events within historical accounts.
- **Depth 3:** cross-account applicant aggregates, recency windows, trends, and source-coverage indicators.

## Data zones

```text
data/raw/        immutable Kaggle files
data/interim/    table-level cleaned Parquet and aggregates
data/processed/  applicant feature mart and split assignments
artifacts/       manifests, metrics, schemas, and model outputs safe for Git when small
```

## Required quality checks

1. File manifest and checksum.
2. Expected schema and type compatibility.
3. Candidate-key uniqueness at expected grain.
4. Exact and near-duplicate rates.
5. Foreign-key coverage and orphan counts.
6. Row counts before and after joins.
7. Null, empty-string, sentinel, and infinity rates.
8. Numeric and cross-field validity rules.
9. Category normalization and unseen-category behavior.
10. Time-availability and future-information checks.
11. Class distribution and source coverage by target.
12. Distribution and missingness comparison across data splits.

## Leakage rules

- No outcome or post-current-decision fields.
- No target-derived aggregates.
- No current-loan payment information.
- All learned imputers, encoders, and selectors fit only on development folds.
- Target encoding, if used, is out-of-fold.
- Applicant IDs never become predictors.
- Any feature with implausibly high standalone discrimination is investigated.

## Feature windows

Historical sources should expose lifetime and recent-window features where their relative month fields support it:

- Last 3 months.
- Last 6 months.
- Last 12 months.
- Last 24 months.
- Lifetime.

Recency windows must be relative to the current application and must never include future records.

## Sensitive and proxy policy

- Sensitive characteristics are excluded from the production candidate model and retained only in a protected audit dataset where permitted.
- Policy-only fields are evaluated outside ML.
- Proxy-prone fields receive ablation and group-impact review.
- Missingness indicators are assessed for disparate impact.

## Compute strategy

- Scan CSVs with DuckDB or Polars rather than loading every file into Pandas simultaneously.
- Materialize each historical table's applicant aggregates independently.
- Cache Parquet outputs keyed by source hash and feature-config version.
- Join only applicant-level aggregates into the final feature mart.
- Begin with application, bureau, previous applications, and installments; add the largest monthly tables after incremental-value measurement.
