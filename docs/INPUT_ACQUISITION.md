# Input acquisition and source marking

## The recommended intake

Do not ask a reviewer or applicant to type 257 model factors. The frozen model consumes 257
selected raw/engineered factors, but most must be assembled from systems of record or calculated.

The simplest responsible workflow is:

1. collect current applicant declarations and requested loan terms through a short form or JSON;
2. retrieve a consented CIBIL Credit Information Report through an authorized member/partner
   integration;
3. retrieve lender/servicer history from partner systems;
4. retrieve consented deposit-account data through a bank or Account Aggregator where income and
   cash-flow verification is required;
5. calculate ratios, recency, aggregates, quality flags and confidence internally;
6. show the reviewer which source supplied or verified each field and whether it is fresh,
   conflicting or unavailable.

The current hackathon runtime implements step 1 for known local applicants and simulates steps 2-5
from the local Home Credit dataset. No live CIBIL, Account Aggregator or bank connector is present.

## Frozen-model factor counts

The primary-source mapping is generated from
`artifacts/history_lightgbm_features.json` by `credifast.input_sources`.

| Primary provenance | Model-ready factors | Current family detail |
|---|---:|---|
| CIBIL report-derived | 48 | 31 bureau-account aggregates, 10 bureau-month history aggregates, 6 enquiry-frequency fields and 1 availability flag. |
| Lending-partner ledger-derived | 102 | 30 prior-application, 20 POS, 25 card, 23 installment and 4 availability factors. |
| Application and documents | 88 | Declared terms, contact/document flags, housing/property fields and other application attributes. |
| Computed automatically | 16 | Ratios, durations, external-source summaries, source counts and missingness. |
| Separate external scores | 3 | Raw Home Credit `EXT_SOURCE_1/2/3`; their provider identity is unspecified and they are not CIBIL scores. |
| **Total** | **257** | Each factor is counted once under its primary current origin. |

These are counts of model-ready factors, not counts of fields returned by an external API. Several
factors are aggregates of many raw accounts or transactions. Sources overlap: a CIBIL report can
also describe cards and loans held with partners, while a lender can hold its own copy of bureau
data. The registry assigns one primary origin to avoid double-counting.

## What CIBIL can provide

TransUnion CIBIL's official consumer-report guidance says a standard report includes personal and
contact information, lender-reported employment/income information, credit-account type and
lender, loan amount or credit limit, outstanding balance, account status, monthly payment history,
and credit enquiries. CIBIL notes that employment/income is what members previously reported; it
should not be treated as verified current income.

For this model, a structured report can primarily support:

- active, closed, bad-debt, sold and overdue account counts;
- account types, original credit, limits, balances and outstanding debt;
- account age and recent credit openings;
- monthly delinquency level, rate, recency and severity;
- recent credit enquiries;
- an explicit bureau-available flag.

This is the conservative 48-factor CIBIL-derived block. Some other lender-ledger factors might be
reconstructed from a sufficiently detailed bureau payload, but their definitions and freshness
must be reconciled before substituting one source for another.

CIBIL access is not an anonymous public API. The official API Marketplace describes JSON API
integration for partners/members and a commercial onboarding process involving registration,
documentation, UAT and production movement. Consumer authentication/consent and the permitted
purpose must be implemented under the applicable CIBIL arrangement.

Official references:

- [CIBIL report section guide](https://www.cibil.com/faq/understand-your-credit-score-and-report)
- [TransUnion CIBIL API Marketplace](https://apimarketplace.transunioncibil.com/)
- [CIBIL consent-based consumer-connect API](https://apimarketplace.transunioncibil.com/products/credit-data/dtc-consumer-connect-services)

## What a bank or Account Aggregator can provide

The RBI Account Aggregator framework requires explicit consent for retrieving or sharing customer
financial information. The customer must be able to revoke consent; an AA must not obtain the
customer's banking passwords, PINs or private keys, and accessed financial information must not
reside with the AA.

The ReBIT deposit schema exposes profile, account summary and transaction sections. Depending on
the participating Financial Information Provider, these include holder details, account type and
status, opening date, current balance, overdraft/drawing limit, and transaction amount, type,
mode, narration, timestamp, reference and transactional balance.

Those raw records can support new engineered evidence such as:

- verified salary or recurring-income detection;
- income stability, volatility and recency;
- average and minimum cash balance;
- EMI/loan-debit detection and missed/bounced-payment signals where reliably identified;
- inflow-to-outflow and debt-service coverage;
- cash-flow stress and end-of-month balance patterns;
- cross-checks between declared income, CIBIL obligations and observed payments.

The frozen V1 model was not trained on deposit-account transactions. These fields may verify the
editable annual-income value, but new bank-transaction features require point-in-time engineering,
training and new untouched validation before they can affect probability.

Official references:

- [RBI NBFC Account Aggregator Master Directions](https://systemhealth.rbi.org.in/Scripts/BS_ViewMasDirections.aspx_id%3D10598%281%29.html)
- [ReBIT Account Aggregator deposit schema](https://specifications.rebit.org.in/api_schema/account_aggregator/documentation/deposit_v2.0.0.html)

## What a lending partner can provide

A lending/origination partner can directly supply the current request, computed proposed annuity,
merchant invoice or goods price, KYC/document results, and its own prior application and servicing
history. A servicing partner may support the current 102-factor block when it can deliver the same
event semantics and point-in-time cutoff:

- previous applications and outcomes;
- POS contract state and remaining installments;
- credit-card balance, utilization, drawings, payments and delinquency;
- scheduled installments, actual payments, shortfalls and lateness;
- source coverage and freshness.

“Partner available” must never mean “observed zero.” Missing data, stale data, consent failure,
schema mismatch and connector outage require separate states and fail-closed routing.

## Source marks on the editable form

| Editable field | Primary source | Verification or secondary evidence | CIBIL? |
|---|---|---|---|
| Annual income | Applicant declaration | Consented bank/AA transaction analysis; payroll/document verification | CIBIL may contain previously lender-reported income, but it can be stale and must be confirmed. |
| Requested credit | Applicant request / lender origination system | Current application record | No; current request is not a bureau fact. |
| Annual proposed repayment | Lender calculation from proposed terms | Product/pricing engine | No; historical EMIs are separate evidence. |
| Goods price | Merchant quote, invoice or application document | Applicant confirmation | No. |

The dashboard renders these labels immediately above each field. It also exposes the five factor
families, their counts and connector status, and allows the full factor-source map to be downloaded
as CSV.

## JSON intake package

For the current local demo, download `configs/intake_template.json` or use the dashboard's
**Download JSON template** action:

```json
{
  "application_id": 176483,
  "annual_income": 157500.0,
  "requested_credit": 440784.0,
  "annual_annuity": 34956.0,
  "goods_price": 360000.0
}
```

The package deliberately supports only a known local application ID and the four controlled
current-term overrides. Unknown keys and invalid values are rejected. The remaining factors are
loaded from the frozen local stores or computed, which preserves the model's trained schema and
prevents an upload from pretending that missing history exists.

For real new applicants, replace the local-ID lookup with a versioned canonical application
package plus raw CIBIL/partner/AA payload adapters. Keep raw source payloads, normalized facts,
model-ready features and reviewer-visible provenance as separate contracts.
