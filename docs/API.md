# Trained-model API

The local API exposes the frozen CrediFast research candidate. It reads the ignored Kaggle files and model binaries from this repository; it does not call an external service.

## Start

```powershell
.\.venv\Scripts\python.exe -m uvicorn credifast.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the generated OpenAPI interface.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process health and local model readiness. |
| `GET` | `/v1/model/status` | Frozen model, artifact, holdout, and approval status. |
| `GET` | `/v1/model/applicants` | Seven curated offline demonstration profiles. |
| `GET` | `/v1/model/input-schema` | Editable fields, JSON template, and primary provenance for all 257 selected model factors. |
| `POST` | `/v1/model/score` | Score one local applicant with optional controlled what-if fields. |
| `POST` | `/v1/score` | Legacy synthetic Phase 0 contract demonstration; not the trained model. |

## Trained-model request

```json
{
  "application_id": 176483,
  "annual_income": 157500,
  "requested_credit": 440784,
  "annual_annuity": 34956,
  "goods_price": 360000,
  "explain": true,
  "simulate_history_unavailable": false
}
```

Only the four declared financial fields may be overridden. The runtime preserves both original and evaluated values in `input_snapshot`. The response includes:

- repayment-difficulty probability, internal score, and risk grade;
- history and application component probabilities plus blend weight;
- source-availability segment, confidence, and quality flags;
- human-review route and separate affordability screen;
- grouped adverse and favorable explanation factors;
- model, calibration, confidence-policy, and gap-strategy versions.

The same fields can be loaded through `configs/intake_template.json` in the reviewer dashboard.
The input-schema endpoint makes the source labels and full factor mapping machine-readable. It
does not imply that live CIBIL or banking connectors are installed; the current model still reads
all non-editable evidence from local Home Credit records.

`DATA_LIMITED_REVIEW`, `BOUNDARY_REVIEW`, `HIGH_RISK_REVIEW`, and `AFFORDABILITY_REVIEW` are review queues, not autonomous lending decisions.

## Failure behavior

- Unknown applicant or invalid override: HTTP 422 with a specific detail.
- Missing runtime artifacts: HTTP 503.
- A simulated or real missing feature-store row: successful score with `FEATURE_STORE_ROW_MISSING`, low confidence, and `DATA_LIMITED_REVIEW`.

## Example PowerShell call

```powershell
$body = @{
    application_id = 176483
    explain = $true
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri http://127.0.0.1:8000/v1/model/score `
    -ContentType application/json `
    -Body $body
```
