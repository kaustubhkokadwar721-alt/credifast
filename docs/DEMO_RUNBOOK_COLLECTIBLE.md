# Demo runbook — collectible-input screens

Five-minute walkthrough of the four screens backed by the collectible-input candidate.
The frozen V1 runbook is separate: `docs/DEMO_RUNBOOK.md`.

## Before the room

```powershell
.\.venv\Scripts\python.exe scripts\generate_demo_workbooks.py
.\.venv\Scripts\python.exe -m streamlit run ui\app.py
```

The generator writes `examples/demo/` (five `.xlsx` plus matching CSV directories). It is
idempotent and takes a few seconds. Everything runs offline.

Confirm before presenting: the app opens on **Review workbench** and the tab strip reads
Review workbench / New application / Scenario analysis / Portfolio / Model evidence /
Responsible AI.

## The five demonstration cases

Every value is synthetic. No real applicant record ships with this repository.

| Case | Sources | FOIR | Estimate | Route |
|---|---:|---:|---:|---|
| `clean_full_file` | 3 of 3 | 24.0% | 2.13% | Standard manual review |
| `thin_new_to_credit` | 0 of 3 | not available | 16.26% | Data-limited manual review |
| `recent_delinquency` | 3 of 3 | 45.0% | 43.03% | Enhanced manual review |
| `high_obligation` | 3 of 3 | 62.5% | 4.42% | Standard manual review |
| `credit_hungry` | 3 of 3 | 36.8% | 38.19% | Enhanced manual review |

## Run of show

### 1. New application — the thing V1 could not do (90 seconds)

Open **New application**. Select `clean_full_file`, Evaluate case.

Say: this scores an applicant from a package they supplied, not from an identifier that
already exists in the training data. The frozen V1 model cannot do this at all — 67 of its
88 application-side factors exist only inside the Home Credit schema.

Point at the parse summary: 4 tradelines, 152 monthly delinquency records, 60 account
months, 1 enquiry. The intake is read and shown before anything is scored.

### 2. Thin file — what happens when there is nothing (45 seconds)

Switch to `thin_new_to_credit`. Evaluate.

Say: no credit report at all. The obligation ratio reads **not available**, not 0%. The
missing features stay missing rather than being read as zero, and the case routes to
data-limited manual review regardless of the estimate. This is the new-to-credit borrower,
and the honest answer is that a human has to gather more evidence.

Then choose **Manual entry** and submit the form unchanged. Same outcome from typed input —
no file needed.

### 3. High obligation — the case that makes the argument (60 seconds)

Switch to `high_obligation`. Evaluate.

This is the strongest case in the set. The estimate is **4.42%** — low, because the
repayment record is genuinely clean. The obligation ratio is **62.5%**.

Say: the risk model and the affordability screen disagree, and the interface shows both
rather than blending them into one number. Repayment risk answers "has this person repaid";
affordability answers "can they carry another instalment". A single score would have hidden
this.

### 4. Scenario analysis — terms move, history does not (75 seconds)

Open **Scenario analysis**, keep `clean_full_file`. Double the requested credit and the
annual repayment.

Watch the request sweep table. Across a 4x range of request size, the estimate stays near
3.4% while the obligation ratio climbs from 24% to 78%.

Say plainly: the risk model is close to insensitive to how much is being asked for. It was
trained to answer whether this borrower repays, not how large a loan they can carry. The
affordability ratio is what actually moves, and it is the constraint that binds. This is
why the two are kept separate.

Point at the locked-history note: only declared terms are editable. A reviewer can ask what
happens if the applicant borrows less. They cannot ask what happens if the arrears vanish.

### 5. Portfolio — allocating attention (45 seconds)

Open **Portfolio**. Five cases, ordered by estimated difficulty, with the queue composition
strip above and the per-case evidence gaps below.

Say: this is a work-allocation view, not a book of decisions. Two enhanced, one
data-limited, two standard — and every one of them goes to a human. The route names the
queue and nothing else.

### 6. Close on the limitation (30 seconds)

Scroll to the disclosure at the foot of any scored screen.

Say: the candidate is unvalidated. It was selected on a calibration partition, and the V1
final holdout is spent, so there is no unbiased final evidence. Overall ROC-AUC is 0.7404 —
but for applicants aged **18-24 it is 0.6637**. That band is ranked less reliably and gets
routed into review more often than under V1. We re-ran the fairness audit rather than
inheriting V1's, found that regression, and put it on the screen instead of in a footnote.

## Language that must not be used

Approval, decline, sanction, rejected, credit limit, pricing, adverse action, "the model
decided". The system produces an internal research score for decision support and names a
human review queue.

Correct phrasing: "repayment-difficulty probability", "internal research score", "decision
support", "manual review".

## If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| "Demonstration files are missing" | Generator not run | `python scripts\generate_demo_workbooks.py` |
| "The candidate model is not available locally" | Model artifacts absent | `python scripts\train_collectible_champion.py --variant v3_extended` |
| Upload rejected with a field error | Workbook missing a required application column | Check `docs/COLLECTIBLE_INPUT_SPEC.md` section 4 |
| Review workbench tab fails | V1 runtime artifacts missing | Restore `artifacts/models/` and `data/processed/history_features.parquet` |

## Questions to expect

**"Why is accuracy lower than the frozen model?"**
Because 145 of the original features cannot be collected from a real applicant. Most of the
loss is the three unnamed external scores, which no lender can obtain. Dropping the other
137 uncollectible fields cost only 0.0132 ROC-AUC — they were carrying almost nothing.
Details in `docs/PHASE7_COLLECTIBLE_MODEL_STATUS.md`.

**"How do you know the uploaded workbook produces the same answer as training?"**
`scripts/verify_collectible_serving.py` scores the same applicant twice, once from the
training frame and once through the upload path, and reports the maximum difference. It is
0.000e+00 across eight applicants. That check found three real defects before it passed.

**"Can this go into production?"**
No. It has no unbiased final evaluation, no out-of-time validation, no lender-owned
outcome definition, and no independent model-risk or legal review. It is a research
prototype.
