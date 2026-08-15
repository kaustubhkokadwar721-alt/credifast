# Calibration and thin-file strategy milestone

## Outcome

Identity, Platt/sigmoid, and isotonic calibration were compared on a 50/50 stratified split of the existing calibration partition. Candidate calibrators were fit on 23,063 rows and evaluated on a disjoint 23,064 rows.

The unmodified history-model probability won by both Brier score and log loss. CrediFast therefore freezes an explicit identity calibrator rather than forcing a probability transformation that worsens held-back calibration performance.

## Candidate comparison

| Method | Brier score | Log loss | ECE | Calibration intercept | Calibration slope |
|---|---:|---:|---:|---:|---:|
| **Identity** | **0.065831** | **0.238328** | **0.00321** | 0.028 | 1.013 |
| Platt/sigmoid | 0.065833 | 0.238342 | 0.00396 | -0.027 | 0.999 |
| Isotonic | 0.065986 | 0.241691 | 0.00541 | -0.191 | 0.917 |

ECE is equal-count expected calibration error. The identity model's 0.00321 ECE means calibration-decile predicted and observed rates differ by about 0.32 percentage points on average, weighted by bin size. This result is specific to the held-back calibration-evaluation half and does not substitute for the sealed holdout.

## Final calibration bundle

After method selection, the identity calibrator is refit on all 46,127 calibration rows and stored with the base-model hash, method, version, seed, and fit-population metadata. Because identity scaling has no learned parameters, the refit preserves the raw probability exactly while maintaining a stable deployable calibration interface.

The next unbiased evaluation of the frozen model-plus-calibrator bundle is the sealed holdout. No holdout features or predictions were created in this phase.

## Thin-file assessment

| History segment | Evaluation rows | Base rate | Mean prediction | ECE | Slope | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Full: 4-5 sources | 19,517 | 7.94% | 7.98% | 0.36% | 1.050 | 0.7876 |
| Partial: 2-3 sources | 2,433 | 10.03% | 9.99% | 1.98% | 0.772 | 0.7141 |
| Thin: 0-1 sources | 1,114 | 6.19% | 5.30% | 1.21% | 1.072 | 0.7660 |

Partial-history applications show the weakest discrimination and the largest within-segment calibration error. The sample sizes also make segment estimates less stable than the full-history result.

## Confidence route

The research confidence policy is deliberately separate from predicted probability:

- four or five historical sources: standard route when risk is not near a grade boundary;
- four or five sources near a boundary: manual review because small probability movement could change the grade;
- zero to three sources: low-confidence, data-limited manual review;
- confidence routing never changes the model probability and is not an approval policy.

This behavior is implemented in `src/credifast/modeling/confidence.py` and configured by `configs/history_confidence_policy.json`.

## Reproducibility artifacts

- `artifacts/history_calibration_metrics.json`
- `artifacts/champion_bundle_manifest.json`
- local ignored `artifacts/models/history_calibrator.joblib`
- local ignored `data/processed/history_calibration_predictions.parquet`

```powershell
credifast-train history-calibrate
```

## Remaining limitations

- Calibration is evaluated on a random competition partition, not out-of-time lender outcomes.
- Segment diagnostics are descriptive and do not replace fairness analysis.
- Risk-grade boundaries are research defaults and have not been linked to lender costs, approval capacity, pricing, or regulation.
- The frozen bundle remains a research champion until explanation, fairness, latency, and sealed-holdout gates pass.
