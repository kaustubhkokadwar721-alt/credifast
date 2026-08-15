# Phase 7 — collectible-input candidate: post-prediction group audit

Date: 2026-08-15. Partition: **calibrator-evaluation half of the calibration partition**
(23,064 rows, 8.07% event rate). The V1 final holdout was not read.

Model: `collectible-input-lightgbm-v3_extended` v0.3.0, identity calibration.
Runner: `scripts/audit_collectible_fairness.py`. Output: `artifacts/collectible_fairness_audit.json`.

## Why this was re-run rather than inherited

V3 dropped 145 of V1's 259 features, including all three external scores. A fairness result
measured on V1's feature set says nothing reliable about V3's. This audit uses the identical
method to the frozen V1 audit — same group metrics, same 95% Wilson intervals, same
suppression rule (under 200 rows or under 20 events/non-events), same fixed 20% review-capacity
proxy — so the two are directly comparable.

Protected fields stay audit-only. The runner raises if `CODE_GENDER` or `DAYS_BIRTH` appears in
the model's feature list.

## Overall

ROC-AUC 0.7404, average precision 0.2272, review threshold 0.114421 at 20% capacity.

## Gender

| Group | Rows | Event rate | Review rate | TPR | FPR | ROC-AUC | Slope | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F | 15,150 | 0.0717 | 0.1842 | 0.4747 | 0.1617 | 0.7356 | 1.065 | 0.0067 |
| M | 7,913 | 0.0979 | 0.2304 | 0.5381 | 0.1970 | 0.7426 | 1.082 | 0.0144 |
| Unknown | 1 | — | — | — | — | suppressed | — | — |

**Routing gaps narrowed against V1.** Review-rate gap 0.0595 → 0.0462, TPR gap 0.0778 → 0.0634,
FPR gap 0.0465 → 0.0353. Ranking gaps widened slightly: ROC-AUC 0.0024 → 0.0070, average
precision 0.0269 → 0.0358.

Net: gender routing is somewhat more even than V1, ranking quality somewhat less even. Neither
movement is large.

## Age band

| Group | Rows | Event rate | Review rate | TPR | FPR | ROC-AUC | Slope | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 18-24 | 639 | 0.1221 | 0.5149 | 0.6923 | 0.4902 | 0.6637 | 0.900 | 0.0394 |
| 25-34 | 5,131 | 0.1062 | 0.3208 | 0.6661 | 0.2798 | 0.7494 | 1.166 | 0.0125 |
| 35-44 | 6,320 | 0.0848 | 0.1959 | 0.4907 | 0.1686 | 0.7494 | 1.095 | 0.0117 |
| 45-54 | 5,395 | 0.0753 | 0.1752 | 0.4483 | 0.1529 | 0.7285 | 1.051 | 0.0061 |
| 55-64 | 4,633 | 0.0540 | 0.0926 | 0.2640 | 0.0828 | 0.6882 | 0.959 | 0.0080 |
| 65+ | 946 | 0.0497 | 0.0275 | 0.1064 | 0.0234 | 0.7205 | 1.268 | 0.0138 |

**This is the material regression, and it must not be softened.**

| Gap | V1 | V3 | Change |
|---|---:|---:|---|
| Review rate | 0.4056 | **0.4874** | wider by 0.0817 |
| False positive rate | 0.3639 | **0.4668** | wider by 0.1029 |
| ROC-AUC | 0.0701 | **0.0857** | wider by 0.0156 |
| True positive rate | 0.6077 | 0.5859 | narrower by 0.0218 |
| Average precision | 0.1956 | 0.1599 | narrower by 0.0357 |

The narrower TPR gap is not an improvement. The 18-24 band's review rate **rose** from 46.2%
to 51.5% while its true positive rate **fell** from 0.8205 to 0.6923. More of that band is
pushed into review and fewer of its actual events are caught. Its ROC-AUC fell from 0.7345 to
0.6637, the worst of any band, and its ECE is 0.0394 with slope 0.900.

Plain statement: **for applicants aged 18-24, the collectible model is both less accurate and
more aggressive than V1.** A 66.4% ROC-AUC means ranking within that band is weak.

The 65+ band moves the opposite way — review rate fell from 5.6% to 2.75% and TPR from 0.2128
to 0.1064 — which widens the spread from both ends.

## Source availability (new dimension)

Segments count only the three families V3 consumes: bureau record, revolving history,
installment history.

| Group | Rows | Event rate | Review rate | TPR | ROC-AUC | Slope | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| full (3 of 3) | 5,473 | 0.0819 | 0.2056 | 0.5067 | 0.7447 | 1.042 | 0.0069 |
| partial (2 of 3) | 14,188 | 0.0782 | 0.1889 | 0.4955 | 0.7422 | 1.111 | 0.0075 |
| thin (0-1) | 3,403 | 0.0893 | 0.2374 | 0.5132 | 0.7214 | 1.000 | 0.0113 |

**Thin-file calibration is markedly better than V1's.** V1's holdout thin segment had slope
1.290, with exact zero-source cases at slope 1.350 and ECE 0.0369. V3's thin segment sits at
slope 1.000 and ECE 0.0113 — the best-calibrated slope of the three segments.

Every source-availability gap falls below the attention thresholds. No segment is being
routed or ranked far out of line with the others.

The plausible reason is that V3 does not depend on `EXT_SOURCE`, which thin-file applicants
disproportionately lack, nor on 137 Home-Credit-only fields. A model built from evidence that
degrades gracefully degrades gracefully.

This does **not** license relaxing the thin-file manual-review rule. That would be a policy
change justified only by calibration-partition evidence, on a candidate with no unbiased final
evaluation.

## Defect found in this audit and corrected

The first version of `_source_segment` counted every `HAS_*` column in the engineered frame —
all five V1 families — while labelling the buckets as thirds. That mislabelled the segments and
produced a spurious thin-file slope of 1.336 alongside a 0.2654 TPR gap. Corrected to count
only the model's own three flags; the numbers above are from after the fix. Recording this
because the erroneous version told a materially worse and wrong story about thin files.

## Assessment

`share_with_caveats`, unchanged from V1.

Ship-blocking for autonomous use: yes, but that was already true. The product routes every
segment to a human queue, so the age-band spread is a disclosure and monitoring obligation
rather than an active harm in the current design.

Required before any interface ships on this model:

1. State the 18-24 result plainly in the model card. Do not let "overall ROC-AUC 0.74" stand
   in for a band where it is 0.66.
2. Keep the fixed-capacity review proxy labelled as an audit device, not a decision threshold.
3. Keep thin and partial files on data-limited manual review despite the improved calibration.

## Limits

- Descriptive only. Observed gaps do not establish discrimination, causality, or legal
  compliance.
- Gender and age are excluded from prediction, but remaining inputs may be correlated proxies.
  Employment vintage, housing type and income type all plausibly track age.
- 18-24 (639 rows) and 65+ (946 rows) are the smallest eligible bands; their intervals are the
  widest and the observed movements are correspondingly less certain.
- Competition data on a random split does not represent an Indian lending population.
- No unbiased final evaluation exists for this candidate.
- Independent legal, compliance and model-risk review is required before any real lending use.
