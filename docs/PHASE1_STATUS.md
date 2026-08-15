# Phase 1 status

## Outcome

The data acquisition and quality layer is implemented and verified against controlled fixtures and the real Home Credit extract.

## Implemented

- Source contracts for ten expected competition files.
- Expected grain, keys, foreign keys, minimum columns, and required fields.
- SHA-256 source hashing.
- File byte, row, column, header, and schema manifesting.
- Missing required-file and invalid-schema reporting.
- Streaming `application_train.csv` profile.
- Applicant-key uniqueness and missing-key checks.
- Target counts and prevalence.
- Per-column missing counts and rates.
- Numeric count, invalid count, minimum, maximum, mean, and standard deviation.
- Core category frequencies.
- Known `DAYS_EMPLOYED=365243` sentinel detection.
- `CODE_GENDER=XNA` detection.
- Nonpositive income, credit, and annuity checks.
- Severity, analytical risk, and remediation for every finding.
- Safe download script that refuses to overwrite raw CSVs.

## Real extract result

- Ten of ten expected files are present.
- Total extracted size: 2,684,261,617 bytes.
- All expected file schemas pass.
- `application_train.csv`: 307,511 rows, 122 columns, 307,511 unique applicant keys.
- Positive target: 24,825 rows, or 8.073%.
- Medium finding: 55,374 `DAYS_EMPLOYED=365243` sentinel values.
- Medium finding: 4 `CODE_GENDER=XNA` values.
- No critical or high application-grain findings.

## Commands after authentication

Configure either the standard `kaggle.json` credential file or the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables. Accept the Home Credit Default Risk competition rules in Kaggle, then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_data.ps1
```

The script will:

1. refuse to overwrite existing raw CSVs;
2. download the competition archive;
3. extract it into `data/raw`;
4. write `artifacts/data_manifest.json`;
5. write `artifacts/application_profile.json`.

## Quality interpretation

The fixture intentionally contains duplicate keys, invalid targets, sentinel values, unknown gender, and nonpositive amounts. The profiler correctly marks it as not ready for modeling with a maximum severity of critical. This verifies failure behavior; it is not evidence about the real Kaggle extract.

## Next gate

After the real profile is generated:

1. review critical and high findings;
2. reconcile table contracts with observed schemas;
3. freeze applicant-level train/calibration/holdout assignments;
4. implement application cleaning and feature generation;
5. train the first base-rate and logistic baselines;
6. add bureau and previous-application aggregates incrementally.
