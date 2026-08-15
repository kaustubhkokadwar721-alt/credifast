# Local data zones

This directory is intentionally data-free in Git.

- `raw/`: immutable Kaggle downloads and extracted CSVs.
- `interim/`: cleaned source tables and table-level applicant aggregates.
- `processed/`: applicant feature marts, split assignments, and model-ready datasets.

Raw, interim, and processed records must not be committed. Reproducible manifests, schemas, and aggregate quality reports belong under `artifacts/` when they contain no applicant-level records.
