# Artifact registry

Small, non-sensitive reproducibility artifacts may be stored here:

- source file manifests and hashes;
- schema and quality summaries;
- split metadata without applicant records;
- evaluation metrics;
- model cards;
- artifact checksums.

Current model-development evidence also includes the history feature/quality reports, model comparison, calibration diagnostics, and the research champion bundle manifest. Applicant-level prediction files remain under ignored `data/processed/` paths.

Model binaries and feature matrices are ignored by Git. Raw applicant rows must never be placed here.
