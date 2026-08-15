"""Create and execute the Phase 3 quality and model-comparison notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        nbformat.v4.new_markdown_cell(
            "# CrediFast Phase 3: history feature quality and model comparison"
        ),
        nbformat.v4.new_markdown_cell(
            """## tl;dr

- The target-free history store reconciles all **356,255** train and test applications at one row per `SK_ID_CURR`; the automated quality gate passes with no invalid rates, negative counts, or non-finite values.
- History coverage ranges from **29.1% for credit cards** to **95.3% for installments**. Missing source blocks are therefore modeled with explicit availability flags, not converted to zero behavior.
- On the untouched calibration partition, the history-enriched LightGBM reaches **0.7814 ROC-AUC** and **0.2739 average precision**, improving over the application-only LightGBM while the final holdout remains sealed.
"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Context & Methods

This notebook is the inspectable audit trail for Phase 3. It reads the saved aggregation, quality, and model metric artifacts produced by repository commands; it does not retrain models or alter data.

### Key Assumptions

- All six history tables describe events observed before the current application decision.
- A missing source history is not equivalent to a measured zero balance or zero delinquency.
- Model comparisons use the same seed and 70/15/15 stratified split. Only the calibration partition is scored here.
"""
        ),
        nbformat.v4.new_markdown_cell("## Data\n\n### 1. Load saved evidence"),
        nbformat.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd

ROOT = Path.cwd()
ARTIFACTS = ROOT / "artifacts"

quality = json.loads((ARTIFACTS / "history_quality_report.json").read_text())
feature_report = json.loads((ARTIFACTS / "history_features_report.json").read_text())
quality["summary"]
"""
        ),
        nbformat.v4.new_markdown_cell("### 2. Review source coverage"),
        nbformat.v4.new_code_cell(
            """coverage = (
    pd.DataFrame(quality["checks"]["source_coverage"])
    .T.rename_axis("source")
    .reset_index()
)
coverage["coverage_percent"] = 100 * coverage["rate"]
coverage[["source", "rows", "coverage_percent"]].sort_values("coverage_percent")
"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Results

### 3. Confirm the automated modeling-readiness gate

These are hard invariants: exact application reconciliation, unique/non-null primary keys, no target field, bounded rate features, nonnegative counts, and finite numeric values.
"""
        ),
        nbformat.v4.new_code_cell(
            """assert quality["automated_gate"]["passed"]
assert quality["summary"]["ready_for_modeling"]
assert quality["summary"]["row_count"] == quality["summary"]["expected_application_rows"]
assert quality["summary"]["unique_keys"] == quality["summary"]["row_count"]
pd.DataFrame({"required_check": quality["automated_gate"]["required"]})
"""
        ),
        nbformat.v4.new_markdown_cell(
            """### 4. Quantify source-availability differences

Availability itself separates populations, so the `HAS_*` indicators are operationally meaningful. These rates are descriptive associations only and are not causal fairness claims.
"""
        ),
        nbformat.v4.new_code_cell(
            """availability_rows = []
for source, groups in quality["checks"]["target_rate_by_source_availability"].items():
    for group in groups:
        availability_rows.append({"source": source, **group})
availability = pd.DataFrame(availability_rows)
availability["target_rate_percent"] = 100 * availability["target_rate"]
availability[["source", "available", "rows", "target_rate_percent"]]
"""
        ),
        nbformat.v4.new_markdown_cell("### 5. Compare models on the same calibration partition"),
        nbformat.v4.new_code_cell(
            """metric_files = [
    "application_baseline_metrics.json",
    "application_lightgbm_metrics.json",
    "history_lightgbm_metrics.json",
]
comparison_rows = []
for metric_file in metric_files:
    artifact = json.loads((ARTIFACTS / metric_file).read_text())
    metrics = artifact["calibration_partition_metrics"]
    comparison_rows.append(
        {
            "model": artifact["model_name"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "brier_score": metrics["brier_score"],
            "top_10_capture": metrics["top_10_percent"]["event_capture_rate"],
            "top_20_capture": metrics["top_20_percent"]["event_capture_rate"],
            "holdout_evaluated": artifact["holdout_evaluated"],
        }
    )
comparison = pd.DataFrame(comparison_rows)
comparison.round(4)
"""
        ),
        nbformat.v4.new_code_cell(
            """application_only = comparison.loc[
    comparison["model"].eq("application-lightgbm-challenger")
].iloc[0]
history_enriched = comparison.loc[
    comparison["model"].eq("history-enriched-lightgbm-challenger")
].iloc[0]
pd.Series(
    {
        "roc_auc_delta": history_enriched.roc_auc - application_only.roc_auc,
        "average_precision_delta": (
            history_enriched.average_precision - application_only.average_precision
        ),
        "brier_score_delta": history_enriched.brier_score - application_only.brier_score,
        "top_20_capture_delta": (
            history_enriched.top_20_capture - application_only.top_20_capture
        ),
    },
    name="history_minus_application_only",
).round(4)
"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Takeaways

1. The history feature store is safe to use for model development under the automated gate and preserves the intended application grain.
2. Sparse credit-card history is structural, not a pipeline failure; source blocks must retain nulls plus availability flags.
3. History improves both ranking and rare-event precision on calibration data. The model is a challenger, not yet a production credit decision system.
4. Next: calibrate probabilities, freeze operating bands and reason-code behavior, then score the sealed holdout once.
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/phase3_history_quality.ipynb"),
    )
    args = parser.parse_args()
    notebook = build_notebook()
    NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=Path.cwd())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"Wrote and executed {args.output}")


if __name__ == "__main__":
    main()
