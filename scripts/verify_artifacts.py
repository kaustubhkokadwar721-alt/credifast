"""Validate versioned JSON artifacts and executed notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact_paths = sorted((root / "artifacts").glob("*.json"))
    if not artifact_paths:
        raise RuntimeError("no JSON artifacts found")
    parsed = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in artifact_paths}
    holdout_artifacts = [
        name
        for name in (
            "application_baseline_metrics.json",
            "application_lightgbm_metrics.json",
            "history_lightgbm_metrics.json",
            "history_calibration_metrics.json",
            "history_gap_strategy_metrics.json",
            "phase3_model_comparison.json",
        )
        if name in parsed
    ]
    for name in holdout_artifacts:
        if parsed[name].get("holdout_evaluated") is not False:
            raise AssertionError(f"{name} does not preserve the sealed holdout")

    final_name = "final_holdout_evaluation.json"
    if final_name in parsed:
        final = parsed[final_name]
        if final.get("status") != "final_random_holdout_evaluation_complete_no_retuning":
            raise AssertionError("final holdout artifact has an unexpected status")
        if final.get("acceptance_gates", {}).get("all_passed") is not True:
            raise AssertionError("final holdout artifact does not pass documented gates")
        if final.get("controls", {}).get("post_holdout_retuning_allowed") is not False:
            raise AssertionError("final holdout artifact permits post-holdout retuning")

    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if any(cell.execution_count is None for cell in code_cells):
            raise AssertionError(f"{path.name} contains an unexecuted code cell")
        errors = [
            output
            for cell in code_cells
            for output in cell.outputs
            if output.output_type == "error"
        ]
        if errors:
            raise AssertionError(f"{path.name} contains executed error outputs")
    print(f"Validated {len(artifact_paths)} JSON artifacts and {len(notebooks)} notebooks")


if __name__ == "__main__":
    main()
