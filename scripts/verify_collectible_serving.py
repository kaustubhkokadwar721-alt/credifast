"""Prove the workbook serving path matches the training path, probability for probability.

Picks applicants from the calibration partition, scores each two ways -- directly from the
engineered training frame, and through the workbook export/parse/score path -- then reports
the maximum absolute difference. Any gap is train/serve skew.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\verify_collectible_serving.py --applicants 5
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credifast.data.workbook_intake import read_intake
from credifast.modeling.application_baseline import assign_splits
from credifast.modeling.application_features import engineer_application_features
from credifast.modeling.collectible_features import add_obligation_ratios, recompute_missing_count
from credifast.modeling.collectible_inference import align_features, load_bundle, score_intake

APPLICATION_PATH = ROOT / "data" / "raw" / "application_train.csv"
HISTORY_PATH = ROOT / "data" / "processed" / "history_features.parquet"
REPORT_OUTPUT = ROOT / "artifacts" / "collectible_serving_verification.json"
TOLERANCE = 1e-9


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--applicants", type=int, default=5)
    arguments = parser.parse_args()

    bundle, calibration_bundle = load_bundle()
    selected = list(bundle["selected_features"])

    _log("loading sources")
    application = pd.read_csv(APPLICATION_PATH, low_memory=False)
    history = pd.read_parquet(HISTORY_PATH)
    frame = application.merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    target = frame["TARGET"].astype("int8")
    splits = assign_splits(target, random_seed=42)
    engineered = add_obligation_ratios(engineer_application_features(frame))
    counted = tuple(f for f in selected if f != "APPLICATION_MISSING_COUNT")
    engineered["APPLICATION_MISSING_COUNT"] = recompute_missing_count(engineered, counted)

    calibration_mask = splits.eq("calibration")
    # Prefer applicants with all three source families present, so the comparison exercises
    # every aggregation path rather than only the application sheet.
    rich = calibration_mask & engineered[
        ["HAS_BUREAU_HISTORY", "HAS_CREDIT_CARD_HISTORY", "HAS_POS_HISTORY"]
    ].fillna(0).eq(1).all(axis=1)
    candidates = engineered.loc[rich, "SK_ID_CURR"].head(arguments.applicants).tolist()
    if not candidates:
        _log("no calibration applicant carries all three source families")
        return 1
    _log(f"comparing {len(candidates)} applicants: {candidates}")

    from export_demo_workbook import export

    results = []
    for applicant in candidates:
        position = engineered.index[engineered["SK_ID_CURR"] == applicant][0]
        aligned, _ = align_features(engineered.loc[[position], selected], selected)
        for column in bundle["numeric_features"]:
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
        direct_raw = float(bundle["model"].predict_proba(bundle["preprocessor"].transform(aligned))[0, 1])
        direct = float(calibration_bundle["calibrator"].predict(np.array([direct_raw]))[0])

        with tempfile.TemporaryDirectory() as workspace:
            destination = Path(workspace) / "workbook"
            export(int(applicant), destination, "application_train.csv")
            served = score_intake(read_intake(destination))

        difference = abs(direct - served.probability)
        results.append(
            {
                "application_id": int(applicant),
                "training_path_probability": direct,
                "workbook_path_probability": served.probability,
                "absolute_difference": difference,
                "features_supplied": served.features_supplied,
            }
        )
        _log(
            f"  {applicant}: training {direct:.10f}  workbook {served.probability:.10f}  "
            f"diff {difference:.3e}"
        )

    worst = max(result["absolute_difference"] for result in results)
    report = {
        "run_version": "1.0.0",
        "tolerance": TOLERANCE,
        "max_absolute_difference": worst,
        "consistent": bool(worst <= TOLERANCE),
        "applicants": results,
        "note": (
            "Compares the same applicant scored from the engineered training frame against the "
            "workbook export/parse/score path. A non-zero difference is train/serve skew."
        ),
    }
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _log(f"max absolute difference {worst:.3e} -> {'CONSISTENT' if report['consistent'] else 'SKEW'}")
    _log(f"wrote {REPORT_OUTPUT}")
    return 0 if report["consistent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
