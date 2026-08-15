"""Score a user-supplied intake workbook with the collectible-input candidate.

This is the path V1 could not offer: an applicant who is not already a row in the training
population gets a repayment-difficulty estimate from data they submitted.

The candidate is selected on the calibration partition only. It has no unbiased final
evidence and must not be presented as validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..data.workbook_intake import IntakeResult, read_intake

DEFAULT_MODEL_PATH = Path("artifacts/models/collectible_lightgbm.joblib")
DEFAULT_CALIBRATOR_PATH = Path("artifacts/models/collectible_calibrator.joblib")

# Source families the reviewer sees named, rather than a coverage count.
SOURCE_FLAGS = {
    "HAS_BUREAU_HISTORY": "credit report",
    "HAS_CREDIT_CARD_HISTORY": "revolving account history",
    "HAS_POS_HISTORY": "installment account history",
}


class CollectibleModelUnavailable(RuntimeError):
    """Raised when the local candidate bundle is missing."""


@dataclass(frozen=True)
class CollectibleScore:
    probability: float
    raw_probability: float
    calibrator_method: str
    model_name: str
    model_version: str
    feature_count: int
    features_supplied: int
    sources_present: tuple[str, ...]
    sources_absent: tuple[str, ...]
    findings: tuple[str, ...]
    evidence: dict
    validated: bool = False

    def to_dict(self) -> dict:
        return {
            "repayment_difficulty_probability": self.probability,
            "raw_probability": self.raw_probability,
            "calibrator_method": self.calibrator_method,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_count": self.feature_count,
            "features_supplied": self.features_supplied,
            "sources_present": list(self.sources_present),
            "sources_absent": list(self.sources_absent),
            "data_findings": list(self.findings),
            "evidence": self.evidence,
            "validated": self.validated,
            "status_note": (
                "Internal research score for decision support. Selected on the calibration "
                "partition only, with no unbiased final evaluation. Not an official bureau "
                "score and not an approval decision."
            ),
        }


def load_bundle(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    calibrator_path: str | Path = DEFAULT_CALIBRATOR_PATH,
) -> tuple[dict, dict]:
    model_file, calibrator_file = Path(model_path), Path(calibrator_path)
    missing = [str(p) for p in (model_file, calibrator_file) if not p.exists()]
    if missing:
        raise CollectibleModelUnavailable(
            "missing collectible model artifacts: "
            + ", ".join(missing)
            + " (run scripts/train_collectible_champion.py)"
        )
    return joblib.load(model_file), joblib.load(calibrator_file)


def align_features(row: pd.DataFrame, selected: list[str]) -> tuple[pd.DataFrame, int]:
    """Place supplied values into the trained column order, leaving absences missing.

    A workbook legitimately omits whole families -- a new-to-credit applicant has no
    tradelines. Those columns must arrive as missing, never as zero, because the model was
    trained with missingness carrying meaning.
    """

    aligned = pd.DataFrame(index=row.index, columns=selected, dtype=object)
    supplied = 0
    for column in selected:
        if column in row.columns:
            aligned[column] = row[column].to_numpy()
            supplied += 1
        else:
            aligned[column] = np.nan

    # Counted over the model's own columns, matching how training computes it. Doing this
    # at intake instead would count whatever the workbook produced and disagree with training.
    if "APPLICATION_MISSING_COUNT" in aligned.columns:
        counted = [c for c in selected if c != "APPLICATION_MISSING_COUNT"]
        aligned["APPLICATION_MISSING_COUNT"] = (
            aligned[counted].isna().sum(axis=1).astype("int16")
        )
    return aligned, supplied


def score_intake(
    intake: IntakeResult,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    calibrator_path: str | Path = DEFAULT_CALIBRATOR_PATH,
) -> CollectibleScore:
    """Score an already-parsed workbook."""

    bundle, calibration_bundle = load_bundle(model_path, calibrator_path)
    selected = list(bundle["selected_features"])
    aligned, supplied = align_features(intake.features, selected)

    for column in bundle["numeric_features"]:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")

    matrix = bundle["preprocessor"].transform(aligned)
    raw = float(bundle["model"].predict_proba(matrix)[0, 1])
    calibrated = float(calibration_bundle["calibrator"].predict(np.array([raw]))[0])

    present, absent = [], []
    for flag, label in SOURCE_FLAGS.items():
        value = intake.features.iloc[0].get(flag)
        (present if value == 1 else absent).append(label)

    return CollectibleScore(
        probability=calibrated,
        raw_probability=raw,
        calibrator_method=calibration_bundle["method"],
        model_name=bundle["model_name"],
        model_version=bundle["model_version"],
        feature_count=len(selected),
        features_supplied=supplied,
        sources_present=tuple(present),
        sources_absent=tuple(absent),
        findings=tuple(intake.findings),
        evidence=intake.evidence,
    )


def score_workbook(
    path: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    calibrator_path: str | Path = DEFAULT_CALIBRATOR_PATH,
) -> CollectibleScore:
    """Read a workbook from disk and score it."""

    return score_intake(
        read_intake(path), model_path=model_path, calibrator_path=calibrator_path
    )
