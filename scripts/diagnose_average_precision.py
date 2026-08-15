"""Diagnose frozen-holdout average precision without changing the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def top_fraction_metrics(target: np.ndarray, probability: np.ndarray, fraction: float) -> dict:
    order = np.argsort(-probability)
    selected = order[: int(np.ceil(len(order) * fraction))]
    positives = float(target.sum())
    true_positives = int(target[selected].sum())
    precision = float(target[selected].mean())
    prevalence = float(target.mean())
    return {
        "population_fraction": fraction,
        "selected_rows": len(selected),
        "probability_threshold": float(probability[selected].min()),
        "true_positives": true_positives,
        "false_positives": int(len(selected) - true_positives),
        "precision": precision,
        "recall": float(true_positives / positives),
        "lift_over_prevalence": float(precision / prevalence),
    }


def precision_recall_targets(target: np.ndarray, probability: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(target, probability)
    precision = precision[:-1]
    recall = recall[:-1]
    result: dict[str, list[dict]] = {"precision_targets": [], "recall_targets": []}
    for target_precision in (0.20, 0.30, 0.40, 0.50):
        eligible = np.flatnonzero(precision >= target_precision)
        if eligible.size:
            index = eligible[np.argmax(recall[eligible])]
            result["precision_targets"].append(
                {
                    "minimum_precision": target_precision,
                    "achieved_precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "threshold": float(thresholds[index]),
                }
            )
    for target_recall in (0.20, 0.40, 0.60, 0.80):
        eligible = np.flatnonzero(recall >= target_recall)
        if eligible.size:
            index = eligible[np.argmax(precision[eligible])]
            result["recall_targets"].append(
                {
                    "minimum_recall": target_recall,
                    "achieved_recall": float(recall[index]),
                    "precision": float(precision[index]),
                    "threshold": float(thresholds[index]),
                }
            )
    f1 = 2 * precision * recall / np.maximum(precision + recall, np.finfo(float).eps)
    best = int(np.argmax(f1))
    result["maximum_f1"] = {
        "f1": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "threshold": float(thresholds[best]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/final_holdout_predictions.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/average_precision_diagnostic.json")
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    target = frame["TARGET"].to_numpy(dtype=int)
    candidate = frame["operational_probability"].to_numpy(dtype=float)
    prevalence = float(target.mean())
    models = {}
    for column in (
        "application_probability",
        "history_probability",
        "operational_probability",
    ):
        ap = float(average_precision_score(target, frame[column]))
        models[column] = {
            "average_precision": ap,
            "lift_over_prevalence": float(ap / prevalence),
        }

    segment_metrics = []
    for segment, group in frame.groupby("history_segment", observed=True):
        segment_target = group["TARGET"].to_numpy(dtype=int)
        segment_probability = group["operational_probability"].to_numpy(dtype=float)
        segment_prevalence = float(segment_target.mean())
        segment_ap = float(average_precision_score(segment_target, segment_probability))
        segment_metrics.append(
            {
                "segment": str(segment),
                "rows": len(group),
                "events": int(segment_target.sum()),
                "prevalence": segment_prevalence,
                "average_precision": segment_ap,
                "lift_over_prevalence": float(segment_ap / segment_prevalence),
            }
        )

    report = {
        "run_version": "1.0.0",
        "status": "diagnostic_only_frozen_holdout_not_retuned",
        "population": {
            "rows": len(frame),
            "events": int(target.sum()),
            "prevalence": prevalence,
        },
        "model_comparison": models,
        "operational_candidate": {
            "absolute_ap_gain_over_random": models["operational_probability"][
                "average_precision"
            ]
            - prevalence,
            "ap_gain_over_application_model": models["operational_probability"][
                "average_precision"
            ]
            - models["application_probability"]["average_precision"],
            "ap_gain_over_history_model": models["operational_probability"][
                "average_precision"
            ]
            - models["history_probability"]["average_precision"],
            "top_fraction_operating_points": [
                top_fraction_metrics(target, candidate, fraction)
                for fraction in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
            ],
            **precision_recall_targets(target, candidate),
        },
        "history_segments": segment_metrics,
        "interpretation": [
            "Average precision must be read against the 8.07% event prevalence; random ranking has AP equal to prevalence.",
            "The metric remains modest in absolute terms and does not establish production-grade credit selection.",
            "The frozen gap-aware candidate preserves, rather than materially improves, the history model's aggregate average precision.",
            "Thresholds are diagnostic operating points, not approval, decline, pricing, or adverse-action policy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
