"""Repeatable model-only and explained inference latency benchmark."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import HistoryModelScorer


def _latency_summary(milliseconds: list[float], rows_per_run: int) -> dict:
    values = np.asarray(milliseconds, dtype=float)
    per_row = values / rows_per_run
    return {
        "repetitions": len(values),
        "rows_per_run": rows_per_run,
        "total_milliseconds_p50": float(np.quantile(values, 0.50)),
        "total_milliseconds_p95": float(np.quantile(values, 0.95)),
        "milliseconds_per_row_p50": float(np.quantile(per_row, 0.50)),
        "milliseconds_per_row_p95": float(np.quantile(per_row, 0.95)),
        "rows_per_second_at_p50": float(1000.0 / np.quantile(per_row, 0.50)),
    }


def benchmark_history_inference(
    application_path: str | Path,
    history_path: str | Path,
    model_path: str | Path,
    calibrator_path: str | Path,
    *,
    output_path: str | Path,
) -> dict:
    """Benchmark a warm scorer; file loading and artifact deserialization are excluded."""

    application = pd.read_csv(application_path, low_memory=False).sort_values("SK_ID_CURR").head(1000)
    history = pd.read_parquet(history_path)
    history = history.loc[history["SK_ID_CURR"].isin(set(application["SK_ID_CURR"]))]
    scorer = HistoryModelScorer(model_path, calibrator_path)
    scorer.score(application.head(10), history, explain=True)

    specifications = {
        "probability_only": {1: 30, 100: 5, 1000: 3},
        "with_reasons": {1: 15, 100: 4, 1000: 2},
    }
    results = {}
    for mode, batches in specifications.items():
        explain = mode == "with_reasons"
        mode_results = {}
        for batch_size, repetitions in batches.items():
            sample = application.head(batch_size)
            durations = []
            for _ in range(repetitions):
                started = time.perf_counter()
                scorer.score(sample, history, explain=explain)
                durations.append((time.perf_counter() - started) * 1000.0)
            mode_results[str(batch_size)] = _latency_summary(durations, batch_size)
        results[mode] = mode_results

    model_only_p95 = results["probability_only"]["1"]["total_milliseconds_p95"]
    explained_p95 = results["with_reasons"]["1"]["total_milliseconds_p95"]
    report = {
        "run_version": "1.0.0",
        "scope": "warm in-process scorer; excludes file reads and joblib deserialization",
        "application_source": Path(application_path).name,
        "benchmark_rows_available": len(application),
        "results": results,
        "gates": {
            "single_row_probability_p95_below_300_ms": model_only_p95 < 300.0,
            "single_row_explained_p95_below_1000_ms": explained_p95 < 1000.0,
            "observed_probability_p95_ms": model_only_p95,
            "observed_explained_p95_ms": explained_p95,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "limitations": [
            "This is a local warm-process benchmark, not an HTTP or concurrent load test.",
            "Production p95 must include request validation, feature retrieval, serialization, and contention.",
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
