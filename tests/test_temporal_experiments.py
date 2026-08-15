from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credifast.modeling.temporal_experiments import (
    _gates,
    _paired_bootstrap,
    engineer_temporal_cross_features,
    temporal_columns_for_variant,
)


def test_temporal_selectors_are_additive_and_bounded_to_declared_families() -> None:
    columns = [
        "HAS_TEMPORAL_INSTALLMENT_HISTORY",
        "TINST_LATE_RATE_EW_90D",
        "TINST_RECENT_EVENT_1_LATE_FLAG",
        "TINST_CURRENT_LATE_STREAK",
        "TPREV_APPLICATION_COUNT_90D",
        "TPOS_ACTIVE_CONCURRENCY_MAX_12M",
        "TCC_UTILIZATION_MEAN_6M",
        "TBUREAU_ACTIVE_DEBT_SUM",
        "CYCLING_SIGNAL_FAMILY_COUNT",
        "BURDEN_ACTIVE_DEBT_TO_INCOME",
        "TARGET",
    ]

    recency = temporal_columns_for_variant(columns, "repayment_recency")
    sequence = temporal_columns_for_variant(columns, "repayment_sequence")
    all_temporal = temporal_columns_for_variant(columns, "all")

    assert "TINST_LATE_RATE_EW_90D" in recency
    assert "TINST_RECENT_EVENT_1_LATE_FLAG" not in recency
    assert "TINST_RECENT_EVENT_1_LATE_FLAG" in sequence
    assert "TINST_CURRENT_LATE_STREAK" in sequence
    assert "TARGET" not in all_temporal
    assert "HAS_TEMPORAL_INSTALLMENT_HISTORY" in recency
    assert set(recency) | set(sequence) <= set(all_temporal)


def test_temporal_cross_features_encode_current_and_historical_burden() -> None:
    frame = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100.0, 0.0],
            "AMT_ANNUITY": [20.0, 15.0],
            "AMT_CREDIT": [200.0, 100.0],
            "TBUREAU_ACTIVE_ANNUITY_SUM": [10.0, 5.0],
            "TBUREAU_ACTIVE_DEBT_SUM": [50.0, 25.0],
            "HAS_TEMPORAL_INSTALLMENT_HISTORY": [1, 0],
            "HAS_TEMPORAL_BUREAU_HISTORY": [1, 1],
            "CYCLING_SIGNAL_FAMILY_COUNT": [2, 0],
        }
    )

    result = engineer_temporal_cross_features(frame)

    assert result.loc[0, "BURDEN_ACTIVE_ANNUITY_TO_INCOME"] == pytest.approx(0.1)
    assert result.loc[0, "BURDEN_PROPOSED_PLUS_ACTIVE_ANNUITY_TO_INCOME"] == pytest.approx(
        0.3
    )
    assert result.loc[0, "BURDEN_TEMPORAL_SOURCE_COUNT"] == 2
    assert result.loc[0, "CYCLING_SIGNAL_BURDEN_INTERACTION"] == pytest.approx(0.6)
    assert np.isnan(result.loc[1, "BURDEN_ACTIVE_DEBT_TO_INCOME"])


def test_paired_bootstrap_detects_strictly_better_ranking() -> None:
    target = np.array([0, 0, 0, 1, 1, 1] * 100)
    baseline = np.full(len(target), 0.5)
    candidate = np.where(target == 1, 0.9, 0.1)

    result = _paired_bootstrap(
        target,
        baseline,
        candidate,
        iterations=30,
        random_seed=42,
    )

    assert result["average_precision_delta"]["probability_positive"] == 1.0
    assert result["brier_delta"]["probability_improved"] == 1.0


def test_research_gate_requires_all_predeclared_conditions() -> None:
    baseline = {
        "average_precision": 0.25,
        "brier_score": 0.070,
        "roc_auc": 0.75,
        "top_20_percent": {"event_capture_rate": 0.55},
    }
    candidate = {
        "average_precision": 0.254,
        "brier_score": 0.0701,
        "roc_auc": 0.7495,
        "top_20_percent": {"event_capture_rate": 0.548},
    }
    uncertainty = {"average_precision_delta": {"probability_positive": 0.95}}
    config = {
        "research_promotion_gates": {
            "minimum_absolute_average_precision_gain": 0.003,
            "minimum_probability_ap_gain_positive": 0.9,
            "maximum_brier_regression": 0.0002,
            "maximum_roc_auc_regression": 0.001,
            "maximum_top_20_capture_regression": 0.005,
        }
    }

    result = _gates(baseline, candidate, uncertainty, config)

    assert result["all_passed"] is True

    candidate["average_precision"] = 0.252
    failed = _gates(baseline, candidate, uncertainty, config)
    assert failed["all_passed"] is False
    assert failed["checks"]["average_precision_gain"] is False
