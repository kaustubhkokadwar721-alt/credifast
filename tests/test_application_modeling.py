import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from credifast.modeling.application_baseline import assign_splits, evaluate_probabilities
from credifast.modeling.application_features import (
    engineer_application_features,
    select_responsible_features,
)

FIXTURE = Path(__file__).parent / "fixtures" / "application_train_sample.csv"


class ApplicationFeatureTests(unittest.TestCase):
    def test_engineering_handles_employment_sentinel_and_ratios(self) -> None:
        frame = pd.read_csv(FIXTURE)
        engineered = engineer_application_features(frame)
        self.assertNotIn("DAYS_EMPLOYED", engineered)
        self.assertEqual(engineered.loc[1, "EMPLOYMENT_SENTINEL"], 1)
        self.assertTrue(np.isnan(engineered.loc[1, "EMPLOYMENT_YEARS"]))
        self.assertAlmostEqual(engineered.loc[0, "CREDIT_TO_INCOME"], 0.4)
        self.assertIn("EXT_SOURCE_MEAN", engineered)

    def test_responsible_selection_excludes_identifiers_and_audit_fields(self) -> None:
        frame = engineer_application_features(pd.read_csv(FIXTURE))
        selection = select_responsible_features(frame, maximum_missing_rate=0.80)
        self.assertNotIn("SK_ID_CURR", selection.selected)
        self.assertNotIn("TARGET", selection.selected)
        self.assertNotIn("CODE_GENDER", selection.selected)
        self.assertNotIn("DAYS_BIRTH", selection.selected)
        self.assertIn("AMT_CREDIT", selection.selected)


class ApplicationSplitAndMetricTests(unittest.TestCase):
    def test_split_assignment_is_deterministic_and_stratified(self) -> None:
        target = pd.Series([0] * 80 + [1] * 20)
        first = assign_splits(target, random_seed=42)
        second = assign_splits(target, random_seed=42)
        self.assertTrue(first.equals(second))
        self.assertEqual(
            first.value_counts().to_dict(), {"development": 70, "calibration": 15, "holdout": 15}
        )
        for split_name in ("development", "calibration", "holdout"):
            self.assertAlmostEqual(target[first.eq(split_name)].mean(), 0.2)

    def test_probability_metrics_reward_useful_ranking(self) -> None:
        target = pd.Series([0, 0, 0, 0, 1, 1])
        probability = np.array([0.05, 0.10, 0.15, 0.20, 0.70, 0.90])
        metrics = evaluate_probabilities(target, probability)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertGreater(metrics["brier_skill_score"], 0)


if __name__ == "__main__":
    unittest.main()
