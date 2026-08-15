import unittest

import numpy as np

from credifast.modeling.fairness import group_fairness_metrics


class FairnessMetricTests(unittest.TestCase):
    def test_capacity_metrics_use_group_denominators(self) -> None:
        target = np.array([1, 1, 0, 0, 0, 0])
        probability = np.array([0.9, 0.2, 0.8, 0.7, 0.1, 0.05])
        metrics = group_fairness_metrics(
            target,
            probability,
            review_threshold=0.75,
            minimum_rows=1,
            minimum_events=1,
        )
        self.assertAlmostEqual(metrics["review_rate"], 2 / 6)
        self.assertAlmostEqual(metrics["true_positive_rate_at_review_capacity"], 1 / 2)
        self.assertAlmostEqual(metrics["false_positive_rate_at_review_capacity"], 1 / 4)
        self.assertTrue(metrics["eligible_for_comparison"])

    def test_small_groups_suppress_unstable_model_metrics(self) -> None:
        metrics = group_fairness_metrics(
            np.array([0, 0, 1]),
            np.array([0.1, 0.2, 0.8]),
            review_threshold=0.5,
        )
        self.assertFalse(metrics["eligible_for_comparison"])
        self.assertIn("roc_auc", metrics["suppressed_metrics"])


if __name__ == "__main__":
    unittest.main()
