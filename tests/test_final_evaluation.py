import unittest

from credifast.modeling.final_evaluation import evaluate_acceptance_gates


class FinalEvaluationTests(unittest.TestCase):
    def test_acceptance_gates_use_predeclared_quality_targets(self) -> None:
        metrics = {
            "ranking": {"roc_auc": 0.78, "average_precision": 0.24},
            "calibration": {
                "base_rate": 0.08,
                "brier_score": 0.06,
                "calibration_slope": 1.02,
            },
        }
        result = evaluate_acceptance_gates(metrics)
        self.assertTrue(result["all_passed"])
        self.assertAlmostEqual(result["observed"]["average_precision_multiple_of_prevalence"], 3.0)

    def test_calibration_slope_failure_is_visible(self) -> None:
        metrics = {
            "ranking": {"roc_auc": 0.78, "average_precision": 0.24},
            "calibration": {
                "base_rate": 0.08,
                "brier_score": 0.06,
                "calibration_slope": 0.80,
            },
        }
        result = evaluate_acceptance_gates(metrics)
        self.assertFalse(result["all_passed"])
        self.assertFalse(result["checks"]["calibration_slope_between_0_90_and_1_10"])


if __name__ == "__main__":
    unittest.main()
