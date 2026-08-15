import unittest

import numpy as np

from credifast.modeling.calibration import calibration_diagnostics, compare_calibrators


class CalibrationTests(unittest.TestCase):
    def test_calibration_diagnostics_reward_well_calibrated_probabilities(self) -> None:
        target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        useful = np.array([0.05, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95])
        reversed_probability = 1.0 - useful
        useful_result = calibration_diagnostics(target, useful)
        reversed_result = calibration_diagnostics(target, reversed_probability)
        self.assertLess(useful_result["brier_score"], reversed_result["brier_score"])
        self.assertEqual(useful_result["rows"], 8)

    def test_candidate_comparison_uses_disjoint_fit_and_evaluation_rows(self) -> None:
        random = np.random.default_rng(42)
        raw = np.linspace(0.01, 0.99, 400)
        true_probability = 1.0 / (1.0 + np.exp(-(-1.0 + 1.8 * np.log(raw / (1 - raw)))))
        target = random.binomial(1, true_probability)
        fit_indices = np.arange(0, 400, 2)
        evaluation_indices = np.arange(1, 400, 2)
        selected, results, _ = compare_calibrators(
            raw,
            target,
            fit_indices=fit_indices,
            evaluation_indices=evaluation_indices,
        )
        self.assertIn(selected, {"identity", "sigmoid", "isotonic"})
        self.assertEqual(set(results), {"identity", "sigmoid", "isotonic"})
        for result in results.values():
            self.assertEqual(result["calibration"]["rows"], len(evaluation_indices))


if __name__ == "__main__":
    unittest.main()
