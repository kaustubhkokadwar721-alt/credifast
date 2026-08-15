import unittest

import numpy as np

from credifast.modeling.gap_strategy import blend_probabilities, select_history_weight


class GapStrategyTests(unittest.TestCase):
    def test_selects_application_fallback_when_history_is_wrong(self) -> None:
        target = np.array([0, 0, 0, 1, 1, 1])
        application = np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.90])
        history = 1.0 - application
        weight, candidates = select_history_weight(target, application, history)
        self.assertEqual(weight, 0.0)
        self.assertEqual(len(candidates), 5)

    def test_blend_validates_shape_and_weight(self) -> None:
        with self.assertRaises(ValueError):
            blend_probabilities(np.array([0.1]), np.array([0.2, 0.3]), 0.5)
        with self.assertRaises(ValueError):
            blend_probabilities(np.array([0.1]), np.array([0.2]), 1.1)


if __name__ == "__main__":
    unittest.main()
