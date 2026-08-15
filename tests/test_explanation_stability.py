import unittest

from credifast.modeling.explanation_stability import _jaccard


class ExplanationStabilityTests(unittest.TestCase):
    def test_jaccard_handles_overlap_and_empty_sets(self) -> None:
        self.assertAlmostEqual(_jaccard({"A", "B"}, {"B", "C"}), 1 / 3)
        self.assertEqual(_jaccard(set(), set()), 1.0)


if __name__ == "__main__":
    unittest.main()
