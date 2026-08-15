import unittest

from credifast.model_runtime import LocalModelRuntime, RuntimePaths

RUNTIME_AVAILABLE = all(path.is_file() for path in RuntimePaths().required().values())


@unittest.skipUnless(RUNTIME_AVAILABLE, "local Kaggle data and ignored model artifacts are required")
class FrozenRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = LocalModelRuntime()

    def test_curated_profiles_preserve_frozen_routes_and_probability_ranges(self) -> None:
        expectations = {
            "strong_full": ("STANDARD_REVIEW", 0.00, 0.01),
            "boundary_full": ("BOUNDARY_REVIEW", 0.09, 0.11),
            "partial_history": ("DATA_LIMITED_REVIEW", 0.05, 0.09),
            "thin_history": ("DATA_LIMITED_REVIEW", 0.00, 0.02),
            "high_risk": ("HIGH_RISK_REVIEW", 0.70, 0.80),
            "affordability_stress": ("AFFORDABILITY_REVIEW", 0.00, 0.01),
            "store_outage": ("DATA_LIMITED_REVIEW", 0.00, 0.01),
        }
        for profile, (route, lower, upper) in expectations.items():
            with self.subTest(profile=profile):
                result = self.runtime.score_profile(profile, explain=False)
                probability = result["repayment_difficulty_probability"]
                self.assertEqual(result["review_route"], route)
                self.assertGreaterEqual(probability, lower)
                self.assertLess(probability, upper)

    def test_partial_history_uses_frozen_blend_and_store_outage_fails_closed(self) -> None:
        partial = self.runtime.score_profile("partial_history", explain=False)
        self.assertEqual(partial["history_segment"], "partial_2_to_3_sources")
        self.assertAlmostEqual(partial["history_weight"], 0.75)

        outage = self.runtime.score_profile("store_outage", explain=False)
        self.assertTrue(outage["feature_store_row_missing"])
        self.assertIn("FEATURE_STORE_ROW_MISSING", outage["data_quality_flags"])
        self.assertEqual(outage["review_route"], "DATA_LIMITED_REVIEW")


if __name__ == "__main__":
    unittest.main()
