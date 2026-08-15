import unittest

from credifast.model_runtime import affordability_context, validate_overrides


class ModelRuntimeTests(unittest.TestCase):
    def test_override_validation_rejects_unknown_and_nonpositive_values(self) -> None:
        self.assertEqual(validate_overrides({"annual_income": 100_000.0}), {"annual_income": 100_000.0})
        with self.assertRaisesRegex(ValueError, "unsupported override"):
            validate_overrides({"age": 35.0})
        with self.assertRaisesRegex(ValueError, "must be positive"):
            validate_overrides({"annual_annuity": 0.0})

    def test_affordability_screen_is_transparent_and_separate(self) -> None:
        result = affordability_context(
            annual_income=100_000.0,
            annual_annuity=50_000.0,
            requested_credit=200_000.0,
        )
        self.assertEqual(result["status"], "SCREEN_REVIEW")
        self.assertAlmostEqual(result["proposed_repayment_to_income"], 0.5)
        self.assertAlmostEqual(result["estimated_credit_at_maximum_ratio"], 160_000.0)


if __name__ == "__main__":
    unittest.main()
