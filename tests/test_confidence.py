import unittest

from credifast.modeling.confidence import assess_history_confidence


class HistoryConfidenceTests(unittest.TestCase):
    def test_partial_history_forces_manual_review_without_changing_risk(self) -> None:
        assessment = assess_history_confidence(
            0.08,
            {
                "HAS_BUREAU_HISTORY": 1,
                "HAS_PREVIOUS_APPLICATION": 1,
                "HAS_POS_HISTORY": 1,
                "HAS_CREDIT_CARD_HISTORY": 0,
                "HAS_INSTALLMENT_HISTORY": 0,
            },
        )
        self.assertEqual(assessment.history_segment, "partial_2_to_3_sources")
        self.assertEqual(assessment.confidence, "LOW")
        self.assertEqual(assessment.route, "MANUAL_REVIEW_DATA_LIMITED")

    def test_full_history_away_from_boundary_uses_standard_route(self) -> None:
        assessment = assess_history_confidence(
            0.35,
            {
                "HAS_BUREAU_HISTORY": 1,
                "HAS_PREVIOUS_APPLICATION": 1,
                "HAS_POS_HISTORY": 1,
                "HAS_CREDIT_CARD_HISTORY": 1,
                "HAS_INSTALLMENT_HISTORY": 1,
            },
        )
        self.assertEqual(assessment.confidence, "HIGH")
        self.assertEqual(assessment.route, "STANDARD_REVIEW")

    def test_full_history_near_boundary_routes_to_review(self) -> None:
        assessment = assess_history_confidence(
            0.101,
            {
                "HAS_BUREAU_HISTORY": 1,
                "HAS_PREVIOUS_APPLICATION": 1,
                "HAS_POS_HISTORY": 1,
                "HAS_CREDIT_CARD_HISTORY": 0,
                "HAS_INSTALLMENT_HISTORY": 1,
            },
        )
        self.assertEqual(assessment.confidence, "MEDIUM")
        self.assertEqual(assessment.route, "MANUAL_REVIEW_NEAR_RISK_BOUNDARY")


if __name__ == "__main__":
    unittest.main()
