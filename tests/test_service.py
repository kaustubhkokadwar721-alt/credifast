import unittest

from credifast.domain import ApplicantProfile
from credifast.service import evaluate_application


class ServiceTests(unittest.TestCase):
    def test_strong_profile_is_approved(self) -> None:
        profile = ApplicantProfile(
            application_id="STRONG",
            requested_amount=300000,
            annual_income=1800000,
            annual_annuity=60000,
            family_size=2,
            employment_years=10,
            existing_monthly_obligations=5000,
            active_accounts=2,
            total_outstanding=50000,
            credit_utilization=0.10,
            recent_inquiries=0,
            delinquent_accounts=0,
            bureau_history_months=120,
            data_coverage=1.0,
        )
        result = evaluate_application(profile)
        self.assertEqual(result["decision"]["recommendation"], "APPROVE")
        self.assertEqual(result["affordability"]["status"], "PASS")
        self.assertEqual(result["data_quality"]["scoring_path"], "FULL_FILE")

    def test_low_coverage_routes_to_manual_review(self) -> None:
        values = {
            "application_id": "THIN",
            "requested_amount": 200000,
            "annual_income": 800000,
            "annual_annuity": 60000,
            "family_size": 2,
            "employment_years": None,
            "existing_monthly_obligations": 5000,
            "active_accounts": 0,
            "total_outstanding": 0,
            "credit_utilization": None,
            "recent_inquiries": 0,
            "delinquent_accounts": 0,
            "bureau_history_months": None,
            "data_coverage": 0.50,
        }
        result = evaluate_application(values)
        self.assertEqual(result["decision"]["recommendation"], "MANUAL_REVIEW")
        self.assertEqual(result["decision"]["confidence"], "LOW")
        self.assertEqual(result["data_quality"]["scoring_path"], "THIN_FILE_DEMO")

    def test_affordability_failure_recommends_lower_amount(self) -> None:
        values = {
            "application_id": "AFFORDABILITY",
            "requested_amount": 300000,
            "annual_income": 900000,
            "annual_annuity": 120000,
            "family_size": 2,
            "employment_years": 8.0,
            "existing_monthly_obligations": 25000,
            "active_accounts": 2,
            "total_outstanding": 100000,
            "credit_utilization": 0.25,
            "recent_inquiries": 0,
            "delinquent_accounts": 0,
            "bureau_history_months": 60,
            "data_coverage": 1.0,
        }
        result = evaluate_application(values)
        self.assertEqual(result["affordability"]["status"], "FAIL")
        self.assertEqual(result["decision"]["recommendation"], "REDUCE_AMOUNT")


if __name__ == "__main__":
    unittest.main()
