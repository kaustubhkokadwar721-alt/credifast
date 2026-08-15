import unittest

from credifast.domain import ApplicantProfile


class ApplicantProfileTests(unittest.TestCase):
    def test_rejects_invalid_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "data_coverage"):
            ApplicantProfile(
                application_id="X",
                requested_amount=100000,
                annual_income=500000,
                annual_annuity=50000,
                family_size=1,
                employment_years=2,
                existing_monthly_obligations=0,
                active_accounts=0,
                total_outstanding=0,
                credit_utilization=0,
                recent_inquiries=0,
                delinquent_accounts=0,
                bureau_history_months=0,
                data_coverage=1.1,
            )

    def test_reports_missing_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            ApplicantProfile.from_mapping({"application_id": "X"})


if __name__ == "__main__":
    unittest.main()
