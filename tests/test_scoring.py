import unittest

from credifast.domain import ApplicantProfile
from credifast.scoring import probability_to_score, score_profile


def strong_profile() -> ApplicantProfile:
    return ApplicantProfile(
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


def risky_profile() -> ApplicantProfile:
    return ApplicantProfile(
        application_id="RISKY",
        requested_amount=900000,
        annual_income=600000,
        annual_annuity=240000,
        family_size=5,
        employment_years=0.5,
        existing_monthly_obligations=25000,
        active_accounts=8,
        total_outstanding=600000,
        credit_utilization=0.95,
        recent_inquiries=6,
        delinquent_accounts=3,
        bureau_history_months=8,
        data_coverage=0.80,
    )


class ScoringTests(unittest.TestCase):
    def test_risky_profile_has_higher_probability(self) -> None:
        strong = score_profile(strong_profile())
        risky = score_profile(risky_profile())
        self.assertLess(strong.probability, risky.probability)
        self.assertGreater(strong.credit_score, risky.credit_score)
        self.assertEqual(strong.risk_grade, "A")
        self.assertEqual(risky.risk_grade, "E")

    def test_score_decreases_when_probability_increases(self) -> None:
        self.assertGreater(probability_to_score(0.02), probability_to_score(0.10))

    def test_probability_range(self) -> None:
        result = score_profile(strong_profile())
        self.assertGreater(result.probability, 0)
        self.assertLess(result.probability, 1)
        self.assertGreaterEqual(result.credit_score, 300)
        self.assertLessEqual(result.credit_score, 850)


if __name__ == "__main__":
    unittest.main()
