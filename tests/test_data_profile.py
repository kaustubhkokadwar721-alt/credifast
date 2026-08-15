import unittest
from pathlib import Path

from credifast.data.profile import profile_application_table

FIXTURE = Path(__file__).parent / "fixtures" / "application_train_sample.csv"


class ApplicationProfileTests(unittest.TestCase):
    def test_profile_finds_key_target_sentinel_and_amount_issues(self) -> None:
        profile = profile_application_table(FIXTURE)
        summary = profile["summary"]
        findings = {item["code"]: item for item in profile["findings"]}

        self.assertEqual(summary["row_count"], 5)
        self.assertEqual(summary["duplicate_key_count"], 1)
        self.assertEqual(summary["target_counts"], {"0": 2, "1": 1, "other": 2})
        self.assertAlmostEqual(summary["target_prevalence"], 0.2)
        self.assertFalse(summary["ready_for_modeling"])
        self.assertEqual(summary["maximum_finding_severity"], "critical")
        self.assertEqual(findings["DQ-APP-001"]["count"], 1)
        self.assertEqual(findings["DQ-APP-002"]["count"], 1)
        self.assertEqual(findings["DQ-APP-003"]["count"], 2)
        self.assertEqual(findings["DQ-APP-004"]["count"], 1)
        self.assertEqual(findings["DQ-APP-005"]["count"], 1)
        self.assertEqual(findings["DQ-APP-006"]["count"], 1)
        self.assertEqual(findings["DQ-APP-007"]["count"], 1)
        self.assertEqual(findings["DQ-APP-008"]["count"], 1)

    def test_profile_tracks_high_missingness(self) -> None:
        profile = profile_application_table(FIXTURE)
        high_missing = {
            item["column"]: item
            for item in profile["missingness"]["columns_at_or_above_50_percent"]
        }
        self.assertIn("EXT_SOURCE_1", high_missing)
        self.assertIn("EXT_SOURCE_3", high_missing)


if __name__ == "__main__":
    unittest.main()
