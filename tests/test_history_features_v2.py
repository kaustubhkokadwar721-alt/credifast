import tempfile
import unittest
from pathlib import Path

import pandas as pd

from credifast.data.history_features_v2 import (
    build_temporal_history_features,
    validate_temporal_history_features,
)


class TemporalHistoryFeatureTests(unittest.TestCase):
    def test_temporal_features_preserve_grain_reconcile_payments_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [1]}).to_csv(
                root / "application_train.csv", index=False
            )
            pd.DataFrame({"SK_ID_CURR": [2]}).to_csv(
                root / "application_test.csv", index=False
            )
            pd.DataFrame(
                {
                    "SK_ID_CURR": [1],
                    "SK_ID_BUREAU": [10],
                    "CREDIT_ACTIVE": ["Active"],
                    "DAYS_CREDIT": [-30],
                    "AMT_CREDIT_SUM_DEBT": [500.0],
                    "AMT_CREDIT_SUM": [1000.0],
                    "AMT_ANNUITY": [100.0],
                    "AMT_CREDIT_SUM_OVERDUE": [0.0],
                }
            ).to_csv(root / "bureau.csv", index=False)
            pd.DataFrame(
                {"SK_ID_BUREAU": [10], "MONTHS_BALANCE": [-1], "STATUS": ["0"]}
            ).to_csv(root / "bureau_balance.csv", index=False)
            pd.DataFrame(
                {
                    "SK_ID_CURR": [1, 1],
                    "SK_ID_PREV": [20, 21],
                    "DAYS_DECISION": [-60, -10],
                    "NAME_CONTRACT_STATUS": ["Refused", "Approved"],
                    "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans"],
                    "NAME_CASH_LOAN_PURPOSE": ["XNA", "Payments on other loans"],
                    "AMT_ANNUITY": [50.0, 60.0],
                    "AMT_CREDIT": [500.0, 600.0],
                }
            ).to_csv(root / "previous_application.csv", index=False)
            pd.DataFrame(
                {
                    "SK_ID_CURR": [1, 1],
                    "SK_ID_PREV": [20, 20],
                    "MONTHS_BALANCE": [-2, -1],
                    "NAME_CONTRACT_STATUS": ["Active", "Active"],
                    "CNT_INSTALMENT_FUTURE": [3.0, 2.0],
                    "SK_DPD": [5, 0],
                }
            ).to_csv(root / "POS_CASH_balance.csv", index=False)
            pd.DataFrame(
                {
                    "SK_ID_CURR": [1, 1],
                    "SK_ID_PREV": [20, 20],
                    "MONTHS_BALANCE": [-2, -1],
                    "AMT_BALANCE": [900.0, 950.0],
                    "AMT_CREDIT_LIMIT_ACTUAL": [1000.0, 1000.0],
                    "AMT_PAYMENT_TOTAL_CURRENT": [40.0, 20.0],
                    "AMT_INST_MIN_REGULARITY": [50.0, 50.0],
                    "AMT_DRAWINGS_ATM_CURRENT": [100.0, 150.0],
                    "SK_DPD": [0, 2],
                }
            ).to_csv(root / "credit_card_balance.csv", index=False)
            pd.DataFrame(
                {
                    "SK_ID_CURR": [1, 1, 1, 1],
                    "SK_ID_PREV": [20, 20, 20, 20],
                    "NUM_INSTALMENT_NUMBER": [1, 2, 3, 3],
                    "DAYS_INSTALMENT": [-100, -70, -40, -40],
                    "DAYS_ENTRY_PAYMENT": [-95, -75, -35, -34],
                    "AMT_INSTALMENT": [100.0, 100.0, 100.0, 100.0],
                    "AMT_PAYMENT": [80.0, 100.0, 60.0, 40.0],
                }
            ).to_csv(root / "installments_payments.csv", index=False)

            feature_path = root / "processed" / "temporal.parquet"
            report = build_temporal_history_features(
                root,
                output_path=feature_path,
                report_path=root / "build_report.json",
                memory_limit="1GB",
                threads=1,
            )
            features = pd.read_parquet(feature_path).set_index("SK_ID_CURR")
            quality = validate_temporal_history_features(
                feature_path,
                root,
                output_path=root / "quality_report.json",
            )

            self.assertEqual(report["row_count"], 2)
            self.assertEqual(report["feature_count_excluding_key"], 111)
            self.assertNotIn("TARGET", features.columns)
            self.assertEqual(features.index.nunique(), 2)
            applicant = features.loc[1]
            self.assertEqual(applicant["TINST_EVENT_COUNT_365D"], 3)
            self.assertEqual(applicant["TINST_SHORT_COUNT_90D"], 0)
            self.assertEqual(applicant["TINST_RECENT_EVENT_1_LATE"], 1)
            self.assertEqual(applicant["TINST_RECENT_EVENT_3_SHORT"], 1)
            self.assertGreater(
                applicant["TINST_WEIGHTED_LATE_RATE_90D"],
                applicant["TINST_WEIGHTED_LATE_RATE_365D"],
            )
            self.assertEqual(applicant["CYCLING_MULTI_SIGNAL_FLAG"], 1)
            self.assertEqual(features.loc[2, "HAS_TEMPORAL_INSTALLMENTS"], 0)
            self.assertTrue(quality["summary"]["ready_for_experiment"])
            self.assertEqual(quality["findings"], [])


if __name__ == "__main__":
    unittest.main()
