import tempfile
import unittest
from pathlib import Path

import pandas as pd

from credifast.data.history_features import build_history_features


class HistoryFeatureTests(unittest.TestCase):
    def test_aggregation_preserves_application_grain_and_target_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({"SK_ID_CURR": [100001], "TARGET": [1]}).to_csv(
                root / "application_train.csv", index=False
            )
            pd.DataFrame({"SK_ID_CURR": [100002]}).to_csv(
                root / "application_test.csv", index=False
            )
            pd.DataFrame(
                {
                    "SK_ID_CURR": [100001],
                    "SK_ID_BUREAU": [200001],
                    "CREDIT_ACTIVE": ["Active"],
                    "CREDIT_TYPE": ["Consumer credit"],
                    "DAYS_CREDIT": [-100],
                    "CREDIT_DAY_OVERDUE": [2],
                    "CNT_CREDIT_PROLONG": [0],
                    "AMT_CREDIT_SUM": [1000.0],
                    "AMT_CREDIT_SUM_DEBT": [250.0],
                    "AMT_CREDIT_SUM_OVERDUE": [10.0],
                    "AMT_CREDIT_SUM_LIMIT": [0.0],
                    "AMT_ANNUITY": [100.0],
                }
            ).to_csv(root / "bureau.csv", index=False)
            pd.DataFrame(
                {
                    "SK_ID_BUREAU": [200001, 200001],
                    "MONTHS_BALANCE": [-1, -2],
                    "STATUS": ["1", "C"],
                }
            ).to_csv(root / "bureau_balance.csv", index=False)
            pd.DataFrame(
                {
                    "SK_ID_PREV": [300001],
                    "SK_ID_CURR": [100001],
                    "NAME_CONTRACT_STATUS": ["Approved"],
                    "DAYS_DECISION": [-200],
                    "AMT_APPLICATION": [500.0],
                    "AMT_CREDIT": [450.0],
                    "AMT_ANNUITY": [50.0],
                    "AMT_DOWN_PAYMENT": [50.0],
                    "RATE_DOWN_PAYMENT": [0.1],
                    "CNT_PAYMENT": [10.0],
                    "NAME_YIELD_GROUP": ["middle"],
                    "NAME_CONTRACT_TYPE": ["Consumer loans"],
                }
            ).to_csv(root / "previous_application.csv", index=False)
            pd.DataFrame(
                {
                    "SK_ID_PREV": [300001],
                    "SK_ID_CURR": [100001],
                    "MONTHS_BALANCE": [-1],
                    "CNT_INSTALMENT": [10.0],
                    "CNT_INSTALMENT_FUTURE": [5.0],
                    "NAME_CONTRACT_STATUS": ["Active"],
                    "SK_DPD": [3],
                    "SK_DPD_DEF": [1],
                }
            ).to_csv(root / "POS_CASH_balance.csv", index=False)
            pd.DataFrame(
                {
                    "SK_ID_PREV": [300001],
                    "SK_ID_CURR": [100001],
                    "MONTHS_BALANCE": [-1],
                    "AMT_BALANCE": [200.0],
                    "AMT_CREDIT_LIMIT_ACTUAL": [1000.0],
                    "AMT_DRAWINGS_CURRENT": [50.0],
                    "AMT_DRAWINGS_ATM_CURRENT": [20.0],
                    "AMT_DRAWINGS_POS_CURRENT": [30.0],
                    "AMT_PAYMENT_CURRENT": [40.0],
                    "AMT_PAYMENT_TOTAL_CURRENT": [45.0],
                    "AMT_INST_MIN_REGULARITY": [25.0],
                    "AMT_TOTAL_RECEIVABLE": [210.0],
                    "CNT_DRAWINGS_CURRENT": [2],
                    "CNT_DRAWINGS_ATM_CURRENT": [1],
                    "CNT_DRAWINGS_POS_CURRENT": [1],
                    "NAME_CONTRACT_STATUS": ["Active"],
                    "SK_DPD": [0],
                    "SK_DPD_DEF": [0],
                }
            ).to_csv(root / "credit_card_balance.csv", index=False)
            pd.DataFrame(
                {
                    "SK_ID_PREV": [300001, 300001],
                    "SK_ID_CURR": [100001, 100001],
                    "NUM_INSTALMENT_VERSION": [1.0, 1.0],
                    "NUM_INSTALMENT_NUMBER": [1, 2],
                    "DAYS_INSTALMENT": [-100, -70],
                    "DAYS_ENTRY_PAYMENT": [-98, -75],
                    "AMT_INSTALMENT": [100.0, 100.0],
                    "AMT_PAYMENT": [90.0, 100.0],
                }
            ).to_csv(root / "installments_payments.csv", index=False)

            output = root / "processed" / "history.parquet"
            report_path = root / "history_report.json"
            report = build_history_features(
                root,
                output_path=output,
                report_path=report_path,
                memory_limit="1GB",
                threads=1,
            )
            features = pd.read_parquet(output)

            self.assertEqual(report["row_count"], 2)
            self.assertEqual(report["duplicate_keys"], 0)
            self.assertFalse(report["target_used"])
            self.assertNotIn("TARGET", features.columns)
            self.assertEqual(features["SK_ID_CURR"].nunique(), 2)
            first = features.set_index("SK_ID_CURR").loc[100001]
            self.assertEqual(first["BUREAU_CREDIT_COUNT"], 1)
            self.assertEqual(first["BB_DELINQUENT_MONTHS"], 1)
            self.assertEqual(first["INST_LATE_PAYMENT_COUNT"], 1)
            self.assertAlmostEqual(first["CC_UTILIZATION_MEAN"], 0.2)
            second = features.set_index("SK_ID_CURR").loc[100002]
            self.assertEqual(second["HAS_BUREAU_HISTORY"], 0)


if __name__ == "__main__":
    unittest.main()
