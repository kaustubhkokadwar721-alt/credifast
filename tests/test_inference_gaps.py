import unittest

import pandas as pd

from credifast.modeling.inference import merge_history_with_gaps


class InferenceGapTests(unittest.TestCase):
    def test_missing_store_row_keeps_values_missing_and_sets_availability_to_zero(self) -> None:
        application = pd.DataFrame({"SK_ID_CURR": [1, 2], "AMT_INCOME_TOTAL": [100.0, 200.0]})
        history = pd.DataFrame(
            {
                "SK_ID_CURR": [1],
                "HAS_BUREAU_HISTORY": [1],
                "BUREAU_DEBT_SUM": [50.0],
            }
        )
        frame, flags, missing = merge_history_with_gaps(
            application,
            history,
            ["HAS_BUREAU_HISTORY", "BUREAU_DEBT_SUM"],
        )
        self.assertEqual(flags, ["HAS_BUREAU_HISTORY"])
        self.assertFalse(bool(missing.iloc[0]))
        self.assertTrue(bool(missing.iloc[1]))
        self.assertEqual(frame.loc[1, "HAS_BUREAU_HISTORY"], 0)
        self.assertTrue(pd.isna(frame.loc[1, "BUREAU_DEBT_SUM"]))

    def test_schema_gap_fails_closed(self) -> None:
        application = pd.DataFrame({"SK_ID_CURR": [1]})
        history = pd.DataFrame({"SK_ID_CURR": [1], "HAS_BUREAU_HISTORY": [1]})
        with self.assertRaisesRegex(ValueError, "schema is missing"):
            merge_history_with_gaps(
                application,
                history,
                ["HAS_BUREAU_HISTORY", "BUREAU_DEBT_SUM"],
            )


if __name__ == "__main__":
    unittest.main()
