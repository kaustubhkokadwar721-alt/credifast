import unittest

import numpy as np

from credifast.modeling.reasons import (
    aggregate_reason_contributions,
    reason_code_for_feature,
)


class ModelReasonTests(unittest.TestCase):
    def test_feature_mapping_prioritizes_missingness_before_generic_history(self) -> None:
        self.assertEqual(
            reason_code_for_feature("numeric__missingindicator_BUREAU_DEBT_SUM"),
            "ML07",
        )
        self.assertEqual(
            reason_code_for_feature("numeric__APPLICATION_MISSING_COUNT"),
            "ML07",
        )
        self.assertEqual(reason_code_for_feature("numeric__BB_DELINQUENT_MONTHS"), "ML01")

    def test_aggregation_returns_controlled_adverse_and_favorable_reasons(self) -> None:
        names = [
            "numeric__BB_DELINQUENT_MONTHS",
            "numeric__CC_UTILIZATION_MEAN",
            "numeric__EXT_SOURCE_MEAN",
            "numeric__APPLICATION_MISSING_COUNT",
        ]
        contributions = np.array([0.6, 0.3, -0.5, 0.1])
        result = aggregate_reason_contributions(names, contributions, top_k=2)
        self.assertEqual(result["adverse"][0]["code"], "ML01")
        self.assertEqual(result["favorable"][0]["code"], "ML06")
        self.assertEqual(result["explanation_space"], "LightGBM raw log odds before probability calibration")


if __name__ == "__main__":
    unittest.main()
