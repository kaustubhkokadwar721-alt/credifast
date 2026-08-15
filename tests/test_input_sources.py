import json
import unittest
from pathlib import Path

from credifast.input_sources import (
    build_input_source_schema,
    classify_model_factor,
    normalize_intake_package,
)


class InputSourceTests(unittest.TestCase):
    def test_factor_source_mapping_matches_frozen_inventory(self) -> None:
        inventory_path = Path(__file__).parents[1] / "artifacts/history_lightgbm_features.json"
        selected = json.loads(inventory_path.read_text(encoding="utf-8"))["selected"]

        schema = build_input_source_schema(selected)
        counts = {
            item["source"]: item["factor_count"] for item in schema["source_summary"]
        }

        self.assertEqual(schema["selected_factor_count"], 257)
        self.assertEqual(
            counts,
            {
                "cibil_report_derived": 48,
                "partner_ledger_derived": 102,
                "application_package": 88,
                "computed": 16,
                "external_score": 3,
            },
        )
        self.assertEqual(sum(counts.values()), 257)

    def test_factor_mapping_does_not_call_external_scores_cibil(self) -> None:
        self.assertEqual(classify_model_factor("BUREAU_DEBT_SUM"), "cibil_report_derived")
        self.assertEqual(classify_model_factor("INST_LATE_PAYMENT_RATE"), "partner_ledger_derived")
        self.assertEqual(classify_model_factor("CREDIT_TO_INCOME"), "computed")
        self.assertEqual(classify_model_factor("EXT_SOURCE_2"), "external_score")

    def test_json_intake_accepts_only_known_applicant_and_editable_terms(self) -> None:
        normalized = normalize_intake_package(
            {
                "application_id": 176483,
                "annual_income": 157500,
                "requested_credit": 440784.0,
            }
        )
        self.assertEqual(normalized["application_id"], 176483)
        self.assertEqual(normalized["annual_income"], 157500.0)

        with self.assertRaisesRegex(ValueError, "requires application_id"):
            normalize_intake_package({"annual_income": 100000})
        with self.assertRaisesRegex(ValueError, "unsupported intake fields"):
            normalize_intake_package({"application_id": 1, "age": 40})
        with self.assertRaisesRegex(ValueError, "positive and below"):
            normalize_intake_package({"application_id": 1, "annual_income": 0})


if __name__ == "__main__":
    unittest.main()
