import tempfile
import unittest
from pathlib import Path
from shutil import copyfile

from credifast.data.contracts import TableContract
from credifast.data.manifest import build_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "application_train_sample.csv"


class DataManifestTests(unittest.TestCase):
    def test_manifest_hashes_counts_and_validates_a_present_file(self) -> None:
        contract = TableContract(
            file_name="application_train.csv",
            table_name="application_train",
            grain="one application",
            primary_key=("SK_ID_CURR",),
            required_columns=("SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL"),
            min_columns=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            copyfile(FIXTURE, raw / contract.file_name)
            manifest = build_manifest(raw, contracts=(contract,))

        entry = manifest["files"][0]
        self.assertEqual(entry["status"], "present")
        self.assertEqual(entry["row_count"], 5)
        self.assertEqual(len(entry["sha256"]), 64)
        self.assertTrue(entry["schema_valid"])
        self.assertTrue(manifest["summary"]["ready_for_profiling"])

    def test_manifest_reports_missing_required_file(self) -> None:
        contract = TableContract(
            file_name="missing.csv",
            table_name="missing",
            grain="one row",
            primary_key=(),
            required_columns=("id",),
            min_columns=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_manifest(directory, contracts=(contract,))

        self.assertEqual(manifest["summary"]["missing_required_files"], ["missing.csv"])
        self.assertFalse(manifest["summary"]["ready_for_profiling"])


if __name__ == "__main__":
    unittest.main()
