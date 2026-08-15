import unittest

from credifast.modeling.benchmark import _latency_summary


class BenchmarkSummaryTests(unittest.TestCase):
    def test_latency_summary_reports_per_row_units(self) -> None:
        summary = _latency_summary([100.0, 200.0, 300.0], rows_per_run=100)
        self.assertEqual(summary["repetitions"], 3)
        self.assertEqual(summary["rows_per_run"], 100)
        self.assertAlmostEqual(summary["milliseconds_per_row_p50"], 2.0)
        self.assertAlmostEqual(summary["rows_per_second_at_p50"], 500.0)


if __name__ == "__main__":
    unittest.main()
