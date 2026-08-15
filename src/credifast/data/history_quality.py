"""Quality gate for the application-grain history feature store."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def validate_history_features(
    feature_path: str | Path,
    application_train_path: str | Path,
    application_test_path: str | Path,
    *,
    output_path: str | Path,
) -> dict:
    """Check grain, completeness, ranges, finite values, and modeling coverage."""

    feature_source = Path(feature_path)
    train_source = Path(application_train_path)
    test_source = Path(application_test_path)
    for path in (feature_source, train_source, test_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    feature_sql = f"read_parquet('{_sql_path(feature_source)}')"
    train_sql = f"read_csv_auto('{_sql_path(train_source)}', header=true, sample_size=200000)"
    test_sql = f"read_csv_auto('{_sql_path(test_source)}', header=true, sample_size=200000)"
    connection = duckdb.connect()
    try:
        schema_rows = connection.execute(f"DESCRIBE SELECT * FROM {feature_sql}").fetchall()
        columns = [row[0] for row in schema_rows]
        numeric_columns = [
            row[0]
            for row in schema_rows
            if any(token in row[1].upper() for token in ("INT", "DOUBLE", "FLOAT", "DECIMAL"))
        ]
        cardinality = connection.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(DISTINCT SK_ID_CURR),
                COUNT(*) FILTER (WHERE SK_ID_CURR IS NULL),
                (SELECT COUNT(*) FROM {train_sql}),
                (SELECT COUNT(*) FROM {test_sql})
            FROM {feature_sql}
            """
        ).fetchone()
        if cardinality is None:
            raise ValueError("could not profile history features")

        null_expressions = [
            f"SUM(CASE WHEN {_identifier(column)} IS NULL THEN 1 ELSE 0 END) AS {_identifier(column)}"
            for column in columns
        ]
        null_counts = connection.execute(
            f"SELECT {', '.join(null_expressions)} FROM {feature_sql}"
        ).fetchone()
        missingness = sorted(
            (
                {
                    "feature": column,
                    "null_count": int(null_counts[index]),
                    "null_rate": float(null_counts[index] / cardinality[0]),
                }
                for index, column in enumerate(columns)
                if column != "SK_ID_CURR"
            ),
            key=lambda item: item["null_rate"],
            reverse=True,
        )

        rate_columns = [column for column in numeric_columns if column.endswith("_RATE")]
        rate_checks = {}
        for column in rate_columns:
            minimum, maximum, invalid = connection.execute(
                f"""
                SELECT
                    MIN({_identifier(column)}),
                    MAX({_identifier(column)}),
                    COUNT(*) FILTER (
                        WHERE {_identifier(column)} < 0 OR {_identifier(column)} > 1
                    )
                FROM {feature_sql}
                """
            ).fetchone()
            rate_checks[column] = {
                "minimum": None if minimum is None else float(minimum),
                "maximum": None if maximum is None else float(maximum),
                "invalid_rows": int(invalid),
            }

        count_columns = [
            column
            for column in numeric_columns
            if "COUNT" in column
            or column.endswith("_MONTHS_OBSERVED")
            or column.startswith("HAS_")
        ]
        negative_counts = {}
        for column in count_columns:
            minimum, invalid = connection.execute(
                f"""
                SELECT MIN({_identifier(column)}),
                       COUNT(*) FILTER (WHERE {_identifier(column)} < 0)
                FROM {feature_sql}
                """
            ).fetchone()
            if invalid:
                negative_counts[column] = {
                    "minimum": float(minimum),
                    "invalid_rows": int(invalid),
                }

        finite_expressions = [
            f"SUM(CASE WHEN {_identifier(column)} IS NOT NULL "
            f"AND NOT isfinite({_identifier(column)}) THEN 1 ELSE 0 END)"
            for column in numeric_columns
        ]
        finite_counts = connection.execute(
            f"SELECT {', '.join(finite_expressions)} FROM {feature_sql}"
        ).fetchone()
        non_finite = {
            column: int(finite_counts[index])
            for index, column in enumerate(numeric_columns)
            if finite_counts[index]
        }

        availability_flags = [column for column in columns if column.startswith("HAS_")]
        coverage = {}
        for flag in availability_flags:
            rows, rate = connection.execute(
                f"SELECT SUM({_identifier(flag)}), AVG({_identifier(flag)}) FROM {feature_sql}"
            ).fetchone()
            coverage[flag] = {"rows": int(rows), "rate": float(rate)}

        target_by_coverage = {}
        for flag in availability_flags:
            target_rows = connection.execute(
                f"""
                SELECT h.{_identifier(flag)}, COUNT(*) AS rows, AVG(a.TARGET) AS target_rate
                FROM {train_sql} a
                JOIN {feature_sql} h USING (SK_ID_CURR)
                GROUP BY h.{_identifier(flag)}
                ORDER BY h.{_identifier(flag)}
                """
            ).fetchall()
            target_by_coverage[flag] = [
                {"available": int(row[0]), "rows": int(row[1]), "target_rate": float(row[2])}
                for row in target_rows
            ]
    finally:
        connection.close()

    findings = []
    expected_rows = int(cardinality[3] + cardinality[4])
    if cardinality[0] != expected_rows:
        findings.append(
            {
                "severity": "critical",
                "check": "application_row_reconciliation",
                "evidence": {"expected": expected_rows, "actual": int(cardinality[0])},
                "risk": "training or scoring populations would be silently lost or duplicated",
            }
        )
    if cardinality[0] != cardinality[1] or cardinality[2]:
        findings.append(
            {
                "severity": "critical",
                "check": "primary_key_uniqueness",
                "evidence": {
                    "rows": int(cardinality[0]),
                    "unique_keys": int(cardinality[1]),
                    "null_keys": int(cardinality[2]),
                },
                "risk": "the application-grain contract is broken",
            }
        )
    invalid_rates = {
        feature: result for feature, result in rate_checks.items() if result["invalid_rows"]
    }
    if invalid_rates:
        findings.append(
            {
                "severity": "high",
                "check": "bounded_rate_validity",
                "evidence": invalid_rates,
                "risk": "invalid aggregates can distort model splits and explanations",
            }
        )
    if negative_counts:
        findings.append(
            {
                "severity": "high",
                "check": "nonnegative_count_validity",
                "evidence": negative_counts,
                "risk": "negative event counts indicate an aggregation or type error",
            }
        )
    if non_finite:
        findings.append(
            {
                "severity": "high",
                "check": "finite_numeric_values",
                "evidence": non_finite,
                "risk": "infinite feature values can break preprocessing and model fitting",
            }
        )

    maximum_severity = "none"
    for severity in ("critical", "high", "medium", "low"):
        if any(finding["severity"] == severity for finding in findings):
            maximum_severity = severity
            break
    report = {
        "run_version": "1.0.0",
        "dataset": "history_features.parquet",
        "intended_use": "history-enriched credit-risk model development",
        "grain": "one row per train or test SK_ID_CURR",
        "summary": {
            "row_count": int(cardinality[0]),
            "column_count": len(columns),
            "feature_count_excluding_key": len(columns) - 1,
            "unique_keys": int(cardinality[1]),
            "null_keys": int(cardinality[2]),
            "expected_application_rows": expected_rows,
            "maximum_finding_severity": maximum_severity,
            "ready_for_modeling": maximum_severity not in {"critical", "high"},
        },
        "checks": {
            "target_column_absent": "TARGET" not in columns,
            "rate_bounds": rate_checks,
            "negative_count_features": negative_counts,
            "non_finite_features": non_finite,
            "source_coverage": coverage,
            "target_rate_by_source_availability": target_by_coverage,
            "top_missing_features": missingness[:20],
        },
        "findings": findings,
        "interpretation": [
            "Nulls in a source-specific feature block are expected when the corresponding HAS_* flag is zero.",
            "Credit-card history has structurally lower coverage than other histories and should not be imputed as evidence of zero debt.",
            "Source-availability flags are retained because absence of history is operationally distinct from a measured zero.",
        ],
        "automated_gate": {
            "required": [
                "row count equals application_train plus application_test",
                "SK_ID_CURR is non-null and unique",
                "TARGET is absent",
                "all count features are nonnegative",
                "all rate features are within zero and one",
                "all populated numeric values are finite",
            ],
            "passed": maximum_severity not in {"critical", "high"} and "TARGET" not in columns,
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
