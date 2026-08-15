"""CLI for CrediFast Phase 1 data inventory and profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.contracts import DATASET_CONTRACTS
from .data.history_features import build_history_features
from .data.history_features_v2 import (
    build_temporal_history_features,
    validate_temporal_history_features,
)
from .data.history_quality import validate_history_features
from .data.manifest import build_manifest, write_manifest
from .data.profile import profile_application_table, write_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CrediFast data-quality utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    contracts = subparsers.add_parser("contracts", help="Print expected dataset contracts")
    contracts.add_argument("--output", type=Path)

    manifest = subparsers.add_parser("manifest", help="Build an immutable source manifest")
    manifest.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    manifest.add_argument("--output", type=Path, default=Path("artifacts/data_manifest.json"))
    manifest.add_argument(
        "--skip-row-count",
        action="store_true",
        help="Hash and validate headers without scanning every CSV row",
    )

    profile = subparsers.add_parser("profile-application", help="Profile application_train.csv")
    profile.add_argument("--input", type=Path, default=Path("data/raw/application_train.csv"))
    profile.add_argument("--output", type=Path, default=Path("artifacts/application_profile.json"))

    history = subparsers.add_parser(
        "aggregate-history", help="Build target-free application-level history features"
    )
    history.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    history.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/history_features.parquet"),
    )
    history.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/history_features_report.json"),
    )
    history.add_argument("--memory-limit", default="4GB")
    history.add_argument("--threads", type=int, default=4)

    quality = subparsers.add_parser(
        "validate-history", help="Run the modeling-readiness gate on history features"
    )
    quality.add_argument(
        "--input", type=Path, default=Path("data/processed/history_features.parquet")
    )
    quality.add_argument(
        "--application-train", type=Path, default=Path("data/raw/application_train.csv")
    )
    quality.add_argument(
        "--application-test", type=Path, default=Path("data/raw/application_test.csv")
    )
    quality.add_argument(
        "--output", type=Path, default=Path("artifacts/history_quality_report.json")
    )

    temporal = subparsers.add_parser(
        "aggregate-history-v2",
        help="Build additive temporal, burden, and credit-cycling challenger features",
    )
    temporal.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    temporal.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/history_temporal_features.parquet"),
    )
    temporal.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/history_temporal_features_report.json"),
    )
    temporal.add_argument("--memory-limit", default="4GB")
    temporal.add_argument("--threads", type=int, default=4)

    temporal_quality = subparsers.add_parser(
        "validate-history-v2", help="Validate additive temporal challenger features"
    )
    temporal_quality.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/history_temporal_features.parquet"),
    )
    temporal_quality.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    temporal_quality.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/history_temporal_quality_report.json"),
    )
    return parser


def _emit(payload: object, output: Path | None) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "contracts":
        _emit([contract.to_dict() for contract in DATASET_CONTRACTS], args.output)
    elif args.command == "manifest":
        manifest = build_manifest(args.raw_dir, include_row_count=not args.skip_row_count)
        destination = write_manifest(manifest, args.output)
        print(
            f"Wrote {destination}; present={manifest['summary']['present_files']} "
            f"missing_required={len(manifest['summary']['missing_required_files'])} "
            f"invalid_schema={len(manifest['summary']['invalid_schema_files'])}"
        )
    elif args.command == "profile-application":
        profile = profile_application_table(args.input)
        destination = write_profile(profile, args.output)
        print(
            f"Wrote {destination}; rows={profile['summary']['row_count']} "
            f"target_prevalence={profile['summary']['target_prevalence']} "
            f"ready_for_modeling={profile['summary']['ready_for_modeling']}"
        )
    elif args.command == "aggregate-history":
        report = build_history_features(
            args.raw_dir,
            output_path=args.output,
            report_path=args.report_output,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        print(
            "History aggregation complete: "
            f"rows={report['row_count']} "
            f"features={report['feature_count_excluding_key']} "
            f"duplicate_keys={report['duplicate_keys']}"
        )
    elif args.command == "validate-history":
        report = validate_history_features(
            args.input,
            args.application_train,
            args.application_test,
            output_path=args.output,
        )
        print(
            "History quality gate complete: "
            f"ready_for_modeling={report['summary']['ready_for_modeling']} "
            f"maximum_severity={report['summary']['maximum_finding_severity']}"
        )
    elif args.command == "aggregate-history-v2":
        report = build_temporal_history_features(
            args.raw_dir,
            output_path=args.output,
            report_path=args.report_output,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        print(
            "Temporal history aggregation complete: "
            f"rows={report['row_count']} "
            f"features={report['feature_count_excluding_key']} "
            f"duplicate_keys={report['duplicate_keys']}"
        )
    elif args.command == "validate-history-v2":
        report = validate_temporal_history_features(
            args.input,
            args.raw_dir,
            output_path=args.output,
        )
        print(
            "Temporal history quality gate complete: "
            f"ready_for_experiment={report['summary']['ready_for_experiment']} "
            f"findings={len(report['findings'])}"
        )


if __name__ == "__main__":
    main()
