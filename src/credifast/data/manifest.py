"""Immutable source-file manifest generation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .contracts import DATASET_CONTRACTS, TableContract


def scan_file(path: Path, include_row_count: bool, chunk_size: int = 4 * 1024 * 1024) -> dict:
    """Hash a file and optionally count physical CSV records in one binary pass.

    Home Credit competition CSVs do not use embedded record newlines. The method is
    recorded in the manifest so the row-count assumption remains inspectable.
    """

    digest = hashlib.sha256()
    newline_count = 0
    final_byte = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            if include_row_count:
                newline_count += chunk.count(b"\n")
                final_byte = chunk[-1:]
    line_count = newline_count + (
        1 if include_row_count and path.stat().st_size and final_byte != b"\n" else 0
    )
    return {
        "sha256": digest.hexdigest(),
        "row_count": max(line_count - 1, 0) if include_row_count else None,
        "row_count_method": "physical_line_count" if include_row_count else None,
    }


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    return scan_file(path, include_row_count=False, chunk_size=chunk_size)["sha256"]


def _open_text(path: Path):
    encoding = (
        "iso-8859-1"
        if path.name.casefold() == "homecredit_columns_description.csv".casefold()
        else "utf-8-sig"
    )
    return path.open("r", encoding=encoding, newline="")


def read_header(path: Path) -> list[str]:
    with _open_text(path) as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _manifest_entry(path: Path, contract: TableContract, include_row_count: bool) -> dict:
    if not path.is_file():
        return {
            "file_name": contract.file_name,
            "table_name": contract.table_name,
            "required": contract.required_file,
            "status": "missing",
            "grain": contract.grain,
        }

    header = read_header(path)
    scan = scan_file(path, include_row_count=include_row_count)
    missing_columns = sorted(set(contract.required_columns) - set(header))
    column_count_valid = len(header) >= contract.min_columns
    schema_valid = not missing_columns and column_count_valid
    return {
        "file_name": contract.file_name,
        "table_name": contract.table_name,
        "required": contract.required_file,
        "status": "present",
        "bytes": path.stat().st_size,
        "sha256": scan["sha256"],
        "row_count": scan["row_count"],
        "row_count_method": scan["row_count_method"],
        "column_count": len(header),
        "columns": header,
        "schema_valid": schema_valid,
        "missing_required_columns": missing_columns,
        "minimum_column_count": contract.min_columns,
        "grain": contract.grain,
        "primary_key": list(contract.primary_key),
    }


def build_manifest(
    raw_directory: str | Path,
    *,
    include_row_count: bool = True,
    contracts: Iterable[TableContract] = DATASET_CONTRACTS,
) -> dict:
    raw_path = Path(raw_directory)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")

    entries = [
        _manifest_entry(raw_path / contract.file_name, contract, include_row_count)
        for contract in contracts
    ]
    missing_required = [
        entry["file_name"]
        for entry in entries
        if entry["required"] and entry["status"] == "missing"
    ]
    invalid_schemas = [
        entry["file_name"]
        for entry in entries
        if entry["status"] == "present" and not entry["schema_valid"]
    ]
    present = [entry for entry in entries if entry["status"] == "present"]

    return {
        "manifest_version": "1.0.0",
        "dataset": "Home Credit Default Risk",
        "kaggle_competition": "home-credit-default-risk",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_directory_label": "data/raw",
        "include_row_count": include_row_count,
        "summary": {
            "expected_files": len(entries),
            "present_files": len(present),
            "missing_required_files": missing_required,
            "invalid_schema_files": invalid_schemas,
            "total_bytes": sum(entry.get("bytes", 0) for entry in present),
            "ready_for_profiling": not missing_required and not invalid_schemas,
        },
        "files": entries,
    }


def write_manifest(manifest: dict, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination
