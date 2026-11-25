#!/usr/bin/env python3
"""Verify pilot/full raw files against the saved row counts and SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for record in manifest["records"]:
        path = args.manifest.parent / record["data_file"]
        actual_hash = sha256_file(path)
        actual_rows = physical_row_count(path)
        expected_rows = record["rows_written"]
        valid = actual_hash == record["sha256"] and actual_rows == expected_rows
        print(
            f"{'OK' if valid else 'FAIL'} {record['repository']} "
            f"rows={actual_rows}/{expected_rows} sha256={actual_hash[:12]}"
        )
        if not valid:
            failures.append(record["repository"])
    if failures:
        print("integrity failures: " + ", ".join(failures))
        return 1
    print(f"verified {len(manifest['records'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

