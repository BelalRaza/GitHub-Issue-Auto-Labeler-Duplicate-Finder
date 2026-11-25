#!/usr/bin/env python3
"""Create an auditable cleaned issue table and provisional chronological splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from issue_triage.preparation import (  # noqa: E402
    assign_chronological_splits,
    build_quality_summary,
    flatten_issue,
    load_taxonomy,
    read_raw_issues,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    taxonomy = load_taxonomy(args.taxonomy)
    rows = [flatten_issue(wrapper, taxonomy) for wrapper in read_raw_issues(args.input)]
    rows.sort(key=lambda row: (row["repository"], row["issue_number"] or 0))
    assign_chronological_splits(rows)
    if not rows:
        raise SystemExit(f"No JSONL issue rows found in {args.input}")

    all_fields = list(rows[0].keys())
    write_csv(args.output / "issues_clean.csv", rows, all_fields)
    model_fields = [
        "repository",
        "issue_number",
        "url",
        "created_at",
        "title",
        "text_clean",
        "classification_target",
        "split",
    ]
    model_rows = [row for row in rows if row["classification_eligible"]]
    write_csv(args.output / "classification_model.csv", model_rows, model_fields)

    quality = build_quality_summary(rows)
    (args.output / "data_quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(quality, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

