#!/usr/bin/env python3
"""Fetch the canonical (original) issue of every verified duplicate pair that our corpus lacks.

A duplicate can only be found if its original is in the search index. In production the index
holds a repository's full history. Our collection is a sample, so many originals are missing.
This adds them, one REST call each, in the same raw snapshot format with a checksum manifest.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from issue_triage.github_client import (  # noqa: E402
    GitHubClient,
    _atomic_write_jsonl,
    _safe_name,
    sha256_file,
    utc_now,
    write_manifest,
)

csv.field_size_limit(10**8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/processed/duplicate_pairs.csv")
    parser.add_argument("--issues", type=Path, default=ROOT / "data/processed/issues_clean.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/canonical")
    args = parser.parse_args()

    with args.issues.open(encoding="utf-8", newline="") as handle:
        have = {(r["repository"], int(r["issue_number"])) for r in csv.DictReader(handle)}
    with args.pairs.open(encoding="utf-8", newline="") as handle:
        wanted = sorted(
            {(r["canonical_repository"], int(r["canonical_issue_number"])) for r in csv.DictReader(handle)} - have
        )
    print(f"{len(wanted)} canonical issues missing from the corpus")

    client = GitHubClient()
    by_repo: dict[str, list[dict]] = {}
    collected_at = utc_now()
    skipped = 0
    for repository, number in wanted:
        payload, headers = client.get_json(f"https://api.github.com/repos/{repository}/issues/{number}")
        if "pull_request" in payload:
            skipped += 1
            continue
        by_repo.setdefault(repository, []).append(
            {"repository": repository, "collected_at": collected_at,
             "source_endpoint": "repos/{owner}/{repo}/issues/{number} (canonical of duplicate)", "issue": payload}
        )
        time.sleep(0.1)
    records = []
    for repository, rows in sorted(by_repo.items()):
        path = args.output / f"{_safe_name(repository)}.jsonl"
        count = _atomic_write_jsonl(path, rows)
        records.append({
            "repository": repository, "mode": "canonical", "rows_written": count, "collected_at": collected_at,
            "authenticated": bool(client.token), "data_file": path.name, "sha256": sha256_file(path),
            "rate_limit": {"remaining": headers.get("x-ratelimit-remaining")},
        })
    write_manifest(args.output, records)
    print(f"wrote {sum(r['rows_written'] for r in records)} canonical issues; skipped {skipped} pull requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
