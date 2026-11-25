#!/usr/bin/env python3
"""Collect public GitHub issues with provenance and rate-limit metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from issue_triage.github_client import (  # noqa: E402
    GitHubClient,
    collect_issue_pages,
    collect_labeled_issue_pages,
    collect_search_snapshot,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["search", "issues", "labeled"], required=True)
    parser.add_argument("--repositories", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--limit-repositories", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.repositories.read_text(encoding="utf-8"))
    repositories = [item["name"] for item in config["repositories"]]
    if args.limit_repositories:
        repositories = repositories[: args.limit_repositories]
    client = GitHubClient()
    taxonomy = None
    if args.mode == "labeled":
        if not args.taxonomy:
            raise SystemExit("--taxonomy is required for --mode labeled")
        taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    records = []
    for index, repository in enumerate(repositories, start=1):
        print(f"[{index}/{len(repositories)}] collecting {repository} ({args.mode})", flush=True)
        if args.mode == "search":
            record = collect_search_snapshot(client, repository, args.output, args.per_page)
        elif args.mode == "issues":
            record = collect_issue_pages(
                client,
                repository,
                args.output,
                max_pages=args.max_pages,
                per_page=args.per_page,
            )
        else:
            target_labels = taxonomy.get("repository_overrides", {}).get(repository, {})
            if not target_labels:
                raise SystemExit(f"No repository_overrides configured for {repository}")
            record = collect_labeled_issue_pages(
                client,
                repository,
                target_labels,
                args.output,
                max_pages=args.max_pages,
                per_page=args.per_page,
            )
        records.append(record)
        remaining = record.get("rate_limit", {}).get("remaining")
        print(f"  wrote {record['rows_written']} rows; rate-limit remaining={remaining}", flush=True)
    manifest = write_manifest(args.output, records)
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

