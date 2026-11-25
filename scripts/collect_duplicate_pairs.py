#!/usr/bin/env python3
"""Resolve canonical duplicate pairs through GitHub GraphQL's structured duplicateOf field."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.github.com/graphql"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issues", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=40)
    return parser.parse_args()


def build_query(rows: list[dict[str, str]]) -> tuple[str, dict[str, dict[str, str]]]:
    parts = ["query {"]
    aliases: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        alias = f"r{index}"
        owner, name = row["repository"].split("/", 1)
        number = int(row["issue_number"])
        aliases[alias] = row
        parts.append(
            f'''  {alias}: repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) {{
    issue(number: {number}) {{
      number url stateReason
      duplicateOf {{ number url repository {{ nameWithOwner }} }}
    }}
  }}'''
        )
    parts.append("}")
    return "\n".join(parts), aliases


def execute_query(query: str, token: str) -> dict:
    request = urllib.request.Request(
        API_URL,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "github-issue-auto-labeler-m2/0.1",
            "X-GitHub-Api-Version": "2026-03-10",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], indent=2))
    return payload["data"]


def parse_batch(data: dict, aliases: dict[str, dict[str, str]]) -> tuple[list[dict], list[dict]]:
    pairs: list[dict] = []
    unresolved: list[dict] = []
    for alias, source in aliases.items():
        issue = (data.get(alias) or {}).get("issue")
        duplicate_of = issue.get("duplicateOf") if issue else None
        if issue and issue.get("stateReason") == "DUPLICATE" and duplicate_of:
            pairs.append(
                {
                    "duplicate_repository": source["repository"],
                    "duplicate_issue_number": source["issue_number"],
                    "duplicate_url": issue["url"],
                    "duplicate_created_at": source["created_at"],
                    "canonical_repository": duplicate_of["repository"]["nameWithOwner"],
                    "canonical_issue_number": duplicate_of["number"],
                    "canonical_url": duplicate_of["url"],
                    "verification_method": "GitHub GraphQL Issue.duplicateOf",
                }
            )
        else:
            unresolved.append(
                {
                    "repository": source["repository"],
                    "issue_number": source["issue_number"],
                    "url": source["url"],
                    "reason": "missing issue, DUPLICATE stateReason, or duplicateOf reference",
                }
            )
    return pairs, unresolved


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required because GitHub GraphQL requires authentication")
    with args.issues.open(encoding="utf-8", newline="") as handle:
        candidates = [
            row for row in csv.DictReader(handle) if row["is_duplicate_verified"].lower() == "true"
        ]
    pairs: list[dict] = []
    unresolved: list[dict] = []
    for offset in range(0, len(candidates), args.batch_size):
        batch = candidates[offset : offset + args.batch_size]
        query, aliases = build_query(batch)
        batch_pairs, batch_unresolved = parse_batch(execute_query(query, token), aliases)
        pairs.extend(batch_pairs)
        unresolved.extend(batch_unresolved)
    pair_fields = [
        "duplicate_repository",
        "duplicate_issue_number",
        "duplicate_url",
        "duplicate_created_at",
        "canonical_repository",
        "canonical_issue_number",
        "canonical_url",
        "verification_method",
    ]
    write_csv(args.output, pairs, pair_fields)
    unresolved_path = args.output.with_name(args.output.stem + "_unresolved.csv")
    write_csv(unresolved_path, unresolved, ["repository", "issue_number", "url", "reason"])
    print(f"verified pairs={len(pairs)} unresolved={len(unresolved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

