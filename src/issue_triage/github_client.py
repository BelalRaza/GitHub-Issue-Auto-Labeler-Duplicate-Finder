"""Small, dependency-free GitHub REST client with auditable collection metadata."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_API_VERSION = "2026-03-10"
USER_AGENT = "github-issue-auto-labeler-m2/0.1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GitHubClient:
    def __init__(self, token: str | None = None, api_version: str = DEFAULT_API_VERSION):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.api_version = api_version

    def get_json(self, url: str) -> tuple[Any, dict[str, str]]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": self.api_version,
            "User-Agent": USER_AGENT,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
                response_headers = {key.lower(): value for key, value in response.headers.items()}
                return payload, response_headers
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            remaining = exc.headers.get("x-ratelimit-remaining")
            reset = exc.headers.get("x-ratelimit-reset")
            detail = f"GitHub API returned HTTP {exc.code}: {body[:500]}"
            if remaining == "0" and reset:
                reset_at = datetime.fromtimestamp(int(reset), timezone.utc).isoformat()
                detail += f"; rate limit resets at {reset_at}"
            raise RuntimeError(detail) from exc


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _safe_name(repository: str) -> str:
    return repository.replace("/", "__")


def collect_search_snapshot(
    client: GitHubClient,
    repository: str,
    output_dir: Path,
    per_page: int = 100,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "q": f"repo:{repository} is:issue",
            "sort": "created",
            "order": "desc",
            "per_page": min(per_page, 100),
            "page": 1,
        }
    )
    url = f"https://api.github.com/search/issues?{query}"
    collected_at = utc_now()
    payload, headers = client.get_json(url)
    rows = (
        {
            "repository": repository,
            "collected_at": collected_at,
            "source_endpoint": "search/issues",
            "issue": issue,
        }
        for issue in payload.get("items", [])
    )
    data_path = output_dir / f"{_safe_name(repository)}.jsonl"
    row_count = _atomic_write_jsonl(data_path, rows)
    return {
        "repository": repository,
        "mode": "search",
        "query": f"repo:{repository} is:issue",
        "sort": "created desc",
        "sample_limit": per_page,
        "github_total_count": payload.get("total_count"),
        "incomplete_results": payload.get("incomplete_results"),
        "rows_written": row_count,
        "collected_at": collected_at,
        "api_version": client.api_version,
        "authenticated": bool(client.token),
        "rate_limit": {
            "limit": headers.get("x-ratelimit-limit"),
            "remaining": headers.get("x-ratelimit-remaining"),
            "reset": headers.get("x-ratelimit-reset"),
            "resource": headers.get("x-ratelimit-resource"),
        },
        "data_file": data_path.name,
        "sha256": sha256_file(data_path),
        "limitation": "Recent-issue pilot snapshot; not a representative full-history sample.",
    }


def collect_issue_pages(
    client: GitHubClient,
    repository: str,
    output_dir: Path,
    max_pages: int,
    per_page: int = 100,
    pause_seconds: float = 0.25,
) -> dict[str, Any]:
    collected_at = utc_now()
    all_rows: list[dict[str, Any]] = []
    rate_limit: dict[str, str | None] = {}
    pages_fetched = 0

    for page in range(1, max_pages + 1):
        query = urllib.parse.urlencode(
            {
                "state": "all",
                "sort": "created",
                "direction": "desc",
                "per_page": min(per_page, 100),
                "page": page,
            }
        )
        url = f"https://api.github.com/repos/{repository}/issues?{query}"
        payload, headers = client.get_json(url)
        pages_fetched += 1
        rate_limit = {
            "limit": headers.get("x-ratelimit-limit"),
            "remaining": headers.get("x-ratelimit-remaining"),
            "reset": headers.get("x-ratelimit-reset"),
            "resource": headers.get("x-ratelimit-resource"),
        }
        for issue in payload:
            if "pull_request" in issue:
                continue
            all_rows.append(
                {
                    "repository": repository,
                    "collected_at": collected_at,
                    "source_endpoint": "repos/{owner}/{repo}/issues",
                    "issue": issue,
                }
            )
        if len(payload) < per_page:
            break
        time.sleep(pause_seconds)

    data_path = output_dir / f"{_safe_name(repository)}.jsonl"
    row_count = _atomic_write_jsonl(data_path, all_rows)
    return {
        "repository": repository,
        "mode": "issues",
        "pages_fetched": pages_fetched,
        "per_page": per_page,
        "rows_written": row_count,
        "collected_at": collected_at,
        "api_version": client.api_version,
        "authenticated": bool(client.token),
        "rate_limit": rate_limit,
        "data_file": data_path.name,
        "sha256": sha256_file(data_path),
    }


def collect_labeled_issue_pages(
    client: GitHubClient,
    repository: str,
    target_labels: dict[str, list[str]],
    output_dir: Path,
    max_pages: int,
    per_page: int = 100,
    pause_seconds: float = 0.25,
) -> dict[str, Any]:
    """Collect each configured target label separately and deduplicate by issue ID."""
    collected_at = utc_now()
    issues_by_id: dict[int, dict[str, Any]] = {}
    collection_routes: dict[int, set[tuple[str, str]]] = {}
    request_log: list[dict[str, Any]] = []
    rate_limit: dict[str, str | None] = {}

    for target, labels in sorted(target_labels.items()):
        for label in labels:
            pages_fetched = 0
            rows_seen = 0
            for page in range(1, max_pages + 1):
                query = urllib.parse.urlencode(
                    {
                        "state": "all",
                        "labels": label,
                        "sort": "created",
                        "direction": "desc",
                        "per_page": min(per_page, 100),
                        "page": page,
                    }
                )
                url = f"https://api.github.com/repos/{repository}/issues?{query}"
                payload, headers = client.get_json(url)
                pages_fetched += 1
                rate_limit = {
                    "limit": headers.get("x-ratelimit-limit"),
                    "remaining": headers.get("x-ratelimit-remaining"),
                    "reset": headers.get("x-ratelimit-reset"),
                    "resource": headers.get("x-ratelimit-resource"),
                }
                for issue in payload:
                    if "pull_request" in issue:
                        continue
                    issue_id = int(issue["id"])
                    issues_by_id[issue_id] = issue
                    collection_routes.setdefault(issue_id, set()).add((target, label))
                    rows_seen += 1
                if len(payload) < per_page:
                    break
                time.sleep(pause_seconds)
            request_log.append(
                {
                    "target": target,
                    "label": label,
                    "pages_fetched": pages_fetched,
                    "issue_rows_seen": rows_seen,
                }
            )

    rows = []
    for issue_id, issue in sorted(issues_by_id.items(), key=lambda item: item[1].get("number", 0)):
        routes = sorted(collection_routes[issue_id])
        rows.append(
            {
                "repository": repository,
                "collected_at": collected_at,
                "source_endpoint": "repos/{owner}/{repo}/issues?labels=...",
                "collection_routes": [
                    {"target": target, "label": label} for target, label in routes
                ],
                "issue": issue,
            }
        )
    data_path = output_dir / f"{_safe_name(repository)}.jsonl"
    row_count = _atomic_write_jsonl(data_path, rows)
    return {
        "repository": repository,
        "mode": "labeled",
        "max_pages_per_label": max_pages,
        "per_page": per_page,
        "requests": request_log,
        "rows_written_after_deduplication": row_count,
        "rows_written": row_count,
        "collected_at": collected_at,
        "api_version": client.api_version,
        "authenticated": bool(client.token),
        "rate_limit": rate_limit,
        "data_file": data_path.name,
        "sha256": sha256_file(data_path),
    }


def write_manifest(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "collection_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    payload = {
        "generated_at": utc_now(),
        "records": records,
        "record_count": len(records),
    }
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path

