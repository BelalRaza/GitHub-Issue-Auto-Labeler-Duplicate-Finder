"""SQLite store for the repositories the app works with and their issues.

Two tables:
    repositories  one row per "owner/repo"
    issues        one row per issue; the full issue is kept as JSON in `data`, the bot's last
                  suggestion as JSON in `prediction`. Deleting a repository deletes its issues.

This store is the bot's memory for duplicate search: a new issue is compared with the issues
stored for its repository. You can add or delete rows through the app's Database tab, the
/db API, or any SQLite tool (e.g. `sqlite3 data/triage.db`).

The training corpus is NOT stored here: it stays in checksummed snapshot files so experiments
remain reproducible.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS repositories (
    id             INTEGER PRIMARY KEY,
    full_name      TEXT NOT NULL UNIQUE COLLATE NOCASE,   -- "owner/repo"
    owner          TEXT NOT NULL,
    name           TEXT NOT NULL,
    source         TEXT NOT NULL,                         -- manual | github | webhook | ...
    added_at       TEXT NOT NULL,
    last_synced_at TEXT                                   -- last full fetch from GitHub
);
CREATE TABLE IF NOT EXISTS issues (
    id            INTEGER PRIMARY KEY,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    number        INTEGER NOT NULL,
    title         TEXT NOT NULL,
    body          TEXT NOT NULL DEFAULT '',
    labels        TEXT NOT NULL DEFAULT '[]',             -- JSON array of label names
    state         TEXT,
    url           TEXT,
    created_at    TEXT,
    source        TEXT NOT NULL,                          -- github | webhook | manual
    data          TEXT NOT NULL,                          -- JSON: the full issue
    prediction    TEXT,                                   -- JSON: the bot's last suggestion
    stored_at     TEXT NOT NULL,
    UNIQUE (repository_id, number)
);
CREATE INDEX IF NOT EXISTS idx_issues_repo ON issues(repository_id, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalise_repository(full_name: str) -> tuple[str, str]:
    """Accept "owner/repo" or a GitHub URL; return (owner, repo)."""
    value = full_name.strip().removesuffix("/").removesuffix(".git")
    if "github.com/" in value:
        value = value.split("github.com/", 1)[1]
    parts = [p for p in value.split("/") if p]
    if len(parts) < 2 or not all(c.isalnum() or c in "-_." for c in parts[0] + parts[1]):
        raise ValueError("expected a repository like owner/repo or https://github.com/owner/repo")
    return parts[0], parts[1]


def _label_names(labels: Any) -> list[str]:
    return [l["name"] if isinstance(l, dict) else str(l) for l in (labels or [])]


class IssueStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.execute("PRAGMA foreign_keys = ON")

    # ------------------------------------------------------------------ repositories
    def _repo_id(self, full_name: str) -> int | None:
        row = self._db.execute("SELECT id FROM repositories WHERE full_name = ?", (full_name,)).fetchone()
        return row["id"] if row else None

    def add_repository(self, full_name: str, source: str = "manual") -> dict:
        owner, name = normalise_repository(full_name)
        canonical = f"{owner}/{name}"
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO repositories (full_name, owner, name, source, added_at) VALUES (?, ?, ?, ?, ?)",
                (canonical, owner, name, source, _now()),
            )
        return self.get_repository(canonical)

    def get_repository(self, full_name: str) -> dict | None:
        owner, name = normalise_repository(full_name)
        row = self._db.execute(
            """SELECT r.*, COUNT(i.id) AS issue_count FROM repositories r
               LEFT JOIN issues i ON i.repository_id = r.id
               WHERE r.full_name = ? GROUP BY r.id""",
            (f"{owner}/{name}",),
        ).fetchone()
        return dict(row) if row else None

    def list_repositories(self) -> list[dict]:
        rows = self._db.execute(
            """SELECT r.*, COUNT(i.id) AS issue_count FROM repositories r
               LEFT JOIN issues i ON i.repository_id = r.id
               GROUP BY r.id ORDER BY r.full_name"""
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_repository(self, full_name: str) -> bool:
        owner, name = normalise_repository(full_name)
        with self._lock, self._db:
            cursor = self._db.execute("DELETE FROM repositories WHERE full_name = ?", (f"{owner}/{name}",))
        return cursor.rowcount > 0

    def mark_synced(self, full_name: str) -> None:
        owner, name = normalise_repository(full_name)
        with self._lock, self._db:
            self._db.execute(
                "UPDATE repositories SET last_synced_at = ? WHERE full_name = ?", (_now(), f"{owner}/{name}")
            )

    def is_synced(self, full_name: str) -> bool:
        repo = self.get_repository(full_name)
        return bool(repo and repo["last_synced_at"])

    # ------------------------------------------------------------------ issues
    def upsert_issue(
        self, full_name: str, issue: dict, source: str = "github", prediction: dict | None = None
    ) -> dict:
        """Insert or update one issue. A GitHub issue dict is stored as-is in `data`.
        An existing prediction is kept unless a new one is given."""
        if "pull_request" in issue:
            raise ValueError("pull requests are not stored")
        repo = self.get_repository(full_name) or self.add_repository(full_name, source=source)
        title = (issue.get("title") or "").strip()
        if not title:
            raise ValueError("an issue needs a title")
        number = issue.get("number")
        with self._lock, self._db:
            if number is None:  # manual issue without a GitHub number: next free number
                row = self._db.execute(
                    "SELECT COALESCE(MAX(number), 0) + 1 AS n FROM issues WHERE repository_id = ?", (repo["id"],)
                ).fetchone()
                number = row["n"]
            data = {**issue, "number": int(number)}
            data.setdefault("created_at", _now())
            self._db.execute(
                """INSERT INTO issues (repository_id, number, title, body, labels, state, url, created_at,
                                       source, data, prediction, stored_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(repository_id, number) DO UPDATE SET
                       title = excluded.title, body = excluded.body, labels = excluded.labels,
                       state = excluded.state, url = COALESCE(excluded.url, issues.url),
                       created_at = COALESCE(issues.created_at, excluded.created_at),
                       data = excluded.data, stored_at = excluded.stored_at,
                       -- a plain GitHub sync never overwrites how an issue first arrived
                       source = CASE WHEN excluded.source = 'github' THEN issues.source ELSE excluded.source END,
                       prediction = COALESCE(excluded.prediction, issues.prediction)""",
                (
                    repo["id"], int(number), title, data.get("body") or "",
                    json.dumps(_label_names(data.get("labels"))), data.get("state") or "open",
                    data.get("html_url") or data.get("url"), data["created_at"], source,
                    json.dumps(data, ensure_ascii=False),
                    json.dumps(prediction) if prediction is not None else None, _now(),
                ),
            )
        return self.get_issue(full_name, int(number))

    def upsert_many(self, full_name: str, issues: list[dict], source: str = "github") -> int:
        stored = 0
        for issue in issues:
            if "pull_request" not in issue and (issue.get("title") or "").strip():
                self.upsert_issue(full_name, issue, source=source)
                stored += 1
        return stored

    def _issue_row(self, row: sqlite3.Row) -> dict:
        return {
            "number": row["number"],
            "title": row["title"],
            "body": row["body"],
            "labels": json.loads(row["labels"]),
            "state": row["state"],
            "html_url": row["url"],
            "created_at": row["created_at"],
            "source": row["source"],
            "stored_at": row["stored_at"],
            "prediction": json.loads(row["prediction"]) if row["prediction"] else None,
            "data": json.loads(row["data"]),
        }

    def get_issue(self, full_name: str, number: int) -> dict | None:
        repo_id = self._repo_id("/".join(normalise_repository(full_name)))
        if repo_id is None:
            return None
        row = self._db.execute(
            "SELECT * FROM issues WHERE repository_id = ? AND number = ?", (repo_id, number)
        ).fetchone()
        return self._issue_row(row) if row else None

    def list_issues(self, full_name: str) -> list[dict]:
        repo_id = self._repo_id("/".join(normalise_repository(full_name)))
        if repo_id is None:
            return []
        rows = self._db.execute(
            "SELECT * FROM issues WHERE repository_id = ? ORDER BY created_at, number", (repo_id,)
        ).fetchall()
        return [self._issue_row(r) for r in rows]

    def delete_issue(self, full_name: str, number: int) -> bool:
        repo_id = self._repo_id("/".join(normalise_repository(full_name)))
        if repo_id is None:
            return False
        with self._lock, self._db:
            cursor = self._db.execute(
                "DELETE FROM issues WHERE repository_id = ? AND number = ?", (repo_id, number)
            )
        return cursor.rowcount > 0

    def close(self) -> None:
        self._db.close()
