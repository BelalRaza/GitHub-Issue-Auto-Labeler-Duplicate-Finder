"""SQLite store: repositories and their issues (standard library only)."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from issue_triage.service.storage import IssueStore, normalise_repository  # noqa: E402


def gh(number, title, labels=(), body="text"):
    return {"number": number, "title": title, "body": body, "labels": [{"name": l} for l in labels],
            "state": "open", "html_url": f"https://github.com/me/demo/issues/{number}",
            "created_at": f"2026-11-25T10:{number:02d}:00Z"}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = IssueStore(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_repository_names_are_normalised(self):
        self.assertEqual(normalise_repository("https://github.com/me/demo.git"), ("me", "demo"))
        self.assertEqual(normalise_repository("me/demo/issues/4"), ("me", "demo"))
        with self.assertRaises(ValueError):
            normalise_repository("not a repo")
        self.store.add_repository("https://github.com/Me/Demo")
        self.store.add_repository("me/demo")  # same repository, different case: no duplicate
        self.assertEqual(len(self.store.list_repositories()), 1)

    def test_issues_are_stored_as_json_and_listed_per_repository(self):
        self.store.upsert_issue("me/demo", gh(1, "Login broken", ["bug"]))
        self.store.upsert_issue("me/other", gh(1, "Unrelated"))
        issues = self.store.list_issues("me/demo")
        self.assertEqual([i["title"] for i in issues], ["Login broken"])
        self.assertEqual(issues[0]["labels"], ["bug"])
        self.assertEqual(issues[0]["data"]["html_url"], "https://github.com/me/demo/issues/1")
        self.assertEqual(self.store.get_repository("me/demo")["issue_count"], 1)

    def test_update_keeps_prediction_unless_replaced(self):
        self.store.upsert_issue("me/demo", gh(1, "Login broken"), prediction={"label": "Bug"})
        self.store.upsert_issue("me/demo", gh(1, "Login broken (edited)", ["bug"]))  # e.g. webhook "edited"
        issue = self.store.get_issue("me/demo", 1)
        self.assertEqual(issue["title"], "Login broken (edited)")
        self.assertEqual(issue["prediction"], {"label": "Bug"})

    def test_manual_issue_gets_next_number(self):
        self.store.upsert_issue("me/demo", gh(7, "From GitHub"))
        manual = self.store.upsert_issue("me/demo", {"title": "Typed by hand", "labels": []}, source="manual")
        self.assertEqual(manual["number"], 8)
        self.assertEqual(manual["source"], "manual")

    def test_delete_issue_and_cascade_on_repository_delete(self):
        for n in (1, 2):
            self.store.upsert_issue("me/demo", gh(n, f"Issue {n}"))
        self.assertTrue(self.store.delete_issue("me/demo", 1))
        self.assertFalse(self.store.delete_issue("me/demo", 1))
        self.assertTrue(self.store.delete_repository("me/demo"))
        self.store.add_repository("me/demo")
        self.assertEqual(self.store.list_issues("me/demo"), [])  # issues went with the repository

    def test_pull_requests_and_untitled_issues_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.upsert_issue("me/demo", {**gh(1, "PR"), "pull_request": {}})
        with self.assertRaises(ValueError):
            self.store.upsert_issue("me/demo", {"number": 2, "title": "  "})

    def test_store_survives_reopening(self):
        self.store.upsert_issue("me/demo", gh(1, "Persisted"))
        self.store.close()
        self.store = IssueStore(Path(self.tmp.name) / "t.db")
        self.assertEqual(self.store.list_issues("me/demo")[0]["title"], "Persisted")


if __name__ == "__main__":
    unittest.main()
