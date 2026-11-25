import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from issue_triage.preparation import (
    clean_issue_text,
    flatten_issue,
    map_issue_type,
    map_labels,
    read_raw_issues,
)


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = json.loads((ROOT / "config" / "label_taxonomy.json").read_text())

    def test_cleaner_retains_code_and_replaces_identity_noise(self):
        text = """Hi @octocat, see [trace](https://example.com/x).\n```py\nraise ValueError('x')\n```"""
        cleaned = clean_issue_text(text)
        self.assertIn("<USER>", cleaned)
        self.assertIn("<CODE_BLOCK>", cleaned)
        self.assertIn("raise ValueError", cleaned)
        self.assertNotIn("https://example.com", cleaned)

    def test_repository_override_maps_to_standard_class(self):
        self.assertEqual(
            map_labels(["C-bug", "A-diagnostics"], "rust-lang/rust", self.taxonomy),
            ["Bug"],
        )

    def test_structured_issue_type_maps_to_standard_class(self):
        self.assertEqual(map_issue_type("Feature", self.taxonomy), ["Feature"])

    def test_cross_class_conflict_is_not_classification_eligible(self):
        wrapper = {
            "repository": "microsoft/TypeScript",
            "collected_at": "2026-10-28T00:00:00Z",
            "source_endpoint": "test",
            "issue": {
                "id": 1,
                "number": 2,
                "title": "Ambiguous",
                "body": "Body",
                "labels": [{"name": "Bug"}, {"name": "Suggestion"}],
                "state": "open",
                "state_reason": None,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "closed_at": None,
                "comments": 0,
            },
        }
        row = flatten_issue(wrapper, self.taxonomy)
        self.assertTrue(row["taxonomy_conflict"])
        self.assertFalse(row["classification_eligible"])
        self.assertEqual(row["classification_target"], "")

    def test_structured_duplicate_state_is_verified(self):
        wrapper = {
            "repository": "flutter/flutter",
            "collected_at": "2026-10-28T00:00:00Z",
            "source_endpoint": "test",
            "issue": {
                "id": 1,
                "number": 3,
                "title": "Duplicate",
                "body": "Body",
                "labels": [{"name": "r: duplicate"}],
                "state": "closed",
                "state_reason": "duplicate",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-02T00:00:00Z",
                "closed_at": "2026-01-02T00:00:00Z",
                "comments": 1,
            },
        }
        row = flatten_issue(wrapper, self.taxonomy)
        self.assertTrue(row["is_duplicate_verified"])
        self.assertTrue(row["has_duplicate_label"])

    def test_repository_scope_split_keeps_time_order_within_each_repo(self):
        from issue_triage.preparation import assign_chronological_splits

        rows = []
        for repo, start_year in (("a/old", 2015), ("b/new", 2026)):
            for i in range(20):
                rows.append({"repository": repo, "issue_number": i, "classification_eligible": True,
                             "created_at": f"{start_year}-01-{i + 1:02d}T00:00:00Z", "split": ""})
        assign_chronological_splits(rows, scope="repository")
        for repo in ("a/old", "b/new"):
            mine = sorted((r for r in rows if r["repository"] == repo), key=lambda r: r["created_at"])
            splits = [r["split"] for r in mine]
            self.assertEqual(splits.count("test"), 3)  # every repo contributes to test
            self.assertEqual(splits[-1], "test")  # the newest issue is tested, never trained on
            self.assertEqual(splits[0], "train")

    def test_jsonl_reader_does_not_split_unicode_line_separator(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            record = {"repository": "x/y", "issue": {"body": "left\u2028right"}}
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            loaded = list(read_raw_issues(Path(directory)))
        self.assertEqual(loaded, [record])


if __name__ == "__main__":
    unittest.main()

