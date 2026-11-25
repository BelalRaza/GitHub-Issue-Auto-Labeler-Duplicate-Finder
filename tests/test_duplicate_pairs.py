import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_duplicate_pairs import build_query, parse_batch


class DuplicatePairTests(unittest.TestCase):
    def test_query_and_parser_use_structured_duplicate_reference(self):
        source = {
            "repository": "flutter/flutter",
            "issue_number": "181034",
            "url": "https://github.com/flutter/flutter/issues/181034",
            "created_at": "2026-01-15T00:00:00Z",
        }
        query, aliases = build_query([source])
        self.assertIn("duplicateOf", query)
        data = {
            "r0": {
                "issue": {
                    "url": source["url"],
                    "stateReason": "DUPLICATE",
                    "duplicateOf": {
                        "number": 150326,
                        "url": "https://github.com/flutter/flutter/issues/150326",
                        "repository": {"nameWithOwner": "flutter/flutter"},
                    },
                }
            }
        }
        pairs, unresolved = parse_batch(data, aliases)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["canonical_issue_number"], 150326)
        self.assertEqual(unresolved, [])


if __name__ == "__main__":
    unittest.main()

