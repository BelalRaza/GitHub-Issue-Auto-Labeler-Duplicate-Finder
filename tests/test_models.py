"""M3 tests. They use lexical models only, so they run in seconds without downloading anything."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import sklearn  # noqa: F401

    HAVE_ML = True
except ImportError:  # the M1/M2 tests must still run with the standard library only
    HAVE_ML = False

if HAVE_ML:
    from issue_triage.classification import MajorityClassifier, TfidfLogReg, predict_labels
    from issue_triage.metrics import retrieval_metrics
    from issue_triage.predictor import TriagePredictor
    from issue_triage.retrieval import IssueRecord, RetrievalIndex

TRAIN = [
    ("App crashes with segmentation fault on startup", "Bug"),
    ("Error: KeyError raised when loading config file", "Bug"),
    ("Crash when clicking save, traceback attached", "Bug"),
    ("Regression: build fails after upgrading to 2.0", "Bug"),
    ("Feature request: add dark mode to settings", "Feature"),
    ("Please support exporting results to CSV format", "Feature"),
    ("Add an option to configure the request timeout", "Feature"),
    ("Docs: typo in installation guide README", "Documentation"),
    ("Documentation for the API is missing examples", "Documentation"),
    ("Docs page has a broken link to the tutorial", "Documentation"),
]


def records() -> list:
    return [
        IssueRecord("o/r", 1, "u1", "Crash on upload", "2026-01-01T00:00:00Z", "app crashes when uploading large images"),
        IssueRecord("o/r", 2, "u2", "Dark mode", "2026-01-02T00:00:00Z", "add a dark theme to the settings page"),
        IssueRecord("o/r", 3, "u3", "Upload crash again", "2026-01-03T00:00:00Z", "crash uploading big image files"),
    ]


@unittest.skipUnless(HAVE_ML, "ML dependencies not installed")
class ClassifierTests(unittest.TestCase):
    def test_majority_always_predicts_most_frequent(self):
        model = MajorityClassifier().fit(["a", "b", "c"], ["Bug", "Bug", "Feature"])
        self.assertEqual(predict_labels(model, ["anything", "else"]), ["Bug", "Bug"])

    def test_tfidf_learns_separable_toy_classes(self):
        texts, labels = zip(*TRAIN)
        model = TfidfLogReg(C=10).fit(texts, labels)
        self.assertEqual(predict_labels(model, ["docs typo in the README guide"]), ["Documentation"])
        self.assertAlmostEqual(float(model.predict_proba(["crash"]).sum()), 1.0, places=5)


@unittest.skipUnless(HAVE_ML, "ML dependencies not installed")
class RetrievalTests(unittest.TestCase):
    def test_finds_lexically_similar_issue(self):
        index = RetrievalIndex(records(), method="tfidf")
        top = index.search("app crash uploading large image", k=1)[0]
        self.assertIn(top["issue_number"], (1, 3))

    def test_past_only_and_self_exclusion(self):
        index = RetrievalIndex(records(), method="bm25")
        results = index.search(
            "crash uploading big image files", k=5, created_before="2026-01-03T00:00:00Z", exclude=("o/r", 3)
        )
        numbers = [r["issue_number"] for r in results]
        self.assertNotIn(3, numbers)  # never itself, never the future
        self.assertEqual(numbers[0], 1)

    def test_add_makes_new_issue_searchable(self):
        index = RetrievalIndex(records(), method="tfidf")
        index.add(IssueRecord("o/r", 4, "u4", "Spinner", "2026-01-04T00:00:00Z", "loading spinner never stops"))
        self.assertEqual(index.search("spinner never stops", k=1)[0]["issue_number"], 4)

    def test_embedding_failure_falls_back_to_tfidf(self):
        index = RetrievalIndex(records(), method="embedding", embedding_model=None)
        self.assertEqual(index.method, "tfidf")

    def test_retrieval_metrics(self):
        metrics = retrieval_metrics([1, 3, None, 7])
        self.assertEqual(metrics["recall@1"], 0.25)
        self.assertEqual(metrics["recall@5"], 0.5)
        self.assertAlmostEqual(metrics["mrr@10"], (1 + 1 / 3 + 1 / 7) / 4, places=3)


@unittest.skipUnless(HAVE_ML, "ML dependencies not installed")
class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts, labels = zip(*TRAIN)
        config = json.loads((ROOT / "config" / "model_config.json").read_text())
        config["retrieval"]["default_method"] = "tfidf"
        cls.predictor = TriagePredictor(
            TfidfLogReg(C=10).fit(texts, labels), RetrievalIndex(records(), method="tfidf"), config
        )

    def test_output_contract(self):
        result = self.predictor.predict("App crashes when uploading large images", "Stack trace attached.")
        self.assertIn(result["classification"]["decision"], {"suggest_label", "needs_manual_review"})
        self.assertIn(result["duplicates"]["decision"], {"possible_duplicate", "check_similar", "no_strong_match"})
        self.assertLessEqual(len(result["duplicates"]["candidates"]), 5)

    def test_short_issue_goes_to_manual_review(self):
        result = self.predictor.predict("help pls")
        self.assertEqual(result["classification"]["decision"], "needs_manual_review")
        self.assertIsNone(result["classification"]["label"])

    def test_duplicate_decisions_are_honest_tiers(self):
        import copy

        from issue_triage.service.webhook import render_comment

        predictor = copy.copy(self.predictor)
        predictor.config = copy.deepcopy(self.predictor.config)
        ret = predictor.config["retrieval"]
        ret.update({"duplicate_threshold": 0.99, "review_threshold": 0.98, "short_issue_review_floor": 0.05})
        # Short issue (< min_words) with a moderate match -> offered for checking (live demo #4/#5).
        short = predictor.predict("upload crash image")
        self.assertEqual(short["duplicates"]["decision"], "check_similar")
        self.assertIn("Might be the same as", render_comment(short))
        # Long issue with the same moderate match -> "no strong match", never "not a duplicate".
        long = predictor.predict(
            "upload crash image",
            "the application window closes unexpectedly whenever someone tries this on a laptop",
        )
        self.assertEqual(long["duplicates"]["decision"], "no_strong_match")
        comment = render_comment(long)
        self.assertIn("Not sure", comment)
        self.assertNotIn("No duplicate", comment)

    def test_empty_issue_is_rejected(self):
        with self.assertRaises(ValueError):
            self.predictor.predict("", "   ")

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            self.predictor.save(Path(directory))
            loaded = TriagePredictor.load(Path(directory))
            self.assertEqual(
                loaded.predict("Docs typo in README guide")["classification"]["top_guess"],
                self.predictor.predict("Docs typo in README guide")["classification"]["top_guess"],
            )


if __name__ == "__main__":
    unittest.main()
