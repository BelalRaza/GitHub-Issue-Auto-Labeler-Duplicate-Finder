"""M5 tests: API, webhook security, bot behaviour, rate limits, ETag cache and poller.

A fake GitHub (httpx.MockTransport) stands in for api.github.com, so nothing goes over the
network and no token is needed.
"""

import hashlib
import hmac
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import fastapi  # noqa: F401
    import httpx
    from fastapi.testclient import TestClient

    HAVE_SERVICE = True
except ImportError:
    HAVE_SERVICE = False

if HAVE_SERVICE:
    from issue_triage.classification import TfidfLogReg
    from issue_triage.predictor import TriagePredictor
    from issue_triage.retrieval import IssueRecord, RetrievalIndex
    from issue_triage.service.app import create_app
    from issue_triage.service.github_api import (
        GitHubClient,
        RateLimited,
        parse_issue_url,
        parse_repo_url,
    )
    from issue_triage.service.settings import Settings
    from issue_triage.service.webhook import BOT_MARKER

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
SECRET = "test-secret"


def toy_predictor():
    config = json.loads((ROOT / "config" / "model_config.json").read_text())
    config["retrieval"]["default_method"] = "tfidf"
    texts, labels = zip(*TRAIN)
    records = [
        IssueRecord("old/repo", 7, "https://github.com/old/repo/issues/7", "Upload crash",
                    "2026-01-01T00:00:00Z", "app crashes when uploading large images out of memory"),
    ]
    return TriagePredictor(TfidfLogReg(C=10).fit(texts, labels), RetrievalIndex(records, method="tfidf"), config)


def gh_issue(number, title, body="", created="2026-11-25T10:00:00Z", labels=(), pr=False):
    issue = {
        "number": number, "title": title, "body": body, "created_at": created, "state": "open",
        "html_url": f"https://github.com/me/demo/issues/{number}",
        "labels": [{"name": name} for name in labels],
    }
    if pr:
        issue["pull_request"] = {}
    return issue


class FakeGitHub:
    """Records every request and answers like api.github.com."""

    def __init__(self):
        self.requests = []
        self.issues = {}
        self.comments = {}
        self.rate_limited = False
        self.etag_hits = 0

    def __call__(self, request: "httpx.Request") -> "httpx.Response":
        self.requests.append((request.method, request.url.path, request.url.query.decode()))
        headers = {"x-ratelimit-limit": "5000", "x-ratelimit-remaining": "4999", "x-ratelimit-reset": "1795590000"}
        if self.rate_limited:
            return httpx.Response(403, json={"message": "API rate limit exceeded"},
                                  headers={**headers, "x-ratelimit-remaining": "0"})
        path = request.url.path
        if request.method == "GET" and path.endswith("/comments"):
            number = int(path.split("/")[-2])
            return httpx.Response(200, json=self.comments.get(number, []), headers=headers)
        if request.method == "POST" and path.endswith("/comments"):
            number = int(path.split("/")[-2])
            body = json.loads(request.content)["body"]
            self.comments.setdefault(number, []).append({"body": body})
            return httpx.Response(201, json={"id": 1}, headers=headers)
        if request.method == "GET" and path.endswith("/issues"):
            if request.headers.get("if-none-match") == '"v1"':
                self.etag_hits += 1
                return httpx.Response(304, headers=headers)
            items = sorted(self.issues.values(), key=lambda i: -i["number"])
            return httpx.Response(200, json=items, headers={**headers, "etag": '"v1"'})
        if request.method == "GET" and "/issues/" in path:
            number = int(path.rsplit("/", 1)[-1])
            if number not in self.issues:
                return httpx.Response(404, json={"message": "Not Found"}, headers=headers)
            return httpx.Response(200, json=self.issues[number], headers=headers)
        return httpx.Response(404, json={"message": "Not Found"}, headers=headers)


def signed(payload: dict) -> tuple[bytes, str]:
    body = json.dumps(payload).encode()
    return body, "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


@unittest.skipUnless(HAVE_SERVICE, "service dependencies not installed")
class ServiceTests(unittest.TestCase):
    def make(self, **settings_kwargs):
        self.fake = FakeGitHub()
        github = GitHubClient(token="x", transport=httpx.MockTransport(self.fake), cache_ttl_seconds=0)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        settings = Settings(webhook_secret=SECRET, db_path=Path(self._tmp.name) / "t.db",
                            local_hosts=frozenset({"testclient"}), **settings_kwargs)
        app = create_app(settings=settings, predictor=toy_predictor(), github=github, load_model=False)
        return TestClient(app)

    def webhook(self, client, payload, delivery="d1", event="issues", signature=None):
        body, sig = signed(payload)
        response = client.post("/webhook/github", content=body, headers={
            "X-GitHub-Event": event, "X-GitHub-Delivery": delivery,
            "X-Hub-Signature-256": signature or sig, "Content-Type": "application/json",
        })
        client.app.state.service.drain()  # wait for the ordered worker to finish
        return response

    # ---------------------------------------------------------------- basic API
    def test_predict_and_empty_input(self):
        with self.make() as client:
            ok = client.post("/predict", json={"title": "App crashes when uploading large images", "body": "out of memory"})
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(ok.json()["duplicates"]["candidates"][0]["issue_number"], 7)
            self.assertEqual(client.post("/predict", json={"title": "", "body": " "}).status_code, 422)
            health = client.get("/health").json()
            self.assertEqual(health["status"], "ok")
            self.assertNotIn(SECRET, json.dumps(health))  # never leak secrets

    def test_issue_url_compares_with_maintainer_label(self):
        with self.make() as client:
            self.fake.issues[3] = gh_issue(3, "Docs: typo in the README installation guide", labels=["documentation"])
            r = client.post("/analyze/issue-url", json={"url": "https://github.com/me/demo/issues/3"}).json()
            self.assertEqual(r["github"]["maintainer_label_mapped"], "Documentation")
            self.assertIn("agrees_with_maintainers", r["github"])

    def test_pull_request_and_missing_issue(self):
        with self.make() as client:
            self.fake.issues[4] = gh_issue(4, "Some PR", pr=True)
            self.assertEqual(client.post("/analyze/issue-url", json={"url": "https://github.com/me/demo/issues/4"}).status_code, 422)
            self.assertEqual(client.post("/analyze/issue-url", json={"url": "https://github.com/me/demo/issues/99"}).status_code, 404)
            self.assertEqual(client.post("/analyze/issue-url", json={"url": "not a link"}).status_code, 422)

    def test_rate_limit_becomes_clear_429(self):
        with self.make() as client:
            self.fake.rate_limited = True
            r = client.post("/analyze/issue-url", json={"url": "https://github.com/me/demo/issues/1"})
            self.assertEqual(r.status_code, 429)
            self.assertIn("rate limit", r.json()["detail"])
            self.assertIn("rate_limit_resets_at", r.json())

    def test_repo_scan_uses_one_call_and_skips_prs(self):
        with self.make() as client:
            self.fake.issues = {1: gh_issue(1, "Crash on start", labels=["bug"]), 2: gh_issue(2, "PR", pr=True)}
            r = client.post("/analyze/repo-url", json={"url": "https://github.com/me/demo"}).json()
            self.assertEqual(r["issues_analyzed"], 1)
            self.assertEqual(len(self.fake.requests), 1)

    # ---------------------------------------------------------------- webhook security
    def test_webhook_rejects_bad_signature(self):
        with self.make() as client:
            r = self.webhook(client, {"action": "opened"}, signature="sha256=deadbeef")
            self.assertEqual(r.status_code, 401)

    def test_webhook_requires_secret_unless_explicitly_allowed(self):
        fake = FakeGitHub()
        github = GitHubClient(transport=httpx.MockTransport(fake))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        app = create_app(settings=Settings(db_path=Path(tmp.name) / "t.db"), predictor=toy_predictor(),
                         github=github, load_model=False)
        with TestClient(app) as client:
            self.assertEqual(client.post("/webhook/github", content=b"{}", headers={"X-GitHub-Event": "ping"}).status_code, 503)

    def test_ping_and_ignored_events(self):
        with self.make() as client:
            self.assertEqual(self.webhook(client, {"zen": "hi"}, event="ping").json()["message"], "pong")
            self.assertFalse(self.webhook(client, {"action": "closed"}, delivery="d2").json()["accepted"])

    # ---------------------------------------------------------------- bot behaviour
    def payload(self, number, title, body=""):
        # Distinct creation times: the search only looks at issues created strictly earlier.
        created = f"2026-11-25T10:{number:02d}:00Z"
        issue = gh_issue(number, title, body, created=created)
        self.fake.issues[number] = issue  # like real GitHub: an opened issue is in the repo's list
        return {"action": "opened", "repository": {"full_name": "me/demo"}, "issue": issue}

    def test_suggestion_only_by_default(self):
        with self.make() as client:
            r = self.webhook(client, self.payload(10, "App crashes when uploading large images", "out of memory"))
            self.assertEqual(r.status_code, 202)
            self.assertEqual(client.get("/metrics").json()["webhook"]["processed"], 1)
            self.assertFalse(any(m == "POST" for m, _, _ in self.fake.requests))  # nothing written

    def test_bot_comments_once_and_finds_repeat(self):
        with self.make(bot_post_comments=True, bot_repositories=["me/demo"]) as client:
            self.webhook(client, self.payload(10, "Spinner never stops on the login page", "loading spinner keeps spinning forever"))
            self.assertIn(BOT_MARKER, self.fake.comments[10][0]["body"])
            # Same delivery again (GitHub redelivery) -> ignored, still one comment.
            self.webhook(client, self.payload(10, "Spinner never stops on the login page"))
            self.assertEqual(len(self.fake.comments[10]), 1)
            # A reworded repeat is now matched against issue 10, which the bot remembered.
            self.webhook(client, self.payload(11, "Login page spinner never stops", "the loading spinner spins forever"),
                         delivery="d3")
            event = client.get("/events").json()["events"][0]
            self.assertEqual(event["issue_number"], 11)
            self.assertEqual(event["top_match"]["issue_number"], 10)

    def test_back_to_back_issues_are_processed_in_order(self):
        # Regression: two webhooks sent without waiting. Issue 21 must still match issue 20,
        # which needs issue 20 to be fully indexed first (single ordered worker).
        with self.make() as client:
            for number, title, delivery in [
                (20, "Spinner never stops on the login page", "x1"),
                (21, "Login page spinner never stops", "x2"),
            ]:
                body, sig = signed(self.payload(number, title, "the loading spinner spins forever"))
                client.post("/webhook/github", content=body, headers={
                    "X-GitHub-Event": "issues", "X-GitHub-Delivery": delivery, "X-Hub-Signature-256": sig})
            client.app.state.service.drain()
            latest = client.get("/events").json()["events"][0]
            self.assertEqual((latest["issue_number"], latest["top_match"]["issue_number"]), (21, 20))
            self.assertEqual(client.get("/metrics").json()["webhook"]["processed"], 2)

    def test_duplicates_are_searched_only_in_the_same_repository(self):
        with self.make() as client:
            # old/repo#7 (in the collected corpus) is about the same crash, but lives in ANOTHER
            # repository, so it must not be suggested for me/demo.
            self.webhook(client, self.payload(30, "App crashes when uploading large images", "out of memory"))
            event = client.get("/events").json()["events"][0]
            self.assertIsNone(event["top_match"])  # nothing earlier in me/demo yet
            self.webhook(client, self.payload(31, "Uploading large images crashes the app", "out of memory"),
                         delivery="s2")
            event = client.get("/events").json()["events"][0]
            self.assertEqual(event["top_match"]["repository"], "me/demo")
            self.assertEqual(event["top_match"]["issue_number"], 30)
            # Without a repository (typed issue), all collected issues are searched.
            typed = client.post("/predict", json={"title": "App crashes when uploading large images"}).json()
            self.assertEqual(typed["duplicates"]["candidates"][0]["repository"], "old/repo")

    # ---------------------------------------------------------------- database
    def test_webhook_issues_and_predictions_are_stored(self):
        with self.make() as client:
            self.webhook(client, self.payload(40, "App crashes when uploading large images", "out of memory"))
            issues = client.get("/db/repos/me/demo/issues").json()["issues"]
            stored = next(i for i in issues if i["number"] == 40)
            self.assertEqual(stored["source"], "webhook")
            self.assertIn("top_guess", stored["prediction"])

    def test_deleted_in_database_means_not_compared_any_more(self):
        with self.make() as client:
            self.webhook(client, self.payload(50, "Spinner never stops on the login page", "spins forever"))
            self.assertEqual(client.delete("/db/repos/me/demo/issues/50").status_code, 200)
            self.webhook(client, self.payload(51, "Login page spinner never stops", "spins forever"), delivery="x51")
            event = client.get("/events").json()["events"][0]
            self.assertIsNone(event["top_match"])  # #50 was deleted from the database

    def test_manual_issue_is_used_for_duplicate_search(self):
        with self.make() as client:
            client.post("/db/repos", json={"repository": "me/demo", "sync": True})
            added = client.post("/db/repos/me/demo/issues",
                                json={"title": "Dark mode toggle resets after reload", "labels": ["bug"],
                                      "created_at": "2026-11-25T09:00:00Z"}).json()
            self.webhook(client, self.payload(added["number"] + 1, "Dark mode toggle resets after a reload",
                                              "the dark mode setting is lost"))
            event = client.get("/events").json()["events"][0]
            self.assertEqual(event["top_match"]["issue_number"], added["number"])

    def test_webhook_delete_and_label_events_update_the_database(self):
        with self.make() as client:
            self.webhook(client, self.payload(60, "Crash on save", "traceback attached"))
            labelled = {**self.fake.issues[60], "labels": [{"name": "bug"}]}
            self.webhook(client, {"action": "labeled", "repository": {"full_name": "me/demo"}, "issue": labelled},
                         delivery="l60")
            self.assertEqual(client.get("/db/repos/me/demo/issues/60").json()["labels"], ["bug"])
            self.webhook(client, {"action": "deleted", "repository": {"full_name": "me/demo"}, "issue": labelled},
                         delivery="d60")
            self.assertEqual(client.get("/db/repos/me/demo/issues/60").status_code, 404)

    def test_database_refuses_requests_through_the_tunnel(self):
        with self.make() as client:
            r = client.get("/db/repos", headers={"X-Forwarded-For": "203.0.113.7"})  # what ngrok adds
            self.assertEqual(r.status_code, 403)
            self.assertEqual(client.delete("/db/repos/me/demo", headers={"X-Forwarded-For": "1.2.3.4"}).status_code, 403)

    def test_delete_repository_removes_its_issues(self):
        with self.make() as client:
            self.webhook(client, self.payload(70, "Crash on save", "traceback"))
            self.assertEqual(client.delete("/db/repos/me/demo").status_code, 200)
            self.assertEqual(client.get("/db/repos/me/demo/issues").status_code, 404)
            self.assertEqual(client.get("/db/repos").json()["repositories"], [])

    def test_bot_never_writes_outside_allowlist(self):
        with self.make(bot_post_comments=True, bot_repositories=["someone/else"]) as client:
            self.webhook(client, self.payload(12, "App crashes when uploading"))
            self.assertEqual(self.fake.comments, {})

    # ---------------------------------------------------------------- poller + cache
    def test_poller_baseline_then_new_issue_and_etag(self):
        # No `with`: the app's startup hook (which would start the background poller thread)
        # does not run, so the test drives poll_once() deterministically.
        client = self.make(poll_repositories=["me/demo"])
        service = client.app.state.service
        self.fake.issues = {1: gh_issue(1, "Old issue about crashes")}
        self.assertEqual(service.poll_once(), 0)  # first poll = baseline only
        self.assertEqual(service.poll_once(), 0)  # unchanged -> 304, free
        self.assertGreaterEqual(self.fake.etag_hits, 1)
        self.fake.issues[2] = gh_issue(2, "New: export to CSV please", "Please support CSV export")
        self.fake.etag_hits = 0
        service.github._cache.clear()
        self.assertEqual(service.poll_once(), 1)


@unittest.skipUnless(HAVE_SERVICE, "service dependencies not installed")
class UrlParsingTests(unittest.TestCase):
    def test_parsers(self):
        self.assertEqual(parse_issue_url("https://github.com/a/b/issues/5#issuecomment-1"), ("a", "b", 5))
        self.assertEqual(parse_repo_url("https://github.com/a/b.git"), ("a", "b"))
        self.assertEqual(parse_repo_url("https://github.com/a/b/issues/5"), ("a", "b"))
        with self.assertRaises(ValueError):
            parse_issue_url("https://example.com/a/b/issues/5")

    def test_rate_limited_is_typed(self):
        self.assertEqual(RateLimited("x", None).status, 429)


if __name__ == "__main__":
    unittest.main()
