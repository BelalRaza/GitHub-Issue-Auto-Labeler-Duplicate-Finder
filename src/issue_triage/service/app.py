"""FastAPI app: the three ways to use the system, plus health, metrics and a live feed.

Run:  .venv/bin/uvicorn issue_triage.service.app:app --app-dir src --port 8000
Open: http://localhost:8000

    GET  /                    web page: paste a link or type an issue, see every step
    POST /predict             {title, body}            -> label + similar issues
    POST /analyze/issue-url   {url}                    -> Way 1: any public GitHub issue
    POST /analyze/repo-url    {url, max_issues}        -> Way 2: a repo's newest issues
    POST /webhook/github      GitHub "issues" events   -> Way 3: live bot
    GET  /health  /metrics  /events
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..predictor import TriagePredictor
from ..preparation import clean_issue_text, map_issue_type, map_labels
from ..retrieval import IssueRecord
from .github_api import GitHubClient, GitHubError, RateLimited, parse_issue_url, parse_repo_url
from .monitoring import Metrics
from .settings import Settings
from .storage import IssueStore, normalise_repository
from .webhook import BOT_MARKER, DeliveryDeduper, render_comment, verify_signature

log = logging.getLogger("issue_triage.service")
STATIC = Path(__file__).parent / "static"


# --------------------------------------------------------------------------- request bodies
class PredictRequest(BaseModel):
    title: str = Field("", max_length=1000)
    body: str = Field("", max_length=200_000)
    repository: str | None = None


class RepoRequest(BaseModel):
    repository: str = Field(..., max_length=300, description="owner/repo or a GitHub URL")
    sync: bool = Field(True, description="also fetch its newest issues from GitHub")


class IssueIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=1000)
    body: str = Field("", max_length=200_000)
    labels: list[str] = Field(default_factory=list)
    number: int | None = Field(None, ge=1, description="leave empty for a manual issue")
    state: str = "open"
    created_at: str | None = None


class UrlRequest(BaseModel):
    url: str = Field(..., max_length=500)
    max_issues: int = Field(20, ge=1, le=100)


# --------------------------------------------------------------------------- service
class TriageService:
    """Holds the model and GitHub client.

    New issues (webhook or poller) go through ONE queue handled by ONE worker thread, so they
    are processed strictly in arrival order. Without this, issue #2 could be searched while
    issue #1 is still being added to the index (about 1 s), and the repeat would be missed.
    A lock also guards the index, because web requests read it while the worker writes."""

    def __init__(self, settings: Settings, predictor: TriagePredictor | None, github: GitHubClient):
        self.settings = settings
        self.predictor = predictor
        self.github = github
        self.metrics = Metrics()
        self.deliveries = DeliveryDeduper()
        self.lock = threading.Lock()
        self.taxonomy = json.loads(settings.taxonomy_path.read_text(encoding="utf-8"))
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._poll_last_seen: dict[str, int] = {}
        self._jobs: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self.store = IssueStore(settings.db_path)

    # ---------------------------------------------------------------- ordered worker
    def start_worker(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._work, name="issue-worker", daemon=True)
            self._worker.start()

    def submit(self, issue: dict, repository: str, source: str, action: str = "opened") -> None:
        """Queue an issue event. Without a running worker (e.g. in a unit test) run it inline.
        Every event goes through the same queue, so "opened" then "deleted" are applied in order."""
        if self._worker is None:
            self._process(issue, repository, source, action)
        else:
            self._jobs.put((issue, repository, source, action))

    def drain(self) -> None:
        """Block until every queued issue is processed (used by tests and shutdown)."""
        self._jobs.join()

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                self._process(*job)
            finally:
                self._jobs.task_done()

    def _process(self, issue: dict, repository: str, source: str, action: str = "opened") -> None:
        try:
            if action == "opened":
                self.handle_new_issue(issue, repository, source)
            elif action == "deleted":
                self.store.delete_issue(repository, int(issue["number"]))
            else:  # edited / labeled / unlabeled / closed / reopened: keep the stored copy current
                self.store.upsert_issue(repository, issue, source="webhook")
            if source == "webhook":
                self.metrics.webhook["processed"] += 1
        except Exception:  # one bad issue must never stop the worker or the server
            self.metrics.record_error(f"{source}_processing")
            log.exception("failed to process %s#%s", repository, issue.get("number"))

    # ---------------------------------------------------------------- prediction
    def _require_model(self) -> TriagePredictor:
        if self.predictor is None:
            raise HTTPException(503, "model not loaded: run scripts/train_models.py first")
        return self.predictor

    def predict(self, **kwargs) -> dict:
        predictor = self._require_model()
        with self.lock:
            return predictor.predict(**kwargs)

    def maintainer_label(self, issue: dict, repository: str) -> dict:
        """What the repo's maintainers labelled it, mapped to our classes (for comparison only:
        never fed to the model)."""
        labels = [l["name"] if isinstance(l, dict) else str(l) for l in issue.get("labels", [])]
        issue_type = issue.get("type")
        if isinstance(issue_type, dict):
            issue_type = issue_type.get("name")
        mapped = sorted(set(map_labels(labels, repository, self.taxonomy)) |
                        set(map_issue_type(str(issue_type or ""), self.taxonomy)))
        return {"original_labels": labels, "mapped": mapped[0] if len(mapped) == 1 else None}

    def to_records(self, issues: list[dict], repository: str) -> list[IssueRecord]:
        """GitHub issues -> searchable records, carrying the maintainers' label (mapped to our
        classes) so the classifier can learn this repository's conventions."""
        return [
            IssueRecord(repository, int(i["number"]), i.get("html_url") or "", i.get("title") or "",
                        i.get("created_at") or "",
                        clean_issue_text(f"{i.get('title') or ''}\n\n{i.get('body') or ''}"),
                        label=self.maintainer_label(i, repository)["mapped"])
            for i in issues if "pull_request" not in i
        ]

    def sync_repository(self, repository: str) -> int:
        """Fetch the repository's newest 100 issues (one API call) into the database."""
        owner, repo = normalise_repository(repository)
        issues = self.github.list_recent_issues(owner, repo, count=100, state="all", fresh=True)
        self.store.add_repository(f"{owner}/{repo}", source="github")
        stored = self.store.upsert_many(f"{owner}/{repo}", issues, source="github")
        self.store.mark_synced(f"{owner}/{repo}")
        return stored

    def repository_index(self, repository: str, live_issues: list[dict] | None = None):
        """Search set for one repository = the issues stored for it in the database (plus any
        collected for it in the training corpus). The first time a repository is seen, its newest
        issues are fetched from GitHub into the database. After that the database is the source of
        truth: what you delete there is no longer compared against."""
        predictor = self._require_model()
        if live_issues is not None:
            self.store.add_repository(repository, source="github")
            self.store.upsert_many(repository, live_issues, source="github")
            self.store.mark_synced(repository)
        elif not self.store.is_synced(repository):
            try:
                self.sync_repository(repository)
            except GitHubError as exc:  # GitHub down: use whatever the database already has
                self.metrics.record_error("github_compare_fetch")
                log.warning("could not fetch %s issues for comparison: %s", repository, exc)
        stored = self.store.list_issues(repository)
        with self.lock:
            return predictor.index.for_repository(repository, self.to_records(stored, repository))

    @staticmethod
    def prediction_summary(result: dict) -> dict:
        cls, dup = result["classification"], result["duplicates"]
        top = dup["candidates"][0] if dup["candidates"] else None
        return {
            "label": cls["label"], "top_guess": cls["top_guess"], "confidence": cls["confidence"],
            "label_decision": cls["decision"], "duplicate_decision": dup["decision"],
            "top_match": None if not top else {
                "repository": top["repository"], "issue_number": top["issue_number"],
                "similarity": top["similarity"], "url": top["url"]},
            "model": result["model"].get("classifier"), "config_version": result["model"].get("version"),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def analyze_github_issue(self, issue: dict, repository: str, source: str, comparison=None) -> dict:
        if "pull_request" in issue:
            raise HTTPException(422, "that link is a pull request, not an issue")
        if comparison is None:
            comparison = self.repository_index(repository)
        result = self.predict(
            title=issue.get("title") or "",
            body=issue.get("body") or "",
            repository=repository,
            created_at=issue.get("created_at"),  # search only issues older than this one
            issue_key=(repository, int(issue["number"])),
            index=comparison,  # and only issues from the same repository
        )
        truth = self.maintainer_label(issue, repository)
        result["github"] = {
            "repository": repository,
            "number": issue["number"],
            "url": issue.get("html_url"),
            "state": issue.get("state"),
            "maintainer_labels": truth["original_labels"],
            "maintainer_label_mapped": truth["mapped"],
            "agrees_with_maintainers": None if truth["mapped"] is None
            else truth["mapped"] == result["classification"]["top_guess"],
        }
        self.metrics.record_prediction(
            result, source, {"repository": repository, "number": issue["number"], "url": issue.get("html_url")}
        )
        try:  # remember the issue and the bot's suggestion in the database
            self.store.upsert_issue(repository, issue, source="webhook" if source == "webhook" else "github",
                                    prediction=self.prediction_summary(result))
        except ValueError:
            pass  # e.g. an issue without a title: nothing useful to store
        return result

    # ---------------------------------------------------------------- live bot
    def handle_new_issue(self, issue: dict, repository: str, source: str) -> dict:
        # The repository's earlier issues are fetched fresh from GitHub, so a repeat of an issue
        # opened a minute ago is found, even right after an app restart.
        result = self.analyze_github_issue(issue, repository, source)
        self._maybe_act_on_github(issue, repository, result)
        return result

    def _maybe_act_on_github(self, issue: dict, repository: str, result: dict) -> None:
        s = self.settings
        if repository.lower() not in s.bot_repositories:
            return  # suggestion-only for every repo not explicitly allowed
        owner, repo = repository.split("/", 1)
        number = int(issue["number"])
        try:
            if s.bot_post_comments:
                existing = self.github.list_comments(owner, repo, number)
                if any(BOT_MARKER in (c.get("body") or "") for c in existing):
                    self.metrics.bot_actions["skipped_existing_comment"] += 1
                else:
                    self.github.post_comment(owner, repo, number, render_comment(result))
                    self.metrics.bot_actions["comments_posted"] += 1
            label = result["classification"]["label"]
            if s.bot_apply_labels and label:
                self.github.add_labels(owner, repo, number, [label.lower()])
                self.metrics.bot_actions["labels_applied"] += 1
        except GitHubError as exc:  # a failed comment must never break anything else
            self.metrics.record_error("github_write")
            log.warning("could not act on %s#%s: %s", repository, number, exc)

    # ---------------------------------------------------------------- optional poller
    def start_poller(self) -> None:
        if not self.settings.poll_repositories or self._poll_thread:
            return
        self._poll_thread = threading.Thread(target=self._poll_loop, name="poller", daemon=True)
        self._poll_thread.start()

    def stop_poller(self) -> None:
        self._poll_stop.set()

    def poll_once(self) -> int:
        new_count = 0
        for full_name in self.settings.poll_repositories:
            owner, repo = full_name.split("/", 1)
            try:
                issues = self.github.list_recent_issues(owner, repo, count=20)
            except GitHubError as exc:
                self.metrics.record_error("poller")
                log.warning("poll %s failed: %s", full_name, exc)
                continue
            newest = max((int(i["number"]) for i in issues), default=0)
            last = self._poll_last_seen.get(full_name)
            self._poll_last_seen[full_name] = max(newest, last or 0)
            if last is None:
                continue  # first look only sets the baseline; old issues are not "new"
            for issue in sorted(issues, key=lambda i: i["number"]):
                if int(issue["number"]) > last:
                    self.submit(issue, full_name, "poller")
                    new_count += 1
        return new_count

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            self.poll_once()
            self._poll_stop.wait(self.settings.poll_interval_seconds)


# --------------------------------------------------------------------------- app factory
def create_app(
    settings: Settings | None = None,
    predictor: TriagePredictor | None = None,
    github: GitHubClient | None = None,
    load_model: bool = True,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service: TriageService = app.state.service
        if service.predictor is None and load_model:
            try:
                service.predictor = TriagePredictor.load(settings.model_dir, config_path=settings.config_path)
                service.predictor.warmup()  # load the embedding model now, not on the first request
            except FileNotFoundError:
                log.error("no model bundle in %s; run scripts/train_models.py", settings.model_dir)
        service.start_worker()
        service.start_poller()
        yield
        service.stop_poller()
        service.drain()
        service.github.close()
        service.store.close()

    app = FastAPI(title="GitHub Issue Auto-Labeler & Duplicate Finder", version="0.5", lifespan=lifespan)
    app.state.service = TriageService(settings, predictor, github or GitHubClient(token=settings.github_token))

    def svc() -> TriageService:
        return app.state.service

    @app.exception_handler(GitHubError)
    async def github_error(_: Request, exc: GitHubError):
        svc().metrics.record_error("rate_limited" if isinstance(exc, RateLimited) else "github")
        content = {"detail": str(exc)}
        if isinstance(exc, RateLimited):
            content["rate_limit_resets_at"] = exc.reset_at
        return JSONResponse(status_code=exc.status, content=content)

    @app.exception_handler(ValueError)
    async def bad_input(_: Request, exc: ValueError):
        svc().metrics.record_error("bad_input")
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        s = svc()
        return {
            "status": "ok" if s.predictor else "degraded: model not loaded",
            "model": None if not s.predictor else {
                **s.predictor.card.get("classifier", {}),
                "trained_at": s.predictor.card.get("trained_at"),
                "config_version": s.predictor.config.get("version"),
                "index": {
                    "method_used": s.predictor.index.method,
                    "documents": len(s.predictor.index.records),
                },
                "thresholds": {
                    "min_confidence": s.predictor.config["classification"]["min_confidence"],
                    "duplicate": s.predictor.config["retrieval"]["duplicate_threshold"],
                    "manual_review": s.predictor.config["retrieval"]["review_threshold"],
                },
            },
            "settings": s.settings.public_view(),
            "github_rate_limit": s.github.rate.as_dict(),
        }

    @app.get("/metrics")
    def metrics():
        return {**svc().metrics.snapshot(), "github_rate_limit": svc().github.rate.as_dict()}

    @app.get("/events")
    def events():
        return {"events": svc().metrics.events()}

    @app.post("/predict")
    def predict(req: PredictRequest):
        svc().metrics.record_request("predict")
        result = svc().predict(title=req.title, body=req.body, repository=req.repository)
        svc().metrics.record_prediction(result, "manual")
        return result

    @app.post("/analyze/issue-url")
    def analyze_issue_url(req: UrlRequest):
        s = svc()
        s.metrics.record_request("analyze_issue_url")
        owner, repo, number = parse_issue_url(req.url)
        issue = s.github.get_issue(owner, repo, number)
        return s.analyze_github_issue(issue, f"{owner}/{repo}", "issue_url")

    @app.post("/analyze/repo-url")
    def analyze_repo_url(req: UrlRequest):
        s = svc()
        s.metrics.record_request("analyze_repo_url")
        owner, repo = parse_repo_url(req.url)
        limit = min(req.max_issues, s.settings.repo_scan_max_issues)
        recent = s.github.list_recent_issues(owner, repo, count=100, state="all")  # one API call
        comparison = s.repository_index(f"{owner}/{repo}", live_issues=recent)
        rows, compared, agreed = [], 0, 0
        for issue in recent[:limit]:
            r = s.analyze_github_issue(issue, f"{owner}/{repo}", "repo_url", comparison=comparison)
            gh, cls, dup = r["github"], r["classification"], r["duplicates"]
            if gh["agrees_with_maintainers"] is not None:
                compared += 1
                agreed += int(gh["agrees_with_maintainers"])
            rows.append({
                "number": gh["number"], "url": gh["url"], "title": r["input"]["title"],
                "predicted": cls["label"], "top_guess": cls["top_guess"], "confidence": cls["confidence"],
                "classification_decision": cls["decision"],
                "maintainer_label": gh["maintainer_label_mapped"], "maintainer_labels": gh["maintainer_labels"],
                "agrees": gh["agrees_with_maintainers"],
                "duplicate_decision": dup["decision"],
                "top_match": dup["candidates"][0] if dup["candidates"] else None,
            })
        return {
            "repository": f"{owner}/{repo}",
            "issues_analyzed": len(rows),
            "github_api_calls_for_this_scan": 1,
            "compared_within_repository": len(comparison.records),
            "agreement_with_maintainers": {
                "issues_with_a_comparable_label": compared,
                "agreed": agreed,
                "rate": round(agreed / compared, 4) if compared else None,
            },
            "issues": rows,
        }

    # ------------------------------------------------------------------ database (local only)
    def local_only(request: Request) -> None:
        """The database pages can delete data, and ngrok makes the app public, so they only
        answer requests made on this computer. ngrok adds X-Forwarded-For to every request it
        forwards, which is how tunnelled traffic is told apart."""
        host = request.client.host if request.client else ""
        if "x-forwarded-for" in request.headers or host not in settings.local_hosts:
            raise HTTPException(403, "the database can only be managed from this computer")

    def _repo_or_404(full_name: str) -> dict:
        repo = svc().store.get_repository(full_name)
        if not repo:
            raise HTTPException(404, f"{full_name} is not in the database")
        return repo

    @app.get("/db/repos")
    def db_list_repos(request: Request):
        local_only(request)
        return {"database": str(settings.db_path), "repositories": svc().store.list_repositories()}

    @app.post("/db/repos", status_code=201)
    def db_add_repo(req: RepoRequest, request: Request):
        local_only(request)
        s = svc()
        owner, name = normalise_repository(req.repository)
        repo = s.store.add_repository(f"{owner}/{name}", source="manual")
        fetched = s.sync_repository(f"{owner}/{name}") if req.sync else 0
        return {"repository": s.store.get_repository(repo["full_name"]), "issues_fetched": fetched}

    @app.delete("/db/repos/{owner}/{name}")
    def db_delete_repo(owner: str, name: str, request: Request):
        local_only(request)
        if not svc().store.delete_repository(f"{owner}/{name}"):
            raise HTTPException(404, f"{owner}/{name} is not in the database")
        return {"deleted": f"{owner}/{name}"}

    @app.post("/db/repos/{owner}/{name}/sync")
    def db_sync_repo(owner: str, name: str, request: Request):
        local_only(request)
        _repo_or_404(f"{owner}/{name}")
        fetched = svc().sync_repository(f"{owner}/{name}")
        return {"repository": svc().store.get_repository(f"{owner}/{name}"), "issues_fetched": fetched}

    @app.get("/db/repos/{owner}/{name}/issues")
    def db_list_issues(owner: str, name: str, request: Request):
        local_only(request)
        repo = _repo_or_404(f"{owner}/{name}")
        return {"repository": repo, "issues": svc().store.list_issues(f"{owner}/{name}")}

    @app.get("/db/repos/{owner}/{name}/issues/{number}")
    def db_get_issue(owner: str, name: str, number: int, request: Request):
        local_only(request)
        issue = svc().store.get_issue(f"{owner}/{name}", number)
        if not issue:
            raise HTTPException(404, f"issue {number} is not in the database")
        return issue

    @app.post("/db/repos/{owner}/{name}/issues", status_code=201)
    def db_add_issue(owner: str, name: str, req: IssueIn, request: Request):
        local_only(request)
        data = req.model_dump(exclude_none=True)
        data["labels"] = [{"name": label} for label in req.labels]
        return svc().store.upsert_issue(f"{owner}/{name}", data, source="manual")

    @app.delete("/db/repos/{owner}/{name}/issues/{number}")
    def db_delete_issue(owner: str, name: str, number: int, request: Request):
        local_only(request)
        if not svc().store.delete_issue(f"{owner}/{name}", number):
            raise HTTPException(404, f"issue {number} is not in the database")
        return {"deleted": number, "repository": f"{owner}/{name}"}

    @app.post("/webhook/github", status_code=202)
    async def webhook(request: Request):
        s = svc()
        s.metrics.webhook["received"] += 1
        body = await request.body()
        if s.settings.webhook_secret:
            if not verify_signature(s.settings.webhook_secret, body, request.headers.get("x-hub-signature-256")):
                s.metrics.webhook["bad_signature"] += 1
                raise HTTPException(401, "invalid webhook signature")
        elif not s.settings.allow_unsigned_webhooks:
            raise HTTPException(503, "webhook secret not configured (set GITHUB_WEBHOOK_SECRET)")

        event = request.headers.get("x-github-event", "")
        if event == "ping":
            return {"ok": True, "message": "pong"}
        payload = json.loads(body or b"{}")
        action = payload.get("action")
        sync_actions = {"edited", "labeled", "unlabeled", "closed", "reopened", "deleted"}
        if event != "issues" or action not in {"opened"} | sync_actions:
            s.metrics.webhook["ignored"] += 1
            return {"accepted": False, "reason": f"ignored event {event}/{action}"}
        if not s.deliveries.first_time(request.headers.get("x-github-delivery")):
            s.metrics.webhook["duplicate_delivery"] += 1
            return {"accepted": False, "reason": "already processed this delivery"}

        if not isinstance(payload.get("repository"), dict) or not isinstance(payload.get("issue"), dict):
            s.metrics.webhook["ignored"] += 1
            return {"accepted": False, "reason": "payload has no repository or issue"}
        repository = payload["repository"]["full_name"]
        issue = payload["issue"]
        if action in sync_actions and not s.store.get_repository(repository):
            s.metrics.webhook["ignored"] += 1  # a repository we don't track: nothing to update
            return {"accepted": False, "reason": f"{repository} is not in the database"}
        # GitHub needs an answer within 10 s, so we queue the work and reply at once.
        s.submit(issue, repository, "webhook", action=action)
        return {"accepted": True, "queued": True, "action": action, "repository": repository,
                "issue": issue.get("number")}

    return app


app = create_app()
