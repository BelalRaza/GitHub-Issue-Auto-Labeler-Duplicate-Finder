"""GitHub REST client for the live service: rate-limit aware, cached, with retries.

Rules it follows (GitHub REST docs, checked 25 Nov 2026):
- Primary limit hit (403/429 with x-ratelimit-remaining: 0): do not retry before
  x-ratelimit-reset. We raise RateLimited so the caller can tell the user when it resets.
- Secondary limit (403/429 with retry-after): wait that many seconds. Short waits are retried
  once; long waits raise RateLimited.
- Conditional requests: we send If-None-Match with the last ETag. A 304 means "unchanged"
  and we reuse the cached body.
- Transient 5xx / network errors: up to 2 retries with exponential backoff.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

API = "https://api.github.com"
API_VERSION = "2026-03-10"
USER_AGENT = "github-issue-auto-labeler/0.5"

ISSUE_URL_RE = re.compile(r"^https?://github\.com/([\w.-]+)/([\w.-]+)/issues/(\d+)/?(?:[#?].*)?$")
REPO_URL_RE = re.compile(r"^https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?(?:[#?].*)?$")


class GitHubError(RuntimeError):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class RateLimited(GitHubError):
    def __init__(self, message: str, reset_at: str | None):
        super().__init__(message, status=429)
        self.reset_at = reset_at


def parse_issue_url(url: str) -> tuple[str, str, int]:
    match = ISSUE_URL_RE.match(url.strip())
    if not match:
        raise ValueError("expected an issue link like https://github.com/owner/repo/issues/123")
    return match.group(1), match.group(2), int(match.group(3))


def parse_repo_url(url: str) -> tuple[str, str]:
    url = url.strip()
    if ISSUE_URL_RE.match(url):
        owner, repo, _ = parse_issue_url(url)
        return owner, repo
    match = REPO_URL_RE.match(url)
    if not match or match.group(2) in {"issues", "pulls"}:
        raise ValueError("expected a repository link like https://github.com/owner/repo")
    return match.group(1), match.group(2)


@dataclass
class RateLimitState:
    limit: int | None = None
    remaining: int | None = None
    reset_epoch: int | None = None
    resource: str | None = None
    calls_made: int = 0
    not_modified_304: int = 0
    rate_limited_responses: int = 0
    errors: int = 0

    def reset_iso(self) -> str | None:
        if not self.reset_epoch:
            return None
        return datetime.fromtimestamp(self.reset_epoch, timezone.utc).isoformat(timespec="seconds")

    def as_dict(self) -> dict:
        return {
            "limit": self.limit,
            "remaining": self.remaining,
            "resets_at": self.reset_iso(),
            "resource": self.resource,
            "calls_made": self.calls_made,
            "not_modified_304": self.not_modified_304,
            "rate_limited_responses": self.rate_limited_responses,
            "errors": self.errors,
        }


@dataclass
class _CacheEntry:
    etag: str | None
    body: Any
    fetched_at: float = field(default_factory=time.monotonic)


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 2,
        max_inline_wait: float = 5.0,
        cache_ttl_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.token = token
        self.max_retries = max_retries
        self.max_inline_wait = max_inline_wait
        self.cache_ttl = cache_ttl_seconds
        self.rate = RateLimitState()
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": USER_AGENT,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(base_url=API, headers=headers, timeout=timeout, transport=transport)

    @property
    def authenticated(self) -> bool:
        return bool(self.token)

    # ------------------------------------------------------------------ core request
    def _record_headers(self, response: httpx.Response) -> None:
        h = response.headers
        with self._lock:
            if "x-ratelimit-limit" in h:
                self.rate.limit = int(h["x-ratelimit-limit"])
                self.rate.remaining = int(h.get("x-ratelimit-remaining", 0))
                self.rate.reset_epoch = int(h.get("x-ratelimit-reset", 0)) or None
                self.rate.resource = h.get("x-ratelimit-resource")

    def request(self, method: str, path: str, *, json: Any = None, use_cache: bool = True) -> Any:
        cache_key = f"{method} {path}"
        cached = self._cache.get(cache_key) if method == "GET" and use_cache else None
        if cached and time.monotonic() - cached.fetched_at < self.cache_ttl:
            return cached.body  # fresh enough: no API call at all

        headers = {"If-None-Match": cached.etag} if cached and cached.etag else {}
        attempt = 0
        while True:
            try:
                with self._lock:
                    self.rate.calls_made += 1
                response = self._http.request(method, path, json=json, headers=headers)
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(0.5 * 2**attempt)
                    continue
                self.rate.errors += 1
                raise GitHubError("GitHub did not answer in time", status=504) from exc
            except httpx.HTTPError as exc:
                self.rate.errors += 1
                raise GitHubError(f"could not reach GitHub: {exc}", status=502) from exc

            self._record_headers(response)
            status = response.status_code

            if status == 304 and cached:
                with self._lock:
                    self.rate.not_modified_304 += 1
                cached.fetched_at = time.monotonic()
                return cached.body

            if status in (403, 429) and (
                response.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in response.headers
            ):
                with self._lock:
                    self.rate.rate_limited_responses += 1
                retry_after = response.headers.get("retry-after")
                if retry_after and float(retry_after) <= self.max_inline_wait and attempt < 1:
                    attempt += 1
                    time.sleep(float(retry_after))
                    continue
                hint = "" if self.authenticated else " Add a GITHUB_TOKEN to raise the limit from 60 to 5,000/hour."
                raise RateLimited(
                    f"GitHub rate limit reached; resets at {self.rate.reset_iso() or 'unknown'}.{hint}",
                    self.rate.reset_iso(),
                )

            if status >= 500 and attempt < self.max_retries:
                attempt += 1
                time.sleep(0.5 * 2**attempt)
                continue

            if status == 404:
                raise GitHubError("not found on GitHub (or the repository is private)", status=404)
            if status == 410:
                raise GitHubError("this issue was deleted on GitHub", status=410)
            if status >= 400:
                self.rate.errors += 1
                try:
                    message = response.json().get("message", "")
                except ValueError:
                    message = response.text[:200]
                raise GitHubError(f"GitHub returned HTTP {status}: {message}", status=502)

            body = response.json() if response.content else None
            if method == "GET":
                self._cache[cache_key] = _CacheEntry(response.headers.get("etag"), body)
            return body

    # ------------------------------------------------------------------ endpoints we use
    def get_issue(self, owner: str, repo: str, number: int) -> dict:
        return self.request("GET", f"/repos/{owner}/{repo}/issues/{number}")

    def list_recent_issues(
        self, owner: str, repo: str, count: int = 30, state: str = "open", fresh: bool = False
    ) -> list[dict]:
        """Newest issues first; pull requests (which this endpoint also returns) are removed.

        Always asks for the maximum 100 items: on busy repos most of the newest items are pull
        requests, and a bigger page costs the same single API call.
        """
        items = self.request(
            "GET", f"/repos/{owner}/{repo}/issues?state={state}&sort=created&direction=desc&per_page=100",
            use_cache=not fresh,  # fresh=True: a just-opened issue must see the one opened a minute ago
        )
        return [item for item in items if "pull_request" not in item][:count]

    def list_comments(self, owner: str, repo: str, number: int) -> list[dict]:
        return self.request("GET", f"/repos/{owner}/{repo}/issues/{number}/comments?per_page=100", use_cache=False)

    def post_comment(self, owner: str, repo: str, number: int, body: str) -> dict:
        return self.request("POST", f"/repos/{owner}/{repo}/issues/{number}/comments", json={"body": body})

    def add_labels(self, owner: str, repo: str, number: int, labels: list[str]) -> Any:
        return self.request("POST", f"/repos/{owner}/{repo}/issues/{number}/labels", json={"labels": labels})

    def refresh_rate_limit(self) -> dict:
        """GET /rate_limit does not count against the primary limit."""
        data = self.request("GET", "/rate_limit", use_cache=False)
        core = (data or {}).get("resources", {}).get("core", {})
        with self._lock:
            self.rate.limit = core.get("limit", self.rate.limit)
            self.rate.remaining = core.get("remaining", self.rate.remaining)
            self.rate.reset_epoch = core.get("reset", self.rate.reset_epoch)
            self.rate.calls_made -= 1  # not a real budget call
        return self.rate.as_dict()

    def close(self) -> None:
        self._http.close()
