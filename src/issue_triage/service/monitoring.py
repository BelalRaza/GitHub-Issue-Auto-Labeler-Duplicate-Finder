"""In-memory monitoring: request counts, latency percentiles, decisions, errors, recent events.

Privacy: we keep issue titles and decisions for the live feed, never issue bodies or authors.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone

import numpy as np


class Metrics:
    def __init__(self, max_latencies: int = 2000, max_events: int = 50):
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests = Counter()  # by endpoint
        self.errors = Counter()  # by kind
        self.classification_decisions = Counter()
        self.predicted_labels = Counter()
        self.duplicate_decisions = Counter()
        self.webhook = Counter()  # received / processed / ignored / duplicate_delivery / bad_signature
        self.bot_actions = Counter()  # comments_posted / labels_applied / skipped_existing_comment
        self.issues_processed = 0
        self._latencies = deque(maxlen=max_latencies)
        self._events = deque(maxlen=max_events)

    def record_request(self, endpoint: str) -> None:
        with self._lock:
            self.requests[endpoint] += 1

    def record_error(self, kind: str) -> None:
        with self._lock:
            self.errors[kind] += 1

    def record_prediction(self, result: dict, source: str, issue: dict | None = None) -> None:
        with self._lock:
            self.issues_processed += 1
            self._latencies.append(result["latency_ms"])
            cls, dup = result["classification"], result["duplicates"]
            self.classification_decisions[cls["decision"]] += 1
            if cls["label"]:
                self.predicted_labels[cls["label"]] += 1
            self.duplicate_decisions[dup["decision"]] += 1
            if issue is not None:
                top = dup["candidates"][0] if dup["candidates"] else None
                self._events.appendleft(
                    {
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "source": source,
                        "repository": issue.get("repository"),
                        "issue_number": issue.get("number"),
                        "url": issue.get("url"),
                        "title": result["input"]["title"][:140],
                        "label": cls["label"],
                        "top_guess": cls["top_guess"],
                        "confidence": cls["confidence"],
                        "classification_decision": cls["decision"],
                        "duplicate_decision": dup["decision"],
                        "top_match": None if not top else {
                            "repository": top["repository"], "issue_number": top["issue_number"],
                            "url": top["url"], "similarity": top["similarity"],
                        },
                        "latency_ms": result["latency_ms"],
                    }
                )

    def events(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def snapshot(self) -> dict:
        with self._lock:
            lat = np.array(self._latencies) if self._latencies else None
            decisions = sum(self.classification_decisions.values())
            return {
                "uptime_seconds": int(time.time() - self.started),
                "issues_processed": self.issues_processed,
                "requests_by_endpoint": dict(self.requests),
                "errors": dict(self.errors),
                "latency_ms": None if lat is None else {
                    "p50": round(float(np.percentile(lat, 50)), 1),
                    "p95": round(float(np.percentile(lat, 95)), 1),
                    "max": round(float(lat.max()), 1),
                    "samples": int(lat.size),
                },
                "classification_decisions": dict(self.classification_decisions),
                "manual_review_rate": round(
                    self.classification_decisions["needs_manual_review"] / decisions, 4
                ) if decisions else None,
                "predicted_labels": dict(self.predicted_labels),
                "duplicate_decisions": dict(self.duplicate_decisions),
                "webhook": dict(self.webhook),
                "bot_actions": dict(self.bot_actions),
            }
