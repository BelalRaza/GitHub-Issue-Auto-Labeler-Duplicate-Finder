"""Webhook security and the bot's comment text."""

from __future__ import annotations

import hashlib
import hmac
import threading
from collections import OrderedDict

BOT_MARKER = "<!-- issue-triage-bot -->"


def verify_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    """GitHub signs each delivery: X-Hub-Signature-256 = 'sha256=' + HMAC-SHA256(secret, body)."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)  # constant-time comparison


class DeliveryDeduper:
    """Remembers recent X-GitHub-Delivery IDs. A redelivery keeps the same ID, so we never
    process (or comment on) the same event twice."""

    def __init__(self, max_size: int = 5000):
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._max = max_size
        self._lock = threading.Lock()

    def first_time(self, delivery_id: str | None) -> bool:
        if not delivery_id:
            return True
        with self._lock:
            if delivery_id in self._seen:
                return False
            self._seen[delivery_id] = None
            if len(self._seen) > self._max:
                self._seen.popitem(last=False)
            return True


def render_comment(result: dict) -> str:
    cls, dup = result["classification"], result["duplicates"]
    lines = [BOT_MARKER, "### 🤖 Triage suggestion", ""]
    if cls["decision"] == "suggest_label":
        lines.append(f"**Suggested label:** `{cls['label']}` (confidence {cls['confidence']:.2f})")
    else:
        lines.append(
            f"**Suggested label:** none, needs a maintainer's review "
            f"({cls['review_reason']}; best guess `{cls['top_guess']}`)"
        )
    lines.append("")
    top = dup["candidates"][0] if dup["candidates"] else None
    link = f"[{top['repository']}#{top['issue_number']}]({top['url']})" if top else ""
    if dup["decision"] == "possible_duplicate":
        lines.append(
            f"**Possible duplicate of** {link} (similarity {top['similarity']:.2f}). "
            "Please check before closing."
        )
    elif dup["decision"] == "check_similar":
        lines.append(
            f"**Might be the same as** {link} (similarity {top['similarity']:.2f}). "
            f"Please check by hand: {dup['reason']}."
        )
    elif top:
        lines.append(
            f"**Not sure.** No strong match; the closest earlier issue is {link} "
            f"(similarity {top['similarity']:.2f})."
        )
    else:
        lines.append("**No earlier issues to compare with yet.**")
    if dup["candidates"]:
        lines += ["", "| Earlier issue | Similarity |", "| --- | ---: |"]
        for c in dup["candidates"][:3]:
            title = c["title"][:80].replace("|", "\\|")
            lines.append(f"| [{c['repository']}#{c['issue_number']}]({c['url']}) {title} | {c['similarity']:.2f} |")
    lines += [
        "",
        "<sub>Automatic suggestion from the NLP issue triage prototype. It never closes or edits "
        "issues by itself; a maintainer always decides.</sub>",
    ]
    return "\n".join(lines)
