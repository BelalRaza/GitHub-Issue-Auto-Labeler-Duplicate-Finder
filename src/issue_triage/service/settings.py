"""Service settings, read from environment variables (and an optional local .env file).

Safe defaults: the bot only *suggests*. Posting comments or applying labels must be switched on
explicitly, and only for repositories listed in BOT_REPOSITORIES.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_dotenv(path: Path) -> None:
    """Minimal .env reader. Real environment variables always win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


@dataclass
class Settings:
    model_dir: Path = ROOT / "models"
    config_path: Path = ROOT / "config" / "model_config.json"
    taxonomy_path: Path = ROOT / "config" / "label_taxonomy.json"
    github_token: str | None = None
    webhook_secret: str | None = None
    allow_unsigned_webhooks: bool = False  # only for local testing
    bot_post_comments: bool = False
    bot_apply_labels: bool = False
    bot_repositories: list[str] = field(default_factory=list)
    poll_repositories: list[str] = field(default_factory=list)
    poll_interval_seconds: int = 60
    repo_scan_max_issues: int = 30
    db_path: Path = ROOT / "data" / "triage.db"
    # Hosts allowed to manage the database (requests forwarded by ngrok are always refused).
    local_hosts: frozenset = frozenset({"127.0.0.1", "::1", "localhost"})

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env")
        return cls(
            model_dir=Path(os.environ.get("MODEL_DIR", ROOT / "models")),
            config_path=Path(os.environ.get("MODEL_CONFIG", ROOT / "config" / "model_config.json")),
            github_token=os.environ.get("GITHUB_TOKEN") or None,
            webhook_secret=os.environ.get("GITHUB_WEBHOOK_SECRET") or None,
            allow_unsigned_webhooks=_flag("ALLOW_UNSIGNED_WEBHOOKS"),
            bot_post_comments=_flag("BOT_POST_COMMENTS"),
            bot_apply_labels=_flag("BOT_APPLY_LABELS"),
            bot_repositories=[r.lower() for r in _list("BOT_REPOSITORIES")],
            poll_repositories=_list("POLL_REPOSITORIES"),
            # Never poll faster than once a minute (see docs/PROJECT_PLAN.md section 5).
            poll_interval_seconds=max(60, int(os.environ.get("POLL_INTERVAL_SECONDS", "60"))),
            repo_scan_max_issues=min(100, int(os.environ.get("REPO_SCAN_MAX_ISSUES", "30"))),
            db_path=Path(os.environ.get("TRIAGE_DB", ROOT / "data" / "triage.db")),
        )

    def public_view(self) -> dict:
        """What /health shows. Never includes the token or the webhook secret."""
        return {
            "github_token_configured": bool(self.github_token),
            "webhook_secret_configured": bool(self.webhook_secret),
            "bot_post_comments": self.bot_post_comments,
            "bot_apply_labels": self.bot_apply_labels,
            "bot_repositories": self.bot_repositories,
            "poll_repositories": self.poll_repositories,
            "poll_interval_seconds": self.poll_interval_seconds,
            "database": str(self.db_path),
        }
