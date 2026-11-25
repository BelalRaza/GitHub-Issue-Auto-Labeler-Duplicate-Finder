"""Cleaning, taxonomy mapping, and deterministic split helpers."""

from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
FENCED_CODE_RE = re.compile(r"```(?:[^\n`]*)\n?(.*?)```", re.DOTALL)
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
# Keep our explicit model tokens while removing ordinary HTML tags.
HTML_TAG_RE = re.compile(r"</?(?!(?:CODE_BLOCK|URL|USER)>)[A-Za-z][^>]*>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_issue_text(value: str | None) -> str:
    """Normalize presentation noise while retaining code/error content."""
    if not value:
        return ""
    text = html.unescape(value.replace("\r\n", "\n").replace("\r", "\n"))
    text = HTML_COMMENT_RE.sub(" ", text)
    text = FENCED_CODE_RE.sub(lambda match: f" <CODE_BLOCK> {match.group(1)} </CODE_BLOCK> ", text)
    text = MARKDOWN_IMAGE_RE.sub(lambda match: f" {match.group(1)} ", text)
    text = MARKDOWN_LINK_RE.sub(lambda match: f" {match.group(1)} ", text)
    text = URL_RE.sub(" <URL> ", text)
    text = MENTION_RE.sub(" <USER> ", text)
    text = text.replace("`", "")
    text = HTML_TAG_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def load_taxonomy(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping_for_repository(taxonomy: dict[str, Any], repository: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for target, labels in taxonomy.get("global", {}).items():
        for label in labels:
            mapping[label.casefold()] = target
    for target, labels in taxonomy.get("repository_overrides", {}).get(repository, {}).items():
        for label in labels:
            mapping[label.casefold()] = target
    return mapping


def map_labels(labels: Iterable[str], repository: str, taxonomy: dict[str, Any]) -> list[str]:
    mapping = _mapping_for_repository(taxonomy, repository)
    return sorted({mapping[label.casefold()] for label in labels if label.casefold() in mapping})


def map_issue_type(issue_type: str, taxonomy: dict[str, Any]) -> list[str]:
    mapped = []
    for target, names in taxonomy.get("issue_types", {}).items():
        if issue_type.casefold() in {name.casefold() for name in names}:
            mapped.append(target)
    return sorted(set(mapped))


def is_duplicate_label(labels: Iterable[str], taxonomy: dict[str, Any]) -> bool:
    aliases = {item.casefold() for item in taxonomy.get("duplicate_evidence_labels", [])}
    return any(label.casefold() in aliases for label in labels)


def read_raw_issues(input_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(input_dir.glob("*.jsonl")):
        # Iterate physical LF-delimited records. str.splitlines() also treats Unicode
        # U+2028/U+2029 inside legitimate issue bodies as record separators.
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def flatten_issue(wrapper: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any]:
    issue = wrapper["issue"]
    repository = wrapper["repository"]
    labels = [item["name"] if isinstance(item, dict) else str(item) for item in issue.get("labels", [])]
    labels_standardized = map_labels(labels, repository, taxonomy)
    issue_type_value = issue.get("type")
    if isinstance(issue_type_value, dict):
        issue_type_value = issue_type_value.get("name") or ""
    issue_type_original = str(issue_type_value or "")
    issue_type_standardized = map_issue_type(issue_type_original, taxonomy)
    standardized = sorted(set(labels_standardized) | set(issue_type_standardized))
    title = (issue.get("title") or "").strip()
    body = issue.get("body") or ""
    body_clean = clean_issue_text(body)
    text_clean = clean_issue_text(f"{title}\n\n{body}")
    target = standardized[0] if len(standardized) == 1 else ""
    return {
        "repository": repository,
        "issue_id": issue.get("id"),
        "issue_number": issue.get("number"),
        "url": issue.get("html_url"),
        "title": title,
        "body": body,
        "body_clean": body_clean,
        "text_clean": text_clean,
        "labels_original": json.dumps(labels, ensure_ascii=False),
        "labels_standardized": json.dumps(labels_standardized),
        "issue_type_original": issue_type_original,
        "issue_type_standardized": json.dumps(issue_type_standardized),
        "target_source": (
            "labels+issue_type"
            if labels_standardized and issue_type_standardized
            else "labels"
            if labels_standardized
            else "issue_type"
            if issue_type_standardized
            else ""
        ),
        "classification_target": target,
        "taxonomy_conflict": len(standardized) > 1,
        "classification_eligible": bool(title and target),
        "state": issue.get("state"),
        "state_reason": issue.get("state_reason"),
        "is_duplicate_verified": issue.get("state_reason") == "duplicate",
        "has_duplicate_label": is_duplicate_label(labels, taxonomy),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "comments_count": issue.get("comments", 0),
        "author_association": issue.get("author_association"),
        "body_missing": not bool(body.strip()),
        "title_length_chars": len(title),
        "body_length_chars": len(body),
        "text_length_words": len(text_clean.split()),
        "collected_at": wrapper.get("collected_at"),
        "source_endpoint": wrapper.get("source_endpoint"),
        "collection_routes": json.dumps(wrapper.get("collection_routes", []), ensure_ascii=False),
        "split": "",
    }


def assign_chronological_splits(rows: list[dict[str, Any]]) -> None:
    """Assign global chronological 70/15/15 splits to classification-eligible rows."""
    eligible = [row for row in rows if row["classification_eligible"] and row.get("created_at")]
    eligible.sort(key=lambda row: (row["created_at"], row["repository"], row["issue_number"] or 0))
    count = len(eligible)
    train_end = int(count * 0.70)
    validation_end = train_end + int(count * 0.15)
    for index, row in enumerate(eligible):
        if index < train_end:
            row["split"] = "train"
        elif index < validation_end:
            row["split"] = "validation"
        else:
            row["split"] = "test"


def build_quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    repositories = Counter(row["repository"] for row in rows)
    targets = Counter(row["classification_target"] for row in rows if row["classification_target"])
    splits = Counter(row["split"] for row in rows if row["split"])
    labels = Counter()
    for row in rows:
        labels.update(json.loads(row["labels_original"]))
    eligible_lengths = [row["text_length_words"] for row in rows if row["classification_eligible"]]
    eligible_lengths.sort()

    def percentile(values: list[int], fraction: float) -> int | None:
        if not values:
            return None
        return values[min(int((len(values) - 1) * fraction), len(values) - 1)]

    unique_keys = {(row["repository"], row["issue_number"]) for row in rows}
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_rows": len(rows),
        "unique_repository_issue_keys": len(unique_keys),
        "duplicate_rows_by_key": len(rows) - len(unique_keys),
        "repositories": dict(sorted(repositories.items())),
        "classification_eligible_rows": sum(bool(row["classification_eligible"]) for row in rows),
        "target_distribution": dict(sorted(targets.items())),
        "target_source_distribution": dict(
            sorted(
                Counter(
                    row["target_source"]
                    for row in rows
                    if row["classification_eligible"] and row["target_source"]
                ).items()
            )
        ),
        "split_distribution": dict(sorted(splits.items())),
        "taxonomy_conflicts": sum(bool(row["taxonomy_conflict"]) for row in rows),
        "missing_bodies": sum(bool(row["body_missing"]) for row in rows),
        "verified_duplicate_status_rows": sum(bool(row["is_duplicate_verified"]) for row in rows),
        "duplicate_label_rows": sum(bool(row["has_duplicate_label"]) for row in rows),
        "text_length_words": {
            "median": percentile(eligible_lengths, 0.50),
            "p90": percentile(eligible_lengths, 0.90),
            "max": max(eligible_lengths) if eligible_lengths else None,
        },
        "top_original_labels": labels.most_common(30),
        "model_contract": {
            "feature_columns": ["text_clean"],
            "optional_context_columns": ["repository"],
            "target_column": "classification_target",
            "metadata_not_features": [
                "issue_id",
                "issue_number",
                "url",
                "labels_original",
                "issue_type_original",
                "state",
                "state_reason",
                "updated_at",
                "closed_at",
                "comments_count",
                "is_duplicate_verified",
                "has_duplicate_label",
            ],
        },
        "warnings": [
            "Pilot search snapshots contain only the newest issues and are not a representative training corpus.",
            "Current labels are retrospective targets; label assignment timestamps are not available in the issue list response.",
            "Verified duplicate status does not always expose the canonical issue in REST list results; use GraphQL duplicateOf or audited maintainer comments for pair extraction.",
            "The chronological split is provisional until the full-history collection and duplicate-pair audit are complete.",
        ],
    }

