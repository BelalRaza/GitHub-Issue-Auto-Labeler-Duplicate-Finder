"""Load the M2 prepared CSVs into model-ready structures."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Sequence

from .retrieval import IssueRecord

csv.field_size_limit(10_000_000)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_classification_splits(path: Path, classes: Sequence[str]) -> dict[str, dict[str, list]]:
    """Returns {"train": {"texts": [...], "labels": [...], "rows": [...]}, "validation": ..., "test": ...}.

    Rows whose target is outside `classes` (e.g. the single Question example) are dropped.
    """
    splits: dict[str, dict[str, list]] = {
        name: {"texts": [], "labels": [], "rows": []} for name in ("train", "validation", "test")
    }
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["classification_target"] not in classes or row["split"] not in splits:
                continue
            bucket = splits[row["split"]]
            bucket["texts"].append(row["text_clean"])
            bucket["labels"].append(row["classification_target"])
            bucket["rows"].append(row)
    return splits


def load_index_records(path: Path) -> list[IssueRecord]:
    """Every collected issue (not only classification-eligible ones) can be a duplicate target."""
    records = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["text_clean"].strip():
                continue
            records.append(
                IssueRecord(
                    repository=row["repository"],
                    issue_number=int(row["issue_number"]),
                    url=row["url"],
                    title=row["title"],
                    created_at=row["created_at"],
                    text=row["text_clean"],
                )
            )
    return records
