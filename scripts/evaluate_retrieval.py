#!/usr/bin/env python3
"""Compare retrieval methods (TF-IDF, BM25, embeddings, hybrid).

Two modes:

--pairs FILE   REAL duplicate evaluation. FILE comes from scripts/collect_duplicate_pairs.py
               (GitHub's structured duplicateOf). For each duplicate, we search only issues
               created before it and record the rank of its canonical issue.

(no --pairs)   PROXY sanity check, used until real pairs are collected: query = an issue's
               TITLE, corpus = every issue's BODY. Can the method find the body that belongs to
               the title? This is NOT duplicate detection, and is reported as a proxy only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from issue_triage.datasets import load_index_records  # noqa: E402
from issue_triage.metrics import retrieval_metrics  # noqa: E402
from issue_triage.preparation import clean_issue_text  # noqa: E402
from issue_triage.retrieval import METHODS, IssueRecord, RetrievalIndex  # noqa: E402

csv.field_size_limit(10_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issues", type=Path, default=ROOT / "data/processed/issues_clean.csv")
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/model_config.json")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--log", type=Path, default=ROOT / "experiments/experiment_log.jsonl")
    parser.add_argument("--min-body-words", type=int, default=20)
    return parser.parse_args()


def rank_of(results: list[dict], key: tuple[str, int]) -> int | None:
    for position, item in enumerate(results, start=1):
        if (item["repository"], item["issue_number"]) == key:
            return position
    return None


def proxy_queries(issues_csv: Path, min_body_words: int) -> tuple[list[IssueRecord], list[tuple[str, tuple]]]:
    corpus, queries = [], []
    with issues_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            body = row["body_clean"]
            if len(body.split()) < min_body_words or not row["title"].strip():
                continue
            record = IssueRecord(
                row["repository"], int(row["issue_number"]), row["url"], row["title"], row["created_at"], body
            )
            corpus.append(record)
            queries.append((clean_issue_text(row["title"]), record.key, None))
    return corpus, queries


def pair_queries(issues_csv: Path, pairs_csv: Path) -> tuple[list[IssueRecord], list, int]:
    corpus = load_index_records(issues_csv)
    by_key = {r.key: r for r in corpus}
    queries, skipped = [], 0
    with pairs_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dup_key = (row["duplicate_repository"], int(row["duplicate_issue_number"]))
            canon_key = (row["canonical_repository"], int(row["canonical_issue_number"]))
            duplicate, canonical = by_key.get(dup_key), by_key.get(canon_key)
            # A pair is only a valid test if both issues are in our corpus and the canonical
            # issue already existed when the duplicate was filed.
            if not duplicate or not canonical or canonical.created_at >= duplicate.created_at:
                skipped += 1
                continue
            queries.append((duplicate.text, canon_key, (duplicate.created_at, dup_key)))
    return corpus, queries, skipped


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mode = "duplicate_pairs" if args.pairs else "proxy_title_to_body"
    if args.pairs:
        corpus, queries, skipped = pair_queries(args.issues, args.pairs)
    else:
        (corpus, queries), skipped = proxy_queries(args.issues, args.min_body_words), 0
    print(f"mode={mode} corpus={len(corpus)} queries={len(queries)} skipped={skipped}")

    for method in args.methods:
        index = RetrievalIndex(
            corpus,
            method=method,
            embedding_model=config["embedding_model"],
            hybrid_weight_embedding=config["retrieval"]["hybrid_weight_embedding"],
        )
        started = time.perf_counter()
        ranks = []
        for text, target, context in queries:
            created_before, exclude = context if context else (None, None)
            results = index.search(text, k=10, created_before=created_before, exclude=exclude)
            ranks.append(rank_of(results, target))
        seconds = time.perf_counter() - started
        metrics = retrieval_metrics(ranks)
        metrics["ms_per_query"] = round(1000 * seconds / max(len(queries), 1), 2)
        run = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "milestone": "M3",
            "task": "retrieval",
            "mode": mode,
            "model": index.method,
            "params": {"hybrid_weight_embedding": config["retrieval"]["hybrid_weight_embedding"]}
            if method == "hybrid"
            else {},
            "corpus": len(corpus),
            "skipped_pairs": skipped,
            "metrics": metrics,
        }
        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(run) + "\n")
        print(f"{index.method:10s} {json.dumps(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
