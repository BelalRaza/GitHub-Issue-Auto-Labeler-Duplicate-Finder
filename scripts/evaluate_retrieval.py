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

import numpy as np  # noqa: E402

from issue_triage.datasets import load_index_records  # noqa: E402
from issue_triage.embeddings import Encoder  # noqa: E402
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
    parser.add_argument("--negatives-per-half", type=int, default=1500)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/m4/retrieval_pairs.json")
    parser.add_argument(
        "--scope", choices=["same_repository", "all"], default="same_repository",
        help="same_repository = search only the query issue's own repo (as the app does)",
    )
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
    queries.sort(key=lambda q: q[2][0])  # oldest duplicate first (used to split tune/test halves)
    return corpus, queries, skipped


def decision_analysis(index: RetrievalIndex, queries: list, issues_csv: Path, per_half: int, same_repo: bool) -> dict:
    """Pick the duplicate threshold on the OLDER half of the pairs, report it on the NEWER half.

    Positives: verified duplicates; a hit only counts if the top-1 result IS the canonical issue.
    Negatives: ordinary issues from the same time window that were not closed as duplicates; any
    flag on them is a false alarm. (Some are unrecorded real repeats, so this over-counts.)
    """
    with issues_csv.open(encoding="utf-8", newline="") as handle:
        ordinary = sorted(
            (r for r in csv.DictReader(handle) if r["state_reason"] != "duplicate" and r["text_clean"].strip()),
            key=lambda r: r["created_at"],
        )
    half = len(queries) // 2
    halves = {"tune (older pairs)": queries[:half], "test (newer pairs)": queries[half:]}
    rng = np.random.default_rng(13)
    scored: dict[str, dict] = {}
    for name, part in halves.items():
        lo, hi = part[0][2][0], part[-1][2][0]
        window = [r for r in ordinary if lo <= r["created_at"] <= hi]
        if len(window) > per_half:
            window = [window[i] for i in sorted(rng.choice(len(window), per_half, replace=False))]
        positives = []
        for text, target, (created, key) in part:
            hits = index.search(text, k=1, created_before=created, exclude=key,
                                repository=key[0] if same_repo else None)
            positives.append(
                (float(hits[0]["similarity"]), (hits[0]["repository"], hits[0]["issue_number"]) == target)
                if hits else (0.0, False)
            )
        negatives = []
        for r in window:
            hits = index.search(r["text_clean"], k=1, created_before=r["created_at"],
                                repository=r["repository"] if same_repo else None,
                                exclude=(r["repository"], int(r["issue_number"])))
            negatives.append(float(hits[0]["similarity"]) if hits else 0.0)
        scored[name] = {"positives": positives, "negatives": negatives, "window": [lo, hi]}

    def at(threshold: float, data: dict) -> dict:
        tp = sum(1 for s, ok in data["positives"] if s >= threshold and ok)
        wrong_target = sum(1 for s, ok in data["positives"] if s >= threshold and not ok)
        fp = sum(1 for s in data["negatives"] if s >= threshold)
        flagged = tp + wrong_target + fp
        precision = tp / flagged if flagged else None
        recall = tp / len(data["positives"]) if data["positives"] else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
        return {"threshold": float(threshold), "precision": None if precision is None else round(precision, 4),
                "recall": None if recall is None else round(recall, 4), "f1": round(f1, 4),
                "false_alarm_rate_on_ordinary_issues": round(fp / max(len(data["negatives"]), 1), 4)}

    grid = [round(float(t), 2) for t in np.arange(0.50, 0.99, 0.02)]
    tune, test = scored["tune (older pairs)"], scored["test (newer pairs)"]
    tune_table = [at(t, tune) for t in grid]

    # A similarity score alone separates duplicates from ordinary issues poorly (precision stays
    # low at every threshold), so thresholds are chosen to cap false alarms on ordinary issues,
    # measured on the TUNE half only:
    #   "possible duplicate" -> lowest threshold flagging <= 0.5% of ordinary issues
    #   "check by hand"      -> lowest threshold flagging <= 2% of ordinary issues
    def lowest(limit: float) -> float:
        ok = [r["threshold"] for r in tune_table if r["false_alarm_rate_on_ordinary_issues"] <= limit]
        return min(ok) if ok else grid[-1]

    duplicate_t, review_t = lowest(0.005), lowest(0.02)

    def summary(data: dict) -> dict:
        pos = np.array([s for s, _ in data["positives"]])
        neg = np.array(data["negatives"])
        return {"top1_similarity_median_duplicates": round(float(np.median(pos)), 4),
                "top1_similarity_median_ordinary": round(float(np.median(neg)), 4)}

    return {
        "method": index.method,
        "rule": "duplicate: false alarms <= 0.5% on tune half; manual review: <= 2% on tune half",
        "tune_pairs": len(tune["positives"]), "tune_negatives": len(tune["negatives"]), "tune_window": tune["window"],
        "test_pairs": len(test["positives"]), "test_negatives": len(test["negatives"]), "test_window": test["window"],
        "chosen_duplicate_threshold": duplicate_t,
        "chosen_review_threshold": review_t,
        "tune_at_chosen": [at(duplicate_t, tune), at(review_t, tune)],
        "test_at_chosen": [at(duplicate_t, test), at(review_t, test)],
        "test_at_other_thresholds": [at(t, test) for t in (0.6, 0.7, 0.8, 0.85, 0.9)],
        "score_overlap": {"tune": summary(tune), "test": summary(test)},
        "tune_table": tune_table,
        "raw_scores": {k: {"positives": v["positives"], "negatives": v["negatives"]} for k, v in scored.items()},
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mode = "duplicate_pairs" if args.pairs else "proxy_title_to_body"
    if args.pairs:
        corpus, queries, skipped = pair_queries(args.issues, args.pairs)
    else:
        (corpus, queries), skipped = proxy_queries(args.issues, args.min_body_words), 0
    print(f"mode={mode} corpus={len(corpus)} queries={len(queries)} skipped={skipped}")

    shared_embeddings = None
    if any(m in ("embedding", "hybrid") for m in args.methods):
        shared_embeddings = Encoder(config["embedding_model"]).encode([r.text for r in corpus])
    report: dict = {"scope": args.scope if args.pairs else "proxy", "mode": mode, "corpus": len(corpus), "queries": len(queries), "skipped_pairs": skipped, "methods": {}}
    for method in args.methods:
        index = RetrievalIndex(
            corpus,
            method=method,
            embedding_model=config["embedding_model"],
            hybrid_weight_embedding=config["retrieval"]["hybrid_weight_embedding"],
            embeddings=shared_embeddings if method in ("embedding", "hybrid") else None,
        )
        started = time.perf_counter()
        ranks = []
        for text, target, context in queries:
            created_before, exclude = context if context else (None, None)
            repo = exclude[0] if (exclude and args.scope == "same_repository") else None
            results = index.search(text, k=10, created_before=created_before, exclude=exclude, repository=repo)
            ranks.append(rank_of(results, target))
        seconds = time.perf_counter() - started
        metrics = retrieval_metrics(ranks)
        metrics["ms_per_query"] = round(1000 * seconds / max(len(queries), 1), 2)
        run = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "milestone": "M3",
            "task": "retrieval",
            "mode": mode,
            "scope": args.scope if args.pairs else "proxy",
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
        report["methods"][index.method] = metrics
        if args.pairs and method == config["retrieval"]["default_method"]:
            report["decision_analysis"] = decision_analysis(
                index, queries, args.issues, args.negatives_per_half, args.scope == "same_repository")
            d = report["decision_analysis"]
            print(f"thresholds chosen on older pairs: duplicate={d['chosen_duplicate_threshold']} "
                  f"review={d['chosen_review_threshold']} -> newer pairs: {d['test_at_chosen']}")
    if args.pairs:
        if args.out.exists():  # keep scores of methods not re-run this time
            previous = json.loads(args.out.read_text(encoding="utf-8")).get("methods", {})
            report["methods"] = {**previous, **report["methods"]}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
