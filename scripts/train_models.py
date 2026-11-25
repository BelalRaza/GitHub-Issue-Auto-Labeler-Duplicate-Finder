#!/usr/bin/env python3
"""M3: train every classifier candidate, log each run, pick the best on VALIDATION, save it.

The test split is not touched here. It is used once, in M4 (scripts/evaluate_models.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from issue_triage.classification import (  # noqa: E402
    EmbeddingLogReg,
    MajorityClassifier,
    TfidfLogReg,
    predict_labels,
)
from issue_triage.datasets import file_sha256, load_classification_splits, load_index_records  # noqa: E402
from issue_triage.metrics import classification_report_dict  # noqa: E402
from issue_triage.predictor import TriagePredictor  # noqa: E402
from issue_triage.retrieval import RetrievalIndex  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/classification_model.csv")
    parser.add_argument("--issues", type=Path, default=ROOT / "data/processed/issues_clean.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config/model_config.json")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--log", type=Path, default=ROOT / "experiments/experiment_log.jsonl")
    parser.add_argument("--skip-embeddings", action="store_true", help="lexical models only (no torch)")
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    classes = config["core_classes"]
    splits = load_classification_splits(args.data, classes)
    train, val = splits["train"], splits["validation"]
    data_hash = file_sha256(args.data)[:16]
    print(f"train={len(train['texts'])} validation={len(val['texts'])} classes={classes}")

    candidates = [MajorityClassifier()]
    candidates += [TfidfLogReg(C=c) for c in (1.0, 4.0, 16.0)]
    if not args.skip_embeddings:
        candidates += [EmbeddingLogReg(config["embedding_model"], C=c) for c in (1.0, 4.0, 16.0)]

    args.log.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for model in candidates:
        started = time.perf_counter()
        model.fit(train["texts"], train["labels"])
        fit_seconds = time.perf_counter() - started
        report = classification_report_dict(val["labels"], predict_labels(model, val["texts"]), classes)
        run = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "milestone": "M3",
            "task": "classification",
            "model": model.name,
            "params": model.params(),
            "data_file": str(args.data.relative_to(ROOT)),
            "data_sha256_16": data_hash,
            "git_commit": git_commit(),
            "train_rows": len(train["texts"]),
            "eval_split": "validation",
            "fit_seconds": round(fit_seconds, 2),
            "metrics": report,
        }
        runs.append((run, model))
        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(run) + "\n")
        print(
            f"{model.name:18s} {json.dumps(model.params()):70.70s} "
            f"macroF1={report['macro_f1']:.3f} acc={report['accuracy']:.3f} ({fit_seconds:.1f}s)"
        )

    # Model selection: validation macro-F1, ties broken by the simpler (earlier) model.
    best_run, best_model = max(runs, key=lambda item: item[0]["metrics"]["macro_f1"])
    print(f"selected: {best_run['model']} {best_run['params']}")

    method = "tfidf" if args.skip_embeddings else config["retrieval"]["default_method"]
    index = RetrievalIndex(
        load_index_records(args.issues),
        method=method,
        embedding_model=config["embedding_model"],
        hybrid_weight_embedding=config["retrieval"]["hybrid_weight_embedding"],
    )
    card = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classifier": {"name": best_run["model"], "params": best_run["params"]},
        "selection": "highest validation macro-F1 among logged M3 candidates",
        "validation_metrics": best_run["metrics"],
        "data_sha256_16": data_hash,
        "git_commit": git_commit(),
    }
    if args.skip_embeddings:
        config = {**config, "retrieval": {**config["retrieval"], "default_method": "tfidf"}}
    TriagePredictor(best_model, index, config, card).save(args.models)
    print(f"saved model bundle to {args.models} (index: {index.describe()['method_used']}, "
          f"{len(index.records)} issues)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
