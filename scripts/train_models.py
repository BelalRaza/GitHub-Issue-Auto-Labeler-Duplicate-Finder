#!/usr/bin/env python3
"""Train every classifier candidate, log each run, select one, save the model bundle.

Selection (M4 improvement): the validation split holds only 8 Feature/Docs issues, so a single
validation score is too noisy to choose a model. `--selection cv` (default) uses 5-fold
stratified cross-validation over train+validation instead, then refits the winner on
train+validation. `--selection validation` reproduces the original M3 behaviour.

The test split is never used here. It is only reported, in scripts/evaluate_models.py.
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

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from issue_triage.classification import (  # noqa: E402
    AveragingEnsemble,
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
    parser.add_argument("--selection", choices=["cv", "validation"], default="cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--choose", choices=["tfidf_repo_balanced"],
        help="skip CV selection and train this model, chosen by the unseen-repository comparison "
             "(scripts/compare_label_strategies.py -> outputs/m4_full/label_strategies.json)",
    )
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

    emb = config["embedding_model"]
    factories = [("majority", {}, lambda: MajorityClassifier())]
    factories += [("tfidf_logreg", {"C": 4.0, "class_weight": "repo_balanced"},
                   lambda: TfidfLogReg(C=4.0, class_weight="repo_balanced"))]
    factories += [("tfidf_logreg", {"C": c}, lambda c=c: TfidfLogReg(C=c)) for c in (1.0, 4.0, 16.0)]
    if not args.skip_embeddings:
        factories += [
            ("embedding_logreg", {"C": c}, lambda c=c: EmbeddingLogReg(emb, C=c, cache=True))
            for c in (1.0, 4.0, 16.0)
        ]
        factories.append(
            ("ensemble", {"C": 4.0},
             lambda: AveragingEnsemble([TfidfLogReg(C=4.0), EmbeddingLogReg(emb, C=4.0, cache=True)]))
        )

    pool_texts = train["texts"] + val["texts"]
    pool_labels = train["labels"] + val["labels"]
    train_groups = [r["repository"] for r in train["rows"]]
    pool_groups = train_groups + [r["repository"] for r in val["rows"]]
    if args.choose:  # selected by the unseen-repository comparison, not by CV
        factories = [f for f in factories if f[1].get("class_weight") == "repo_balanced"]
    folds = list(
        StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=13).split(pool_texts, pool_labels)
    )

    args.log.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for name, _, factory in factories:
        started = time.perf_counter()
        model = factory().fit(train["texts"], train["labels"], groups=train_groups)
        fit_seconds = time.perf_counter() - started
        report = classification_report_dict(val["labels"], predict_labels(model, val["texts"]), classes)
        cv_scores = []
        for train_idx, test_idx in ([] if args.choose else folds):
            fold_model = factory().fit([pool_texts[i] for i in train_idx], [pool_labels[i] for i in train_idx],
                                       groups=[pool_groups[i] for i in train_idx])
            fold_pred = predict_labels(fold_model, [pool_texts[i] for i in test_idx])
            cv_scores.append(
                classification_report_dict([pool_labels[i] for i in test_idx], fold_pred, classes)["macro_f1"]
            )
        run = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "milestone": "M4" if args.selection == "cv" else "M3",
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
            "cv": None if not cv_scores else {
                "folds": args.folds,
                "pool": "train+validation",
                "macro_f1_mean": round(float(np.mean(cv_scores)), 4),
                "macro_f1_std": round(float(np.std(cv_scores)), 4),
            },
        }
        runs.append((run, factory))
        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(run) + "\n")
        cv_text = (f"cv macroF1={run['cv']['macro_f1_mean']:.3f}±{run['cv']['macro_f1_std']:.3f}"
                   if run["cv"] else "cv skipped (--choose)")
        print(f"{name:18s} {json.dumps(_):40s} val macroF1={report['macro_f1']:.3f}  {cv_text}")

    # Selection; ties go to the simpler (earlier) model because max() keeps the first maximum.
    if args.choose:
        score = lambda r: 0.0  # noqa: E731 - only the chosen model is in `runs`
    elif args.selection == "cv":
        score = lambda r: r["cv"]["macro_f1_mean"]  # noqa: E731
    else:
        score = lambda r: r["metrics"]["macro_f1"]  # noqa: E731
    best_run, best_factory = max(runs, key=lambda item: score(item[0]))
    best_model = best_factory()
    if args.selection == "cv":
        best_model.fit(pool_texts, pool_labels, groups=pool_groups)  # final model sees all non-test data
    else:
        best_model.fit(train["texts"], train["labels"], groups=train_groups)
    if hasattr(best_model, "cache"):
        best_model.cache = False
    for member in getattr(best_model, "members", []):
        if hasattr(member, "cache"):
            member.cache = False
    print(f"selected ({args.selection}): {best_run['model']} {best_run['params']}")

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
        "selection": (
            "best unseen-repository macro-F1 with repo adaptation (leave-one-repository-out on "
            "train+validation: outputs/m4_full/label_strategies.json); refit on train+validation"
            if args.choose else
            f"highest mean {args.folds}-fold CV macro-F1 on train+validation; refit on train+validation"
            if args.selection == "cv" else "highest validation macro-F1; fit on train"
        ),
        "validation_metrics": best_run["metrics"],
        "cv_metrics": best_run["cv"],
        "fitted_on": "train+validation" if args.selection == "cv" else "train",
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
