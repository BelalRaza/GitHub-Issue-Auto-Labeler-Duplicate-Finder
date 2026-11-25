#!/usr/bin/env python3
"""M4: evaluate on the held-out TEST split once, plus robustness, thresholds and held-out repos.

Writes machine-readable results to outputs/m4/. The human-readable write-up is
outputs/M4_EVALUATION.md.

Rule: every *choice* (model, C, confidence threshold) is made on validation data. The test split
only *reports* what those choices achieve.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from issue_triage.classification import (  # noqa: E402
    AveragingEnsemble,
    EmbeddingLogReg,
    MajorityClassifier,
    TfidfLogReg,
    predict_labels,
)
from issue_triage.datasets import load_classification_splits  # noqa: E402
from issue_triage.metrics import classification_report_dict  # noqa: E402
from issue_triage.predictor import TriagePredictor  # noqa: E402

csv.field_size_limit(10_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/classification_model.csv")
    parser.add_argument("--issues", type=Path, default=ROOT / "data/processed/issues_clean.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config/model_config.json")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/m4")
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    return parser.parse_args()


# --------------------------------------------------------------------------- text perturbations
def add_typos(text: str, rng: random.Random, rate: float = 0.15) -> str:
    words = text.split()
    for i, word in enumerate(words):
        if len(word) > 3 and rng.random() < rate:
            j = rng.randrange(len(word) - 1)
            words[i] = word[:j] + word[j + 1] + word[j] + word[j + 2 :]  # swap two letters
    return " ".join(words)


PERTURBATIONS = {
    "original": lambda row, rng: row["text_clean"],
    "title_only (empty body)": lambda row, rng: row["title"],
    "first_20_words": lambda row, rng: " ".join(row["text_clean"].split()[:20]),
    "typos_15pct_words": lambda row, rng: add_typos(row["text_clean"], rng),
    "lowercase_no_punctuation": lambda row, rng: "".join(
        ch for ch in row["text_clean"].lower() if ch.isalnum() or ch.isspace()
    ),
    "very_long (x5 + log noise)": lambda row, rng: " ".join([row["text_clean"]] * 5)
    + " INFO started worker pid=1234" * 200,
}


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    classes = config["core_classes"]
    splits = load_classification_splits(args.data, classes)
    train, val, test = splits["train"], splits["validation"], splits["test"]
    card = json.loads((args.models / "model_card.json").read_text(encoding="utf-8"))
    emb = config["embedding_model"]

    def same_family_as_chosen():
        """A fresh, unfitted model of the type the training script selected."""
        name, params = card["classifier"]["name"], card["classifier"]["params"]
        if name == "tfidf_logreg":
            return TfidfLogReg(C=params["C"], class_weight=params.get("class_weight", "balanced"))
        if name == "embedding_logreg":
            return EmbeddingLogReg(emb, C=params["C"], class_weight=params.get("class_weight", "balanced"), cache=True)
        if name == "ensemble":
            return AveragingEnsemble([TfidfLogReg(C=4.0), EmbeddingLogReg(emb, C=4.0, cache=True)])
        return MajorityClassifier()
    results: dict = {"split_sizes": {k: len(v["texts"]) for k, v in splits.items()}}
    print(results["split_sizes"])

    # 1. Model comparison on TEST. All candidates are fitted on the same data as the saved model
    #    (train+validation under CV selection), so the comparison is like-for-like.
    fit_texts, fit_labels = train["texts"], train["labels"]
    fit_groups = [r["repository"] for r in train["rows"]]
    if card.get("fitted_on") == "train+validation":
        fit_texts, fit_labels = train["texts"] + val["texts"], train["labels"] + val["labels"]
        fit_groups = fit_groups + [r["repository"] for r in val["rows"]]
    models = {
        "Majority class": MajorityClassifier(),
        "TF-IDF + LogReg (C=4)": TfidfLogReg(C=4.0),
        "TF-IDF + LogReg, repo-balanced (C=4)": TfidfLogReg(C=4.0, class_weight="repo_balanced"),
        "MiniLM embeddings + LogReg (C=4)": EmbeddingLogReg(emb, C=4.0, cache=True),
        "Ensemble: TF-IDF + MiniLM": AveragingEnsemble(
            [TfidfLogReg(C=4.0), EmbeddingLogReg(emb, C=4.0, cache=True)]
        ),
    }
    results["fitted_on"] = card.get("fitted_on", "train")
    results["chosen_model"] = card["classifier"]
    results["test_comparison"] = {}
    for name, model in models.items():
        model.fit(fit_texts, fit_labels, groups=fit_groups)
        report = classification_report_dict(test["labels"], predict_labels(model, test["texts"]), classes)
        results["test_comparison"][name] = report
        print(f"TEST {name:28s} acc={report['accuracy']:.3f} macroF1={report['macro_f1']:.3f}")

    predictor = TriagePredictor.load(args.models, config_path=args.config)
    chosen = predictor.classifier

    # 2. Errors of the chosen model on TEST, for manual error analysis -------------------------
    probs = chosen.predict_proba(test["texts"])
    preds = [str(chosen.classes_[i]) for i in probs.argmax(axis=1)]
    with (args.out / "test_errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repository", "issue_number", "url", "title", "true", "predicted", "confidence", "text_start"])
        for row, true, pred, p in zip(test["rows"], test["labels"], preds, probs):
            if true != pred:
                writer.writerow(
                    [row["repository"], row["issue_number"], row["url"], row["title"], true, pred,
                     round(float(p.max()), 3), row["text_clean"][:300]]
                )

    # 3. Confidence threshold: choose on VALIDATION, report on TEST ---------------------------
    #    The threshold must be chosen on data the model was NOT fitted on, so we fit a
    #    same-family model on train only and read its validation confidences.
    threshold_model = same_family_as_chosen().fit(
        train["texts"], train["labels"], groups=[r["repository"] for r in train["rows"]])
    def coverage_table(model, texts, labels):
        p = model.predict_proba(texts)
        conf, pred = p.max(axis=1), [str(model.classes_[i]) for i in p.argmax(axis=1)]
        rows = []
        for t in np.round(np.arange(0.35, 0.96, 0.05), 2):
            keep = conf >= t
            n = int(keep.sum())
            acc = float(np.mean([pred[i] == labels[i] for i in np.where(keep)[0]])) if n else None
            rows.append({"threshold": float(t), "coverage": round(n / len(labels), 4),
                         "accuracy_on_auto_labelled": None if acc is None else round(acc, 4)})
        return rows

    val_table = coverage_table(threshold_model, val["texts"], val["labels"])
    ok = [r for r in val_table if r["accuracy_on_auto_labelled"] is not None
          and r["accuracy_on_auto_labelled"] >= args.target_accuracy]
    picked = min(ok, key=lambda r: r["threshold"])["threshold"] if ok else 0.9
    test_table = coverage_table(chosen, test["texts"], test["labels"])
    results["confidence_threshold"] = {
        "rule": f"lowest threshold whose VALIDATION accuracy on auto-labelled issues >= {args.target_accuracy}",
        "picked_on_validation": picked,
        "validation": val_table,
        "test_at_picked": next(r for r in test_table if r["threshold"] == picked),
        "test": test_table,
    }
    print("picked confidence threshold", picked, results["confidence_threshold"]["test_at_picked"])

    # 4. Robustness: perturb TEST inputs -----------------------------------------------------------
    rng = random.Random(13)
    base_preds = preds
    results["robustness"] = {}
    for name, fn in PERTURBATIONS.items():
        texts = [fn(row, rng) for row in test["rows"]]
        started = time.perf_counter()
        p_preds = predict_labels(chosen, texts)
        ms = 1000 * (time.perf_counter() - started) / len(texts)
        report = classification_report_dict(test["labels"], p_preds, classes)
        results["robustness"][name] = {
            "macro_f1": report["macro_f1"],
            "accuracy": report["accuracy"],
            "same_prediction_as_original": round(np.mean([a == b for a, b in zip(p_preds, base_preds)]), 4),
            "ms_per_issue": round(ms, 2),
        }
    edge = {}
    for label, (title, body) in {
        "emoji_only": ("🔥🔥🔥", "😭"),
        "non_english": ("La aplicación se bloquea al subir imágenes grandes", "Error de memoria"),
        "50k_characters": ("Huge log", "x " * 25_000),
        "only_code_block": ("", "```\nTraceback (most recent call last):\nKeyError: 'id'\n```"),
    }.items():
        out = predictor.predict(title, body)
        edge[label] = {
            "classification_decision": out["classification"]["decision"],
            "top_guess": out["classification"]["top_guess"],
            "confidence": out["classification"]["confidence"],
            "duplicate_decision": out["duplicates"]["decision"],
            "latency_ms": out["latency_ms"],
        }
    results["edge_cases"] = edge

    # 5. Repository held-out: train on 9 repos, test on the 10th ---------------------------------
    all_rows = train["rows"] + val["rows"] + test["rows"]
    held = {}
    for repo in sorted({r["repository"] for r in all_rows}):
        tr = [r for r in all_rows if r["repository"] != repo]
        te = [r for r in all_rows if r["repository"] == repo]
        present = sorted({r["classification_target"] for r in te})
        if len(te) < 15 or len(present) < 2:
            held[repo] = {"skipped": f"{len(te)} rows, classes present {present}"}
            continue
        model = same_family_as_chosen().fit(
            [r["text_clean"] for r in tr], [r["classification_target"] for r in tr],
            groups=[r["repository"] for r in tr],
        )
        y = [r["classification_target"] for r in te]
        report = classification_report_dict(y, predict_labels(model, [r["text_clean"] for r in te]), present)
        held[repo] = {"rows": len(te), "classes": present, "macro_f1": report["macro_f1"], "accuracy": report["accuracy"]}
    scored = [v["macro_f1"] for v in held.values() if "macro_f1" in v]
    results["repository_held_out"] = {"per_repository": held, "mean_macro_f1": round(float(np.mean(scored)), 4)}
    print("held-out repo mean macro-F1", results["repository_held_out"]["mean_macro_f1"])

    # 6. Duplicate threshold, interim: how often would ordinary new issues be flagged? ---------
    with args.issues.open(encoding="utf-8", newline="") as handle:
        issues = sorted(csv.DictReader(handle), key=lambda r: r["created_at"])
    recent = issues[int(len(issues) * 0.7):]  # the newest 30%, so each has a real past to search
    top1, top1_dup = [], []
    for row in recent:
        hits = predictor.index.search(
            row["text_clean"], k=1, created_before=row["created_at"],
            exclude=(row["repository"], int(row["issue_number"])),
        )
        if not hits:
            continue
        (top1_dup if row["state_reason"] == "duplicate" else top1).append(hits[0]["similarity"])
    flag_rates = {
        str(t): round(float(np.mean(np.array(top1) >= t)), 4) for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)
    }
    results["duplicate_threshold_interim"] = {
        "method": predictor.index.method,
        "queries_not_marked_duplicate": len(top1),
        "queries_closed_as_duplicate": len(top1_dup),
        "top1_similarity_percentiles_non_duplicates": {
            p: round(float(np.percentile(top1, p)), 4) for p in (50, 90, 95, 99)
        },
        "top1_similarity_closed_as_duplicate": sorted(round(x, 4) for x in top1_dup),
        "share_of_non_duplicates_flagged_at_threshold": flag_rates,
        "caveat": "No canonical pairs yet (needs a token). Flag rate on non-duplicates approximates the false-alarm rate only.",
    }
    print("flag rates", flag_rates, "dup sims", results["duplicate_threshold_interim"]["top1_similarity_closed_as_duplicate"])

    (args.out / "m4_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'm4_results.json'} and {args.out / 'test_errors.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
