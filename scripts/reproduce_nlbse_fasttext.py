#!/usr/bin/env python3
"""Reproduce the NLBSE'24 per-repository fastText classification baseline.

This follows `4-Template-fastText.ipynb` from the public anchor repository:
- concatenate title and body;
- remove fastText's reserved `__label__` sequence;
- collapse consecutive whitespace with gensim;
- shuffle train and test with seed 42;
- train one supervised fastText model per repository for 100 epochs;
- evaluate bug/feature/question with per-class and weighted-average P/R/F1;
- average repository metrics for the reported cross-repository result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# Keep all Hugging Face cache writes inside the task workspace. This must be
# configured before importing `datasets`, which otherwise defaults to ~/.cache.
_DEFAULT_HF_HOME = (Path.cwd() / "work" / "hf_cache").resolve()
os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(_DEFAULT_HF_HOME / "datasets"))

import fasttext
import gensim
from datasets import Dataset, disable_progress_bar
from numpy import mean
from sklearn.metrics import classification_report, confusion_matrix


RANDOM_SEED = 42
EPOCHS = 100
LABELS = ["bug", "feature", "question"]
METRICS = ["precision", "recall", "f1-score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_dataset(example: dict) -> dict:
    text = (example["title"] or "") + " " + (example["body"] or "")
    text = text.replace("__label__", "")
    text = gensim.parsing.preprocessing.strip_multiple_whitespaces(text)
    example["text"] = text
    return example


def repo_eq(repository: str):
    return lambda example: example["repo"] == repository


def write_fasttext_train_file(dataset, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in dataset:
            handle.write(f"__label__{example['label']} {example['text']}\n")


def evaluate(references: dict[str, list[str]], predictions: dict[str, list[str]], repos: list[str]) -> dict:
    results: dict[str, dict] = {}
    for repository in repos:
        report = classification_report(
            references[repository],
            predictions[repository],
            labels=LABELS,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        report["average"] = report["weighted avg"]
        results[repository] = {
            label: {metric: report[label][metric] for metric in METRICS}
            for label in LABELS + ["average"]
        }
    results["overall"] = {
        label: {
            metric: float(mean([results[repository][label][metric] for repository in repos]))
            for metric in METRICS
        }
        for label in LABELS + ["average"]
    }
    return results


def metric_gaps(original: dict, reproduced: dict) -> dict:
    gaps: dict[str, dict] = {}
    for repository, label_values in reproduced.items():
        gaps[repository] = {}
        for label, metric_values in label_values.items():
            gaps[repository][label] = {
                metric: metric_values[metric] - original[repository][label][metric]
                for metric in METRICS
            }
    return gaps


def installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def memory_gib() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 2)
    except (AttributeError, ValueError, OSError):
        return None


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    disable_progress_bar()

    train_path = args.anchor_dir / "data" / "issues_train.csv"
    test_path = args.anchor_dir / "data" / "issues_test.csv"
    original_path = args.anchor_dir / "output" / "fasttext" / "results.json"
    for path in (train_path, test_path, original_path):
        if not path.exists():
            raise SystemExit(f"Missing anchor artifact: {path}")

    started_at = time.time()
    dataset = Dataset.from_csv(
        {"train": str(train_path), "test": str(test_path)},
        cache_dir=str(args.work_dir / "hf_cache"),
    )
    repositories = dataset["train"].unique("repo")
    dataset = dataset.shuffle(seed=RANDOM_SEED)
    dataset = dataset.map(process_dataset)

    references: dict[str, list[str]] = {}
    predictions: dict[str, list[str]] = {}
    durations: dict[str, float] = {}
    prediction_rows: list[dict[str, object]] = []
    model_arguments: dict[str, object] | None = None

    for repository in repositories:
        safe_name = repository.replace("/", "-")
        train_file = args.work_dir / f"issues-{safe_name}.train"
        model_file = args.work_dir / f"issues-{safe_name}.bin"
        train_set = dataset.filter(repo_eq(repository))["train"]
        write_fasttext_train_file(train_set, train_file)

        train_started = time.time()
        model = fasttext.train_supervised(
            input=str(train_file),
            epoch=EPOCHS,
            seed=RANDOM_SEED,
        )
        durations[repository] = round(time.time() - train_started, 4)
        model.save_model(str(model_file))
        if model_arguments is None:
            raw_args = model.f.getArgs()
            for name in ("lr", "dim", "ws", "epoch", "minCount", "wordNgrams", "bucket", "thread"):
                if hasattr(raw_args, name):
                    model_arguments = model_arguments or {}
                    model_arguments[name] = getattr(raw_args, name)

        test_set = dataset.filter(repo_eq(repository))["test"]
        repo_references = list(test_set["label"])
        raw_predictions = model.predict(test_set["text"], k=1)[0]
        repo_predictions = [prediction[0].replace("__label__", "") for prediction in raw_predictions]
        references[repository] = repo_references
        predictions[repository] = repo_predictions

        for index, (expected, predicted) in enumerate(zip(repo_references, repo_predictions)):
            prediction_rows.append(
                {
                    "repository": repository,
                    "test_index_within_repository": index,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": expected == predicted,
                    "title": test_set[index]["title"],
                }
            )

    reproduced = evaluate(references, predictions, repositories)
    original = json.loads(original_path.read_text(encoding="utf-8"))
    gaps = metric_gaps(original, reproduced)

    (args.output / "fasttext_reproduced_results.json").write_text(
        json.dumps(reproduced, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "fasttext_anchor_results.json").write_text(
        json.dumps(original, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "fasttext_metric_gaps.json").write_text(
        json.dumps(gaps, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    with (args.output / "confusion_matrices.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["repository", "actual", *[f"predicted_{label}" for label in LABELS]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for repository in repositories:
            matrix = confusion_matrix(references[repository], predictions[repository], labels=LABELS)
            for actual, counts in zip(LABELS, matrix):
                writer.writerow(
                    {
                        "repository": repository,
                        "actual": actual,
                        **{f"predicted_{label}": int(value) for label, value in zip(LABELS, counts)},
                    }
                )

    package_versions = {
        package: installed_version(package)
        for package in [
            "python",
            "pandas",
            "numpy",
            "scikit-learn",
            "datasets",
            "pyarrow",
            "fasttext",
            "gensim",
            "scipy",
            "huggingface-hub",
        ]
    }
    package_versions["python"] = platform.python_version()
    manifest = {
        "anchor_repository": "https://github.com/nlbse2024/issue-report-classification",
        "anchor_commit": subprocess.check_output(
            ["git", "-C", str(args.anchor_dir), "rev-parse", "HEAD"], text=True
        ).strip(),
        "baseline": "Per-repository supervised fastText from 4-Template-fastText.ipynb",
        "data": {
            "train_rows": len(dataset["train"]),
            "test_rows": len(dataset["test"]),
            "repositories": repositories,
            "labels": LABELS,
            "train_sha256": sha256_file(train_path),
            "test_sha256": sha256_file(test_path),
            "training_label_counts": dict(Counter(dataset["train"]["label"])),
            "test_label_counts": dict(Counter(dataset["test"]["label"])),
        },
        "preprocessing": [
            "concatenate title and body with one space",
            "remove reserved __label__ sequence",
            "collapse consecutive whitespace with gensim",
            "shuffle train and test independently with seed 42",
        ],
        "training": {
            "one_model_per_repository": True,
            "seed": RANDOM_SEED,
            "epochs": EPOCHS,
            "fasttext_arguments_observed": model_arguments,
            "duration_seconds_by_repository": durations,
            "total_duration_seconds": round(time.time() - started_at, 4),
        },
        "evaluation": {
            "class_metrics": METRICS,
            "labels": LABELS,
            "repository_average": "scikit-learn weighted average; equivalent to macro average because each repository test set is balanced",
            "cross_repository_average": "arithmetic mean of five repository metrics",
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "memory_gib": memory_gib(),
            "packages": package_versions,
        },
        "result_summary": {
            "anchor_overall_average_f1": original["overall"]["average"]["f1-score"],
            "reproduced_overall_average_f1": reproduced["overall"]["average"]["f1-score"],
            "absolute_gap": (
                reproduced["overall"]["average"]["f1-score"]
                - original["overall"]["average"]["f1-score"]
            ),
        },
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.output / "environment_freeze.txt").write_text(freeze, encoding="utf-8")

    print(json.dumps(manifest["result_summary"], indent=2))
    print(f"wrote reproduction artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

