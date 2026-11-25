"""Classification and retrieval metrics used by M3/M4 reports."""

from __future__ import annotations

from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_report_dict(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(classes), zero_division=0
    )
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(
                f1_score(y_true, y_pred, labels=list(classes), average="weighted", zero_division=0)
            ),
            4,
        ),
        "per_class": {
            name: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for name, p, r, f, s in zip(classes, precision, recall, f1, support)
        },
        "confusion_matrix": {
            "labels": list(classes),
            "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=list(classes)).tolist(),
        },
    }


def retrieval_metrics(ranks: Sequence[int | None], ks: Sequence[int] = (1, 5, 10)) -> dict:
    """`ranks` holds the 1-based rank of the correct item per query, or None if not retrieved."""
    n = len(ranks)
    if n == 0:
        return {"queries": 0}
    result: dict = {"queries": n}
    for k in ks:
        result[f"recall@{k}"] = round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)
    result["mrr@10"] = round(sum(1.0 / r for r in ranks if r is not None and r <= 10) / n, 4)
    return result
