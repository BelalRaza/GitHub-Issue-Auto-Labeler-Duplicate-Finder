"""The single prediction core shared by the CLI, the FastAPI endpoints and the webhook.

    predictor = TriagePredictor.load(Path("models"))
    result = predictor.predict(title="App crashes on upload", body="...")
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from .preparation import clean_issue_text
from .retrieval import IssueRecord, RetrievalIndex

MODEL_FILES = {
    "classifier": "classifier.joblib",
    "records": "index_records.json",
    "embeddings": "index_embeddings.npy",
    "card": "model_card.json",
}


class TriagePredictor:
    def __init__(self, classifier, index: RetrievalIndex, config: dict, card: dict | None = None):
        self.classifier = classifier
        self.index = index
        self.config = config
        self.card = card or {}

    # ------------------------------------------------------------------ persistence
    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, model_dir / MODEL_FILES["classifier"])
        (model_dir / MODEL_FILES["records"]).write_text(
            json.dumps(self.index.records_as_dicts(), ensure_ascii=False), encoding="utf-8"
        )
        if self.index.embeddings is not None:
            np.save(model_dir / MODEL_FILES["embeddings"], self.index.embeddings)
        card = {**self.card, "config": self.config, "index": self.index.describe()}
        (model_dir / MODEL_FILES["card"]).write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(
        cls, model_dir: Path, method: str | None = None, config_path: Path | None = None
    ) -> "TriagePredictor":
        """`config_path` lets thresholds (tuned in M4) change without retraining the models."""
        card = json.loads((model_dir / MODEL_FILES["card"]).read_text(encoding="utf-8"))
        config = card["config"]
        if config_path and config_path.exists():
            live = json.loads(config_path.read_text(encoding="utf-8"))
            config = {
                **config,
                "classification": {**config["classification"], **live.get("classification", {})},
                "retrieval": {**config["retrieval"], **live.get("retrieval", {})},
            }
        classifier = joblib.load(model_dir / MODEL_FILES["classifier"])
        records = [
            IssueRecord(**row)
            for row in json.loads((model_dir / MODEL_FILES["records"]).read_text(encoding="utf-8"))
        ]
        method = method or config["retrieval"]["default_method"]
        embeddings_path = model_dir / MODEL_FILES["embeddings"]
        embeddings = None
        if method in ("embedding", "hybrid") and embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            if len(embeddings) != len(records):
                embeddings = None  # stale cache -> re-encode
        index = RetrievalIndex(
            records,
            method=method,
            embedding_model=config["embedding_model"],
            hybrid_weight_embedding=config["retrieval"]["hybrid_weight_embedding"],
            embeddings=embeddings,
        )
        return cls(classifier, index, config, card)

    # ------------------------------------------------------------------ inference
    def predict(
        self,
        title: str,
        body: str = "",
        repository: str | None = None,
        created_at: str | None = None,
        issue_key: tuple[str, int] | None = None,
        top_k: int | None = None,
    ) -> dict:
        started = time.perf_counter()
        title = (title or "").strip()
        body = body or ""
        if not title and not body.strip():
            raise ValueError("an issue needs a title or a body")
        text = clean_issue_text(f"{title}\n\n{body}")

        # Task 1: classification
        cls_cfg = self.config["classification"]
        probs = self.classifier.predict_proba([text])[0]
        order = np.argsort(-probs)
        best = int(order[0])
        confidence = float(probs[best])
        too_short = len(text.split()) < cls_cfg.get("min_words", 0)
        confident = confidence >= cls_cfg["min_confidence"] and not too_short
        if too_short:
            reason = "issue text is too short to label safely"
        elif not confident:
            reason = f"confidence {confidence:.2f} is below {cls_cfg['min_confidence']}"
        else:
            reason = None
        classification = {
            "label": str(self.classifier.classes_[best]) if confident else None,
            "top_guess": str(self.classifier.classes_[best]),
            "confidence": round(confidence, 4),
            "decision": "suggest_label" if confident else "needs_manual_review",
            "review_reason": reason,
            "probabilities": {
                str(self.classifier.classes_[i]): round(float(probs[i]), 4) for i in order
            },
        }

        # Task 2: similar-issue retrieval, past issues only, never the issue itself
        ret_cfg = self.config["retrieval"]
        candidates = self.index.search(
            text,
            k=top_k or ret_cfg["top_k"],
            created_before=created_at,
            exclude=issue_key,
        )
        top_similarity = candidates[0]["similarity"] if candidates else 0.0
        if top_similarity >= ret_cfg["duplicate_threshold"]:
            dup_decision = "possible_duplicate"
        elif top_similarity >= ret_cfg["review_threshold"]:
            dup_decision = "needs_manual_review"
        else:
            dup_decision = "no_duplicate_found"
        for candidate in candidates:
            candidate["above_review_threshold"] = candidate["similarity"] >= ret_cfg["review_threshold"]

        return {
            "input": {
                "title": title,
                "repository": repository,
                "body_chars": len(body),
                "created_at": created_at,
            },
            "preprocessing": {
                "cleaned_text_preview": text[:600],
                "word_count": len(text.split()),
            },
            "classification": classification,
            "duplicates": {
                "decision": dup_decision,
                "top_similarity": round(float(top_similarity), 4),
                "thresholds": {
                    "possible_duplicate": ret_cfg["duplicate_threshold"],
                    "manual_review": ret_cfg["review_threshold"],
                },
                "method": self.index.method,
                "searched_issues": len(self.index.records),
                "candidates": candidates,
            },
            "model": {
                "classifier": self.card.get("classifier", {}).get("name"),
                "version": self.config.get("version"),
                "trained_at": self.card.get("trained_at"),
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def warmup(self) -> None:
        """Load lazy models now (at startup) so the first real request isn't slow."""
        self.predict("warm up the models", "loading the embedding model before traffic arrives")

    def remember(self, record: IssueRecord) -> None:
        """Add a newly seen issue so later issues can be matched against it."""
        self.index.add(record)
