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
                "version": live.get("version", config.get("version")),
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
        index: RetrievalIndex | None = None,
    ) -> dict:
        """`index` overrides the search set (e.g. one repository's issues fetched live). Without
        it, a known `repository` restricts the search to that repository; with neither, all
        collected issues are searched (the "type an issue" demo)."""
        started = time.perf_counter()
        title = (title or "").strip()
        body = body or ""
        if not title and not body.strip():
            raise ValueError("an issue needs a title or a body")
        text = clean_issue_text(f"{title}\n\n{body}")

        # Where to look: the issue's own repository whenever we know it (see retrieval scope).
        ret_cfg = self.config["retrieval"]
        same_repo = ret_cfg.get("scope", "same_repository") == "same_repository"
        if index is not None:
            search_index, repo_filter = index, None
            scope = f"same repository ({repository})" if repository else "given issue set"
        elif repository and same_repo:
            search_index, repo_filter = self.index, repository
            scope = f"same repository ({repository})"
        else:
            search_index, repo_filter = self.index, None
            scope = "all collected repositories (no repository given)"
        in_one_repo = index is not None or repo_filter is not None

        # Task 1: classification = general model, adjusted by this repository's own labels
        cls_cfg = self.config["classification"]
        classes = [str(c) for c in self.classifier.classes_]
        general = self.classifier.predict_proba([text])[0]
        probs, evidence, weight = general, [], 0.0
        adapt = cls_cfg.get("repo_adaptation", {})
        if adapt.get("enabled") and in_one_repo:
            probs, evidence, weight = self._adapt_to_repository(
                text, general, classes, search_index, repo_filter, created_at, issue_key, adapt
            )
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
            "label": classes[best] if confident else None,
            "top_guess": classes[best],
            "confidence": round(confidence, 4),
            "decision": "suggest_label" if confident else "needs_manual_review",
            "review_reason": reason,
            "probabilities": {classes[i]: round(float(probs[i]), 4) for i in order},
            "general_model_probabilities": {classes[i]: round(float(general[i]), 4) for i in np.argsort(-general)},
            "repository_evidence": {"weight": round(weight, 3), "similar_labelled_issues": evidence},
        }

        # Task 2: similar-issue retrieval, past issues only, never the issue itself
        candidates = search_index.search(
            text,
            k=top_k or ret_cfg["top_k"],
            created_before=created_at,
            exclude=issue_key,
            repository=repo_filter,
        )
        searched = (
            sum(1 for r in search_index.records if r.repository.lower() == repo_filter.lower())
            if repo_filter else len(search_index.records)
        )
        # Three honest outcomes. M4 showed a similarity score alone cannot prove a duplicate, so
        # the system never says "not a duplicate": below the thresholds it says "no strong match"
        # and still shows the closest issues.
        top_similarity = candidates[0]["similarity"] if candidates else 0.0
        short_floor = ret_cfg.get("short_issue_review_floor", ret_cfg["review_threshold"])
        if not candidates:
            dup_decision, dup_reason = "no_strong_match", "no earlier issue in this repository to compare with"
        elif top_similarity >= ret_cfg["duplicate_threshold"]:
            dup_decision, dup_reason = "possible_duplicate", (
                f"very similar to an earlier issue ({top_similarity:.2f} >= {ret_cfg['duplicate_threshold']})"
            )
        elif top_similarity >= ret_cfg["review_threshold"]:
            dup_decision, dup_reason = "check_similar", (
                f"similar to an earlier issue ({top_similarity:.2f} >= {ret_cfg['review_threshold']}); a human should check"
            )
        elif too_short and top_similarity >= short_floor:
            # Short texts score low even when they repeat an earlier issue (live demo #4/#5),
            # so for them a moderate match is worth a human look.
            dup_decision, dup_reason = "check_similar", (
                f"short issue: scores run low, so the closest match ({top_similarity:.2f}) is worth checking"
            )
        else:
            dup_decision, dup_reason = "no_strong_match", (
                f"closest earlier issue scores {top_similarity:.2f}, below {ret_cfg['review_threshold']}"
            )
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
                "reason": dup_reason,
                "top_similarity": round(float(top_similarity), 4),
                "thresholds": {
                    "possible_duplicate": ret_cfg["duplicate_threshold"],
                    "manual_review": ret_cfg["review_threshold"],
                    "short_issue_review": short_floor,
                },
                "method": search_index.method,
                "scope": scope,
                "searched_issues": searched,
                "candidates": candidates,
            },
            "model": {
                "classifier": self.card.get("classifier", {}).get("name"),
                "version": self.config.get("version"),
                "trained_at": self.card.get("trained_at"),
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    @staticmethod
    def _adapt_to_repository(text, general, classes, search_index, repo_filter, created_at, issue_key, adapt):
        """Blend the general model with labels the repository's maintainers already gave to
        similar, earlier issues (a similarity-weighted vote of the nearest labelled neighbours).

        The general model knows what bugs and feature requests sound like in English; the
        repository's own labels capture its conventions. The vote's weight grows with the
        number of similar labelled issues found, up to `max_weight`.
        """
        neighbours = search_index.search(
            text, k=adapt.get("k", 10), created_before=created_at, exclude=issue_key, repository=repo_filter
        )
        votes = np.zeros(len(classes))
        evidence = []
        for n in neighbours:
            if n.get("label") in classes and n["similarity"] >= adapt.get("min_similarity", 0.45):
                votes[classes.index(n["label"])] += n["similarity"]
                evidence.append({"issue_number": n["issue_number"], "url": n["url"], "label": n["label"],
                                 "similarity": n["similarity"]})
        if not evidence:
            return general, [], 0.0
        weight = adapt.get("max_weight", 0.6) * min(1.0, len(evidence) / adapt.get("full_weight_at", 3))
        return (1 - weight) * general + weight * votes / votes.sum(), evidence, weight

    def warmup(self) -> None:
        """Load lazy models now (at startup) so the first real request isn't slow."""
        self.predict("warm up the models", "loading the embedding model before traffic arrives")

    def remember(self, record: IssueRecord) -> None:
        """Add a newly seen issue so later issues can be matched against it."""
        self.index.add(record)
