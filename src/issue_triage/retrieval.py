"""Similar-issue search over historical issues: TF-IDF, BM25, sentence embeddings, or a hybrid.

Leakage rule (from M2): a query may only be compared with issues created *before* it, and never
with itself. `search(..., created_before=..., exclude=...)` enforces both.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .embeddings import Encoder

log = logging.getLogger(__name__)

Method = Literal["tfidf", "bm25", "embedding", "hybrid"]
METHODS: tuple[str, ...] = ("tfidf", "bm25", "embedding", "hybrid")


@dataclass
class IssueRecord:
    repository: str
    issue_number: int
    url: str
    title: str
    created_at: str  # ISO-8601 UTC "YYYY-MM-DDTHH:MM:SSZ" -> string order == time order
    text: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.repository, int(self.issue_number))


class _BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, texts: list[str]) -> "_BM25":
        self.vectorizer = CountVectorizer(lowercase=True)
        tf = self.vectorizer.fit_transform(texts).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        self.avgdl = float(lengths.mean()) if n_docs else 1.0
        self.weights = self._weigh(tf)
        return self

    def _weigh(self, tf: sparse.csr_matrix) -> sparse.csr_matrix:
        tf = tf.tocsr().astype(np.float32, copy=True)
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * lengths / self.avgdl)
        row_of_entry = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        values = tf.data
        tf.data = self.idf[tf.indices] * values * (self.k1 + 1) / (values + norm[row_of_entry])
        return tf

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return np.asarray((self.weights @ q.T).todense()).ravel()


class RetrievalIndex:
    def __init__(
        self,
        records: list[IssueRecord],
        method: Method = "hybrid",
        embedding_model: str | None = None,
        hybrid_weight_embedding: float = 0.6,
        embeddings: np.ndarray | None = None,
    ):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        self.records = list(records)
        self.method_requested = method
        self.method = method
        self.hybrid_weight = hybrid_weight_embedding
        self.encoder = Encoder(embedding_model) if embedding_model else None
        texts = [r.text for r in self.records]
        self._created = np.array([r.created_at or "" for r in self.records], dtype=object)

        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        self.bm25 = _BM25().fit(texts) if method == "bm25" else None

        self.embeddings: np.ndarray | None = None
        if method in ("embedding", "hybrid"):
            try:
                if self.encoder is None:
                    raise RuntimeError("no embedding model configured")
                self.embeddings = embeddings if embeddings is not None else self.encoder.encode(texts)
            except Exception as exc:  # model download/load failure must not take the service down
                log.warning("embedding index unavailable (%s); falling back to tfidf", exc)
                self.method = "tfidf"

    # ------------------------------------------------------------------ scoring
    def _scores(self, text: str) -> np.ndarray:
        if self.method == "bm25":
            return self.bm25.scores(text)
        lexical = (self.tfidf_matrix @ self.tfidf.transform([text]).T).toarray().ravel()
        if self.method == "tfidf":
            return lexical
        semantic = self.embeddings @ self.encoder.encode([text])[0]
        if self.method == "embedding":
            return semantic
        return self.hybrid_weight * semantic + (1 - self.hybrid_weight) * lexical

    def search(
        self,
        text: str,
        k: int = 5,
        created_before: str | None = None,
        exclude: tuple[str, int] | None = None,
        repository: str | None = None,
    ) -> list[dict]:
        if not self.records or not text.strip():
            return []
        scores = self._scores(text).astype(np.float64)
        allowed = np.ones(len(self.records), dtype=bool)
        if created_before:
            allowed &= self._created < created_before
        if exclude is not None:
            allowed &= np.array([r.key != exclude for r in self.records])
        if repository:
            allowed &= np.array([r.repository == repository for r in self.records])
        scores[~allowed] = -np.inf
        k = min(k, int(allowed.sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            {
                "repository": self.records[i].repository,
                "issue_number": int(self.records[i].issue_number),
                "url": self.records[i].url,
                "title": self.records[i].title,
                "created_at": self.records[i].created_at,
                "similarity": round(float(scores[i]), 4),
            }
            for i in top
        ]

    # ------------------------------------------------------------------ updates
    def add(self, record: IssueRecord) -> None:
        """Add one new issue (e.g. from the webhook) so later issues can match it.

        The lexical vocabulary is re-fitted: with a frozen vocabulary, words that first appear
        in the new issue would be invisible and it could never be found. Re-fitting a few
        thousand short documents takes well under a second; embeddings need no re-fit.
        """
        if any(r.key == record.key for r in self.records):
            return
        self.records.append(record)
        self._created = np.append(self._created, record.created_at or "")
        texts = [r.text for r in self.records]
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        if self.bm25 is not None:
            self.bm25.fit(texts)
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, self.encoder.encode([record.text])])

    def describe(self) -> dict:
        return {
            "method_requested": self.method_requested,
            "method_used": self.method,
            "documents": len(self.records),
            "repositories": sorted({r.repository for r in self.records}),
        }

    def records_as_dicts(self) -> list[dict]:
        return [asdict(r) for r in self.records]
