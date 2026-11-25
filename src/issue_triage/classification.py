"""Issue-type classifiers: majority baseline, TF-IDF + logistic regression, embeddings + LR.

Every model exposes the same two methods, so the API and the evaluation code don't care which
one is plugged in:

    fit(texts, labels) -> self
    predict_proba(texts) -> (n, n_classes) array, columns ordered as `self.classes_`
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .embeddings import Encoder


def group_balanced_weights(labels: Sequence[str], groups: Sequence[str]) -> np.ndarray:
    """Weight each example by 1 / (examples sharing its repository AND label).

    Plain class balancing only equalises Bug/Feature/Docs overall. If one repository supplies
    most of a class (VS Code: 498 Feature vs 139 Bug), the model can learn "this repository's
    writing style -> Feature". Balancing every (repository, label) group removes that shortcut.
    """
    counts = Counter(zip(groups, labels))
    weights = np.array([1.0 / counts[(g, l)] for g, l in zip(groups, labels)])
    return weights * len(weights) / weights.sum()


def _lr_settings(class_weight, labels, groups):
    """Translate our class_weight option into (sklearn class_weight, sample_weight)."""
    if class_weight == "repo_balanced":
        if groups is None:
            raise ValueError("class_weight='repo_balanced' needs the repository of each example")
        return None, group_balanced_weights(labels, groups)
    return class_weight, None


class MajorityClassifier:
    """Always predicts the most frequent training label (shows what 'no understanding' scores)."""

    name = "majority"

    def fit(self, texts: Sequence[str], labels: Sequence[str], groups=None) -> "MajorityClassifier":
        counts = Counter(labels)
        self.classes_ = np.array(sorted(counts))
        self.majority_ = counts.most_common(1)[0][0]
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        probs = np.zeros((len(texts), len(self.classes_)))
        probs[:, list(self.classes_).index(self.majority_)] = 1.0
        return probs

    def params(self) -> dict:
        return {}


class TfidfLogReg:
    """Word 1-2 grams + character 3-5 grams. Char n-grams catch identifiers like `KeyError`."""

    name = "tfidf_logreg"

    def __init__(self, C: float = 4.0, class_weight: str | None = "balanced"):
        self.C = C
        self.class_weight = class_weight

    def fit(self, texts: Sequence[str], labels: Sequence[str], groups=None) -> "TfidfLogReg":
        lr_class_weight, sample_weight = _lr_settings(self.class_weight, labels, groups)
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=100_000, sublinear_tf=True
                    ),
                ),
            ]
        )
        self.pipeline_ = Pipeline(
            [
                ("features", features),
                (
                    "clf",
                    LogisticRegression(C=self.C, class_weight=lr_class_weight, max_iter=3000),
                ),
            ]
        )
        # The vectoriser vocabulary is learned here, from training texts only (no leakage).
        fit_params = {} if sample_weight is None else {"clf__sample_weight": sample_weight}
        self.pipeline_.fit(list(texts), list(labels), **fit_params)
        self.classes_ = self.pipeline_.classes_
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline_.predict_proba(list(texts))

    def params(self) -> dict:
        return {"C": self.C, "class_weight": self.class_weight, "word_ngrams": [1, 2], "char_ngrams": [3, 5]}


class EmbeddingLogReg:
    """Pretrained sentence embeddings (frozen) + logistic regression on top."""

    name = "embedding_logreg"

    def __init__(
        self, model_name: str, C: float = 4.0, class_weight: str | None = "balanced", cache: bool = False
    ):
        self.model_name = model_name
        self.C = C
        self.class_weight = class_weight
        self.cache = cache

    def _encoder(self) -> Encoder:
        return Encoder(self.model_name)

    def __getstate__(self):  # never pickle the experiment-only cache flag into the served model
        return {**self.__dict__, "cache": False}

    def fit(self, texts: Sequence[str], labels: Sequence[str], groups=None) -> "EmbeddingLogReg":
        lr_class_weight, sample_weight = _lr_settings(self.class_weight, labels, groups)
        vectors = self._encoder().encode(list(texts), cache=self.cache)
        self.clf_ = LogisticRegression(C=self.C, class_weight=lr_class_weight, max_iter=3000)
        self.clf_.fit(vectors, list(labels), sample_weight=sample_weight)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.clf_.predict_proba(self._encoder().encode(list(texts), cache=self.cache))

    def params(self) -> dict:
        return {"embedding_model": self.model_name, "C": self.C, "class_weight": self.class_weight}


class AveragingEnsemble:
    """Averages the probabilities of several models: word-matching + meaning-matching."""

    name = "ensemble"

    def __init__(self, members: Sequence):
        self.members = list(members)

    def fit(self, texts: Sequence[str], labels: Sequence[str], groups=None) -> "AveragingEnsemble":
        for member in self.members:
            member.fit(texts, labels, groups=groups)
        self.classes_ = self.members[0].classes_
        for member in self.members[1:]:
            if list(member.classes_) != list(self.classes_):
                raise ValueError("ensemble members disagree on class order")
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return np.mean([member.predict_proba(texts) for member in self.members], axis=0)

    def params(self) -> dict:
        return {"members": [{"model": m.name, **m.params()} for m in self.members]}


def predict_labels(model, texts: Sequence[str]) -> list[str]:
    probs = model.predict_proba(texts)
    return [str(model.classes_[i]) for i in probs.argmax(axis=1)]
