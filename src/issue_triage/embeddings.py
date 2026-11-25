"""Sentence-embedding encoder with lazy loading, so lexical-only code paths never import torch."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load(model_name: str):
    from sentence_transformers import SentenceTransformer  # heavy import, deferred on purpose

    return SentenceTransformer(model_name)


class Encoder:
    """Turns texts into L2-normalised vectors, so a dot product equals cosine similarity."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        model = _load(self.model_name)
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)
