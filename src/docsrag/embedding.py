"""Embedders turn text into unit-length vectors. Swap them without touching callers."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

TOKEN = re.compile(r"[a-z0-9_]+")
STOP = frozenset(
    "a an and are as at be but by do does for from how i if in is it its of on or "
    "so that the their then there these this to was what when where which who why "
    "will with you your can my not".split()
)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), rows L2-normalised."""
        ...


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class HashingEmbedder:
    """Deterministic, offline, no model download: hashed bag of words and bigrams.

    A *lexical* embedder: it matches shared words, not meaning. It exists so the
    whole pipeline can be tested without a neural model; production swaps in a
    real model (SentenceTransformerEmbedder) behind the same interface.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _slot(self, feature: str) -> tuple[int, float]:
        # hashlib, not hash(): Python's built-in hash is salted per process.
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokens(text)
            feats = Counter(toks) + Counter(
                f"{a} {b}" for a, b in zip(toks, toks[1:], strict=False)
            )
            for feature, count in feats.items():
                slot, sign = self._slot(feature)
                out[row, slot] += sign * (1.0 + math.log(count))
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder:
    """A real neural embedder (needs `uv sync --extra hf` and a model download)."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        from sentence_transformers import SentenceTransformer  # heavy: import lazily

        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)
