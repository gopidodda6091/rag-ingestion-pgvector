"""Embedder interface. HashingEmbedder is the deterministic, download-free default;
USE_ST_MODEL=1 switches to sentence-transformers all-MiniLM-L6-v2 when it is installed."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np

LOG = logging.getLogger(__name__)
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the and or of to in on for is are do does i my how what when can".split())


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _stem(token: str) -> str:
    """Tiny suffix stripper so 'claims'/'claim' and 'covered'/'cover' share a bucket."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


class HashingEmbedder:
    """Signed feature hashing of stemmed words and bigrams. Same input -> same vector, always."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        words = [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]
        return words + [f"{a}_{b}" for a, b in pairwise(words)]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = int.from_bytes(
                    hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little"
                )
                out[row, digest % self.dim] += 1.0 if digest >> 63 else -1.0
        return _normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = ST_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return _normalize(np.asarray(self._model.encode(list(texts), batch_size=32)))


def get_embedder() -> Embedder:
    if os.getenv("USE_ST_MODEL", "0") == "1":
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # any import/download failure -> deterministic fallback
            LOG.warning("sentence-transformers unavailable (%s); using HashingEmbedder", exc)
    return HashingEmbedder()
