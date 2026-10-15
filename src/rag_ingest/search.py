"""Hybrid retrieval: BM25 (rank_bm25) + vector search, fused with reciprocal rank fusion."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from rag_ingest.embed import Embedder
from rag_ingest.store import StoredChunk, VectorStore

_TOKEN = re.compile(r"[a-z0-9]+")
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """score(d) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    heading: str
    text: str


class HybridSearcher:
    """Builds the BM25 index from the store's current chunks at construction time."""

    def __init__(self, store: VectorStore, embedder: Embedder, candidates: int = 20) -> None:
        self.store, self.embedder, self.candidates = store, embedder, candidates
        self.chunks: dict[str, StoredChunk] = {c.chunk_id: c for c in store.chunks()}
        self._ids = list(self.chunks)
        corpus = [tokenize(self.chunks[i].text) for i in self._ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def bm25(self, query: str, k: int) -> list[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self._ids)), key=lambda i: (-scores[i], self._ids[i]))
        return [self._ids[i] for i in order[:k] if scores[i] > 0]

    def vector(self, query: str, k: int) -> list[str]:
        vec = self.embedder.embed([query])[0]
        return [chunk_id for chunk_id, _ in self.store.vector_search(vec, k)]

    def ranked_ids(self, query: str, mode: str = "hybrid", k: int = 5) -> list[str]:
        if mode == "bm25":
            return self.bm25(query, k)
        if mode == "vector":
            return self.vector(query, k)
        if mode != "hybrid":
            raise ValueError(f"unknown mode {mode!r}")
        fused = reciprocal_rank_fusion(
            [self.bm25(query, self.candidates), self.vector(query, self.candidates)]
        )
        return [chunk_id for chunk_id, _ in fused[:k]]

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        ids = self.ranked_ids(query, mode, k)
        return [
            Hit(i, 1.0 / (rank + 1), self.chunks[i].heading, self.chunks[i].text)
            for rank, i in enumerate(ids)
            if i in self.chunks
        ]
