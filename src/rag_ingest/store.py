"""VectorStore interface with a Postgres/pgvector implementation and an in-memory one for tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag_ingest.chunk import Chunk


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    path: str
    title: str
    content_hash: str


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    doc_id: str
    version: int
    heading: str
    text: str


class VectorStore(Protocol):
    def current_hash(self, doc_id: str) -> str | None: ...
    def doc_ids(self) -> set[str]: ...
    def replace_document(
        self, doc: DocMeta, chunks: Sequence[Chunk], vectors: np.ndarray, model: str
    ) -> int: ...
    def delete_document(self, doc_id: str) -> None: ...
    def chunks(self) -> list[StoredChunk]: ...
    def vector_search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]: ...


class InMemoryStore:
    def __init__(self) -> None:
        self.versions: dict[str, list[DocMeta]] = {}
        self.current: dict[str, DocMeta] = {}
        self._chunks: dict[str, StoredChunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def current_hash(self, doc_id: str) -> str | None:
        meta = self.current.get(doc_id)
        return meta.content_hash if meta else None

    def doc_ids(self) -> set[str]:
        return set(self.current)

    def replace_document(
        self, doc: DocMeta, chunks: Sequence[Chunk], vectors: np.ndarray, model: str
    ) -> int:
        self._drop_chunks(doc.doc_id)
        history = self.versions.setdefault(doc.doc_id, [])
        history.append(doc)
        self.current[doc.doc_id] = doc
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.chunk_id] = StoredChunk(
                chunk.chunk_id, doc.doc_id, len(history), chunk.heading, chunk.text
            )
            self._vectors[chunk.chunk_id] = vector
        return len(history)

    def delete_document(self, doc_id: str) -> None:
        self._drop_chunks(doc_id)
        self.current.pop(doc_id, None)

    def _drop_chunks(self, doc_id: str) -> None:
        for chunk_id in [c for c, v in self._chunks.items() if v.doc_id == doc_id]:
            del self._chunks[chunk_id], self._vectors[chunk_id]

    def chunks(self) -> list[StoredChunk]:
        return sorted(self._chunks.values(), key=lambda c: c.chunk_id)

    def vector_search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._vectors:
            return []
        ids = list(self._vectors)
        scores = np.vstack([self._vectors[i] for i in ids]) @ query
        order = np.argsort(-scores, kind="stable")[:k]
        return [(ids[i], float(scores[i])) for i in order]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS doc_versions (
    doc_id       text        NOT NULL,
    version      integer     NOT NULL,
    content_hash text        NOT NULL,
    path         text        NOT NULL,
    title        text,
    is_current   boolean     NOT NULL DEFAULT true,
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS doc_versions_one_current ON doc_versions (doc_id)
    WHERE is_current;
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id text    PRIMARY KEY,
    doc_id   text    NOT NULL,
    version  integer NOT NULL,
    ordinal  integer NOT NULL,
    heading  text,
    text     text    NOT NULL,
    FOREIGN KEY (doc_id, version) REFERENCES doc_versions (doc_id, version)
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id  text PRIMARY KEY REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    model     text NOT NULL,
    embedding vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS embeddings_hnsw ON embeddings
    USING hnsw (embedding vector_cosine_ops);
"""


class PgVectorStore:
    """Postgres + pgvector. Each document replacement is one transaction."""

    def __init__(self, dsn: str, dim: int, schema: str = "public") -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        self.conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        self.conn.execute(f'SET search_path TO "{schema}", public')
        register_vector(self.conn)
        self.conn.execute(SCHEMA_SQL.format(dim=int(dim)))

    def close(self) -> None:
        self.conn.close()

    def current_hash(self, doc_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT content_hash FROM doc_versions WHERE doc_id = %s AND is_current", (doc_id,)
        ).fetchone()
        return row[0] if row else None

    def doc_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT doc_id FROM doc_versions WHERE is_current").fetchall()
        return {r[0] for r in rows}

    def replace_document(
        self, doc: DocMeta, chunks: Sequence[Chunk], vectors: np.ndarray, model: str
    ) -> int:
        with self.conn.transaction():
            row = self.conn.execute(
                "SELECT coalesce(max(version), 0) + 1 FROM doc_versions WHERE doc_id = %s",
                (doc.doc_id,),
            ).fetchone()
            version = int(row[0]) if row else 1
            self._retire(doc.doc_id)
            self.conn.execute(
                "INSERT INTO doc_versions (doc_id, version, content_hash, path, title) "
                "VALUES (%s, %s, %s, %s, %s)",
                (doc.doc_id, version, doc.content_hash, doc.path, doc.title),
            )
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, version, ordinal, heading, text) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (c.chunk_id, doc.doc_id, version, c.ordinal, c.heading, c.text)
                        for c in chunks
                    ],
                )
                cur.executemany(
                    "INSERT INTO embeddings (chunk_id, model, embedding) VALUES (%s, %s, %s)",
                    [(c.chunk_id, model, v) for c, v in zip(chunks, vectors, strict=True)],
                )
        return version

    def _retire(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        self.conn.execute(
            "UPDATE doc_versions SET is_current = false WHERE doc_id = %s AND is_current",
            (doc_id,),
        )

    def delete_document(self, doc_id: str) -> None:
        with self.conn.transaction():
            self._retire(doc_id)

    def chunks(self) -> list[StoredChunk]:
        rows = self.conn.execute(
            "SELECT chunk_id, doc_id, version, heading, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
        return [StoredChunk(*r) for r in rows]

    def vector_search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        rows = self.conn.execute(
            "SELECT chunk_id, 1 - (embedding <=> %s) AS score FROM embeddings "
            "ORDER BY embedding <=> %s LIMIT %s",
            (query, query, k),
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def embedding_count(self) -> int:
        row = self.conn.execute("SELECT count(*) FROM embeddings").fetchone()
        return int(row[0]) if row else 0
