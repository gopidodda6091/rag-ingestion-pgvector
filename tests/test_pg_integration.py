"""Runs only when PG_DSN points at a Postgres with the pgvector extension available."""

import os
import uuid

import pytest

from rag_ingest.ingest import ingest
from rag_ingest.search import HybridSearcher

PG_DSN = os.getenv("PG_DSN")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not PG_DSN, reason="PG_DSN not set")]


@pytest.fixture
def pg_store(embedder):
    import psycopg

    from rag_ingest.store import PgVectorStore

    schema = f"rag_test_{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(PG_DSN, embedder.dim, schema)
    yield store
    store.close()
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def test_pgvector_incremental_ingest_and_search(pg_store, corpus, embedder, tmp_path) -> None:
    audit = tmp_path / "audit.jsonl"
    first = ingest(corpus, pg_store, embedder, audit)
    assert pg_store.embedding_count() == first.chunks_embedded > 0
    index = pg_store.conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'embeddings_hnsw' "
        "AND schemaname = current_schema()"
    ).fetchone()
    assert index and "hnsw" in index[0]

    assert ingest(corpus, pg_store, embedder, audit).embedded == []
    edited = corpus / "pharmacy-benefits.md"
    edited.write_text(edited.read_text() + "\n## Vaccines\n\nFlu shots are free at pharmacies.\n")
    third = ingest(corpus, pg_store, embedder, audit)
    assert third.embedded == ["pharmacy-benefits"]
    versions = pg_store.conn.execute(
        "SELECT version, is_current FROM doc_versions WHERE doc_id = 'pharmacy-benefits' "
        "ORDER BY version"
    ).fetchall()
    assert versions == [(1, False), (2, True)]
    assert pg_store.embedding_count() == len(pg_store.chunks())

    hits = HybridSearcher(pg_store, embedder).search("free flu shots", k=3)
    assert hits[0].chunk_id == "pharmacy-benefits::vaccines::0"
