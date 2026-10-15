from pathlib import Path

from rag_ingest.ingest import ingest
from rag_ingest.store import InMemoryStore


def test_reingest_is_idempotent_and_only_changed_doc_is_reembedded(
    corpus: Path, embedder, tmp_path: Path
) -> None:
    store, audit = InMemoryStore(), tmp_path / "audit.jsonl"
    first = ingest(corpus, store, embedder, audit)
    n_docs = len(first.embedded)
    assert n_docs >= 9 and first.unchanged == []
    chunks_before = {c.chunk_id: c for c in store.chunks()}
    calls_after_first = embedder.calls

    second = ingest(corpus, store, embedder, audit)
    assert second.embedded == [] and len(second.unchanged) == n_docs
    assert embedder.calls == calls_after_first
    assert {c.chunk_id: c for c in store.chunks()} == chunks_before

    edited = corpus / "telehealth-policy.md"
    edited.write_text(edited.read_text().replace("$0 copay", "$5 copay"))
    third = ingest(corpus, store, embedder, audit)
    assert third.embedded == ["telehealth-policy"]
    assert embedder.calls - calls_after_first == third.chunks_embedded
    assert (
        store.versions["telehealth-policy"][-1].content_hash
        != store.versions["telehealth-policy"][0].content_hash
    )
    assert len(store.versions["telehealth-policy"]) == 2
    assert len(store.versions["member-handbook"]) == 1
    assert any("$5 copay" in c.text for c in store.chunks())


def test_deleted_document_is_retired(corpus: Path, embedder, tmp_path: Path) -> None:
    store = InMemoryStore()
    ingest(corpus, store, embedder, tmp_path / "a.jsonl")
    (corpus / "faq-billing.md").unlink()
    report = ingest(corpus, store, embedder, tmp_path / "a.jsonl")
    assert report.retired == ["faq-billing"] and report.embedded == []
    assert not any(c.doc_id == "faq-billing" for c in store.chunks())
