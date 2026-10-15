"""Incremental, idempotent ingestion: only documents whose content hash changed are
redacted, chunked, embedded and written. Documents removed from disk are retired."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from rag_ingest.chunk import Chunk, chunk_document, slugify
from rag_ingest.embed import Embedder
from rag_ingest.parse import Document, Section, discover, parse_file
from rag_ingest.redact import AuditEntry, AuditLog, redact_text
from rag_ingest.store import DocMeta, VectorStore


@dataclass
class IngestReport:
    run_id: str
    unchanged: list[str] = field(default_factory=list)
    embedded: list[str] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)
    chunks_embedded: int = 0
    redactions: int = 0

    def summary(self) -> str:
        return (
            f"run {self.run_id}: {len(self.embedded)} re-embedded {self.embedded}, "
            f"{len(self.unchanged)} unchanged, {len(self.retired)} retired, "
            f"{self.chunks_embedded} chunks embedded, {self.redactions} redactions"
        )


def redact_document(doc: Document) -> tuple[Document, list[AuditEntry]]:
    """Redact per section, before windowing, so PII can never straddle a chunk boundary."""
    sections, audit = [], []
    for section in doc.sections:
        location = f"{doc.doc_id}::{slugify(section.heading)}"
        text, entries = redact_text(section.text, doc.doc_id, location)
        sections.append(Section(section.heading, section.level, text))
        audit += entries
    return Document(doc.doc_id, doc.path, doc.doc_type, doc.title, sections), audit


def prepare(doc: Document, max_words: int, overlap: int) -> tuple[list[Chunk], list[AuditEntry]]:
    redacted, audit = redact_document(doc)
    return chunk_document(redacted, max_words, overlap), audit


def ingest(
    corpus: Path,
    store: VectorStore,
    embedder: Embedder,
    audit_path: Path,
    max_words: int = 120,
    overlap: int = 20,
) -> IngestReport:
    report = IngestReport(run_id=uuid.uuid4().hex[:12])
    audit_log = AuditLog(audit_path)
    seen: set[str] = set()
    for path in discover(corpus):
        doc = parse_file(path, corpus)
        seen.add(doc.doc_id)
        content_hash = doc.content_hash
        if store.current_hash(doc.doc_id) == content_hash:
            report.unchanged.append(doc.doc_id)
            continue
        chunks, audit = prepare(doc, max_words, overlap)
        vectors = embedder.embed([c.text for c in chunks])
        store.replace_document(
            DocMeta(doc.doc_id, doc.path, doc.title, content_hash), chunks, vectors, embedder.name
        )
        audit_log.write(audit, report.run_id)
        report.embedded.append(doc.doc_id)
        report.chunks_embedded += len(chunks)
        report.redactions += len(audit)
    for doc_id in sorted(store.doc_ids() - seen):
        store.delete_document(doc_id)
        report.retired.append(doc_id)
    return report
