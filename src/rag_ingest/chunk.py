"""Heading-aware chunking: never cross a section boundary; overlap windows inside a section."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_ingest.parse import Document, Section

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    heading: str
    text: str

    @property
    def token_count(self) -> int:
        return len(self.text.split())


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def window_words(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    """Fixed-size windows with `overlap` shared words between neighbours."""
    if max_words <= 0 or not 0 <= overlap < max_words:
        raise ValueError("require max_words > 0 and 0 <= overlap < max_words")
    if len(words) <= max_words:
        return [words] if words else []
    step = max_words - overlap
    windows = []
    for start in range(0, len(words), step):
        windows.append(words[start : start + max_words])
        if start + max_words >= len(words):
            break
    return windows


def chunk_section(doc_id: str, section: Section, max_words: int, overlap: int) -> list[Chunk]:
    slug = slugify(section.heading) or "section"
    parts = window_words(section.text.split(), max_words, overlap)
    return [
        Chunk(
            f"{doc_id}::{slug}::{i}",
            doc_id,
            0,
            section.heading,
            f"{section.heading}\n{' '.join(words)}",
        )
        for i, words in enumerate(parts)
    ]


def chunk_document(doc: Document, max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    """Chunk ids are `doc::heading-slug::n`, stable when other sections change."""
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for section in doc.sections:
        for chunk in chunk_section(doc.doc_id, section, max_words, overlap):
            seen[chunk.chunk_id] = seen.get(chunk.chunk_id, 0) + 1
            chunk_id = chunk.chunk_id
            if seen[chunk_id] > 1:  # repeated heading text in one document
                chunk_id = f"{chunk_id}-{seen[chunk_id]}"
            chunks.append(Chunk(chunk_id, doc.doc_id, len(chunks), chunk.heading, chunk.text))
    return chunks


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text) if s]
