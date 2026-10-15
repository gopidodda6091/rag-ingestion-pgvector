"""Optional answer step behind an interface. The bundled adapter is extractive and makes
no network or API calls; an LLM-backed adapter can implement the same protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_ingest.chunk import split_sentences
from rag_ingest.search import Hit, tokenize


class AnswerAdapter(Protocol):
    def answer(self, question: str, contexts: Sequence[Hit]) -> str: ...


class ExtractiveAnswerer:
    """Returns the context sentence with the largest word overlap with the question."""

    def answer(self, question: str, contexts: Sequence[Hit]) -> str:
        q_words = set(tokenize(question))
        best, best_score = "", -1
        for hit in contexts:
            body = hit.text.split("\n", 1)[-1]
            for sentence in split_sentences(body):
                score = len(q_words & set(tokenize(sentence)))
                if score > best_score:
                    best, best_score = f"{sentence} [{hit.chunk_id}]", score
        return best or "No answer found in the indexed documents."
