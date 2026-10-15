from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from rag_ingest.embed import HashingEmbedder

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
QUESTIONS = ROOT / "eval" / "questions.jsonl"


class CountingEmbedder(HashingEmbedder):
    """HashingEmbedder that records how many texts it embedded."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed(self, texts) -> np.ndarray:  # type: ignore[no-untyped-def]
        self.calls += len(texts)
        return super().embed(texts)


@pytest.fixture(autouse=True)
def no_model_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_ST_MODEL", "0")


@pytest.fixture
def embedder() -> CountingEmbedder:
    return CountingEmbedder()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A private copy of the committed sample corpus (md + html; PDFs if generated)."""
    target = tmp_path / "corpus"
    shutil.copytree(DOCS, target)
    return target
