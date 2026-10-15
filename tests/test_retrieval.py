import pytest

from rag_ingest.answer import ExtractiveAnswerer
from rag_ingest.evaluate import evaluate, load_questions, recall_at_k, reciprocal_rank
from rag_ingest.ingest import ingest
from rag_ingest.search import HybridSearcher, reciprocal_rank_fusion
from rag_ingest.store import InMemoryStore

from .conftest import QUESTIONS


def test_recall_and_mrr() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["c"], 2) == 0.0
    assert recall_at_k(retrieved, ["c"], 3) == 1.0
    assert recall_at_k(retrieved, ["a", "z"], 5) == 0.5
    assert reciprocal_rank(retrieved, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(retrieved, ["z"]) == 0.0


def test_evaluate_aggregates() -> None:
    class Q:
        def __init__(self, q, g):
            self.question, self.gold_chunk_ids = q, g

    answers = {"q1": ["x", "g1"], "q2": ["g2"]}
    metrics = evaluate([Q("q1", ("g1",)), Q("q2", ("g2",))], answers.__getitem__, ks=(1, 2))
    assert metrics == {"recall@1": 0.5, "recall@2": 1.0, "mrr": 0.75}


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert {fused[0][0], fused[1][0]} == {"a", "b"}
    assert fused[-1][0] in {"c", "d"}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_hybrid_retrieval_quality_on_sample_corpus(corpus, embedder, tmp_path) -> None:
    store = InMemoryStore()
    ingest(corpus, store, embedder, tmp_path / "audit.jsonl")
    searcher = HybridSearcher(store, embedder)
    questions = [
        q
        for q in load_questions(QUESTIONS)
        if q.gold_chunk_ids[0].split("::")[0] in store.doc_ids()
    ]
    assert len(questions) >= 15
    metrics = evaluate(questions, lambda q: searcher.ranked_ids(q, "hybrid", 5))
    assert metrics["recall@5"] >= 0.9
    assert metrics["mrr"] >= 0.8
    hits = searcher.search("appeal a denied claim deadline", k=3)
    answer = ExtractiveAnswerer().answer("How many days to appeal a denied claim?", hits)
    assert "60 days" in answer
