"""Command line: ingest, search, eval and the scripted demo."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

from rag_ingest.answer import ExtractiveAnswerer
from rag_ingest.embed import Embedder, get_embedder
from rag_ingest.evaluate import evaluate, format_table, load_questions
from rag_ingest.ingest import ingest
from rag_ingest.search import HybridSearcher
from rag_ingest.store import InMemoryStore, PgVectorStore, VectorStore

LOG = logging.getLogger("rag_ingest")
OUTPUT = Path(os.getenv("OUTPUT_DIR", "output"))
AUDIT_PATH = OUTPUT / "audit" / "redactions.jsonl"
SCHEMA = os.getenv("PG_SCHEMA", "rag")


def open_store(embedder: Embedder, fresh: bool = False) -> VectorStore:
    """PgVectorStore when PG_DSN is set and reachable, else InMemoryStore."""
    dsn = os.getenv("PG_DSN")
    if not dsn:
        print("PG_DSN not set: using InMemoryStore")
        return InMemoryStore()
    try:
        if fresh:
            import psycopg

            with psycopg.connect(dsn, autocommit=True) as conn:
                conn.execute(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE')
        store = PgVectorStore(dsn, embedder.dim, SCHEMA)
    except Exception as exc:  # unreachable database -> keep the demo runnable
        print(f"Postgres unavailable ({type(exc).__name__}): using InMemoryStore")
        return InMemoryStore()
    print(f"using PgVectorStore (schema {SCHEMA})")
    return store


def run_eval(store: VectorStore, embedder: Embedder, questions: Path, k: int = 5) -> str:
    searcher = HybridSearcher(store, embedder)
    qs = load_questions(questions)
    by_mode = {
        mode: evaluate(qs, lambda q, m=mode: searcher.ranked_ids(q, m, k))
        for mode in ("bm25", "vector", "hybrid")
    }
    table = format_table(by_mode)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "retrieval_eval.md").write_text(table + "\n")
    return table


def demo(corpus: Path, questions: Path) -> None:
    embedder = get_embedder()
    store = open_store(embedder, fresh=True)
    work = OUTPUT / "corpus"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(corpus, work)
    AUDIT_PATH.unlink(missing_ok=True)

    print("1) initial ingest:\n  ", ingest(work, store, embedder, AUDIT_PATH).summary())
    print("2) re-ingest, nothing changed:\n  ", ingest(work, store, embedder, AUDIT_PATH).summary())
    edited = work / "telehealth-policy.md"
    edited.write_text(
        edited.read_text() + "\n## Mental Health Apps\n\nTwo approved "
        "mindfulness apps are free for members.\n"
    )
    print(
        "3) edited telehealth-policy.md, re-ingest:\n  ",
        ingest(work, store, embedder, AUDIT_PATH).summary(),
    )
    print(
        f"4) retrieval eval ({len(load_questions(questions))} questions, embedder "
        f"{embedder.name}):\n{run_eval(store, embedder, questions)}"
    )
    question = "How long do I have to appeal a denied claim?"
    hits = HybridSearcher(store, embedder).search(question, k=3)
    print(
        f"5) answer adapter (extractive, no API): {question}\n   "
        f"{ExtractiveAnswerer().answer(question, hits)}"
    )
    print(f"audit log: {AUDIT_PATH} (types and keyed hashes only)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag_ingest")
    parser.add_argument("--corpus", type=Path, default=Path("docs"))
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    sub.add_parser("eval")
    sub.add_parser("demo")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    search.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.cmd == "demo":
        demo(args.corpus, args.questions)
        return
    embedder = get_embedder()
    store = open_store(embedder)
    if isinstance(store, InMemoryStore) or args.cmd == "ingest":
        print(ingest(args.corpus, store, embedder, AUDIT_PATH).summary())
    if args.cmd == "eval":
        print(run_eval(store, embedder, args.questions))
    elif args.cmd == "search":
        for hit in HybridSearcher(store, embedder).search(args.query, args.k, args.mode):
            print(f"{hit.chunk_id:55s} {hit.text.splitlines()[-1][:90]}")


if __name__ == "__main__":
    main()
