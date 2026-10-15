"""Retrieval evaluation: recall@k and MRR against questions with gold chunk ids."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    gold_chunk_ids: tuple[str, ...]


def load_questions(path: Path) -> list[EvalQuestion]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [EvalQuestion(r["question"], tuple(r["gold_chunk_ids"])) for r in rows]


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Share of gold chunks found in the top k."""
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def evaluate(
    questions: Sequence[EvalQuestion],
    retrieve: Callable[[str], list[str]],
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    if not questions:
        raise ValueError("no evaluation questions")
    results = [(retrieve(q.question), q.gold_chunk_ids) for q in questions]
    metrics = {
        f"recall@{k}": sum(recall_at_k(r, g, k) for r, g in results) / len(results) for k in ks
    }
    metrics["mrr"] = sum(reciprocal_rank(r, g) for r, g in results) / len(results)
    return metrics


def format_table(by_mode: dict[str, dict[str, float]]) -> str:
    cols = list(next(iter(by_mode.values())))
    lines = ["| mode | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for mode, metrics in by_mode.items():
        lines.append(f"| {mode} | " + " | ".join(f"{metrics[c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)
