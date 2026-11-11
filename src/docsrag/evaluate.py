"""Retrieval evaluation: did the right section come back, and how high?"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .store import Hit


def first_relevant_rank(hits: Sequence[Hit], expected_section: str) -> int | None:
    """1-based rank of the first hit whose section matches, else None."""
    for rank, hit in enumerate(hits, start=1):
        if hit.section == expected_section:
            return rank
    return None


@dataclass(frozen=True)
class RetrievalReport:
    name: str
    k: int
    hit_rate: float  # fraction of questions with a relevant hit in the top k
    mrr: float  # mean reciprocal rank (0 when not found)
    misses: list[str]


def evaluate(
    name: str,
    search: Callable[[str, int], list[Hit]],
    eval_set: Sequence[tuple[str, str]],
    k: int = 5,
) -> RetrievalReport:
    ranks = [first_relevant_rank(search(q, k), sec) for q, sec in eval_set]
    found = [r for r in ranks if r is not None]
    return RetrievalReport(
        name=name,
        k=k,
        hit_rate=len(found) / len(ranks),
        mrr=sum(1.0 / r for r in found) / len(ranks),
        misses=[q for (q, _), r in zip(eval_set, ranks, strict=True) if r is None],
    )
