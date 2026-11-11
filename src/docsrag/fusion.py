"""Combine ranked lists from different retrievers. Pure functions: easy to test."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Merge ranked id lists. An id's score is the sum of 1 / (k + rank) over lists.

    Rank-based fusion needs no score calibration: cosine similarity and text-search
    ranks live on different scales, but positions in a list are comparable.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
