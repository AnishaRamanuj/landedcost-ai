"""Ranking metrics over HTS candidate lists."""

from __future__ import annotations

from collections.abc import Collection, Sequence


def first_hit_rank(codes: Sequence[str], gold: Collection[str]) -> int | None:
    """1-based rank of the first candidate code that falls under any gold code.

    ``codes`` are dotted HTS codes; a candidate matches a gold heading ``8507`` or
    subheading ``8507.60`` when it starts with it (so ``8507.60.00.20`` hits both).
    """
    for rank, code in enumerate(codes, start=1):
        if any(code == g or code.startswith(g + ".") for g in gold):
            return rank
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Share of queries whose first relevant candidate is within the top ``k``."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None], depth: int) -> float:
    """MRR@depth: mean of 1/rank, counting misses (or hits beyond ``depth``) as 0."""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None and r <= depth) / len(ranks)
