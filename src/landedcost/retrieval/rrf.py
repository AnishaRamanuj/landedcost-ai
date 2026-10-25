"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

``score(d) = sum over retrievers r of  w_r / (k + rank_r(d))`` with 1-based ranks.

RRF only looks at ranks, so it needs no score calibration between a cosine similarity and
a BM25 (or ``ts_rank_cd``) score that live on unrelated scales; ``k`` (60 in the paper)
damps the influence of the very top ranks so one retriever's #1 cannot dominate. The
optional weights ``w_r`` (default 1.0) let a stronger retriever count for more.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FusedItem:
    id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def best_rank(self) -> int:
        return min(self.ranks.values())


def max_rrf_score(
    retrievers: Sequence[str], k: int, weights: Mapping[str, float] | None = None
) -> float:
    """Score of an item ranked first by every listed retriever (the normaliser for gating)."""
    w = weights or {}
    return sum(w.get(name, 1.0) / (k + 1) for name in retrievers)


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[str]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[FusedItem]:
    """Fuse ranked id lists. Duplicate ids within one list count once, at their best rank.

    Ties are broken by best individual rank, then by id, so output is deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    w = weights or {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for retriever, ids in ranked_lists.items():
        weight = w.get(retriever, 1.0)
        seen: set[str] = set()
        rank = 0
        for item_id in ids:
            if item_id in seen:
                continue
            seen.add(item_id)
            rank += 1
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
            ranks.setdefault(item_id, {})[retriever] = rank
    fused = [FusedItem(i, s, ranks[i]) for i, s in scores.items()]
    fused.sort(key=lambda f: (-f.score, f.best_rank, f.id))
    return fused
