"""A small Okapi BM25 index used by the in-memory store."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence


class Bm25Index:
    """BM25 with OR semantics: any document sharing at least one term is a candidate."""

    def __init__(self, documents: Sequence[Sequence[str]], *, k1: float = 1.2, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths = [len(doc) for doc in documents]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        for doc_id, doc in enumerate(documents):
            for term, tf in Counter(doc).items():
                self._postings[term].append((doc_id, tf))
        n = len(documents)
        self._idf = {
            term: math.log(1 + (n - len(posts) + 0.5) / (len(posts) + 0.5))
            for term, posts in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self._lengths)

    def search(self, query_terms: Sequence[str], limit: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in dict.fromkeys(query_terms):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self._postings[term]:
                norm = 1 - self._b + self._b * (self._lengths[doc_id] / self._avg_len)
                scores[doc_id] += idf * tf * (self._k1 + 1) / (tf + self._k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:limit]
