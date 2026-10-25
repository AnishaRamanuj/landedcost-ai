"""In-memory store with the same interface as the Postgres store.

Vector search is exact (numpy dot product over normalised vectors); text search is BM25
over the same text Postgres indexes, with the line's own description counted twice to
mirror the ``A``/``B`` weights of the Postgres ``tsvector``. Used by unit tests, the
offline demo mode (``STORE_BACKEND=memory``) and as a reference for the SQL behaviour.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

from landedcost.embeddings.base import Matrix, Vector
from landedcost.hts.models import HtsLine
from landedcost.rulings.models import Ruling
from landedcost.store.base import IndexInfo, ScoredLine, ScoredRuling, check_embedder
from landedcost.store.bm25 import Bm25Index
from landedcost.text import bm25_terms


def _line_terms(line: HtsLine) -> list[str]:
    return bm25_terms(line.description) * 2 + bm25_terms(" ".join(line.path))


def _ruling_terms(ruling: Ruling) -> list[str]:
    return bm25_terms(ruling.product_description) * 2 + bm25_terms(ruling.excerpt)


class InMemoryStore:
    def __init__(self) -> None:
        self._lines: list[HtsLine] = []
        self._by_code: dict[str, HtsLine] = {}
        self._line_vectors: Matrix = np.zeros((0, 0), dtype=np.float32)
        self._line_bm25 = Bm25Index([])
        self._rulings: dict[str, Ruling] = {}
        self._ruling_vectors: dict[str, Vector] = {}
        self._ruling_bm25 = Bm25Index([])
        self._ruling_order: list[str] = []
        self._embedder: str | None = None
        self._rulings_embedder: str | None = None
        self._dim: int | None = None
        self._hts_version: str | None = None

    # -- writes ---------------------------------------------------------------------------
    async def replace_hts_lines(
        self, lines: Sequence[HtsLine], embeddings: Matrix, *, hts_version: str, embedder: str
    ) -> int:
        if len(lines) != len(embeddings):
            raise ValueError("lines and embeddings differ in length")
        self._lines = list(lines)
        self._by_code = {line.code: line for line in self._lines}
        self._line_vectors = np.asarray(embeddings, dtype=np.float32)
        self._line_bm25 = Bm25Index([_line_terms(line) for line in self._lines])
        self._embedder = embedder
        self._dim = int(self._line_vectors.shape[1]) if len(self._lines) else None
        self._hts_version = hts_version
        return len(self._lines)

    async def upsert_rulings(
        self, rulings: Sequence[Ruling], embeddings: Matrix, *, embedder: str
    ) -> int:
        if len(rulings) != len(embeddings):
            raise ValueError("rulings and embeddings differ in length")
        check_embedder(await self.index_info(), embedder, int(embeddings.shape[1]))
        self._store_rulings(rulings, embeddings, embedder)
        return len(rulings)

    async def replace_rulings(
        self, rulings: Sequence[Ruling], embeddings: Matrix, *, embedder: str
    ) -> int:
        if len(rulings) != len(embeddings):
            raise ValueError("rulings and embeddings differ in length")
        check_embedder(
            await self.index_info(), embedder, int(embeddings.shape[1]), include_rulings=False
        )
        self._rulings.clear()
        self._ruling_vectors.clear()
        self._store_rulings(rulings, embeddings, embedder)
        return len(rulings)

    def _store_rulings(self, rulings: Sequence[Ruling], embeddings: Matrix, embedder: str) -> None:
        for ruling, vector in zip(rulings, embeddings, strict=True):
            self._rulings[ruling.ruling_number] = ruling
            self._ruling_vectors[ruling.ruling_number] = np.asarray(vector, dtype=np.float32)
        self._ruling_order = sorted(self._rulings)
        self._ruling_bm25 = Bm25Index([_ruling_terms(self._rulings[n]) for n in self._ruling_order])
        self._rulings_embedder = embedder if self._rulings else None

    # -- HTS search -----------------------------------------------------------------------
    async def search_hts_vector(self, embedding: Vector, limit: int) -> list[ScoredLine]:
        if not self._lines:
            return []
        scores = self._line_vectors @ np.asarray(embedding, dtype=np.float32)
        top = np.argsort(-scores, kind="stable")[:limit]
        return [ScoredLine(self._lines[i], float(scores[i])) for i in top]

    async def search_hts_text(self, query: str, limit: int) -> list[ScoredLine]:
        hits = self._line_bm25.search(bm25_terms(query), limit)
        return [ScoredLine(self._lines[i], score) for i, score in hits]

    # -- ruling search --------------------------------------------------------------------
    async def search_rulings_vector(
        self, embedding: Vector, limit: int, exclude: Collection[str] = ()
    ) -> list[ScoredRuling]:
        query = np.asarray(embedding, dtype=np.float32)
        scored = [
            ScoredRuling(self._rulings[n], float(self._ruling_vectors[n] @ query))
            for n in self._ruling_order
            if n not in exclude
        ]
        scored.sort(key=lambda s: (-s.score, s.ruling.ruling_number))
        return scored[:limit]

    async def search_rulings_text(
        self, query: str, limit: int, exclude: Collection[str] = ()
    ) -> list[ScoredRuling]:
        hits = self._ruling_bm25.search(bm25_terms(query), limit + len(exclude))
        results = [
            ScoredRuling(self._rulings[self._ruling_order[i]], score)
            for i, score in hits
            if self._ruling_order[i] not in exclude
        ]
        return results[:limit]

    # -- lookups --------------------------------------------------------------------------
    async def get_hts_lines(self, codes: Sequence[str]) -> dict[str, HtsLine]:
        return {c: self._by_code[c] for c in codes if c in self._by_code}

    async def find_hts_by_prefix(self, prefix: str, limit: int) -> list[HtsLine]:
        matches = sorted(
            (ln for ln in self._lines if ln.code.startswith(prefix)), key=lambda ln: ln.code
        )
        return matches[:limit]

    async def list_rulings(self) -> list[Ruling]:
        return [self._rulings[n] for n in self._ruling_order]

    async def index_info(self) -> IndexInfo:
        return IndexInfo(
            hts_lines=len(self._lines),
            rulings=len(self._rulings),
            embedder=self._embedder,
            embedding_dim=self._dim,
            hts_version=self._hts_version,
            rulings_embedder=self._rulings_embedder,
        )

    async def close(self) -> None:
        return None
