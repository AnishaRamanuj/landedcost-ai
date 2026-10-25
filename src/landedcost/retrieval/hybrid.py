"""Hybrid retriever over HTS lines and (optionally) CBP rulings.

For a query it runs, concurrently:

* ``vector``: HNSW cosine search over HTS line embeddings,
* ``fts``:    full-text search over the weighted ``tsvector`` (GIN-filtered, ranked with
  Okapi BM25 by default; Postgres ``ts_rank_cd`` is available via ``FTS_RANKING``),
* the same two searches over rulings, when any rulings are loaded.

Rulings are fused among themselves first; the HTS lines they cite then form a third ranked
list (``rulings``) that votes in the HTS fusion. A ruling therefore cannot become a
candidate itself, but it can lift the tariff line a similar product was classified under,
and it is returned alongside as citable evidence.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from landedcost.embeddings.base import Embedder, Vector
from landedcost.hts.codes import CodeLevel, HtsCode
from landedcost.hts.models import HtsLine
from landedcost.retrieval.rrf import FusedItem, reciprocal_rank_fusion
from landedcost.rulings.models import Ruling
from landedcost.store.base import IndexInfo, ScoredLine, ScoredRuling, Store, check_embedder

VECTOR = "vector"
FTS = "fts"
RULINGS = "rulings"


class RetrievalMode(StrEnum):
    VECTOR = "vector"
    FTS = "fts"
    HYBRID = "hybrid"

    @property
    def retrievers(self) -> tuple[str, ...]:
        return {
            RetrievalMode.VECTOR: (VECTOR,),
            RetrievalMode.FTS: (FTS,),
            RetrievalMode.HYBRID: (VECTOR, FTS),
        }[self]


@dataclass(frozen=True, slots=True)
class Candidate:
    line: HtsLine
    fused_score: float
    ranks: dict[str, int]
    scores: dict[str, float]
    """Raw per-retriever scores (cosine similarity, BM25 or ``ts_rank_cd``), for tracing only;
    fusion uses ranks, because these scales are not comparable."""
    supporting_rulings: tuple[str, ...] = ()

    @property
    def chunk_id(self) -> str:
        return self.line.chunk_id


@dataclass(frozen=True, slots=True)
class RulingCandidate:
    ruling: Ruling
    fused_score: float
    ranks: dict[str, int]

    @property
    def chunk_id(self) -> str:
        return self.ruling.chunk_id


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    mode: RetrievalMode
    rrf_k: int
    candidates: list[Candidate]
    rulings: list[RulingCandidate] = field(default_factory=list)
    hits_per_retriever: dict[str, int] = field(default_factory=dict)
    retrievers_run: tuple[str, ...] = ()
    rrf_weights: Mapping[str, float] = field(default_factory=dict)
    """Per-retriever RRF weights used for this fusion (missing names weigh 1.0)."""
    indexed_lines: int | None = None
    """HTS lines in the index when the query ran (0 means nothing was ingested yet)."""


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    rrf_k: int = 60
    candidate_pool: int = 50
    """Hits requested from each underlying retriever before fusion."""
    ruling_pool: int = 10
    """Top fused rulings whose cited HTS lines vote in the HTS fusion."""
    rrf_weights: Mapping[str, float] = field(default_factory=dict)
    """Per-retriever weights (``vector``, ``fts``, ``rulings``); missing names weigh 1.0."""
    index_check_ttl_s: float = 30.0
    """How long a successful embedder/row-count check of the index is reused."""


@dataclass(frozen=True, slots=True)
class _IndexCheck:
    info: IndexInfo
    at: float


class HybridRetriever:
    def __init__(self, store: Store, embedder: Embedder, config: RetrieverConfig | None = None):
        self._store = store
        self._embedder = embedder
        self._config = config or RetrieverConfig()
        self._index_check: _IndexCheck | None = None

    @property
    def config(self) -> RetrieverConfig:
        return self._config

    async def _checked_index(self) -> IndexInfo:
        """Index metadata, verified against the configured embedder.

        A passing check is reused for ``index_check_ttl_s`` so a long-running API notices a
        re-ingest (or newly imported rulings) without paying a metadata query per request.
        An empty index is never cached: the next request checks again.
        """
        cached = self._index_check
        now = time.monotonic()
        if cached is not None and now - cached.at < self._config.index_check_ttl_s:
            return cached.info
        info = await self._store.index_info()
        check_embedder(info, self._embedder.name, self._embedder.dim)
        self._index_check = _IndexCheck(info, now) if info.embedder is not None else None
        return info

    async def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 10,
        exclude_rulings: Collection[str] = (),
        use_rulings: bool = True,
    ) -> RetrievalResult:
        """Retrieve and fuse candidates for ``query``.

        ``exclude_rulings`` supports leave-one-out evaluation on ruling-derived items.
        """
        info = await self._checked_index()
        pool = max(self._config.candidate_pool, top_k)
        embedding: Vector | None = None
        if VECTOR in mode.retrievers:
            embedding = await self._embedder.embed_query(query)

        # With no rulings loaded, skip their searches entirely: each would hold a pooled
        # connection for nothing while the HTS searches run.
        search_rulings = use_rulings and info.rulings > 0
        hts_lists, ruling_hits = await asyncio.gather(
            self._search_hts(query, embedding, mode, pool),
            self._search_rulings(query, embedding, mode, exclude_rulings)
            if search_rulings
            else _no_rulings(),
        )

        ranked: dict[str, list[str]] = {}
        lines: dict[str, HtsLine] = {}
        scores: dict[str, dict[str, float]] = {}
        for name, hits in hts_lists.items():
            ranked[name] = [h.line.chunk_id for h in hits]
            for h in hits:
                lines[h.line.chunk_id] = h.line
                scores.setdefault(h.line.chunk_id, {})[name] = h.score

        ruling_candidates = _fuse_rulings(ruling_hits, self._config.rrf_k)
        retrievers_run = mode.retrievers
        support: dict[str, list[str]] = {}
        if ruling_candidates:
            cited = await self._lines_cited_by(ruling_candidates[: self._config.ruling_pool])
            ranked[RULINGS] = [line.chunk_id for line, _ in cited]
            for line, ruling_number in cited:
                lines.setdefault(line.chunk_id, line)
                support.setdefault(line.chunk_id, []).append(ruling_number)
            retrievers_run = (*retrievers_run, RULINGS)

        weights = dict(self._config.rrf_weights)
        fused = reciprocal_rank_fusion(ranked, k=self._config.rrf_k, weights=weights)
        candidates = [_candidate(item, lines, scores, support) for item in fused[: max(top_k, 1)]]
        return RetrievalResult(
            query=query,
            mode=mode,
            rrf_k=self._config.rrf_k,
            candidates=candidates,
            rulings=ruling_candidates,
            hits_per_retriever={name: len(ids) for name, ids in ranked.items()},
            retrievers_run=retrievers_run,
            rrf_weights=weights,
            indexed_lines=info.hts_lines,
        )

    async def _search_hts(
        self, query: str, embedding: Vector | None, mode: RetrievalMode, pool: int
    ) -> dict[str, list[ScoredLine]]:
        names: list[str] = []
        tasks = []
        if embedding is not None:
            names.append(VECTOR)
            tasks.append(self._store.search_hts_vector(embedding, pool))
        if FTS in mode.retrievers:
            names.append(FTS)
            tasks.append(self._store.search_hts_text(query, pool))
        results = await asyncio.gather(*tasks)
        return dict(zip(names, results, strict=True))

    async def _search_rulings(
        self,
        query: str,
        embedding: Vector | None,
        mode: RetrievalMode,
        exclude: Collection[str],
    ) -> dict[str, list[ScoredRuling]]:
        pool = self._config.ruling_pool * 2
        names: list[str] = []
        tasks = []
        if embedding is not None:
            names.append(VECTOR)
            tasks.append(self._store.search_rulings_vector(embedding, pool, exclude))
        if FTS in mode.retrievers:
            names.append(FTS)
            tasks.append(self._store.search_rulings_text(query, pool, exclude))
        results = await asyncio.gather(*tasks)
        return dict(zip(names, results, strict=True))

    async def _lines_cited_by(
        self, rulings: Sequence[RulingCandidate]
    ) -> list[tuple[HtsLine, str]]:
        """Resolve each ruling's HTS codes to indexed lines, in ruling rank order.

        A ruling may cite a 10-digit code that is not in this HTS edition; we fall back to
        its nearest indexed ancestor (8, 6, then 4 digits).
        """
        wanted: list[tuple[str, list[str]]] = []
        lookup: set[str] = set()
        for rc in rulings:
            for raw in rc.ruling.hts_codes:
                code = HtsCode.parse(raw)
                chain = [
                    code.truncate(level).dotted
                    for level in (
                        CodeLevel.STATISTICAL,
                        CodeLevel.TARIFF_LINE,
                        CodeLevel.SUBHEADING,
                        CodeLevel.HEADING,
                    )
                ]
                chain = list(dict.fromkeys(chain))
                wanted.append((rc.ruling.ruling_number, chain))
                lookup.update(chain)
        found = await self._store.get_hts_lines(sorted(lookup))
        cited: list[tuple[HtsLine, str]] = []
        for ruling_number, chain in wanted:
            line = next((found[c] for c in chain if c in found), None)
            if line is not None:
                cited.append((line, ruling_number))
        return cited


async def _no_rulings() -> dict[str, list[ScoredRuling]]:
    return {}


def _fuse_rulings(hits: dict[str, list[ScoredRuling]], k: int) -> list[RulingCandidate]:
    if not hits:
        return []
    by_id = {h.ruling.chunk_id: h.ruling for lst in hits.values() for h in lst}
    fused = reciprocal_rank_fusion(
        {n: [h.ruling.chunk_id for h in lst] for n, lst in hits.items()}, k=k
    )
    return [RulingCandidate(by_id[f.id], f.score, f.ranks) for f in fused]


def _candidate(
    item: FusedItem,
    lines: dict[str, HtsLine],
    scores: dict[str, dict[str, float]],
    support: dict[str, list[str]],
) -> Candidate:
    return Candidate(
        line=lines[item.id],
        fused_score=item.score,
        ranks=item.ranks,
        scores=scores.get(item.id, {}),
        supporting_rulings=tuple(dict.fromkeys(support.get(item.id, []))),
    )
