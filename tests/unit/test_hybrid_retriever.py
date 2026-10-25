import asyncio
from collections.abc import Collection, Sequence

import pytest

from landedcost.embeddings.base import Matrix, Vector
from landedcost.embeddings.hashing import HashEmbedder
from landedcost.hts.models import HtsLine
from landedcost.retrieval.hybrid import (
    FTS,
    RULINGS,
    VECTOR,
    HybridRetriever,
    RetrievalMode,
    RetrieverConfig,
)
from landedcost.rulings.models import Ruling
from landedcost.store.base import EmbedderMismatchError, IndexInfo, ScoredLine, ScoredRuling
from landedcost.store.memory import InMemoryStore


async def test_hybrid_fuses_both_retrievers(
    memory_store: InMemoryStore, embedder: HashEmbedder
) -> None:
    retriever = HybridRetriever(memory_store, embedder)
    result = await retriever.retrieve("rechargeable lithium-ion battery pack", top_k=5)
    assert result.mode is RetrievalMode.HYBRID
    assert result.retrievers_run == (VECTOR, FTS)
    top = result.candidates[0]
    assert top.line.code.startswith("8507.60")
    assert set(top.ranks) == {VECTOR, FTS}
    assert top.fused_score == pytest.approx(sum(1 / (60 + r) for r in top.ranks.values()))
    assert [c.fused_score for c in result.candidates] == sorted(
        (c.fused_score for c in result.candidates), reverse=True
    )
    assert len(result.candidates) == 5
    assert result.rulings == []


@pytest.mark.parametrize("mode", [RetrievalMode.VECTOR, RetrievalMode.FTS])
async def test_single_retriever_modes(
    memory_store: InMemoryStore, embedder: HashEmbedder, mode: RetrievalMode
) -> None:
    result = await HybridRetriever(memory_store, embedder).retrieve("natural honey", mode=mode)
    assert result.retrievers_run == (mode.value,)
    assert all(set(c.ranks) == {mode.value} for c in result.candidates)
    assert result.candidates[0].line.heading == "0409"


async def test_rulings_vote_for_the_lines_they_cite(
    store_with_rulings: InMemoryStore, embedder: HashEmbedder
) -> None:
    retriever = HybridRetriever(store_with_rulings, embedder)
    result = await retriever.retrieve("lithium battery module for electric kick scooters")
    assert result.retrievers_run == (VECTOR, FTS, RULINGS)
    assert result.rulings[0].ruling.ruling_number == "TEST-0001"
    cited = next(c for c in result.candidates if c.line.code == "8507.60.00.90")
    assert RULINGS in cited.ranks
    assert "TEST-0001" in cited.supporting_rulings


async def test_ruling_citing_unknown_suffix_falls_back_to_ancestor(
    store_with_rulings: InMemoryStore, embedder: HashEmbedder
) -> None:
    # TEST-0003 cites 8507.60.0020, which does not exist in this HTS edition.
    result = await HybridRetriever(store_with_rulings, embedder).retrieve(
        "power bank cells assembled into a battery pack"
    )
    supported = {c.line.code: c.supporting_rulings for c in result.candidates}
    assert "TEST-0003" in supported["8507.60.00"]


async def test_exclude_rulings_supports_leave_one_out(
    store_with_rulings: InMemoryStore, embedder: HashEmbedder
) -> None:
    result = await HybridRetriever(store_with_rulings, embedder).retrieve(
        "clover honey squeeze bottles", exclude_rulings={"TEST-0002"}
    )
    assert "TEST-0002" not in {r.ruling.ruling_number for r in result.rulings}
    without = await HybridRetriever(store_with_rulings, embedder).retrieve(
        "clover honey squeeze bottles", use_rulings=False
    )
    assert without.rulings == []
    assert RULINGS not in without.retrievers_run


class _ConcurrencyProbeStore(InMemoryStore):
    """A two-way barrier: each search announces it started, then waits for the other one.

    Sequential execution (in either order) deadlocks the first search until its timeout, so
    the test passes only if both searches are in flight at the same time.
    """

    def __init__(self, timeout_s: float = 2.0) -> None:
        super().__init__()
        self.timeout_s = timeout_s
        self.vector_started = asyncio.Event()
        self.text_started = asyncio.Event()

    async def search_hts_vector(self, embedding: Vector, limit: int) -> list[ScoredLine]:
        self.vector_started.set()
        await asyncio.wait_for(self.text_started.wait(), timeout=self.timeout_s)
        return await super().search_hts_vector(embedding, limit)

    async def search_hts_text(self, query: str, limit: int) -> list[ScoredLine]:
        self.text_started.set()
        await asyncio.wait_for(self.vector_started.wait(), timeout=self.timeout_s)
        return await super().search_hts_text(query, limit)


class _SequentialRetriever(HybridRetriever):
    """Control for the probe: awaits the two searches one after the other."""

    async def _search_hts(
        self, query: str, embedding: Vector | None, mode: RetrievalMode, pool: int
    ) -> dict[str, list[ScoredLine]]:
        assert embedding is not None
        text = await self._store.search_hts_text(query, pool)
        vector = await self._store.search_hts_vector(embedding, pool)
        return {VECTOR: vector, FTS: text}


async def test_vector_and_text_searches_run_concurrently(
    sample_lines: list[HtsLine], embedder: HashEmbedder
) -> None:
    store = _ConcurrencyProbeStore()
    vectors = await embedder.embed_documents([line.embedding_text for line in sample_lines])
    await store.replace_hts_lines(sample_lines, vectors, hts_version="v", embedder=embedder.name)
    result = await HybridRetriever(store, embedder).retrieve("honey")
    assert result.candidates

    sequential = _ConcurrencyProbeStore(timeout_s=0.2)
    await sequential.replace_hts_lines(
        sample_lines, vectors, hts_version="v", embedder=embedder.name
    )
    with pytest.raises(TimeoutError):  # the probe really detects sequential awaits
        await _SequentialRetriever(sequential, embedder).retrieve("honey")


async def test_candidate_pool_and_top_k(
    memory_store: InMemoryStore, embedder: HashEmbedder
) -> None:
    retriever = HybridRetriever(memory_store, embedder, RetrieverConfig(candidate_pool=3))
    result = await retriever.retrieve("cotton t-shirts", top_k=2)
    assert len(result.candidates) == 2
    assert result.hits_per_retriever[VECTOR] == 3


class _StaleIndexStore(InMemoryStore):
    async def index_info(self) -> IndexInfo:
        return IndexInfo(1, 0, "openai:text-embedding-3-small:384", 384, "v")


async def test_refuses_index_built_by_another_embedder(embedder: HashEmbedder) -> None:
    with pytest.raises(EmbedderMismatchError):
        await HybridRetriever(_StaleIndexStore(), embedder).retrieve("honey")


class _NoopStore(InMemoryStore):
    async def upsert_rulings(
        self, rulings: Sequence[Ruling], embeddings: Matrix, *, embedder: str
    ) -> int:
        return 0

    async def search_rulings_text(
        self, query: str, limit: int, exclude: Collection[str] = ()
    ) -> list[ScoredRuling]:
        raise AssertionError("not called when use_rulings=False")


async def test_use_rulings_false_skips_ruling_searches(embedder: HashEmbedder) -> None:
    result = await HybridRetriever(_NoopStore(), embedder).retrieve("honey", use_rulings=False)
    assert result.candidates == []


async def test_stale_ruling_vectors_are_refused(
    store_with_rulings: InMemoryStore, sample_lines: list[HtsLine]
) -> None:
    # Re-index the HTS directly (bypassing ingest_hts, which would re-embed the rulings).
    other = HashEmbedder(char_ngram_range=(2, 4))
    vectors = await other.embed_documents([line.embedding_text for line in sample_lines])
    await store_with_rulings.replace_hts_lines(
        sample_lines, vectors, hts_version="v", embedder=other.name
    )
    with pytest.raises(EmbedderMismatchError, match="rulings were embedded"):
        await HybridRetriever(store_with_rulings, other).retrieve("honey")


class _CountingStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.info_calls = 0

    async def index_info(self) -> IndexInfo:
        self.info_calls += 1
        return await super().index_info()

    async def search_rulings_vector(
        self, embedding: Vector, limit: int, exclude: Collection[str] = ()
    ) -> list[ScoredRuling]:
        if not self._rulings:
            raise AssertionError("ruling search must be skipped when no rulings are loaded")
        return await super().search_rulings_vector(embedding, limit, exclude)


async def test_index_check_is_cached_only_after_a_successful_ingest(
    sample_lines: list[HtsLine], embedder: HashEmbedder
) -> None:
    store = _CountingStore()
    retriever = HybridRetriever(store, embedder, RetrieverConfig(index_check_ttl_s=3600))
    assert (await retriever.retrieve("honey")).candidates == []  # empty index: not cached
    other = HashEmbedder(char_ngram_range=(2, 4))
    vectors = await other.embed_documents([line.embedding_text for line in sample_lines])
    await store.replace_hts_lines(sample_lines, vectors, hts_version="v", embedder=other.name)
    with pytest.raises(EmbedderMismatchError):
        await retriever.retrieve("honey")

    good = _CountingStore()
    vectors = await embedder.embed_documents([line.embedding_text for line in sample_lines])
    await good.replace_hts_lines(sample_lines, vectors, hts_version="v", embedder=embedder.name)
    cached = HybridRetriever(good, embedder, RetrieverConfig(index_check_ttl_s=3600))
    for _ in range(3):
        await cached.retrieve("honey")
    assert good.info_calls == 1
    uncached = HybridRetriever(good, embedder, RetrieverConfig(index_check_ttl_s=0))
    for _ in range(3):
        await uncached.retrieve("honey")
    assert good.info_calls == 4


async def test_rrf_weights_are_applied_and_reported(
    memory_store: InMemoryStore, embedder: HashEmbedder
) -> None:
    weights = {VECTOR: 0.5, FTS: 1.0}
    retriever = HybridRetriever(memory_store, embedder, RetrieverConfig(rrf_weights=weights))
    result = await retriever.retrieve("rechargeable lithium-ion battery pack", top_k=5)
    assert result.rrf_weights == weights
    for c in result.candidates:
        expected = sum(weights[name] / (60 + rank) for name, rank in c.ranks.items())
        assert c.fused_score == pytest.approx(expected)
    assert set(result.candidates[0].scores) == {VECTOR, FTS}
