import asyncio

import numpy as np
import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from landedcost.embeddings.hashing import HashEmbedder
from landedcost.hts.models import HtsLine
from landedcost.ingest import import_rulings_file, ingest_hts
from landedcost.retrieval.hybrid import RULINGS, HybridRetriever, RetrievalMode
from landedcost.store.base import EmbedderMismatchError, StoreUnavailableError
from landedcost.store.migrate import apply_migrations
from landedcost.store.postgres import _VECTOR_SQL, PostgresStore
from tests.conftest import SAMPLE_SNAPSHOT, SYNTHETIC_RULINGS

pytestmark = pytest.mark.integration


async def test_ingest_loads_lines_and_metadata(
    pg_store: PostgresStore, sample_lines: list[HtsLine]
) -> None:
    info = await pg_store.index_info()
    assert info.hts_lines == len(sample_lines)
    assert (info.embedder, info.embedding_dim, info.hts_version) == (
        "hash-v2-c35",
        384,
        "hts-2026-09-04",
    )
    line = (await pg_store.get_hts_lines(["8507.60.00.90"]))["8507.60.00.90"]
    assert line == next(x for x in sample_lines if x.code == "8507.60.00.90")


async def test_tsvector_column_weights_own_description_higher(
    pg_store: PostgresStore, pg_schema: str
) -> None:
    async with await psycopg.AsyncConnection.connect(pg_schema) as conn:
        cur = await conn.execute(
            "SELECT code FROM hts_lines "
            "WHERE search_tsv @@ to_tsquery('english', 'lithium:A') ORDER BY code"
        )
        own = [r[0] for r in await cur.fetchall()]
        cur = await conn.execute(
            "SELECT count(*) FROM hts_lines WHERE search_tsv @@ to_tsquery('english', 'lithium:B')"
        )
        inherited = (await cur.fetchone() or [0])[0]
    assert own == ["8507.60.00"]  # only the line whose own text says "Lithium-ion"
    assert inherited >= 3  # its statistical children carry it as context


async def test_queries_use_hnsw_and_gin_indexes(
    pg_store: PostgresStore, pg_schema: str, embedder: HashEmbedder
) -> None:
    vector = embedder.embed_one("natural honey")
    async with await psycopg.AsyncConnection.connect(pg_schema) as conn, conn.transaction():
        await conn.execute("SET LOCAL enable_seqscan = off")
        literal = "[" + ",".join(f"{x:.6f}" for x in vector) + "]"
        explain = "EXPLAIN " + _VECTOR_SQL.replace("%(q)s", "%(q)s::vector")
        cur = await conn.execute(explain.encode(), {"q": literal, "fetch": 10, "limit": 5})
        vector_plan = "\n".join(r[0] for r in await cur.fetchall())
        cur = await conn.execute(
            "EXPLAIN SELECT code FROM hts_lines "
            "WHERE search_tsv @@ websearch_to_tsquery('english', 'honey or jar')"
        )
        fts_plan = "\n".join(r[0] for r in await cur.fetchall())
    assert "hts_lines_embedding_hnsw" in vector_plan
    assert "hts_lines_search_tsv_gin" in fts_plan


async def test_vector_and_bm25_search(pg_store: PostgresStore, embedder: HashEmbedder) -> None:
    vec = await pg_store.search_hts_vector(await embedder.embed_query("natural honey"), 5)
    assert vec[0].line.heading == "0409"
    assert vec[0].score > vec[-1].score
    text = await pg_store.search_hts_text("natural honey in glass jars", 10)
    assert text[0].line.heading == "0409"  # rare "honey" outweighs common "glass"/"jar"
    assert any(h.line.heading == "7010" for h in text)
    assert await pg_store.search_hts_text("the 750 ml", 5) == []


async def test_ts_rank_cd_variant(pg_schema: str) -> None:
    store = await PostgresStore.connect(pg_schema, fts_ranking="ts_rank_cd", max_size=2)
    try:
        await ingest_hts(store, HashEmbedder(), SAMPLE_SNAPSHOT, chapters=["04", "70"])
        hits = await store.search_hts_text("natural honey in glass jars", 50)
        assert {h.line.heading for h in hits} == {"0409", "7010"}
        assert all(h.line.chapter in {"04", "70"} for h in hits)
    finally:
        await store.close()


async def test_bm25_statistics_are_refreshed(pg_store: PostgresStore, pg_schema: str) -> None:
    async with await psycopg.AsyncConnection.connect(pg_schema) as conn:
        cur = await conn.execute("SELECT ndoc FROM hts_lexeme_stats WHERE lexeme = 'honey'")
        honey = (await cur.fetchone() or [0])[0]
        cur = await conn.execute(
            "SELECT key, value FROM index_meta WHERE key LIKE 'fts_%' ORDER BY key"
        )
        meta: dict[str, str] = dict(await cur.fetchall())
    assert honey >= 6
    assert int(meta["fts_doc_count"]) == (await pg_store.index_info()).hts_lines
    assert float(meta["fts_avg_doc_len"]) > 1


async def test_prefix_lookup_escapes_like_wildcards(pg_store: PostgresStore) -> None:
    lines = await pg_store.find_hts_by_prefix("8507.6", 3)
    assert [line.code for line in lines] == ["8507.60.00", "8507.60.00.10", "8507.60.00.30"]
    assert await pg_store.find_hts_by_prefix("%", 3) == []
    assert await pg_store.get_hts_lines([]) == {}


async def _hnsw_index_count(url: str) -> int:
    async with await psycopg.AsyncConnection.connect(url) as conn:
        cur = await conn.execute(
            "SELECT count(*) FROM pg_indexes WHERE indexname = 'hts_lines_embedding_hnsw'"
        )
        return int((await cur.fetchone() or [0])[0])


async def test_reingest_replaces_rows_and_keeps_the_hnsw_index(
    pg_store: PostgresStore, pg_schema: str, embedder: HashEmbedder
) -> None:
    before = (await pg_store.index_info()).hts_lines
    report = await ingest_hts(pg_store, embedder, SAMPLE_SNAPSHOT)
    assert report.lines == before == (await pg_store.index_info()).hts_lines
    assert await _hnsw_index_count(pg_schema) == 1


async def test_failed_reingest_rolls_back_to_the_old_index(
    pg_store: PostgresStore, pg_schema: str, sample_lines: list[HtsLine]
) -> None:
    before = await pg_store.index_info()
    wrong_dim = np.ones((len(sample_lines), 383), dtype=np.float32)  # column is vector(384)
    with pytest.raises(psycopg.Error):
        await pg_store.replace_hts_lines(
            sample_lines, wrong_dim, hts_version="broken", embedder="broken"
        )
    after = await pg_store.index_info()
    assert after == before  # rows, embedder and version all untouched
    assert await _hnsw_index_count(pg_schema) == 1  # the DROP INDEX was rolled back too
    assert (await pg_store.search_hts_text("natural honey", 3))[0].line.heading == "0409"


async def test_drop_existing_also_drops_bm25_statistics(
    pg_store: PostgresStore, pg_schema: str
) -> None:
    await apply_migrations(pg_schema, embedding_dim=384, drop_existing=True)
    async with await psycopg.AsyncConnection.connect(pg_schema) as conn:
        cur = await conn.execute("SELECT count(*) FROM hts_lexeme_stats")
        assert (await cur.fetchone() or [1])[0] == 0


async def test_missing_schema_is_reported_as_unavailable(pg_schema: str) -> None:
    async with await psycopg.AsyncConnection.connect(pg_schema, autocommit=True) as conn:
        await conn.execute(
            "DROP TABLE hts_lines, rulings, hts_lexeme_stats, index_meta, schema_migrations"
        )
    store = await PostgresStore.connect(pg_schema, max_size=1)
    try:
        with pytest.raises(StoreUnavailableError, match="landedcost migrate"):
            await store.index_info()
    finally:
        await store.close()


async def test_bad_credentials_fail_fast_with_the_server_message(pg_url: str) -> None:
    wrong = make_conninfo(pg_url, password="definitely-wrong")
    assert conninfo_to_dict(wrong)["password"] == "definitely-wrong"
    with pytest.raises(StoreUnavailableError, match="cannot connect to PostgreSQL"):
        await PostgresStore.connect(wrong, open_timeout_s=1)


async def test_hts_reingest_with_a_new_embedder_reembeds_rulings(
    pg_store: PostgresStore, embedder: HashEmbedder
) -> None:
    await import_rulings_file(pg_store, embedder, SYNTHETIC_RULINGS)
    assert (await pg_store.index_info()).rulings_embedder == embedder.name
    other = HashEmbedder(char_ngram_range=(2, 4))  # same dimension, different vectors
    report = await ingest_hts(pg_store, other, SAMPLE_SNAPSHOT)
    assert report.rulings_reembedded == 4
    info = await pg_store.index_info()
    assert (info.embedder, info.rulings_embedder, info.rulings) == (other.name, other.name, 4)
    result = await HybridRetriever(pg_store, other).retrieve("lithium battery for kick scooters")
    assert RULINGS in result.retrievers_run
    with pytest.raises(EmbedderMismatchError):
        await HybridRetriever(pg_store, embedder).retrieve("honey")


async def test_rulings_round_trip(pg_store: PostgresStore, embedder: HashEmbedder) -> None:
    report = await import_rulings_file(pg_store, embedder, SYNTHETIC_RULINGS)
    assert report.imported == 4
    again = await import_rulings_file(pg_store, embedder, SYNTHETIC_RULINGS)  # upsert
    assert again.imported == 4
    assert (await pg_store.index_info()).rulings == 4
    text = await pg_store.search_rulings_text("lithium battery scooter", 5)
    assert text[0].ruling.ruling_number == "TEST-0001"
    excluded = await pg_store.search_rulings_text("lithium battery scooter", 5, {"TEST-0001"})
    assert "TEST-0001" not in {h.ruling.ruling_number for h in excluded}
    vec = await pg_store.search_rulings_vector(await embedder.embed_query("clover honey"), 2)
    assert vec[0].ruling.ruling_number == "TEST-0002"
    stored = {r.ruling_number: r for r in await pg_store.list_rulings()}
    assert stored["TEST-0004"].hts_codes == ("6109.10.00.12", "6109.10.00.04")
    with pytest.raises(EmbedderMismatchError):
        await import_rulings_file(
            pg_store, HashEmbedder(dim=384, char_ngram_range=(2, 4)), SYNTHETIC_RULINGS
        )


async def test_hybrid_retrieval_with_ruling_votes(
    pg_store: PostgresStore, embedder: HashEmbedder
) -> None:
    await import_rulings_file(pg_store, embedder, SYNTHETIC_RULINGS)
    retriever = HybridRetriever(pg_store, embedder)
    result = await retriever.retrieve("lithium-ion battery module for electric kick scooters")
    assert RULINGS in result.retrievers_run
    assert result.candidates[0].line.code.startswith("8507.60")
    cited = {c.line.code: c.supporting_rulings for c in result.candidates}
    assert "TEST-0001" in cited["8507.60.00.90"]
    loo = await retriever.retrieve(
        "lithium-ion battery module for electric kick scooters", exclude_rulings={"TEST-0001"}
    )
    assert "TEST-0001" not in {r.ruling.ruling_number for r in loo.rulings}


async def test_concurrent_queries_share_the_pool(
    pg_store: PostgresStore, embedder: HashEmbedder
) -> None:
    retriever = HybridRetriever(pg_store, embedder)
    queries = [
        "natural honey",
        "cotton t-shirt",
        "olive oil",
        "glass bottles",
        "lithium battery",
    ] * 4
    results = await asyncio.gather(
        *(retriever.retrieve(q, mode=RetrievalMode.HYBRID) for q in queries)
    )
    assert all(r.candidates for r in results)
