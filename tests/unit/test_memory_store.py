import numpy as np
import pytest

from landedcost.embeddings.hashing import HashEmbedder
from landedcost.hts.models import HtsLine
from landedcost.ingest import import_rulings_file, ingest_hts
from landedcost.store.base import EmbedderMismatchError, IndexInfo, check_embedder
from landedcost.store.memory import InMemoryStore
from tests.conftest import SAMPLE_SNAPSHOT, SYNTHETIC_RULINGS


async def test_vector_and_text_search(memory_store: InMemoryStore, embedder: HashEmbedder) -> None:
    hits = await memory_store.search_hts_vector(await embedder.embed_query("natural honey"), 3)
    assert hits[0].line.heading == "0409"
    assert hits[0].score >= hits[-1].score
    text_hits = await memory_store.search_hts_text("honey in glass jars", 5)
    assert text_hits[0].line.heading == "0409"
    assert any(h.line.heading == "7010" for h in text_hits)


async def test_lookups(memory_store: InMemoryStore) -> None:
    found = await memory_store.get_hts_lines(["8507.60.00.90", "9999.99"])
    assert list(found) == ["8507.60.00.90"]
    prefix = await memory_store.find_hts_by_prefix("8507.60", 10)
    assert [line.code for line in prefix][:2] == ["8507.60.00", "8507.60.00.10"]
    info = await memory_store.index_info()
    assert info.embedder == "hash-v2-c35"
    assert info.embedding_dim == 384
    assert info.hts_version == "hts-2026-09-04"
    assert info.rulings == 0


async def test_rulings_upsert_search_and_exclude(
    memory_store: InMemoryStore, embedder: HashEmbedder
) -> None:
    await import_rulings_file(memory_store, embedder, SYNTHETIC_RULINGS)
    assert (await memory_store.index_info()).rulings == 4
    hits = await memory_store.search_rulings_text("lithium battery scooter", 5)
    assert hits[0].ruling.ruling_number == "TEST-0001"
    excluded = await memory_store.search_rulings_text("lithium battery scooter", 5, {"TEST-0001"})
    assert "TEST-0001" not in {h.ruling.ruling_number for h in excluded}
    vec = await memory_store.search_rulings_vector(
        await embedder.embed_query("clover honey"), 2, exclude={"TEST-0003"}
    )
    assert vec[0].ruling.ruling_number == "TEST-0002"
    assert [r.ruling_number for r in await memory_store.list_rulings()] == [
        "TEST-0001",
        "TEST-0002",
        "TEST-0003",
        "TEST-0004",
    ]


async def test_rulings_from_a_different_embedder_are_rejected(memory_store: InMemoryStore) -> None:
    other = HashEmbedder(dim=64, char_ngram_range=(2, 4))
    with pytest.raises(EmbedderMismatchError):
        await import_rulings_file(memory_store, other, SYNTHETIC_RULINGS)


async def test_length_mismatch_is_rejected(sample_lines: list[HtsLine]) -> None:
    store = InMemoryStore()
    with pytest.raises(ValueError, match="differ in length"):
        await store.replace_hts_lines(
            sample_lines[:2], np.zeros((1, 4), dtype=np.float32), hts_version="x", embedder="e"
        )


async def test_empty_store_returns_nothing() -> None:
    store = InMemoryStore()
    assert await store.search_hts_vector(np.ones(4, dtype=np.float32), 5) == []
    assert await store.search_hts_text("honey", 5) == []


def test_check_embedder() -> None:
    check_embedder(IndexInfo(0, 0, None, None, None), "anything", 10)
    check_embedder(IndexInfo(1, 0, "hash-v2-c35", 384, "v"), "hash-v2-c35", 384)
    with pytest.raises(EmbedderMismatchError, match="re-run"):
        check_embedder(IndexInfo(1, 0, "hash-v2-c35", 384, "v"), "openai:x:384", 384)
    with pytest.raises(EmbedderMismatchError):
        check_embedder(IndexInfo(1, 0, "hash-v2-c35", 384, "v"), "hash-v2-c35", 128)


def test_check_embedder_covers_rulings_separately() -> None:
    hts_ok = IndexInfo(1, 4, "hash-v2-c35", 384, "v", rulings_embedder="hash-v2-c24")
    with pytest.raises(EmbedderMismatchError, match="rulings were embedded"):
        check_embedder(hts_ok, "hash-v2-c35", 384)
    check_embedder(hts_ok, "hash-v2-c35", 384, include_rulings=False)
    check_embedder(IndexInfo(1, 0, "hash-v2-c35", 384, "v", "other"), "hash-v2-c35", 384)
    check_embedder(IndexInfo(1, 4, "hash-v2-c35", 384, "v", None), "hash-v2-c35", 384)


async def test_rulings_imported_before_the_hts_record_their_embedder(
    embedder: HashEmbedder,
) -> None:
    store = InMemoryStore()
    await import_rulings_file(store, embedder, SYNTHETIC_RULINGS)
    info = await store.index_info()
    assert (info.embedder, info.rulings_embedder) == (None, embedder.name)
    with pytest.raises(EmbedderMismatchError, match="rulings were embedded"):
        await import_rulings_file(store, HashEmbedder(char_ngram_range=(2, 4)), SYNTHETIC_RULINGS)


async def test_hts_reingest_with_a_new_embedder_reembeds_stored_rulings(
    store_with_rulings: InMemoryStore,
) -> None:
    other = HashEmbedder(char_ngram_range=(2, 4))  # same dim, different vectors
    report = await ingest_hts(store_with_rulings, other, SAMPLE_SNAPSHOT)
    assert report.rulings_reembedded == 4
    info = await store_with_rulings.index_info()
    assert (info.embedder, info.rulings_embedder, info.rulings) == (other.name, other.name, 4)
    again = await ingest_hts(store_with_rulings, other, SAMPLE_SNAPSHOT)
    assert again.rulings_reembedded == 0
