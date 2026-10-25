import numpy as np
import pytest

from landedcost.embeddings.base import l2_normalize
from landedcost.embeddings.hashing import HashEmbedder
from landedcost.text import bm25_terms, fts_query, stem, words


def test_words_drop_stopwords_numbers_and_units() -> None:
    assert words("Extra virgin olive oil in 1 litre glass bottles, 750 ml") == [
        "extra",
        "virgin",
        "olive",
        "oil",
        "litre",
        "glass",
        "bottles",
    ]
    assert words("36V 10Ah 16GB of the other") == []


@pytest.mark.parametrize(
    ("token", "stemmed"),
    [
        ("batteries", "battery"),
        ("boxes", "box"),
        ("shoes", "shoe"),
        ("glass", "glass"),
        ("bus", "bus"),
    ],
)
def test_stem(token: str, stemmed: str) -> None:
    assert stem(token) == stemmed


def test_bm25_terms_are_stemmed() -> None:
    assert bm25_terms("Lithium batteries") == ["lithium", "battery"]


def test_fts_query_ors_terms_and_strips_operators() -> None:
    assert (
        fts_query("rechargeable lithium-ion battery") == "rechargeable or lithium or ion or battery"
    )
    assert fts_query("men's or boys' -shirts") == "men or boys or shirts"
    assert fts_query("the of 123") == ""


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    a, b = HashEmbedder(), HashEmbedder()
    va = a.embed_one("Lithium-ion batteries")
    vb = b.embed_one("Lithium-ion batteries")
    assert va.dtype == np.float32
    assert va.shape == (384,)
    np.testing.assert_array_equal(va, vb)
    assert float(np.linalg.norm(va)) == pytest.approx(1.0, abs=1e-5)


def test_hash_embedder_similarity_is_lexical() -> None:
    e = HashEmbedder()
    battery = e.embed_one("rechargeable lithium battery pack")
    batteries = e.embed_one("Lithium-ion batteries")
    honey = e.embed_one("Natural honey")
    assert float(battery @ batteries) > float(battery @ honey) + 0.2


def test_hash_embedder_ignores_numbers_and_handles_empty_text() -> None:
    e = HashEmbedder(dim=64)
    np.testing.assert_array_equal(e.embed_one("750 ml bottle"), e.embed_one("bottle"))
    assert not e.embed_one("123 456").any()
    assert e.embed_many([]).shape == (0, 64)
    assert e.embed_many(["a b", "glass jar"]).shape == (2, 64)


async def test_async_interface_matches_sync() -> None:
    e = HashEmbedder()
    np.testing.assert_array_equal(await e.embed_query("honey"), e.embed_one("honey"))
    assert (await e.embed_documents(["honey", "glass"])).shape == (2, e.dim)
    assert e.name == "hash-v2-c35"


@pytest.mark.parametrize("kwargs", [{"dim": 0}, {"char_ngram_range": (4, 2)}])
def test_hash_embedder_validates_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        HashEmbedder(**kwargs)  # type: ignore[arg-type]


def test_l2_normalize_keeps_zero_rows() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])
