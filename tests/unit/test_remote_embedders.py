import sys
import types
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, cast

import numpy as np
import pytest
from openai import AsyncOpenAI

from landedcost.embeddings.openai_embedder import OpenAIEmbedder


class _Item:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _FakeEmbeddings:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model: str, input: list[str], dimensions: int) -> Any:
        self.calls.append({"model": model, "input": input, "dimensions": dimensions})
        data = [_Item(i, [float(i + 1)] * self.dim) for i in range(len(input))]
        return types.SimpleNamespace(data=list(reversed(data)))  # out of order on purpose


def embedder(
    dim: int = 4, returned_dim: int | None = None
) -> tuple[OpenAIEmbedder, _FakeEmbeddings]:
    fake = _FakeEmbeddings(returned_dim or dim)
    client = types.SimpleNamespace(embeddings=fake)
    return (
        OpenAIEmbedder(
            cast(AsyncOpenAI, client), model="text-embedding-3-small", dim=dim, batch_size=2
        ),
        fake,
    )


async def test_openai_embedder_batches_orders_and_normalises() -> None:
    emb, fake = embedder()
    matrix = await emb.embed_documents(["a", "b", "c"])
    assert matrix.shape == (3, 4)
    assert [len(c["input"]) for c in fake.calls] == [2, 1]
    assert fake.calls[0]["dimensions"] == 4
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-6)
    assert emb.name == "openai:text-embedding-3-small:4"
    assert (await emb.embed_query("x")).shape == (4,)
    assert (await emb.embed_documents([])).shape == (0, 4)


async def test_openai_embedder_rejects_wrong_dimension() -> None:
    emb, _ = embedder(dim=4, returned_dim=8)
    with pytest.raises(ValueError, match="expected 4-dim"):
        await emb.embed_documents(["a"])


class _FakeTextEmbedding:
    """Mimics fastembed's asymmetric API: queries and passages embed differently."""

    calls: ClassVar[list[str]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def passage_embed(self, texts: Sequence[str]) -> Iterator[np.ndarray]:
        self.calls.append("passage")
        for i, _ in enumerate(texts):
            yield np.full(3, float(i + 1), dtype=np.float32)

    def query_embed(self, query: Sequence[str]) -> Iterator[np.ndarray]:
        self.calls.append("query")
        for _ in query:
            yield np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> Iterator[np.ndarray]:
        raise AssertionError("use passage_embed / query_embed")


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("fastembed")
    module.TextEmbedding = _FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)


@pytest.mark.usefixtures("fake_fastembed")
async def test_fastembed_wrapper() -> None:
    from landedcost.embeddings.fastembed_embedder import FastEmbedEmbedder

    _FakeTextEmbedding.calls.clear()
    emb = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", dim=3)
    assert emb.name == "fastembed:BAAI/bge-small-en-v1.5"
    matrix = await emb.embed_documents(["a", "b"])
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-6)
    query = await emb.embed_query("a")
    np.testing.assert_allclose(query, [1.0, 0.0, 0.0])
    assert _FakeTextEmbedding.calls == ["passage", "query"]
    assert (await emb.embed_documents([])).shape == (0, 3)
    with pytest.raises(ValueError, match="EMBEDDING_DIM"):
        await FastEmbedEmbedder("m", dim=384).embed_documents(["a"])
