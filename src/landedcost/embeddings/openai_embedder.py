"""Embeddings via the official ``openai`` SDK (OpenAI or Azure OpenAI)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from openai import AsyncOpenAI

from landedcost.embeddings.base import Matrix, Vector, l2_normalize


class OpenAIEmbedder:
    """Calls ``embeddings.create`` in batches.

    Works for both ``AsyncOpenAI`` and ``AsyncAzureOpenAI`` clients; for Azure, ``model`` is
    the *deployment* name. ``text-embedding-3-*`` models accept a ``dimensions`` argument,
    which lets the same ``vector(dim)`` schema serve the hash baseline and OpenAI vectors.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        dim: int,
        provider: str = "openai",
        batch_size: int = 256,
    ) -> None:
        self._client = client
        self._model = model
        self._dim = dim
        self._provider = provider
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return f"{self._provider}:{self._model}:{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    async def aclose(self) -> None:
        """Close the SDK client's connection pool."""
        await self._client.close()

    async def embed_documents(self, texts: Sequence[str]) -> Matrix:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = await self._client.embeddings.create(
                model=self._model, input=batch, dimensions=self._dim
            )
            rows.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        if not rows:
            return np.zeros((0, self._dim), dtype=np.float32)
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.shape[1] != self._dim:
            raise ValueError(f"expected {self._dim}-dim embeddings, got {matrix.shape[1]}")
        return l2_normalize(matrix)

    async def embed_query(self, text: str) -> Vector:
        vector: Vector = (await self.embed_documents([text]))[0]
        return vector
