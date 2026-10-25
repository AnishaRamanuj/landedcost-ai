"""Local ONNX embeddings via the optional ``fastembed`` extra (``uv sync --extra fastembed``)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import numpy as np

from landedcost.embeddings.base import Matrix, Vector, l2_normalize


class FastEmbedEmbedder:
    """Wraps ``fastembed.TextEmbedding``; the model is downloaded on first use.

    Documents go through ``passage_embed`` and queries through ``query_embed`` so models
    that expect asymmetric instructions or prefixes (BGE, E5, ...) get them as fastembed
    defines them for each model.
    """

    def __init__(self, model_name: str, dim: int) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "EMBEDDING_PROVIDER=fastembed requires the optional extra: "
                "uv sync --extra fastembed"
            ) from exc
        self._model: Any = TextEmbedding(model_name=model_name)
        self._model_name = model_name
        self._dim = dim

    @property
    def name(self) -> str:
        return f"fastembed:{self._model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_sync(self, texts: Sequence[str], *, query: bool) -> Matrix:
        embed = self._model.query_embed if query else self._model.passage_embed
        rows = [np.asarray(v, dtype=np.float32) for v in embed(list(texts))]
        if not rows:
            return np.zeros((0, self._dim), dtype=np.float32)
        matrix = np.vstack(rows)
        if matrix.shape[1] != self._dim:
            raise ValueError(
                f"{self._model_name} produces {matrix.shape[1]}-dim vectors but EMBEDDING_DIM="
                f"{self._dim}"
            )
        return l2_normalize(matrix)

    async def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return await asyncio.to_thread(self._embed_sync, texts, query=False)

    async def embed_query(self, text: str) -> Vector:
        vector: Vector = (await asyncio.to_thread(self._embed_sync, [text], query=True))[0]
        return vector
