"""Embedder interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]


class Embedder(Protocol):
    """Turns text into L2-normalised float32 vectors of a fixed dimension."""

    @property
    def name(self) -> str:
        """Identifier stored with the index, e.g. ``hash-v1`` or ``openai:text-embedding-3-small``.

        Vectors from different embedders are not comparable; the store refuses to query an
        index built with a different embedder.
        """
        ...

    @property
    def dim(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    async def embed_query(self, text: str) -> Vector: ...


def l2_normalize(matrix: NDArray[np.floating]) -> Matrix:
    """Row-wise L2 normalisation; all-zero rows stay zero."""
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(m / norms, dtype=np.float32)
