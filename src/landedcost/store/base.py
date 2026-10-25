"""Store interface shared by the PostgreSQL and in-memory implementations."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol

from landedcost.embeddings.base import Matrix, Vector
from landedcost.hts.models import HtsLine
from landedcost.rulings.models import Ruling


class EmbedderMismatchError(RuntimeError):
    """The index was built with a different embedder (or dimension) than the one configured."""


class StoreUnavailableError(RuntimeError):
    """The store cannot be reached or its schema is missing (the message says which)."""


@dataclass(frozen=True, slots=True)
class ScoredLine:
    line: HtsLine
    score: float


@dataclass(frozen=True, slots=True)
class ScoredRuling:
    ruling: Ruling
    score: float


@dataclass(frozen=True, slots=True)
class IndexInfo:
    hts_lines: int
    rulings: int
    embedder: str | None
    """Embedder that built the HTS line vectors (``None`` before the first ingest)."""
    embedding_dim: int | None
    hts_version: str | None
    rulings_embedder: str | None = None
    """Embedder that built the ruling vectors (``None`` if unknown or no rulings)."""


class Store(Protocol):
    """Persistence and search for HTS lines and rulings.

    Vector searches take an L2-normalised query vector and score by cosine similarity;
    text searches take raw query text and apply the backend's own keyword ranking.
    """

    async def search_hts_vector(self, embedding: Vector, limit: int) -> list[ScoredLine]: ...

    async def search_hts_text(self, query: str, limit: int) -> list[ScoredLine]: ...

    async def search_rulings_vector(
        self, embedding: Vector, limit: int, exclude: Collection[str] = ()
    ) -> list[ScoredRuling]: ...

    async def search_rulings_text(
        self, query: str, limit: int, exclude: Collection[str] = ()
    ) -> list[ScoredRuling]: ...

    async def get_hts_lines(self, codes: Sequence[str]) -> dict[str, HtsLine]:
        """Exact lookup by canonical dotted code; missing codes are absent from the result."""
        ...

    async def find_hts_by_prefix(self, prefix: str, limit: int) -> list[HtsLine]:
        """Lines whose dotted code starts with ``prefix``, in code order."""
        ...

    async def list_rulings(self) -> list[Ruling]: ...

    async def replace_hts_lines(
        self, lines: Sequence[HtsLine], embeddings: Matrix, *, hts_version: str, embedder: str
    ) -> int:
        """Atomically replace the whole HTS index."""
        ...

    async def upsert_rulings(
        self, rulings: Sequence[Ruling], embeddings: Matrix, *, embedder: str
    ) -> int:
        """Insert or update rulings; refuses vectors from an embedder other than the index's."""
        ...

    async def replace_rulings(
        self, rulings: Sequence[Ruling], embeddings: Matrix, *, embedder: str
    ) -> int:
        """Atomically replace every ruling (used to re-embed them after an HTS re-ingest)."""
        ...

    async def index_info(self) -> IndexInfo: ...

    async def close(self) -> None: ...


def check_embedder(
    info: IndexInfo, embedder_name: str, dim: int, *, include_rulings: bool = True
) -> None:
    """Fail fast when querying (or extending) an index built by another embedder.

    HTS lines and rulings are checked separately: re-ingesting the HTS with a new embedder
    must not leave rulings embedded by the old one voting in the fusion.
    """
    if info.embedder is not None and (
        info.embedder != embedder_name or info.embedding_dim not in (None, dim)
    ):
        raise EmbedderMismatchError(
            f"index was built with embedder {info.embedder!r} (dim={info.embedding_dim}) but "
            f"{embedder_name!r} (dim={dim}) is configured; re-run `landedcost ingest-hts`"
        )
    if include_rulings and info.rulings and info.rulings_embedder not in (None, embedder_name):
        raise EmbedderMismatchError(
            f"rulings were embedded with {info.rulings_embedder!r} but {embedder_name!r} is "
            "configured; re-run `landedcost ingest-hts` (it re-embeds stored rulings)"
        )
