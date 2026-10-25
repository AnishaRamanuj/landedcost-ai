"""Ingestion use cases shared by the CLI, the API's memory mode and tests."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

from landedcost.embeddings.base import Embedder
from landedcost.hts.loader import load_lines
from landedcost.hts.parser import DEFAULT_EXCLUDED_CHAPTERS
from landedcost.rulings.importer import read_rulings
from landedcost.store.base import Store


@dataclass(frozen=True, slots=True)
class HtsIngestReport:
    hts_version: str
    lines: int
    skipped: Counter[str]
    embed_seconds: float
    load_seconds: float
    rulings_reembedded: int = 0
    """Stored rulings re-embedded because they were built by a different embedder."""


@dataclass(frozen=True, slots=True)
class RulingsImportReport:
    imported: int
    duplicates: int
    errors: list[str] = field(default_factory=list)


async def ingest_hts(
    store: Store,
    embedder: Embedder,
    snapshot_path: Path,
    *,
    chapters: Collection[str] | None = None,
    excluded_chapters: frozenset[str] = DEFAULT_EXCLUDED_CHAPTERS,
) -> HtsIngestReport:
    """Parse, embed and (atomically) load the HTS snapshot. ``chapters`` restricts to a subset.

    Rulings already in the store are re-embedded when they were built by another embedder,
    so a switch of ``EMBEDDING_PROVIDER`` never leaves stale ruling vectors voting in the
    fusion.
    """
    snapshot, parsed = load_lines(snapshot_path, excluded_chapters=excluded_chapters)
    lines = parsed.lines
    if chapters:
        wanted = {c.zfill(2) for c in chapters}
        lines = [line for line in lines if line.chapter in wanted]
    started = time.perf_counter()
    vectors = await embedder.embed_documents([line.embedding_text for line in lines])
    embedded = time.perf_counter()
    count = await store.replace_hts_lines(
        lines, vectors, hts_version=snapshot.version, embedder=embedder.name
    )
    loaded = time.perf_counter()
    reembedded = await reembed_stale_rulings(store, embedder)
    return HtsIngestReport(
        hts_version=snapshot.version,
        lines=count,
        skipped=parsed.skipped,
        embed_seconds=round(embedded - started, 2),
        load_seconds=round(loaded - embedded, 2),
        rulings_reembedded=reembedded,
    )


async def reembed_stale_rulings(store: Store, embedder: Embedder) -> int:
    """Re-embed every stored ruling unless it is known to come from ``embedder``."""
    info = await store.index_info()
    if not info.rulings or info.rulings_embedder == embedder.name:
        return 0
    rulings = await store.list_rulings()
    vectors = await embedder.embed_documents([r.search_text for r in rulings])
    return await store.replace_rulings(rulings, vectors, embedder=embedder.name)


async def import_rulings_file(store: Store, embedder: Embedder, path: Path) -> RulingsImportReport:
    """Validate a CSV/JSONL rulings file and upsert the valid rows."""
    report = read_rulings(path)
    imported = 0
    if report.rulings:
        vectors = await embedder.embed_documents([r.search_text for r in report.rulings])
        imported = await store.upsert_rulings(report.rulings, vectors, embedder=embedder.name)
    return RulingsImportReport(
        imported=imported, duplicates=report.duplicates, errors=report.errors
    )
