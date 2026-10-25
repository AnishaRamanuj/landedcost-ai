"""Load the HTS snapshot shipped in ``data/hts``."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from landedcost.hts.parser import DEFAULT_EXCLUDED_CHAPTERS, ParseResult, parse_rows


@dataclass(frozen=True, slots=True)
class HtsSnapshot:
    """Raw snapshot metadata and rows, as written by ``scripts/fetch_hts.py``."""

    fetched_at: str
    source: str
    license: str
    rows: list[dict[str, Any]]

    @property
    def version(self) -> str:
        return f"hts-{self.fetched_at}"


def load_snapshot(path: Path) -> HtsSnapshot:
    """Read a snapshot from ``.json`` or ``.json.gz``."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"{path} is not an HTS snapshot (expected an object with a 'rows' list)")
    return HtsSnapshot(
        fetched_at=str(payload.get("fetched_at", "unknown")),
        source=str(payload.get("source", "unknown")),
        license=str(payload.get("license", "unknown")),
        rows=payload["rows"],
    )


def load_lines(
    path: Path, *, excluded_chapters: frozenset[str] = DEFAULT_EXCLUDED_CHAPTERS
) -> tuple[HtsSnapshot, ParseResult]:
    """Load a snapshot and parse it in one step."""
    snapshot = load_snapshot(path)
    return snapshot, parse_rows(snapshot.rows, excluded_chapters=excluded_chapters)
