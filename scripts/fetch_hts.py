"""Download a fresh HTS snapshot from the official USITC REST export.

Usage:
    uv run python scripts/fetch_hts.py --out data/hts/hts_$(date +%F).json.gz

The output has the same shape as the shipped snapshot
(``{"fetched_at", "source", "license", "rows": [...]}``), so ``landedcost ingest-hts
--snapshot <file>`` can load it directly.

The export endpoint and its ``from``/``to``/``format``/``styles`` parameters are what the
public HTS site uses for its "Export" button; USITC does not version this interface, so the
script validates every response and fails loudly instead of writing a partial snapshot.
It downloads one chapter at a time to keep responses small. The shipped snapshot was not
produced by this script (see data/hts/PROVENANCE.md), so treat its first run against the
live endpoint as a test of the endpoint as well, and diff the result with ``landedcost stats``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

EXPORT_URL = "https://hts.usitc.gov/reststop/exportList"
LICENSE = "US Government work (public domain)"
REQUIRED_KEYS = {"htsno", "indent", "description", "general", "special", "other"}


def fetch_chapter(
    client: httpx.Client, url: str, chapter: int, retries: int = 3
) -> list[dict[str, Any]]:
    params = {
        "from": f"{chapter:02d}00",
        "to": f"{chapter:02d}99",
        "format": "JSON",
        "styles": "false",
    }
    for attempt in range(1, retries + 1):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            rows = response.json()
            break
        except (httpx.HTTPError, ValueError) as exc:
            if attempt == retries:
                raise RuntimeError(f"chapter {chapter:02d}: {exc}") from exc
            time.sleep(2**attempt)
    if not isinstance(rows, list):
        raise RuntimeError(
            f"chapter {chapter:02d}: expected a JSON list, got {type(rows).__name__}"
        )
    for row in rows:
        if not isinstance(row, dict) or not row.keys() >= REQUIRED_KEYS:
            raise RuntimeError(f"chapter {chapter:02d}: unexpected row shape: {str(row)[:200]}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True, help="output .json or .json.gz path")
    parser.add_argument(
        "--url", default=EXPORT_URL, help="export endpoint (override if USITC moves it)"
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout, follow_redirects=True) as client:
        for chapter in range(1, 100):  # chapter 77 is reserved and comes back empty
            chapter_rows = fetch_chapter(client, args.url, chapter)
            print(f"chapter {chapter:02d}: {len(chapter_rows)} rows", file=sys.stderr)
            rows.extend(chapter_rows)
    if len(rows) < 30_000:
        print(
            f"refusing to write a suspiciously small snapshot ({len(rows)} rows)", file=sys.stderr
        )
        return 1

    snapshot = {
        "fetched_at": datetime.now(UTC).date().isoformat(),
        "source": args.url,
        "license": LICENSE,
        "rows": rows,
    }
    payload = json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".gz":
        args.out.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    else:
        args.out.write_bytes(payload)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {len(rows)} rows to {args.out} (sha256 {digest})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
