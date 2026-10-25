"""Best-effort helper to pull CBP rulings from CROSS (rulings.cbp.gov) into the import CSV.

UNOFFICIAL, UNDOCUMENTED ENDPOINT - VERIFY BEFORE USE.
CROSS has no published API. This script calls the JSON endpoints that the CROSS web app
itself appears to use (``/api/search`` and ``/api/ruling/{number}``); their paths, parameters
and field names may change without notice, and they may be rate limited. Treat the output as
a draft: open each ruling's URL, confirm the number, date, HTS codes and excerpt by hand, and
only then run ``landedcost import-rulings``. The script never invents values; rows missing a
ruling number, description or HTS code are dropped and reported. Because the endpoints are
undocumented, expect to adjust the field mapping in ``to_row`` after a first test run.

Usage:
    uv run python scripts/fetch_rulings.py --term "lithium-ion battery" --limit 25 \
        --out data/rulings/lithium_draft.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "https://rulings.cbp.gov"
COLUMNS = ["ruling_number", "ruling_date", "product_description", "hts_codes", "url", "excerpt"]
_HTS_LIKE = re.compile(r"\b\d{4}\.\d{2}(?:\.\d{2,4}){0,2}\b")


def _first(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value).strip()
    return ""


def _codes(record: dict[str, Any]) -> list[str]:
    raw = record.get("tariffs") or record.get("hts") or record.get("tariffNumbers") or []
    if isinstance(raw, str):
        raw = [raw]
    codes = [str(c).strip() for c in raw if str(c).strip()]
    if not codes:  # fall back to codes quoted in the subject line
        codes = _HTS_LIKE.findall(_first(record, "subject", "title"))
    return list(dict.fromkeys(codes))


def to_row(record: dict[str, Any]) -> dict[str, str] | None:
    number = _first(record, "rulingNumber", "ruling_number", "number")
    description = _first(record, "subject", "title", "description")
    codes = _codes(record)
    if not (number and description and codes):
        return None
    date = _first(record, "rulingDate", "ruling_date", "date")[:10]
    return {
        "ruling_number": number,
        "ruling_date": date,
        "product_description": description,
        "hts_codes": ";".join(codes),
        "url": f"{BASE_URL}/ruling/{number}",
        "excerpt": _first(record, "text", "summary")[:1000],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--term", required=True, help="CROSS search term")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    params: dict[str, str | int] = {
        "term": args.term,
        "collection": "ALL",
        "pageSize": min(args.limit, 100),
        "page": 1,
        "sortBy": "RELEVANCE",
    }
    with httpx.Client(timeout=30.0, headers={"Accept": "application/json"}) as client:
        response = client.get(f"{args.base_url}/api/search", params=params)
        response.raise_for_status()
        payload = response.json()
    records = payload.get("rulings", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        print("unexpected response shape; the endpoint may have changed", file=sys.stderr)
        return 1

    rows, dropped = [], 0
    for record in records[: args.limit]:
        row = to_row(record) if isinstance(record, dict) else None
        if row is None:
            dropped += 1
        else:
            rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {len(rows)} DRAFT rows to {args.out} ({dropped} dropped for missing fields). "
        "Verify every row against rulings.cbp.gov before importing.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
