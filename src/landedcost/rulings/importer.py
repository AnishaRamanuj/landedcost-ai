"""Validate and import rulings from CSV or JSONL.

Expected columns / keys: ``ruling_number, ruling_date, product_description, hts_codes,
url, excerpt``. ``hts_codes`` may hold several codes separated by ``;``, ``,``, ``|`` or
whitespace (or be a JSON list in JSONL). Rows are validated individually so one bad row
does not abort an import; problems are reported with their line numbers.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from landedcost.hts.codes import HtsCode, InvalidHtsCodeError
from landedcost.rulings.models import Ruling

REQUIRED_COLUMNS = (
    "ruling_number",
    "ruling_date",
    "product_description",
    "hts_codes",
    "url",
    "excerpt",
)
_CODE_SPLIT = re.compile(r"[;,|\s]+")


@dataclass(slots=True)
class ImportReport:
    rulings: list[Ruling] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duplicates: int = 0


def _parse_codes(value: object) -> tuple[str, ...]:
    raw: Iterable[str]
    if isinstance(value, list):
        raw = (str(v) for v in value)
    else:
        raw = _CODE_SPLIT.split(str(value or ""))
    codes = [HtsCode.parse(c).dotted for c in raw if c.strip()]
    if not codes:
        raise ValueError("hts_codes is empty")
    return tuple(dict.fromkeys(codes))


def _parse_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"ruling_date {text!r} is not ISO-8601 (YYYY-MM-DD)") from exc


def parse_ruling(record: Mapping[str, Any]) -> Ruling:
    """Validate one record. Raises ``ValueError`` with a human-readable reason."""
    number = str(record.get("ruling_number") or "").strip()
    if not number:
        raise ValueError("ruling_number is empty")
    description = " ".join(str(record.get("product_description") or "").split())
    if not description:
        raise ValueError("product_description is empty")
    try:
        codes = _parse_codes(record.get("hts_codes"))
    except InvalidHtsCodeError as exc:
        raise ValueError(str(exc)) from exc
    return Ruling(
        ruling_number=number,
        ruling_date=_parse_date(record.get("ruling_date")),
        product_description=description,
        hts_codes=codes,
        url=str(record.get("url") or "").strip(),
        excerpt=" ".join(str(record.get("excerpt") or "").split()),
    )


def _records(path: Path) -> Iterator[tuple[int, Mapping[str, Any] | str]]:
    """Yield ``(line number, record)``, or ``(line number, error)`` for an unreadable row.

    Files are read as ``utf-8-sig`` so a byte-order mark (as written by Excel's
    "CSV UTF-8" export) does not corrupt the first column name.
    """
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8-sig") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    yield lineno, f"invalid JSON ({exc.msg})"
                    continue
                if not isinstance(record, dict):
                    yield lineno, "expected a JSON object"
                    continue
                yield lineno, record
        return
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{path}: missing columns {missing}; expected {list(REQUIRED_COLUMNS)}"
            )
        for row in reader:
            yield reader.line_num, row


def read_rulings(path: Path) -> ImportReport:
    """Read and validate every row of ``path``; later duplicates of a ruling number win.

    Row-level problems (bad JSON, missing fields, invalid codes or dates) are collected in
    ``errors`` and the remaining rows are still imported. Only a file that cannot be read as
    a whole (e.g. a CSV without the required columns) raises ``ValueError``.
    """
    report = ImportReport()
    by_number: dict[str, Ruling] = {}
    for lineno, record in _records(path):
        if isinstance(record, str):
            report.errors.append(f"line {lineno}: {record}")
            continue
        try:
            ruling = parse_ruling(record)
        except ValueError as exc:
            report.errors.append(f"line {lineno}: {exc}")
            continue
        if ruling.ruling_number in by_number:
            report.duplicates += 1
        by_number[ruling.ruling_number] = ruling
    report.rulings = list(by_number.values())
    return report
