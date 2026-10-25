"""Turn raw USITC export rows into :class:`HtsLine` records.

The USITC export is a flat list in document order. Hierarchy is encoded only by the
``indent`` column, and many context rows (``superior`` rows such as "Other:" or
"Men's or boys':") carry no code at all. We rebuild the tree with a stack keyed by indent.

Duty rates are printed on 8-digit tariff lines; 10-digit statistical suffixes usually leave
the rate columns blank and inherit them. We copy the nearest ancestor's rates onto such
lines and record where they came from (``rate_source_code``).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from landedcost.hts.codes import CodeLevel, HtsCode, InvalidHtsCodeError
from landedcost.hts.models import HtsLine

DEFAULT_EXCLUDED_CHAPTERS: frozenset[str] = frozenset({"98", "99"})
"""Chapter 98 holds special classification provisions (returned goods, repairs, ...) and
chapter 99 holds temporary legislation and additional duties (Section 301/232, ...). Neither
classifies a product on its own, so they are excluded from the retrieval index by default."""

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class _Frame:
    indent: int
    description: str
    code: str | None
    general: str
    special: str
    other: str


@dataclass(slots=True)
class ParseResult:
    """Parsed lines plus counts of rows that were skipped and why."""

    lines: list[HtsLine] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)


def clean_text(value: object) -> str:
    """Collapse whitespace (``\\s`` includes NBSP, U+00A0) and strip a trailing colon."""
    text = _WHITESPACE.sub(" ", str(value or "")).strip()
    return text[:-1].rstrip() if text.endswith(":") else text


def _rate(value: object) -> str:
    return _WHITESPACE.sub(" ", str(value or "")).strip()


def _indent(value: object) -> int:
    try:
        return int(str(value or "0").strip() or "0")
    except ValueError:
        return 0


def _footnotes(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    notes: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            text = clean_text(item.get("value"))
            if text:
                notes.append(text)
    return tuple(notes)


def _units(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(u for u in (_rate(x) for x in value) if u)


def _inherited_rates(stack: list[_Frame]) -> tuple[str, str, str, str | None]:
    for frame in reversed(stack):
        if frame.code is not None and frame.general:
            return frame.general, frame.special, frame.other, frame.code
    return "", "", "", None


def parse_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    excluded_chapters: frozenset[str] = DEFAULT_EXCLUDED_CHAPTERS,
) -> ParseResult:
    """Parse USITC export rows (in document order) into :class:`HtsLine` records."""
    result = ParseResult()
    stack: list[_Frame] = []

    for row in rows:
        indent = _indent(row.get("indent"))
        while stack and stack[-1].indent >= indent:
            stack.pop()

        description = clean_text(row.get("description"))
        raw_code = str(row.get("htsno") or "").strip()
        general, special, other = (_rate(row.get(k)) for k in ("general", "special", "other"))

        code: HtsCode | None = None
        if raw_code:
            try:
                code = HtsCode.parse(raw_code)
            except InvalidHtsCodeError:
                result.skipped["invalid_code"] += 1

        if code is not None:
            if code.chapter in excluded_chapters:
                result.skipped[f"excluded_chapter_{code.chapter}"] += 1
            else:
                rate_source: str | None = code.dotted if general else None
                if not general and code.level is CodeLevel.STATISTICAL:
                    general, special, other, rate_source = _inherited_rates(stack)
                result.lines.append(
                    HtsLine(
                        code=code.dotted,
                        level=code.level,
                        description=description,
                        path=tuple(f.description for f in stack if f.description),
                        indent=indent,
                        units=_units(row.get("units")),
                        general_rate=general,
                        special_rate=special,
                        other_rate=other,
                        rate_source_code=rate_source,
                        footnotes=_footnotes(row.get("footnotes")),
                    )
                )
        elif not raw_code:
            result.skipped["context_row"] += 1

        stack.append(
            _Frame(
                indent=indent,
                description=description,
                code=code.dotted if code else None,
                general=general if code else "",
                special=special if code else "",
                other=other if code else "",
            )
        )

    return result
