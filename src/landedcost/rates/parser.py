"""Parse HTS "General" (column 1, MFN) duty strings into a structured form.

Roughly 98% of general rates in the 2026 HTS are one of: ``Free``, an ad valorem
percentage (``6.8%``), a specific rate (``2.5¢/kg``, ``$1.03/doz.``, ``4.4¢ each``) or a
compound of those (``3.2¢/kg + 5%``). Everything else (rates "on the case", "on drained
weight", formulas referring to other subheadings, ...) is returned as ``UNKNOWN`` so that
callers surface it for broker review instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from landedcost.rates.units import canonical_unit


class RateKind(StrEnum):
    NONE = "none"
    FREE = "free"
    AD_VALOREM = "ad_valorem"
    SPECIFIC = "specific"
    COMPOUND = "compound"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SpecificComponent:
    """A per-unit duty, normalised to US dollars."""

    amount_usd: Decimal
    unit: str


@dataclass(frozen=True, slots=True)
class ParsedRate:
    raw: str
    kind: RateKind
    ad_valorem_pct: Decimal | None = None
    specific: tuple[SpecificComponent, ...] = ()
    note: str | None = None

    @property
    def required_units(self) -> tuple[str, ...]:
        return tuple(component.unit for component in self.specific)


_NUMBER = r"(\d+(?:\.\d+)?)"
_AD_VALOREM = re.compile(rf"^{_NUMBER}\s*%$")
# "2.5¢/kg", "$1.03/doz.", "4.4¢ each", "1.9¢/ liter", "$3.50/t"
_SPECIFIC = re.compile(rf"^(\$)?\s*{_NUMBER}\s*(¢)?\s*(?:/\s*|\s+)(?P<unit>[a-zA-Z.²³0-9 ]+?)\s*$")
_CENT = Decimal("0.01")


def _parse_component(text: str) -> tuple[Decimal | None, SpecificComponent | None]:
    if m := _AD_VALOREM.match(text):
        return Decimal(m.group(1)), None
    if m := _SPECIFIC.match(text):
        dollars, amount, cents = m.group(1), Decimal(m.group(2)), m.group(3)
        if bool(dollars) == bool(cents):
            raise ValueError(f"ambiguous currency in {text!r}")
        unit = canonical_unit(m.group("unit"))
        if unit is None:
            raise ValueError(f"unsupported unit in {text!r}")
        return None, SpecificComponent(amount * _CENT if cents else amount, unit)
    raise ValueError(f"unsupported rate component {text!r}")


def parse_rate(raw: str) -> ParsedRate:
    """Parse a general-rate string. Never raises: unsupported text becomes ``UNKNOWN``."""
    text = " ".join(raw.split())  # str.split() also splits on NBSP (U+00A0)
    if not text:
        return ParsedRate(raw=raw, kind=RateKind.NONE, note="no rate printed for this line")
    if text.lower() == "free":
        return ParsedRate(raw=raw, kind=RateKind.FREE)

    ad_valorem: Decimal | None = None
    specific: list[SpecificComponent] = []
    try:
        for part in (p.strip() for p in text.split("+")):
            pct, component = _parse_component(part)
            if pct is not None:
                if ad_valorem is not None:
                    raise ValueError("more than one ad valorem component")
                ad_valorem = pct
            elif component is not None:
                specific.append(component)
    except ValueError as exc:
        return ParsedRate(raw=raw, kind=RateKind.UNKNOWN, note=str(exc))

    if specific and ad_valorem is not None:
        kind = RateKind.COMPOUND
    elif specific:
        kind = RateKind.SPECIFIC
    else:
        kind = RateKind.AD_VALOREM
    return ParsedRate(raw=raw, kind=kind, ad_valorem_pct=ad_valorem, specific=tuple(specific))
