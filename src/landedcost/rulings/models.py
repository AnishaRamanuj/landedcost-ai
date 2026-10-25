"""Ruling records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from landedcost.hts.codes import HtsCode


@dataclass(frozen=True, slots=True)
class Ruling:
    """A CBP classification ruling as imported by ``landedcost import-rulings``."""

    ruling_number: str
    ruling_date: date | None
    product_description: str
    hts_codes: tuple[str, ...]
    url: str
    excerpt: str

    @property
    def chunk_id(self) -> str:
        return f"ruling:{self.ruling_number}"

    @property
    def search_text(self) -> str:
        return f"{self.product_description} {self.excerpt}".strip()

    @property
    def headings(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(HtsCode.parse(c).heading for c in self.hts_codes))

    @property
    def subheadings(self) -> tuple[str, ...]:
        subs = (HtsCode.parse(c).subheading for c in self.hts_codes)
        return tuple(dict.fromkeys(s for s in subs if s))
