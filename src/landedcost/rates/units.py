"""Canonical quantity units shared by rate parsing, HTS reporting units and user input."""

from __future__ import annotations

_ALIASES: dict[str, str] = {
    # mass
    "kg": "kg",
    "kgs": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "t": "t",
    "ton": "t",
    "tons": "t",
    "tonne": "t",
    "tonnes": "t",
    "metric_ton": "t",
    # volume
    "liter": "liter",
    "liters": "liter",
    "litre": "liter",
    "litres": "liter",
    "l": "liter",
    "pf.liter": "proof_liter",
    "pf. liter": "proof_liter",
    "proof_liter": "proof_liter",
    "bbl": "bbl",
    "barrel": "bbl",
    "barrels": "bbl",
    "m3": "m3",
    "m³": "m3",
    # area
    "m2": "m2",
    "m²": "m2",
    # count
    "each": "each",
    "no.": "each",
    "no": "each",
    "pcs": "each",
    "pieces": "each",
    "units": "each",
    "article": "each",
    "head": "head",
    "doz.": "doz",
    "doz": "doz",
    "dozen": "doz",
    "pr.": "pair",
    "pr": "pair",
    "prs.": "pair",
    "prs": "pair",
    "pair": "pair",
    "pairs": "pair",
    "doz. prs.": "doz_pair",
    "gross": "gross",
    "thousand": "thousand",
}


def canonical_unit(raw: str) -> str | None:
    """Map a unit spelling (``"No."``, ``"liters"``, ``"prs."``) to its canonical name."""
    key = " ".join(raw.strip().lower().split())
    return _ALIASES.get(key)
