"""Evaluation items: hand-labelled JSONL, plus items derived from imported rulings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from landedcost.hts.codes import CodeLevel, HtsCode
from landedcost.rulings.models import Ruling


@dataclass(frozen=True, slots=True)
class EvalItem:
    id: str
    description: str
    gold_headings: tuple[str, ...]
    gold_subheadings: tuple[str, ...] = ()
    expect_abstain: bool = False
    source: str = ""
    exclude_rulings: tuple[str, ...] = ()
    """Ruling numbers hidden from retrieval for this item (leave-one-out)."""


def _codes(values: object, level: CodeLevel) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError("gold codes must be a list")
    out = []
    for value in values:
        code = HtsCode.parse(str(value))
        if code.level is not level:
            raise ValueError(f"{value!r} is not a {level.value}")
        out.append(code.dotted)
    return tuple(out)


def load_eval_items(path: Path) -> list[EvalItem]:
    """Load a JSONL file of ``{id, description, gold_headings, gold_subheadings?, ...}``."""
    items: list[EvalItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                item = EvalItem(
                    id=str(record["id"]),
                    description=str(record["description"]).strip(),
                    gold_headings=_codes(record.get("gold_headings", []), CodeLevel.HEADING),
                    gold_subheadings=_codes(
                        record.get("gold_subheadings", []), CodeLevel.SUBHEADING
                    ),
                    expect_abstain=bool(record.get("expect_abstain", False)),
                    source=str(record.get("source", "")),
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if item.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {item.id!r}")
            if not item.expect_abstain and not item.gold_headings:
                raise ValueError(f"{path}:{lineno}: answerable item needs gold_headings")
            for sub in item.gold_subheadings:
                if sub[:4] not in item.gold_headings:
                    raise ValueError(f"{path}:{lineno}: {sub} is not under a gold heading")
            seen.add(item.id)
            items.append(item)
    return items


def items_from_rulings(rulings: list[Ruling]) -> list[EvalItem]:
    """One item per ruling: its product description should retrieve its HTS code(s).

    The ruling itself is excluded from retrieval for its own item, otherwise the ruling
    would trivially vote for its own answer.
    """
    return [
        EvalItem(
            id=f"ruling:{r.ruling_number}",
            description=r.product_description,
            gold_headings=r.headings,
            gold_subheadings=r.subheadings,
            source=f"ruling {r.ruling_number}",
            exclude_rulings=(r.ruling_number,),
        )
        for r in rulings
    ]
