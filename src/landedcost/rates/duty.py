"""Compute MFN duty from a parsed rate, when that is possible without broker judgement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from landedcost.rates.parser import ParsedRate, RateKind

BROKER_REVIEW = "requires broker review"
_CENTS = Decimal("0.01")


def to_cents(value: Decimal) -> Decimal:
    return value.quantize(_CENTS, rounding=ROUND_HALF_UP)


class DutyStatus(StrEnum):
    COMPUTED = "computed"
    REQUIRES_QUANTITY = "requires_quantity"
    REQUIRES_BROKER_REVIEW = "requires_broker_review"


@dataclass(frozen=True, slots=True)
class DutyEstimate:
    status: DutyStatus
    amount_usd: Decimal | None
    explanation: str
    missing_units: tuple[str, ...] = ()


def estimate_duty(
    rate: ParsedRate,
    customs_value_usd: Decimal,
    quantities: Mapping[str, Decimal] | None = None,
) -> DutyEstimate:
    """Estimate MFN duty on ``customs_value_usd``.

    ``quantities`` maps canonical units (``kg``, ``liter``, ``each``, ``doz``, ``pair``, ...)
    to amounts and is only needed for specific or compound rates.
    """
    quantities = quantities or {}
    if rate.kind is RateKind.FREE:
        return DutyEstimate(DutyStatus.COMPUTED, Decimal("0.00"), "General rate is Free.")
    if rate.kind in (RateKind.NONE, RateKind.UNKNOWN):
        reason = rate.note or "rate could not be parsed"
        if rate.raw.strip() and rate.raw.strip() not in reason:
            reason = f"{reason} (rate {rate.raw!r})"
        return DutyEstimate(DutyStatus.REQUIRES_BROKER_REVIEW, None, f"{BROKER_REVIEW}: {reason}")

    missing = tuple(c.unit for c in rate.specific if c.unit not in quantities)
    if missing:
        return DutyEstimate(
            DutyStatus.REQUIRES_QUANTITY,
            None,
            f"Rate {rate.raw!r} is specific; supply quantity in: {', '.join(missing)}.",
            missing_units=missing,
        )

    total = Decimal("0")
    parts: list[str] = []
    if rate.ad_valorem_pct is not None:
        av = customs_value_usd * rate.ad_valorem_pct / Decimal(100)
        total += av
        parts.append(f"{rate.ad_valorem_pct}% x ${customs_value_usd} = ${to_cents(av)}")
    for component in rate.specific:
        qty = quantities[component.unit]
        amount = component.amount_usd * qty
        total += amount
        parts.append(f"${component.amount_usd}/{component.unit} x {qty} = ${to_cents(amount)}")
    return DutyEstimate(DutyStatus.COMPUTED, to_cents(total), "; ".join(parts))
