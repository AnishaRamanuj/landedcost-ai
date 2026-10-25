from decimal import Decimal
from pathlib import Path

import pytest

from landedcost.hts.loader import load_lines
from landedcost.rates.duty import DutyStatus, estimate_duty, to_cents
from landedcost.rates.parser import RateKind, SpecificComponent, parse_rate
from landedcost.rates.units import canonical_unit


@pytest.mark.parametrize(
    ("raw", "kind", "pct", "specific"),
    [
        ("Free", RateKind.FREE, None, ()),
        ("free", RateKind.FREE, None, ()),
        ("6.8%", RateKind.AD_VALOREM, Decimal("6.8"), ()),
        ("2.5¢/kg", RateKind.SPECIFIC, None, (SpecificComponent(Decimal("0.025"), "kg"),)),
        ("$1.03/doz.", RateKind.SPECIFIC, None, (SpecificComponent(Decimal("1.03"), "doz"),)),
        ("4.4¢ each", RateKind.SPECIFIC, None, (SpecificComponent(Decimal("0.044"), "each"),)),
        ("1.9¢/ liter", RateKind.SPECIFIC, None, (SpecificComponent(Decimal("0.019"), "liter"),)),
        ("$3.50/t", RateKind.SPECIFIC, None, (SpecificComponent(Decimal("3.50"), "t"),)),
        (
            "3.2¢/kg + 5%",
            RateKind.COMPOUND,
            Decimal("5"),
            (SpecificComponent(Decimal("0.032"), "kg"),),
        ),
        (
            "1.5¢ each + 4.9%",
            RateKind.COMPOUND,
            Decimal("4.9"),
            (SpecificComponent(Decimal("0.015"), "each"),),
        ),
    ],
)
def test_parse_supported_rates(
    raw: str, kind: RateKind, pct: Decimal | None, specific: tuple[SpecificComponent, ...]
) -> None:
    rate = parse_rate(raw)
    assert rate.kind is kind
    assert rate.ad_valorem_pct == pct
    assert rate.specific == specific
    assert rate.note is None


@pytest.mark.parametrize(
    "raw",
    [
        "The rate applicable to each garment in the ensemble if separately entered",
        "6% on the case",
        "5¢/kg on drained weight",
        "18.7¢/clean kg",
        "5% + 3%",
        "2.5¢$/kg",
    ],
)
def test_unsupported_rates_are_unknown_not_guessed(raw: str) -> None:
    rate = parse_rate(raw)
    assert rate.kind is RateKind.UNKNOWN
    assert rate.note


def test_empty_rate_is_none() -> None:
    assert parse_rate("  ").kind is RateKind.NONE


def test_rate_parser_covers_most_of_the_shipped_snapshot() -> None:
    _, parsed = load_lines(Path("data/hts/hts_2026-09-04.json.gz"))
    rated = [line for line in parsed.lines if line.general_rate]
    unknown = sum(1 for line in rated if parse_rate(line.general_rate).kind is RateKind.UNKNOWN)
    assert unknown / len(rated) < 0.03


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("No.", "each"),
        ("kg", "kg"),
        ("liters", "liter"),
        ("prs.", "pair"),
        ("m²", "m2"),
        ("Doz.", "doz"),
    ],
)
def test_canonical_units(raw: str, canonical: str) -> None:
    assert canonical_unit(raw) == canonical
    assert canonical_unit("furlongs") is None


def test_duty_free_and_ad_valorem_rounding() -> None:
    assert estimate_duty(parse_rate("Free"), Decimal("1000")).amount_usd == Decimal("0.00")
    duty = estimate_duty(parse_rate("3.4%"), Decimal("12000"))
    assert duty.status is DutyStatus.COMPUTED
    assert duty.amount_usd == Decimal("408.00")
    # 2.5% of 33.30 = 0.8325 -> rounds half up to 0.83; 0.835 -> 0.84
    assert estimate_duty(parse_rate("2.5%"), Decimal("33.30")).amount_usd == Decimal("0.83")
    assert to_cents(Decimal("0.835")) == Decimal("0.84")


def test_specific_rate_needs_quantity() -> None:
    duty = estimate_duty(parse_rate("2.5¢/kg"), Decimal("500"))
    assert duty.status is DutyStatus.REQUIRES_QUANTITY
    assert duty.missing_units == ("kg",)
    assert duty.amount_usd is None
    computed = estimate_duty(parse_rate("2.5¢/kg"), Decimal("500"), {"kg": Decimal("120")})
    assert computed.amount_usd == Decimal("3.00")


def test_compound_rate_adds_both_parts() -> None:
    duty = estimate_duty(parse_rate("3¢ each + 4.5%"), Decimal("1000"), {"each": Decimal("200")})
    assert duty.amount_usd == Decimal("51.00")  # 6.00 specific + 45.00 ad valorem
    assert "4.5%" in duty.explanation


@pytest.mark.parametrize("raw", ["6% on the case", ""])
def test_unparseable_rate_requires_broker_review(raw: str) -> None:
    duty = estimate_duty(parse_rate(raw), Decimal("1000"))
    assert duty.status is DutyStatus.REQUIRES_BROKER_REVIEW
    assert duty.amount_usd is None
    assert "requires broker review" in duty.explanation
