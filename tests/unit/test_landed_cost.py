import json
from datetime import date
from decimal import Decimal

import httpx
import pytest

from landedcost.landed.calculator import LandedCostError, LandedCostService
from landedcost.landed.fees import (
    FeeSchedule,
    federal_fiscal_year,
    harbor_maintenance_fee,
    merchandise_processing_fee,
)
from landedcost.landed.freight import (
    FreightQuoteError,
    FreightQuoteRequest,
    HttpFreightQuoteProvider,
    StaticFreightQuoteProvider,
    TransportMode,
)
from landedcost.landed.models import CHAPTER_99_WARNING, LandedCostRequest
from landedcost.store.memory import InMemoryStore

FEES = FeeSchedule()
TODAY = date(2026, 10, 24)  # inside FY2027, the fiscal year of the default fee schedule


@pytest.mark.parametrize(
    ("value", "expected", "note"),
    [
        (Decimal("5000"), Decimal("34.58"), "minimum"),  # 0.3464% = 17.32 -> raised to min
        (Decimal("50000"), Decimal("173.20"), "of $50000"),  # within band
        (Decimal("500000"), Decimal("670.86"), "maximum"),  # 1732.00 -> capped at max
    ],
)
def test_mpf_is_clamped_to_min_and_max(value: Decimal, expected: Decimal, note: str) -> None:
    fee = merchandise_processing_fee(value, FEES)
    assert fee.amount_usd == expected
    assert note in fee.explanation


def test_hmf_is_ad_valorem() -> None:
    assert harbor_maintenance_fee(Decimal("12000"), FEES).amount_usd == Decimal("15.00")


def test_fee_schedule_validation_and_note() -> None:
    with pytest.raises(ValueError, match="minimum exceeds maximum"):
        FeeSchedule(mpf_min_usd=Decimal(10), mpf_max_usd=Decimal(5))
    with pytest.raises(ValueError, match="fractions"):
        FeeSchedule(mpf_rate=Decimal("0.3464"), hmf_rate=Decimal(2))
    with pytest.raises(ValueError, match="fiscal year"):
        FeeSchedule(fiscal_year="2027")
    assert "Verify against CBP" in FEES.verification_note
    assert "FY2027" in FEES.verification_note
    assert "min $34.58 / max $670.86" in FEES.verification_note


@pytest.mark.parametrize(
    ("day", "fiscal_year"),
    [(date(2026, 9, 30), 2026), (date(2026, 10, 1), 2027), (date(2027, 9, 30), 2027)],
)
def test_federal_fiscal_year_starts_on_october_first(day: date, fiscal_year: int) -> None:
    assert federal_fiscal_year(day) == fiscal_year


def test_stale_fee_schedule_is_flagged() -> None:
    assert FEES.staleness_warning(TODAY) is None
    assert FeeSchedule(fiscal_year="FY2028").staleness_warning(TODAY) is None
    warning = FeeSchedule(fiscal_year="FY2026").staleness_warning(TODAY)
    assert warning is not None
    assert "FY2026" in warning
    assert "falls in FY2027" in warning


def service(
    store: InMemoryStore, fees: FeeSchedule = FEES, today: date = TODAY
) -> LandedCostService:
    return LandedCostService(store, StaticFreightQuoteProvider(), fees, today=lambda: today)


async def test_ocean_shipment_with_ad_valorem_rate(memory_store: InMemoryStore) -> None:
    result = await service(memory_store).estimate(
        LandedCostRequest(
            hts_code="8507600090",
            customs_value_usd=Decimal("12000"),
            mode=TransportMode.OCEAN,
            freight_usd=Decimal("850"),
            insurance_usd=Decimal("60"),
        )
    )
    assert result.hts_code == "8507.60.00.90"
    assert result.rate_inherited_from == "8507.60.00"
    assert result.duty.amount_usd == Decimal("408.00")  # 3.4% x 12,000
    assert result.mpf.amount_usd == Decimal("41.57")  # 0.3464% x 12,000 = 41.568
    assert result.hmf is not None
    assert result.hmf.amount_usd == Decimal("15.00")
    assert result.subtotal_excluding_duty_usd == Decimal("12966.57")
    assert result.total_landed_cost_usd == Decimal("13374.57")
    assert result.freight.source == "provided"
    assert CHAPTER_99_WARNING in result.warnings
    assert any("FY2027" in w for w in result.warnings)
    assert not any("update MPF_MIN_USD" in w for w in result.warnings)


async def test_estimate_warns_when_the_fee_year_has_ended(memory_store: InMemoryStore) -> None:
    result = await service(memory_store, today=date(2027, 10, 2)).estimate(
        LandedCostRequest(
            hts_code="8507.60.00.90", customs_value_usd=Decimal("100"), freight_usd=Decimal("1")
        )
    )
    assert any("falls in FY2028" in w for w in result.warnings)


async def test_air_shipment_has_no_hmf_and_uses_quoted_freight(memory_store: InMemoryStore) -> None:
    result = await service(memory_store).estimate(
        LandedCostRequest(
            hts_code="0409.00.00.10",
            customs_value_usd=Decimal("2000"),
            mode=TransportMode.AIR,
            weight_kg=Decimal("100"),
            volume_cbm=Decimal("1"),
        )
    )
    assert result.hmf is None
    assert result.freight.source == "static-mock"
    assert result.freight.amount_usd == Decimal("751.50")  # max(100, 1*167) kg x $4.50
    assert result.duty.amount_usd is not None


async def test_specific_rate_uses_weight_when_kg_missing(memory_store: InMemoryStore) -> None:
    # 0102.29.40 carries "1¢/kg"; its statistical lines inherit it.
    result = await service(memory_store).estimate(
        LandedCostRequest(
            hts_code="0102.29.40.24",
            customs_value_usd=Decimal("10000"),
            freight_usd=Decimal("0"),
            weight_kg=Decimal("1500"),
            mode=TransportMode.TRUCK,
        )
    )
    assert result.rate_kind == "specific"
    assert result.duty.amount_usd == Decimal("15.00")
    assert any("weight_kg was used" in w for w in result.warnings)


async def test_compound_rate_without_quantity_omits_total(memory_store: InMemoryStore) -> None:
    # 8213.00.30.00 (scissors) has a compound "each + %" rate.
    result = await service(memory_store).estimate(
        LandedCostRequest(
            hts_code="8213.00.30.00", customs_value_usd=Decimal("1000"), freight_usd=Decimal("50")
        )
    )
    assert result.rate_kind == "compound"
    assert result.duty.status == "requires_quantity"
    assert result.duty.missing_units == ["each"]
    assert result.total_landed_cost_usd is None
    assert any("Total omitted" in w for w in result.warnings)
    priced = await service(memory_store).estimate(
        LandedCostRequest(
            hts_code="8213.00.30.00",
            customs_value_usd=Decimal("1000"),
            freight_usd=Decimal("50"),
            quantities={"No.": Decimal("400")},
        )
    )
    assert priced.duty.status == "computed"
    assert priced.total_landed_cost_usd is not None


async def test_unparseable_rate_requires_broker_review(memory_store: InMemoryStore) -> None:
    result = await service(memory_store).estimate(
        LandedCostRequest(
            hts_code="9103.10.40", customs_value_usd=Decimal("1000"), freight_usd=Decimal("10")
        )
    )
    assert result.duty.status == "requires_broker_review"
    assert result.total_landed_cost_usd is None


@pytest.mark.parametrize(
    ("code", "match"),
    [
        ("8507", "carries no duty rate"),
        ("9999.99.99.99", "not in the loaded"),
        ("85", "not an HTS code"),
    ],
)
async def test_unpriceable_codes(memory_store: InMemoryStore, code: str, match: str) -> None:
    with pytest.raises(LandedCostError, match=match):
        await service(memory_store).estimate(
            LandedCostRequest(
                hts_code=code, customs_value_usd=Decimal("100"), freight_usd=Decimal("1")
            )
        )


async def test_freight_needs_weight_or_volume(memory_store: InMemoryStore) -> None:
    with pytest.raises(FreightQuoteError, match="weight_kg or volume_cbm"):
        await service(memory_store).estimate(
            LandedCostRequest(hts_code="8507.60.00.90", customs_value_usd=Decimal("100"))
        )


def test_request_validates_units_and_amounts() -> None:
    req = LandedCostRequest(
        hts_code="x",
        customs_value_usd=Decimal(1),
        quantities={"No.": Decimal(3), "liters": Decimal(2)},
    )
    assert req.quantities == {"each": Decimal(3), "liter": Decimal(2)}
    with pytest.raises(ValueError, match="unknown unit"):
        LandedCostRequest(
            hts_code="x", customs_value_usd=Decimal(1), quantities={"cubits": Decimal(1)}
        )
    with pytest.raises(ValueError, match="greater than 0"):
        LandedCostRequest(hts_code="x", customs_value_usd=Decimal(0))


@pytest.mark.parametrize(
    "fields",
    [
        {"customs_value_usd": "1e30"},
        {"customs_value_usd": "1e100"},
        {"customs_value_usd": "NaN"},
        {"customs_value_usd": "1", "freight_usd": "2e12"},
        {"customs_value_usd": "1", "weight_kg": "0.00001"},
        {"customs_value_usd": "1", "quantities": {"kg": "-1"}},
        {"customs_value_usd": "1", "quantities": {"kg": "1e40"}},
    ],
)
def test_request_rejects_out_of_range_numbers(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="validation error"):
        LandedCostRequest.model_validate({"hts_code": "8507.60.00.90", **fields})


async def test_largest_accepted_amounts_still_price(memory_store: InMemoryStore) -> None:
    result = await service(memory_store).estimate(
        LandedCostRequest(
            hts_code="8507.60.00.90",
            customs_value_usd=Decimal("1e12"),
            freight_usd=Decimal("999999999999.9999"),
            insurance_usd=Decimal("1e12"),
        )
    )
    assert result.mpf.amount_usd == FEES.mpf_max_usd
    assert result.total_landed_cost_usd is not None


# -- freight providers ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "weight", "volume", "amount"),
    [
        (TransportMode.OCEAN, "8000", "12", "1140.00"),  # W/M = max(8 t, 12 m3) x $95
        (TransportMode.OCEAN, "100", None, "150.00"),  # minimum
        (TransportMode.AIR, "50", "1", "751.50"),  # volumetric 167 kg x $4.50
        (TransportMode.TRUCK, "2000", None, "700.00"),
        (TransportMode.RAIL, "2000", None, "400.00"),
    ],
)
async def test_static_freight_table(
    mode: TransportMode, weight: str, volume: str | None, amount: str
) -> None:
    quote = await StaticFreightQuoteProvider().quote(
        FreightQuoteRequest(
            mode=mode, weight_kg=Decimal(weight), volume_cbm=Decimal(volume) if volume else None
        )
    )
    assert quote.amount_usd == Decimal(amount)
    assert "MOCK" in quote.note
    assert "E+" not in quote.note  # no scientific notation


async def test_static_freight_note_prints_plain_numbers() -> None:
    quote = await StaticFreightQuoteProvider().quote(
        FreightQuoteRequest(mode=TransportMode.TRUCK, weight_kg=Decimal("1000"))
    )
    assert quote.note.startswith("MOCK quote: 1000 kg x $0.35")


def _http_provider(handler: httpx.MockTransport) -> HttpFreightQuoteProvider:
    return HttpFreightQuoteProvider(httpx.AsyncClient(transport=handler), "http://quotes.test/")


async def test_http_freight_provider_success() -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "amount_usd": "1234.567",
                "quote_id": "Q1",
                "carrier": "SimLine",
                "transit_days": 21,
            },
        )

    quote = await _http_provider(httpx.MockTransport(handle)).quote(
        FreightQuoteRequest(
            mode=TransportMode.OCEAN, weight_kg=Decimal("800"), origin="CNSHA", destination="USLAX"
        )
    )
    assert seen["url"] == "http://quotes.test/v1/quotes"
    assert seen["body"] == {
        "mode": "ocean",
        "weight_kg": "800",
        "volume_cbm": None,
        "origin": "CNSHA",
        "destination": "USLAX",
    }
    assert quote.amount_usd == Decimal("1234.57")
    assert (quote.quote_id, quote.carrier, quote.transit_days) == ("Q1", "SimLine", 21)


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(503, text="down"), "HTTP 503"),
        (httpx.Response(200, text="not json"), "invalid body"),
        (httpx.Response(200, json={"price": 1}), "invalid body"),
        (httpx.Response(200, json={"amount_usd": -5}), "negative"),
    ],
)
async def test_http_freight_provider_failures(response: httpx.Response, match: str) -> None:
    provider = _http_provider(httpx.MockTransport(lambda _: response))
    with pytest.raises(FreightQuoteError, match=match) as info:
        await provider.quote(FreightQuoteRequest(mode=TransportMode.AIR, weight_kg=Decimal(1)))
    assert info.value.upstream


async def test_http_freight_provider_unreachable() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(FreightQuoteError, match="unreachable"):
        await _http_provider(httpx.MockTransport(boom)).quote(
            FreightQuoteRequest(mode=TransportMode.AIR, weight_kg=Decimal(1))
        )


def test_percent_formatting_in_explanations() -> None:
    assert merchandise_processing_fee(Decimal("50000"), FEES).explanation == "0.3464% of $50000"
    assert harbor_maintenance_fee(Decimal("100"), FEES).explanation == "0.125% of $100"


async def test_subheading_without_its_own_row_points_to_rate_lines(
    memory_store: InMemoryStore,
) -> None:
    with pytest.raises(LandedCostError, match=r"8507\.60 carries no duty rate.*8507\.60\.00"):
        await service(memory_store).estimate(
            LandedCostRequest(
                hts_code="8507.60", customs_value_usd=Decimal("100"), freight_usd=Decimal("1")
            )
        )
