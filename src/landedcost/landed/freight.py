"""Freight quotes behind a small interface.

* :class:`StaticFreightQuoteProvider` (default) returns deterministic, clearly-labelled
  mock quotes from a rate table; it exists so landed-cost estimates work offline.
* :class:`HttpFreightQuoteProvider` POSTs to an external quote API that implements the
  JSON contract documented on the class (for example an adapter in front of a carrier
  sandbox).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

import httpx
from pydantic import BaseModel, ValidationError

from landedcost.rates.duty import to_cents


class TransportMode(StrEnum):
    OCEAN = "ocean"
    AIR = "air"
    TRUCK = "truck"
    RAIL = "rail"


class FreightQuoteError(RuntimeError):
    """A quote could not be produced (bad input or upstream failure)."""

    def __init__(self, message: str, *, upstream: bool = False) -> None:
        super().__init__(message)
        self.upstream = upstream


@dataclass(frozen=True, slots=True)
class FreightQuoteRequest:
    mode: TransportMode
    weight_kg: Decimal | None = None
    volume_cbm: Decimal | None = None
    origin: str | None = None
    destination: str | None = None


@dataclass(frozen=True, slots=True)
class FreightQuote:
    amount_usd: Decimal
    provider: str
    note: str
    quote_id: str | None = None
    carrier: str | None = None
    transit_days: int | None = None


class FreightQuoteProvider(Protocol):
    @property
    def name(self) -> str: ...

    async def quote(self, request: FreightQuoteRequest) -> FreightQuote: ...


@dataclass(frozen=True, slots=True)
class _ModeRate:
    per_chargeable_unit: Decimal
    minimum: Decimal
    unit: str


# Illustrative numbers only - not market rates.
_MOCK_RATES: dict[TransportMode, _ModeRate] = {
    TransportMode.OCEAN: _ModeRate(Decimal("95.00"), Decimal("150.00"), "W/M (t or m3)"),
    TransportMode.AIR: _ModeRate(Decimal("4.50"), Decimal("120.00"), "chargeable kg"),
    TransportMode.TRUCK: _ModeRate(Decimal("0.35"), Decimal("250.00"), "kg"),
    TransportMode.RAIL: _ModeRate(Decimal("0.20"), Decimal("300.00"), "kg"),
}
_AIR_VOLUMETRIC_KG_PER_CBM = Decimal(167)


class StaticFreightQuoteProvider:
    """Deterministic mock freight from a per-mode rate table (not market rates)."""

    name = "static-mock"

    async def quote(self, request: FreightQuoteRequest) -> FreightQuote:
        weight = request.weight_kg or Decimal(0)
        volume = request.volume_cbm or Decimal(0)
        if weight <= 0 and volume <= 0:
            raise FreightQuoteError("weight_kg or volume_cbm is required for a freight quote")
        rate = _MOCK_RATES[request.mode]
        if request.mode is TransportMode.OCEAN:
            chargeable = max(weight / Decimal(1000), volume)  # weight-or-measure
        elif request.mode is TransportMode.AIR:
            chargeable = max(weight, volume * _AIR_VOLUMETRIC_KG_PER_CBM)
        else:
            chargeable = weight
        amount = max(to_cents(chargeable * rate.per_chargeable_unit), rate.minimum)
        return FreightQuote(
            amount_usd=amount,
            provider=self.name,
            note=(
                f"MOCK quote: {chargeable.normalize():f} {rate.unit} x ${rate.per_chargeable_unit} "
                f"(minimum ${rate.minimum}); illustrative, not a market rate"
            ),
        )


class _QuoteResponse(BaseModel):
    amount_usd: Decimal
    quote_id: str | None = None
    carrier: str | None = None
    transit_days: int | None = None


class HttpFreightQuoteProvider:
    """Calls ``POST {base_url}/v1/quotes``.

    Request body: ``{"mode", "weight_kg", "volume_cbm", "origin", "destination"}``.
    Expected response: ``{"amount_usd": number, "quote_id"?, "carrier"?, "transit_days"?}``.
    """

    name = "http"

    def __init__(
        self, client: httpx.AsyncClient, base_url: str, *, path: str = "/v1/quotes"
    ) -> None:
        self._client = client
        self._url = base_url.rstrip("/") + path

    async def quote(self, request: FreightQuoteRequest) -> FreightQuote:
        payload = {
            "mode": request.mode.value,
            "weight_kg": str(request.weight_kg) if request.weight_kg is not None else None,
            "volume_cbm": str(request.volume_cbm) if request.volume_cbm is not None else None,
            "origin": request.origin,
            "destination": request.destination,
        }
        try:
            response = await self._client.post(self._url, json=payload)
            response.raise_for_status()
            data = _QuoteResponse.model_validate(response.json())
        except httpx.HTTPStatusError as exc:
            raise FreightQuoteError(
                f"freight API returned HTTP {exc.response.status_code}", upstream=True
            ) from exc
        except httpx.HTTPError as exc:
            raise FreightQuoteError(f"freight API unreachable: {exc}", upstream=True) from exc
        except (ValueError, ValidationError) as exc:
            raise FreightQuoteError(
                f"freight API returned an invalid body: {exc}", upstream=True
            ) from exc
        if data.amount_usd < 0:
            raise FreightQuoteError("freight API returned a negative amount", upstream=True)
        return FreightQuote(
            amount_usd=to_cents(data.amount_usd),
            provider=self.name,
            note=f"quote from {self._url}",
            quote_id=data.quote_id,
            carrier=data.carrier,
            transit_days=data.transit_days,
        )
