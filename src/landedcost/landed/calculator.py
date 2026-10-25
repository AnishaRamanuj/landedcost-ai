"""Landed cost = customs value + freight + insurance + MFN duty + MPF + HMF (ocean)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from decimal import Decimal

from landedcost.hts.codes import HtsCode, InvalidHtsCodeError
from landedcost.hts.models import HtsLine
from landedcost.landed.fees import FeeSchedule, harbor_maintenance_fee, merchandise_processing_fee
from landedcost.landed.freight import (
    FreightQuoteProvider,
    FreightQuoteRequest,
    TransportMode,
)
from landedcost.landed.models import (
    CHAPTER_99_WARNING,
    FORMAL_ENTRY_NOTE,
    MFN_ONLY_WARNING,
    VALUATION_NOTE,
    DutyOut,
    FeeOut,
    FreightOut,
    LandedCostRequest,
    LandedCostResult,
)
from landedcost.rates.duty import DutyStatus, estimate_duty, to_cents
from landedcost.rates.parser import parse_rate
from landedcost.store.base import Store


class LandedCostError(ValueError):
    """The request cannot be priced (unknown code, code without a rate, ...)."""


class LandedCostService:
    def __init__(
        self,
        store: Store,
        freight: FreightQuoteProvider,
        fees: FeeSchedule,
        *,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._store = store
        self._freight = freight
        self._fees = fees
        self._today = today

    async def _resolve_line(self, raw_code: str) -> HtsLine:
        try:
            code = HtsCode.parse(raw_code)
        except InvalidHtsCodeError as exc:
            raise LandedCostError(str(exc)) from exc
        line = (await self._store.get_hts_lines([code.dotted])).get(code.dotted)
        if line is not None and line.general_rate:
            return line
        children = await self._store.find_hts_by_prefix(code.dotted + ".", 50)
        rated = [c.code for c in children if c.general_rate][:5]
        if line is None and not children:
            raise LandedCostError(f"HTS {code.dotted} is not in the loaded tariff schedule")
        hint = ", ".join(rated) if rated else "a more specific code"
        raise LandedCostError(
            f"HTS {code.dotted} carries no duty rate; rates are set on 8/10-digit lines "
            f"(e.g. {hint})"
        )

    async def estimate(self, request: LandedCostRequest) -> LandedCostResult:
        """Raises ``LandedCostError`` for unpriceable codes and ``FreightQuoteError`` when no
        freight was provided and the quote provider fails."""
        line = await self._resolve_line(request.hts_code)
        rate = parse_rate(line.general_rate)
        warnings = [CHAPTER_99_WARNING, MFN_ONLY_WARNING, self._fees.verification_note]
        if stale := self._fees.staleness_warning(self._today()):
            warnings.append(stale)

        quantities = dict(request.quantities)
        if "kg" in rate.required_units and "kg" not in quantities and request.weight_kg:
            quantities["kg"] = request.weight_kg
            warnings.append(
                "weight_kg was used as the 'kg' quantity for the specific rate; HTS specific "
                "rates generally apply to net weight."
            )
        value = request.customs_value_usd
        duty = estimate_duty(rate, value, quantities)

        if request.freight_usd is not None:
            freight = FreightOut(
                amount_usd=to_cents(request.freight_usd),
                source="provided",
                note="supplied by caller",
            )
        else:
            quote = await self._freight.quote(
                FreightQuoteRequest(
                    mode=request.mode,
                    weight_kg=request.weight_kg,
                    volume_cbm=request.volume_cbm,
                    origin=request.origin,
                    destination=request.destination,
                )
            )
            freight = FreightOut(
                amount_usd=quote.amount_usd,
                source=quote.provider,
                note=quote.note,
                quote_id=quote.quote_id,
                carrier=quote.carrier,
                transit_days=quote.transit_days,
            )

        insurance = to_cents(request.insurance_usd or Decimal(0))
        mpf = merchandise_processing_fee(value, self._fees)
        hmf = (
            harbor_maintenance_fee(value, self._fees)
            if request.mode is TransportMode.OCEAN
            else None
        )
        subtotal = to_cents(
            value + freight.amount_usd + insurance + mpf.amount_usd + (hmf.amount_usd if hmf else 0)
        )
        total = (
            to_cents(subtotal + duty.amount_usd)
            if duty.status is DutyStatus.COMPUTED and duty.amount_usd is not None
            else None
        )
        if total is None:
            warnings.append(f"Total omitted: duty {duty.status.value.replace('_', ' ')}.")
        warnings.extend([VALUATION_NOTE, FORMAL_ENTRY_NOTE])

        return LandedCostResult(
            hts_code=line.code,
            description=line.description,
            breadcrumb=line.breadcrumb,
            general_rate=line.general_rate,
            rate_kind=rate.kind.value,
            rate_inherited_from=line.rate_source_code if line.rate_inherited else None,
            customs_value_usd=to_cents(value),
            freight=freight,
            insurance_usd=insurance,
            duty=DutyOut(
                status=duty.status.value,
                amount_usd=duty.amount_usd,
                explanation=duty.explanation,
                missing_units=list(duty.missing_units),
            ),
            mpf=FeeOut(amount_usd=mpf.amount_usd, explanation=mpf.explanation),
            hmf=FeeOut(amount_usd=hmf.amount_usd, explanation=hmf.explanation) if hmf else None,
            subtotal_excluding_duty_usd=subtotal,
            total_landed_cost_usd=total,
            fee_schedule=self._fees.fiscal_year,
            warnings=warnings,
        )
