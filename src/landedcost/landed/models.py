"""Landed-cost request/response models (also the API schema).

Money is ``Decimal`` end to end and serialises to JSON strings (``"1234.50"``) so no
amount is ever rounded through a binary float.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from landedcost import DISCLAIMER
from landedcost.landed.freight import TransportMode
from landedcost.rates.units import canonical_unit

CHAPTER_99_WARNING = (
    "Chapter 99 additional duties (Section 301, Section 232, IEEPA and other temporary "
    "tariffs) and antidumping/countervailing duties are NOT included. They can exceed the "
    "MFN duty; check the country of origin against HTS chapter 99 and AD/CVD orders."
)
MFN_ONLY_WARNING = (
    "Duty uses the column 1 General (MFN) rate only; free-trade-agreement (Special) and "
    "column 2 rates are not applied."
)
VALUATION_NOTE = (
    "Duty, MPF and HMF are computed on the customs (transaction) value; freight and "
    "insurance are added to landed cost but not to the dutiable value."
)
FORMAL_ENTRY_NOTE = "MPF assumes a formal entry; informal entries use a different fee."

MAX_INPUT = Decimal("1e12")
"""Upper bound for every amount and quantity a caller supplies. Together with the digit
limits it keeps every intermediate product well inside Decimal's 28-digit context, so
rounding to cents can never fail on absurd input such as ``1e30``."""

NonNegativeInput = Annotated[
    Decimal, Field(ge=0, le=MAX_INPUT, max_digits=18, decimal_places=4, allow_inf_nan=False)
]
PositiveInput = Annotated[
    Decimal, Field(gt=0, le=MAX_INPUT, max_digits=18, decimal_places=4, allow_inf_nan=False)
]


class LandedCostRequest(BaseModel):
    hts_code: str = Field(examples=["8507.60.00.90"], description="8- or 10-digit HTS code")
    customs_value_usd: PositiveInput = Field(examples=["12000"])
    mode: TransportMode = TransportMode.OCEAN
    freight_usd: NonNegativeInput | None = Field(
        default=None, description="Known freight cost; omit to ask the freight quote provider"
    )
    insurance_usd: NonNegativeInput | None = None
    weight_kg: PositiveInput | None = None
    volume_cbm: PositiveInput | None = None
    quantities: dict[str, NonNegativeInput] = Field(
        default_factory=dict,
        description='Quantities for specific rates, e.g. {"kg": 120, "each": 500, "doz": 40}',
        examples=[{"kg": "850", "each": "200"}],
    )
    origin: str | None = None
    destination: str | None = None

    @field_validator("quantities")
    @classmethod
    def _canonical_units(cls, value: dict[str, Decimal]) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for unit, qty in value.items():
            canonical = canonical_unit(unit)
            if canonical is None:
                raise ValueError(f"unknown unit {unit!r}")
            out[canonical] = qty
        return out


class FreightOut(BaseModel):
    amount_usd: Decimal
    source: str = Field(description="'provided' or the quote provider's name")
    note: str
    quote_id: str | None = None
    carrier: str | None = None
    transit_days: int | None = None


class DutyOut(BaseModel):
    status: str
    amount_usd: Decimal | None
    explanation: str
    missing_units: list[str] = Field(default_factory=list)


class FeeOut(BaseModel):
    amount_usd: Decimal
    explanation: str


class LandedCostResult(BaseModel):
    hts_code: str
    description: str
    breadcrumb: str
    general_rate: str
    rate_kind: str
    rate_inherited_from: str | None = None
    customs_value_usd: Decimal
    freight: FreightOut
    insurance_usd: Decimal
    duty: DutyOut
    mpf: FeeOut
    hmf: FeeOut | None = Field(description="Harbor Maintenance Fee; ocean shipments only")
    subtotal_excluding_duty_usd: Decimal
    total_landed_cost_usd: Decimal | None = Field(
        description="None when the duty needs a quantity or broker review"
    )
    fee_schedule: str
    warnings: list[str]
    disclaimer: str = DISCLAIMER
