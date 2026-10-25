"""CBP user-fee parameters.

The MPF ad valorem rate (0.3464%) and HMF rate (0.125%) come from 19 U.S.C. 58c and
26 U.S.C. 4461. The MPF minimum and maximum for formal entries are adjusted for inflation
every federal fiscal year (starting October 1) in a CBP Decision published in the Federal
Register. The defaults below are the FY2027 values (effective 2026-10-01) announced in
CBP Dec. 26-14: minimum $34.58, maximum $670.86, ad valorem rate unchanged (FY2026 was
$33.58 / $651.50). They are configuration, NOT verified at runtime: check the current
notice and set ``MPF_MIN_USD``, ``MPF_MAX_USD`` and ``FEE_SCHEDULE_FISCAL_YEAR`` before
relying on an estimate. Every estimate repeats the fiscal year it used and warns when that
year has already ended.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from landedcost.rates.duty import to_cents

_FISCAL_YEAR = re.compile(r"^FY(\d{4})$")


def percent(fraction: Decimal) -> str:
    """``Decimal("0.003464")`` -> ``"0.3464%"``."""
    return f"{(fraction * 100).normalize():f}%"


def federal_fiscal_year(day: date) -> int:
    """US federal fiscal year containing ``day`` (FY2027 runs 2026-10-01 to 2027-09-30)."""
    return day.year + 1 if day.month >= 10 else day.year


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    fiscal_year: str = "FY2027"
    mpf_rate: Decimal = Decimal("0.003464")
    mpf_min_usd: Decimal = Decimal("34.58")
    mpf_max_usd: Decimal = Decimal("670.86")
    hmf_rate: Decimal = Decimal("0.00125")

    def __post_init__(self) -> None:
        if not _FISCAL_YEAR.match(self.fiscal_year):
            raise ValueError(f"fiscal year must look like 'FY2027', got {self.fiscal_year!r}")
        if self.mpf_min_usd > self.mpf_max_usd:
            raise ValueError("MPF minimum exceeds maximum")
        if not (Decimal(0) <= self.mpf_rate < 1 and Decimal(0) <= self.hmf_rate < 1):
            raise ValueError("fee rates must be fractions, e.g. 0.003464 for 0.3464%")

    @property
    def verification_note(self) -> str:
        return (
            f"MPF/HMF parameters are configured for {self.fiscal_year} "
            f"(MPF {percent(self.mpf_rate)} with min ${self.mpf_min_usd} / "
            f"max ${self.mpf_max_usd}; HMF {percent(self.hmf_rate)}). "
            "Verify against CBP's current user-fee table."
        )

    def staleness_warning(self, today: date) -> str | None:
        """A warning when ``today`` falls after the configured fiscal year, else ``None``."""
        match = _FISCAL_YEAR.match(self.fiscal_year)
        assert match is not None  # checked in __post_init__
        current = federal_fiscal_year(today)
        if int(match.group(1)) >= current:
            return None
        return (
            f"The fee parameters are for {self.fiscal_year}, but {today.isoformat()} falls in "
            f"FY{current}: update MPF_MIN_USD, MPF_MAX_USD and FEE_SCHEDULE_FISCAL_YEAR from "
            f"CBP's FY{current} user-fee notice."
        )


@dataclass(frozen=True, slots=True)
class FeeAmount:
    amount_usd: Decimal
    explanation: str


def merchandise_processing_fee(value_usd: Decimal, fees: FeeSchedule) -> FeeAmount:
    """MPF for a formal entry: ad valorem, clamped to the fiscal-year min/max."""
    raw = value_usd * fees.mpf_rate
    rate = percent(fees.mpf_rate)
    if raw < fees.mpf_min_usd:
        return FeeAmount(fees.mpf_min_usd, f"{rate} = ${to_cents(raw)}, raised to minimum")
    if raw > fees.mpf_max_usd:
        return FeeAmount(fees.mpf_max_usd, f"{rate} = ${to_cents(raw)}, capped at maximum")
    return FeeAmount(to_cents(raw), f"{rate} of ${value_usd}")


def harbor_maintenance_fee(value_usd: Decimal, fees: FeeSchedule) -> FeeAmount:
    """HMF applies to commercial cargo arriving by vessel at US ports."""
    return FeeAmount(
        to_cents(value_usd * fees.hmf_rate), f"{percent(fees.hmf_rate)} of ${value_usd}"
    )
