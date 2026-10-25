"""HTS code normalisation.

The HTS is hierarchical by digit count: chapter (2) > heading (4) > subheading (6) >
tariff line (8, where duty rates live) > statistical suffix (10). Users and rulings write
the same code many ways (``8507600090``, ``8507.60.0090``, ``8507.60.00.90``); everything
in this package works with the canonical dotted form produced here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_SEPARATORS = re.compile(r"[\s.\-]")


class InvalidHtsCodeError(ValueError):
    """Raised when a string cannot be interpreted as an HTS code."""


class CodeLevel(StrEnum):
    """Granularity of an HTS code."""

    HEADING = "heading"
    SUBHEADING = "subheading"
    TARIFF_LINE = "tariff_line"
    STATISTICAL = "statistical"


_LEVEL_BY_LENGTH: dict[int, CodeLevel] = {
    4: CodeLevel.HEADING,
    6: CodeLevel.SUBHEADING,
    8: CodeLevel.TARIFF_LINE,
    10: CodeLevel.STATISTICAL,
}
_LENGTH_BY_LEVEL: dict[CodeLevel, int] = {level: n for n, level in _LEVEL_BY_LENGTH.items()}


@dataclass(frozen=True, slots=True, order=True)
class HtsCode:
    """A validated HTS code stored as its digits."""

    digits: str

    @classmethod
    def parse(cls, raw: str) -> HtsCode:
        """Parse any common spelling of an HTS code.

        Raises:
            InvalidHtsCodeError: if the input is not 4, 6, 8 or 10 digits after removing
                dots, dashes and whitespace.
        """
        digits = _SEPARATORS.sub("", raw)
        if not digits.isdigit() or len(digits) not in _LEVEL_BY_LENGTH:
            raise InvalidHtsCodeError(f"not an HTS code: {raw!r} (expected 4, 6, 8 or 10 digits)")
        return cls(digits)

    @property
    def level(self) -> CodeLevel:
        return _LEVEL_BY_LENGTH[len(self.digits)]

    @property
    def chapter(self) -> str:
        return self.digits[:2]

    @property
    def heading(self) -> str:
        """The 4-digit heading, e.g. ``8507``."""
        return self.digits[:4]

    @property
    def subheading(self) -> str | None:
        """The dotted 6-digit subheading (``8507.60``), or ``None`` for a bare heading."""
        if len(self.digits) < 6:
            return None
        return f"{self.digits[:4]}.{self.digits[4:6]}"

    @property
    def dotted(self) -> str:
        """Canonical dotted form: ``8507``, ``8507.60``, ``8507.60.00`` or ``8507.60.00.90``."""
        d = self.digits
        parts = [d[:4]] + [d[i : i + 2] for i in range(4, len(d), 2)]
        return ".".join(parts)

    def truncate(self, level: CodeLevel) -> HtsCode:
        """Return the ancestor code at ``level`` (or ``self`` if already coarser)."""
        n = _LENGTH_BY_LEVEL[level]
        return self if len(self.digits) <= n else HtsCode(self.digits[:n])

    def is_within(self, other: HtsCode) -> bool:
        """True if ``self`` equals ``other`` or is one of its descendants."""
        return self.digits.startswith(other.digits)

    def __str__(self) -> str:
        return self.dotted


def normalize_code(raw: str) -> str:
    """Return the canonical dotted spelling of ``raw``."""
    return HtsCode.parse(raw).dotted
