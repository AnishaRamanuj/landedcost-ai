"""Parsed HTS records."""

from __future__ import annotations

from dataclasses import dataclass

from landedcost.hts.codes import CodeLevel, HtsCode


@dataclass(frozen=True, slots=True)
class HtsLine:
    """One numbered HTS line with its full hierarchical context.

    ``path`` holds the descriptions of every ancestor row (numbered or not), root first, so
    a leaf such as ``8507.60.00.90 Other`` keeps the context that makes it meaningful:
    ``Electric storage batteries ... > Lithium-ion batteries > Other > Other``.
    """

    code: str
    level: CodeLevel
    description: str
    path: tuple[str, ...]
    indent: int
    units: tuple[str, ...]
    general_rate: str
    special_rate: str
    other_rate: str
    rate_source_code: str | None
    footnotes: tuple[str, ...]

    @property
    def hts_code(self) -> HtsCode:
        return HtsCode.parse(self.code)

    @property
    def chapter(self) -> str:
        return self.code[:2]

    @property
    def heading(self) -> str:
        return self.code[:4]

    @property
    def subheading(self) -> str | None:
        return self.hts_code.subheading

    @property
    def chunk_id(self) -> str:
        """Stable identifier used for retrieval results and citations."""
        return f"hts:{self.code}"

    @property
    def breadcrumb(self) -> str:
        return " > ".join((*self.path, self.description))

    @property
    def rate_inherited(self) -> bool:
        """True when the general rate was taken from an ancestor tariff line."""
        return self.rate_source_code is not None and self.rate_source_code != self.code

    @property
    def embedding_text(self) -> str:
        """Text embedded for vector search: the line's own wording first, then its full path.

        Leading with (and so repeating) the line's own description gives it more weight than
        the shared ancestor text, the same emphasis the full-text index applies with its
        ``A``/``B`` weights.
        """
        return f"{self.description}. {self.breadcrumb}"
