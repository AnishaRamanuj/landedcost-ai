"""Request/response models that are specific to the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from landedcost import DISCLAIMER


class ClassifyRequest(BaseModel):
    description: str = Field(
        min_length=3,
        max_length=2000,
        examples=["Rechargeable lithium-ion battery pack for electric bicycles, 36V 10Ah"],
    )
    top_k: int | None = Field(default=None, ge=1, le=10)


class HtsLineOut(BaseModel):
    hts_code: str
    level: str
    description: str
    breadcrumb: str
    units: list[str]
    general_rate: str
    special_rate: str
    other_rate: str
    rate_inherited_from: str | None
    rate_kind: str
    footnotes: list[str]
    disclaimer: str = DISCLAIMER


class HealthOut(BaseModel):
    status: str
    store: str
    hts_lines: int
    rulings: int
    hts_version: str | None
    embedder: str
    index_embedder: str | None
    llm: str | None
    retrieval_mode: str
