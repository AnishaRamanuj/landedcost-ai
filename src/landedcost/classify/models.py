"""Public result types (also the API response schema) and the LLM output contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from landedcost import DISCLAIMER


class Citation(BaseModel):
    """A retrieved source an answer relies on."""

    chunk_id: str = Field(examples=["hts:8507.60.00.90"])
    kind: Literal["hts_line", "ruling"]
    reference: str = Field(description="HTS code or ruling number")
    text: str = Field(description="Tariff text (with hierarchy) or ruling excerpt")
    url: str | None = None


class RetrievalTrace(BaseModel):
    """How a candidate was found: per-retriever ranks, raw scores and the fused RRF score."""

    fused_score: float
    ranks: dict[str, int]
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Raw retriever scores (vector: cosine similarity; fts: BM25 or ts_rank_cd). "
        "Not comparable across retrievers; fusion uses the ranks.",
    )


class CandidateOut(BaseModel):
    hts_code: str
    level: str
    description: str
    breadcrumb: str
    general_rate: str
    rate_inherited_from: str | None = None
    rationale: str
    citations: list[str] = Field(description="chunk_ids of the sources supporting this code")
    retrieval: RetrievalTrace


class EvidenceSignals(BaseModel):
    """Inputs to the abstention gate. Heuristic, not a calibrated probability."""

    support: float = Field(description="Top candidate's fused score / best possible fused score")
    heading_consensus: float = Field(
        description="Share of fused score in the top window that agrees with the top heading"
    )
    fts_hits: int | None = None
    vector_hits: int | None = None
    ruling_hits: int = 0


class ClassificationResult(BaseModel):
    query: str
    candidates: list[CandidateOut]
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool
    abstain_reason: str | None = None
    citations: list[Citation] = Field(description="Every source referenced by a candidate")
    evidence: EvidenceSignals
    answer_mode: Literal["extractive", "llm"]
    llm: str | None = Field(default=None, description="provider:model when an LLM was used")
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


# -- LLM output contract ------------------------------------------------------------------


class LlmCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hts_code: str
    description: str
    general_rate: str
    rationale: str
    citations: list[str]


class LlmClassification(BaseModel):
    """What the model must return. Validated, then checked by the citation guard."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[LlmCandidate]
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool
    abstain_reason: str


LLM_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates", "confidence", "abstained", "abstain_reason"],
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hts_code", "description", "general_rate", "rationale", "citations"],
                "properties": {
                    "hts_code": {"type": "string"},
                    "description": {"type": "string"},
                    "general_rate": {"type": "string"},
                    "rationale": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "confidence": {"type": "number"},
        "abstained": {"type": "boolean"},
        "abstain_reason": {"type": "string"},
    },
}
"""Hand-written JSON Schema for structured outputs (no ``$ref``/``anyOf``, every object
closed with ``additionalProperties: false`` and a full ``required`` list). A unit test keeps
it in sync with the Pydantic models above."""
