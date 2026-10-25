"""Classification pipeline: retrieve -> gate -> (LLM + citation guard | extractive) -> result."""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from landedcost.classify.gate import EvidenceGate, GateDecision
from landedcost.classify.guard import GuardedCandidate, apply_citation_guard
from landedcost.classify.models import (
    LLM_OUTPUT_SCHEMA,
    CandidateOut,
    Citation,
    ClassificationResult,
    LlmClassification,
    RetrievalTrace,
)
from landedcost.classify.prompts import SYSTEM_PROMPT, build_user_prompt
from landedcost.llm.base import LlmProvider, LlmProviderError
from landedcost.retrieval.hybrid import (
    Candidate,
    HybridRetriever,
    RetrievalMode,
    RetrievalResult,
    RulingCandidate,
)

logger = logging.getLogger(__name__)

_RETRIEVER_LABELS = {"vector": "vector", "fts": "full-text", "rulings": "ruling-citation"}
LLM_FALLBACK_WARNING = "LLM unavailable"
"""Prefix of the warning added when a provider call fails and the extractive answer is used."""


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    mode: RetrievalMode = RetrievalMode.HYBRID
    answer_candidates: int = 3
    """Candidates returned to the caller."""
    context_candidates: int = 12
    """HTS lines shown to the LLM (and considered by the gate)."""
    context_rulings: int = 5


class Classifier:
    def __init__(
        self,
        retriever: HybridRetriever,
        gate: EvidenceGate,
        llm: LlmProvider | None = None,
        config: ClassifierConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._gate = gate
        self._llm = llm
        self._config = config or ClassifierConfig()

    @property
    def llm_label(self) -> str | None:
        """``provider:model`` of the configured LLM, or ``None`` for extractive answers."""
        return self._llm.label if self._llm is not None else None

    async def classify(
        self,
        description: str,
        *,
        top_k: int | None = None,
        exclude_rulings: Collection[str] = (),
    ) -> ClassificationResult:
        """Retrieve, gate and answer. ``exclude_rulings`` supports leave-one-out evals."""
        cfg = self._config
        answer_k = top_k or cfg.answer_candidates
        retrieval = await self._retriever.retrieve(
            description,
            mode=cfg.mode,
            top_k=max(cfg.context_candidates, self._gate.config.window, answer_k),
            exclude_rulings=exclude_rulings,
        )
        decision = self._gate.evaluate(retrieval)
        if decision.abstain:
            return _abstained(retrieval, decision, decision.reason or "weak evidence")
        if self._llm is None:
            return _extractive(retrieval, decision, answer_k)
        return await self._classify_with_llm(retrieval, decision, answer_k, self._llm)

    async def _classify_with_llm(
        self, retrieval: RetrievalResult, decision: GateDecision, answer_k: int, llm: LlmProvider
    ) -> ClassificationResult:
        cfg = self._config
        context = retrieval.candidates[: cfg.context_candidates]
        rulings = retrieval.rulings[: cfg.context_rulings]
        try:
            raw = await llm.generate_structured(
                system=SYSTEM_PROMPT.format(max_candidates=answer_k),
                user=build_user_prompt(retrieval.query, context, rulings),
                schema=LLM_OUTPUT_SCHEMA,
                schema_name="hts_classification",
            )
        except LlmProviderError as exc:
            logger.warning("LLM call failed, falling back to extractive answer: %s", exc)
            result = _extractive(retrieval, decision, answer_k)
            result.warnings.append(
                f"{LLM_FALLBACK_WARNING} ({exc}); returned an extractive answer."
            )
            return result

        if raw.refused:
            return _abstained(
                retrieval, decision, f"LLM declined: {raw.refusal_reason}", llm=llm.label
            )
        try:
            answer = LlmClassification.model_validate_json(raw.text or "")
        except ValidationError as exc:
            logger.warning("LLM output failed validation: %s", exc)
            return _abstained(
                retrieval, decision, "LLM output failed schema validation", llm=llm.label
            )
        if answer.abstained:
            return _abstained(
                retrieval, decision, f"LLM abstained: {answer.abstain_reason}", llm=llm.label
            )

        outcome = apply_citation_guard(
            answer,
            {c.line.code: c for c in context},
            frozenset([c.chunk_id for c in context] + [r.chunk_id for r in rulings]),
        )
        warnings = [f"Rejected by citation guard: {r}" for r in outcome.rejected]
        if outcome.dropped_citations:
            warnings.append(f"Dropped unknown citations: {', '.join(outcome.dropped_citations)}")
        if not outcome.kept:
            result = _abstained(
                retrieval,
                decision,
                "no LLM candidate survived the citation guard",
                llm=llm.label,
            )
            result.warnings.extend(warnings)
            return result

        kept = outcome.kept[:answer_k]
        return ClassificationResult(
            query=retrieval.query,
            candidates=[_candidate_out(g.candidate, g.rationale, g.citations) for g in kept],
            confidence=round(min(answer.confidence, decision.confidence), 4),
            abstained=False,
            citations=_citations(kept, context, retrieval.rulings),
            evidence=decision.signals,
            answer_mode="llm",
            llm=llm.label,
            warnings=warnings,
        )


def _rationale(candidate: Candidate) -> str:
    parts = [
        f"{_RETRIEVER_LABELS.get(name, name)} rank {rank}"
        for name, rank in sorted(candidate.ranks.items(), key=lambda kv: kv[1])
    ]
    text = f"Retrieved by {', '.join(parts)} (RRF score {candidate.fused_score:.4f})."
    if candidate.supporting_rulings:
        text += f" Cited by ruling(s): {', '.join(candidate.supporting_rulings)}."
    return text


def _candidate_out(candidate: Candidate, rationale: str, citations: Sequence[str]) -> CandidateOut:
    line = candidate.line
    return CandidateOut(
        hts_code=line.code,
        level=line.level.value,
        description=line.description,
        breadcrumb=line.breadcrumb,
        general_rate=line.general_rate,
        rate_inherited_from=line.rate_source_code if line.rate_inherited else None,
        rationale=rationale,
        citations=list(citations),
        retrieval=RetrievalTrace(
            fused_score=round(candidate.fused_score, 6),
            ranks=candidate.ranks,
            scores={name: round(score, 4) for name, score in candidate.scores.items()},
        ),
    )


def _citations(
    kept: Sequence[GuardedCandidate],
    context: Sequence[Candidate],
    rulings: Sequence[RulingCandidate],
) -> list[Citation]:
    """Resolve every chunk id cited by a kept candidate to its source text.

    ``context`` holds the HTS lines the answer could cite (the guard has already removed any
    id that was not retrieved), so a candidate citing a *neighbouring* tariff line gets that
    line's text too, and ``result.citations`` covers every id in ``candidate.citations``.
    """
    hts_by_id = {c.chunk_id: c.line for c in context}
    for g in kept:
        hts_by_id.setdefault(g.candidate.chunk_id, g.candidate.line)
    by_ruling = {r.chunk_id: r.ruling for r in rulings}
    out: dict[str, Citation] = {}
    for g in kept:
        for chunk_id in g.citations:
            if chunk_id in out:
                continue
            if chunk_id in hts_by_id:
                line = hts_by_id[chunk_id]
                out[chunk_id] = Citation(
                    chunk_id=chunk_id, kind="hts_line", reference=line.code, text=line.breadcrumb
                )
            elif chunk_id in by_ruling:
                ruling = by_ruling[chunk_id]
                out[chunk_id] = Citation(
                    chunk_id=chunk_id,
                    kind="ruling",
                    reference=ruling.ruling_number,
                    text=ruling.excerpt or ruling.product_description,
                    url=ruling.url or None,
                )
    return list(out.values())


def _extractive(
    retrieval: RetrievalResult, decision: GateDecision, answer_k: int
) -> ClassificationResult:
    """No-LLM answer: the top fused lines, each citing its own tariff text (and rulings)."""
    ruling_ids = {r.ruling.ruling_number: r.chunk_id for r in retrieval.rulings}
    kept = [
        GuardedCandidate(
            c,
            _rationale(c),
            (c.chunk_id, *(ruling_ids[n] for n in c.supporting_rulings if n in ruling_ids)),
        )
        for c in retrieval.candidates[:answer_k]
    ]
    return ClassificationResult(
        query=retrieval.query,
        candidates=[_candidate_out(g.candidate, g.rationale, g.citations) for g in kept],
        confidence=decision.confidence,
        abstained=False,
        citations=_citations(kept, retrieval.candidates, retrieval.rulings),
        evidence=decision.signals,
        answer_mode="extractive",
    )


def _abstained(
    retrieval: RetrievalResult, decision: GateDecision, reason: str, *, llm: str | None = None
) -> ClassificationResult:
    return ClassificationResult(
        query=retrieval.query,
        candidates=[],
        confidence=0.0,
        abstained=True,
        abstain_reason=reason,
        citations=[],
        evidence=decision.signals,
        answer_mode="llm" if llm else "extractive",
        llm=llm,
    )
