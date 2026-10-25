"""Evidence gate: decide whether retrieval found enough to answer at all.

Abstaining is the right output when evidence is weak: a wrong HTS code shifts duty, can
trigger penalties and is costly to unwind, while "I don't know, ask a broker" is cheap.
The gate runs *before* any LLM call, so a weak retrieval never reaches a model that might
confabulate a plausible-looking code.

Signals (all computed from the fused ranking, so they mean the same with any embedder):

* ``support``: the top candidate's (weighted) RRF score from the tariff-text retrievers
  (vector and/or full-text) divided by the best achievable score (ranked first by each of
  them). In hybrid mode it is high only when both retrievers rank that line near the top.
* ``heading_consensus``: of the fused score in the top ``window`` candidates, the share
  belonging to the top candidate's 4-digit heading. Low values mean the evidence is spread
  across unrelated headings.
* zero full-text hits in hybrid mode: no tariff text shares a single informative term with
  the description.
"""

from __future__ import annotations

from dataclasses import dataclass

from landedcost.classify.models import EvidenceSignals
from landedcost.retrieval.hybrid import FTS, RULINGS, VECTOR, RetrievalResult
from landedcost.retrieval.rrf import max_rrf_score
from landedcost.text import words


@dataclass(frozen=True, slots=True)
class GateConfig:
    min_support: float = 0.9
    min_heading_consensus: float = 0.3
    window: int = 10
    require_fts_hit: bool = True


@dataclass(frozen=True, slots=True)
class GateDecision:
    abstain: bool
    reason: str | None
    signals: EvidenceSignals
    confidence: float


def evidence_signals(result: RetrievalResult, window: int) -> EvidenceSignals:
    hits = result.hits_per_retriever
    if not result.candidates:
        return EvidenceSignals(
            support=0.0,
            heading_consensus=0.0,
            fts_hits=hits.get(FTS),
            vector_hits=hits.get(VECTOR),
            ruling_hits=len(result.rulings),
        )
    top = result.candidates[0]
    head = result.candidates[:window]
    total = sum(c.fused_score for c in head)
    agreeing = sum(c.fused_score for c in head if c.line.heading == top.line.heading)
    # Support uses only the tariff-text retrievers of the mode. Ruling votes can lift a
    # candidate to the top, but a missing ruling must not make strong text evidence look weak.
    retrievers = result.mode.retrievers
    weights = result.rrf_weights
    best = max_rrf_score(retrievers, result.rrf_k, weights)
    achieved = sum(
        weights.get(r, 1.0) / (result.rrf_k + top.ranks[r]) for r in retrievers if r in top.ranks
    )
    support = achieved / best if best else 0.0
    return EvidenceSignals(
        support=round(min(support, 1.0), 4),
        heading_consensus=round(agreeing / total if total else 0.0, 4),
        fts_hits=hits.get(FTS),
        vector_hits=hits.get(VECTOR),
        ruling_hits=len(result.rulings) if RULINGS in result.retrievers_run else 0,
    )


class EvidenceGate:
    def __init__(self, config: GateConfig | None = None) -> None:
        self._config = config or GateConfig()

    @property
    def config(self) -> GateConfig:
        return self._config

    def evaluate(self, result: RetrievalResult) -> GateDecision:
        cfg = self._config
        signals = evidence_signals(result, cfg.window)
        confidence = round(signals.support * signals.heading_consensus, 4)

        def abstain(reason: str) -> GateDecision:
            return GateDecision(True, reason, signals, confidence)

        if not words(result.query):
            return abstain("the description has no informative words to search for")
        if not result.candidates:
            if result.indexed_lines == 0:
                return abstain("the HTS index is empty; run `landedcost ingest-hts` first")
            return abstain("no tariff lines were retrieved")
        if cfg.require_fts_hit and FTS in result.retrievers_run and not signals.fts_hits:
            return abstain("no tariff text shares a keyword with the description")
        if signals.support < cfg.min_support:
            return abstain(
                f"retrievers disagree on the best match (support {signals.support:.2f} "
                f"< {cfg.min_support:.2f})"
            )
        if signals.heading_consensus < cfg.min_heading_consensus:
            return abstain(
                "evidence is spread across unrelated headings (consensus "
                f"{signals.heading_consensus:.2f} < {cfg.min_heading_consensus:.2f})"
            )
        return GateDecision(False, None, signals, confidence)
