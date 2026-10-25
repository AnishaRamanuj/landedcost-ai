import pytest

from landedcost.classify.gate import EvidenceGate, GateConfig
from landedcost.classify.guard import apply_citation_guard
from landedcost.classify.models import LlmCandidate, LlmClassification
from landedcost.hts.models import HtsLine
from landedcost.retrieval.hybrid import (
    FTS,
    RULINGS,
    VECTOR,
    Candidate,
    RetrievalMode,
    RetrievalResult,
)


def cand(line: HtsLine, ranks: dict[str, int], k: int = 60) -> Candidate:
    return Candidate(line, sum(1 / (k + r) for r in ranks.values()), ranks, {})


def result(
    candidates: list[Candidate],
    *,
    query: str = "lithium battery",
    fts_hits: int = 10,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    with_rulings: bool = False,
) -> RetrievalResult:
    hits = {VECTOR: 10, FTS: fts_hits}
    run = mode.retrievers + ((RULINGS,) if with_rulings else ())
    return RetrievalResult(
        query=query,
        mode=mode,
        rrf_k=60,
        candidates=candidates,
        hits_per_retriever=hits,
        retrievers_run=run,
    )


@pytest.fixture
def gate() -> EvidenceGate:
    return EvidenceGate(GateConfig(min_support=0.9, min_heading_consensus=0.3, window=10))


def test_strong_agreeing_evidence_passes(
    gate: EvidenceGate, lines_by_code: dict[str, HtsLine]
) -> None:
    decision = gate.evaluate(
        result(
            [
                cand(lines_by_code["8507.60.00"], {VECTOR: 1, FTS: 1}),
                cand(lines_by_code["8507.60.00.90"], {VECTOR: 2, FTS: 3}),
                cand(lines_by_code["0409.00.00"], {VECTOR: 3}),
            ]
        )
    )
    assert not decision.abstain
    assert decision.signals.support == 1.0
    assert decision.signals.heading_consensus > 0.8
    assert decision.confidence == pytest.approx(
        decision.signals.support * decision.signals.heading_consensus, abs=1e-3
    )


def test_single_retriever_support_abstains(
    gate: EvidenceGate, lines_by_code: dict[str, HtsLine]
) -> None:
    decision = gate.evaluate(result([cand(lines_by_code["8507.60.00"], {VECTOR: 1})]))
    assert decision.abstain
    assert decision.signals.support == pytest.approx(0.5)
    assert "disagree" in (decision.reason or "")


def test_spread_evidence_abstains(gate: EvidenceGate, lines_by_code: dict[str, HtsLine]) -> None:
    codes = ["8507.60.00", "0409.00.00", "1509.20", "6109.10.00", "7010.90", "0101.30.00.00"]
    candidates = [cand(lines_by_code[codes[0]], {VECTOR: 1, FTS: 1})] + [
        cand(lines_by_code[c], {VECTOR: i + 2, FTS: i + 2}) for i, c in enumerate(codes[1:])
    ]
    decision = gate.evaluate(result(candidates))
    assert decision.abstain
    assert "spread" in (decision.reason or "")


def test_no_keyword_match_abstains(gate: EvidenceGate, lines_by_code: dict[str, HtsLine]) -> None:
    decision = gate.evaluate(result([cand(lines_by_code["8507.60.00"], {VECTOR: 1})], fts_hits=0))
    assert decision.abstain
    assert "keyword" in (decision.reason or "")


def test_empty_query_and_no_candidates_abstain(gate: EvidenceGate) -> None:
    assert gate.evaluate(result([], query="the 123")).reason == (
        "the description has no informative words to search for"
    )
    assert gate.evaluate(result([])).reason == "no tariff lines were retrieved"


def test_rulings_list_does_not_dilute_support(
    gate: EvidenceGate, lines_by_code: dict[str, HtsLine]
) -> None:
    decision = gate.evaluate(
        result([cand(lines_by_code["8507.60.00"], {VECTOR: 1, FTS: 1})], with_rulings=True)
    )
    assert decision.signals.support == 1.0
    assert not decision.abstain


def test_single_mode_support_is_relative_to_that_retriever(
    gate: EvidenceGate, lines_by_code: dict[str, HtsLine]
) -> None:
    decision = gate.evaluate(
        result([cand(lines_by_code["0409.00.00"], {FTS: 1})], mode=RetrievalMode.FTS, query="honey")
    )
    assert decision.signals.support == 1.0


def test_support_uses_the_rrf_weights(
    gate: EvidenceGate, lines_by_code: dict[str, HtsLine]
) -> None:
    top = Candidate(lines_by_code["0409.00.00"], 1.0 / 61, {FTS: 1}, {})
    weighted = RetrievalResult(
        query="honey",
        mode=RetrievalMode.HYBRID,
        rrf_k=60,
        candidates=[top],
        hits_per_retriever={VECTOR: 10, FTS: 10},
        retrievers_run=(VECTOR, FTS),
        rrf_weights={VECTOR: 0.5, FTS: 1.0},
    )
    assert gate.evaluate(weighted).signals.support == pytest.approx(1 / 1.5, abs=1e-4)


# -- citation guard ----------------------------------------------------------------------


def llm_answer(*candidates: LlmCandidate) -> LlmClassification:
    return LlmClassification(
        candidates=list(candidates), confidence=0.8, abstained=False, abstain_reason=""
    )


def llm_cand(code: str, citations: list[str]) -> LlmCandidate:
    return LlmCandidate(
        hts_code=code,
        description="model text",
        general_rate="99%",
        rationale=" why ",
        citations=citations,
    )


def test_guard_keeps_grounded_candidates_and_normalises_codes(
    lines_by_code: dict[str, HtsLine],
) -> None:
    c = cand(lines_by_code["8507.60.00.90"], {VECTOR: 1, FTS: 1})
    outcome = apply_citation_guard(
        llm_answer(llm_cand("8507600090", ["hts:8507.60.00.90", "ruling:TEST-0001"])),
        {c.line.code: c},
        frozenset({"hts:8507.60.00.90", "ruling:TEST-0001"}),
    )
    assert [k.candidate.line.code for k in outcome.kept] == ["8507.60.00.90"]
    assert outcome.kept[0].citations == ("hts:8507.60.00.90", "ruling:TEST-0001")
    assert outcome.kept[0].rationale == "why"
    assert outcome.rejected == []


def test_guard_rejects_hallucinated_codes_and_citations(lines_by_code: dict[str, HtsLine]) -> None:
    c = cand(lines_by_code["8507.60.00.90"], {VECTOR: 1, FTS: 1})
    context = {c.line.code: c}
    ids = frozenset({"hts:8507.60.00.90"})
    outcome = apply_citation_guard(
        llm_answer(
            llm_cand("8506.50.00", ["hts:8507.60.00.90"]),  # code not retrieved
            llm_cand("not-a-code", ["hts:8507.60.00.90"]),
            llm_cand("8507.60.00.90", ["hts:9999.99.99.99"]),  # no valid citation
        ),
        context,
        ids,
    )
    assert outcome.kept == []
    assert len(outcome.rejected) == 3
    assert "not among the retrieved" in outcome.rejected[0]
    assert "not a valid HTS code" in outcome.rejected[1]
    assert "no citation" in outcome.rejected[2]
    assert outcome.dropped_citations == ["hts:9999.99.99.99"]


def test_guard_drops_unknown_citations_but_keeps_valid_ones_and_dedupes(
    lines_by_code: dict[str, HtsLine],
) -> None:
    c = cand(lines_by_code["0409.00.00"], {VECTOR: 1, FTS: 1})
    outcome = apply_citation_guard(
        llm_answer(
            llm_cand("0409.00.00", ["hts:0409.00.00", "made-up"]),
            llm_cand("0409.00.00", ["hts:0409.00.00"]),
        ),
        {c.line.code: c},
        frozenset({"hts:0409.00.00"}),
    )
    assert len(outcome.kept) == 1
    assert outcome.dropped_citations == ["made-up"]


def test_guard_rejects_candidate_that_only_cites_another_line(
    lines_by_code: dict[str, HtsLine],
) -> None:
    leaf = cand(lines_by_code["8507.60.00.90"], {VECTOR: 2, FTS: 2})
    parent = cand(lines_by_code["8507.60.00"], {VECTOR: 1, FTS: 1})
    outcome = apply_citation_guard(
        llm_answer(llm_cand("8507.60.00.90", ["hts:8507.60.00"])),
        {c.line.code: c for c in (leaf, parent)},
        frozenset({"hts:8507.60.00.90", "hts:8507.60.00"}),
    )
    assert outcome.kept == []
    assert outcome.rejected == [
        "8507.60.00.90: does not cite its own tariff line (hts:8507.60.00.90)"
    ]


def test_guard_puts_the_own_line_first_and_keeps_supporting_lines(
    lines_by_code: dict[str, HtsLine],
) -> None:
    leaf = cand(lines_by_code["8507.60.00.90"], {VECTOR: 2, FTS: 2})
    parent = cand(lines_by_code["8507.60.00"], {VECTOR: 1, FTS: 1})
    outcome = apply_citation_guard(
        llm_answer(llm_cand("8507.60.00.90", ["hts:8507.60.00", "hts:8507.60.00.90"])),
        {c.line.code: c for c in (leaf, parent)},
        frozenset({"hts:8507.60.00.90", "hts:8507.60.00"}),
    )
    assert outcome.kept[0].citations == ("hts:8507.60.00.90", "hts:8507.60.00")


def test_empty_index_abstains_with_an_ingest_hint(gate: EvidenceGate) -> None:
    empty = RetrievalResult(
        query="honey", mode=RetrievalMode.HYBRID, rrf_k=60, candidates=[], indexed_lines=0
    )
    assert "ingest-hts" in (gate.evaluate(empty).reason or "")
