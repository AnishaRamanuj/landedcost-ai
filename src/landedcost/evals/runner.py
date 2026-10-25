"""Run retrieval and abstention evals and render markdown reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from landedcost.classify.gate import EvidenceGate
from landedcost.classify.service import LLM_FALLBACK_WARNING, Classifier
from landedcost.evals.dataset import EvalItem
from landedcost.evals.metrics import first_hit_rank, mean_reciprocal_rank, recall_at_k
from landedcost.retrieval.hybrid import HybridRetriever, RetrievalMode, RetrievalResult

SUPPORT_SWEEP: tuple[float, ...] = (0.0, 0.6, 0.8, 0.9, 0.95)
"""``min_support`` values reported in the coverage/accuracy table (other settings fixed)."""


@dataclass(frozen=True, slots=True)
class ItemOutcome:
    item_id: str
    mode: RetrievalMode
    heading_rank: int | None
    subheading_rank: int | None
    top_codes: tuple[str, ...]
    abstained: bool
    abstain_reason: str | None


@dataclass(frozen=True, slots=True)
class LevelMetrics:
    n: int
    recall: dict[int, float]
    mrr: float


@dataclass(frozen=True, slots=True)
class ModeReport:
    mode: RetrievalMode
    heading: LevelMetrics
    subheading: LevelMetrics
    outcomes: list[ItemOutcome] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AbstentionReport:
    """Gate behaviour at one setting: coverage, selective accuracy, out-of-scope handling."""

    min_support: float
    min_heading_consensus: float
    answerable: int
    answered: int
    answered_correct_at_1: int
    correct_at_1_overall: int
    unanswerable: int
    unanswerable_abstained: int


@dataclass(frozen=True, slots=True)
class AnswerOutcome:
    item_id: str
    abstained: bool
    top_code: str | None
    heading_correct: bool
    subheading_correct: bool | None
    """``None`` when the item has no gold subheading."""
    llm_fallback: bool
    abstain_reason: str | None


@dataclass(frozen=True, slots=True)
class AnswerReport:
    """End-to-end quality of the classifier's final answers (gate + LLM or extractive)."""

    answer_mode: str
    answerable: int
    answered: int
    heading_correct: int
    subheading_answered: int
    subheading_correct: int
    unanswerable: int
    unanswerable_abstained: int
    llm_fallbacks: int
    outcomes: list[AnswerOutcome] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvalReport:
    ks: tuple[int, ...]
    modes: list[ModeReport]
    abstention: AbstentionReport | None = None
    abstention_sweep: list[AbstentionReport] = field(default_factory=list)
    answers: AnswerReport | None = None


GateInput = tuple[EvalItem, RetrievalResult, int | None]


def _level_metrics(ranks: Sequence[int | None], ks: Sequence[int]) -> LevelMetrics:
    return LevelMetrics(
        n=len(ranks),
        recall={k: recall_at_k(ranks, k) for k in ks},
        mrr=mean_reciprocal_rank(ranks, max(ks)),
    )


def _abstention(gate: EvidenceGate, inputs: Sequence[GateInput]) -> AbstentionReport:
    answerable = [(r, rank) for item, r, rank in inputs if not item.expect_abstain]
    unanswerable = [r for item, r, _ in inputs if item.expect_abstain]
    answered = [rank for r, rank in answerable if not gate.evaluate(r).abstain]
    return AbstentionReport(
        min_support=gate.config.min_support,
        min_heading_consensus=gate.config.min_heading_consensus,
        answerable=len(answerable),
        answered=len(answered),
        answered_correct_at_1=sum(1 for rank in answered if rank == 1),
        correct_at_1_overall=sum(1 for _, rank in answerable if rank == 1),
        unanswerable=len(unanswerable),
        unanswerable_abstained=sum(1 for r in unanswerable if gate.evaluate(r).abstain),
    )


async def run_eval(
    retriever: HybridRetriever,
    gate: EvidenceGate,
    items: Sequence[EvalItem],
    *,
    modes: Sequence[RetrievalMode],
    ks: Sequence[int],
) -> EvalReport:
    """Evaluate each mode over ``items``.

    Recall@k and MRR are computed on answerable items only. The abstention tables use the
    hybrid run (or the first mode if hybrid is not requested) and include items marked
    ``expect_abstain`` (descriptions of services or non-goods that have no HTS code).
    """
    ks = tuple(sorted(set(ks)))
    depth = max(ks)
    gate_mode = RetrievalMode.HYBRID if RetrievalMode.HYBRID in modes else modes[0]
    gate_inputs: list[GateInput] = []
    by_id = {i.id: i for i in items}
    reports: list[ModeReport] = []
    for mode in modes:
        outcomes: list[ItemOutcome] = []
        for item in items:
            result = await retriever.retrieve(
                item.description,
                mode=mode,
                top_k=max(depth, gate.config.window),
                exclude_rulings=item.exclude_rulings,
            )
            codes = tuple(c.line.code for c in result.candidates)[:depth]
            heading_rank = first_hit_rank(codes, item.gold_headings) if item.gold_headings else None
            sub_rank = (
                first_hit_rank(codes, item.gold_subheadings) if item.gold_subheadings else None
            )
            decision = gate.evaluate(result)
            if mode is gate_mode:
                gate_inputs.append((item, result, heading_rank))
            outcomes.append(
                ItemOutcome(
                    item.id, mode, heading_rank, sub_rank, codes, decision.abstain, decision.reason
                )
            )
        answerable = [o for o in outcomes if not by_id[o.item_id].expect_abstain]
        with_sub = [o for o in answerable if by_id[o.item_id].gold_subheadings]
        reports.append(
            ModeReport(
                mode=mode,
                heading=_level_metrics([o.heading_rank for o in answerable], ks),
                subheading=_level_metrics([o.subheading_rank for o in with_sub], ks),
                outcomes=outcomes,
            )
        )
    if not gate_inputs:
        return EvalReport(ks=ks, modes=reports)
    sweep = [
        _abstention(EvidenceGate(replace(gate.config, min_support=t)), gate_inputs)
        for t in SUPPORT_SWEEP
    ]
    return EvalReport(
        ks=ks,
        modes=reports,
        abstention=_abstention(gate, gate_inputs),
        abstention_sweep=sweep,
    )


def _in_gold(code: str, gold: Sequence[str]) -> bool:
    return first_hit_rank([code], gold) == 1


async def run_answer_eval(classifier: Classifier, items: Sequence[EvalItem]) -> AnswerReport:
    """Score the classifier's first answer per item, exactly as the API would return it.

    With ``LLM_PROVIDER=none`` this measures the extractive answer (and must agree with the
    gate table); with an LLM configured it measures the model's pick after the citation
    guard, and counts provider failures that fell back to the extractive answer.
    """
    outcomes: list[AnswerOutcome] = []
    for item in items:
        result = await classifier.classify(item.description, exclude_rulings=item.exclude_rulings)
        top = result.candidates[0].hts_code if result.candidates else None
        outcomes.append(
            AnswerOutcome(
                item_id=item.id,
                abstained=result.abstained,
                top_code=top,
                heading_correct=top is not None and _in_gold(top, item.gold_headings),
                subheading_correct=(
                    top is not None and _in_gold(top, item.gold_subheadings)
                    if item.gold_subheadings
                    else None
                ),
                llm_fallback=any(w.startswith(LLM_FALLBACK_WARNING) for w in result.warnings),
                abstain_reason=result.abstain_reason,
            )
        )
    by_id = {i.id: i for i in items}
    answerable = [o for o in outcomes if not by_id[o.item_id].expect_abstain]
    answered = [o for o in answerable if not o.abstained]
    with_sub = [o for o in answered if o.subheading_correct is not None]
    unanswerable = [o for o in outcomes if by_id[o.item_id].expect_abstain]
    label = classifier.llm_label
    return AnswerReport(
        answer_mode=f"llm ({label})" if label else "extractive (no LLM)",
        answerable=len(answerable),
        answered=len(answered),
        heading_correct=sum(o.heading_correct for o in answered),
        subheading_answered=len(with_sub),
        subheading_correct=sum(bool(o.subheading_correct) for o in with_sub),
        unanswerable=len(unanswerable),
        unanswerable_abstained=sum(o.abstained for o in unanswerable),
        llm_fallbacks=sum(o.llm_fallback for o in outcomes),
        outcomes=outcomes,
    )


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _ratio(num: int, den: int) -> str:
    return f"{num}/{den} ({_fmt(num / den if den else 0.0)})"


def render_markdown(report: EvalReport, *, title: str, meta: dict[str, str]) -> str:
    """Render the report as GitHub-flavoured markdown."""
    ks = report.ks
    lines = [f"## {title}", ""]
    lines += [f"- **{key}:** {value}" for key, value in meta.items()]
    lines += ["", "### Retrieval", ""]
    header = "| Mode | Level | n | " + " | ".join(f"Recall@{k}" for k in ks) + f" | MRR@{max(ks)} |"
    lines += [header, "|" + "---|" * (4 + len(ks))]
    for mode_report in report.modes:
        for level_name, metrics in (
            ("heading (4)", mode_report.heading),
            ("subheading (6)", mode_report.subheading),
        ):
            cells = [mode_report.mode.value, level_name, str(metrics.n)]
            cells += [_fmt(metrics.recall[k]) for k in ks] + [_fmt(metrics.mrr)]
            lines.append("| " + " | ".join(cells) + " |")

    if report.abstention:
        a = report.abstention
        lines += [
            "",
            f"### Abstention gate (configured: min_support={a.min_support}, "
            f"min_heading_consensus={a.min_heading_consensus})",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Answerable items answered (coverage) | {_ratio(a.answered, a.answerable)} |",
            "| Heading accuracy@1 on answered items | "
            f"{_ratio(a.answered_correct_at_1, a.answered)} |",
            "| Heading accuracy@1 on all answerable items (no gate) | "
            f"{_ratio(a.correct_at_1_overall, a.answerable)} |",
            "| Out-of-scope items abstained | "
            f"{_ratio(a.unanswerable_abstained, a.unanswerable)} |",
        ]
    if report.abstention_sweep:
        lines += [
            "",
            "### Coverage vs. accuracy by min_support (other gate settings as configured)",
            "",
            "| min_support | Coverage | Heading acc@1 when answering | Out-of-scope abstained |",
            "|---|---|---|---|",
        ]
        for s in report.abstention_sweep:
            lines.append(
                f"| {s.min_support} | {_ratio(s.answered, s.answerable)} | "
                f"{_ratio(s.answered_correct_at_1, s.answered)} | "
                f"{_ratio(s.unanswerable_abstained, s.unanswerable)} |"
            )
    if report.answers:
        a2 = report.answers
        lines += [
            "",
            f"### Final answers (answer mode: {a2.answer_mode})",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Answerable items answered (coverage) | {_ratio(a2.answered, a2.answerable)} |",
            f"| Heading accuracy@1 on answered items | {_ratio(a2.heading_correct, a2.answered)} |",
            "| Subheading accuracy@1 on answered items | "
            f"{_ratio(a2.subheading_correct, a2.subheading_answered)} |",
            "| Out-of-scope items abstained | "
            f"{_ratio(a2.unanswerable_abstained, a2.unanswerable)} |",
            f"| LLM calls that fell back to the extractive answer | {a2.llm_fallbacks} |",
        ]
    return "\n".join(lines) + "\n"
