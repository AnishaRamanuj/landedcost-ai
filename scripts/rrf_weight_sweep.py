"""Sweep the RRF weight of the vector retriever and cross-validate the choice.

Equal-weight RRF is the default because it needs no tuning. When one retriever is clearly
weaker (the offline hash embedder is lexical, so its "vector" leg mostly repeats what BM25
already found, with more noise), down-weighting it can help. This script measures that
honestly on the starter eval set:

* hybrid retrieval is run once per vector weight (full-text weight fixed at 1.0) over the
  answerable items, and heading/subheading Recall@k and MRR@10 are reported for each;
* a two-fold split (even/odd item positions) picks the weight with the best heading MRR@10
  on one fold and reports it on the other, so the "tuned" number is not measured on the
  items it was tuned on;
* the configured abstention gate is re-applied for each weight, because the weights change
  the ``support`` signal too.

    uv run python scripts/rrf_weight_sweep.py --output docs/eval/rrf-weights.md

The index (embedder, store backend, HNSW settings) comes from the usual settings.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from landedcost.config import DEFAULT_EVAL_DATASET, Settings, resolve_data_path
from landedcost.container import Services, build_services
from landedcost.evals.dataset import EvalItem, load_eval_items
from landedcost.evals.metrics import mean_reciprocal_rank, recall_at_k
from landedcost.evals.runner import ItemOutcome, run_eval
from landedcost.retrieval.hybrid import (
    FTS,
    VECTOR,
    HybridRetriever,
    RetrievalMode,
    RetrieverConfig,
)

DEFAULT_WEIGHTS = (2.0, 1.0, 0.75, 0.5, 0.35, 0.25, 0.1)
KS = (1, 3, 5, 10)


@dataclass(frozen=True, slots=True)
class Row:
    label: str
    vector_weight: float | None
    outcomes: list[ItemOutcome]
    coverage: tuple[int, int]
    answered_correct: int


def _mrr(outcomes: Sequence[ItemOutcome], *, sub: bool = False) -> float:
    ranks = [o.subheading_rank if sub else o.heading_rank for o in outcomes]
    return mean_reciprocal_rank(ranks, max(KS))


def _fold(outcomes: Sequence[ItemOutcome], parity: int) -> list[ItemOutcome]:
    return [o for i, o in enumerate(outcomes) if i % 2 == parity]


async def _run(
    services: Services, items: list[EvalItem], mode: RetrievalMode, vector_weight: float | None
) -> Row:
    weights = dict(services.settings.rrf_weights)
    if vector_weight is not None:
        weights[VECTOR] = vector_weight
    retriever = HybridRetriever(
        services.store,
        services.embedder,
        RetrieverConfig(
            rrf_k=services.settings.rrf_k,
            candidate_pool=services.settings.retrieval_candidate_pool,
            rrf_weights=weights,
        ),
    )
    report = await run_eval(retriever, services.gate, items, modes=[mode], ks=KS)
    gate = report.abstention
    assert gate is not None
    label = f"hybrid, vector weight {vector_weight:g}" if vector_weight is not None else mode.value
    return Row(
        label=label,
        vector_weight=vector_weight,
        outcomes=report.modes[0].outcomes,
        coverage=(gate.answered, gate.answerable),
        answered_correct=gate.answered_correct_at_1,
    )


def _best(rows: Sequence[Row], parity: int) -> Row:
    # Highest heading MRR@10 on the fold; ties go to the weight closest to 1.0 (no tuning).
    return max(
        rows,
        key=lambda r: (_mrr(_fold(r.outcomes, parity)), -abs((r.vector_weight or 1.0) - 1.0)),
    )


def render(rows: Sequence[Row], baselines: Sequence[Row], meta: dict[str, str]) -> str:
    lines = ["## RRF weight sweep", ""]
    lines += [f"- **{k}:** {v}" for k, v in meta.items()]
    header = (
        "| Retrieval | Heading R@1 | Heading R@10 | Heading MRR@10 | Subheading MRR@10 "
        "| Gate coverage | Gate acc@1 when answering |"
    )
    lines += ["", header, "|---|---|---|---|---|---|---|"]
    for row in [*baselines, *rows]:
        heading = [o.heading_rank for o in row.outcomes]
        answered, answerable = row.coverage
        acc = row.answered_correct / answered if answered else 0.0
        lines.append(
            f"| {row.label} | {recall_at_k(heading, 1):.3f} | {recall_at_k(heading, 10):.3f} "
            f"| {_mrr(row.outcomes):.3f} | {_mrr(row.outcomes, sub=True):.3f} "
            f"| {answered}/{answerable} | {row.answered_correct}/{answered} ({acc:.3f}) |"
        )
    lines += [
        "",
        "### Two-fold check (weight chosen on one half, measured on the other)",
        "",
        "| Tuned on | Chosen vector weight | Heading MRR@10 on the other half "
        "| Equal weights on the other half | Full-text only on the other half |",
        "|---|---|---|---|---|",
    ]
    equal = next(r for r in rows if r.vector_weight == 1.0)
    fts = next(r for r in baselines if r.label == RetrievalMode.FTS.value)
    for tune, test in ((0, 1), (1, 0)):
        best = _best(rows, tune)
        lines.append(
            f"| {'even' if tune == 0 else 'odd'} items | {best.vector_weight:g} "
            f"| {_mrr(_fold(best.outcomes, test)):.3f} "
            f"| {_mrr(_fold(equal.outcomes, test)):.3f} "
            f"| {_mrr(_fold(fts.outcomes, test)):.3f} |"
        )
    return "\n".join(lines) + "\n"


async def main(args: argparse.Namespace) -> str:
    settings = Settings()
    items = [i for i in load_eval_items(resolve_data_path(args.dataset)) if not i.expect_abstain]
    services = await build_services(settings)
    try:
        info = await services.store.index_info()
        baselines = [
            await _run(services, items, RetrievalMode.FTS, None),
            await _run(services, items, RetrievalMode.VECTOR, None),
        ]
        rows = [await _run(services, items, RetrievalMode.HYBRID, w) for w in args.weights]
    finally:
        await services.aclose()
    meta = {
        "Command": "uv run python scripts/rrf_weight_sweep.py --weights "
        + " ".join(f"{w:g}" for w in args.weights),
        "Dataset": f"{Path(args.dataset).name} ({len(items)} answerable items)",
        "HTS data": f"{info.hts_version} ({info.hts_lines} indexed lines)",
        "Embedder": services.embedder.name,
        "Fixed": f"{FTS} weight {settings.rrf_weight_fts:g}, RRF k={settings.rrf_k}, "
        f"gate min_support={settings.gate_min_support}, "
        f"min_heading_consensus={settings.gate_min_heading_consensus}",
        "Run at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }
    return render(rows, baselines, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_EVAL_DATASET)
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    if 1.0 not in arguments.weights:
        parser.error("--weights must include 1.0 (the equal-weight reference)")
    markdown = asyncio.run(main(arguments))
    print(markdown)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(markdown, encoding="utf-8")
