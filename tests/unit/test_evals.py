import json
from dataclasses import replace
from pathlib import Path

import pytest

from landedcost.classify.gate import EvidenceGate, GateConfig
from landedcost.classify.service import Classifier
from landedcost.embeddings.hashing import HashEmbedder
from landedcost.evals.dataset import EvalItem, items_from_rulings, load_eval_items
from landedcost.evals.metrics import first_hit_rank, mean_reciprocal_rank, recall_at_k
from landedcost.evals.runner import SUPPORT_SWEEP, render_markdown, run_answer_eval, run_eval
from landedcost.hts.loader import load_lines
from landedcost.llm.base import LlmProviderError
from landedcost.retrieval.hybrid import HybridRetriever, RetrievalMode
from landedcost.store.memory import InMemoryStore
from tests.fakes import FakeLlm

STARTER = Path("data/eval/starter_eval.jsonl")


def test_first_hit_rank_uses_code_prefixes() -> None:
    codes = ["8506.50.00", "8507.60.00.90", "8507.20"]
    assert first_hit_rank(codes, ["8507"]) == 2
    assert first_hit_rank(codes, ["8507.60"]) == 2
    assert first_hit_rank(codes, ["8507.20"]) == 3
    assert first_hit_rank(codes, ["8509"]) is None
    assert first_hit_rank(["85071"], ["8507"]) is None  # prefix must end on a code boundary


def test_recall_and_mrr() -> None:
    ranks = [1, 3, None, 12]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.5
    assert recall_at_k(ranks, 20) == 0.75
    assert mean_reciprocal_rank(ranks, depth=10) == pytest.approx((1 + 1 / 3) / 4)
    assert recall_at_k([], 5) == 0.0
    assert mean_reciprocal_rank([], 5) == 0.0


def test_starter_set_shape() -> None:
    items = load_eval_items(STARTER)
    assert 40 <= len(items) <= 60
    answerable = [i for i in items if not i.expect_abstain]
    assert len(answerable) >= 40
    assert all(i.source == "hand-labeled against HTS 2026-09-04" for i in answerable)
    assert any(i.expect_abstain for i in items)


def test_every_gold_code_exists_in_the_shipped_hts() -> None:
    _, parsed = load_lines(Path("data/hts/hts_2026-09-04.json.gz"))
    codes = [line.code for line in parsed.lines]
    missing = [
        (item.id, gold)
        for item in load_eval_items(STARTER)
        for gold in item.gold_headings + item.gold_subheadings
        if not any(c == gold or c.startswith(gold + ".") for c in codes)
    ]
    assert missing == []


def test_descriptions_are_not_copied_from_tariff_text() -> None:
    _, parsed = load_lines(Path("data/hts/hts_2026-09-04.json.gz"))
    tariff_text = {line.description.lower() for line in parsed.lines}
    copied = [i.id for i in load_eval_items(STARTER) if i.description.lower() in tariff_text]
    assert copied == []


@pytest.mark.parametrize(
    ("record", "match"),
    [
        ({"id": "a", "description": "x", "gold_headings": ["8507.60"]}, "not a heading"),
        ({"id": "a", "description": "x", "gold_headings": []}, "needs gold_headings"),
        (
            {
                "id": "a",
                "description": "x",
                "gold_headings": ["8507"],
                "gold_subheadings": ["8506.50"],
            },
            "not under a gold heading",
        ),
        ({"description": "x"}, "id"),
    ],
)
def test_dataset_validation(tmp_path: Path, record: dict[str, object], match: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_eval_items(path)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    record = {"id": "a", "description": "x", "gold_headings": ["8507"]}
    path = tmp_path / "dup.jsonl"
    path.write_text(json.dumps(record) + "\n\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_eval_items(path)


async def test_ruling_items_hide_their_own_ruling(store_with_rulings: InMemoryStore) -> None:
    items = items_from_rulings(await store_with_rulings.list_rulings())
    first = next(i for i in items if i.id == "ruling:TEST-0001")
    assert first.gold_headings == ("8507",)
    assert first.gold_subheadings == ("8507.60",)
    assert first.exclude_rulings == ("TEST-0001",)


async def test_run_eval_and_render(memory_store: InMemoryStore, embedder: HashEmbedder) -> None:
    items = [
        EvalItem("honey", "natural honey in jars", ("0409",), ("0409.00",)),
        EvalItem("battery", "lithium-ion battery pack", ("8507",), ("8507.60",)),
        EvalItem("tee", "cotton t-shirt", ("6109",)),
        EvalItem("oos", "zzqx vvbn", (), expect_abstain=True),
    ]
    report = await run_eval(
        HybridRetriever(memory_store, embedder),
        EvidenceGate(GateConfig()),
        items,
        modes=[RetrievalMode.VECTOR, RetrievalMode.FTS, RetrievalMode.HYBRID],
        ks=[3, 1],
    )
    assert report.ks == (1, 3)
    hybrid = report.modes[-1]
    assert hybrid.heading.n == 3
    assert hybrid.subheading.n == 2
    assert hybrid.heading.recall[3] == 1.0
    assert report.abstention is not None
    assert report.abstention.unanswerable == 1
    assert report.abstention.unanswerable_abstained == 1
    assert [s.min_support for s in report.abstention_sweep] == list(SUPPORT_SWEEP)
    markdown = render_markdown(report, title="T", meta={"Embedder": embedder.name})
    assert "| Mode | Level | n | Recall@1 | Recall@3 | MRR@3 |" in markdown
    assert "| hybrid | heading (4) | 3 |" in markdown
    assert "**Embedder:** hash-v2-c35" in markdown
    assert "Out-of-scope items abstained | 1/1 (1.000)" in markdown


async def test_extractive_answer_eval_agrees_with_the_gate_table(
    store_with_rulings: InMemoryStore, embedder: HashEmbedder
) -> None:
    items = [
        EvalItem("honey", "natural honey", ("0409",), ("0409.00",)),
        EvalItem("battery", "lithium-ion battery pack for scooters", ("8507",), ("8507.60",)),
        EvalItem("tee", "cotton t-shirt", ("6109",)),
        EvalItem("wrong", "lithium-ion battery pack", ("0409",)),
        EvalItem("oos", "zzqx vvbn", (), expect_abstain=True),
        *items_from_rulings(await store_with_rulings.list_rulings()),
    ]
    retriever = HybridRetriever(store_with_rulings, embedder)
    gate = EvidenceGate(GateConfig(min_support=0.6, min_heading_consensus=0.2))
    report = await run_eval(retriever, gate, items, modes=[RetrievalMode.HYBRID], ks=[1])
    answers = await run_answer_eval(Classifier(retriever, gate), items)
    assert report.abstention is not None
    assert answers.answer_mode == "extractive (no LLM)"
    assert (answers.answerable, answers.answered, answers.heading_correct) == (
        report.abstention.answerable,
        report.abstention.answered,
        report.abstention.answered_correct_at_1,
    )
    assert answers.unanswerable_abstained == report.abstention.unanswerable_abstained == 1
    assert answers.llm_fallbacks == 0
    wrong = next(o for o in answers.outcomes if o.item_id == "wrong")
    assert not wrong.heading_correct
    assert "### Final answers" in render_markdown(
        replace(report, answers=answers), title="T", meta={}
    )


async def test_answer_eval_counts_llm_fallbacks(
    memory_store: InMemoryStore, embedder: HashEmbedder
) -> None:
    retriever = HybridRetriever(memory_store, embedder)
    gate = EvidenceGate(GateConfig(min_support=0.6, min_heading_consensus=0.2))
    llm = FakeLlm(error=LlmProviderError("timed out"))
    items = [EvalItem("battery", "lithium-ion battery pack", ("8507",), ("8507.60",))]
    answers = await run_answer_eval(Classifier(retriever, gate, llm), items)
    assert answers.answer_mode == "llm (fake:test-model)"
    assert answers.llm_fallbacks == 1
    assert answers.subheading_correct == answers.subheading_answered == 1
