import pytest

from landedcost.retrieval.rrf import max_rrf_score, reciprocal_rank_fusion
from landedcost.store.bm25 import Bm25Index


def test_bm25_rare_terms_outweigh_common_terms() -> None:
    docs = [
        ["glass", "jar", "glass", "bottle"],
        ["natural", "honey"],
        ["glass", "sheet"],
        ["glass", "fiber"],
    ]
    index = Bm25Index(docs)
    ranked = index.search(["natural", "honey", "glass", "jar"], limit=4)
    assert ranked[0][0] == 1  # "honey" is rare, so it beats two matches of common "glass"
    assert {doc for doc, _ in ranked} == {0, 1, 2, 3}  # OR semantics


def test_bm25_unknown_terms_and_limit() -> None:
    index = Bm25Index([["a"], ["b"]])
    assert index.search(["zzz"], limit=5) == []
    assert len(index.search(["a", "b"], limit=1)) == 1
    assert len(Bm25Index([])) == 0


def test_bm25_prefers_shorter_documents_for_equal_matches() -> None:
    index = Bm25Index([["battery", "x", "y", "z", "w"], ["battery"]])
    assert index.search(["battery"], limit=2)[0][0] == 1


def test_rrf_scores_follow_the_formula() -> None:
    fused = reciprocal_rank_fusion({"vector": ["a", "b", "c"], "fts": ["b", "a", "d"]}, k=60)
    scores = {f.id: f.score for f in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 63)
    by_id = {f.id: f for f in fused}
    assert by_id["a"].ranks == {"vector": 1, "fts": 2}
    assert by_id["d"].ranks == {"fts": 3}


def test_rrf_ties_are_deterministic() -> None:
    fused = reciprocal_rank_fusion({"vector": ["a", "b", "c"], "fts": ["b", "a", "d"]}, k=60)
    # a and b tie on score and best rank -> ordered by id; c and d tie -> ordered by id
    assert [f.id for f in fused] == ["a", "b", "c", "d"]


def test_rrf_counts_duplicates_once_and_applies_weights() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": ["a", "a", "b"], "fts": ["b"]}, k=0, weights={"fts": 2.0}
    )
    scores = {f.id: f.score for f in fused}
    assert scores["a"] == pytest.approx(1.0)
    assert scores["b"] == pytest.approx(1 / 2 + 2.0)
    assert fused[0].id == "b"


def test_rrf_edge_cases() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"vector": []}) == []
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion({"vector": ["a"]}, k=-1)
    assert max_rrf_score(["vector", "fts"], 60) == pytest.approx(2 / 61)
