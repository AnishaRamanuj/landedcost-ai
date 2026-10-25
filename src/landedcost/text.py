"""Tokenisation shared by the hash embedder, the in-memory BM25 index and query building."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

# fmt: off
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "s", "such", "than", "that", "the",
    "their", "these", "this", "those", "to", "was", "were", "which", "with", "without",
    "not", "nor",
    # tariff boilerplate
    "other", "thereof", "therefor", "whether", "including",
    # bare units of measure (their numbers are dropped, see words())
    "kg", "g", "lb", "lbs", "oz", "ml", "l", "cl", "cm", "mm", "m", "ft", "inch", "cu",
    "w", "v", "kw", "kva", "ah", "mah", "gb", "tb", "pcs", "pc",
})
# fmt: on
"""English function words, tariff boilerplate ("other", "thereof", "whether or not") that
appears in thousands of HTS lines, and bare units of measure. None carry product signal."""


def words(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens without stopwords or tokens containing digits.

    Keyword and hashed retrievers cannot reason about numeric ranges ("exceeding 1,500 cc
    but not exceeding 3,000 cc"); numbers and sizes like ``750``, ``36v`` or ``16gb`` only
    produce spurious matches (chemical names such as "2,4,6-..." are full of digits).
    """
    return [
        t
        for t in _TOKEN.findall(text.lower())
        if t not in STOPWORDS and not any(ch.isdigit() for ch in t)
    ]


def stem(token: str) -> str:
    """A deliberately tiny English plural stemmer (batteries -> battery, boxes -> box)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "sses", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def bm25_terms(text: str) -> list[str]:
    return [stem(t) for t in words(text)]


def fts_query(text: str) -> str:
    """Build a ``websearch_to_tsquery`` input that ORs the informative terms of ``text``.

    ``websearch_to_tsquery`` ANDs bare words, which on free-text product descriptions
    ("rechargeable lithium-ion battery pack for e-bikes") almost never matches a whole
    tariff line. OR-ing the terms lets the GIN index select every line sharing a term, and
    the ranking (BM25 by default, ``ts_rank_cd`` via ``FTS_RANKING``) rewards lines that
    match more, and rarer, terms: a classic keyword retriever. Postgres still applies its own
    English stemming and stopword list on top. ``words`` already drops "or", "and" and "not"
    (stopwords), so no term can be read as a ``websearch_to_tsquery`` operator.
    """
    return " or ".join(words(text))
