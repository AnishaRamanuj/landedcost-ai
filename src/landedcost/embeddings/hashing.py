"""Deterministic, offline feature-hashing embedder.

This is a *lexical* baseline, not a semantic model: it projects word unigrams, word bigrams
and character n-grams into a fixed number of signed buckets (the "hashing trick"). Character
n-grams make it tolerant of morphology and spelling variants ("batteries" / "battery",
"lithium-ion" / "lithium ion"), but it cannot know that "laptop" and "automatic data
processing machine" mean the same thing. It exists so the whole pipeline, its tests and CI
run without network access or model downloads; swap in ``openai``/``azure_openai``/
``fastembed`` for semantic retrieval.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from functools import lru_cache
from itertools import pairwise

import numpy as np

from landedcost.embeddings.base import Matrix, Vector, l2_normalize
from landedcost.text import stem, words

_WORD_WEIGHT = 1.0
_BIGRAM_WEIGHT = 0.5
_CHAR_WEIGHT = 1.0  # spread across all n-grams of a word


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = zlib.crc32(feature.encode("utf-8"))
    return h % dim, (1.0 if (h >> 31) & 1 else -1.0)


class HashEmbedder:
    """Signed feature hashing over words, word bigrams and character n-grams."""

    def __init__(self, dim: int = 384, char_ngram_range: tuple[int, int] = (3, 5)) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        lo, hi = char_ngram_range
        if not 1 <= lo <= hi:
            raise ValueError("invalid char_ngram_range")
        self._dim = dim
        self._ngram_range = (lo, hi)
        self._word_features = lru_cache(maxsize=200_000)(self._compute_word_features)

    @property
    def name(self) -> str:
        lo, hi = self._ngram_range
        return f"hash-v2-c{lo}{hi}"

    @property
    def dim(self) -> int:
        return self._dim

    def _compute_word_features(self, word: str) -> tuple[tuple[int, float], ...]:
        # Whole-word features use the stem so "batteries" and "battery" share a bucket;
        # character n-grams use the surface form and add fuzzier subword overlap.
        features = [(*_bucket(f"w:{stem(word)}", self._dim), _WORD_WEIGHT)]
        padded = f"<{word}>"
        grams = [
            padded[i : i + n]
            for n in range(self._ngram_range[0], self._ngram_range[1] + 1)
            for i in range(len(padded) - n + 1)
        ]
        if grams:
            per_gram = _CHAR_WEIGHT / len(grams)
            features.extend((*_bucket(f"c:{g}", self._dim), per_gram) for g in grams)
        return tuple((idx, sign * weight) for idx, sign, weight in features)

    def embed_one(self, text: str) -> Vector:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = words(text)
        for token in tokens:
            for idx, value in self._word_features(token):
                vec[idx] += value
        for left, right in pairwise(tokens):
            idx, sign = _bucket(f"b:{stem(left)}_{stem(right)}", self._dim)
            vec[idx] += sign * _BIGRAM_WEIGHT
        normalized: Vector = l2_normalize(vec)[0]
        return normalized

    def embed_many(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack([self.embed_one(t) for t in texts])

    async def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return self.embed_many(texts)

    async def embed_query(self, text: str) -> Vector:
        return self.embed_one(text)
