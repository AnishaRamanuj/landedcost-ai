# Retrieval tuning notes

The measurements behind the retrieval defaults. Every table comes from the command shown,
run against the full HTS snapshot (2026-09-04, 26,246 indexed lines) in PostgreSQL 16 +
pgvector 0.6.0 with the offline `hash-v2-c35` embedder, unless stated otherwise. The eval set
is `data/eval/starter_eval.jsonl` (54 answerable items, 6 out-of-scope items), which is small
and was also used to choose these defaults; treat the numbers as indicative.

## BM25 vs. Postgres `ts_rank_cd`

`FTS_RANKING=ts_rank_cd uv run landedcost eval --output docs/eval/ts-rank-cd-baseline.md`
against the default BM25 run ([`docs/eval/hash-embedder.md`](eval/hash-embedder.md);
full baseline report: [`docs/eval/ts-rank-cd-baseline.md`](eval/ts-rank-cd-baseline.md)):

| Full-text ranking | Heading R@1 | Heading R@10 | Heading MRR@10 | Subheading R@1 | Subheading MRR@10 |
|---|---|---|---|---|---|
| `ts_rank_cd` (Postgres built-in) | 0.241 | 0.593 | 0.338 | 0.148 | 0.227 |
| BM25 over the same `tsvector` (default) | 0.574 | 0.778 | 0.645 | 0.463 | 0.545 |

`ts_rank_cd` has no inverse document frequency: "natural honey in 500 g glass jars" ranked
glass containers (7010) above honey (0409) because "glass" and "jar" outnumber "honey". BM25
weights rare terms. Ingest refreshes a lexeme document-frequency table from the computed
`tsvector` column (`ts_stat`), and the query scores the GIN-filtered candidates in SQL.

The same baseline run also shows when fusion pays off: with the weaker `ts_rank_cd` leg,
hybrid heading MRR@10 is **0.502**, above both the vector leg (0.402) and the full-text leg
(0.338), because the two legs make different mistakes. With BM25 the full-text leg is strong
enough that equal-weight fusion with the lexical hash-vector leg pulls it down (next section).

## RRF weights

`uv run python scripts/rrf_weight_sweep.py --output docs/eval/rrf-weights.md`
([report](eval/rrf-weights.md)). Vector weight varied, full-text weight fixed at 1.0, gate as
configured:

| Retrieval | Heading R@1 | Heading R@10 | Heading MRR@10 | Subheading MRR@10 | Gate coverage | Gate acc@1 when answering |
|---|---|---|---|---|---|---|
| fts only | 0.574 | 0.778 | 0.645 | 0.545 | 42/54 | 25/42 (0.595) |
| vector only | 0.370 | 0.481 | 0.402 | 0.342 | 28/54 | 16/28 (0.571) |
| hybrid, vector weight 2 | 0.463 | 0.556 | 0.494 | 0.447 | 28/54 | 21/28 (0.750) |
| hybrid, vector weight 1 (default) | 0.481 | 0.722 | 0.554 | 0.494 | 28/54 | 22/28 (0.786) |
| hybrid, vector weight 0.75 | 0.500 | 0.722 | 0.570 | 0.494 | 29/54 | 22/29 (0.759) |
| hybrid, vector weight 0.5 | 0.500 | 0.722 | 0.575 | 0.480 | 31/54 | 22/31 (0.710) |
| hybrid, vector weight 0.35 | 0.537 | 0.759 | 0.602 | 0.506 | 32/54 | 23/32 (0.719) |
| hybrid, vector weight 0.25 | 0.537 | 0.778 | 0.606 | 0.507 | 34/54 | 23/34 (0.676) |
| hybrid, vector weight 0.1 | 0.519 | 0.778 | 0.606 | 0.507 | 44/54 | 25/44 (0.568) |

Two-fold check (weight picked on one half of the items by heading MRR@10, measured on the
other half):

| Tuned on | Chosen vector weight | MRR@10 on the other half | Equal weights | Full-text only |
|---|---|---|---|---|
| even items | 0.1 | 0.625 | 0.569 | 0.677 |
| odd items | 0.25 | 0.564 | 0.539 | 0.614 |

What this says, with the hash embedder:

- Down-weighting the lexical vector leg improves hybrid ranking (0.554 to about 0.60), and
  the gain holds on the held-out half, but **no weight makes hybrid beat BM25 alone** on
  ranking. The hash "vector" leg mostly repeats BM25's evidence with more noise.
- The weights also move the abstention gate: lower vector weights raise coverage but cut
  precision, because a line ranked first by BM25 alone then looks better supported. Equal
  weights give the most precise gate (22 of 28 answers right).
- The default therefore stays at equal weights (`RRF_WEIGHT_VECTOR=1.0`). The weights are
  configuration, and the right values depend on the embedder: rerun the sweep after
  re-ingesting with a semantic embedder (`EMBEDDING_PROVIDER=fastembed`, `openai` or
  `azure_openai`).

## Abstention gate

From the main eval run (hybrid mode, `GATE_MIN_HEADING_CONSENSUS=0.3`):

| min_support | Coverage | Heading acc@1 when answering | Out-of-scope abstained |
|---|---|---|---|
| 0.0 | 35/54 (0.648) | 23/35 (0.657) | 3/6 (0.500) |
| 0.6 | 33/54 (0.611) | 22/33 (0.667) | 3/6 (0.500) |
| 0.8 | 33/54 (0.611) | 22/33 (0.667) | 4/6 (0.667) |
| 0.9 (default) | 28/54 (0.519) | 22/28 (0.786) | 5/6 (0.833) |
| 0.95 | 26/54 (0.481) | 21/26 (0.808) | 6/6 (1.000) |

The default (0.9) was chosen from this sweep on this same small set, so re-tune it on your
own labelled data. The one out-of-scope item answered at 0.9 ("annual maintenance contract
for HVAC equipment") retrieves vehicle lighting equipment that both retrievers happen to
agree on. A known weakness: `heading_consensus` favours headings with many lines, so
"natural honey in glass jars" retrieves 0409 first but abstains because the many 7010
glass-container lines fill the window.

## HNSW recall vs. exact search

`uv run python scripts/hnsw_recall.py --ef-search 40 100 200 400 --rebuild 16/64 16/200`
(60 eval queries; recall@10 of the HNSW index against a sequential scan, ties counted; the
two rebuilds are temporary and the migrated index is restored afterwards). One run; the graph
build is randomised, and individual cells moved by up to about 0.03 between rebuilds:

| index | ef_search | recall | ms/query |
|---|---|---|---|
| as migrated (m=32, ef_construction=400) | 40 | 0.867 | 1.2 |
| as migrated (m=32, ef_construction=400) | 100 | 0.943 | 1.7 |
| as migrated (m=32, ef_construction=400) | 200 | 0.988 | 2.9 |
| as migrated (m=32, ef_construction=400) | 400 | 1.000 | 5.1 |
| m=16, ef_construction=64 (pgvector defaults; build 2.3 s) | 40 | 0.620 | 0.8 |
| m=16, ef_construction=64 (pgvector defaults; build 2.3 s) | 100 | 0.745 | 1.0 |
| m=16, ef_construction=64 (pgvector defaults; build 2.3 s) | 200 | 0.827 | 1.3 |
| m=16, ef_construction=64 (pgvector defaults; build 2.3 s) | 400 | 0.908 | 2.1 |
| m=16, ef_construction=200 (build 4.5 s) | 40 | 0.752 | 0.8 |
| m=16, ef_construction=200 (build 4.5 s) | 100 | 0.893 | 1.1 |
| m=16, ef_construction=200 (build 4.5 s) | 200 | 0.960 | 1.5 |
| m=16, ef_construction=200 (build 4.5 s) | 400 | 0.990 | 2.3 |

5,092 of the 26,246 lines embed to exactly the same tokens as another line (typically a line
and its "Other" child, since "other" is a stopword), and those duplicate vectors leave the
graph poorly connected at the default build settings. An average recall near 0.99 was not
enough: with `m=16, ef_construction=200`, six consecutive re-ingests gave vector-mode heading
Recall@1 of 0.370 four times and 0.352 twice, because the query "acoustic guitar" lost its
true nearest neighbour (9202.90.40.00) in some randomly built graphs. With `m=32,
ef_construction=400` (the shipped migration) the same six-rebuild check produced identical
eval metrics every time (individual rankings below the cut-offs can still shift slightly).
The cost is index build time (part of the ~17 s load step) and ~5 ms per vector query at
`ef_search=400`, both fine at this corpus size. Vector results are also ordered by
(distance, code), so ties between identical vectors do not depend on the graph.

## Bulk load

Measured on the full HTS with the shipped index: roughly 190-200 s to load with the HNSW
index maintained row by row during `COPY` (two runs), about 17 s when the index is dropped, the rows copied and
the index rebuilt in the same transaction (the shipped behaviour). Readers block on the
`TRUNCATE`/`DROP INDEX` locks until the load commits, then see the new index; a failed load
rolls back to the old one.
