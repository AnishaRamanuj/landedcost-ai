## Retrieval eval

- **Dataset:** starter_eval.jsonl (60 items)
- **HTS data:** hts-2026-09-04 (26246 indexed lines)
- **Embedder:** hash-v2-c35
- **Store:** postgres (full-text ranking: ts_rank_cd, hnsw.ef_search: 400)
- **RRF:** k=60, weights vector=1, fts=1, rulings=1
- **Candidate pool per retriever:** 50
- **Run at:** 2026-10-24 20:51 UTC

### Retrieval

| Mode | Level | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| vector | heading (4) | 54 | 0.370 | 0.426 | 0.444 | 0.481 | 0.402 |
| vector | subheading (6) | 54 | 0.296 | 0.352 | 0.407 | 0.481 | 0.342 |
| fts | heading (4) | 54 | 0.241 | 0.389 | 0.519 | 0.593 | 0.338 |
| fts | subheading (6) | 54 | 0.148 | 0.259 | 0.370 | 0.444 | 0.227 |
| hybrid | heading (4) | 54 | 0.463 | 0.519 | 0.556 | 0.630 | 0.502 |
| hybrid | subheading (6) | 54 | 0.389 | 0.426 | 0.481 | 0.574 | 0.429 |

### Abstention gate (configured: min_support=0.9, min_heading_consensus=0.3)

| Metric | Value |
|---|---|
| Answerable items answered (coverage) | 20/54 (0.370) |
| Heading accuracy@1 on answered items | 14/20 (0.700) |
| Heading accuracy@1 on all answerable items (no gate) | 25/54 (0.463) |
| Out-of-scope items abstained | 4/6 (0.667) |

### Coverage vs. accuracy by min_support (other gate settings as configured)

| min_support | Coverage | Heading acc@1 when answering | Out-of-scope abstained |
|---|---|---|---|
| 0.0 | 33/54 (0.611) | 20/33 (0.606) | 2/6 (0.333) |
| 0.6 | 30/54 (0.556) | 18/30 (0.600) | 3/6 (0.500) |
| 0.8 | 29/54 (0.537) | 17/29 (0.586) | 4/6 (0.667) |
| 0.9 | 20/54 (0.370) | 14/20 (0.700) | 4/6 (0.667) |
| 0.95 | 15/54 (0.278) | 11/15 (0.733) | 5/6 (0.833) |
