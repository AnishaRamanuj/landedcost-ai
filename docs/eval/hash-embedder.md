## Retrieval eval

- **Dataset:** starter_eval.jsonl (60 items)
- **HTS data:** hts-2026-09-04 (26246 indexed lines)
- **Embedder:** hash-v2-c35
- **Store:** postgres (full-text ranking: bm25, hnsw.ef_search: 400)
- **RRF:** k=60, weights vector=1, fts=1, rulings=1
- **Candidate pool per retriever:** 50
- **Run at:** 2026-10-24 20:51 UTC

### Retrieval

| Mode | Level | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| vector | heading (4) | 54 | 0.370 | 0.426 | 0.444 | 0.481 | 0.402 |
| vector | subheading (6) | 54 | 0.296 | 0.352 | 0.407 | 0.481 | 0.342 |
| fts | heading (4) | 54 | 0.574 | 0.722 | 0.741 | 0.778 | 0.645 |
| fts | subheading (6) | 54 | 0.463 | 0.556 | 0.685 | 0.741 | 0.545 |
| hybrid | heading (4) | 54 | 0.481 | 0.593 | 0.630 | 0.722 | 0.554 |
| hybrid | subheading (6) | 54 | 0.426 | 0.519 | 0.593 | 0.685 | 0.494 |

### Abstention gate (configured: min_support=0.9, min_heading_consensus=0.3)

| Metric | Value |
|---|---|
| Answerable items answered (coverage) | 28/54 (0.519) |
| Heading accuracy@1 on answered items | 22/28 (0.786) |
| Heading accuracy@1 on all answerable items (no gate) | 26/54 (0.481) |
| Out-of-scope items abstained | 5/6 (0.833) |

### Coverage vs. accuracy by min_support (other gate settings as configured)

| min_support | Coverage | Heading acc@1 when answering | Out-of-scope abstained |
|---|---|---|---|
| 0.0 | 35/54 (0.648) | 23/35 (0.657) | 3/6 (0.500) |
| 0.6 | 33/54 (0.611) | 22/33 (0.667) | 3/6 (0.500) |
| 0.8 | 33/54 (0.611) | 22/33 (0.667) | 4/6 (0.667) |
| 0.9 | 28/54 (0.519) | 22/28 (0.786) | 5/6 (0.833) |
| 0.95 | 26/54 (0.481) | 21/26 (0.808) | 6/6 (1.000) |

### Final answers (answer mode: extractive (no LLM))

| Metric | Value |
|---|---|
| Answerable items answered (coverage) | 28/54 (0.519) |
| Heading accuracy@1 on answered items | 22/28 (0.786) |
| Subheading accuracy@1 on answered items | 19/28 (0.679) |
| Out-of-scope items abstained | 5/6 (0.833) |
| LLM calls that fell back to the extractive answer | 0 |
