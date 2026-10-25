## RRF weight sweep

- **Command:** uv run python scripts/rrf_weight_sweep.py --weights 2 1 0.75 0.5 0.35 0.25 0.1
- **Dataset:** starter_eval.jsonl (54 answerable items)
- **HTS data:** hts-2026-09-04 (26246 indexed lines)
- **Embedder:** hash-v2-c35
- **Fixed:** fts weight 1, RRF k=60, gate min_support=0.9, min_heading_consensus=0.3
- **Run at:** 2026-10-24 20:51 UTC

| Retrieval | Heading R@1 | Heading R@10 | Heading MRR@10 | Subheading MRR@10 | Gate coverage | Gate acc@1 when answering |
|---|---|---|---|---|---|---|
| fts | 0.574 | 0.778 | 0.645 | 0.545 | 42/54 | 25/42 (0.595) |
| vector | 0.370 | 0.481 | 0.402 | 0.342 | 28/54 | 16/28 (0.571) |
| hybrid, vector weight 2 | 0.463 | 0.556 | 0.494 | 0.447 | 28/54 | 21/28 (0.750) |
| hybrid, vector weight 1 | 0.481 | 0.722 | 0.554 | 0.494 | 28/54 | 22/28 (0.786) |
| hybrid, vector weight 0.75 | 0.500 | 0.722 | 0.570 | 0.494 | 29/54 | 22/29 (0.759) |
| hybrid, vector weight 0.5 | 0.500 | 0.722 | 0.575 | 0.480 | 31/54 | 22/31 (0.710) |
| hybrid, vector weight 0.35 | 0.537 | 0.759 | 0.602 | 0.506 | 32/54 | 23/32 (0.719) |
| hybrid, vector weight 0.25 | 0.537 | 0.778 | 0.606 | 0.507 | 34/54 | 23/34 (0.676) |
| hybrid, vector weight 0.1 | 0.519 | 0.778 | 0.606 | 0.507 | 44/54 | 25/44 (0.568) |

### Two-fold check (weight chosen on one half, measured on the other)

| Tuned on | Chosen vector weight | Heading MRR@10 on the other half | Equal weights on the other half | Full-text only on the other half |
|---|---|---|---|---|
| even items | 0.1 | 0.625 | 0.569 | 0.677 |
| odd items | 0.25 | 0.564 | 0.539 | 0.614 |
