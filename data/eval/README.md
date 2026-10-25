# Starter eval set

`starter_eval.jsonl` holds 60 items:

* **54 answerable items**: product descriptions written the way an importer or a commercial
  invoice would describe goods (brand-free, with sizes and materials, not copied from tariff
  text), each labelled with its gold HTS heading (4 digits) and subheading (6 digits).
* **6 out-of-scope items** (`expect_abstain: true`): services and nonsense strings that have
  no HTS classification. They only feed the abstention table.

Labelling method: each code was chosen by reading the HTS 2026-09-04 text for the candidate
headings and confirming that the subheading wording matches the goods (material, function,
capacity thresholds). Draft items whose subheading could not be pinned down from the
description alone, or that are known judgement calls (electric kettles, for example), were
dropped rather than guessed, and a few more were dropped to keep the set at 60.
`tests/unit/test_evals.py` checks that every gold code exists in the shipped snapshot and that
no description is a verbatim HTS description.

Limitations: one labeller, no second review, and a small set that was also used while
choosing retrieval and gate defaults (there is no separate held-out set; only the RRF weight
sweep in `scripts/rrf_weight_sweep.py` uses a two-fold split of it). Treat the numbers as a
regression signal for this repository, not as a benchmark.

Format:

```json
{"id": "olive-oil", "description": "Extra virgin olive oil in 1 litre glass bottles",
 "gold_headings": ["1509"], "gold_subheadings": ["1509.20"],
 "source": "hand-labeled against HTS 2026-09-04"}
```
