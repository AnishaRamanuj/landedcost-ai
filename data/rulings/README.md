# CBP rulings corpus

The repository ships **no rulings**: `rulings_template.csv` has the header only. CBP's
Customs Rulings Online Search System (CROSS, <https://rulings.cbp.gov>) has no documented
public API, so the corpus is something you build and verify yourself.

## Format

CSV (or JSONL with the same keys):

| Column | Example | Notes |
|---|---|---|
| `ruling_number` | `NXXXXXX` (placeholder) | Required, unique. Re-importing a number updates it. |
| `ruling_date` | `2024-05-17` | ISO-8601, optional. |
| `product_description` | `Lithium-ion battery pack for e-bikes` | Required; what is embedded and searched. |
| `hts_codes` | `8507.60.0090;8507.60.0010` | Required. Separate several with `;` `,` `|` or spaces. |
| `url` | the ruling's page on rulings.cbp.gov | Shown with citations. |
| `excerpt` | short quote of the holding | Searched (lower weight) and shown as citation text. |

```bash
uv run landedcost import-rulings data/rulings/rulings.csv
```

Invalid rows (missing number/description, malformed HTS code or date, a JSONL line that is
not a JSON object) are skipped and listed with their line numbers; valid rows are still
imported. CSV files saved by Excel as "CSV UTF-8" (with a byte-order mark) are accepted.

## Hand-collecting rulings from CROSS

1. Search CROSS for the product family (for example `lithium-ion battery`).
2. Open each relevant ruling and copy the ruling number, date, the HTS code(s) from the
   holding, the ruling URL and one or two sentences from the holding as the excerpt.
3. Write the product description in your own words, the way an importer would describe the
   goods. It becomes both the search text and, for `landedcost eval --rulings`, an eval query.
4. Save it as `data/rulings/rulings.csv` and commit it, so imports and the
   `eval --rulings` numbers are reproducible. (Other files in this folder, such as drafts
   from `scripts/fetch_rulings.py`, are git-ignored until you have verified them.)

`scripts/fetch_rulings.py` can draft such a file from the endpoints the CROSS web app uses.
Those endpoints are **unofficial and undocumented**: verify every drafted row against the
ruling page before importing it.

## How rulings are used

* Retrieval searches rulings (vector + full-text) alongside tariff lines. The HTS lines cited
  by the top rulings form an extra ranked list in Reciprocal Rank Fusion, so a ruling on a
  similar product can lift the tariff line it was classified under.
* Answers cite supporting rulings next to the tariff text.
* `landedcost eval --rulings` adds one eval item per ruling (description -> its HTS codes),
  with that ruling hidden from retrieval so it cannot vote for its own answer.

Never commit invented ruling numbers or text. The only rulings in this repository are the
obviously synthetic `TEST-xxxx` rows under `tests/fixtures/`.
