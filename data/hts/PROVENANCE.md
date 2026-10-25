# HTS snapshot provenance

| | |
|---|---|
| File | `hts_2026-09-04.json.gz` (gzip, `gzip -n`, no embedded name or timestamp) |
| SHA-256 (gz) | `e5606cdaf742f80d5608ede021397eb489d64d1206d945445b331264bd326346` |
| Content | Harmonized Tariff Schedule of the United States, all chapters, 35,804 rows |
| Fetched | 2026-09-04 |
| Original source | U.S. International Trade Commission HTS REST export, `https://hts.usitc.gov/reststop/exportList` |
| License | U.S. Government work, public domain (17 U.S.C. 105) |

## How this copy was obtained

The rows were obtained from the snapshot bundled in the `tariff-resolver` npm package, which
packages the output of the official USITC export API above. The tariff data itself is a U.S.
Government work in the public domain; only the transport differed. Nothing in the rows was
edited. The JSON keeps the export's fields per row: `htsno`, `indent`, `description`,
`superior`, `units`, `general`, `special`, `other`, `footnotes`, `quotaQuantity`,
`additionalDuties` (plus the export's misspelled duplicate `addiitionalDuties`), wrapped in
`{"fetched_at", "source", "license", "rows"}`.

## Refreshing

`scripts/fetch_hts.py` downloads a fresh copy straight from the USITC endpoint, one chapter at
a time, validates every response and writes the same structure:

```bash
uv run python scripts/fetch_hts.py --out data/hts/hts_$(date +%F).json.gz
uv run landedcost ingest-hts --snapshot data/hts/hts_$(date +%F).json.gz
```

The shipped file was not produced by this script, and USITC does not version the export
interface, so treat the script's first run as a test of the endpoint as well: compare the
output of `uv run landedcost stats --snapshot <new file>` with the shipped snapshot.

## What the parser does with it

* Chapters 98 (special classification provisions) and 99 (temporary legislation and
  additional duties such as Section 301/232) are parsed but excluded from the retrieval index;
  they do not classify a product on their own.
* Unnumbered "superior" rows ("Other:", "Men's or boys':") are kept as hierarchy context.
* 10-digit statistical suffixes inherit the rate columns of their 8-digit parent line.

`uv run landedcost stats` prints the counts for the shipped file.
