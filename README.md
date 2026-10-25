# landedcost-ai

Suggests US HTS codes for imported goods with cited tariff text and CBP rulings, abstains
when the evidence is weak, and estimates landed cost (duty, MPF, HMF, freight).

[![CI](https://github.com/AnishaRamanuj/landedcost-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AnishaRamanuj/landedcost-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-3776AB)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)

> **Not legal or customs advice — designed to assist licensed customs brokers.** Every answer
> carries this disclaimer. Classification and entry remain the broker's responsibility.

**Stack:** Python · FastAPI · PostgreSQL + pgvector (HNSW) · Azure OpenAI / Claude · Docker

## Why

Importers describe goods in invoice language ("36V e-bike battery pack"), while the
Harmonized Tariff Schedule describes them in legal language spread over a 35,000-row
hierarchy where the leaf that carries the duty rate is often just "Other". A wrong code
changes the duty owed and can trigger penalties, so a tool in this space has to show its
sources and say "I don't know" rather than guess. This project is a retrieval system over the
official HTS text that does both, plus the landed-cost arithmetic a broker does next.

## What it does

- **Parses the official USITC HTS** (snapshot 2026-09-04, shipped in the repo) into 26,246
  searchable lines, each with its full hierarchy ("Electric storage batteries … > Lithium-ion
  batteries > Other > Other") and the duty rate inherited from its parent tariff line.
- **Hybrid retrieval in PostgreSQL**: pgvector HNSW (cosine) and full-text search ranked with
  BM25, run concurrently and merged with **Reciprocal Rank Fusion** (k=60, optional
  per-retriever weights).
- **CBP rulings** you import (CSV/JSONL) are searched too; the tariff lines they cite vote in
  the fusion and are cited in answers.
- **Evidence gate**: abstains, with a reason, when the retrievers disagree or the evidence is
  spread across unrelated headings.
- **Optional LLM** (Claude via the `anthropic` SDK, or OpenAI / Azure OpenAI via `openai`)
  answers only from the retrieved excerpts with JSON-schema output; a **citation guard**
  rejects any code or citation that was not retrieved, and any code that does not cite its
  own tariff line. Default is a no-LLM extractive answer.
- **Rate parser** turns "Free", "6.8%", "2.5¢/kg", "$1.03/doz." and "3.2¢/kg + 5%" into
  structured rates (97.5% of rated lines); everything else is flagged for broker review.
- **Landed cost** = customs value + freight + insurance + MFN duty + MPF (with min/max) + HMF
  (ocean), with explicit warnings for what is excluded (Chapter 99 / Section 301 / AD/CVD).
- **Evals**: recall@k and MRR at heading and subheading level for `vector`, `fts` and
  `hybrid`, abstention coverage/accuracy, and end-to-end answer accuracy, on a hand-labelled
  starter set.
- FastAPI service (`/classify`, `/landed-cost`, `/hts/{code}`, `/healthz`, OpenAPI docs, a
  small HTML page) and a Typer CLI.

## Architecture

```mermaid
flowchart LR
    subgraph ingest["Ingest (landedcost ingest-hts / import-rulings)"]
        A["USITC HTS snapshot<br/>data/hts/*.json.gz"] --> B["Parser<br/>hierarchy + rate inheritance"]
        R["CBP rulings<br/>CSV / JSONL"] --> E1
        B --> E1["Embedder<br/>hash · OpenAI · Azure · fastembed"]
        E1 --> DB[("PostgreSQL 16 + pgvector<br/>hts_lines · rulings<br/>HNSW · GIN tsvector · BM25 stats")]
    end
    subgraph query["Query (POST /classify)"]
        Q["Product description"] --> H["HybridRetriever"]
        H -- "vector top-k (HNSW)" --> DB
        H -- "full-text top-k (BM25)" --> DB
        H --> F["Reciprocal Rank Fusion<br/>vector + full-text + ruling votes"]
        F --> G{"Evidence gate"}
        G -- weak --> X["Abstain + reason"]
        G -- strong --> L{"LLM configured?"}
        L -- no --> EX["Extractive answer"]
        L -- yes --> M["Claude / Azure OpenAI<br/>JSON-schema output"]
        M --> CG["Citation guard"]
        CG --> OUT["Candidates + citations + confidence"]
        EX --> OUT
    end
    OUT -. "8/10-digit code" .-> LC["POST /landed-cost<br/>duty + MPF + HMF + freight"]
```

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12+, PostgreSQL 16 with pgvector
installed and a role allowed to `CREATE EXTENSION vector` (or Docker).

### Local (the path used for every output in this README)

```bash
uv sync
export DATABASE_URL=postgresql://dev:dev@localhost:5432/landedcost   # any empty database
uv run landedcost migrate        # idempotent schema migration
uv run landedcost ingest-hts     # parse + embed + load 26,246 HTS lines (~20 s)
uv run landedcost classify "Rechargeable lithium-ion battery pack for e-bikes, 36V 10Ah"
uv run landedcost serve          # http://127.0.0.1:8201  (docs at /docs)
```

No database at hand? `STORE_BACKEND=memory uv run landedcost classify "..."` loads the HTS
into an in-memory store (exact vector search + BM25 in Python) at startup.

### Docker Compose

```bash
docker compose up --build        # Postgres (pgvector/pgvector:pg16) + API on 127.0.0.1:8201
```

The API container runs `migrate`, then `ingest-hts --if-empty`, then `serve`. The CI
workflow's `docker` job builds the image and smoke-tests `/healthz` and `/classify` in a
container.

## Demo (captured from a real run)

Full HTS loaded into PostgreSQL, hash embedder, no LLM configured.

```text
$ uv run landedcost classify "Rechargeable lithium-ion battery pack for e-bikes, 36V 10Ah"
query: Rechargeable lithium-ion battery pack for e-bikes, 36V 10Ah
mode: extractive  confidence: 0.71  support: 1.00  heading consensus: 0.71
1. 8507.60.00  general rate: 3.4%
   Electric storage batteries, including separators therefor, whether or not rectangular (including square); parts thereof > Lithium-ion batteries
   Retrieved by vector rank 1, full-text rank 1 (RRF score 0.0328).
   citations: hts:8507.60.00
2. 8507.60.00.90  general rate: 3.4%
   Electric storage batteries, including separators therefor, whether or not rectangular (including square); parts thereof > Lithium-ion batteries > Other > Other
   Retrieved by vector rank 2, full-text rank 2 (RRF score 0.0323).
   citations: hts:8507.60.00.90
3. 8506.50.00  general rate: 2.7%
   Primary cells and primary batteries; parts thereof > Lithium
   Retrieved by vector rank 3, full-text rank 5 (RRF score 0.0313).
   citations: hts:8506.50.00
Not legal or customs advice. Suggestions are designed to assist licensed customs brokers, who remain responsible for the final classification and entry.

$ uv run landedcost classify "Cloud accounting software subscription, annual license"
query: Cloud accounting software subscription, annual license
mode: extractive  confidence: 0.00  support: 0.50  heading consensus: 0.10
ABSTAINED: retrievers disagree on the best match (support 0.50 < 0.90)
Not legal or customs advice. Suggestions are designed to assist licensed customs brokers, who remain responsible for the final classification and entry.
```

<!-- demo.gif: record with `asciinema rec` + `agg` while running the two classify commands above -->

`POST /classify` returns the same evidence as JSON, including per-retriever ranks and raw
scores (cosine similarity for `vector`, BM25 for `fts`; fusion uses only the ranks):

```text
$ curl -s -X POST localhost:8201/classify -H 'Content-Type: application/json' \
    -d '{"description": "Extra virgin olive oil in 1 litre glass bottles", "top_k": 2}' \
    | python3 -m json.tool --indent 4
```

<details>
<summary>Response (unedited)</summary>

```json
{
    "query": "Extra virgin olive oil in 1 litre glass bottles",
    "candidates": [
        {
            "hts_code": "1509.20",
            "level": "subheading",
            "description": "Extra virgin olive oil",
            "breadcrumb": "Olive oil and its fractions, whether or not refined, but not chemically modified > Extra virgin olive oil",
            "general_rate": "",
            "rate_inherited_from": null,
            "rationale": "Retrieved by vector rank 1, full-text rank 1 (RRF score 0.0328).",
            "citations": [
                "hts:1509.20"
            ],
            "retrieval": {
                "fused_score": 0.032787,
                "ranks": {
                    "vector": 1,
                    "fts": 1
                },
                "scores": {
                    "vector": 0.6998,
                    "fts": 40.9267
                }
            }
        },
        {
            "hts_code": "1509.20.90",
            "level": "tariff_line",
            "description": "Other",
            "breadcrumb": "Olive oil and its fractions, whether or not refined, but not chemically modified > Extra virgin olive oil > Other",
            "general_rate": "3.4\u00a2/kg",
            "rate_inherited_from": null,
            "rationale": "Retrieved by vector rank 2, full-text rank 2 (RRF score 0.0323).",
            "citations": [
                "hts:1509.20.90"
            ],
            "retrieval": {
                "fused_score": 0.032258,
                "ranks": {
                    "vector": 2,
                    "fts": 2
                },
                "scores": {
                    "vector": 0.6162,
                    "fts": 34.4148
                }
            }
        }
    ],
    "confidence": 1.0,
    "abstained": false,
    "abstain_reason": null,
    "citations": [
        {
            "chunk_id": "hts:1509.20",
            "kind": "hts_line",
            "reference": "1509.20",
            "text": "Olive oil and its fractions, whether or not refined, but not chemically modified > Extra virgin olive oil",
            "url": null
        },
        {
            "chunk_id": "hts:1509.20.90",
            "kind": "hts_line",
            "reference": "1509.20.90",
            "text": "Olive oil and its fractions, whether or not refined, but not chemically modified > Extra virgin olive oil > Other",
            "url": null
        }
    ],
    "evidence": {
        "support": 1.0,
        "heading_consensus": 1.0,
        "fts_hits": 50,
        "vector_hits": 50,
        "ruling_hits": 0
    },
    "answer_mode": "extractive",
    "llm": null,
    "warnings": [],
    "disclaimer": "Not legal or customs advice. Suggestions are designed to assist licensed customs brokers, who remain responsible for the final classification and entry."
}
```

</details>

The subheading (1509.20, extra virgin) is right, but the second candidate, 1509.20.90, is
the line for containers of 18 kg or more; a 1-litre bottle belongs under 1509.20.20 ("under
18 kg"). Keyword and hashed retrieval cannot reason about that threshold. 1509.20.20 is in the
12 lines the LLM path receives as context (fused rank 10), which is where a model, or the broker, picks the
right 8-digit line.

Landed cost for the battery line, with mock ocean freight from weight/volume:

```text
$ uv run landedcost landed-cost --hts 8507.60.00.90 --value 12000 --mode ocean --weight-kg 850 --volume-cbm 2.4 --insurance 60
8507.60.00.90  Electric storage batteries, including separators therefor, whether or not rectangular (including square); parts thereof > Lithium-ion batteries > Other > Other
  customs value            $12,000.00
  freight (static-mock)       $228.00
  insurance                    $60.00
  duty @ 3.4%                 $408.00
  MPF                          $41.57
  HMF                          $15.00
  TOTAL landed cost        $12,752.57
  duty: 3.4% x $12000 = $408.00
  ! Chapter 99 additional duties (Section 301, Section 232, IEEPA and other temporary tariffs) and antidumping/countervailing duties are NOT included. They can exceed the MFN duty; check the country of origin against HTS chapter 99 and AD/CVD orders.
  ! Duty uses the column 1 General (MFN) rate only; free-trade-agreement (Special) and column 2 rates are not applied.
  ! MPF/HMF parameters are configured for FY2027 (MPF 0.3464% with min $34.58 / max $670.86; HMF 0.125%). Verify against CBP's current user-fee table.
  ! Duty, MPF and HMF are computed on the customs (transaction) value; freight and insurance are added to landed cost but not to the dutiable value.
  ! MPF assumes a formal entry; informal entries use a different fee.
Not legal or customs advice. Suggestions are designed to assist licensed customs brokers, who remain responsible for the final classification and entry.
```

When a rate cannot be computed safely, the total is withheld instead of guessed (MPF is
raised to the FY2027 minimum here):

```text
$ uv run landedcost landed-cost --hts 1509.20.20 --value 4800 --mode ocean --weight-kg 1100 --freight 640
1509.20.20  Olive oil and its fractions, whether or not refined, but not chemically modified > Extra virgin olive oil > Weighing with the immediate container under 18 kg
  customs value                              $4,800.00
  freight (provided)                           $640.00
  insurance                                      $0.00
  duty @ 5¢/kg on contents and container           n/a
  MPF                                           $34.58
  HMF                                            $6.00
  TOTAL landed cost                                n/a
  duty: requires broker review: unsupported unit in '5¢/kg on contents and container'
  ! Chapter 99 additional duties (Section 301, Section 232, IEEPA and other temporary tariffs) and antidumping/countervailing duties are NOT included. They can exceed the MFN duty; check the country of origin against HTS chapter 99 and AD/CVD orders.
  ! Duty uses the column 1 General (MFN) rate only; free-trade-agreement (Special) and column 2 rates are not applied.
  ! MPF/HMF parameters are configured for FY2027 (MPF 0.3464% with min $34.58 / max $670.86; HMF 0.125%). Verify against CBP's current user-fee table.
  ! Total omitted: duty requires broker review.
  ! Duty, MPF and HMF are computed on the customs (transaction) value; freight and insurance are added to landed cost but not to the dutiable value.
  ! MPF assumes a formal entry; informal entries use a different fee.
Not legal or customs advice. Suggestions are designed to assist licensed customs brokers, who remain responsible for the final classification and entry.
```

(The `POST /landed-cost` response for the battery example is the same calculation as JSON;
money fields are decimal strings such as `"12752.57"`.)

## Evaluation

All numbers come from the commands shown, run against the full HTS snapshot (26,246
indexed lines) in PostgreSQL 16 + pgvector 0.6.0 with the default configuration: embedder
`hash-v2-c35`, RRF k=60 with equal weights, 50 candidates per retriever, BM25 full-text
ranking, `hnsw.ef_search` 400. Dataset: `data/eval/starter_eval.jsonl`, 54 answerable items
with gold heading and subheading plus 6 out-of-scope items, written in importer language and
hand-labelled against the shipped HTS.

`uv run landedcost eval --answers --output docs/eval/hash-embedder.md --details docs/eval/hash-embedder-details.jsonl`
(full report: [`docs/eval/hash-embedder.md`](docs/eval/hash-embedder.md); per-item results:
[`docs/eval/hash-embedder-details.jsonl`](docs/eval/hash-embedder-details.jsonl))

| Mode | Level | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| vector | heading (4) | 54 | 0.370 | 0.426 | 0.444 | 0.481 | 0.402 |
| vector | subheading (6) | 54 | 0.296 | 0.352 | 0.407 | 0.481 | 0.342 |
| fts | heading (4) | 54 | 0.574 | 0.722 | 0.741 | 0.778 | 0.645 |
| fts | subheading (6) | 54 | 0.463 | 0.556 | 0.685 | 0.741 | 0.545 |
| hybrid | heading (4) | 54 | 0.481 | 0.593 | 0.630 | 0.722 | 0.554 |
| hybrid | subheading (6) | 54 | 0.426 | 0.519 | 0.593 | 0.685 | 0.494 |

| Final answers (hybrid, gate at `min_support=0.9`, extractive) | Value |
|---|---|
| Answerable items answered (coverage) | 28/54 (0.519) |
| Heading accuracy@1 on answered items | 22/28 (0.786) |
| Heading accuracy@1 without the gate | 26/54 (0.481) |
| Out-of-scope items abstained | 5/6 (0.833) |

**Takeaway.** With the offline, lexical hash embedder, equal-weight hybrid ranks below BM25
alone; its value here is the abstention signal (answering only when both retrievers agree
lifts heading accuracy from 48% to 79% at half the coverage). Semantic embedders are
supported but not yet measured; [the tuning notes](docs/retrieval-tuning.md) cover BM25 vs.
`ts_rank_cd` (where hybrid does beat both legs), an RRF weight sweep with a held-out check,
the gate threshold sweep and HNSW recall.

## Design decisions

**Hybrid retrieval + RRF.** Tariff text is a vocabulary problem: exact legal terms
("accumulators", "knitted or crocheted") favour keyword search, paraphrases favour
embeddings. RRF fuses ranked lists without calibrating a cosine similarity against a BM25
score, has one main parameter (k=60, from the original paper; per-retriever weights are
optional), and gives per-retriever ranks that double as explainability ("vector rank 1,
full-text rank 3") and as the abstention signal. Both searches run concurrently on separate
pooled connections (`asyncio.gather`; a unit test with a two-way barrier fails if they run
one after the other).

**BM25 inside Postgres.** `ts_rank_cd` ignores term rarity. Rather than adding a search
engine, ingest refreshes a lexeme document-frequency table from the computed `tsvector` column
(`ts_stat`), and the query scores GIN-filtered candidates with Okapi BM25 in SQL, counting the
line's own description (weight A) twice and its ancestors' text (weight B) once. It costs one
extra table and nearly doubled full-text MRR on the eval (0.338 to 0.645).

**Why abstain.** A confidently wrong code is worse than no answer. The gate runs before any
LLM call, so weak retrieval never reaches a model that could produce a plausible-looking
code. Signals are rank-based (`support`: how close the top line is to being ranked first by
both retrievers; `heading_consensus`: how much of the top-10 fused score agrees on its
heading), so they mean the same thing with any embedder. Confidence is reported as a
heuristic, not a calibrated probability.

**Citation guard.** Structured outputs guarantee shape, not truth. The LLM may only return
codes that normalise to a retrieved HTS line, every candidate must cite its own tariff line
(citing only a neighbouring line, such as the parent of an "Other" suffix, gets it
rejected), and other citations must be retrieved chunk ids. Rejections are visible as
warnings, every cited id resolves to its source text in `citations[]`, and the description
and rate shown are re-read from the tariff data, never from the model. If nothing survives,
the answer is an abstention.

**LLM providers.** `anthropic`: official SDK, default model `claude-opus-5-5`,
`output_config={"format": {"type": "json_schema", ...}, "effort": "medium"}`, no sampling
parameters, server-side refusal fallbacks (`betas=["server-side-fallback-2026-07-01"]`,
`fallbacks="default"`), `stop_reason == "refusal"` checked before reading content and turned
into an abstention, typed SDK errors mapped most-specific-first. `openai` / `azure_openai`:
Chat Completions with a strict JSON schema; Azure uses deployment names. Every SDK client gets
an explicit timeout and retry budget (`LLM_TIMEOUT_S=30`, `LLM_MAX_RETRIES=2`; the SDK default
timeout is 10 minutes), and a provider failure falls back to the extractive answer with a
warning. `landedcost eval --answers` scores whichever answer path is configured.

**pgvector instead of a separate vector database.** The corpus is tens of thousands of rows;
one PostgreSQL holds vectors, full-text, metadata and rulings, so hybrid queries, joins and a
transactional re-index need no second system and no cross-store consistency story. The HNSW
index is dropped before the bulk `COPY` and re-created from its own catalog definition
afterwards (about 17 s for the full HTS instead of roughly 190-200 s with the index
maintained row by row). Readers block until the load commits and then see the new index; a failed load
rolls back to the old one (integration-tested).

**The hash embedder trade-off.** Signed feature hashing over stemmed words, word bigrams and
character 3–5-grams into 384 dimensions: deterministic, offline, ~3 s for the whole HTS, and
good enough for CI and demos. It is lexical, so it cannot know a laptop is an "automatic data
processing machine". `EMBEDDING_DIM=384` is shared with `text-embedding-3-*` (via its
`dimensions` parameter) and `bge-small-en-v1.5`. The index records which embedder built the
HTS vectors and, separately, the ruling vectors: querying with a different embedder fails
fast with a re-ingest message, and `ingest-hts` re-embeds stored rulings when the embedder
changes.

**Rulings as votes, not candidates.** Rulings are fused among themselves, then the HTS lines
they cite become a third RRF list. A ruling on a similar product can lift the line it was
classified under but can never be returned as a "code". Ruling-derived eval items hide their
own ruling from retrieval (leave-one-out) so they cannot vote for their own answer. A ruling
citing a statistical suffix missing from this HTS edition falls back to its nearest ancestor.

**Money and fees.** Amounts are `Decimal` end to end, serialise as strings, and are bounded
on input (at most 10^12, 4 decimal places), so absurd values get a 422 instead of a
rounding error. MPF/HMF rates and the MPF min/max live in configuration with the fiscal year
they apply to; every estimate repeats them and warns once that fiscal year has ended.
Specific and compound rates are computed only when the needed quantity is supplied; anything
the parser cannot represent returns "requires broker review" and no total.

**Composition root.** `container.py` builds every adapter from settings and owns their
lifecycle (store pool, SDK clients, freight HTTP client); the API and CLI receive a
`Services` object, and tests inject an in-memory store, fake LLMs and mock HTTP transports
through the same seam.

## Testing

```bash
uv run pytest tests/unit                       # no services needed
make check                                     # ruff, mypy --strict, all tests with coverage
make test TEST_DB=postgresql://dev:dev@localhost:5432/landedcost_test   # + PostgreSQL tests
```

Integration tests drop and recreate the tables in the database you point them at
(`TEST_DB`, or `LANDEDCOST_TEST_DATABASE_URL` in the environment). Without one they start a
throwaway `pgvector/pgvector:pg16` container via Testcontainers when Docker is available, and
skip cleanly otherwise. CI runs them against a GitHub Actions service container on Python
3.12 and 3.13, and fails the eval job if hybrid or full-text heading MRR drops below a floor.

Last local run: **279 tests passed** (257 unit, 22 integration against PostgreSQL 16 +
pgvector 0.6.0) on Python 3.12 and 3.13, **98% line coverage**, `ruff` and `mypy --strict`
clean; without a database, 258 pass and the 21 PostgreSQL tests skip. They cover HTS parsing
and rate inheritance (including a full-snapshot check), rate parsing and duty/fee math with
input bounds and fiscal-year staleness, RRF/BM25 math, retrieval modes with ruling votes,
leave-one-out, concurrency and embedder-mismatch checks, the gate and citation guard
(hallucinated codes, unknown citations, a candidate citing only a neighbouring line), the
classifier with a fake LLM (refusal, invalid JSON, outage, prompt-injection escaping), exact
Anthropic/OpenAI request shapes against fake SDK clients, PostgreSQL behaviour (index use in
query plans, BM25, re-ingest rollback, rulings re-embedding, connection and missing-schema
errors, API and CLI end to end) and eval-data integrity (every gold code exists; no
description copied from tariff text).

## Project structure

```text
landedcost-ai/
├── src/landedcost/
│   ├── hts/            codes.py (normalisation) · parser.py (hierarchy, rate inheritance) · loader.py
│   ├── rates/          parser.py (Free / % / ¢/kg / compound) · duty.py · units.py
│   ├── embeddings/     hashing.py (offline baseline) · openai_embedder.py · fastembed_embedder.py
│   ├── store/          postgres.py (pgvector + BM25 SQL) · memory.py · bm25.py · migrate.py · migrations/*.sql
│   ├── retrieval/      rrf.py · hybrid.py (concurrent vector + full-text + ruling votes)
│   ├── classify/       gate.py · guard.py · prompts.py · service.py · models.py (API + LLM schema)
│   ├── llm/            anthropic_provider.py · openai_provider.py (OpenAI + Azure) · base.py
│   ├── landed/         calculator.py · fees.py (MPF/HMF) · freight.py (static + HTTP) · models.py
│   ├── rulings/        importer.py (CSV/JSONL validation) · models.py
│   ├── evals/          dataset.py · metrics.py · runner.py (retrieval, gate and answer scoring)
│   ├── api/            app.py (FastAPI) · schemas.py · templates/index.html
│   ├── cli.py          Typer: migrate, ingest-hts, import-rulings, classify, landed-cost, eval, stats, serve
│   ├── config.py       pydantic-settings
│   ├── container.py    composition root
│   └── ingest.py       ingestion use cases
├── data/
│   ├── hts/            hts_2026-09-04.json.gz · PROVENANCE.md
│   ├── rulings/        rulings_template.csv · README.md (how to hand-collect from CROSS)
│   └── eval/           starter_eval.jsonl · README.md
├── docs/               retrieval-tuning.md · eval/ (reports written by `landedcost eval` and scripts)
├── scripts/            fetch_hts.py · fetch_rulings.py · hnsw_recall.py · rrf_weight_sweep.py
├── tests/              unit/ · integration/ · fixtures/ (HTS subset, synthetic TEST-xxxx rulings)
├── Dockerfile · docker-compose.yml · Makefile · .github/workflows/ci.yml
└── pyproject.toml · uv.lock
```

## Configuration

Environment variables (or a `.env` file; see [`.env.example`](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `STORE_BACKEND` | `postgres` | `postgres` or `memory` (loads the HTS in-process) |
| `DATABASE_URL` | `postgresql://dev:dev@localhost:5432/landedcost` | PostgreSQL connection |
| `DB_POOL_MIN_SIZE` / `DB_POOL_MAX_SIZE` | `1` / `10` | Connection pool (a hybrid query uses up to 4 connections) |
| `DB_CONNECT_TIMEOUT_S` | `5` | Fail fast with the server's message when the database is unreachable |
| `HTS_SNAPSHOT_PATH` | `data/hts/hts_2026-09-04.json.gz` | Snapshot for ingest / memory mode |
| `EMBEDDING_PROVIDER` | `hash` | `hash`, `openai`, `azure_openai`, `fastembed` (extra) |
| `EMBEDDING_DIM` | `384` | Vector dimension (schema and embedder) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local ONNX model (`uv sync --extra fastembed`) |
| `EMBEDDING_TIMEOUT_S` / `EMBEDDING_MAX_RETRIES` | `60` / `3` | OpenAI / Azure embedding requests |
| `RETRIEVAL_MODE` | `hybrid` | `vector`, `fts`, `hybrid` |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RRF_WEIGHT_VECTOR` / `RRF_WEIGHT_FTS` / `RRF_WEIGHT_RULINGS` | `1.0` each | Per-retriever RRF weights ([sweep](docs/retrieval-tuning.md#rrf-weights)) |
| `RETRIEVAL_CANDIDATE_POOL` | `50` | Hits per retriever before fusion |
| `FTS_RANKING` | `bm25` | `bm25` or `ts_rank_cd` |
| `HNSW_EF_SEARCH` | `400` | HNSW search breadth |
| `GATE_MIN_SUPPORT` | `0.9` | Abstain below this cross-retriever support |
| `GATE_MIN_HEADING_CONSENSUS` | `0.3` | Abstain below this heading agreement |
| `GATE_WINDOW` | `10` | Candidates considered by the gate |
| `LLM_PROVIDER` | `none` | `none`, `anthropic`, `openai`, `azure_openai` |
| `LLM_TIMEOUT_S` / `LLM_MAX_RETRIES` | `30` / `2` | Per-request timeout and SDK retries for LLM calls |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | — / `claude-opus-5-5` | Claude provider |
| `ANTHROPIC_EFFORT` / `ANTHROPIC_USE_FALLBACKS` | `medium` / `true` | Effort level, server-side refusal fallbacks |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | — / `gpt-4.1-mini` | OpenAI provider |
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` | — | Azure OpenAI resource |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | Azure API version |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` / `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | — | Azure deployment names |
| `FEE_SCHEDULE_FISCAL_YEAR` | `FY2027` | Fiscal year of the fee values (estimates warn once it has ended) |
| `MPF_RATE` / `MPF_MIN_USD` / `MPF_MAX_USD` | `0.003464` / `34.58` / `670.86` | Merchandise Processing Fee, FY2027 per CBP Dec. 26-14 (verify yearly) |
| `HMF_RATE` | `0.00125` | Harbor Maintenance Fee (ocean only) |
| `FREIGHT_PROVIDER` | `static` | `static` (mock table) or `http` |
| `FREIGHT_API_URL` / `FREIGHT_API_TIMEOUT_S` | — / `10` | HTTP quote API (`POST {url}/v1/quotes`) |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8201` | `landedcost serve` bind address |
| `LANDEDCOST_TEST_DATABASE_URL` | — | Disposable database for integration tests (shell only) |

## Data provenance

- **HTS**: official USITC Harmonized Tariff Schedule export, fetched 2026-09-04
  (`https://hts.usitc.gov/reststop/exportList`), a U.S. Government work in the public domain.
  This copy came from the snapshot bundled in the `tariff-resolver` npm package, which
  packages that official export; see [`data/hts/PROVENANCE.md`](data/hts/PROVENANCE.md) for
  the checksum and refresh steps (`scripts/fetch_hts.py`).
- **CBP rulings**: none are shipped yet. CROSS has no documented API, so the corpus is
  hand-collected into `data/rulings/rulings.csv` and imported; see
  [`data/rulings/README.md`](data/rulings/README.md). `scripts/fetch_rulings.py` is an
  unofficial helper against undocumented endpoints. Test rulings are obviously synthetic
  (`TEST-0001`…) and live only in `tests/fixtures/`.
- **Fees**: MPF/HMF defaults are the FY2027 values announced in CBP Dec. 26-14 (effective
  2026-10-01). They are configuration; check CBP's current notice before relying on them.
- **Freight**: the default provider returns **mock** quotes from a fixed rate table; the HTTP
  provider calls whatever quote API you configure. No carrier integration is implied.

## Limitations and roadmap

- **Published numbers use the offline hash embedder and extractive answers.** The semantic
  embedders and the Claude / Azure OpenAI path are implemented and covered by unit tests
  against fake SDK clients, but no semantic-embedding or LLM-in-the-loop eval is published
  yet. `EMBEDDING_PROVIDER=fastembed uv run landedcost ingest-hts` followed by
  `uv run landedcost eval --answers` (optionally with `LLM_PROVIDER=anthropic`) produces those
  tables.
- **Small, single-labeller eval set** (54 + 6 items) that was also used to pick the gate
  defaults and retrieval choices; only the RRF weight sweep uses a held-out split.
  Ruling-derived items (`eval --rulings`) are supported, but no rulings ship yet.
- **The gate is a heuristic.** `heading_consensus` favours headings with many lines (see the
  honey example in [the tuning notes](docs/retrieval-tuning.md#abstention-gate)). A model
  calibrated on labelled outcomes would do better.
- **Candidates can be headings or subheadings without a rate**; landed cost needs an 8- or
  10-digit line. The General Rules of Interpretation, section and chapter notes are not
  modelled; tariff lines are retrieved by their own text and hierarchy only.
- **Duty is MFN (column 1 General) only**: no free-trade-agreement preferences, no Chapter 99
  (Section 301/232, IEEPA), no AD/CVD; each estimate says so. MPF assumes a formal entry.
  About 2.5% of rated lines (582) use rate text the parser does not model and return
  "requires broker review".
- **No authentication or rate limiting** on the API. Run it behind a gateway before exposing
  it beyond localhost, especially with LLM keys configured (Compose binds it to 127.0.0.1).
- `scripts/fetch_hts.py` and `scripts/fetch_rulings.py` target interfaces USITC and CBP do
  not version (the CROSS endpoints are undocumented); treat a first run as a test of the
  endpoint.
- Roadmap: semantic-embedding and LLM eval runs, a hand-collected rulings corpus with an
  `eval --rulings` table, a larger held-out eval split, chapter/section notes as retrievable
  context, FTA preference programs, and Chapter 99 lookups by country of origin.

## License

[MIT](LICENSE) © 2026 Anisha V. Shah. The HTS data is a U.S. Government work in the public
domain.
