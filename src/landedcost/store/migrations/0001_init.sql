-- Schema for HTS lines and CBP rulings.
-- {{EMBEDDING_DIM}} is substituted by landedcost.store.migrate from EMBEDDING_DIM.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS index_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE TABLE IF NOT EXISTS hts_lines (
    code             text PRIMARY KEY,
    level            text NOT NULL
        CHECK (level IN ('heading', 'subheading', 'tariff_line', 'statistical')),
    chapter          char(2) NOT NULL,
    heading          char(4) NOT NULL,
    description      text NOT NULL,
    path             text[] NOT NULL DEFAULT '{}',
    context          text NOT NULL DEFAULT '',
    indent           smallint NOT NULL,
    units            text[] NOT NULL DEFAULT '{}',
    general_rate     text NOT NULL DEFAULT '',
    special_rate     text NOT NULL DEFAULT '',
    other_rate       text NOT NULL DEFAULT '',
    rate_source_code text,
    footnotes        text[] NOT NULL DEFAULT '{}',
    embedding        vector({{EMBEDDING_DIM}}) NOT NULL,
    -- The line's own description is weighted A, its ancestors' descriptions B.
    search_tsv       tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', description), 'A')
        || setweight(to_tsvector('english', context), 'B')
    ) STORED
);

-- m and ef_construction are raised from pgvector's defaults (16, 64): thousands of HTS lines
-- share their text, and therefore their embedding, with a sibling ("Other"), and the
-- resulting duplicate vectors leave the default graph poorly connected. With m=32 and
-- ef_construction=400, eval results were identical across repeated rebuilds; see
-- scripts/hnsw_recall.py for the measured recall/latency trade-off.
CREATE INDEX IF NOT EXISTS hts_lines_embedding_hnsw
    ON hts_lines USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 400);
CREATE INDEX IF NOT EXISTS hts_lines_search_tsv_gin ON hts_lines USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS hts_lines_code_prefix ON hts_lines (code text_pattern_ops);

CREATE TABLE IF NOT EXISTS rulings (
    ruling_number       text PRIMARY KEY,
    ruling_date         date,
    product_description text NOT NULL,
    hts_codes           text[] NOT NULL CHECK (cardinality(hts_codes) > 0),
    url                 text NOT NULL DEFAULT '',
    excerpt             text NOT NULL DEFAULT '',
    embedding           vector({{EMBEDDING_DIM}}) NOT NULL,
    search_tsv          tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', product_description), 'A')
        || setweight(to_tsvector('english', excerpt), 'B')
    ) STORED,
    imported_at         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rulings_embedding_hnsw
    ON rulings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS rulings_search_tsv_gin ON rulings USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS rulings_hts_codes_gin ON rulings USING gin (hts_codes);
