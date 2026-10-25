-- Document frequencies of every lexeme in hts_lines.search_tsv, refreshed by each ingest.
-- Postgres's ts_rank/ts_rank_cd have no IDF component: a common word ("glass") counts as
-- much as a rare one ("honey"). These statistics let the store rank full-text matches with
-- Okapi BM25 while the GIN index still does the candidate filtering.
CREATE TABLE IF NOT EXISTS hts_lexeme_stats (
    lexeme text PRIMARY KEY,
    ndoc   integer NOT NULL CHECK (ndoc > 0)
);
