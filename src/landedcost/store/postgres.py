"""PostgreSQL + pgvector store.

* HTS lines and rulings live in separate tables, each with an ``embedding vector(dim)``
  column (HNSW, cosine) and a computed, weighted ``tsvector`` column (GIN).
* Vector and full-text queries are independent statements, so the hybrid retriever runs
  them concurrently on two pooled connections.
* ``replace_hts_lines`` replaces the whole index inside one transaction (DROP INDEX +
  TRUNCATE + COPY + index rebuild). Readers never see a half-loaded index, but TRUNCATE and
  DROP INDEX take ACCESS EXCLUSIVE locks, so queries block until the load commits (about
  17 s for the full HTS) and then see the new index; a failed load rolls back to the old one.
  A staging table swapped in with ``ALTER TABLE ... RENAME`` would avoid the pause; at this
  corpus size and re-ingest frequency (a new HTS edition a few times a year) it is not worth
  the extra code.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import psycopg
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from landedcost.embeddings.base import Matrix, Vector
from landedcost.hts.codes import CodeLevel
from landedcost.hts.models import HtsLine
from landedcost.rulings.models import Ruling
from landedcost.store.base import (
    IndexInfo,
    ScoredLine,
    ScoredRuling,
    StoreUnavailableError,
    check_embedder,
)
from landedcost.text import fts_query, words

_HTS_COLUMNS = (
    "code, level, description, path, indent, units, general_rate, special_rate, other_rate, "
    "rate_source_code, footnotes"
)
_HTS_COLUMNS_H = ", ".join(f"h.{c.strip()}" for c in _HTS_COLUMNS.split(","))
_RULING_COLUMNS = "ruling_number, ruling_date, product_description, hts_codes, url, excerpt"
_HNSW_INDEX = "hts_lines_embedding_hnsw"

# Okapi BM25 over the computed tsvector column. Term frequency counts a lexeme position in the
# line's own description (weight A) twice and one in its ancestors' text (weight B) once,
# mirroring the in-memory store. N and avgdl are written to index_meta at ingest.
_BM25_SQL = f"""
WITH params AS (
    SELECT (SELECT value::float FROM index_meta WHERE key = 'fts_doc_count') AS n,
           (SELECT value::float FROM index_meta WHERE key = 'fts_avg_doc_len') AS avgdl,
           1.2::float AS k1, 0.75::float AS b
),
query_terms AS (
    SELECT s.lexeme, ln(1 + (p.n - s.ndoc + 0.5) / (s.ndoc + 0.5)) AS idf
    FROM params p,
         unnest(tsvector_to_array(to_tsvector('english', %(terms)s))) AS t(lexeme)
    JOIN hts_lexeme_stats s ON s.lexeme = t.lexeme
),
candidates AS (
    SELECT h.code, h.search_tsv,
           (SELECT sum(array_length(u.positions, 1)) FROM unnest(h.search_tsv) u) AS dl
    FROM hts_lines h
    WHERE h.search_tsv @@ websearch_to_tsquery('english', %(tsquery)s)
),
scored AS (
    SELECT c.code,
           sum(q.idf * (tf.tf * (p.k1 + 1)) / (tf.tf + p.k1 * (1 - p.b + p.b * c.dl / p.avgdl)))
               AS score
    FROM candidates c
    CROSS JOIN params p
    CROSS JOIN LATERAL (
        SELECT u.lexeme,
               (SELECT 2 * count(*) FILTER (WHERE w = 'A') + count(*) FILTER (WHERE w <> 'A')
                FROM unnest(u.weights) AS w)::float AS tf
        FROM unnest(c.search_tsv) AS u
    ) tf
    JOIN query_terms q ON q.lexeme = tf.lexeme
    GROUP BY c.code
)
SELECT {_HTS_COLUMNS_H}, s.score
FROM scored s JOIN hts_lines h ON h.code = s.code
ORDER BY s.score DESC, h.code
LIMIT %(limit)s
"""

# Many HTS lines share their text (and so their vector) with a sibling. The inner query is
# the HNSW index scan; the outer ORDER BY makes ties deterministic (by code) instead of
# depending on the randomised graph build, so evals are reproducible across re-ingests.
_TIE_BREAK_FACTOR = 2
_VECTOR_SQL = f"""
SELECT * FROM (
    SELECT {_HTS_COLUMNS}, embedding <=> %(q)s AS distance
    FROM hts_lines
    ORDER BY embedding <=> %(q)s
    LIMIT %(fetch)s
) AS nearest
ORDER BY distance, code
LIMIT %(limit)s
"""

_TS_RANK_SQL = f"""
SELECT {_HTS_COLUMNS}, ts_rank_cd(search_tsv, q) AS score
FROM hts_lines, websearch_to_tsquery('english', %(tsquery)s) AS q
WHERE search_tsv @@ q
ORDER BY score DESC, code
LIMIT %(limit)s
"""


FtsRanking = Literal["bm25", "ts_rank_cd"]


def _line_from_row(row: DictRow) -> HtsLine:
    return HtsLine(
        code=row["code"],
        level=CodeLevel(row["level"]),
        description=row["description"],
        path=tuple(row["path"]),
        indent=row["indent"],
        units=tuple(row["units"]),
        general_rate=row["general_rate"],
        special_rate=row["special_rate"],
        other_rate=row["other_rate"],
        rate_source_code=row["rate_source_code"],
        footnotes=tuple(row["footnotes"]),
    )


def _ruling_from_row(row: DictRow) -> Ruling:
    return Ruling(
        ruling_number=row["ruling_number"],
        ruling_date=row["ruling_date"],
        product_description=row["product_description"],
        hts_codes=tuple(row["hts_codes"]),
        url=row["url"],
        excerpt=row["excerpt"],
    )


async def _configure(conn: AsyncConnection[Any]) -> None:
    await register_vector_async(conn)


async def _diagnose(conninfo: str, timeout_s: float) -> str:
    """One direct connection attempt, to report why the pool could not open."""
    try:
        conn = await AsyncConnection.connect(conninfo, connect_timeout=max(1, round(timeout_s)))
    except psycopg.Error as exc:
        return f"cannot connect to PostgreSQL (check DATABASE_URL): {exc}".strip()
    try:
        await register_vector_async(conn)
    except psycopg.Error as exc:
        return f"connected, but pgvector is not usable (run `landedcost migrate`): {exc}"
    finally:
        await conn.close()
    return f"timed out after {timeout_s}s opening the PostgreSQL connection pool"


_SCHEMA_HINT = "the landedcost schema is missing; run `landedcost migrate` first"


class PostgresStore:
    """Async store backed by a ``psycopg_pool.AsyncConnectionPool``."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        *,
        ef_search: int = 400,
        fts_ranking: FtsRanking = "bm25",
    ):
        self._pool = pool
        self._ef_search = ef_search
        self._fts_ranking = fts_ranking

    @classmethod
    async def connect(
        cls,
        conninfo: str,
        *,
        min_size: int = 1,
        max_size: int = 8,
        open_timeout_s: float = 5.0,
        ef_search: int = 400,
        fts_ranking: FtsRanking = "bm25",
    ) -> PostgresStore:
        """Open a pool. The schema must already exist (``landedcost migrate``).

        Raises :class:`StoreUnavailableError` with the server's own message (bad password,
        unknown host, missing database, ...) when no connection opens within
        ``open_timeout_s``, instead of hanging on a bare pool timeout.
        """
        pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
            conninfo,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            configure=_configure,
            open=False,
        )
        try:
            await pool.open(wait=True, timeout=open_timeout_s)
        except PoolTimeout as exc:
            await pool.close()
            raise StoreUnavailableError(await _diagnose(conninfo, open_timeout_s)) from exc
        return cls(pool, ef_search=ef_search, fts_ranking=fts_ranking)

    async def close(self) -> None:
        await self._pool.close()

    # -- HTS search -----------------------------------------------------------------------
    async def search_hts_vector(self, embedding: Vector, limit: int) -> list[ScoredLine]:
        fetch = limit * _TIE_BREAK_FACTOR
        async with self._pool.connection() as conn, conn.transaction():
            # HNSW returns at most ef_search rows; keep it above what we fetch.
            await conn.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(max(self._ef_search, fetch)),),
            )
            cur = await conn.execute(
                _VECTOR_SQL.encode(),
                {"q": np.asarray(embedding, dtype=np.float32), "fetch": fetch, "limit": limit},
            )
            rows = await cur.fetchall()
        return [ScoredLine(_line_from_row(r), 1.0 - float(r["distance"])) for r in rows]

    async def search_hts_text(self, query: str, limit: int) -> list[ScoredLine]:
        terms = words(query)
        if not terms:
            return []
        sql = _BM25_SQL if self._fts_ranking == "bm25" else _TS_RANK_SQL
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                sql.encode(),
                {"terms": " ".join(terms), "tsquery": fts_query(query), "limit": limit},
            )
            rows = await cur.fetchall()
        return [ScoredLine(_line_from_row(r), float(r["score"])) for r in rows]

    # -- ruling search --------------------------------------------------------------------
    async def search_rulings_vector(
        self, embedding: Vector, limit: int, exclude: Collection[str] = ()
    ) -> list[ScoredRuling]:
        fetch = (limit + len(exclude)) * _TIE_BREAK_FACTOR
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(max(self._ef_search, fetch)),),
            )
            cur = await conn.execute(
                f"""
                SELECT * FROM (
                    SELECT {_RULING_COLUMNS}, embedding <=> %(q)s AS distance
                    FROM rulings
                    WHERE NOT (ruling_number = ANY(%(exclude)s))
                    ORDER BY embedding <=> %(q)s
                    LIMIT %(fetch)s
                ) AS nearest
                ORDER BY distance, ruling_number
                LIMIT %(limit)s
                """,
                {
                    "q": np.asarray(embedding, dtype=np.float32),
                    "fetch": fetch,
                    "limit": limit,
                    "exclude": list(exclude),
                },
            )
            rows = await cur.fetchall()
        return [ScoredRuling(_ruling_from_row(r), 1.0 - float(r["distance"])) for r in rows]

    async def search_rulings_text(
        self, query: str, limit: int, exclude: Collection[str] = ()
    ) -> list[ScoredRuling]:
        tsquery = fts_query(query)
        if not tsquery:
            return []
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT {_RULING_COLUMNS}, ts_rank_cd(search_tsv, q) AS score
                FROM rulings, websearch_to_tsquery('english', %(q)s) AS q
                WHERE search_tsv @@ q AND NOT (ruling_number = ANY(%(exclude)s))
                ORDER BY score DESC, ruling_number
                LIMIT %(limit)s
                """,
                {"q": tsquery, "limit": limit, "exclude": list(exclude)},
            )
            rows = await cur.fetchall()
        return [ScoredRuling(_ruling_from_row(r), float(r["score"])) for r in rows]

    # -- lookups --------------------------------------------------------------------------
    async def get_hts_lines(self, codes: Sequence[str]) -> dict[str, HtsLine]:
        if not codes:
            return {}
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_HTS_COLUMNS} FROM hts_lines WHERE code = ANY(%s)", (list(codes),)
            )
            rows = await cur.fetchall()
        return {r["code"]: _line_from_row(r) for r in rows}

    async def find_hts_by_prefix(self, prefix: str, limit: int) -> list[HtsLine]:
        pattern = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_HTS_COLUMNS} FROM hts_lines WHERE code LIKE %s ORDER BY code LIMIT %s",
                (pattern, limit),
            )
            rows = await cur.fetchall()
        return [_line_from_row(r) for r in rows]

    async def list_rulings(self) -> list[Ruling]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_RULING_COLUMNS} FROM rulings ORDER BY ruling_number"
            )
            rows = await cur.fetchall()
        return [_ruling_from_row(r) for r in rows]

    async def index_info(self) -> IndexInfo:
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    """
                    SELECT (SELECT count(*) FROM hts_lines) AS hts_lines,
                           (SELECT count(*) FROM rulings) AS rulings
                    """
                )
                counts = await cur.fetchone()
                cur = await conn.execute("SELECT key, value FROM index_meta")
                meta = {r["key"]: r["value"] for r in await cur.fetchall()}
        except psycopg.errors.UndefinedTable as exc:
            raise StoreUnavailableError(_SCHEMA_HINT) from exc
        assert counts is not None
        dim = meta.get("embedding_dim")
        return IndexInfo(
            hts_lines=int(counts["hts_lines"]),
            rulings=int(counts["rulings"]),
            embedder=meta.get("embedder"),
            embedding_dim=int(dim) if dim else None,
            hts_version=meta.get("hts_version"),
            rulings_embedder=meta.get("rulings_embedder"),
        )

    # -- writes ---------------------------------------------------------------------------
    async def _write_meta(self, conn: AsyncConnection[DictRow], meta: dict[str, str]) -> None:
        async with conn.cursor() as cur:
            await cur.executemany(
                """
                INSERT INTO index_meta (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                list(meta.items()),
            )

    async def replace_hts_lines(
        self, lines: Sequence[HtsLine], embeddings: Matrix, *, hts_version: str, embedder: str
    ) -> int:
        if len(lines) != len(embeddings):
            raise ValueError("lines and embeddings differ in length")
        dim = int(embeddings.shape[1])
        async with self._pool.connection() as conn, conn.transaction():
            # Building an HNSW index once after a bulk load is several times faster than
            # maintaining it row by row during COPY. We re-create it from its own catalog
            # definition so the migration stays the single source of index parameters.
            cur = await conn.execute(
                "SELECT pg_get_indexdef(to_regclass(%s)) AS ddl", (_HNSW_INDEX,)
            )
            row = await cur.fetchone()
            index_ddl = row["ddl"] if row else None
            if index_ddl:
                await conn.execute(f"DROP INDEX {_HNSW_INDEX}")
            await conn.execute("TRUNCATE hts_lines")
            async with conn.cursor() as copy_cur:
                copy_sql = (
                    "COPY hts_lines (code, level, chapter, heading, description, path, context, "
                    "indent, units, general_rate, special_rate, other_rate, rate_source_code, "
                    "footnotes, embedding) FROM STDIN"
                )
                async with copy_cur.copy(copy_sql) as copy:
                    for line, vector in zip(lines, embeddings, strict=True):
                        await copy.write_row(
                            (
                                line.code,
                                line.level.value,
                                line.chapter,
                                line.heading,
                                line.description,
                                list(line.path),
                                " ".join(line.path),
                                line.indent,
                                list(line.units),
                                line.general_rate,
                                line.special_rate,
                                line.other_rate,
                                line.rate_source_code,
                                list(line.footnotes),
                                np.asarray(vector, dtype=np.float32),
                            )
                        )
            if index_ddl:
                await conn.execute("SET LOCAL maintenance_work_mem = '256MB'")
                await conn.execute(index_ddl.encode())
            fts_meta = await self._refresh_lexeme_stats(conn)
            await self._write_meta(
                conn,
                {
                    "embedder": embedder,
                    "embedding_dim": str(dim),
                    "hts_version": hts_version,
                    "hts_ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    **fts_meta,
                },
            )
        return len(lines)

    async def _refresh_lexeme_stats(self, conn: AsyncConnection[DictRow]) -> dict[str, str]:
        """Recompute BM25 statistics (document frequencies, N, average length)."""
        await conn.execute("TRUNCATE hts_lexeme_stats")
        await conn.execute(
            """
            INSERT INTO hts_lexeme_stats (lexeme, ndoc)
            SELECT word, ndoc FROM ts_stat('SELECT search_tsv FROM hts_lines')
            """
        )
        cur = await conn.execute(
            """
            SELECT count(*) AS n,
                   coalesce(avg((SELECT sum(array_length(u.positions, 1))
                                 FROM unnest(search_tsv) u)), 0) AS avgdl
            FROM hts_lines
            """
        )
        row = await cur.fetchone()
        assert row is not None
        return {"fts_doc_count": str(row["n"]), "fts_avg_doc_len": str(float(row["avgdl"]))}

    async def upsert_rulings(
        self, rulings: Sequence[Ruling], embeddings: Matrix, *, embedder: str
    ) -> int:
        if len(rulings) != len(embeddings):
            raise ValueError("rulings and embeddings differ in length")
        check_embedder(await self.index_info(), embedder, int(embeddings.shape[1]))
        async with self._pool.connection() as conn, conn.transaction():
            await self._insert_rulings(conn, rulings, embeddings)
            await self._write_meta(conn, {"rulings_embedder": embedder})
        return len(rulings)

    async def replace_rulings(
        self, rulings: Sequence[Ruling], embeddings: Matrix, *, embedder: str
    ) -> int:
        if len(rulings) != len(embeddings):
            raise ValueError("rulings and embeddings differ in length")
        check_embedder(
            await self.index_info(), embedder, int(embeddings.shape[1]), include_rulings=False
        )
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute("TRUNCATE rulings")
            await self._insert_rulings(conn, rulings, embeddings)
            await self._write_meta(conn, {"rulings_embedder": embedder})
        return len(rulings)

    async def _insert_rulings(
        self, conn: AsyncConnection[DictRow], rulings: Sequence[Ruling], embeddings: Matrix
    ) -> None:
        rows = [
            (
                r.ruling_number,
                r.ruling_date,
                r.product_description,
                list(r.hts_codes),
                r.url,
                r.excerpt,
                np.asarray(v, dtype=np.float32),
            )
            for r, v in zip(rulings, embeddings, strict=True)
        ]
        async with conn.cursor() as cur:
            await cur.executemany(
                f"""
                INSERT INTO rulings ({_RULING_COLUMNS}, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ruling_number) DO UPDATE SET
                    ruling_date = EXCLUDED.ruling_date,
                    product_description = EXCLUDED.product_description,
                    hts_codes = EXCLUDED.hts_codes,
                    url = EXCLUDED.url,
                    excerpt = EXCLUDED.excerpt,
                    embedding = EXCLUDED.embedding,
                    imported_at = now()
                """,
                rows,
            )
