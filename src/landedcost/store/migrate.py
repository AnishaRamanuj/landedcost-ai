"""Apply the plain-SQL migrations in ``store/migrations`` idempotently.

All pending files run in order inside a single transaction (so a failing file leaves the
schema untouched) and each is recorded in ``schema_migrations``. A transaction-scoped
advisory lock serialises concurrent runners (e.g. an API container and a CLI invocation
starting at the same time).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import psycopg
from psycopg.rows import TupleRow

_ADVISORY_LOCK_KEY = 0x1A9DED0C057  # arbitrary, constant per application
_DIM_PLACEHOLDER = "{{EMBEDDING_DIM}}"


@dataclass(frozen=True, slots=True)
class Migration:
    version: str
    sql: str


def load_migrations(embedding_dim: int) -> list[Migration]:
    if not 1 <= embedding_dim <= 2000:
        raise ValueError("EMBEDDING_DIM must be between 1 and 2000 (pgvector HNSW limit)")
    folder = resources.files("landedcost.store") / "migrations"
    files = sorted((f for f in folder.iterdir() if f.name.endswith(".sql")), key=lambda f: f.name)
    return [
        Migration(
            f.name.removesuffix(".sql"),
            f.read_text("utf-8").replace(_DIM_PLACEHOLDER, str(embedding_dim)),
        )
        for f in files
    ]


async def embedding_column_dim(conn: psycopg.AsyncConnection[TupleRow]) -> int | None:
    """Dimension of ``hts_lines.embedding`` if the table exists (kept in ``atttypmod``)."""
    cur = await conn.execute(
        """
        SELECT a.atttypmod FROM pg_attribute a
        WHERE a.attrelid = to_regclass('hts_lines') AND a.attname = 'embedding'
        """
    )
    row = await cur.fetchone()
    return int(row[0]) if row else None


async def apply_migrations(
    conninfo: str, *, embedding_dim: int, drop_existing: bool = False
) -> list[str]:
    """Apply pending migrations; return the versions that were applied."""
    applied: list[str] = []
    async with await psycopg.AsyncConnection.connect(conninfo) as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,))
        if drop_existing:
            await conn.execute(
                "DROP TABLE IF EXISTS hts_lines, rulings, hts_lexeme_stats, index_meta, "
                "schema_migrations"
            )
        await conn.execute(
            """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version text PRIMARY KEY,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
        )
        existing_dim = await embedding_column_dim(conn)
        if existing_dim is not None and existing_dim != embedding_dim:
            raise RuntimeError(
                f"hts_lines.embedding is vector({existing_dim}) but EMBEDDING_DIM={embedding_dim}; "
                "run `landedcost migrate --drop-existing` and re-ingest"
            )
        cur = await conn.execute("SELECT version FROM schema_migrations")
        done = {str(r[0]) for r in await cur.fetchall()}
        for migration in load_migrations(embedding_dim):
            if migration.version in done:
                continue
            await conn.execute(migration.sql.encode())
            await conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (migration.version,)
            )
            applied.append(migration.version)
    return applied
