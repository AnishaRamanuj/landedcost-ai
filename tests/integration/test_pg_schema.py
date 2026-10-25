import psycopg
import pytest

from landedcost.store.migrate import apply_migrations, load_migrations

pytestmark = pytest.mark.integration


async def test_migrations_are_idempotent_and_recorded(pg_schema: str) -> None:
    assert await apply_migrations(pg_schema, embedding_dim=384) == []
    async with await psycopg.AsyncConnection.connect(pg_schema) as conn:
        cur = await conn.execute("SELECT version FROM schema_migrations ORDER BY version")
        assert [r[0] for r in await cur.fetchall()] == ["0001_init", "0002_bm25_stats"]


async def test_indexes_exist_with_the_right_access_methods(pg_schema: str) -> None:
    async with await psycopg.AsyncConnection.connect(pg_schema) as conn:
        cur = await conn.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename IN ('hts_lines', 'rulings')"
        )
        defs: dict[str, str] = dict(await cur.fetchall())
    assert "USING hnsw (embedding vector_cosine_ops)" in defs["hts_lines_embedding_hnsw"]
    assert "m='32', ef_construction='400'" in defs["hts_lines_embedding_hnsw"]
    assert "USING gin (search_tsv)" in defs["hts_lines_search_tsv_gin"]
    assert "USING hnsw" in defs["rulings_embedding_hnsw"]
    assert "USING gin (hts_codes)" in defs["rulings_hts_codes_gin"]


async def test_dimension_change_is_refused(pg_schema: str) -> None:
    with pytest.raises(RuntimeError, match="drop-existing"):
        await apply_migrations(pg_schema, embedding_dim=128)
    # ...and allowed after dropping everything
    assert await apply_migrations(pg_schema, embedding_dim=128, drop_existing=True)
    await apply_migrations(pg_schema, embedding_dim=384, drop_existing=True)


def test_dimension_bounds_and_placeholder_substitution() -> None:
    with pytest.raises(ValueError, match="between 1 and 2000"):
        load_migrations(4096)
    sql = load_migrations(384)[0].sql
    assert "vector(384)" in sql
    assert "{{" not in sql
