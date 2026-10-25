"""PostgreSQL + pgvector fixtures.

Point ``LANDEDCOST_TEST_DATABASE_URL`` at a disposable database (CI uses a
``pgvector/pgvector:pg16`` service container). Without it, a throwaway container is started
with Testcontainers when Docker is reachable; otherwise the integration tests are skipped.
The tests drop and recreate the landedcost tables, so never point this at real data.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import pytest

from landedcost.embeddings.hashing import HashEmbedder
from landedcost.ingest import ingest_hts
from landedcost.store.migrate import apply_migrations
from landedcost.store.postgres import PostgresStore
from tests.conftest import SAMPLE_SNAPSHOT

TEST_DB_ENV = "LANDEDCOST_TEST_DATABASE_URL"


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    url = os.environ.get(TEST_DB_ENV)
    if url:
        yield url
        return
    try:
        from testcontainers.community.postgres import PostgresContainer
    except ImportError:  # pragma: no cover - dev dependency normally installed
        pytest.skip(f"set {TEST_DB_ENV} to run PostgreSQL integration tests")
    try:
        container = PostgresContainer("pgvector/pgvector:pg16", driver=None)
        container.start()
    except Exception as exc:  # any Docker failure means "skip", not "fail"
        pytest.skip(f"set {TEST_DB_ENV} or start Docker for Testcontainers ({type(exc).__name__})")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture
async def pg_schema(pg_url: str) -> str:
    await apply_migrations(pg_url, embedding_dim=384, drop_existing=True)
    return pg_url


@pytest.fixture
async def pg_store(pg_schema: str, embedder: HashEmbedder) -> AsyncIterator[PostgresStore]:
    store = await PostgresStore.connect(pg_schema, max_size=4)
    try:
        await ingest_hts(store, embedder, SAMPLE_SNAPSHOT)
        yield store
    finally:
        await store.close()
