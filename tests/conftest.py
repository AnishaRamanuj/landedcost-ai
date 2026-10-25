"""Shared fixtures: a real-data HTS subset loaded into the in-memory store."""

from __future__ import annotations

from pathlib import Path

import pytest

from landedcost.config import Settings, StoreBackend
from landedcost.container import Services, assemble
from landedcost.embeddings.hashing import HashEmbedder
from landedcost.hts.loader import load_lines
from landedcost.hts.models import HtsLine
from landedcost.ingest import import_rulings_file
from landedcost.store.memory import InMemoryStore

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent
SAMPLE_SNAPSHOT = FIXTURES / "hts_sample.json"
SYNTHETIC_RULINGS = FIXTURES / "rulings_synthetic.csv"


@pytest.fixture(scope="session")
def sample_lines() -> list[HtsLine]:
    _, parsed = load_lines(SAMPLE_SNAPSHOT)
    return parsed.lines


@pytest.fixture(scope="session")
def lines_by_code(sample_lines: list[HtsLine]) -> dict[str, HtsLine]:
    return {line.code: line for line in sample_lines}


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
async def memory_store(sample_lines: list[HtsLine], embedder: HashEmbedder) -> InMemoryStore:
    store = InMemoryStore()
    vectors = await embedder.embed_documents([line.embedding_text for line in sample_lines])
    await store.replace_hts_lines(
        sample_lines, vectors, hts_version="hts-2026-09-04", embedder=embedder.name
    )
    return store


@pytest.fixture
async def store_with_rulings(memory_store: InMemoryStore, embedder: HashEmbedder) -> InMemoryStore:
    report = await import_rulings_file(memory_store, embedder, SYNTHETIC_RULINGS)
    assert report.imported == 4
    return memory_store


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        store_backend=StoreBackend.MEMORY,
        hts_snapshot_path=SAMPLE_SNAPSHOT,
    )


@pytest.fixture
def services(settings: Settings, memory_store: InMemoryStore, embedder: HashEmbedder) -> Services:
    return assemble(settings, store=memory_store, embedder=embedder)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every setting to its default so neither the developer's shell nor a local ``.env``
    (environment variables take precedence over the file) leaks into tests. Tests that need
    another value set it with ``monkeypatch.setenv``."""
    for name, field in Settings.model_fields.items():
        default = field.default
        monkeypatch.setenv(name.upper(), "" if default is None else str(default))
    monkeypatch.chdir(REPO_ROOT)
