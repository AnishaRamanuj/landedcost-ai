from decimal import Decimal

import httpx
import pytest
from pydantic import SecretStr

from landedcost import container
from landedcost.config import EmbeddingProvider, LlmProviderName, Settings
from landedcost.container import (
    ConfigurationError,
    assemble,
    build_embedder,
    build_fee_schedule,
    build_llm,
    build_services,
)
from landedcost.embeddings.hashing import HashEmbedder
from landedcost.embeddings.openai_embedder import OpenAIEmbedder
from landedcost.llm.anthropic_provider import AnthropicProvider
from landedcost.llm.base import aclose_all
from landedcost.llm.openai_provider import OpenAIChatProvider
from landedcost.store.base import StoreUnavailableError
from landedcost.store.memory import InMemoryStore


def make(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[arg-type]


def test_defaults_are_offline() -> None:
    settings = make()
    assert isinstance(build_embedder(settings), HashEmbedder)
    assert build_llm(settings) is None
    assert settings.api_port == 8201
    assert settings.anthropic_model == "claude-opus-5-5"


def test_embedder_selection() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_embedder(make(embedding_provider=EmbeddingProvider.OPENAI))
    openai_embedder = build_embedder(
        make(embedding_provider=EmbeddingProvider.OPENAI, openai_api_key=SecretStr("sk-test"))
    )
    assert isinstance(openai_embedder, OpenAIEmbedder)
    assert openai_embedder.name == "openai:text-embedding-3-small:384"
    with pytest.raises(ConfigurationError, match="EMBEDDING_DEPLOYMENT"):
        build_embedder(make(embedding_provider=EmbeddingProvider.AZURE_OPENAI))
    azure = build_embedder(
        make(
            embedding_provider=EmbeddingProvider.AZURE_OPENAI,
            azure_openai_embedding_deployment="emb",
            azure_openai_endpoint="https://example.invalid",
            azure_openai_api_key=SecretStr("k"),
        )
    )
    assert azure.name == "azure_openai:emb:384"


def test_llm_selection() -> None:
    claude = build_llm(
        make(llm_provider=LlmProviderName.ANTHROPIC, anthropic_api_key=SecretStr("k"))
    )
    assert isinstance(claude, AnthropicProvider)
    assert claude.label == "anthropic:claude-opus-5-5"
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_llm(make(llm_provider=LlmProviderName.OPENAI))
    with pytest.raises(ConfigurationError, match="CHAT_DEPLOYMENT"):
        build_llm(make(llm_provider=LlmProviderName.AZURE_OPENAI))
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_ENDPOINT"):
        build_llm(make(llm_provider=LlmProviderName.AZURE_OPENAI, azure_openai_chat_deployment="d"))
    gpt = build_llm(make(llm_provider=LlmProviderName.OPENAI, openai_api_key=SecretStr("k")))
    assert isinstance(gpt, OpenAIChatProvider)


def test_fee_schedule_from_settings() -> None:
    fees = build_fee_schedule(make(mpf_min_usd="40", fee_schedule_fiscal_year="FY2027"))
    assert fees.mpf_min_usd == Decimal("40")
    assert fees.fiscal_year == "FY2027"


async def test_http_freight_requires_url() -> None:
    with pytest.raises(ConfigurationError, match="FREIGHT_API_URL"):
        await build_services(make(freight_provider="http"))


def test_sdk_clients_get_explicit_timeouts_and_retries() -> None:
    claude = build_llm(
        make(
            llm_provider=LlmProviderName.ANTHROPIC,
            anthropic_api_key=SecretStr("k"),
            llm_timeout_s=12,
            llm_max_retries=1,
        )
    )
    assert isinstance(claude, AnthropicProvider)
    assert (claude._client.timeout, claude._client.max_retries) == (12, 1)
    gpt = build_llm(make(llm_provider=LlmProviderName.OPENAI, openai_api_key=SecretStr("k")))
    assert isinstance(gpt, OpenAIChatProvider)
    assert (gpt._client.timeout, gpt._client.max_retries) == (30, 2)
    emb = build_embedder(
        make(embedding_provider=EmbeddingProvider.OPENAI, openai_api_key=SecretStr("k"))
    )
    assert isinstance(emb, OpenAIEmbedder)
    assert (emb._client.timeout, emb._client.max_retries) == (60, 3)


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


async def test_services_aclose_releases_every_client(
    settings: Settings, memory_store: InMemoryStore, embedder: HashEmbedder
) -> None:
    llm, http_client = _Closable(), _Closable()
    services = assemble(settings, store=memory_store, embedder=embedder)
    services.llm = llm  # type: ignore[assignment]
    services.http_client = http_client  # type: ignore[assignment]
    await services.aclose()
    assert llm.closed
    assert http_client.closed


async def test_build_services_closes_clients_when_the_store_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[httpx.AsyncClient] = []

    class _TrackedClient(httpx.AsyncClient):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)  # type: ignore[arg-type]
            created.append(self)

    async def failing_store(settings: Settings, embedder: object) -> None:
        raise StoreUnavailableError("db down")

    monkeypatch.setattr(httpx, "AsyncClient", _TrackedClient)
    monkeypatch.setattr(container, "build_store", failing_store)
    with pytest.raises(StoreUnavailableError):
        await build_services(make(freight_provider="http", freight_api_url="http://quotes.test"))
    assert len(created) == 1
    assert created[0].is_closed


async def test_aclose_all_closes_everything_then_reraises() -> None:
    class _Broken:
        async def aclose(self) -> None:
            raise RuntimeError("boom")

    ok = _Closable()
    with pytest.raises(RuntimeError, match="boom"):
        await aclose_all(_Broken(), None, ok)
    assert ok.closed
