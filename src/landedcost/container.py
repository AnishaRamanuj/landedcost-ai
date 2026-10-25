"""Composition root: build every service from :class:`Settings` in one place."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx
from pydantic import SecretStr

from landedcost.classify.gate import EvidenceGate, GateConfig
from landedcost.classify.service import Classifier, ClassifierConfig
from landedcost.config import (
    EmbeddingProvider,
    FreightProviderName,
    LlmProviderName,
    Settings,
    StoreBackend,
)
from landedcost.embeddings.base import Embedder
from landedcost.embeddings.hashing import HashEmbedder
from landedcost.hts.loader import load_lines
from landedcost.landed.calculator import LandedCostService
from landedcost.landed.fees import FeeSchedule
from landedcost.landed.freight import (
    FreightQuoteProvider,
    HttpFreightQuoteProvider,
    StaticFreightQuoteProvider,
)
from landedcost.llm.base import LlmProvider, aclose_all
from landedcost.retrieval.hybrid import HybridRetriever, RetrieverConfig
from landedcost.store.base import Store
from landedcost.store.memory import InMemoryStore

if TYPE_CHECKING:
    from openai import AsyncAzureOpenAI

    from landedcost.store.postgres import PostgresStore


class ConfigurationError(RuntimeError):
    """Settings are incomplete for the selected providers."""


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value is not None else None


def build_embedder(settings: Settings) -> Embedder:
    match settings.embedding_provider:
        case EmbeddingProvider.HASH:
            return HashEmbedder(dim=settings.embedding_dim)
        case EmbeddingProvider.OPENAI:
            from openai import AsyncOpenAI

            from landedcost.embeddings.openai_embedder import OpenAIEmbedder

            key = _secret(settings.openai_api_key)
            if not key:
                raise ConfigurationError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
            return OpenAIEmbedder(
                AsyncOpenAI(
                    api_key=key,
                    timeout=settings.embedding_timeout_s,
                    max_retries=settings.embedding_max_retries,
                ),
                model=settings.openai_embedding_model,
                dim=settings.embedding_dim,
            )
        case EmbeddingProvider.AZURE_OPENAI:
            from landedcost.embeddings.openai_embedder import OpenAIEmbedder

            deployment = settings.azure_openai_embedding_deployment
            if not deployment:
                raise ConfigurationError(
                    "EMBEDDING_PROVIDER=azure_openai requires AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
                )
            return OpenAIEmbedder(
                _azure_client(
                    settings,
                    timeout_s=settings.embedding_timeout_s,
                    max_retries=settings.embedding_max_retries,
                ),
                model=deployment,
                dim=settings.embedding_dim,
                provider="azure_openai",
            )
        case EmbeddingProvider.FASTEMBED:
            from landedcost.embeddings.fastembed_embedder import FastEmbedEmbedder

            return FastEmbedEmbedder(settings.fastembed_model, settings.embedding_dim)


def _azure_client(settings: Settings, *, timeout_s: float, max_retries: int) -> AsyncAzureOpenAI:
    from openai import AsyncAzureOpenAI

    key = _secret(settings.azure_openai_api_key)
    if not settings.azure_openai_endpoint or not key:
        raise ConfigurationError(
            "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
        )
    return AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=key,
        api_version=settings.azure_openai_api_version,
        timeout=timeout_s,
        max_retries=max_retries,
    )


def build_llm(settings: Settings) -> LlmProvider | None:
    """Build the configured LLM provider.

    Every SDK client gets an explicit timeout and retry budget (``LLM_TIMEOUT_S``,
    ``LLM_MAX_RETRIES``): the SDK defaults (10 minutes, 2 retries) would let one
    ``/classify`` request hang for half an hour before falling back to the extractive answer.
    """
    timeout, retries = settings.llm_timeout_s, settings.llm_max_retries
    match settings.llm_provider:
        case LlmProviderName.NONE:
            return None
        case LlmProviderName.ANTHROPIC:
            import anthropic

            from landedcost.llm.anthropic_provider import AnthropicProvider

            # With no explicit key the SDK resolves credentials from its own environment.
            key = _secret(settings.anthropic_api_key)
            client = (
                anthropic.AsyncAnthropic(api_key=key, timeout=timeout, max_retries=retries)
                if key
                else anthropic.AsyncAnthropic(timeout=timeout, max_retries=retries)
            )
            return AnthropicProvider(
                client,
                model=settings.anthropic_model,
                effort=settings.anthropic_effort,
                use_fallbacks=settings.anthropic_use_fallbacks,
            )
        case LlmProviderName.OPENAI:
            from openai import AsyncOpenAI

            from landedcost.llm.openai_provider import OpenAIChatProvider

            key = _secret(settings.openai_api_key)
            if not key:
                raise ConfigurationError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
            return OpenAIChatProvider(
                AsyncOpenAI(api_key=key, timeout=timeout, max_retries=retries),
                model=settings.openai_model,
            )
        case LlmProviderName.AZURE_OPENAI:
            from landedcost.llm.openai_provider import OpenAIChatProvider

            deployment = settings.azure_openai_chat_deployment
            if not deployment:
                raise ConfigurationError(
                    "LLM_PROVIDER=azure_openai requires AZURE_OPENAI_CHAT_DEPLOYMENT"
                )
            return OpenAIChatProvider(
                _azure_client(settings, timeout_s=timeout, max_retries=retries),
                model=deployment,
                provider="azure_openai",
            )


def build_fee_schedule(settings: Settings) -> FeeSchedule:
    return FeeSchedule(
        fiscal_year=settings.fee_schedule_fiscal_year,
        mpf_rate=settings.mpf_rate,
        mpf_min_usd=settings.mpf_min_usd,
        mpf_max_usd=settings.mpf_max_usd,
        hmf_rate=settings.hmf_rate,
    )


@dataclass
class Services:
    """Everything the API and CLI need, wired together."""

    settings: Settings
    store: Store
    embedder: Embedder
    retriever: HybridRetriever
    gate: EvidenceGate
    classifier: Classifier
    landed_cost: LandedCostService
    llm: LlmProvider | None
    http_client: httpx.AsyncClient | None = field(default=None, repr=False)

    async def aclose(self) -> None:
        """Release the store's pool and every network client (SDK and freight HTTP)."""
        try:
            await self.store.close()
        finally:
            await aclose_all(self.embedder, self.llm, self.http_client)


async def connect_postgres(settings: Settings) -> PostgresStore:
    """Open the PostgreSQL store with the pool and search settings from ``settings``."""
    from landedcost.store.postgres import PostgresStore

    return await PostgresStore.connect(
        settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        open_timeout_s=settings.db_connect_timeout_s,
        ef_search=settings.hnsw_ef_search,
        fts_ranking=settings.fts_ranking,
    )


async def build_store(settings: Settings, embedder: Embedder) -> Store:
    if settings.store_backend is StoreBackend.POSTGRES:
        return await connect_postgres(settings)
    store = InMemoryStore()
    snapshot, parsed = load_lines(settings.hts_snapshot_path)
    vectors = await embedder.embed_documents([line.embedding_text for line in parsed.lines])
    await store.replace_hts_lines(
        parsed.lines, vectors, hts_version=snapshot.version, embedder=embedder.name
    )
    return store


def assemble(
    settings: Settings,
    *,
    store: Store,
    embedder: Embedder,
    llm: LlmProvider | None = None,
    freight: FreightQuoteProvider | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Services:
    """Wire services around already-built adapters (used by tests to inject fakes)."""
    retriever = HybridRetriever(
        store,
        embedder,
        RetrieverConfig(
            rrf_k=settings.rrf_k,
            candidate_pool=settings.retrieval_candidate_pool,
            rrf_weights=settings.rrf_weights,
        ),
    )
    gate = EvidenceGate(
        GateConfig(
            min_support=settings.gate_min_support,
            min_heading_consensus=settings.gate_min_heading_consensus,
            window=settings.gate_window,
        )
    )
    classifier = Classifier(retriever, gate, llm, ClassifierConfig(mode=settings.retrieval_mode))
    landed = LandedCostService(
        store, freight or StaticFreightQuoteProvider(), build_fee_schedule(settings)
    )
    return Services(
        settings=settings,
        store=store,
        embedder=embedder,
        retriever=retriever,
        gate=gate,
        classifier=classifier,
        landed_cost=landed,
        llm=llm,
        http_client=http_client,
    )


async def build_services(settings: Settings) -> Services:
    """Build every adapter from ``settings``; nothing is leaked if a later step fails."""
    embedder = build_embedder(settings)
    llm: LlmProvider | None = None
    http_client: httpx.AsyncClient | None = None
    try:
        llm = build_llm(settings)
        freight: FreightQuoteProvider = StaticFreightQuoteProvider()
        if settings.freight_provider is FreightProviderName.HTTP:
            if not settings.freight_api_url:
                raise ConfigurationError("FREIGHT_PROVIDER=http requires FREIGHT_API_URL")
            http_client = httpx.AsyncClient(timeout=settings.freight_api_timeout_s)
            freight = HttpFreightQuoteProvider(http_client, settings.freight_api_url)
        store = await build_store(settings, embedder)
    except BaseException:
        await aclose_all(embedder, llm, http_client)
        raise
    return assemble(
        settings, store=store, embedder=embedder, llm=llm, freight=freight, http_client=http_client
    )
