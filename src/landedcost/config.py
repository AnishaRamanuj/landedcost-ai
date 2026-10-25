"""Runtime configuration from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from landedcost.retrieval.hybrid import FTS, RULINGS, VECTOR, RetrievalMode

DEFAULT_SNAPSHOT = Path("data/hts/hts_2026-09-04.json.gz")
DEFAULT_EVAL_DATASET = Path("data/eval/starter_eval.jsonl")
SOURCE_ROOT = Path(__file__).resolve().parents[2]
"""Repository root when running from a source checkout (``src/landedcost/config.py``)."""


def resolve_data_path(path: Path) -> Path:
    """Resolve a relative data path against the working directory, falling back to the
    repository root, so commands also work when started outside the checkout."""
    if path.is_absolute() or path.exists():
        return path
    candidate = SOURCE_ROOT / path
    return candidate if candidate.exists() else path


class StoreBackend(StrEnum):
    POSTGRES = "postgres"
    MEMORY = "memory"


class EmbeddingProvider(StrEnum):
    HASH = "hash"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    FASTEMBED = "fastembed"


class LlmProviderName(StrEnum):
    NONE = "none"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"


class FreightProviderName(StrEnum):
    STATIC = "static"
    HTTP = "http"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # storage
    store_backend: StoreBackend = StoreBackend.POSTGRES
    database_url: str = "postgresql://dev:dev@localhost:5432/landedcost"
    hts_snapshot_path: Path = DEFAULT_SNAPSHOT
    hnsw_ef_search: int = Field(default=400, ge=1, le=1000)
    fts_ranking: Literal["bm25", "ts_rank_cd"] = "bm25"
    db_pool_min_size: int = Field(default=1, ge=0, le=100)
    db_pool_max_size: int = Field(default=10, ge=1, le=100)
    db_connect_timeout_s: float = Field(default=5.0, gt=0, le=120)

    # embeddings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HASH
    embedding_dim: int = Field(default=384, ge=1, le=2000)
    openai_embedding_model: str = "text-embedding-3-small"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    embedding_timeout_s: float = Field(default=60.0, gt=0, le=600)
    embedding_max_retries: int = Field(default=3, ge=0, le=10)

    # retrieval + gate
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    rrf_k: int = Field(default=60, ge=0)
    rrf_weight_vector: float = Field(default=1.0, gt=0, le=10)
    rrf_weight_fts: float = Field(default=1.0, gt=0, le=10)
    rrf_weight_rulings: float = Field(default=1.0, gt=0, le=10)
    retrieval_candidate_pool: int = Field(default=50, ge=1, le=500)
    gate_min_support: float = Field(default=0.9, ge=0.0, le=1.0)
    gate_min_heading_consensus: float = Field(default=0.3, ge=0.0, le=1.0)
    gate_window: int = Field(default=10, ge=1)

    # LLM
    llm_provider: LlmProviderName = LlmProviderName.NONE
    llm_timeout_s: float = Field(default=30.0, gt=0, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    anthropic_use_fallbacks: bool = True
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    # landed cost (defaults: FY2027, CBP Dec. 26-14; see landedcost.landed.fees)
    fee_schedule_fiscal_year: str = Field(default="FY2027", pattern=r"^FY\d{4}$")
    mpf_rate: Decimal = Decimal("0.003464")
    mpf_min_usd: Decimal = Decimal("34.58")
    mpf_max_usd: Decimal = Decimal("670.86")
    hmf_rate: Decimal = Decimal("0.00125")
    freight_provider: FreightProviderName = FreightProviderName.STATIC
    freight_api_url: str | None = None
    freight_api_timeout_s: float = Field(default=10.0, gt=0)

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8201

    @field_validator("hts_snapshot_path")
    @classmethod
    def _resolve_snapshot(cls, value: Path) -> Path:
        return resolve_data_path(value)

    @model_validator(mode="after")
    def _check_pool(self) -> Self:
        if self.db_pool_min_size > self.db_pool_max_size:
            raise ValueError("DB_POOL_MIN_SIZE must not exceed DB_POOL_MAX_SIZE")
        return self

    @property
    def rrf_weights(self) -> dict[str, float]:
        return {
            VECTOR: self.rrf_weight_vector,
            FTS: self.rrf_weight_fts,
            RULINGS: self.rrf_weight_rulings,
        }
