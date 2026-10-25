"""Provider-neutral interface for JSON-schema constrained generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class LlmProviderError(RuntimeError):
    """A provider call failed (network, rate limit, server error, truncation, ...).

    Transient failures (429, 5xx, connection errors) have already been retried by the SDK
    client (``LLM_MAX_RETRIES``, with backoff) by the time this is raised; the classifier
    then falls back to the extractive answer instead of retrying again.
    """


@dataclass(frozen=True, slots=True)
class StructuredResult:
    """Raw JSON text from the model, or a refusal."""

    text: str | None
    model: str
    refused: bool = False
    refusal_reason: str | None = None


class LlmProvider(Protocol):
    @property
    def label(self) -> str:
        """``provider:model``, reported in API responses."""
        ...

    async def generate_structured(
        self, *, system: str, user: str, schema: dict[str, Any], schema_name: str
    ) -> StructuredResult: ...


@runtime_checkable
class SupportsAclose(Protocol):
    """Adapters that own a network client (SDK or HTTP) and must release it."""

    async def aclose(self) -> None: ...


async def aclose_all(*resources: object) -> None:
    """Close every resource that supports it; one failure does not stop the others."""
    errors: list[BaseException] = []
    for resource in resources:
        if isinstance(resource, SupportsAclose):
            try:
                await resource.aclose()
            except Exception as exc:  # keep closing the rest, re-raise the first failure
                errors.append(exc)
    if errors:
        raise errors[0]
