"""OpenAI and Azure OpenAI via the official ``openai`` SDK (Chat Completions + JSON Schema)."""

from __future__ import annotations

from typing import Any

import openai

from landedcost.llm.base import LlmProviderError, StructuredResult


class OpenAIChatProvider:
    """Works with ``AsyncOpenAI`` and ``AsyncAzureOpenAI``.

    For Azure, ``model`` is the chat *deployment* name and ``provider`` should be
    ``"azure_openai"`` so responses report where the answer came from.
    """

    def __init__(self, client: openai.AsyncOpenAI, *, model: str, provider: str = "openai") -> None:
        self._client = client
        self._model = model
        self._provider = provider

    @property
    def label(self) -> str:
        return f"{self._provider}:{self._model}"

    async def aclose(self) -> None:
        """Close the SDK client's connection pool."""
        await self._client.close()

    async def generate_structured(
        self, *, system: str, user: str, schema: dict[str, Any], schema_name: str
    ) -> StructuredResult:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                },
            )
        except openai.RateLimitError as exc:
            raise LlmProviderError(f"{self._provider} rate limit: {exc.message}") from exc
        except openai.APIStatusError as exc:
            raise LlmProviderError(
                f"{self._provider} API error {exc.status_code}: {exc.message}"
            ) from exc
        except openai.APIConnectionError as exc:
            raise LlmProviderError(f"{self._provider} connection error: {exc}") from exc

        if not response.choices:
            raise LlmProviderError(f"{self._provider} returned no choices")
        choice = response.choices[0]
        if choice.message.refusal:
            return StructuredResult(
                None, response.model, refused=True, refusal_reason=choice.message.refusal
            )
        if choice.finish_reason == "content_filter":
            return StructuredResult(
                None, response.model, refused=True, refusal_reason="blocked by content filter"
            )
        if choice.finish_reason == "length":
            raise LlmProviderError(f"{self._provider} response truncated")
        if not choice.message.content:
            raise LlmProviderError(f"{self._provider} returned empty content")
        return StructuredResult(choice.message.content, response.model)
