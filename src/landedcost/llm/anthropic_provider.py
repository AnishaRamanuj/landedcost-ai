"""Claude via the official ``anthropic`` SDK.

* Structured outputs: ``output_config.format`` with a closed JSON Schema.
* ``output_config.effort`` controls thinking depth (Claude Opus 5.5 defaults to ``medium``;
  we set it explicitly). No ``temperature``/``top_p``/``top_k``: current models reject them.
* Server-side refusal fallbacks (beta ``server-side-fallback-2026-07-01``,
  ``fallbacks="default"``): if a safety classifier declines, the API retries on Anthropic's
  recommended fallback model inside the same call.
* ``stop_reason == "refusal"`` is checked before reading content and mapped to an
  abstention, never parsed as an answer.
"""

from __future__ import annotations

from typing import Any, Literal

import anthropic
from anthropic import Omit, omit
from anthropic.types.anthropic_beta_param import AnthropicBetaParam

from landedcost.llm.base import LlmProviderError, StructuredResult

Effort = Literal["low", "medium", "high", "xhigh", "max"]
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        model: str = "claude-opus-5-5",
        effort: Effort = "medium",
        max_tokens: int = 16000,
        use_fallbacks: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._effort: Effort = effort
        self._max_tokens = max_tokens
        self._use_fallbacks = use_fallbacks

    @property
    def label(self) -> str:
        return f"anthropic:{self._model}"

    async def aclose(self) -> None:
        """Close the SDK client's connection pool."""
        await self._client.close()

    async def generate_structured(
        self, *, system: str, user: str, schema: dict[str, Any], schema_name: str
    ) -> StructuredResult:
        betas: list[AnthropicBetaParam] | Omit = [FALLBACK_BETA] if self._use_fallbacks else omit
        fallbacks: Literal["default"] | Omit = "default" if self._use_fallbacks else omit
        try:
            response = await self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self._effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                betas=betas,
                fallbacks=fallbacks,
            )
        except anthropic.RateLimitError as exc:
            raise LlmProviderError(f"Anthropic rate limit: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LlmProviderError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LlmProviderError(f"Anthropic connection error: {exc}") from exc

        if response.stop_reason == "refusal":
            details = response.stop_details
            reason = "model declined the request"
            if details is not None and details.category:
                reason += f" (category: {details.category})"
            if details is not None and details.explanation:
                reason += f": {details.explanation}"
            return StructuredResult(None, response.model, refused=True, refusal_reason=reason)
        if response.stop_reason == "max_tokens":
            raise LlmProviderError("Anthropic response truncated at max_tokens")

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LlmProviderError("Anthropic response contained no text block")
        return StructuredResult(text, response.model)
