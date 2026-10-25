"""Test doubles shared across the suite."""

from __future__ import annotations

import json
from typing import Any

from landedcost.llm.base import LlmProviderError, StructuredResult


class FakeLlm:
    """Returns a scripted response and records every request."""

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        text: str | None = None,
        refusal: str | None = None,
        error: LlmProviderError | None = None,
    ) -> None:
        self._payload = payload
        self._text = text
        self._refusal = refusal
        self._error = error
        self.calls: list[dict[str, Any]] = []

    @property
    def label(self) -> str:
        return "fake:test-model"

    async def generate_structured(
        self, *, system: str, user: str, schema: dict[str, Any], schema_name: str
    ) -> StructuredResult:
        self.calls.append({"system": system, "user": user, "schema": schema, "name": schema_name})
        if self._error is not None:
            raise self._error
        if self._refusal is not None:
            return StructuredResult(None, "fake", refused=True, refusal_reason=self._refusal)
        text = self._text if self._text is not None else json.dumps(self._payload)
        return StructuredResult(text, "fake")
