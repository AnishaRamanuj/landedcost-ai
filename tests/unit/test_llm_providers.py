from typing import Any, cast

import anthropic
import httpx2
import openai
import pytest
from anthropic.types.beta import BetaMessage
from openai.types.chat import ChatCompletion

from landedcost.classify.models import LLM_OUTPUT_SCHEMA
from landedcost.llm.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from landedcost.llm.base import LlmProviderError
from landedcost.llm.openai_provider import OpenAIChatProvider

REQUEST = httpx2.Request("POST", "https://api.example.invalid/v1/messages")


def beta_message(stop_reason: str, content: list[dict[str, Any]], **extra: Any) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
            **extra,
        }
    )


class _FakeAnthropicMessages:
    def __init__(self, outcome: BetaMessage | Exception) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> BetaMessage:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class _FakeAnthropic:
    def __init__(self, outcome: BetaMessage | Exception) -> None:
        self.messages = _FakeAnthropicMessages(outcome)
        self.beta = self


def anthropic_provider(
    outcome: BetaMessage | Exception, **kwargs: Any
) -> tuple[AnthropicProvider, _FakeAnthropicMessages]:
    fake = _FakeAnthropic(outcome)
    return AnthropicProvider(cast(anthropic.AsyncAnthropic, fake), **kwargs), fake.messages


async def generate(provider: AnthropicProvider | OpenAIChatProvider) -> Any:
    return await provider.generate_structured(
        system="sys", user="usr", schema=LLM_OUTPUT_SCHEMA, schema_name="hts_classification"
    )


async def test_anthropic_request_shape() -> None:
    provider, messages = anthropic_provider(
        beta_message("end_turn", [{"type": "text", "text": "{}"}])
    )
    result = await generate(provider)
    assert result.text == "{}"
    assert not result.refused
    kw = messages.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": LLM_OUTPUT_SCHEMA},
    }
    assert kw["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    assert kw["system"] == "sys"
    assert kw["messages"] == [{"role": "user", "content": "usr"}]
    assert not {"temperature", "top_p", "top_k"} & set(kw)
    assert provider.label == "anthropic:claude-opus-5-5"


async def test_anthropic_without_fallbacks_omits_beta_parameters() -> None:
    provider, messages = anthropic_provider(
        beta_message("end_turn", [{"type": "text", "text": "{}"}]), use_fallbacks=False, model="m"
    )
    await generate(provider)
    assert isinstance(messages.kwargs["betas"], anthropic.Omit)
    assert isinstance(messages.kwargs["fallbacks"], anthropic.Omit)


async def test_anthropic_refusal_is_reported_not_parsed() -> None:
    message = beta_message(
        "refusal",
        [],
        stop_details={"type": "refusal", "category": "cyber", "explanation": "declined"},
    )
    provider, _ = anthropic_provider(message)
    result = await generate(provider)
    assert result.refused
    assert result.text is None
    assert result.refusal_reason == "model declined the request (category: cyber): declined"


async def test_anthropic_reads_first_text_block_after_thinking() -> None:
    message = beta_message(
        "end_turn",
        [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": '{"a":1}'},
        ],
    )
    provider, _ = anthropic_provider(message)
    assert (await generate(provider)).text == '{"a":1}'


@pytest.mark.parametrize(
    ("message", "match"),
    [
        (beta_message("max_tokens", [{"type": "text", "text": "{"}]), "truncated"),
        (beta_message("end_turn", []), "no text block"),
    ],
)
async def test_anthropic_unusable_responses_raise(message: BetaMessage, match: str) -> None:
    provider, _ = anthropic_provider(message)
    with pytest.raises(LlmProviderError, match=match):
        await generate(provider)


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (
            anthropic.RateLimitError(
                "slow down", response=httpx2.Response(429, request=REQUEST), body=None
            ),
            "rate limit",
        ),
        (
            anthropic.InternalServerError(
                "boom", response=httpx2.Response(500, request=REQUEST), body=None
            ),
            "API error 500",
        ),
        (
            anthropic.BadRequestError(
                "bad", response=httpx2.Response(400, request=REQUEST), body=None
            ),
            "API error 400",
        ),
        (anthropic.APIConnectionError(request=REQUEST), "connection error"),
        (anthropic.APITimeoutError(request=REQUEST), "connection error"),
    ],
)
async def test_anthropic_errors_are_mapped(error: Exception, match: str) -> None:
    provider, _ = anthropic_provider(error)
    with pytest.raises(LlmProviderError, match=match):
        await generate(provider)


# -- OpenAI / Azure OpenAI ------------------------------------------------------------------


def chat_completion(
    content: str | None, finish_reason: str = "stop", refusal: str | None = None
) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content, "refusal": refusal},
                }
            ],
        }
    )


class _FakeOpenAI:
    def __init__(self, outcome: ChatCompletion | Exception) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, Any] = {}
        self.chat = self
        self.completions = self

    async def create(self, **kwargs: Any) -> ChatCompletion:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def openai_provider(outcome: ChatCompletion | Exception) -> tuple[OpenAIChatProvider, _FakeOpenAI]:
    fake = _FakeOpenAI(outcome)
    return OpenAIChatProvider(
        cast(openai.AsyncOpenAI, fake), model="dep", provider="azure_openai"
    ), fake


async def test_openai_request_uses_strict_json_schema() -> None:
    provider, fake = openai_provider(chat_completion("{}"))
    result = await generate(provider)
    assert result.text == "{}"
    fmt = fake.kwargs["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"] == {
        "name": "hts_classification",
        "schema": LLM_OUTPUT_SCHEMA,
        "strict": True,
    }
    assert fake.kwargs["model"] == "dep"
    assert [m["role"] for m in fake.kwargs["messages"]] == ["system", "user"]
    assert provider.label == "azure_openai:dep"


@pytest.mark.parametrize(
    ("completion", "reason"),
    [
        (chat_completion(None, refusal="I can't help with that"), "I can't help with that"),
        (chat_completion(None, finish_reason="content_filter"), "blocked by content filter"),
    ],
)
async def test_openai_refusals(completion: ChatCompletion, reason: str) -> None:
    provider, _ = openai_provider(completion)
    result = await generate(provider)
    assert result.refused
    assert result.refusal_reason == reason


async def test_openai_truncation_and_errors() -> None:
    provider, _ = openai_provider(chat_completion("{", finish_reason="length"))
    with pytest.raises(LlmProviderError, match="truncated"):
        await generate(provider)
    err = openai.RateLimitError("slow", response=httpx2.Response(429, request=REQUEST), body=None)
    provider, _ = openai_provider(err)
    with pytest.raises(LlmProviderError, match="rate limit"):
        await generate(provider)
    provider, _ = openai_provider(openai.APIConnectionError(request=REQUEST))
    with pytest.raises(LlmProviderError):
        await generate(provider)
