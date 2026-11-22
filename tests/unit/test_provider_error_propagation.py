"""
RateLimitError and ContextOverflowError must escape every HTTP provider.

A provider that swallows RateLimitError turns a 429 storm into a failed chunk
instead of the pipeline's pause-and-checkpoint path; one that swallows
ContextOverflowError prevents the chunk from being split. Each provider is
driven through a fake httpx transport, so no network is used.
"""

import asyncio

import httpx
import pytest

from src.core.llm.exceptions import ContextOverflowError, RateLimitError

FAKE_KEY = "YOUR_API_KEY_HERE"


async def _no_sleep(*_args, **_kwargs):
    return None


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skip every backoff sleep (handler and provider modules share asyncio)."""
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


def _build(name):
    from src.core.llm.providers.deepseek import DeepSeekProvider
    from src.core.llm.providers.gemini import GeminiProvider
    from src.core.llm.providers.mistral import MistralProvider
    from src.core.llm.providers.ollama import OllamaProvider
    from src.core.llm.providers.openai import OpenAICompatibleProvider
    from src.core.llm.providers.openrouter import OpenRouterProvider
    from src.core.llm.providers.poe import PoeProvider

    builders = {
        "mistral": lambda: MistralProvider(api_key=FAKE_KEY, model="mistral-small"),
        "deepseek": lambda: DeepSeekProvider(api_key=FAKE_KEY, model="deepseek-chat",
                                             disable_thinking=False),
        "poe": lambda: PoeProvider(api_key=FAKE_KEY, model="GPT-4o"),
        "openrouter": lambda: OpenRouterProvider(api_key=FAKE_KEY, model="openai/gpt-4o",
                                                 disable_thinking=False),
        "gemini": lambda: GeminiProvider(api_key=FAKE_KEY, model="gemini-2.0-flash"),
        "openai": lambda: OpenAICompatibleProvider(
            api_endpoint="https://api.example.test/v1/chat/completions",
            model="gpt-4o", api_key=FAKE_KEY),
        "openai-keyless": lambda: OpenAICompatibleProvider(
            api_endpoint="http://localhost:8080/v1/chat/completions", model="local"),
        "ollama": lambda: OllamaProvider(
            api_endpoint="http://localhost:11434/api/generate", model="llama3",
            context_window=2048),
    }
    return builders[name]()


def _attach_transport(provider, handler):
    calls = []

    def wrapped(request):
        calls.append(request)
        return handler(request)

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return calls


HTTP_PROVIDERS = ["mistral", "deepseek", "poe", "openrouter", "gemini",
                  "openai", "openai-keyless", "ollama"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", HTTP_PROVIDERS)
async def test_persistent_429_raises_rate_limit_error(name):
    provider = _build(name)
    calls = _attach_transport(
        provider,
        lambda request: httpx.Response(429, headers={"Retry-After": "1"},
                                       json={"error": "Too Many Requests"}),
    )
    try:
        with pytest.raises(RateLimitError):
            await provider.generate("Translate this")
    finally:
        await provider.close()
    assert calls, "the provider never sent a request"


OVERFLOW_BODY = {
    "error": {
        "message": "This model's maximum context length is 4096 tokens. "
                   "However, your messages resulted in 9000 tokens.",
    }
}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", HTTP_PROVIDERS)
async def test_context_overflow_propagates(name):
    provider = _build(name)
    _attach_transport(provider, lambda request: httpx.Response(400, json=OVERFLOW_BODY))
    try:
        with pytest.raises(ContextOverflowError):
            await provider.generate("Translate this")
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_keyless_openai_429_honours_retry_after_then_raises(monkeypatch):
    """A local/keyless OpenAI-compatible server has no key pool to rotate:
    each retry waits the server's Retry-After (capped), and a 429 that
    persists becomes a RateLimitError carrying the full hint."""
    from src.core.llm.providers import openai as openai_module
    from src.core.llm.providers.openai import OpenAICompatibleProvider

    waits = []

    async def record_sleep(seconds, *_args, **_kwargs):
        waits.append(seconds)

    monkeypatch.setattr(openai_module.asyncio, "sleep", record_sleep)
    provider = OpenAICompatibleProvider(
        api_endpoint="http://localhost:8080/v1/chat/completions", model="local",
    )
    _attach_transport(
        provider,
        lambda request: httpx.Response(429, headers={"Retry-After": "120"},
                                       json={"error": "slow down"}),
    )
    try:
        with pytest.raises(RateLimitError) as info:
            await provider.generate("Translate this")
    finally:
        await provider.close()
    assert waits and all(w == openai_module.MAX_HINTED_RETRY_WAIT for w in waits)
    assert info.value.retry_after == 120


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, expect",
    [
        ({"error": {"code": 429, "message": "rate limited upstream"}}, RateLimitError),
        ({"error": {"code": 402, "message": "insufficient credits"}}, None),
        ({"choices": [{"message": {"content": "half a transl"}, "finish_reason": "error"}],
          "error": {"code": 502, "message": "upstream died"}}, None),
    ],
    ids=["rate-limit", "no-credit", "mid-answer-error"],
)
async def test_openrouter_error_inside_200_answer(body, expect):
    """OpenRouter reports late failures in a 200 body; none may come back as
    translated text."""
    from src.core.llm.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider(api_key=FAKE_KEY, model="openai/gpt-4o")
    _attach_transport(provider, lambda request: httpx.Response(200, json=body))
    try:
        if expect is RateLimitError:
            with pytest.raises(RateLimitError):
                await provider.generate("Translate this")
        else:
            result = await provider.generate("Translate this")
            assert result is None or not getattr(result, "content", "")
    finally:
        await provider.close()
