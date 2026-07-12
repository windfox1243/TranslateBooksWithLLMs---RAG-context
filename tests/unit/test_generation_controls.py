import httpx
import pytest

from src.core.llm import LLMGenerationOptions, LLMResponse
from src.core.llm.generation_controls import (
    adaptive_retry_output_tokens,
    generation_capabilities,
    resolve_editor_output_tokens,
    resolve_thinking_controls,
)
from src.core.llm.providers.openai import OpenAICompatibleProvider
from src.core.llm.providers.openrouter import OpenRouterProvider


def test_gemini_3_editor_auto_uses_minimal_thinking():
    capabilities = generation_capabilities(
        "gemini", "gemini-3-flash-preview"
    )
    assert capabilities.thinking_control == "level"
    assert capabilities.default_thinking_mode == "high"
    assert "minimal" in capabilities.thinking_modes
    assert resolve_thinking_controls(
        "gemini", "gemini-3-flash-preview", "auto", role="editor"
    ) == {
        "mode": "minimal",
        "thinking_level": "minimal",
        "thinking_budget": None,
    }
    assert resolve_thinking_controls(
        "gemini", "gemini-3-flash-preview", "auto", role="draft"
    )["thinking_level"] is None


def test_gemini_25_and_output_budgets_are_bounded():
    controls = resolve_thinking_controls(
        "gemini", "gemini-2.5-flash", "auto", role="editor"
    )
    assert controls["mode"] == "off"
    assert controls["thinking_budget"] == 0
    assert resolve_editor_output_tokens(
        "gemini", "gemini-3-flash-preview", "auto", "minimal"
    ) == 4096
    assert resolve_editor_output_tokens(
        "gemini", "gemini-3-flash-preview", "model_max", "high",
        reported_limit=32768,
    ) == 32768
    assert resolve_editor_output_tokens(
        "gemini", "gemini-3-flash-preview", "999999", "high",
        reported_limit=65536,
    ) == 65536
    assert adaptive_retry_output_tokens(4096, 65536) == 8192


def test_other_provider_capabilities_are_conservative():
    assert generation_capabilities(
        "ollama", "gpt-oss:20b"
    ).thinking_modes == ("auto", "low", "medium", "high")
    assert generation_capabilities(
        "ollama", "qwen3:8b"
    ).can_disable_thinking is True
    assert generation_capabilities(
        "openai", "o4-mini", endpoint="https://api.openai.com/v1"
    ).thinking_control == "effort"
    assert generation_capabilities(
        "openai", "o4-mini", endpoint="http://localhost:1234/v1"
    ).thinking_supported is False


@pytest.mark.parametrize(
    "finish_reason",
    ["MAX_TOKENS", "length", "max_output_tokens", "token-limit"],
)
def test_provider_finish_reasons_normalize_truncation(finish_reason):
    assert LLMResponse(content="partial", finish_reason=finish_reason).was_truncated
    assert generation_capabilities(
        "mistral", "mistral-large-latest"
    ).thinking_supported is False


class _JsonResponse:
    def __init__(self, url, payload):
        self._payload = payload
        self.headers = {}
        self.request = httpx.Request("POST", url)
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_openai_reasoning_effort_and_tokens(monkeypatch):
    provider = OpenAICompatibleProvider(
        api_endpoint="https://api.openai.com/v1",
        model="o4-mini",
        api_key="YOUR_API_KEY_HERE",
    )
    sent = []

    class Client:
        async def post(self, url, json=None, headers=None, timeout=None):
            sent.append(json)
            return _JsonResponse(url, {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "completion_tokens_details": {"reasoning_tokens": 12},
                },
            })

    async def get_client():
        return Client()

    monkeypatch.setattr(provider, "_get_client", get_client)
    response = await provider.generate(
        "hello",
        generation_options=LLMGenerationOptions(
            reasoning_effort="low", max_output_tokens=8192,
        ),
    )
    assert sent[0]["reasoning_effort"] == "low"
    assert sent[0]["max_completion_tokens"] == 8192
    assert "max_tokens" not in sent[0]
    assert response.thinking_tokens == 12
    assert response.total_tokens == 30


@pytest.mark.asyncio
async def test_openrouter_reasoning_effort_and_tokens(monkeypatch):
    provider = OpenRouterProvider(
        api_key="YOUR_API_KEY_HERE",
        model="openai/o4-mini",
    )
    sent = []

    class Client:
        async def post(self, url, json=None, headers=None, timeout=None):
            sent.append(json)
            return _JsonResponse(url, {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "completion_tokens_details": {"reasoning_tokens": 9},
                },
            })

    async def get_client():
        return Client()

    monkeypatch.setattr(provider, "_get_client", get_client)
    response = await provider.generate(
        "hello",
        generation_options=LLMGenerationOptions(reasoning_effort="minimal"),
    )
    assert sent[0]["reasoning"] == {"effort": "minimal", "exclude": True}
    assert "thinking" not in sent[0]
    assert response.thinking_tokens == 9
    assert response.total_tokens == 30
