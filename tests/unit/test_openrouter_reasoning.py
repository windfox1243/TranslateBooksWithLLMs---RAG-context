"""
Unit tests for the OpenRouter provider's reasoning override.

OpenRouter takes one unified `reasoning` request parameter and describes each
model's reasoning support in /api/v1/models. Mandatory-reasoning models reject
a disable request, so the override is derived from that metadata. These tests
pin the mapping using real catalog entries; the live behaviour is covered by
tests/standalone/manual_openrouter_reasoning.py.
"""

import pytest

from src.core.llm.providers.openrouter import OpenRouterProvider


def _provider(model, disable_thinking=True):
    return OpenRouterProvider(
        api_key="sk-or-xxxxxxxx",
        model=model,
        disable_thinking=disable_thinking,
    )


# Real entries (trimmed to the fields used) from GET https://openrouter.ai/api/v1/models
DEEPSEEK_V41_FLASH = {
    "id": "deepseek/deepseek-v4.1-flash",
    "supported_parameters": ["include_reasoning", "reasoning", "reasoning_effort", "temperature"],
    "reasoning": {
        "mandatory": False,
        "default_enabled": True,
        "supported_efforts": ["max", "high", "low"],
        "default_effort": "high",
    },
}

QWEN_37_FLASH = {
    "id": "qwen/qwen3.7-flash",
    "supported_parameters": ["include_reasoning", "reasoning", "temperature"],
    "reasoning": {"mandatory": False, "default_enabled": True, "supports_max_tokens": True},
}

GLM_53_PRIME = {
    "id": "z-ai/glm-5.3-prime",
    "supported_parameters": ["include_reasoning", "reasoning", "reasoning_effort"],
    "reasoning": {
        "mandatory": True,
        "default_enabled": True,
        "supported_efforts": ["max", "high", "low"],
        "default_effort": "max",
    },
}

PERCEPTRON_MK15 = {
    "id": "perceptron/perceptron-mk1.5",
    "supported_parameters": ["reasoning", "reasoning_effort"],
    "reasoning": {
        "mandatory": True,
        "supported_efforts": ["high", "medium", "low", "minimal", "none"],
        "default_effort": "high",
    },
}

MANDATORY_BUDGET_ONLY = {
    "id": "example/budget-only",
    "supported_parameters": ["reasoning"],
    "reasoning": {"mandatory": True, "supports_max_tokens": True},
}

NO_REASONING = {
    "id": "meta-llama/llama-3.3-70b-instruct",
    "supported_parameters": ["temperature", "top_p", "max_tokens"],
}


@pytest.mark.parametrize("model_info, expected", [
    (DEEPSEEK_V41_FLASH, {"enabled": False}),
    (QWEN_37_FLASH, {"enabled": False}),
    (GLM_53_PRIME, {"effort": "low"}),
    (NO_REASONING, {}),
])
def test_picks_the_right_override(model_info, expected):
    provider = _provider(model_info["id"])
    assert provider._pick_reasoning_override(model_info) == expected


def test_mandatory_models_never_get_none():
    # "none" is advertised but a mandatory model rejects it: next lowest wins.
    provider = _provider(PERCEPTRON_MK15["id"])
    assert provider._pick_reasoning_override(PERCEPTRON_MK15) == {"effort": "minimal"}


def test_mandatory_without_effort_list_keeps_the_default():
    provider = _provider(MANDATORY_BUDGET_ONLY["id"])
    assert provider._pick_reasoning_override(MANDATORY_BUDGET_ONLY) == {}


def _catalog_of(*entries):
    async def catalog():
        return {e["id"]: e for e in entries}
    return catalog


@pytest.mark.asyncio
async def test_opting_out_sends_nothing_and_skips_the_catalog(monkeypatch):
    provider = _provider(DEEPSEEK_V41_FLASH["id"], disable_thinking=False)

    async def fail():
        raise AssertionError("catalog must not be fetched")

    monkeypatch.setattr(provider, "_load_model_catalog", fail)
    assert await provider._get_reasoning_override() == {}


@pytest.mark.asyncio
async def test_override_is_resolved_once_per_model(monkeypatch):
    monkeypatch.setattr(OpenRouterProvider, "_reasoning_overrides", {})
    calls = []

    async def catalog():
        calls.append(1)
        return {DEEPSEEK_V41_FLASH["id"]: DEEPSEEK_V41_FLASH}

    provider = _provider(DEEPSEEK_V41_FLASH["id"])
    monkeypatch.setattr(provider, "_load_model_catalog", catalog)

    assert await provider._get_reasoning_override() == {"enabled": False}
    assert await provider._get_reasoning_override() == {"enabled": False}
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_unreachable_catalog_leaves_the_model_default(monkeypatch):
    monkeypatch.setattr(OpenRouterProvider, "_reasoning_overrides", {})
    provider = _provider(DEEPSEEK_V41_FLASH["id"])

    async def unreachable():
        return None

    monkeypatch.setattr(provider, "_load_model_catalog", unreachable)
    assert await provider._get_reasoning_override() == {}
    # Not cached: a later call may still resolve once the catalog is back.
    assert DEEPSEEK_V41_FLASH["id"] not in OpenRouterProvider._reasoning_overrides


@pytest.mark.asyncio
async def test_unknown_model_sends_nothing(monkeypatch):
    monkeypatch.setattr(OpenRouterProvider, "_reasoning_overrides", {})
    provider = _provider("example/not-in-catalog")
    monkeypatch.setattr(provider, "_load_model_catalog", _catalog_of(DEEPSEEK_V41_FLASH))
    assert await provider._get_reasoning_override() == {}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _RecordingClient:
    def __init__(self):
        self.payloads = []

    async def post(self, url, headers=None, json=None, timeout=None):
        self.payloads.append(json)
        return _FakeResponse({
            "choices": [{"message": {"content": "<TRANSLATED>Hola</TRANSLATED>"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "cost": 0.0,
        })


@pytest.mark.asyncio
async def test_generate_sends_the_unified_parameter_only(monkeypatch):
    monkeypatch.setattr(OpenRouterProvider, "_reasoning_overrides", {})
    provider = _provider(DEEPSEEK_V41_FLASH["id"])
    client = _RecordingClient()

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(provider, "_load_model_catalog", _catalog_of(DEEPSEEK_V41_FLASH))

    response = await provider.generate("Translate: hello")

    assert response is not None
    payload = client.payloads[0]
    assert payload["reasoning"] == {"enabled": False}
    # The old ad-hoc fields are not OpenRouter parameters and must be gone.
    assert "thinking" not in payload
    assert "enable_thinking" not in payload


# The fork's model-aware thinking controls pass an explicit reasoning effort
# through LLMGenerationOptions. That choice must win over the default switch-off,
# and an explicit "none" must switch reasoning off the way the catalog allows.

async def _generate_with(monkeypatch, provider, catalog_entry, options):
    client = _RecordingClient()

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(provider, "_load_model_catalog", _catalog_of(catalog_entry))
    await provider.generate("Translate: hello", generation_options=options)
    return client.payloads[0]


@pytest.mark.asyncio
async def test_explicit_effort_wins_over_the_default_switch_off(monkeypatch):
    from src.core.llm.base import LLMGenerationOptions

    monkeypatch.setattr(OpenRouterProvider, "_reasoning_overrides", {})
    provider = _provider(DEEPSEEK_V41_FLASH["id"])
    payload = await _generate_with(
        monkeypatch, provider, DEEPSEEK_V41_FLASH,
        LLMGenerationOptions(reasoning_effort="low"),
    )
    assert payload["reasoning"] == {"effort": "low", "exclude": True}


@pytest.mark.asyncio
async def test_explicit_none_switches_off_even_when_env_keeps_reasoning(monkeypatch):
    from src.core.llm.base import LLMGenerationOptions

    monkeypatch.setattr(OpenRouterProvider, "_reasoning_overrides", {})
    provider = _provider(GLM_53_PRIME["id"], disable_thinking=False)
    payload = await _generate_with(
        monkeypatch, provider, GLM_53_PRIME,
        LLMGenerationOptions(reasoning_effort="none"),
    )
    # Mandatory reasoning cannot be disabled, so the lowest effort is asked for.
    assert payload["reasoning"] == {"effort": "low"}
