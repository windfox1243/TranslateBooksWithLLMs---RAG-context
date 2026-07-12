from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.core.llm import LLMGenerationOptions, LLMResponse
from src.core.llm.exceptions import StructuredOutputSchemaError
from src.core.llm.providers.gemini import GeminiProvider
from src.prompts.prompts import REFLECTION_RESPONSE_SCHEMA


class _GeminiResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "candidates": [{
                "content": {"parts": [{"text": '{"status":"no_issues","issues":[]}'}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 4,
                "thoughtsTokenCount": 5,
                "totalTokenCount": 12,
            },
        }


def _http_error_response(url: str, body: str) -> httpx.Response:
    request = httpx.Request("POST", url)
    return httpx.Response(400, text=body, request=request)


@pytest.mark.asyncio
async def test_gemini_sends_editor_contract_as_json_schema(monkeypatch):
    provider = GeminiProvider(
        api_key="YOUR_API_KEY_HERE",
        model="gemini-3-flash-preview",
    )
    sent_payloads = []

    class FakeClient:
        async def post(self, url, headers=None, json=None, timeout=None):
            sent_payloads.append(json)
            return _GeminiResponse()

    async def fake_get_client():
        return FakeClient()

    monkeypatch.setattr(provider, "_get_client", fake_get_client)
    original_schema = deepcopy(REFLECTION_RESPONSE_SCHEMA)

    result = await provider.generate(
        "Review this translation.",
        generation_options=LLMGenerationOptions(
            response_schema=REFLECTION_RESPONSE_SCHEMA,
        ),
    )

    assert result.content == '{"status":"no_issues","issues":[]}'
    assert result.thinking_tokens == 5
    assert result.total_tokens == 12
    generation_config = sent_payloads[0]["generationConfig"]
    assert generation_config["responseMimeType"] == "application/json"
    assert "responseSchema" not in generation_config
    gemini_schema = generation_config["responseJsonSchema"]
    assert gemini_schema["additionalProperties"] is False
    assert gemini_schema["propertyOrdering"] == ["status", "issues"]
    issue_schema = gemini_schema["properties"]["issues"]["items"]
    assert issue_schema["propertyOrdering"][0] == "issue_id"
    replacement_schema = issue_schema["properties"]["draft_replacement"]
    assert replacement_schema["anyOf"][0] == {"type": "null"}
    assert replacement_schema["anyOf"][1]["additionalProperties"] is False
    assert REFLECTION_RESPONSE_SCHEMA == original_schema


@pytest.mark.asyncio
async def test_gemini_schema_rejection_is_typed_and_not_retried(monkeypatch):
    provider = GeminiProvider(
        api_key="YOUR_API_KEY_HERE",
        model="gemini-3-flash-preview",
    )
    calls = 0

    class FakeClient:
        async def post(self, url, headers=None, json=None, timeout=None):
            nonlocal calls
            calls += 1
            return _http_error_response(
                url,
                '{"error":{"code":400,"message":"Unknown name '
                '\"additionalProperties\" at '
                'generation_config.response_schema.properties"}}',
            )

    async def fake_get_client():
        return FakeClient()

    monkeypatch.setattr(provider, "_get_client", fake_get_client)

    with pytest.raises(StructuredOutputSchemaError):
        await provider.generate(
            "Review this translation.",
            generation_options=LLMGenerationOptions(
                response_schema=REFLECTION_RESPONSE_SCHEMA,
            ),
        )

    assert calls == 1


@pytest.fixture
def clear_editor_schema_capabilities():
    from src.core import translator

    translator._EDITOR_SCHEMA_UNSUPPORTED.clear()
    yield
    translator._EDITOR_SCHEMA_UNSUPPORTED.clear()


@pytest.mark.asyncio
async def test_editor_caches_only_explicit_schema_rejections(
    clear_editor_schema_capabilities,
):
    from src.core.translator import run_chunk_reflection_pass

    seen_schemas = []

    async def generate_async(**kwargs):
        schema = kwargs.get("response_schema")
        seen_schemas.append(schema)
        if schema is not None:
            raise StructuredOutputSchemaError("schema rejected")
        return LLMResponse(content='{"status":"no_issues","issues":[]}')

    client = MagicMock()
    client.generate_async = AsyncMock(side_effect=generate_async)
    options = {
        "editor_provider_resolved": "gemini",
        "editor_model_resolved": "gemini-3-flash-preview",
    }

    for _ in range(2):
        result = await run_chunk_reflection_pass(
            source_chunk="Source text.",
            draft_translation="Draft text.",
            target_language="English",
            model_name="gemini-3-flash-preview",
            llm_client=client,
            prompt_options=options,
        )
        assert result == "Draft text."

    assert seen_schemas[0] is REFLECTION_RESPONSE_SCHEMA
    assert seen_schemas[1:] == [None, None]


@pytest.mark.asyncio
async def test_editor_does_not_cache_empty_schema_requests(
    clear_editor_schema_capabilities,
):
    from src.core.translator import run_chunk_reflection_pass

    seen_schemas = []
    responses = iter([
        None,
        LLMResponse(content='{"status":"no_issues","issues":[]}'),
    ])

    async def generate_async(**kwargs):
        seen_schemas.append(kwargs.get("response_schema"))
        return next(responses)

    client = MagicMock()
    client.generate_async = AsyncMock(side_effect=generate_async)
    options = {
        "editor_provider_resolved": "gemini",
        "editor_model_resolved": "gemini-3-flash-preview",
    }

    for _ in range(2):
        await run_chunk_reflection_pass(
            source_chunk="Source text.",
            draft_translation="Draft text.",
            target_language="English",
            model_name="gemini-3-flash-preview",
            llm_client=client,
            prompt_options=options,
        )

    assert seen_schemas[0] is REFLECTION_RESPONSE_SCHEMA
    # Empty/transport failures do not trigger an unstructured duplicate call
    # and do not poison the endpoint-specific capability cache.
    assert seen_schemas[1] is REFLECTION_RESPONSE_SCHEMA


@pytest.mark.asyncio
async def test_editor_retries_truncation_with_more_output_and_less_thinking(
    clear_editor_schema_capabilities,
):
    from src.core.translator import run_chunk_reflection_pass

    truncated = LLMResponse(
        content='{',
        prompt_tokens=100,
        completion_tokens=60,
        thinking_tokens=3936,
        total_tokens=4096,
        was_truncated=True,
        finish_reason="MAX_TOKENS",
    )
    complete = LLMResponse(
        content='{"status":"no_issues","issues":[]}',
        prompt_tokens=100,
        completion_tokens=20,
        thinking_tokens=40,
        total_tokens=160,
        finish_reason="STOP",
    )
    client = MagicMock()
    client.generate_async = AsyncMock(side_effect=[truncated, complete])

    result = await run_chunk_reflection_pass(
        source_chunk="Hello.",
        draft_translation="Bonjour.",
        target_language="French",
        model_name="gemini-3-flash-preview",
        llm_client=client,
        prompt_options={
            "editor_provider_resolved": "gemini",
            "editor_model_resolved": "gemini-3-flash-preview",
            "editor_thinking_level": "auto",
            "editor_max_output_tokens": "auto",
        },
    )

    assert result == "Bonjour."
    assert client.generate_async.await_count == 2
    first = client.generate_async.call_args_list[0].kwargs
    retry = client.generate_async.call_args_list[1].kwargs
    assert first["max_output_tokens"] == 4096
    assert first["thinking_level"] == "minimal"
    assert retry["max_output_tokens"] == 8192
    assert retry["thinking_level"] == "minimal"


@pytest.mark.asyncio
async def test_native_json_prompt_and_tagged_schema_fallback_do_not_conflict(
    clear_editor_schema_capabilities,
):
    from src.core.llm.exceptions import StructuredOutputSchemaError
    from src.core.translator import run_chunk_reflection_pass

    calls = []

    async def generate_async(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise StructuredOutputSchemaError("schema rejected")
        return LLMResponse(content='{"status":"no_issues","issues":[]}')

    client = MagicMock()
    client.generate_async = AsyncMock(side_effect=generate_async)
    await run_chunk_reflection_pass(
        source_chunk="Source.", draft_translation="Draft.",
        target_language="English", model_name="editor", llm_client=client,
        prompt_options={
            "editor_provider_resolved": "gemini",
            "editor_model_resolved": "editor",
            "editor_api_endpoint": "https://example.invalid/a",
        },
    )
    native_text = calls[0]["system_prompt"] + calls[0]["prompt"]
    fallback_text = calls[1]["system_prompt"] + calls[1]["prompt"]
    assert calls[0]["response_schema"] is REFLECTION_RESPONSE_SCHEMA
    assert "<REFLECTION_JSON>" not in native_text
    assert calls[1]["response_schema"] is None
    assert "<REFLECTION_JSON>" in fallback_text
    await run_chunk_reflection_pass(
        source_chunk="Source.", draft_translation="Draft.",
        target_language="English", model_name="editor", llm_client=client,
        prompt_options={
            "editor_provider_resolved": "gemini",
            "editor_model_resolved": "editor",
            "editor_api_endpoint": "https://example.invalid/b",
        },
    )
    assert calls[2]["response_schema"] is REFLECTION_RESPONSE_SCHEMA
