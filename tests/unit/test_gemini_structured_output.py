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
    assert seen_schemas[1] is None
    assert seen_schemas[2] is REFLECTION_RESPONSE_SCHEMA
