"""EditorRequestSession: schema fallback and the truncation retry."""

from types import SimpleNamespace

import pytest

from src.core import translator as translator_module
from src.core.editor.requests import EditorRequestSession
from src.core.llm.exceptions import StructuredOutputSchemaError


def _response(content, truncated=False):
    return SimpleNamespace(
        content=content,
        was_truncated=truncated,
        prompt_tokens=1,
        completion_tokens=1,
        thinking_tokens=0,
        total_tokens=2,
        finish_reason="length" if truncated else "stop",
    )


@pytest.mark.asyncio
async def test_truncated_tagged_fallback_is_retried_with_the_tagged_prompt(monkeypatch):
    calls = []

    async def fake_generate(**kwargs):
        calls.append(kwargs)
        if kwargs["response_schema"] is not None:
            raise StructuredOutputSchemaError("schema rejected")
        if len(calls) == 2:
            return _response("<cut off", truncated=True)
        return _response("complete answer")

    monkeypatch.setattr(translator_module, "_generate_editor_response", fake_generate)
    monkeypatch.setattr(translator_module, "_EDITOR_SCHEMA_UNSUPPORTED", set())
    session = EditorRequestSession(
        options={"llm_provider": "openai"},
        llm_client=object(),
        model_name="test-model",
        target_language="Vietnamese",
    )

    response = await session.generate(
        prompt="SCHEMA PROMPT",
        system_prompt="SCHEMA SYSTEM",
        temperature=0.0,
        max_output_tokens=1024,
        stage="reflection",
        structured=True,
        fallback_prompt="TAGGED PROMPT",
        fallback_system_prompt="TAGGED SYSTEM",
    )

    assert response.content == "complete answer"
    assert [c["stage"] for c in calls] == [
        "reflection",
        "reflection_tagged_fallback",
        "reflection_max_tokens_retry",
    ]
    retry = calls[2]
    assert retry["prompt"] == "TAGGED PROMPT"
    assert retry["system_prompt"] == "TAGGED SYSTEM"
    assert retry["response_schema"] is None
