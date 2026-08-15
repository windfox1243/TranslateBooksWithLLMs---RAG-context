"""What the editor is told after a rewrite of its own was rejected.

A rejected rewrite comes back as a reason code written for the record. Two
measured chunks were handed one twice and answered with the same text both
times, so the run ended in review with nothing repaired. These pin that the
next attempt is told what to do instead of what went wrong.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.translator import run_chunk_reflection_pass

SOURCE = (
    "September came to Tracen Academy. The classmates who had gone to summer "
    "camp had changed beyond recognition."
)
DRAFT = (
    "Tháng Chín ập đến Học viện Tracen Academy. Những bạn học đã đi trại hè "
    "thay đổi đến mức không nhận ra."
)
# The rewrite the model answered with: the English name is gone from it.
DROPS_THE_NAME = (
    "Tháng Chín ập đến Học viện Tracen. Những bạn học từng đi trại hè đã thay "
    "đổi đến mức không nhận ra."
)
KEEPS_THE_NAME = (
    "Tháng Chín ập đến Học viện Tracen Academy. Những bạn học từng đi trại hè "
    "đã thay đổi đến mức không nhận ra."
)

REFLECTION = json.dumps({
    "status": "needs_repair",
    "issues": [{
        "issue_id": "ISSUE-001",
        "category": "style",
        "severity": "major",
        "confidence": 0.9,
        "repair_kind": "rewrite",
        "instruction": "The second sentence reads stiffly.",
    }],
    "voice_observations": [],
})


def _client(*repairs):
    prompts = []

    async def generate_async(**kwargs):
        from src.core.llm_client import LLMResponse

        prompts.append(kwargs.get("prompt") or "")
        if len(prompts) == 1:
            return LLMResponse(content=REFLECTION)
        body = repairs[min(len(prompts) - 2, len(repairs) - 1)]
        return LLMResponse(content=f"<TRANSLATION>{body}</TRANSLATION>")

    # Spec'd down to the one method: a bare MagicMock answers
    # `extract_translation` with a mock, and the rewrite under test never
    # reaches the validation.
    client = MagicMock(spec=["generate_async"])
    client.generate_async = AsyncMock(side_effect=generate_async)
    return client, prompts


async def _run(client):
    return await run_chunk_reflection_pass(
        source_chunk=SOURCE,
        draft_translation=DRAFT,
        target_language="Vietnamese",
        model_name="editor-model",
        llm_client=client,
        prompt_options={
            "context_contract_version": 5,
            "source_language": "English",
            "editor_provider_resolved": "gemini",
            "editor_model_resolved": "editor-model",
        },
    )


@pytest.mark.asyncio
async def test_a_dropped_name_is_asked_for_back_in_words_the_rewrite_can_follow():
    client, prompts = _client(DROPS_THE_NAME, KEEPS_THE_NAME)

    result = await _run(client)

    second_repair = prompts[2]
    assert 'keep the name "Tracen Academy" exactly as spelled' in second_repair
    assert "1 time(s) in the draft" in second_repair
    assert result == KEEPS_THE_NAME


@pytest.mark.asyncio
async def test_a_rewrite_that_keeps_the_name_is_never_argued_with():
    client, prompts = _client(KEEPS_THE_NAME)

    result = await _run(client)

    assert result == KEEPS_THE_NAME
    assert len(prompts) == 2
