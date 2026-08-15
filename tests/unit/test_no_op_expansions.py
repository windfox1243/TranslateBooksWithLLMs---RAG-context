"""An edit the draft already reads, written as an expansion of what it quotes.

Two measured chunks were given one: part of a passage quoted, the whole of it
offered back, and the draft already saying exactly that. Applied literally it
would repeat the words standing beside the span, so nothing applied it, and
each chunk spent two focused retries before going to review.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.translator import run_chunk_reflection_pass
from src.utils.translation_quality import replacement_already_in_draft

# The measured chunk, kept in its own words: a book title quoted from its
# second half, answered with the whole title the draft already carries.
DRAFT = (
    'Hả? "Chỉ cần làm điều này vào ngày mai! Phương pháp tập luyện chiến thắng '
    'tối thượng!" Cái tiêu đề nghe như câu view vậy.'
)
SOURCE = (
    'Huh? "Just Do This Tomorrow! The Ultimate Winning Training Method!" That '
    "title reads like clickbait."
)
ISSUE = {
    "issue_id": "ISSUE-001",
    "category": "mistranslation",
    "severity": "major",
    "confidence": 0.9,
    "repair_kind": "local_replace",
    "instruction": "The title is missing its first half.",
    "draft_quote": '"Chỉ cần làm điều này vào ngày mai! Phương pháp tập luyện '
                   'chiến thắng tối thượng!"',
    "draft_replacement": {
        "draft": "Phương pháp tập luyện chiến thắng tối thượng!",
        "replacement": "Chỉ cần làm điều này vào ngày mai! Phương pháp tập "
                       "luyện chiến thắng tối thượng!",
    },
}


def test_an_expansion_the_draft_already_reads_is_a_no_op():
    assert replacement_already_in_draft(DRAFT, ISSUE) is True


def test_an_expansion_that_adds_something_is_not_a_no_op():
    issue = {"draft_replacement": {
        "draft": "Tam quan mùa thu",
        "replacement": "Tam quan mùa thu dành cho ngựa cái ba tuổi",
    }}

    assert replacement_already_in_draft(
        "Cô ấy sẽ càn quét Tam quan mùa thu.", issue,
    ) is False


def test_one_occurrence_reading_the_replacement_does_not_excuse_the_other():
    # "beta" reads the replacement in the first sentence and not in the second,
    # so the edit has somewhere to land and must not be dropped.
    issue = {"draft_replacement": {
        "draft": "beta", "replacement": "alpha beta",
    }}

    assert replacement_already_in_draft("alpha beta. Rồi beta.", issue) is False


@pytest.mark.asyncio
async def test_a_retry_that_answers_with_one_ends_the_chunk_instead_of_looping():
    # The audit screens these out, so reaching the retry takes an issue that
    # locates badly first; the relocated answer is the edit the draft already
    # reads. Before, that failed validation twice and the chunk went to review.
    unlocatable = dict(
        ISSUE,
        draft_quote="một câu không có trong bản dịch",
        draft_replacement={"draft": "không có", "replacement": "vẫn không có"},
    )
    answers = [
        json.dumps({
            "status": "needs_repair",
            "issues": [unlocatable],
            "voice_observations": [],
        }),
        json.dumps({
            "status": "needs_repair",
            "issues": [ISSUE],
            "voice_observations": [],
        }),
    ]
    prompts = []

    async def generate_async(**kwargs):
        from src.core.llm_client import LLMResponse

        prompts.append(kwargs.get("prompt") or "")
        return LLMResponse(content=answers[min(len(prompts) - 1, len(answers) - 1)])

    client = MagicMock(spec=["generate_async"])
    client.generate_async = AsyncMock(side_effect=generate_async)

    result = await run_chunk_reflection_pass(
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

    assert result == DRAFT
    # One audit, one relocation, and no third request arguing with the answer.
    assert len(prompts) == 2


@pytest.mark.asyncio
async def test_the_pass_ignores_it_instead_of_retrying_it():
    prompts = []

    async def generate_async(**kwargs):
        from src.core.llm_client import LLMResponse

        prompts.append(kwargs.get("prompt") or "")
        return LLMResponse(content=json.dumps({
            "status": "needs_repair",
            "issues": [ISSUE],
            "voice_observations": [],
        }))

    client = MagicMock(spec=["generate_async"])
    client.generate_async = AsyncMock(side_effect=generate_async)

    result = await run_chunk_reflection_pass(
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

    assert result == DRAFT
    assert len(prompts) == 1
