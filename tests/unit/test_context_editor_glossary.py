"""
Unit tests for Glossary and Custom Instructions integration across
Dynamic Novel Context updates and Senior Editor Reflection passes.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.utils.novel_context import (
    NovelContextSession,
    update_novel_context_chunk,
)
from src.core.translator import run_chunk_reflection_pass


@pytest.mark.asyncio
async def test_update_novel_context_chunk_renders_custom_instructions_and_glossary():
    """Verify update_novel_context_chunk includes custom instructions and glossary block in LLM prompt."""
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=MagicMock(content="[DYNAMIC_STATE]\n- A ↔ B: Allies")
    )

    custom_inst = "DO NOT translate character name Elric."
    glossary = "Elric -> Élric"

    await update_novel_context_chunk(
        llm_client=mock_llm,
        model_name="test-model",
        current_global_lore="",
        current_dynamic_state="",
        source_chunk="Elric walked into the tavern.",
        translated_chunk="Élric đi vào quán rượu.",
        source_language="English",
        target_language="Vietnamese",
        custom_instructions=custom_inst,
        glossary_block=glossary,
    )

    assert mock_llm.generate.called
    call_kwargs = mock_llm.generate.call_args[1]
    prompt = call_kwargs["prompt"]

    assert "CUSTOM INSTRUCTIONS & STYLE GUIDELINES:" in prompt
    assert custom_inst in prompt
    assert "ACTIVE PROJECT GLOSSARY:" in prompt
    assert glossary in prompt


@pytest.mark.asyncio
async def test_novel_context_session_passes_prompt_options():
    """Verify NovelContextSession passes custom_instructions and glossary_block from prompt_options."""
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=MagicMock(content="[DYNAMIC_STATE]\n- A ↔ B: Allies")
    )

    options = {
        "custom_instructions": "Always use formal tone.",
        "glossary_block": "Guild -> Hiệp hội",
    }
    session = NovelContextSession(
        path=MagicMock(name="context.txt", parent=MagicMock()),
        prompt_options=options,
        global_lore="",
        dynamic_state="",
    )

    with patch("src.utils.novel_context.save_novel_context"):
        await session.analyze_source(
            llm_client=mock_llm,
            model_name="test-model",
            source_chunk="The Guild Master spoke.",
            source_language="English",
            target_language="Vietnamese",
            chunk_index=1,
            total_chunks=1,
        )

    assert mock_llm.generate.called
    prompt = mock_llm.generate.call_args[1]["prompt"]
    assert "Always use formal tone." in prompt
    assert "Guild -> Hiệp hội" in prompt


@pytest.mark.asyncio
async def test_run_chunk_reflection_pass_includes_glossary_and_custom_instructions():
    """Verify Senior Editor reflection pass embeds glossary and custom instructions in evaluation prompt."""
    mock_llm = MagicMock()
    mock_llm.generate_async = AsyncMock(
        return_value=MagicMock(content="NO_ISSUES")
    )

    custom_inst = "Use short sentences."
    glossary = "Mana -> Năng lượng"

    result = await run_chunk_reflection_pass(
        source_chunk="Mana increased.",
        draft_translation="Năng lượng tăng lên.",
        target_language="Vietnamese",
        model_name="test-model",
        llm_client=mock_llm,
        custom_instructions=custom_inst,
        glossary_block=glossary,
    )

    assert result == "Năng lượng tăng lên."
    assert mock_llm.generate_async.called
    user_prompt = mock_llm.generate_async.call_args[1]["prompt"]
    assert "CUSTOM INSTRUCTIONS & STYLE GUIDELINES:" in user_prompt
    assert custom_inst in user_prompt
    assert "GLOSSARY & TERM MAPPING:" in user_prompt
    assert glossary in user_prompt


@pytest.mark.asyncio
async def test_xhtml_translator_reflection_mode():
    """Verify xhtml_translator invokes Senior Editor pass when reflection_mode is enabled."""
    from src.core.epub.xhtml_translator import translate_chunk_with_fallback, TranslationMetrics

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=MagicMock(content="<TRANSLATION>[0] Năng lượng tăng lên.</TRANSLATION>")
    )
    mock_llm.extract_translation = MagicMock(return_value="[0] Năng lượng tăng lên.")

    metrics = TranslationMetrics()
    prompt_options = {
        "reflection_mode": True,
        "custom_instructions": "Custom rule",
        "glossary_block": "Mana -> Năng lượng",
    }

    with patch("src.core.translator.run_chunk_reflection_pass", new_callable=AsyncMock) as mock_reflection:
        mock_reflection.return_value = "[0] Năng lượng tăng cao lên."

        result = await translate_chunk_with_fallback(
            chunk_text="[0] Mana increased.",
            local_tag_map={"0": "<span>Mana increased.</span>"},
            global_indices=[0],
            source_language="English",
            target_language="Vietnamese",
            model_name="test-model",
            llm_client=mock_llm,
            stats=metrics,
            prompt_options=prompt_options,
        )

        assert mock_reflection.called
        call_kwargs = mock_reflection.call_args[1]
        assert call_kwargs["custom_instructions"] == "Custom rule"
        assert call_kwargs["glossary_block"] == "Mana -> Năng lượng"
        assert result.succeeded
