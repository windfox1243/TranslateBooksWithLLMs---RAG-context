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
async def test_run_chunk_reflection_pass_accepts_two_argument_log_callback():
    """Reflection logging must not crash callbacks that only accept event and message."""
    from src.prompts.prompts import REFLECTION_JSON_TAG_IN, REFLECTION_JSON_TAG_OUT

    repaired_json = f"""{REFLECTION_JSON_TAG_IN}
{{"status": "no_issues", "issues": []}}
{REFLECTION_JSON_TAG_OUT}"""

    mock_llm = MagicMock()
    mock_llm.generate_async = AsyncMock(side_effect=[
        MagicMock(content='{"status": "needs_repair", "issues": ['),
        MagicMock(content=repaired_json),
    ])
    events = []

    def two_arg_logger(event, message):
        events.append((event, message))

    result = await run_chunk_reflection_pass(
        source_chunk="Mana increased.",
        draft_translation="Năng lượng tăng lên.",
        target_language="Vietnamese",
        model_name="test-model",
        llm_client=mock_llm,
        log_callback=two_arg_logger,
    )

    assert result == "Năng lượng tăng lên."
    assert mock_llm.generate_async.await_count == 2
    assert any(event == "reflection_parse_retry" for event, _ in events)


@pytest.mark.asyncio
async def test_invalid_reflection_json_does_not_silently_become_no_issues():
    """Malformed JSON is retried and then logged as parse failure, not no issues."""
    mock_llm = MagicMock()
    mock_llm.generate_async = AsyncMock(side_effect=[
        MagicMock(content='{"status": "needs_repair", "issues": ['),
        MagicMock(content='{"status": "needs_repair", "issues": ['),
    ])
    events = []

    def two_arg_logger(event, message):
        events.append((event, message))

    result = await run_chunk_reflection_pass(
        source_chunk="Mana increased.",
        draft_translation="Mana.",
        target_language="Vietnamese",
        model_name="test-model",
        llm_client=mock_llm,
        log_callback=two_arg_logger,
    )

    assert result == "Mana."
    assert mock_llm.generate_async.await_count == 2
    assert any(event == "reflection_parse_failed" for event, _ in events)
    assert not any(
        event == "reflection_complete" and "No issues" in message
        for event, message in events
    )


@pytest.mark.asyncio
async def test_run_chunk_reflection_pass_rejects_invalid_repair():
    """Senior Editor repairs must pass the adapter validator before replacement."""
    from src.prompts.prompts import REFLECTION_JSON_TAG_IN, REFLECTION_JSON_TAG_OUT

    critique_output = f"""{REFLECTION_JSON_TAG_IN}
{{
  "status": "needs_repair",
  "issues": [
    {{
      "category": "format",
      "severity": "major",
      "source_quote": "[0] Mana increased.",
      "draft_quote": "[0] Mana.",
      "instruction": "Restore the omitted detail without changing markers.",
      "term_replacement": null
    }}
  ]
}}
{REFLECTION_JSON_TAG_OUT}"""

    mock_llm = MagicMock()
    mock_llm.generate_async = AsyncMock(side_effect=[
        MagicMock(content=critique_output),
        MagicMock(content="<TRANSLATION>Mana increased.</TRANSLATION>"),
    ])
    mock_llm.extract_translation = MagicMock(return_value="Mana increased.")

    result = await run_chunk_reflection_pass(
        source_chunk="[0] Mana increased.",
        draft_translation="[0] Mana.",
        target_language="Vietnamese",
        model_name="test-model",
        llm_client=mock_llm,
        repair_validator=lambda repaired: (
            None if repaired.startswith("[0]") else "missing marker"
        ),
    )

    assert result == "[0] Mana."


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


def test_extract_term_replacements_from_critique():
    """Verify term replacement extraction from Senior Editor critique text."""
    from src.core.translator import extract_term_replacements_from_critique

    sample_critique = """
    1. **Global Replace:** Change all instances of "Học viện Đào tạo Mã nương Nhật Bản" to **"Học viện Tracen"** to maintain natural register.
    2. **Consistency:** Replace "Tracen Academy" with "Học viện Tracen".
    3. **Term Fix:** Change "Mã nương" -> "Umamusume".
    4. **Dialogue Line Fix:** Change "huh!? you're double trigger-san!?" -> "hả!? cậu là double trigger-san sao!?".
    5. **Dialogue Interjection:** Replace "eh? u-um... is that a compliment?" with "hả? ừ-ừm... đó là một lời khen sao?".
    """
    extracted = extract_term_replacements_from_critique(sample_critique)
    assert ("Học viện Đào tạo Mã nương Nhật Bản", "Học viện Tracen") in extracted
    assert ("Tracen Academy", "Học viện Tracen") in extracted
    assert ("Umamusume", "Mã nương") in extracted
    # Full dialogue quotes and exclamations with punctuation must be rejected
    assert ("huh!? you're double trigger-san!?", "hả!? cậu là double trigger-san sao!?") not in extracted
    assert ("eh? u-um... is that a compliment?", "hả? ừ-ừm... đó là một lời khen sao?") not in extracted


def test_structured_reflection_parser_does_not_substring_skip_no_issues():
    """Embedded NO_ISSUES text must not hide actionable legacy defects."""
    from src.core.translator import parse_reflection_result

    result = parse_reflection_result(
        "NO_ISSUES is not valid here.\n- Restore the omitted second sentence."
    )

    assert result.needs_repair
    assert result.issues[0]["instruction"] == "Restore the omitted second sentence."


def test_structured_reflection_term_replacement_extraction():
    """Structured Senior Editor issues expose glossary replacements directly."""
    from src.core.translator import extract_term_replacements_from_critique
    from src.prompts.prompts import REFLECTION_JSON_TAG_IN, REFLECTION_JSON_TAG_OUT

    critique = f"""{REFLECTION_JSON_TAG_IN}
{{
  "status": "needs_repair",
  "issues": [
    {{
      "category": "glossary",
      "severity": "major",
      "source_quote": "Tracen Academy",
      "draft_quote": "Học viện Đào tạo Mã nương Nhật Bản",
      "instruction": "Use the concise established academy name.",
      "term_replacement": {{"source": "Tracen Academy", "target": "Học viện Tracen"}}
    }}
  ]
}}
{REFLECTION_JSON_TAG_OUT}"""

    assert extract_term_replacements_from_critique(critique) == [
        ("Tracen Academy", "Học viện Tracen")
    ]


def test_reflection_prompt_uses_json_contract():
    """Senior Editor prompts must request structured JSON instead of free-form bullets."""
    from src.prompts.prompts import (
        REFLECTION_JSON_TAG_IN,
        generate_chunk_reflection_prompt,
    )

    prompt_pair = generate_chunk_reflection_prompt(
        source_chunk="Mana increased.",
        draft_translation="Năng lượng tăng lên.",
        target_language="Vietnamese",
    )

    assert REFLECTION_JSON_TAG_IN in prompt_pair.system
    assert '"status": "needs_repair"' in prompt_pair.system
    assert "Do not include prose, markdown, bullets, or comments outside the JSON object" in prompt_pair.system
    assert "Output NO_ISSUES" not in prompt_pair.system


@pytest.mark.asyncio
async def test_run_chunk_reflection_pass_sends_context_and_structured_feedback_to_repair():
    """Repair receives active lore plus parsed structured issues."""
    from src.prompts.prompts import REFLECTION_JSON_TAG_IN, REFLECTION_JSON_TAG_OUT

    critique_output = f"""{REFLECTION_JSON_TAG_IN}
{{
  "status": "needs_repair",
  "issues": [
    {{
      "category": "omission",
      "severity": "major",
      "source_quote": "Mana increased.",
      "draft_quote": "",
      "instruction": "Restore the source detail about mana increasing.",
      "term_replacement": null
    }}
  ]
}}
{REFLECTION_JSON_TAG_OUT}"""

    mock_llm = MagicMock()
    mock_llm.generate_async = AsyncMock(side_effect=[
        MagicMock(content=critique_output),
        MagicMock(content="<TRANSLATION>Năng lượng tăng lên.</TRANSLATION>"),
    ])
    mock_llm.extract_translation = MagicMock(return_value="Năng lượng tăng lên.")

    repaired = await run_chunk_reflection_pass(
        source_chunk="Mana increased.",
        draft_translation="Mana.",
        target_language="Vietnamese",
        model_name="test-model",
        llm_client=mock_llm,
        novel_context="Alice: Female mage.",
    )

    assert repaired == "Năng lượng tăng lên."
    repair_prompt = mock_llm.generate_async.call_args_list[1][1]["prompt"]
    assert "# ACTIVE NOVEL LORE & ADDRESSING RULES:" in repair_prompt
    assert "Alice: Female mage." in repair_prompt
    assert '"status": "needs_repair"' in repair_prompt
    assert "Restore the source detail about mana increasing." in repair_prompt


@pytest.mark.asyncio
async def test_subtitle_reflection_rejects_marker_corrupting_repair(monkeypatch):
    """Subtitle reflection repairs are accepted only when local markers survive exactly."""
    from src.core import subtitle_translator

    class FakeSubtitleClient:
        def __init__(self):
            self.closed = False

        async def make_request(self, prompt, model, system_prompt=None):
            return MagicMock(
                content="<TRANSLATION>[0] Năng lượng tăng lên.\n[1] Chạy đi.</TRANSLATION>"
            )

        def extract_translation(self, content):
            return content.replace("<TRANSLATION>", "").replace("</TRANSLATION>", "")

        async def close(self):
            self.closed = True

    fake_client = FakeSubtitleClient()
    monkeypatch.setattr(
        subtitle_translator,
        "create_llm_client",
        lambda *args, **kwargs: fake_client,
    )

    async def corrupting_repair(**kwargs):
        return "[0] Năng lượng tăng lên.\nChạy đi."

    monkeypatch.setattr(
        "src.core.translator.run_chunk_reflection_pass",
        corrupting_repair,
    )

    translations = await subtitle_translator.translate_subtitles_in_blocks(
        subtitle_blocks=[
            [
                {"number": "1", "text": "Mana increased."},
                {"number": "2", "text": "Run."},
            ]
        ],
        source_language="English",
        target_language="Vietnamese",
        model_name="test-model",
        api_endpoint="http://localhost",
        prompt_options={"reflection_mode": True},
    )

    assert translations == {
        0: "Năng lượng tăng lên.",
        1: "Chạy đi.",
    }


@pytest.mark.asyncio
async def test_register_editor_terms_updates_session_and_glossary(tmp_path):
    """Verify NovelContextSession.register_editor_terms updates global lore and prompt options."""
    from src.utils.novel_context import NovelContextSession

    context_file = tmp_path / "test_context.txt"
    context_file.write_text("=== GLOBAL LORE ===\n[GLOSSARY]\n- Initial: Value\n", encoding="utf-8")

    prompt_options = {"glossary_terms": {}}
    session = NovelContextSession(
        path=context_file,
        prompt_options=prompt_options,
        global_lore="=== GLOBAL LORE ===\n[GLOSSARY]\n- Initial: Value\n",
        dynamic_state="",
    )

    term_pairs = [("Học viện Đào tạo Mã nương Nhật Bản", "Học viện Tracen")]
    session.register_editor_terms(term_pairs)

    assert "Học viện Tracen" in session.global_lore
    assert prompt_options["glossary_terms"]["Học viện Đào tạo Mã nương Nhật Bản"] == "Học viện Tracen"
    assert "Học viện Tracen" in prompt_options["novel_context"]


@pytest.mark.asyncio
async def test_run_chunk_reflection_pass_registers_editor_terms(tmp_path):
    """Verify run_chunk_reflection_pass extracts and registers editor terms into context_session."""
    from src.core.translator import run_chunk_reflection_pass
    from src.utils.novel_context import NovelContextSession

    context_file = tmp_path / "test_context.txt"
    context_file.write_text("=== GLOBAL LORE ===\n[GLOSSARY]\n", encoding="utf-8")

    prompt_options = {"glossary_terms": {}}
    session = NovelContextSession(
        path=context_file,
        prompt_options=prompt_options,
        global_lore="=== GLOBAL LORE ===\n[GLOSSARY]\n",
        dynamic_state="",
    )

    critique_output = '1. **Global Replace:** Change all instances of "Học viện Đào tạo Mã nương Nhật Bản" to "Học viện Tracen".'

    mock_llm = MagicMock()
    mock_llm.generate_async = AsyncMock(side_effect=[
        MagicMock(content=critique_output),
        MagicMock(content="<TRANSLATION>[0] Sửa lại tại Học viện Tracen.</TRANSLATION>"),
    ])
    mock_llm.extract_translation = MagicMock(return_value="[0] Sửa lại tại Học viện Tracen.")

    repaired = await run_chunk_reflection_pass(
        source_chunk="[0] At Tracen Academy.",
        draft_translation="[0] Tại Học viện Đào tạo Mã nương Nhật Bản.",
        target_language="Vietnamese",
        model_name="test-model",
        llm_client=mock_llm,
        context_session=session,
    )

    assert repaired == "[0] Sửa lại tại Học viện Tracen."
    assert "Học viện Tracen" in session.global_lore
    assert prompt_options["glossary_terms"]["Học viện Đào tạo Mã nương Nhật Bản"] == "Học viện Tracen"


@pytest.mark.asyncio
async def test_sync_translated_output_with_llm_client(tmp_path):
    """Verify sync_translated_output runs update_novel_context_chunk without NameError for detect_dialogue_turns."""
    from src.utils.novel_context import NovelContextSession

    context_file = tmp_path / "test_context.txt"
    context_file.write_text("=== GLOBAL LORE ===\n[GLOSSARY]\n", encoding="utf-8")

    prompt_options = {}
    session = NovelContextSession(
        path=context_file,
        prompt_options=prompt_options,
        global_lore="=== GLOBAL LORE ===\n[GLOSSARY]\n",
        dynamic_state="",
    )

    mock_llm = MagicMock()
    mock_llm.generate_async = AsyncMock(return_value=MagicMock(content="[NEW_CHARACTERS]\n[IDENTITY_LINKS]\n[NEW_GLOSSARY]\n"))

    success = await session.sync_translated_output(
        translated_chunk="Tất nhiên rồi.",
        source_chunk="\"Of course.\"",
        llm_client=mock_llm,
        model_name="test-model",
    )

    assert success is True
