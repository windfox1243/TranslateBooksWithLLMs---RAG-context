import pytest
import tempfile
import shutil
from pathlib import Path
from src.utils.novel_context import (
    build_novel_context,
    compress_dynamic_state,
    decode_context_snapshot,
    is_safe_filename,
    list_novel_contexts,
    load_novel_context,
    make_novel_context_filename,
    merge_dynamic_state,
    normalize_novel_context_content,
    normalize_novel_context_filename,
    save_novel_context,
    resolve_novel_context_path,
)
from src.prompts.prompts import (
    generate_translation_prompt,
    generate_refinement_prompt,
    generate_subtitle_block_prompt,
    generate_subtitle_refinement_block_prompt,
)

def test_is_safe_filename():
    assert is_safe_filename("novel.txt") is True
    assert is_safe_filename("novel-name_1.txt") is True
    assert is_safe_filename("novel/name.txt") is False
    assert is_safe_filename("../novel.txt") is False
    assert is_safe_filename("novel.json") is False
    assert is_safe_filename("") is False

def test_novel_context_operations():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Test load when missing (creates template)
        content = load_novel_context("test_novel.txt", tmp_path)
        assert "# CHARACTERS & GENDERS" in content
        assert "test_novel.txt" in [f["filename"] for f in list_novel_contexts(tmp_path)]
        
        # Test save and load
        new_content = "Custom content here."
        save_novel_context("test_novel.txt", tmp_path, new_content)
        loaded = load_novel_context("test_novel.txt", tmp_path)
        assert loaded == new_content
        
        # Test resolve path
        resolved = resolve_novel_context_path("test_novel.txt", tmp_path)
        assert resolved == tmp_path / "test_novel.txt"

        # Test absolute path redirection (old unbuilt path containing Novel_Contexts)
        old_unbuilt_path = r"C:\unbuilt_folder\Novel_Contexts\test_novel.txt"
        resolved_old = resolve_novel_context_path(old_unbuilt_path, tmp_path)
        assert resolved_old == tmp_path / "test_novel.txt"

        # Test absolute path redirection (old app data path containing TranslateBook_Data)
        old_data_path = r"C:\Users\lmao\Downloads\Compressed\TranslateBooksWithLLMs\TranslateBook_Data\Novel_Contexts\test_novel.txt"
        resolved_data = resolve_novel_context_path(old_data_path, tmp_path)
        assert resolved_data == tmp_path / "test_novel.txt"

        # Test normal absolute path (should not redirect in dev/unfrozen mode)
        other_abs_path = r"C:\external_folder\my_test_context.txt"
        resolved_other = resolve_novel_context_path(other_abs_path, tmp_path)
        assert resolved_other == Path(other_abs_path).resolve()

def test_prompt_injection_translation():
    novel_context = "Li Fan is Male.\nSibling: Elder Brother."
    prompt_pair = generate_translation_prompt(
        main_content="Hello brother.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Vietnamese",
        prompt_options={"novel_context": novel_context}
    )
    
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" in prompt_pair.system
    assert novel_context in prompt_pair.system

def test_prompt_injection_refinement():
    novel_context = "Li Fan is Male."
    prompt_pair = generate_refinement_prompt(
        draft_translation="Bonjour",
        target_language="French",
        prompt_options={"novel_context": novel_context}
    )
    
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" in prompt_pair.system
    assert novel_context in prompt_pair.system

def test_prompt_injection_subtitles():
    novel_context = "Address: Anh / Em."
    subtitles = [(0, "Hello"), (1, "Yes")]
    
    # Translation
    prompt_pair_trans = generate_subtitle_block_prompt(
        subtitle_blocks=subtitles,
        previous_translation_block="",
        source_language="English",
        target_language="Vietnamese",
        prompt_options={"novel_context": novel_context}
    )
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" in prompt_pair_trans.system
    assert novel_context in prompt_pair_trans.system
    
    # Refinement
    prompt_pair_refine = generate_subtitle_refinement_block_prompt(
        subtitle_blocks=subtitles,
        previous_refined_block="",
        target_language="Vietnamese",
        prompt_options={"novel_context": novel_context}
    )
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" in prompt_pair_refine.system
    assert novel_context in prompt_pair_refine.system


@pytest.mark.asyncio
async def test_resync_context_snapshots_logic():
    from unittest.mock import MagicMock, AsyncMock, patch
    from src.core.adapters.generic_translator import _resync_context_snapshots_async
    from src.utils.novel_context import compress_dynamic_state

    translation_id = "test_resync_job"
    initial_snapshot = compress_dynamic_state("Dynamic state initial")

    mock_state_mgr = MagicMock()
    mock_checkpoint_mgr = MagicMock()
    mock_db = MagicMock()
    mock_checkpoint_mgr.db = mock_db
    mock_state_mgr.checkpoint_manager = mock_checkpoint_mgr

    # Setup the mock checkpoint data containing job and chunks
    mock_checkpoint_data = {
        'job': {
            'config': {
                'llm_provider': 'ollama',
                'model': 'test-model',
                'prompt_options': {
                    'novel_context_file': 'resync_novel.txt'
                },
                'source_language': 'English',
                'target_language': 'Vietnamese'
            }
        },
        'chunks': [
            {
                'chunk_index': 0,
                'status': 'completed',
                'original_text': 'Source 1',
                'translated_text': 'Translation 1',
                'chunk_data': {'context_snapshot': initial_snapshot}
            },
            {
                'chunk_index': 1,
                'status': 'completed',
                'original_text': 'Source 2',
                'translated_text': 'Translation 2',
                'chunk_data': {}
            }
        ]
    }
    mock_checkpoint_mgr.load_checkpoint.return_value = mock_checkpoint_data

    # We mock LLM client and update_novel_context_chunk
    mock_global_lore = "Global Lore"
    mock_dynamic_state = "Dynamic state updated"
    mock_change_logs = ["[Novel Context] Dynamic relationship state / addressing forms updated."]

    with patch('src.api.translation_state.get_state_manager', return_value=mock_state_mgr), \
         patch('src.core.llm_client.LLMClient') as mock_llm_class, \
         patch('src.utils.novel_context.update_novel_context_chunk', new_callable=AsyncMock) as mock_update:
        
        mock_update.return_value = (mock_global_lore, mock_dynamic_state, mock_change_logs)
        
        # Run resync for chunk_index > 0 (start_chunk_index=0, so index 1 is resynced)
        await _resync_context_snapshots_async(
            translation_id=translation_id,
            start_chunk_index=0,
            initial_compressed_snapshot=initial_snapshot,
            socketio=None
        )

        # Verify load_checkpoint was called
        mock_checkpoint_mgr.load_checkpoint.assert_called_with(translation_id)
        
        # Verify update_novel_context_chunk was called for chunk 1
        mock_update.assert_called_once()
        
        # Verify database save_chunk was called to store the new compressed context snapshot
        mock_db.save_chunk.assert_called_once()
        args, kwargs = mock_db.save_chunk.call_args
        assert kwargs['translation_id'] == translation_id
        assert kwargs['chunk_index'] == 1
        assert 'context_snapshot' in kwargs['chunk_data']
        
        # Verify logging appends logs successfully
        assert mock_state_mgr.append_log.called


@pytest.mark.asyncio
async def test_resync_last_chunk_writes_edited_full_snapshot_without_nesting(
    monkeypatch,
    tmp_path,
):
    from unittest.mock import MagicMock
    from src.core.adapters.generic_translator import _resync_context_snapshots_async
    import src.config

    fallback = build_novel_context("FILE GLOBAL", "FILE DYNAMIC")
    edited = build_novel_context("EDITED GLOBAL", "EDITED DYNAMIC")
    save_novel_context("resync.txt", tmp_path, fallback)

    state_manager = MagicMock()
    checkpoint_manager = MagicMock()
    checkpoint_manager.load_checkpoint.return_value = {
        "job": {
            "config": {
                "prompt_options": {"novel_context_file": "resync.txt"},
            }
        },
        "chunks": [
            {
                "chunk_index": 0,
                "status": "completed",
                "original_text": "Source",
                "translated_text": "Translation",
                "chunk_data": {},
            }
        ],
    }
    state_manager.checkpoint_manager = checkpoint_manager
    resume_callback = MagicMock()

    monkeypatch.setattr(src.config, "NOVEL_CONTEXTS_DIR", tmp_path)
    monkeypatch.setattr(
        "src.api.translation_state.get_state_manager",
        lambda: state_manager,
    )

    result = await _resync_context_snapshots_async(
        translation_id="job",
        start_chunk_index=0,
        initial_compressed_snapshot=compress_dynamic_state(edited),
        auto_resume_callback=resume_callback,
    )

    assert result is True
    assert load_novel_context("resync.txt", tmp_path) == edited
    assert load_novel_context("resync.txt", tmp_path).count("---DYNAMIC_STATE_START---") == 1
    resume_callback.assert_called_once()


@pytest.mark.asyncio
async def test_resync_failure_does_not_auto_resume(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock, MagicMock, patch
    from src.core.adapters.generic_translator import _resync_context_snapshots_async
    import src.config

    initial = build_novel_context("GLOBAL", "DYNAMIC")
    save_novel_context("resync.txt", tmp_path, initial)
    snapshot = compress_dynamic_state(initial)

    state_manager = MagicMock()
    checkpoint_manager = MagicMock()
    checkpoint_manager.load_checkpoint.return_value = {
        "job": {
            "config": {
                "llm_provider": "ollama",
                "model_name": "model",
                "source_language": "English",
                "target_language": "French",
                "prompt_options": {"novel_context_file": "resync.txt"},
            }
        },
        "chunks": [
            {"chunk_index": 0, "status": "completed", "chunk_data": {}},
            {
                "chunk_index": 1,
                "status": "completed",
                "original_text": "Source",
                "translated_text": "Translation",
                "chunk_data": {},
            },
        ],
    }
    state_manager.checkpoint_manager = checkpoint_manager
    resume_callback = MagicMock()

    monkeypatch.setattr(src.config, "NOVEL_CONTEXTS_DIR", tmp_path)
    monkeypatch.setattr(
        "src.api.translation_state.get_state_manager",
        lambda: state_manager,
    )

    with patch("src.core.llm_client.LLMClient"), patch(
        "src.utils.novel_context.update_novel_context_chunk",
        new_callable=AsyncMock,
        side_effect=RuntimeError("context replay failed"),
    ):
        result = await _resync_context_snapshots_async(
            translation_id="job",
            start_chunk_index=0,
            initial_compressed_snapshot=snapshot,
            auto_resume_callback=resume_callback,
        )

    assert result is False
    resume_callback.assert_not_called()


def test_merge_new_lore_updates_and_corrections():
    from src.utils.novel_context import merge_new_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Li Fan: Male, elder brother.\n"
        "- Wang Lin: Female, student.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- [Apple]: [Táo]\n"
        "- Banana: Chuối\n"
    )

    new_chars = (
        "- Li Fan: Male, elder brother, possessed by a spirit.\n"
        "- Sect Master: Female, leader of Sect.\n"
    )
    new_glossary = (
        "- Apple: Quả Táo\n"
        "- Orange: Cam\n"
    )

    updated_lore, logs = merge_new_lore(initial_lore, new_chars, new_glossary)

    # Check updates
    assert "elder brother, possessed by a spirit." in updated_lore
    assert "Sect Master" in updated_lore
    assert "Quả Táo" in updated_lore
    assert "Orange: Cam" in updated_lore
    assert "Wang Lin: Female, student." in updated_lore  # Untouched

    # Check logs
    assert any("Corrected/Updated Character 'li fan'" in log for log in logs)
    assert any("Added Character" in log and "Sect Master" in log for log in logs)
    assert any("Corrected/Updated Glossary Entry 'apple'" in log for log in logs)
    assert any("Added Glossary Entry" in log and "Orange" in log for log in logs)


def test_character_gender_does_not_flip_without_explicit_correction():
    from src.utils.novel_context import merge_new_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Alex: Male, recurring officer.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    guessed_lore, _ = merge_new_lore(
        initial_lore,
        "- Alex: Female, recurring officer seen in the next scene.",
        "",
    )
    assert "- Alex: Male," in guessed_lore
    assert "- Alex: Female," not in guessed_lore

    corrected_lore, _ = merge_new_lore(
        guessed_lore,
        "- Alex: CORRECTION: [Female, recurring officer explicitly called she.]",
        "",
    )
    assert "- Alex: Female," in corrected_lore


def test_trailing_gender_correction_is_summarized_into_primary_gender():
    from src.utils.novel_context import merge_new_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kriha: Unspecified, Captain, subordinate of Valentine.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    update = (
        '- Kriha: Unspecified, Captain, subordinate of Valentine. '
        '(Correction: Gender confirmed as female via source text '
        '"a woman with blonde hair and blue eyes").'
    )

    updated_lore, _ = merge_new_lore(initial_lore, update, "")

    assert "- Kriha: Female, Captain, subordinate of Valentine" in updated_lore
    assert "Correction:" not in updated_lore
    assert "Unspecified" not in updated_lore


def test_structured_dynamic_state_preserves_addressing_when_only_relationship_changes():
    current = (
        "## CURRENT ADDRESSING FORMS\n"
        '- Valentine → Eric: source "Captain" | target "anh" | intimate\n\n'
        "## RELATIONSHIP EVOLUTION\n"
        "- Valentine ↔ Eric: Deep mutual trust."
    )
    proposed = (
        "## CURRENT ADDRESSING FORMS\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Valentine ↔ Eric: They plan to marry."
    )

    merged = merge_dynamic_state(current, proposed)

    assert 'source "Captain" | target "anh" | intimate' in merged
    assert "- Valentine ↔ Eric: They plan to marry." in merged
    assert "Deep mutual trust" not in merged


def test_context_normalization_removes_placeholders_merges_aliases_and_plain_text_arrows():
    raw_context = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- [Name]: [Gender, role, and description]\n"
        "- Valentine: [Female, protagonist and vampire officer.]\n"
        "- Valentine (Awakened state): [Female, experiencing a PTSD breakdown.]\n"
        "- [None]\n"
        "- The Emperor: [Female, ruler of the Empire.]\n"
        "- Emperor Serena Augusta: [Female, wise ruler of the Empire.]\n"
        "- Serena Augusta: [Female, charismatic sovereign.]\n"
        "- Eric's sibling: [Male, deceased younger sibling.]\n"
        "- Eric's younger sibling: [Male, murdered by vampires.]\n"
        "- Eric's sister: [Female, deceased sister.]\n\n"
        "- Vampire soldier: [Male, unnamed, killed in one scene.]\n"
        "- Allied commander: [Male, unnamed military commander.]\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- [Source Term]: [Target Term]\n\n"
        "---DYNAMIC_STATE_START---\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "- [Character A] $\\rightarrow$ [Character B]: [Reason]\n"
        "- Valentine $\\rightarrow$ Eric: Deeply in love.\n"
        "- Eric $\\rightarrow$ Valentine: Devoted.\n"
        "- Valentine $\\leftrightarrow$ Eric: Mutual commitment.\n"
        "---DYNAMIC_STATE_END---"
    )

    normalized = normalize_novel_context_content(raw_context)

    assert normalized.count("- Valentine:") == 1
    assert normalized.count("- Serena Augusta:") == 1
    assert "The Emperor:" not in normalized
    assert "Emperor Serena Augusta:" not in normalized
    assert normalized.count("Eric's younger sibling:") == 1
    assert "Eric's sibling:" not in normalized
    assert "Eric's sister:" in normalized
    assert "Vampire soldier:" not in normalized
    assert "Allied commander:" not in normalized
    assert "[Name]" not in normalized
    assert "[None]" not in normalized
    assert "[Source Term]" not in normalized
    assert "\\rightarrow" not in normalized
    assert "\\leftrightarrow" not in normalized
    assert "$" not in normalized
    assert "- Valentine → Eric: Deeply in love." in normalized
    assert "- Valentine ↔ Eric: Mutual commitment." in normalized


def test_save_and_load_normalize_existing_context(tmp_path):
    malformed = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- None\n"
            "- Captain Kyle: [Male, loyal subordinate.]\n"
            "- Kyle: [Male, wounded officer.]\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        ),
        "- Kyle $\\rightarrow$ Valentine: Loyal.",
    )

    save_novel_context("cleaned.txt", tmp_path, malformed)
    loaded = load_novel_context("cleaned.txt", tmp_path)

    assert loaded.count("- Kyle:") == 1
    assert "Captain Kyle:" not in loaded
    assert "- None" not in loaded
    assert "$" not in loaded
    assert "- Kyle → Valentine: Loyal." in loaded


def test_dynamic_state_delta_updates_one_relationship_without_erasing_others():
    current = (
        "- Valentine → Eric: Cautious trust.\n"
        "- Kyle → Valentine: Loyal subordinate."
    )
    proposed = "- Valentine $\\rightarrow$ Eric: Deeply in love."

    merged = merge_dynamic_state(current, proposed)

    assert "- Valentine → Eric: Deeply in love." in merged
    assert "Cautious trust" not in merged
    assert "- Kyle → Valentine: Loyal subordinate." in merged


def test_distinct_named_monarchs_are_not_merged():
    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Emperor Nero: Male, ruler of the Western Empire.\n"
            "- Emperor Claudius: Male, ruler of the Eastern Empire.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        ),
        "",
    )

    assert context.count("- Nero:") == 1
    assert context.count("- Claudius:") == 1


@pytest.mark.asyncio
async def test_update_novel_context_chunk_parsing():
    from unittest.mock import MagicMock, AsyncMock
    from src.utils.novel_context import update_novel_context_chunk

    mock_client = MagicMock()
    mock_client.generate = AsyncMock()

    # Setup mock LLM response content using strict format tags
    response_content = (
        "[NEW_CHARACTERS]\n"
        "- Li Fan: Male, possessed.\n\n"
        "[NEW_GLOSSARY]\n"
        "- Apple: Trái Táo\n\n"
        "[DYNAMIC_STATE]\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "Li Fan -> Sect Master: Respectful\n"
    )
    mock_response = MagicMock()
    mock_response.content = response_content
    mock_client.generate.return_value = mock_response

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Li Fan: Male.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- Apple: Táo\n"
    )
    initial_dynamic = "Li Fan -> Sect Master: Neutral"

    updated_lore, updated_dynamic, logs = await update_novel_context_chunk(
        llm_client=mock_client,
        model_name="test-model",
        current_global_lore=initial_lore,
        current_dynamic_state=initial_dynamic,
        source_chunk="Hello Sect Master",
        translated_chunk="Xin chào Sect Master",
        source_language="English",
        target_language="Vietnamese"
    )

    assert "Li Fan: Male, possessed." in updated_lore
    assert "Apple: Trái Táo" in updated_lore
    assert "## CURRENT ADDRESSING FORMS" in updated_dynamic
    assert "## RELATIONSHIP EVOLUTION" in updated_dynamic
    assert "- Li Fan → Sect Master: Respectful" in updated_dynamic
    assert any("Corrected/Updated Character 'li fan'" in log for log in logs)


@pytest.mark.asyncio
async def test_update_novel_context_chunk_deduplicates_headers():
    from unittest.mock import MagicMock, AsyncMock
    from src.utils.novel_context import update_novel_context_chunk

    mock_client = MagicMock()
    mock_client.generate = AsyncMock()

    # LLM returns repeated headers
    response_content = (
        "[NEW_CHARACTERS]\n\n"
        "[NEW_GLOSSARY]\n\n"
        "[DYNAMIC_STATE]\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "Li Fan -> Sect Master: Intimate\n"
    )
    mock_response = MagicMock()
    mock_response.content = response_content
    mock_client.generate.return_value = mock_response

    updated_lore, updated_dynamic, logs = await update_novel_context_chunk(
        llm_client=mock_client,
        model_name="test-model",
        current_global_lore="Global Lore",
        current_dynamic_state="Li Fan -> Sect Master: Neutral",
        source_chunk="Hello Sect Master",
        translated_chunk="Xin chào Sect Master",
        source_language="English",
        target_language="Vietnamese"
    )

    assert updated_dynamic.count("## CURRENT ADDRESSING FORMS") == 1
    assert updated_dynamic.count("## RELATIONSHIP EVOLUTION") == 1
    assert "- Li Fan → Sect Master: Intimate" in updated_dynamic



def test_novel_context_filename_regex_substitution():
    import re
    from pathlib import Path

    # Mimic the regex pattern and logic in translation_routes.py
    def clean_stem(filename):
        stem = Path(filename).stem
        cleaned = re.sub(r'[^a-zA-Z0-9_\-.]', '_', stem)
        return cleaned or 'translation'

    assert clean_stem("Vampire.epub") == "Vampire"
    assert clean_stem("Vampire-1.2_3!@#$.epub") == "Vampire-1.2_3____"
    assert clean_stem("!!!.epub") == "___"
    assert clean_stem("") == "translation"


def test_canonical_snapshot_decodes_full_and_legacy_formats():
    fallback = build_novel_context("GLOBAL", "FILE DYNAMIC")
    full = build_novel_context("EDITED GLOBAL", "EDITED DYNAMIC")

    decoded_full, global_lore, dynamic_state = decode_context_snapshot(
        compress_dynamic_state(full),
        fallback,
    )
    assert decoded_full == full
    assert global_lore == "EDITED GLOBAL"
    assert "## CURRENT ADDRESSING FORMS" in dynamic_state
    assert "## RELATIONSHIP EVOLUTION\nEDITED DYNAMIC" in dynamic_state

    decoded_legacy, global_lore, dynamic_state = decode_context_snapshot(
        compress_dynamic_state("LEGACY DYNAMIC"),
        fallback,
    )
    assert global_lore == "GLOBAL"
    assert dynamic_state == "LEGACY DYNAMIC"
    assert decoded_legacy == build_novel_context("GLOBAL", "LEGACY DYNAMIC")


def test_make_novel_context_filename_is_safe_for_every_input_name():
    assert make_novel_context_filename("Book Name.epub") == "Book_Name_context.txt"
    assert make_novel_context_filename("日本語.docx") == "____context.txt"
    assert make_novel_context_filename("", "epub") == "epub_context.txt"
    assert is_safe_filename(make_novel_context_filename("日本語.docx"))
    assert normalize_novel_context_filename(
        r"C:\old\Novel_Contexts\novel_context.txt"
    ) == "novel_context.txt"
    with pytest.raises(ValueError):
        normalize_novel_context_filename("../outside.json")


@pytest.mark.asyncio
async def test_source_first_context_analysis_uses_no_translation():
    from unittest.mock import AsyncMock, MagicMock
    from src.utils.novel_context import update_novel_context_chunk

    response = MagicMock()
    response.content = (
        "[NEW_CHARACTERS]\n\n"
        "[NEW_GLOSSARY]\n- Sect Master: Maître de secte\n\n"
        "[DYNAMIC_STATE]\nCurrent addressing remains formal."
    )
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)

    lore, dynamic, _ = await update_novel_context_chunk(
        llm_client=client,
        model_name="model",
        current_global_lore="# GLOBAL LORE\n\n## GLOSSARY & TERMINOLOGY\n",
        current_dynamic_state="",
        source_chunk="The Sect Master entered.",
        translated_chunk=None,
        source_language="English",
        target_language="French",
        chunk_index=1,
        total_chunks=2,
    )

    call = client.generate.call_args.kwargs
    assert "LATEST TRANSLATION" not in call["prompt"]
    assert "Analyze the source for context needed by its translation" in call["prompt"]
    assert "Sect Master: Maître de secte" in lore
    assert "## CURRENT ADDRESSING FORMS" in dynamic
    assert "## RELATIONSHIP EVOLUTION\nCurrent addressing remains formal." in dynamic


@pytest.mark.asyncio
async def test_plain_text_context_is_prepared_before_translation(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    from src.core.common import plain_text_pipeline
    import src.config

    events = []

    async def fake_update(**kwargs):
        events.append(("analyze", kwargs["source_chunk"]))
        return (
            "# GLOBAL LORE\n\n## GLOSSARY & TERMINOLOGY\n- Master: Maître",
            "Formal addressing",
            [],
        )

    async def fake_translate(*, main_content, prompt_options, **kwargs):
        events.append(("translate", main_content))
        assert "Master: Maître" in prompt_options["novel_context"]
        return f"FR::{main_content}"

    monkeypatch.setattr(src.config, "NOVEL_CONTEXTS_DIR", tmp_path)
    monkeypatch.setattr("src.utils.novel_context.update_novel_context_chunk", fake_update)
    monkeypatch.setattr(plain_text_pipeline, "generate_translation_request", fake_translate)
    monkeypatch.setattr(plain_text_pipeline, "clean_translated_text", lambda value: value)

    checkpoint_manager = MagicMock()
    output, _, interrupted = await plain_text_pipeline.translate_paragraphs_plain(
        paragraphs=["Master arrived.", "Master spoke."],
        source_language="English",
        target_language="French",
        model_name="model",
        llm_client=object(),
        max_tokens_per_chunk=3,
        prompt_options={
            "auto_update_context": True,
            "input_filename": "novel.docx",
        },
        parallel_workers=4,
        checkpoint_manager=checkpoint_manager,
        translation_id="plain-job",
        global_chunk_offset=10,
    )

    assert interrupted is False
    assert all(value.startswith("FR::") for value in output)
    assert [kind for kind, _ in events] == [
        "analyze", "translate", "analyze", "translate"
    ]
    saved_calls = checkpoint_manager.db.save_chunk.call_args_list
    assert [call.kwargs["chunk_index"] for call in saved_calls] == [10, 11]
    for call in saved_calls:
        snapshot = call.kwargs["chunk_data"]["context_snapshot"]
        decoded, _, _ = decode_context_snapshot(snapshot)
        assert "---DYNAMIC_STATE_START---" in decoded


@pytest.mark.asyncio
async def test_xhtml_context_is_prepared_before_translation(monkeypatch, tmp_path):
    from src.core.epub import xhtml_translator
    import src.config

    events = []

    async def fake_update(**kwargs):
        events.append(("analyze", kwargs["source_chunk"]))
        return (
            "# GLOBAL LORE\n\n## GLOSSARY & TERMINOLOGY\n- Master: Maître",
            "Formal addressing",
            [],
        )

    async def fake_translate(*, chunk_text, prompt_options, **kwargs):
        events.append(("translate", chunk_text))
        assert "Master: Maître" in prompt_options["novel_context"]
        return f"FR::{chunk_text}"

    monkeypatch.setattr(src.config, "NOVEL_CONTEXTS_DIR", tmp_path)
    monkeypatch.setattr("src.utils.novel_context.update_novel_context_chunk", fake_update)
    monkeypatch.setattr(xhtml_translator, "translate_chunk_with_fallback", fake_translate)

    chunks = [
        {"text": "Master arrived.", "local_tag_map": {}, "global_indices": []},
        {"text": "Master spoke.", "local_tag_map": {}, "global_indices": []},
    ]
    translated, _, interrupted = await xhtml_translator._translate_all_chunks_with_checkpoint(
        chunks=chunks,
        source_language="English",
        target_language="French",
        model_name="model",
        llm_client=object(),
        max_retries=0,
        context_manager=None,
        placeholder_format=("[[", "]]"),
        prompt_options={
            "auto_update_context": True,
            "input_filename": "novel.epub",
        },
        parallel_workers=3,
    )

    assert interrupted is False
    assert translated == ["FR::Master arrived.", "FR::Master spoke."]
    assert [kind for kind, _ in events] == [
        "analyze", "translate", "analyze", "translate"
    ]


@pytest.mark.asyncio
async def test_epub_file_checkpoint_does_not_overwrite_chunk_snapshots(tmp_path):
    from unittest.mock import MagicMock
    from lxml import etree
    from src.core.epub.translator import _save_checkpoint

    manager = MagicMock()
    manager.save_epub_file.return_value = True
    file_path = tmp_path / "chapter.xhtml"
    file_path.write_text("<html/>", encoding="utf-8")

    await _save_checkpoint(
        checkpoint_manager=manager,
        translation_id="job",
        file_idx=0,
        content_href="chapter.xhtml",
        doc_root=etree.fromstring(b"<html/>"),
        file_path=str(file_path),
        temp_dir=str(tmp_path),
        total_chunks=1,
        completed_chunks=1,
    )

    manager.db.save_chunk.assert_not_called()
    manager.db.update_job_progress.assert_called_once()
