import json
import logging
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from src.utils.novel_context import (
    build_novel_context,
    compress_dynamic_state,
    decode_context_snapshot,
    infer_dynamic_address_identity_links,
    is_safe_filename,
    list_novel_contexts,
    load_novel_context,
    make_novel_context_filename,
    merge_new_lore,
    merge_dynamic_state,
    normalize_novel_context_content,
    render_novel_context_update_view,
    render_novel_context_for_prompt,
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
    assert is_safe_filename(
        "\u6211\u600e\u4e48\u53ef\u80fd\u4f1a\u53d8\u6210gal"
        "\u5973\u4e3b\u554a_context.txt"
    ) is True
    assert is_safe_filename("novel/name.txt") is False
    assert is_safe_filename("../novel.txt") is False
    assert is_safe_filename("novel:name.txt") is False
    assert is_safe_filename("con.txt") is False
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
    
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" not in prompt_pair.system
    assert novel_context not in prompt_pair.system
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" in prompt_pair.user
    assert novel_context in prompt_pair.user
    assert "Treat stated character genders as binding continuity facts" in prompt_pair.user


def test_novel_context_keeps_system_prompt_cacheable():
    first = generate_translation_prompt(
        main_content="Eric spoke.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Vietnamese",
        prompt_options={"novel_context": "Eric: Male."},
    )
    second = generate_translation_prompt(
        main_content="Eric spoke.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Vietnamese",
        prompt_options={"novel_context": "Eric: Female."},
    )

    assert first.system == second.system
    assert "Eric: Male." in first.user
    assert "Eric: Female." in second.user


def test_hard_glossary_has_priority_over_novel_context_hints():
    prompt_pair = generate_translation_prompt(
        main_content="The lieutenant colonel entered.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Vietnamese",
        prompt_options={
            "novel_context": (
                "# GLOBAL LORE\n\n"
                "## GLOSSARY & TERMINOLOGY\n"
                "- lieutenant colonel: colonel"
            )
        },
        glossary_block=(
            "# GLOSSARY - REQUIRED TRANSLATIONS\n\n"
            "MANDATORY: use these EXACT translations whenever the source term appears.\n"
            "  - lieutenant colonel -> trung tá\n"
        ),
    )

    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" in prompt_pair.user
    assert "# GLOSSARY - REQUIRED TRANSLATIONS" in prompt_pair.user
    assert "required glossary wins" in prompt_pair.user
    assert prompt_pair.user.index(
        "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)"
    ) < prompt_pair.user.index("# GLOSSARY - REQUIRED TRANSLATIONS")
    assert "# GLOSSARY - REQUIRED TRANSLATIONS" not in prompt_pair.system


def test_prompt_context_selector_prefers_relevant_dormant_relationships():
    filler_characters = "\n".join(
        f"- Irrelevant {index}: Unspecified, background figure with a very long "
        f"description that should be trimmed before relevant Eric context is lost."
        for index in range(40)
    )
    filler_relationships = "\n".join(
        f"- Irrelevant {index} ↔ Someone: stale background relationship."
        for index in range(30)
    )
    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Valentine: Female, protagonist and imperial major.\n"
            "- Eric: Male, lieutenant colonel and Valentine romantic partner.\n"
            f"{filler_characters}\n\n"
            "## CHARACTER ALIASES\n"
            "- Lieutenant Colonel: Eric\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
            "- Hero Medal: Hero's Medal"
        ),
        (
            "## CURRENT ADDRESSING FORMS\n"
            "- Valentine → Eric: intimate romantic address.\n\n"
            "## RELATIONSHIP EVOLUTION\n"
            "- Valentine ↔ Eric: deeply in love after a long separation.\n"
            f"{filler_relationships}"
        ),
    )

    selected = render_novel_context_for_prompt(
        context,
        reference_text='Eric looked at Valentine. "...Lieutenant Colonel."',
        max_tokens=20,
    )

    assert len(selected) <= 1000
    assert "Valentine: Female" in selected
    assert "Eric: Male" in selected
    assert "Lieutenant Colonel: Eric" in selected
    assert "Valentine → Eric" in selected
    assert "Valentine ↔ Eric" in selected
    assert "Irrelevant 39" not in selected


def test_prompt_context_selector_uses_short_names_without_leaking_one_sided_rows():
    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Frondier De Roach: Male, first-year student at Constel, Aster Evans's peer.\n"
            "- Aster Evans: Male, first-year student at Constel.\n"
            "- Ellen Evans: Female, student, Aster Evans's older sister.\n"
            "- Maid: Female, servant of the Roach family.\n"
            "- Enfer De Roach: Male, Frondier De Roach's father."
        ),
        (
            "## CURRENT ADDRESSING FORMS\n"
            '- Ellen Evans → Frondier De Roach: source form "Frondier" | '
            'target-language form "Frondier" | informal, peer\n'
            '- Maid → Frondier De Roach: source form "Master Frondier" | '
            'target-language form "Cậu chủ Frondier" | formal, respectful\n'
            '- Frondier De Roach → Enfer De Roach: source form "Father" | '
            'target-language form "Cha" | formal, respectful\n\n'
            "## RELATIONSHIP EVOLUTION\n"
            "- Ellen Evans ↔ Frondier De Roach: Ellen receives help from Frondier.\n"
            "- Enfer De Roach ↔ Frondier De Roach: father and son."
        ),
    )

    selected = render_novel_context_for_prompt(
        context,
        reference_text='Ellen looked at Frondier. "Frondier, are you with Aster?"',
    )

    assert "Frondier De Roach: Male" in selected
    assert "Ellen Evans: Female" in selected
    assert "Aster Evans: Male" in selected
    assert "Ellen Evans → Frondier De Roach" in selected
    assert "Ellen Evans ↔ Frondier De Roach" in selected
    assert "Maid → Frondier De Roach" not in selected
    assert "Frondier De Roach → Enfer De Roach" not in selected
    assert "Enfer De Roach ↔ Frondier De Roach" not in selected


def test_prompt_context_selector_filters_small_unrelated_context_by_default():
    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Alice: Female, mage from a distant kingdom.\n"
            "- Bob: Male, knight guarding the capital.\n\n"
            "## CHARACTER ALIASES\n"
            "- Captain: Bob\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
            "- mana: magical energy"
        ),
        (
            "## CURRENT ADDRESSING FORMS\n"
            "- Alice → Mentor: formal address.\n\n"
            "## RELATIONSHIP EVOLUTION\n"
            "- Alice ↔ Mentor: student and teacher."
        ),
    )

    selected = render_novel_context_for_prompt(
        context,
        reference_text="Bob lifted his sword at the city gate.",
    )

    assert "Bob: Male" in selected
    assert "Captain: Bob" in selected
    assert "Alice: Female" in selected
    assert "Alice: Female, mage from a distant kingdom" not in selected
    assert "Alice → Mentor" not in selected
    assert "Alice ↔ Mentor" not in selected
    assert "mana: magical energy" not in selected


def test_prompt_context_selector_keeps_compact_gender_roster_without_matches():
    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Alice: Female, mage from a distant kingdom.\n"
            "- Bob: Male, knight guarding the capital."
        ),
        "",
    )

    selected = render_novel_context_for_prompt(
        context,
        reference_text="A nameless guard closed the door.",
    )

    assert "Alice: Female" in selected
    assert "Bob: Male" in selected
    assert "mage from a distant kingdom" not in selected
    assert "knight guarding the capital" not in selected


def test_prompt_context_selector_can_use_legacy_full_injection():
    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Alice: Female, mage from a distant kingdom.\n"
            "- Bob: Male, knight guarding the capital."
        ),
        "",
    )

    selected = render_novel_context_for_prompt(
        context,
        reference_text="A nameless guard closed the door.",
        selective=False,
    )

    assert "Alice: Female" in selected
    assert "Bob: Male" in selected


def test_translation_prompt_uses_selective_context_injection_by_default():
    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Alice: Female, mage from a distant kingdom.\n"
            "- Bob: Male, knight guarding the capital."
        ),
        "",
    )

    prompt_pair = generate_translation_prompt(
        main_content="Bob watched the gate.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Vietnamese",
        prompt_options={"novel_context": context},
    )

    assert "Bob: Male" in prompt_pair.user
    assert "Alice: Female" in prompt_pair.user
    assert "Alice: Female, mage from a distant kingdom" not in prompt_pair.user


def test_translation_prompt_can_disable_selective_context_injection():
    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Alice: Female, mage from a distant kingdom.\n"
            "- Bob: Male, knight guarding the capital."
        ),
        "",
    )

    prompt_pair = generate_translation_prompt(
        main_content="Bob watched the gate.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Vietnamese",
        prompt_options={
            "novel_context": context,
            "novel_context_selective_injection": False,
        },
    )

    assert "Bob: Male" in prompt_pair.user
    assert "Alice: Female" in prompt_pair.user


def test_context_update_view_uses_selective_lore_without_mutating_source():
    global_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Alice: Female, mage from a distant kingdom.\n"
        "- Bob: Male, knight guarding the capital.\n\n"
        "## CHARACTER ALIASES\n"
        "- Captain: Bob"
    )
    dynamic_state = (
        "## CURRENT ADDRESSING FORMS\n"
        "- Alice → Mentor: formal address.\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Alice ↔ Mentor: student and teacher."
    )

    prompt_lore, prompt_dynamic = render_novel_context_update_view(
        global_lore,
        dynamic_state,
        reference_text="Bob lifted his sword at the city gate.",
    )

    assert "Bob: Male" in prompt_lore
    assert "Captain: Bob" in prompt_lore
    assert "Alice: Female" not in prompt_lore
    assert "Alice → Mentor" not in prompt_dynamic
    assert "Alice: Female" in global_lore


@pytest.mark.asyncio
async def test_update_novel_context_chunk_sends_selective_lore_prompt():
    from src.utils.novel_context import update_novel_context_chunk

    response = MagicMock()
    response.content = json.dumps(
        {
            "new_characters": [],
            "identity_links": [],
            "new_glossary": [],
            "dynamic_state": {
                "current_addressing_forms": [],
                "relationship_evolution": [],
            },
            "dialogue_attribution": {"turns": [], "state_after": {}},
        }
    )
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    global_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Alice: Female, mage from a distant kingdom.\n"
        "- Bob: Male, knight guarding the capital."
    )

    updated_lore, _dynamic, _logs = await update_novel_context_chunk(
        llm_client=client,
        model_name="test-model",
        current_global_lore=global_lore,
        current_dynamic_state="",
        source_chunk="Bob lifted his sword at the city gate.",
        translated_chunk=None,
        source_language="English",
        target_language="Vietnamese",
    )

    sent_prompt = client.generate.call_args.kwargs["prompt"]
    assert "Bob: Male" in sent_prompt
    assert "Alice: Female" not in sent_prompt
    assert "Alice: Female" in updated_lore


def test_prompt_injection_refinement():
    novel_context = "Li Fan is Male."
    prompt_pair = generate_refinement_prompt(
        draft_translation="Bonjour",
        target_language="French",
        prompt_options={"novel_context": novel_context}
    )
    
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" not in prompt_pair.system
    assert novel_context not in prompt_pair.system
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" in prompt_pair.user
    assert novel_context in prompt_pair.user


def test_prompts_guard_relationship_based_addressing_for_all_languages():
    translation_prompt = generate_translation_prompt(
        main_content='Ellen looked at Frondier. "Frondier, are you with Aster?"',
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Japanese",
        has_placeholders=False,
    )
    refinement_prompt = generate_refinement_prompt(
        draft_translation="Ellen looked at Frondier.",
        target_language="French",
        has_placeholders=False,
    )

    for prompt in (translation_prompt, refinement_prompt):
        assert "RELATIONSHIP AND ADDRESSING GUARDRAILS" in prompt.system
        assert "direct address and indirect references" in prompt.system
        assert "kinship, seniority, or respect markers" in prompt.system
        assert "VIETNAMESE STYLE GUARDRAILS" not in prompt.system


def test_vietnamese_prompts_guard_first_person_pronoun_consistency():
    novel_context = (
        "## CHARACTERS & GENDERS\n"
        "- Frondier De Roach: Male, first-year student at Constel, Aster Evans's peer.\n"
        "- Aster Evans: Male, first-year student at Constel.\n"
        "- Ellen Evans: Female, student, Aster Evans's older sister.\n\n"
        "---DYNAMIC_STATE_START---\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "## CURRENT ADDRESSING FORMS\n"
        '- Ellen Evans → Frondier De Roach: source form "Frondier" | '
        'target-language form "Frondier" | informal, peer\n'
        "---DYNAMIC_STATE_END---"
    )
    translation_prompt = generate_translation_prompt(
        main_content='Ellen looked at Frondier. "Frondier, are you with Aster?"',
        context_before="",
        context_after="",
        previous_translation_context=(
            "Bởi vì đã lâu lắm rồi, tôi mới lại cảm nhận được tình cảm."
        ),
        source_language="English",
        target_language="Vietnamese",
        has_placeholders=False,
        prompt_options={"novel_context": novel_context},
    )
    refinement_prompt = generate_refinement_prompt(
        draft_translation=(
            'Ellen nhìn Frondier. "Anh Frondier, anh đi cùng Aster à?" '
            'Cô nghĩ anh ấy thật khó đoán.'
        ),
        previous_refined_context=(
            "Bởi vì đã lâu lắm rồi, tôi mới lại cảm nhận được tình cảm."
        ),
        target_language="Vietnamese",
        has_placeholders=False,
        prompt_options={"novel_context": novel_context},
    )

    for prompt in (translation_prompt, refinement_prompt):
        assert "VIETNAMESE STYLE GUARDRAILS" in prompt.system
        assert '"tôi"' in prompt.system
        assert '"mình"' in prompt.system
        assert "CURRENT ADDRESSING FORMS" in prompt.system
        assert '"anh", "chị", or "em"' in prompt.system
        assert "paired social system" in prompt.system
        assert '"em-cô", "em-thầy"' in prompt.system
        assert '"bố/mẹ-con", "tớ-cậu"' in prompt.system
        assert "include, but are not limited to" in prompt.system
        assert "do not call or refer to that character" in prompt.system
        assert "indirect references in dialogue, thoughts, and narration" in prompt.system
        assert "Sino-Vietnamese literary renderings" in prompt.system
        assert "named powers and terminology, not people" in prompt.system
        assert 'target-language form "Frondier"' in prompt.user


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
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" not in prompt_pair_trans.system
    assert novel_context not in prompt_pair_trans.system
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" in prompt_pair_trans.user
    assert novel_context in prompt_pair_trans.user
    
    # Refinement
    prompt_pair_refine = generate_subtitle_refinement_block_prompt(
        subtitle_blocks=subtitles,
        previous_refined_block="",
        target_language="Vietnamese",
        prompt_options={"novel_context": novel_context}
    )
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" not in prompt_pair_refine.system
    assert novel_context not in prompt_pair_refine.system
    assert "# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)" in prompt_pair_refine.user
    assert novel_context in prompt_pair_refine.user


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
    mock_change_logs = ["[Novel Context] Dynamic state updated."]

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
async def test_resync_context_snapshots_resets_dialogue_state_on_scene_key_fallback():
    from unittest.mock import MagicMock, AsyncMock, patch
    from src.core.adapters.generic_translator import _resync_context_snapshots_async
    from src.utils.novel_context import compress_dynamic_state

    translation_id = "test_resync_scene_reset"
    initial_snapshot = compress_dynamic_state("Dynamic state initial")

    mock_state_mgr = MagicMock()
    mock_checkpoint_mgr = MagicMock()
    mock_db = MagicMock()
    mock_checkpoint_mgr.db = mock_db
    mock_state_mgr.checkpoint_manager = mock_checkpoint_mgr

    # We mock dialogue state to verify reset logic
    # Chunk 0 has dialogue attribution with scene_key="1" and state_after = {"speaker": "A", "addressee": "B"}
    # Chunk 1 has no chapter_index at root of chunk_data, but has dialogue_attribution with scene_key="2" (transition!)
    # Chunk 2 has no chapter_index at root, but has dialogue_attribution with scene_key="2" (no transition!)
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
                'chunk_data': {
                    'context_snapshot': initial_snapshot,
                    'dialogue_attribution': {
                        'scene_key': '1',
                        'state_after': {'speaker': 'A', 'addressee': 'B'}
                    }
                }
            },
            {
                'chunk_index': 1,
                'status': 'completed',
                'original_text': 'Source 2',
                'translated_text': 'Translation 2',
                'chunk_data': {
                    'dialogue_attribution': {
                        'scene_key': '2',
                        'state_after': {'speaker': 'C', 'addressee': 'D'}
                    }
                }
            },
            {
                'chunk_index': 2,
                'status': 'completed',
                'original_text': 'Source 3',
                'translated_text': 'Translation 3',
                'chunk_data': {
                    'dialogue_attribution': {
                        'scene_key': '2',
                        'state_after': {'speaker': 'E', 'addressee': 'F'}
                    }
                }
            }
        ]
    }
    mock_checkpoint_mgr.load_checkpoint.return_value = mock_checkpoint_data

    mock_global_lore = (
        "## CHARACTERS & GENDERS\n"
        "- A (Unknown): A\n"
        "- B (Unknown): B\n"
        "- C (Unknown): C\n"
        "- D (Unknown): D\n"
        "- E (Unknown): E\n"
        "- F (Unknown): F\n\n"
        "## CHARACTER ALIASES\n"
        "- A -> A\n"
        "- B -> B\n"
        "- C -> C\n"
        "- D -> D\n"
        "- E -> E\n"
        "- F -> F"
    )
    mock_dynamic_state = "Dynamic state updated"
    mock_change_logs = []

    async def update_side_effect(
        llm_client, model_name, current_global_lore, current_dynamic_state,
        source_chunk, translated_chunk, source_language, target_language,
        chunk_index, total_chunks, source_context, dialogue_turns,
        current_dialogue_state, dialogue_attribution_sink, **_kwargs
    ):
        if chunk_index == 2:  # idx is 1, so idx + 1 = 2
            dialogue_attribution_sink['state_after'] = {'speaker': 'C', 'addressee': 'D'}
        elif chunk_index == 3:  # idx is 2, so idx + 1 = 3
            dialogue_attribution_sink['state_after'] = {'speaker': 'E', 'addressee': 'F'}
        return mock_global_lore, mock_dynamic_state, mock_change_logs

    with patch('src.api.translation_state.get_state_manager', return_value=mock_state_mgr), \
         patch('src.core.llm_client.LLMClient'), \
         patch('src.utils.novel_context.update_novel_context_chunk', new_callable=AsyncMock) as mock_update:
        
        mock_update.side_effect = update_side_effect
        
        # Run resync starting from chunk 0. This will process chunk 1 and chunk 2.
        await _resync_context_snapshots_async(
            translation_id=translation_id,
            start_chunk_index=0,
            initial_compressed_snapshot=initial_snapshot,
            socketio=None
        )
        
        # Since we processed chunk 1 (scene_key='2') and chunk 2 (scene_key='2'):
        # For chunk 1: the initial state of chunk 0 (scene_key='1') was reset because scene_key transitioned ('1' != '2').
        # So current_dialogue_state passed to update_novel_context_chunk for chunk 1 must be {} (empty).
        # For chunk 2: scene_key didn't transition ('2' == '2'), so current_dialogue_state is carried from chunk 1's state_after, i.e. {'speaker': 'C', 'addressee': 'D'}.
        
        assert mock_update.call_count == 2
        
        # Check call arguments for chunk 1 (first call)
        first_call_kwargs = mock_update.call_args_list[0][1]
        assert first_call_kwargs['current_dialogue_state'] == {}
        
        # Check call arguments for chunk 2 (second call)
        second_call_kwargs = mock_update.call_args_list[1][1]
        assert second_call_kwargs['current_dialogue_state'] == {'speaker': 'C', 'addressee': 'D'}


@pytest.mark.asyncio
async def test_global_only_resync_propagates_lore_without_llm(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock, MagicMock, patch
    from src.core.adapters.generic_translator import _resync_context_snapshots_async
    import src.config

    old_global = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Seria: Male, divine woman.\n"
    )
    edited_global = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Seria: Female, divine woman.\n"
    )
    initial = build_novel_context(edited_global, "DYN0")
    chunk_one = build_novel_context(old_global, "DYN1")
    chunk_two = build_novel_context(old_global, "DYN2")
    save_novel_context("resync.txt", tmp_path, build_novel_context(old_global, "DYN2"))

    checkpoint_data = {
        "job": {
            "config": {
                "llm_provider": "ollama",
                "model": "test-model",
                "prompt_options": {"novel_context_file": "resync.txt"},
                "source_language": "English",
                "target_language": "Vietnamese",
            }
        },
        "chunks": [
            {
                "chunk_index": 0,
                "status": "completed",
                "original_text": "Source 1",
                "translated_text": "Translation 1",
                "chunk_data": {
                    "context_snapshot": compress_dynamic_state(initial),
                },
            },
            {
                "chunk_index": 1,
                "status": "completed",
                "original_text": "Source 2",
                "translated_text": "Translation 2",
                "chunk_data": {
                    "context_snapshot": compress_dynamic_state(chunk_one),
                    "dialogue_attribution": {"state_after": {"speaker": "Seria"}},
                },
            },
            {
                "chunk_index": 2,
                "status": "completed",
                "original_text": "Source 3",
                "translated_text": "Translation 3",
                "chunk_data": {
                    "context_snapshot": compress_dynamic_state(chunk_two),
                },
            },
        ],
    }

    state_manager = MagicMock()
    checkpoint_manager = MagicMock()
    checkpoint_manager.load_checkpoint.return_value = checkpoint_data
    checkpoint_manager.get_job.return_value = checkpoint_data["job"]
    checkpoint_manager.update_job_config.return_value = True
    checkpoint_manager.db = MagicMock()
    state_manager.checkpoint_manager = checkpoint_manager
    state_manager.exists.side_effect = [False] + [True] * 20

    monkeypatch.setattr(src.config, "NOVEL_CONTEXTS_DIR", tmp_path)
    monkeypatch.setattr(
        "src.api.translation_state.get_state_manager",
        lambda: state_manager,
    )

    with patch("src.core.llm_client.LLMClient") as llm_client, patch(
        "src.utils.novel_context.update_novel_context_chunk",
        new_callable=AsyncMock,
    ) as update_context:
        result = await _resync_context_snapshots_async(
            translation_id="job",
            start_chunk_index=0,
            initial_compressed_snapshot=compress_dynamic_state(initial),
            global_only_resync=True,
        )

    assert result is True
    state_manager.restore_job_from_checkpoint.assert_called_once_with("job")
    llm_client.assert_not_called()
    update_context.assert_not_called()
    assert checkpoint_manager.db.save_chunk.call_count == 2

    saved_snapshots = {
        call.kwargs["chunk_index"]: call.kwargs["chunk_data"]["context_snapshot"]
        for call in checkpoint_manager.db.save_chunk.call_args_list
    }
    _, chunk_one_global, chunk_one_dynamic = decode_context_snapshot(
        saved_snapshots[1],
        "",
    )
    _, chunk_two_global, chunk_two_dynamic = decode_context_snapshot(
        saved_snapshots[2],
        "",
    )

    assert "Seria: Female" in chunk_one_global
    assert "Seria: Female" in chunk_two_global
    assert chunk_one_dynamic == decode_context_snapshot(
        compress_dynamic_state(chunk_one),
        "",
    )[2]
    assert chunk_two_dynamic == decode_context_snapshot(
        compress_dynamic_state(chunk_two),
        "",
    )[2]
    assert "Seria: Female" in load_novel_context("resync.txt", tmp_path)
    assert "DYN2" in load_novel_context("resync.txt", tmp_path)
    progress_logs = [
        call.args[1]
        for call in state_manager.append_log.call_args_list
        if "Global context propagation progress" in call.args[1]
    ]
    assert len(progress_logs) == 2
    assert "1/2" in progress_logs[0]
    assert "2/2" in progress_logs[1]


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
    refinement_callback = MagicMock()

    monkeypatch.setattr(src.config, "NOVEL_CONTEXTS_DIR", tmp_path)
    monkeypatch.setattr(
        "src.api.translation_state.get_state_manager",
        lambda: state_manager,
    )

    result = await _resync_context_snapshots_async(
        translation_id="job",
        start_chunk_index=0,
        initial_compressed_snapshot=compress_dynamic_state(edited),
        post_resync_callback=refinement_callback,
        post_resync_message="Starting corrective refinement...",
    )

    assert result is True
    assert load_novel_context("resync.txt", tmp_path) == edited
    assert load_novel_context("resync.txt", tmp_path).count("---DYNAMIC_STATE_START---") == 1
    refinement_callback.assert_called_once()
    assert any(
        "Starting corrective refinement..." in call.args[1]
        for call in state_manager.append_log.call_args_list
    )


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
    assert any("Updated character 'li fan'" in log for log in logs)
    assert any("Added character 'sect master'" in log for log in logs)
    assert any("Updated glossary term 'apple'" in log for log in logs)
    assert any("Added glossary term 'orange'" in log for log in logs)


def test_merge_new_lore_returns_logs_without_printing(capsys):
    from src.utils.novel_context import merge_new_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Alice: Female, mage.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    _, logs = merge_new_lore(
        initial_lore,
        "- Bob: Male, knight.\n",
        "",
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert any("Added character 'bob'" in log for log in logs)


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


def test_source_gate_downgrades_unproven_new_character_gender_guess():
    from src.utils.novel_context import merge_new_lore
    from src import config

    original_bypass = getattr(config, 'BYPASS_CONTEXT_GATING', True)
    config.BYPASS_CONTEXT_GATING = False

    try:
        initial_lore = (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        )
        source = (
            "During a regular health checkup, the doctor addressed Kim Ji-an. "
            "The lover I trusted abandoned me. Seeing a sick person hurts her "
            "heart, she said. She just cheated on me with a healthy guy."
        )
        model_guess = (
            "- Kim Ji-an: Female, protagonist, suffering from various "
            "blood-related diseases and anemia.\n"
            "- Ex-lover: Female, former romantic partner who abandoned Kim Ji-an."
        )

        updated_lore, _ = merge_new_lore(
            initial_lore,
            model_guess,
            "",
            source_text=source,
        )

        assert (
            "- Kim Ji-an: Unspecified, protagonist, suffering from various "
            "blood-related diseases and anemia."
        ) in updated_lore
        assert "- Kim Ji-an: Female" not in updated_lore
        assert "- Ex-lover: Female," in updated_lore
    finally:
        config.BYPASS_CONTEXT_GATING = original_bypass


def test_source_gate_accepts_source_proven_new_character_gender():
    from src.utils.novel_context import merge_new_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    source = "Eric suspected Kriha of hiding her identity."

    updated_lore, _ = merge_new_lore(
        initial_lore,
        "- Kriha: Female, soldier subordinate of Valentine.",
        "",
        source_text=source,
    )

    assert "- Kriha: Female, soldier subordinate of Valentine." in updated_lore


def test_source_gate_allows_non_risky_new_character_gender():
    from src.utils.novel_context import merge_new_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    source = "Captain Valentine called Kyle, Jenny, Berman, and Kriha forward."

    updated_lore, _ = merge_new_lore(
        initial_lore,
        "- Jenny: Female, loyal soldier subordinate of Valentine.",
        "",
        source_text=source,
    )

    assert "- Jenny: Female, loyal soldier subordinate of Valentine." in updated_lore


def test_source_gate_accepts_gender_backed_by_incoming_details():
    from src.utils.novel_context import merge_new_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    source = "When I woke up again, I had become a girl with a very small build."

    updated_lore, _ = merge_new_lore(
        initial_lore,
        "- Valentine: Female, reincarnated as a small-build Vampire girl.",
        "",
        source_text=source,
    )

    assert "- Valentine: Female, reincarnated as a small-build Vampire girl." in updated_lore


def test_source_gate_rejects_unproven_explicit_gender_correction():
    from src.utils.novel_context import merge_new_lore
    from src import config

    original_bypass = getattr(config, 'BYPASS_CONTEXT_GATING', True)
    config.BYPASS_CONTEXT_GATING = False

    try:
        initial_lore = (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Alex: Male, veteran investigator.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        )
        source = "Alex interviewed the witness before leaving the station."

        updated_lore, _ = merge_new_lore(
            initial_lore,
            "- Alex: CORRECTION: [Female, veteran investigator with a new lead.]",
            "",
            source_text=source,
        )

        assert "- Alex: Male, veteran investigator" in updated_lore
        assert "- Alex: Female" not in updated_lore
    finally:
        config.BYPASS_CONTEXT_GATING = original_bypass


def test_bypass_context_gating_trusts_new_gender_guess():
    from src.utils.novel_context import merge_new_lore
    from src import config

    original_bypass = getattr(config, 'BYPASS_CONTEXT_GATING', True)
    config.BYPASS_CONTEXT_GATING = True

    try:
        initial_lore = (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        )
        source = (
            "During a regular health checkup, the doctor addressed Kim Ji-an. "
            "The lover I trusted abandoned me. Seeing a sick person hurts her "
            "heart, she said. She just cheated on me with a healthy guy."
        )
        model_guess = (
            "- Kim Ji-an: Female, protagonist, suffering from various "
            "blood-related diseases and anemia.\n"
            "- Ex-lover: Female, former romantic partner who abandoned Kim Ji-an."
        )

        updated_lore, _ = merge_new_lore(
            initial_lore,
            model_guess,
            "",
            source_text=source,
        )

        # When bypass is True, we trust the LLM's classification directly instead of gating it
        assert "- Kim Ji-an: Female, protagonist" in updated_lore
        assert "- Ex-lover: Female," in updated_lore
    finally:
        config.BYPASS_CONTEXT_GATING = original_bypass


def test_bypass_context_gating_accepts_correction():
    from src.utils.novel_context import merge_new_lore
    from src import config

    original_bypass = getattr(config, 'BYPASS_CONTEXT_GATING', True)
    config.BYPASS_CONTEXT_GATING = True

    try:
        initial_lore = (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Alex: Male, veteran investigator.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        )
        source = "Alex interviewed the witness before leaving the station."

        updated_lore, _ = merge_new_lore(
            initial_lore,
            "- Alex: CORRECTION: [Female, veteran investigator with a new lead.]",
            "",
            source_text=source,
        )

        # When bypass is True, corrections are accepted without source text regex checks
        assert "- Alex: Female, veteran investigator with a new lead." in updated_lore
    finally:
        config.BYPASS_CONTEXT_GATING = original_bypass


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


def test_unspecified_gender_is_promoted_by_later_direct_evidence():
    from src.utils.novel_context import merge_new_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kyle: Unspecified, loyal subordinate of Valentine.\n"
        "- Jenny: Unspecified, Captain and subordinate of Valentine.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    updated_lore, _ = merge_new_lore(
        initial_lore,
        (
            "- Kyle: Male, loyal subordinate of Valentine.\n"
            "- Jenny: Female, Captain and loyal subordinate of Valentine."
        ),
        "",
    )

    assert "- Kyle: Male, loyal subordinate of Valentine" in updated_lore
    assert "- Jenny: Female, Captain, loyal subordinate of Valentine" in updated_lore
    assert "Unspecified" not in updated_lore


def test_character_evidence_notes_are_removed_without_rewriting_existing_gender():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kyle: Male, Captain and subordinate of Valentine "
        '(Gender confirmed by source text "Captain Kyle, please take your '
        'spouse away" and context of military roles).\n\n'
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- Kyle: Male, Captain, subordinate of Valentine." in normalized
    assert "Gender confirmed" not in normalized
    assert "source text" not in normalized


def test_direct_gendered_noun_in_description_promotes_unspecified_gender():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kriha: Unspecified, subordinate of Valentine; described as a "
        "blonde-haired girl.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- Kriha: Female," in normalized
    assert "blonde-haired girl" in normalized
    assert "Unspecified" not in normalized


def test_articleless_gendered_noun_in_description_promotes_unspecified_gender():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Tuhee: Unspecified, leader of the special forces/summoner group, "
        "white-haired woman.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- Tuhee: Female," in normalized
    assert "white-haired woman" in normalized
    assert "Unspecified" not in normalized


def test_embedded_gender_fragments_collapse_to_one_canonical_gender():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Valentine: Male, reincarnation into the game world; "
        "Female, reincarnation of Kim Ji-an.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- Valentine: Female," in normalized
    assert "Male, reincarnation" not in normalized
    assert "; Female," not in normalized
    assert "reincarnation of Kim Ji-an into the game world" in normalized


def test_unique_self_title_merges_role_entry_without_explicit_alias():
    from src.utils.novel_context import character_alias_map, normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, suspicious of Valentine; main protagonist of the game "
        "world; Lieutenant Colonel, superior officer of Valentine and Eric "
        "who suspects her true identity.\n"
        "- Lieutenant Colonel: Unspecified, superior officer of Valentine and "
        "Eric who suspects her true identity.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert normalized.count("- Eric:") == 1
    assert "- Lieutenant Colonel:" not in normalized
    assert "Lieutenant Colonel, superior officer of Valentine who suspects" in normalized
    assert "and Eric who suspects" not in normalized
    assert character_alias_map(normalized)["lieutenant colonel"] == "Eric"


def test_reincarnated_current_form_gender_overrides_previous_body_gender():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Valentine: Male, protagonist, a terminally ill man who reincarnates "
        "as a vampire named Valentine; vampire, the reincarnated form of "
        "Kim Ji-an serving as a Major in the imperial army.\n"
        "- Lieutenant Colonel: Male, superior officer to Valentine, suspicious "
        "of her identity.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- Valentine: Female," in normalized
    assert "terminally ill man who reincarnates" not in normalized
    assert "reincarnated from a terminally ill man as a vampire" in normalized


def test_source_detectors_are_file_type_agnostic_plain_text_backstops():
    from src.utils.novel_context import (
        infer_source_gender_updates,
        infer_source_identity_links,
    )

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Valentine: Male, protagonist.\n"
        "- Eric: Male, protagonist of the game.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    source = (
        "[Valentine, starting reincarnation.] When I woke up again, I had "
        "become a girl with a very small build.\n"
        "The Lieutenant Colonel's office was silent. I stared at Eric, who "
        "had dragged me into the room."
    )

    assert infer_source_gender_updates(source, lore) == (
        "- Valentine: CORRECTION: [Female]"
    )
    assert infer_source_identity_links(source, lore) == (
        "- Lieutenant Colonel: Eric"
    )


def test_source_pronoun_correction_targets_named_object_not_sentence_subject():
    from src.utils.novel_context import infer_source_gender_updates

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Valentine: Male, protagonist.\n"
        "- Eric: Male, imperial officer.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    source = "Eric suspected Valentine of using blood magic and hiding her identity."

    assert infer_source_gender_updates(source, lore) == (
        "- Valentine: CORRECTION: [Female]"
    )


def test_cross_character_pronoun_repair_drops_context_control_text():
    from src.utils.novel_context import merge_new_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Valentine: Male, protagonist, a terminally ill man who reincarnates "
        "as Valentine.\n"
        "- Eric: Male, protagonist of the game \"Glory of Victory\", a "
        "vengeful soldier currently holding the rank of Second Lieutenant.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    update = (
        "- Eric: Male, superior officer who suspects Valentine of using blood "
        "magic and her identity; Eric's current rank and title; "
        "title/nickname for Eric; a soldier known for his reckless combat style."
    )

    updated_lore, _ = merge_new_lore(initial_lore, update, "")

    assert "- Valentine: Female," in updated_lore
    assert "- Eric: Male," in updated_lore
    assert "current rank and title" not in updated_lore
    assert "title/nickname for Eric" not in updated_lore
    assert "reckless combat style" in updated_lore


def test_context_normalization_drops_proof_labels_descriptor_names_and_background_roles():
    from src.utils.novel_context import normalize_novel_context_content

    raw_context = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Protagonist of Glory of Victory: Male, fictional character, a "
        "soldier who rises to the rank of Lieutenant Colonel in the imperial "
        "army to seek revenge against the Vampire Kingdom.\n"
        "- Eric: Male, protagonist of Glory of Victory, vampire killer and "
        "imperial officer; source pronoun evidence.\n"
        "- Serena Augusta: Female, ruler of the Empire, character from Glory "
        "of Victory; Emperor, ruler of the Empire.\n"
        "- Doctor: Male, medical professional, attending physician of Kim Ji-an.\n"
        "- Vampire Kingdom: Unspecified, faction, the enemy nation in the "
        "game Glory of Victory.\n"
        "- Marketing Manager: Unspecified, employee of the game company, "
        "author of the advertisement text.\n"
        "- Wounded Soldier 1: Male, imperial soldier, a soldier suffering from "
        "a severed arm.\n"
        "- Wounded Soldier 2: Male, imperial soldier, a soldier searching for "
        "his missing leg.\n"
        "- Wounded Soldier 3: Unspecified, imperial soldier, a soldier "
        "screaming in pain.\n"
        "- Wounded Soldier 4: Unspecified, imperial soldier, a soldier with a "
        "severe abdominal wound.\n\n"
        "## CHARACTER ALIASES\n"
        "- The protagonist: Eric\n"
        "- The protagonist: Eric\n"
        "- The girl: Kim Ji-an\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "\n"
        "---DYNAMIC_STATE_START---\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "## CURRENT ADDRESSING FORMS\n"
        "- Kim Ji-an → Protagonist of Glory of Victory: narrative reference.\n"
        "- Valentine → Wounded Soldier 1: background casualty.\n"
        "- Wounded Soldier 2 → Valentine: background casualty.\n"
        "- Valentine → Doctor: one-off medical scene.\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Kim Ji-an ↔ Protagonist of Glory of Victory: former fan.\n"
        "- Valentine ↔ Wounded Soldier 3: background casualty.\n"
        "---DYNAMIC_STATE_END---"
    )

    normalized = normalize_novel_context_content(raw_context)

    assert "- Eric: Male," in normalized
    assert "Protagonist of Glory of Victory" not in normalized
    assert "The protagonist" not in normalized
    assert "The girl" not in normalized
    assert "source pronoun evidence" not in normalized
    assert "fictional character" not in normalized
    assert normalized.count("- Serena Augusta:") == 1
    assert "Emperor of the Empire" in normalized
    assert "ruler of the Empire, character from" not in normalized
    assert "- Doctor:" not in normalized
    assert "- Vampire Kingdom:" not in normalized
    assert "- Marketing Manager:" not in normalized
    assert "Wounded Soldier" not in normalized
    assert "- Kim Ji-an ↔ Eric: former fan." in normalized


def test_bare_protagonist_entries_and_aliases_cannot_hijack_characters():
    from src.utils.novel_context import merge_new_lore, normalize_global_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kim Ji-an: Male, terminally ill beta tester using the account "
        "name Valentine.\n"
        "- Valentine: Female, reincarnated vampire girl and current named "
        "form of Kim Ji-an.\n"
        "- Eric: Male, protagonist of Glory of Victory and imperial vampire "
        "hunter.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    normalized = normalize_global_lore(
        initial_lore.replace(
            "## CHARACTERS & GENDERS\n",
            "## CHARACTERS & GENDERS\n"
            "- Protagonist: Male, the dying man who is a beta tester. "
            "(Merged Kim Ji-an and Valentine).\n",
        )
    )

    assert "- Protagonist:" not in normalized
    assert "Merged Kim Ji-an and Valentine" not in normalized

    updated_lore, _ = merge_new_lore(
        normalized,
        "- Eric: Male, protagonist of the game Glory of Victory, currently "
        "Second Lieutenant.",
        "",
        "- Protagonist: Eric\n- The protagonist: Eric\n- Valentine: Protagonist",
    )

    assert "- Eric: Male," in updated_lore
    assert "Second Lieutenant" in updated_lore
    assert "Merged Kim Ji-an and Valentine" not in updated_lore
    assert "Protagonist: Eric" not in updated_lore
    assert "The protagonist" not in updated_lore
    assert "Valentine: Protagonist" not in updated_lore
    assert "- Kim Ji-an: Male, terminally ill beta tester" in updated_lore
    assert "- Valentine: Female, reincarnated vampire girl" in updated_lore


def test_bare_protagonist_dynamic_rows_are_dropped_when_unresolved():
    from src.utils.novel_context import normalize_novel_context_content

    raw_context = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Valentine: Female, reincarnated vampire girl.\n"
        "- Eric: Male, protagonist of Glory of Victory.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "\n"
        "---DYNAMIC_STATE_START---\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "## CURRENT ADDRESSING FORMS\n"
        "- Doctor → Protagonist: \"Kim Ji-an\" | \"Kim Ji-an\" | old scene.\n"
        "- Protagonist → Doctor: (none used) | (none used) | old scene.\n"
        "- Eric → Valentine: \"Captain\" | \"Đại úy\" | hostile.\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Protagonist ↔ Eric: player and game character.\n"
        "- Valentine ↔ Eric: hostile superior and subordinate.\n"
        "---DYNAMIC_STATE_END---"
    )

    normalized = normalize_novel_context_content(raw_context)

    assert "Protagonist →" not in normalized
    assert "→ Protagonist" not in normalized
    assert "Protagonist ↔" not in normalized
    assert "↔ Protagonist" not in normalized
    assert "- Eric → Valentine:" in normalized
    assert "- Valentine ↔ Eric:" in normalized


def test_recurring_or_named_generic_roles_are_preserved():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Doctor: Female, recurring physician and mentor to the protagonist.\n"
        "- Soldier 76: Male, source-named callsign and recurring squad leader.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- Doctor: Female, recurring physician" in normalized
    assert "- Soldier 76: Male, source-named callsign" in normalized


def test_bare_physical_labels_are_not_durable_characters():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- boy: Male, character, the protagonist and summoner of the game.\n"
        "- young woman: Female, unnamed person seen in one scene.\n"
        "- Kim Si-hu: Male, academy student and summoner.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- boy:" not in normalized
    assert "- young woman:" not in normalized
    assert "- Kim Si-hu: Male" in normalized


def test_distinctive_physical_descriptors_can_be_tracked_until_named():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- boy in the black coat: Male, recurring observer wearing a black coat.\n"
        "- woman with the scar: Female, recurring swordswoman with a facial scar.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- boy in the black coat: Male" in normalized
    assert "recurring observer wearing a black coat" in normalized
    assert "- woman with the scar: Female" in normalized


def test_metadata_skills_and_incidental_referees_are_not_characters():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- FIVEONE: Unspecified, author, writer of the current work.\n"
        "- Bloodstained Swordswoman A: Unspecified, skill, a combat skill "
        "that upgrades sword-related aptitudes to A.\n"
        "- Duel Referee: Unspecified, character, an NPC entity that oversees "
        "duels.\n"
        "- Seria Bladi Demonkill: Female, S-rank vampire summon.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- FIVEONE:" not in normalized
    assert "- Bloodstained Swordswoman A:" not in normalized
    assert "- Duel Referee:" not in normalized
    assert "- Seria Bladi Demonkill: Female" in normalized


def test_similar_full_name_typos_merge_when_descriptions_overlap():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Seria Bladi Demonkill: Female, S-rank vampire summon with blonde "
        "hair, red eyes, and arrogant chuunibyou behavior.\n"
        "- Seria Blady Demonkill: Female, S-rank vampire summon with blonde "
        "hair, red eyes, and arrogant chuunibyou behavior during battle.\n"
        "- Seria Vladi Demonkill: Female, S-rank vampire summon with blonde "
        "hair, red eyes, and arrogant chuunibyou behavior after injury.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert normalized.count("Demonkill: Female") == 1
    assert "- Seria Blady Demonkill: Seria Bladi Demonkill" in normalized
    assert "- Seria Vladi Demonkill: Seria Bladi Demonkill" in normalized
    assert "during battle" in normalized
    assert "after injury" in normalized


def test_similar_full_name_typos_do_not_merge_without_overlap():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Mira Bladi Cross: Female, herbalist traveling through the capital.\n"
        "- Mira Blady Cross: Female, knight guarding the northern fortress.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- Mira Bladi Cross: Female, herbalist" in normalized
    assert "- Mira Blady Cross: Female, knight" in normalized
    assert "- Mira Blady Cross: Mira Bladi Cross" not in normalized


def test_real_vampire_source_backstops_repair_current_form_gender_and_title_alias():
    from src.utils.novel_context import (
        infer_source_gender_updates,
        infer_source_identity_links,
    )

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kim Ji-an: Female, protagonist, a patient suffering from "
        "blood-related diseases; protagonist, reincarnated as a vampire girl "
        "with a small build.\n"
        "- Valentine: Male, user of the game, a patient suffering from "
        "terminal illness.\n"
        "- Eric: Male, protagonist of Glory of Victory, a soldier currently "
        "holding the rank of Second Lieutenant.\n"
        "- Lieutenant Colonel: Male, superior officer, a vampire killer who "
        "suspects Valentine's identity.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    reincarnation_source = (
        "[Valentine, starting reincarnation.] When I woke up again, I had "
        "become a girl with a very small build. Moreover, a ragged girl in "
        "tattered clothes."
    )
    office_source = (
        "The Lieutenant Colonel's office was silent. I stared at Eric, who "
        "had dragged me into the room."
    )

    assert infer_source_gender_updates(reincarnation_source, lore) == (
        "- Valentine: CORRECTION: [Female]"
    )
    assert infer_source_identity_links(office_source, lore) == (
        "- Lieutenant Colonel: Eric"
    )


def test_vampire_context_normalization_removes_work_and_repairs_current_form():
    from src.utils.novel_context import normalize_novel_context_content

    raw_context = (
        "# GLOBAL LORE\n"
        "(Characters, genders, and terminology; canonical names only.)\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kim Ji-an: Female, protagonist, a patient suffering from "
        "blood-related diseases; protagonist, reincarnated as a small, "
        "ragged vampire girl in the game world.\n"
        "- Glory of Victory: Unspecified, game, a strategy mobile game about "
        "a protagonist seeking revenge against a Vampire Kingdom.\n"
        "- Valentine: Male, protagonist, the user identity for the game "
        "Glory of Victory.\n"
        "- Operator: Unspecified, game administrator, the entity managing the "
        "beta test for Glory of Victory.\n\n"
        "## CHARACTER ALIASES\n"
        "- glory of victory 2: Glory of Victory\n"
        "- Glovic: Glory of Victory\n"
        "- Captain: Valentine\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- Glovic: Glovic\n\n"
        "---DYNAMIC_STATE_START---\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "## CURRENT ADDRESSING FORMS\n"
        "- Operator → Valentine: formal monitoring.\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Kim Ji-an ↔ Glory of Victory: former obsession with the game.\n"
        "- Kim Ji-an ↔ Valentine: Kim Ji-an is the human host currently "
        "undergoing the reincarnation process into the game character "
        "Valentine.\n"
        "---DYNAMIC_STATE_END---"
    )

    normalized = normalize_novel_context_content(raw_context)

    assert "- Glory of Victory:" not in normalized
    assert "- Valentine: Female," in normalized
    assert "reincarnated form of Kim Ji-an as a small, ragged vampire girl" in normalized
    assert "- Operator: Unspecified, game administrator" in normalized
    assert "Glovic: Glory of Victory" not in normalized
    assert "- Glovic: Glovic" in normalized
    assert "Kim Ji-an ↔ Glory of Victory" not in normalized


def test_source_relationship_pronoun_backstop_promotes_kim_jian_gender():
    from src.utils.novel_context import infer_source_gender_updates

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kim Ji-an: Unspecified, protagonist, suffering from blood-related "
        "diseases and facing terminal illness.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    source = "Kim Ji-an's ex-girlfriend cheated on him and abandoned him."

    assert infer_source_gender_updates(source, lore) == (
        "- Kim Ji-an: CORRECTION: [Male]"
    )


def test_source_relationship_label_alone_does_not_promote_gender():
    from src.utils.novel_context import infer_source_gender_updates

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kim Ji-an: Unspecified, protagonist, suffering from blood-related "
        "diseases and facing terminal illness.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    source = "Kim Ji-an's ex-girlfriend moved away without saying goodbye."

    assert infer_source_gender_updates(source, lore) == ""


def test_named_ex_girlfriend_descriptor_does_not_flip_to_object_gender():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Yuna: Unspecified, Kim Ji-an's ex-girlfriend who cheated on him.\n"
        "- Kim Ji-an: Male, terminally ill beta tester.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert (
        "- Yuna: Female, Kim Ji-an's ex-girlfriend who cheated on him."
        in normalized
    )
    assert "- Yuna: Male," not in normalized


def test_romantic_alias_and_unresolved_lover_relation_are_dropped():
    from src.utils.novel_context import normalize_novel_context_content

    raw_context = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kim Ji-an: Unspecified, protagonist, suffering from blood-related "
        "diseases and facing terminal illness.\n\n"
        "## CHARACTER ALIASES\n"
        "- Lover: Kim Ji-an\n\n"
        "## GLOSSARY & TERMINOLOGY\n\n"
        "---DYNAMIC_STATE_START---\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "## CURRENT ADDRESSING FORMS\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Kim Ji-an ↔ Lover: Broken up; she cheated on him and abandoned him.\n"
        "- Kim Ji-an ↔ Kim Ji-an: Broken up; she cheated on him and abandoned him.\n"
        "---DYNAMIC_STATE_END---"
    )

    normalized = normalize_novel_context_content(raw_context)

    assert "Lover: Kim Ji-an" not in normalized
    assert "Kim Ji-an ↔ Lover" not in normalized
    assert "Kim Ji-an ↔ Kim Ji-an" not in normalized


def test_old_identity_does_not_absorb_reincarnated_girl_body_gender():
    from src.utils.novel_context import normalize_novel_context_content

    raw_context = (
        "# GLOBAL LORE\n"
        "(Characters, genders, and terminology; canonical names only.)\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kim Ji-an: Female, protagonist, a terminally ill man suffering "
        "from blood-related diseases and facing terminal illness; his "
        "ex-girlfriend cheated on him and abandoned him; reincarnated as a "
        "small, ragged Vampire girl in the world of Glory of Victory 2.\n"
        "- Valentine: Male, protagonist, the user identity for the game "
        "Glory of Victory.\n\n"
        "## GLOSSARY & TERMINOLOGY\n\n"
        "---DYNAMIC_STATE_START---\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "## CURRENT ADDRESSING FORMS\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Kim Ji-an ↔ Valentine: Kim Ji-an is the human host currently "
        "undergoing the reincarnation process into the game character "
        "Valentine.\n"
        "---DYNAMIC_STATE_END---"
    )

    normalized = normalize_novel_context_content(raw_context)

    assert "- Kim Ji-an: Male," in normalized
    assert "- Kim Ji-an: Female," not in normalized
    assert "- Valentine: Female," in normalized


@pytest.mark.asyncio
async def test_source_memory_backstop_repairs_gender_across_chunk_boundary():
    from src.utils.novel_context import update_novel_context_chunk

    response = MagicMock()
    response.content = (
        "[NEW_CHARACTERS]\n\n"
        "[IDENTITY_LINKS]\n\n"
        "[NEW_GLOSSARY]\n\n"
        "[DYNAMIC_STATE]\n"
        "## CURRENT ADDRESSING FORMS\n\n"
        "## RELATIONSHIP EVOLUTION\n\n"
        "[DIALOGUE_ATTRIBUTION]\n"
        '{"turns":[],"state_after":{}}'
    )
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Valentine: Male, protagonist, the user identity for the game.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    previous_source = (
        "I thought I would die like that. [Valentine, starting "
        "reincarnation.] With that phrase, I fell asleep."
    )
    current_source = (
        "When I woke up again, I had become a girl with a very small build. "
        "Moreover, a ragged girl in tattered clothes."
    )

    updated_lore, _dynamic, _logs = await update_novel_context_chunk(
        llm_client=client,
        model_name="model",
        current_global_lore=lore,
        current_dynamic_state="",
        source_context=previous_source,
        source_chunk=current_source,
        translated_chunk=None,
        source_language="English",
        target_language="Vietnamese",
    )

    assert "- Valentine: Female," in updated_lore


@pytest.mark.asyncio
async def test_context_session_passes_bounded_previous_source_memory(tmp_path):
    from src.utils.novel_context import NovelContextSession

    response = MagicMock()
    response.content = (
        "[NEW_CHARACTERS]\n\n"
        "[IDENTITY_LINKS]\n\n"
        "[NEW_GLOSSARY]\n\n"
        "[DYNAMIC_STATE]\n"
        "## CURRENT ADDRESSING FORMS\n\n"
        "## RELATIONSHIP EVOLUTION\n\n"
        "[DIALOGUE_ATTRIBUTION]\n"
        '{"turns":[],"state_after":{}}'
    )
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    session = NovelContextSession(
        path=tmp_path / "novel.txt",
        prompt_options={},
        global_lore=(
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Valentine: Male, protagonist.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        ),
        dynamic_state="",
    )

    await session.analyze_source(
        llm_client=client,
        model_name="model",
        source_chunk="[Valentine, starting reincarnation.]",
        source_language="English",
        target_language="Vietnamese",
        chunk_index=1,
        total_chunks=2,
    )
    await session.analyze_source(
        llm_client=client,
        model_name="model",
        source_chunk="When I woke up again, I had become a girl.",
        source_language="English",
        target_language="Vietnamese",
        chunk_index=2,
        total_chunks=2,
    )

    second_prompt = client.generate.call_args_list[1].kwargs["prompt"]
    assert "### RECENT SOURCE MEMORY" in second_prompt
    assert "[Valentine, starting reincarnation.]" in second_prompt
    assert "When I woke up again, I had become a girl." in second_prompt


def test_source_identity_links_direct_addressed_title_to_named_responder():
    from src.utils.novel_context import infer_source_identity_links

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Valentine: Female, protagonist.\n"
        "- Eric: Male, protagonist of the game.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    source = (
        '"...Lieutenant Colonel." "You are awake?" Eric was looking at me, '
        "his face contorted in a deep frown."
    )

    assert infer_source_identity_links(source, lore) == (
        "- Lieutenant Colonel: Eric"
    )


def test_explicit_identity_link_merges_rank_entry_and_rewrites_relationships():
    from src.utils.novel_context import build_novel_context

    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Eric: Male, vampire hunter and imperial officer.\n"
            "- Lieutenant Colonel: Unspecified, superior officer of Valentine.\n\n"
            "## CHARACTER ALIASES\n"
            "- Lieutenant Colonel: Eric\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        ),
        (
            "## CURRENT ADDRESSING FORMS\n"
            '- Valentine → Lieutenant Colonel: source form "Lieutenant Colonel" '
            "| formal\n\n"
            "## RELATIONSHIP EVOLUTION\n"
            "- Lieutenant Colonel ↔ Valentine: Mutual suspicion."
        ),
    )

    assert context.count("- Eric:") == 1
    assert "- Lieutenant Colonel: Unspecified" not in context
    assert "- Lieutenant Colonel: Eric" in context
    assert "- Valentine → Eric:" in context
    assert "- Eric ↔ Valentine:" in context


def test_identity_link_update_merges_existing_rank_entry_into_named_character():
    from src.utils.novel_context import merge_new_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, vampire hunter and imperial officer.\n"
        "- Lieutenant Colonel: Unspecified, superior officer of Valentine.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    updated, _ = merge_new_lore(
        initial_lore,
        "",
        "",
        "- Lieutenant Colonel: Eric",
    )

    assert updated.count("- Eric:") == 1
    assert "- Lieutenant Colonel: Unspecified" not in updated
    assert "- Lieutenant Colonel: Eric" in updated
    assert "superior officer of Valentine" in updated


def test_generic_rank_entries_remain_distinct_without_source_proven_link():
    from src.utils.novel_context import normalize_global_lore

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, imperial officer.\n"
        "- Lieutenant Colonel: Unspecified, superior officer in another unit.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(lore)

    assert normalized.count("- Eric:") == 1
    assert normalized.count("- Lieutenant Colonel:") == 1


def test_explicit_identity_link_allows_physical_unstable_aliases():
    from src.utils.novel_context import merge_new_lore

    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, imperial officer.\n"
        "- boy: Unspecified, mysterious child watch from the shadow.\n"
        "- child: Unspecified, youth from the slums.\n"
        "- 소년: Unspecified, boy in Korean.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    updated, _ = merge_new_lore(
        initial_lore,
        "",
        "",
        "- boy: Eric\n- child: Eric\n- 소년: Eric\n- Protagonist: Eric",
    )

    assert updated.count("- Eric:") == 1
    # Physical/non-English aliases should successfully merge
    assert "- boy: Unspecified" not in updated
    assert "- boy: Eric" in updated
    assert "- child: Unspecified" not in updated
    assert "- child: Eric" in updated
    assert "- 소년: Unspecified" not in updated
    assert "- 소년: Eric" in updated
    assert "mysterious child watch from the shadow" in updated
    assert "youth from the slums" in updated
    assert "boy in Korean" in updated
    
    # Meta roles (like Protagonist) should still be rejected/ignored
    assert "- Protagonist:" not in updated
    assert "- Protagonist: Eric" not in updated


def test_save_and_load_preserve_explicit_identity_links(tmp_path):
    from src.utils.novel_context import load_novel_context, save_novel_context

    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Eric: Male, imperial officer.\n"
            "- Lieutenant Colonel: Unspecified, superior officer.\n\n"
            "## CHARACTER ALIASES\n"
            "- Lieutenant Colonel: Eric\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        ),
        "- Lieutenant Colonel → Valentine: Formal command.",
    )

    save_novel_context("identity.txt", tmp_path, context)
    loaded = load_novel_context("identity.txt", tmp_path)

    assert loaded.count("- Eric:") == 1
    assert "- Lieutenant Colonel: Eric" in loaded
    assert "- Lieutenant Colonel →" not in loaded
    assert "- Eric → Valentine: Formal command." in loaded


def test_identity_links_support_non_latin_source_aliases():
    from src.utils.novel_context import character_alias_map

    lore = normalize_novel_context_content(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Eric: Male, imperial officer.\n\n"
            "## CHARACTER ALIASES\n"
            "- 中佐: Eric\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        )
    )

    assert "- 中佐: Eric" in lore
    assert character_alias_map(lore)["中佐"] == "Eric"


def test_context_normalization_merges_named_unique_title_and_repairs_explicit_pronoun_evidence():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Emperor: Female, ruler of the Empire with brilliant white hair.\n"
        "- Serena Augusta: Female, Emperor of the Empire.\n"
        "- Private: Unspecified, a recruit who is mourning his brother.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert normalized.count("- Serena Augusta:") == 1
    assert "- Emperor:" not in normalized
    assert "Emperor of the Empire" in normalized
    assert "ruler of the Empire" not in normalized
    assert "- Private: Male, a recruit who is mourning his brother" in normalized


def test_gender_repair_does_not_use_pronouns_that_refer_to_another_character():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric's sibling: Unspecified, Eric's sibling seen in his dream.\n"
        "- Guard: Unspecified, a bodyguard assigned to her unit.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- Eric's sibling: Unspecified" in normalized
    assert "- Guard: Unspecified" in normalized


def test_character_name_alone_never_promotes_unspecified_gender():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Andrea: Unspecified, recurring officer.\n"
        "- Sasha: Unspecified, recurring medic.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- Andrea: Unspecified" in normalized
    assert "- Sasha: Unspecified" in normalized


def test_character_name_alone_promotes_direct_gender_and_kinship_words():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Female Officer: Unspecified, recurring officer in the front row.\n"
        "- Male Guard: Unspecified, recurring guard in the front row.\n"
        "- Shigure Father: Unspecified, father of Shigure Aya.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- Female Officer: Female, recurring officer in the front row." in normalized
    assert "- Male Guard: Male, recurring guard in the front row." in normalized
    assert "- Shigure Father: Male, father of Shigure Aya." in normalized


def test_disposable_generic_roles_filtering_logic():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Female Student: Unspecified, recurring student acting like a programmed machine.\n"
        "- Teacher: Unspecified, recurring teacher at the school.\n"
        "- Bystander: Unspecified, one-off bystander standing nearby.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    # Female Student should be discarded because "programmed machine" is in incidental markers
    assert "Female Student" not in normalized
    # Bystander should be discarded because it is a generic role and not marked as recurring
    assert "Bystander" not in normalized
    # Teacher should be kept because it has no incidental markers and is marked as recurring
    assert "- Teacher: Unspecified, recurring teacher at the school." in normalized


def test_character_details_compact_repeated_subordinate_roles():
    from src.utils.novel_context import normalize_global_lore

    raw_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Jenny: Female, a loyal subordinate of Valentine; "
        "Captain and subordinate of Valentine.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(raw_lore)

    assert "- Jenny: Female, Captain, loyal subordinate of Valentine" in normalized
    assert normalized.count("subordinate of Valentine") == 1


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


def test_dynamic_state_preserves_dormant_couple_for_arbitrary_chunk_gaps():
    state = (
        "## CURRENT ADDRESSING FORMS\n"
        '- Alice → Bob: source form "Bob" | target-language form "anh" | intimate\n\n'
        "## RELATIONSHIP EVOLUTION\n"
        "- Alice ↔ Bob: Established romantic couple."
    )

    for index in range(1000):
        state = merge_dynamic_state(
            state,
            (
                "## CURRENT ADDRESSING FORMS\n"
                '- Guard → Captain: source form "Captain" | formal\n\n'
                "## RELATIONSHIP EVOLUTION\n"
                f"- Guard → Captain: Temporary scene relationship {index}."
            ),
        )

    assert '- Alice → Bob: source form "Bob"' in state
    assert 'target-language form "anh" | intimate' in state
    assert "- Alice ↔ Bob: Established romantic couple." in state
    assert "- Guard → Captain: Temporary scene relationship 999." in state


def test_dynamic_state_updates_existing_pair_without_erasing_dormant_pairs():
    current = (
        "## CURRENT ADDRESSING FORMS\n"
        '- Alice → Bob: source form "Bob" | target-language form "anh" | intimate\n'
        '- Kriha → Valentine: source form "Major" | formal\n\n'
        "## RELATIONSHIP EVOLUTION\n"
        "- Alice ↔ Bob: Established romantic couple.\n"
        "- Kriha → Valentine: Loyal subordinate."
    )
    proposed = (
        "## CURRENT ADDRESSING FORMS\n"
        '- Kriha → Valentine: source form "Valentine" | intimate\n\n'
        "## RELATIONSHIP EVOLUTION\n"
        "- Kriha → Valentine: Deep mutual trust."
    )

    merged = merge_dynamic_state(current, proposed)

    assert '- Alice → Bob: source form "Bob"' in merged
    assert "- Alice ↔ Bob: Established romantic couple." in merged
    assert 'source form "Valentine" | intimate' in merged
    assert 'source form "Major" | formal' not in merged
    assert "- Kriha → Valentine: Deep mutual trust." in merged
    assert "- Kriha → Valentine: Loyal subordinate." not in merged


def test_dynamic_state_requires_explicit_delete_for_durable_entries():
    current = (
        "## CURRENT ADDRESSING FORMS\n"
        '- Alice → Bob: source form "Bob" | target-language form "anh" | intimate\n'
        '- Kriha → Valentine: source form "Major" | formal\n\n'
        "## RELATIONSHIP EVOLUTION\n"
        "- Alice ↔ Bob: Established romantic couple.\n"
        "- Kriha → Valentine: Loyal subordinate."
    )
    proposed = (
        "## CURRENT ADDRESSING FORMS\n"
        "- Alice → Bob: DELETE\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Alice ↔ Bob: DELETE"
    )

    merged = merge_dynamic_state(current, proposed)

    assert "Alice → Bob" not in merged
    assert "Alice ↔ Bob" not in merged
    assert "- Kriha → Valentine: source form" in merged
    assert "- Kriha → Valentine: Loyal subordinate." in merged


def test_dynamic_state_delete_resolves_character_aliases():
    current = (
        "## CURRENT ADDRESSING FORMS\n"
        "- Serena Augusta → Valentine: source form \"Major\" | formal\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Serena Augusta ↔ Valentine: Trusted allies."
    )
    aliases = {
        "the emperor": "Serena Augusta",
        "serena augusta": "Serena Augusta",
        "valentine": "Valentine",
    }
    proposed = (
        "## CURRENT ADDRESSING FORMS\n"
        "- The Emperor → Valentine: DELETE\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- The Emperor ↔ Valentine: DELETE"
    )

    merged = merge_dynamic_state(current, proposed, aliases)

    assert "Serena Augusta → Valentine" not in merged
    assert "Serena Augusta ↔ Valentine" not in merged


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


def test_new_file_context_session_reuses_lore_without_importing_resume_state(tmp_path):
    from src.utils.novel_context import open_novel_context_session

    saved_context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Serena Augusta: Female, Emperor of the Empire.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        ),
        (
            "## CURRENT ADDRESSING FORMS\n\n"
            "## RELATIONSHIP EVOLUTION\n"
            "- Valentine ↔ Eric: They plan to marry."
        ),
    )
    save_novel_context("continuation_context.txt", tmp_path, saved_context)
    prompt_options = {
        "novel_context_file": "continuation_context.txt",
        "auto_update_context": True,
    }

    session = open_novel_context_session(
        prompt_options=prompt_options,
        novel_contexts_dir=tmp_path,
        input_filename="new_chapters.epub",
    )

    assert session is not None
    assert "Serena Augusta" in session.global_lore
    assert "They plan to marry" in session.dynamic_state
    assert session.dialogue_state == {}
    assert session.dialogue_scene_key is None
    assert "dialogue_attribution" not in prompt_options


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


def test_title_only_monarch_does_not_collapse_two_named_monarchs():
    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Emperor: Male, ruler seen in the opening scene.\n"
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
        "[IDENTITY_LINKS]\n\n"
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
    assert any("Updated character 'li fan'" in log for log in logs)


@pytest.mark.asyncio
async def test_update_novel_context_chunk_parses_legacy_blocks_with_dialogue_json():
    from src.utils.dialogue_attribution import detect_dialogue_turns
    from src.utils.novel_context import update_novel_context_chunk

    source_text = "Sect Master Chen raised his hand. “Stand down,” Li Fan said."
    candidates = detect_dialogue_turns(source_text)
    response = MagicMock()
    response.content = (
        "[NEW_CHARACTERS]\n"
        "- Li Fan: Male, possessed cultivator.\n\n"
        "[IDENTITY_LINKS]\n"
        "- Sect Master: Master Chen\n\n"
        "[NEW_GLOSSARY]\n"
        "- Apple: Trái Táo\n\n"
        "[DYNAMIC_STATE]\n"
        "## CURRENT ADDRESSING FORMS\n"
        "- Li Fan → Sect Master: source form \"Sect Master\" | "
        "target-language form \"Sư phụ\" | formal respect\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Li Fan ↔ Sect Master: student showing formal respect\n\n"
        "[DIALOGUE_ATTRIBUTION]\n"
        + json.dumps(
            {
                "turns": [
                    {
                        "id": candidates[0]["id"],
                        "speaker": "Li Fan",
                        "addressee": "Sect Master",
                        "confidence": 0.92,
                    }
                ],
                "state_after": {
                    "speaker": "Li Fan",
                    "addressee": "Sect Master",
                },
            },
            ensure_ascii=False,
        )
    )
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    sink = {}
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Li Fan: Male.\n"
        "- Master Chen: Male, sect leader.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- Apple: Táo\n"
    )

    updated_lore, updated_dynamic, logs = await update_novel_context_chunk(
        llm_client=client,
        model_name="test-model",
        current_global_lore=initial_lore,
        current_dynamic_state="",
        source_chunk=source_text,
        translated_chunk="Dừng lại.",
        source_language="English",
        target_language="Vietnamese",
        dialogue_turns=candidates,
        dialogue_attribution_sink=sink,
    )

    assert "Li Fan: Male, possessed cultivator." in updated_lore
    assert "- Sect Master: Master Chen" in updated_lore
    assert "Apple: Trái Táo" in updated_lore
    assert "- Li Fan → Master Chen:" in updated_dynamic
    assert "- Li Fan ↔ Master Chen: student showing formal respect" in updated_dynamic
    assert sink["turns"][0]["speaker"] == "Li Fan"
    assert sink["turns"][0]["addressee"] == "Master Chen"
    assert sink["state_after"]["speaker"] == "Li Fan"
    assert any("Updated character 'li fan'" in log for log in logs)


@pytest.mark.asyncio
async def test_json_only_context_update_does_not_modify_lore():
    from src.utils.novel_context import update_novel_context_chunk

    response = MagicMock()
    response.content = json.dumps(
        {
            "newCharacters": {
                "Li Fan": {
                    "gender": "Male",
                    "description": "possessed cultivator",
                }
            },
            "identityLinks": {
                "Sect Master": "Master Chen",
            },
            "newGlossary": {
                "Apple": {
                    "targetTerm": "Trái Táo",
                }
            },
            "dynamicState": {
                "currentAddressingForms": [
                    {
                        "speaker": "Li Fan",
                        "addressee": "Sect Master",
                        "sourceForm": "Sect Master",
                        "targetForm": "Sư phụ",
                        "register": "formal respect",
                    }
                ],
                "relationshipEvolution": [
                    {
                        "characterA": "Li Fan",
                        "characterB": "Sect Master",
                        "relationship": "student showing formal respect",
                    }
                ],
            },
        },
        ensure_ascii=False,
    )
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Li Fan: Male.\n"
        "- Master Chen: Male, sect leader.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- Apple: Táo\n"
    )

    updated_lore, updated_dynamic, _ = await update_novel_context_chunk(
        llm_client=client,
        model_name="test-model",
        current_global_lore=initial_lore,
        current_dynamic_state="",
        source_chunk="Li Fan greeted Sect Master Chen.",
        translated_chunk=None,
        source_language="English",
        target_language="Vietnamese",
    )

    assert "Li Fan: Male, possessed cultivator." not in updated_lore
    assert "- Sect Master: Master Chen" not in updated_lore
    assert "Apple: Trái Táo" not in updated_lore
    assert "Li Fan → Master Chen" not in updated_dynamic
    assert "Li Fan ↔ Master Chen" not in updated_dynamic
    assert "{'gender'" not in updated_lore


@pytest.mark.asyncio
async def test_unproven_model_identity_link_does_not_merge_named_characters(caplog):
    from src.utils.novel_context import update_novel_context_chunk

    response = MagicMock()
    response.content = (
        "[NEW_CHARACTERS]\n\n"
        "[IDENTITY_LINKS]\n"
        "- Alice: Bob\n\n"
        "[NEW_GLOSSARY]\n\n"
        "[DYNAMIC_STATE]\n"
        "## CURRENT ADDRESSING FORMS\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Alice ↔ Clara: Alice protects Clara\n"
    )
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Alice: Female, healer from the east.\n"
        "- Bob: Male, blacksmith from the west.\n"
        "- Clara: Female, Alice's apprentice.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    with caplog.at_level(logging.WARNING, logger="novel_context"):
        updated_lore, updated_dynamic, _ = await update_novel_context_chunk(
            llm_client=client,
            model_name="test-model",
            current_global_lore=initial_lore,
            current_dynamic_state="",
            source_chunk="Alice protects Clara while Bob repairs a sword.",
            translated_chunk=None,
            source_language="English",
            target_language="Vietnamese",
        )

    assert "- Alice: Female, healer from the east." in updated_lore
    assert "- Bob: Male, blacksmith from the west." in updated_lore
    assert "- Alice: Bob" not in updated_lore
    assert "healer from the east; blacksmith from the west" not in updated_lore
    assert "- Alice ↔ Clara: Alice protects Clara" in updated_dynamic
    assert "- Bob ↔ Clara" not in updated_dynamic
    assert "Skipped unsafe identity link 'alice' -> 'Bob'" in caplog.text
    assert "alias is already a named character" in caplog.text


@pytest.mark.asyncio
async def test_model_identity_link_accepts_explicit_parenthetical_alias():
    from src.utils.novel_context import update_novel_context_chunk

    response = MagicMock()
    response.content = (
        "[NEW_CHARACTERS]\n\n"
        "[IDENTITY_LINKS]\n"
        "- Imperial Hawk: Eric\n\n"
        "[NEW_GLOSSARY]\n\n"
        "[DYNAMIC_STATE]\n"
        "## CURRENT ADDRESSING FORMS\n\n"
        "## RELATIONSHIP EVOLUTION\n"
    )
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, imperial officer.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    updated_lore, _, _ = await update_novel_context_chunk(
        llm_client=client,
        model_name="test-model",
        current_global_lore=initial_lore,
        current_dynamic_state="",
        source_chunk="The Imperial Hawk (Eric) entered the room.",
        translated_chunk=None,
        source_language="English",
        target_language="Vietnamese",
    )

    assert "- Imperial Hawk: Eric" in updated_lore
    assert updated_lore.count("- Eric:") == 1


def test_role_only_summoner_update_is_quarantined_not_character():
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kim Hyun-woo: Male, 24-year-old player of Summoner Fantasism.\n"
        "- Seria Bladi Demonkill: Female, S-rank vampire summon.\n"
        "- Valentine: Female, gacha-room NPC.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- Summoner Fantasism: Summoner Fantasism\n"
    )
    new_characters = (
        "- Summoner: Female, 24-year-old unemployed high school graduate "
        "who transmigrated into Seria Bladi Demonkill.\n"
        "- Kim Hyun-woo: Male, soul placed into Seria Bladi Demonkill's body."
    )
    new_glossary = "- Summoner: Triệu hồi sư"

    updated, logs = merge_new_lore(
        initial_lore,
        new_characters,
        new_glossary,
        source_text=(
            "Kim Hyun-woo was placed into Seria Bladi Demonkill's body. "
            "The Summoner is the protagonist of the original game."
        ),
    )

    assert "- Summoner: Female" not in updated
    assert "who transmigrated into Seria Bladi Demonkill" not in updated
    assert "- Kim Hyun-woo: Male," in updated
    assert "soul placed into Seria Bladi Demonkill's body" in updated
    assert "- Seria Bladi Demonkill: Female" in updated
    assert "- Valentine: Female" in updated
    assert "- Summoner: Triệu hồi sư" in updated
    assert any("Quarantined role-like character 'summoner'" in log for log in logs)


def test_context_glossary_character_alias_merges_incoming_alias():
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, lieutenant colonel.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- El Lobo: Eric\n"
    )

    updated, _ = merge_new_lore(
        initial_lore,
        "- El Lobo: Male, masked codename used by the rebels.",
        "",
    )

    assert updated.count("- Eric:") == 1
    assert "- El Lobo: Male" not in updated
    assert "- El Lobo: Eric" in updated
    assert "masked codename used by the rebels" in updated


def test_context_glossary_cjk_full_name_adds_short_source_alias():
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Houjou Takuhei: Male, galgame male lead.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- \u51e4\u591c\u62d3\u5e73: Houjou Takuhei\n"
    )

    updated, _ = merge_new_lore(
        initial_lore,
        "- \u62d3\u5e73: Male, student and childhood friend of Toda Hitona.",
        "",
    )

    assert updated.count("- Houjou Takuhei:") == 1
    assert "- \u62d3\u5e73: Male" not in updated
    assert "- \u62d3\u5e73: Houjou Takuhei" in updated
    assert "childhood friend of Toda Hitona" in updated


def test_context_glossary_korean_full_name_adds_given_name_alias():
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kim Min-su: Male, student.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- \uae40\ubbfc\uc218: Kim Min-su\n"
    )

    updated, _ = merge_new_lore(
        initial_lore,
        "- \ubbfc\uc218: Male, classmate who helps the protagonist.",
        "",
    )

    assert updated.count("- Kim Min-su:") == 1
    assert "- \ubbfc\uc218: Male" not in updated
    assert "- \ubbfc\uc218: Kim Min-su" in updated
    assert "classmate who helps the protagonist" in updated


def test_address_suffix_alias_merges_into_base_character():
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Akane: Female, shy student.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    updated, _ = merge_new_lore(
        initial_lore,
        "- Akane-san: Female, addressed politely by classmates.",
        "",
    )

    assert updated.count("- Akane:") == 1
    assert "- Akane-san: Female" not in updated
    assert "addressed politely by classmates" in updated


def test_korean_romanized_ssi_suffix_merges_into_base_character():
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kim Min-su: Male, student.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    updated, _ = merge_new_lore(
        initial_lore,
        "- Kim Min-su-ssi: Male, addressed politely by classmates.",
        "",
    )

    assert updated.count("- Kim Min-su:") == 1
    assert "- Kim Min-su-ssi: Male" not in updated
    assert "addressed politely by classmates" in updated


def test_dynamic_address_source_form_becomes_trusted_identity_link():
    global_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Houjo Takuhei: Male, student.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    dynamic = (
        "## CURRENT ADDRESSING FORMS\n"
        "- Toda Hitona → Houjo Takuhei: \"\u62d3\u5e73\" | "
        "\"Takuhei\" | casual, classmates\n\n"
        "## RELATIONSHIP EVOLUTION\n"
    )

    assert infer_dynamic_address_identity_links(dynamic, global_lore) == (
        "- \u62d3\u5e73: Houjo Takuhei"
    )


def test_trusted_dynamic_alias_bypasses_source_proof_gate():
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Houjo Takuhei: Male, student.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    updated, _ = merge_new_lore(
        initial_lore,
        "- \u62d3\u5e73: Male, childhood friend of Toda Hitona.",
        "",
        source_text="\u62d3\u5e73 looked at Toda Hitona.",
        trusted_aliases="- \u62d3\u5e73: Houjo Takuhei",
    )

    assert updated.count("- Houjo Takuhei:") == 1
    assert "- \u62d3\u5e73: Male" not in updated
    assert "- \u62d3\u5e73: Houjo Takuhei" in updated
    assert "childhood friend of Toda Hitona" in updated


def test_name_translation_map_renders_auditable_source_name_mappings():
    context = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Houya Takuhei: Male, student.\n"
        "- Shirasagi Akane: Female, main heroine.\n\n"
        "## CHARACTER ALIASES\n"
        "- 凤夜拓平: Houya Takuhei\n"
        "- 拓平: Houya Takuhei\n"
        "- 白鹭茜: Shirasagi Akane\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- 凤夜拓平: Houya Takuhei\n"
        "- 拓平: Takuhei\n"
        "- 白鹭茜: Shirasagi Akane\n"
        "- gal女主: nữ chính game gal\n"
    )

    normalized = normalize_novel_context_content(context)

    assert "## NAME TRANSLATION MAP" in normalized
    assert "- 凤夜拓平: Houya Takuhei" in normalized
    assert "- 拓平: Takuhei" in normalized
    assert "- 白鹭茜: Shirasagi Akane" in normalized
    name_map = normalized.split("## NAME TRANSLATION MAP", 1)[1].split(
        "## GLOSSARY & TERMINOLOGY",
        1,
    )[0]
    assert "gal女主" not in name_map


def test_source_character_glossary_mapping_appears_in_name_translation_map():
    context = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- 户田瞳奈: Female, protagonist.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- 户田瞳奈: Toda Hitona\n"
        "- 运动文胸: Sports bra\n"
    )

    normalized = normalize_novel_context_content(context)
    name_map = normalized.split("## NAME TRANSLATION MAP", 1)[1].split(
        "## GLOSSARY & TERMINOLOGY",
        1,
    )[0]

    assert "- 户田瞳奈: Toda Hitona" in name_map
    assert "运动文胸" not in name_map


def test_name_translation_map_marks_unmapped_source_names_without_generic_roles():
    context = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- 户田瞳奈: Female, protagonist.\n"
        "- 男同学: Male, incidental character who overhears a monologue.\n"
        "- 菜单栏小姐: Unspecified, system entity.\n\n"
        "## CHARACTER ALIASES\n"
        "- 瞳奈: 户田瞳奈\n"
        "- 美少女: 户田瞳奈\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_novel_context_content(context)
    name_map = normalized.split("## NAME TRANSLATION MAP", 1)[1].split(
        "## GLOSSARY & TERMINOLOGY",
        1,
    )[0]

    assert "- 户田瞳奈: (not set)" in name_map
    assert "- 瞳奈: (not set)" in name_map
    assert "- 菜单栏小姐: (not set)" in name_map
    assert "男同学" not in normalized
    assert "美少女" not in name_map


def test_unique_short_name_full_name_entries_merge_with_durable_alias():
    from src.utils.novel_context import character_alias_map, normalize_global_lore

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Vera: Female, summon, sly knight in gray armor who holds a grudge "
        "against Seria Bladi Demon Kill.\n"
        "- Vera Rasvin: Female, knight, former commander of the Holy Knights "
        "who is currently hostile toward the protagonist.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(lore)

    assert "- Vera: Female" not in normalized
    assert normalized.count("- Vera Rasvin:") == 1
    assert "sly knight in gray armor" in normalized
    assert "former commander of the Holy Knights" in normalized
    assert "- Vera: Vera Rasvin" in normalized
    assert character_alias_map(normalized)["vera"] == "Vera Rasvin"


def test_unique_short_name_full_name_merge_rejects_ambiguous_family_name():
    from src.utils.novel_context import normalize_global_lore

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Kim: Male, academy student and summoner.\n"
        "- Kim Si-hu: Male, academy student and protagonist.\n"
        "- Kim Hyun-woo: Male, player and narrator.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(lore)

    assert "- Kim: Male, academy student and summoner." in normalized
    assert "- Kim Si-hu: Male, academy student and protagonist." in normalized
    assert "- Kim Hyun-woo: Male, player and narrator." in normalized
    assert "- Kim: Kim Si-hu" not in normalized
    assert "- Kim: Kim Hyun-woo" not in normalized


def test_singular_plural_entity_entries_merge_when_descriptions_overlap():
    from src.utils.novel_context import normalize_global_lore

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Death God: Unspecified, entity, a creature possessing Kim Si-hu's "
        "body and controlling his actions.\n"
        "- Death Gods: Unspecified, multiple entities, creatures possessing "
        "Kim Si-hu's body and controlling his actions, appearing as a flock "
        "of crows.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(lore)

    assert normalized.count("- Death God:") == 1
    assert "- Death Gods: Unspecified" not in normalized
    assert "flock of crows" in normalized
    assert "- Death Gods: Death God" in normalized


def test_singular_plural_entity_entries_do_not_merge_on_weak_overlap():
    from src.utils.novel_context import normalize_global_lore

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Palace Guard: Male, named royal guard who escorts Valentine.\n"
        "- Palace Guards: Unspecified, multiple background guards stationed "
        "around the palace gate.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(lore)

    assert "- Palace Guard: Male, named royal guard who escorts Valentine." in normalized
    assert "- Palace Guards: Unspecified, multiple background guards stationed around the palace gate." in normalized
    assert "- Palace Guards: Palace Guard" not in normalized


def test_dynamic_state_canonicalizes_compact_and_typo_name_variants():
    from src.utils.novel_context import character_alias_map

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Seria Bladi Demon Kill: Female, S-class vampire summon.\n"
        "- Kim Si-hu: Male, Summoner Academy student.\n"
        "- Vera Rasvin: Female, Holy Knight.\n\n"
        "## CHARACTER ALIASES\n"
        "- Seria Blady Demonkill: Seria Bladi Demon Kill\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    dynamic = (
        "## CURRENT ADDRESSING FORMS\n"
        '- Kim Si-hu → Seria Bladi Demonkill: "Seria" | "Seria" | trust\n'
        '- Seria Blady Demonkill → Vera Rasvin: "Vera" | "Vera" | hostile\n\n'
        "## RELATIONSHIP EVOLUTION\n"
        "- Seria Bladi Demonkill ↔ Seria Bladi Demon Kill: duplicate self row\n"
        "- Yohan → Kim Si-hu & Seria Bladi Demonkill: group targeting\n"
        "- Seria Blady Demonkill ↔ Vera Rasvin: hostile rivals\n"
    )

    updated = merge_dynamic_state("", dynamic, character_alias_map(lore))

    assert '- Kim Si-hu → Seria Bladi Demon Kill: "Seria"' in updated
    assert '- Seria Bladi Demon Kill → Vera Rasvin: "Vera"' in updated
    assert "- Yohan → Kim Si-hu & Seria Bladi Demon Kill: group targeting" in updated
    assert "- Seria Bladi Demon Kill ↔ Vera Rasvin: hostile rivals" in updated
    assert "duplicate self row" not in updated
    assert "Seria Bladi Demonkill →" not in updated
    assert "Seria Blady Demonkill →" not in updated


def test_short_name_full_name_merge_requires_shared_identity_evidence():
    from src.utils.novel_context import normalize_global_lore

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Alice: Female, apothecary traveling through the capital.\n"
        "- Alice Hart: Female, swordswoman guarding the northern gate.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    normalized = normalize_global_lore(lore)

    assert "- Alice: Female, apothecary traveling through the capital." in normalized
    assert "- Alice Hart: Female, swordswoman guarding the northern gate." in normalized
    assert "- Alice: Alice Hart" not in normalized


def test_incoming_full_name_merges_existing_short_name_and_persists_alias():
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Vera: Female, summon, sly knight in gray armor who holds a grudge "
        "against Seria Bladi Demon Kill.\n\n"
        "## CHARACTER ALIASES\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    new_characters = (
        "- Vera Rasvin: Female, knight, former commander of the Holy Knights "
        "who is currently hostile toward the protagonist."
    )

    updated, logs = merge_new_lore(initial_lore, new_characters, "")

    assert "- Vera: Female" not in updated
    assert updated.count("- Vera Rasvin:") == 1
    assert "- Vera: Vera Rasvin" in updated
    assert "sly knight in gray armor" in updated
    assert "former commander of the Holy Knights" in updated
    assert any("Linked identity 'Vera' -> 'Vera Rasvin'" in log for log in logs)


def test_role_only_existing_character_is_excluded_from_prompt_and_dynamic():
    context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Summoner: Female, wrongly stored transmigrated protagonist.\n"
            "- Kim Hyun-woo: Male, soul placed into Seria's body.\n"
            "- Seria Bladi Demonkill: Female, S-rank vampire summon.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
            "- Summoner: Triệu hồi sư\n"
        ),
        (
            "## CURRENT ADDRESSING FORMS\n"
            "- Seria Bladi Demonkill → Summoner: source form \"Summoner-nim\" "
            "| target-language form \"Triệu hồi sư-nim\" | formal\n\n"
            "## RELATIONSHIP EVOLUTION\n"
            "- Seria Bladi Demonkill ↔ Summoner: Summon and Summoner"
        ),
    )

    selected = render_novel_context_for_prompt(
        context,
        reference_text="The Summoner opened the gacha room.",
    )

    assert "wrongly stored transmigrated protagonist" not in selected
    assert "- Summoner: Triệu hồi sư" in selected
    assert "Seria Bladi Demonkill → Summoner" not in selected
    assert "Seria Bladi Demonkill ↔ Summoner" not in selected


@pytest.mark.asyncio
async def test_update_chunk_identity_link_canonicalizes_every_context_layer():
    from src.utils.dialogue_attribution import detect_dialogue_turns
    from src.utils.novel_context import update_novel_context_chunk

    source_text = (
        "The Lieutenant Colonel's office was silent. Eric was watching "
        "Valentine closely. “Stand down,” he said."
    )
    candidates = detect_dialogue_turns(source_text)
    response = MagicMock()
    response.content = (
        "[NEW_CHARACTERS]\n"
        "- Lieutenant Colonel: Unspecified, superior officer of Valentine.\n\n"
        "[IDENTITY_LINKS]\n"
        "- Lieutenant Colonel: Eric\n\n"
        "[NEW_GLOSSARY]\n\n"
        "[DYNAMIC_STATE]\n"
        "## CURRENT ADDRESSING FORMS\n"
        '- Valentine → Lieutenant Colonel: source form "Lieutenant Colonel" '
        "| formal\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Lieutenant Colonel ↔ Valentine: Mutual suspicion.\n\n"
        "[DIALOGUE_ATTRIBUTION]\n"
        + json.dumps({
            "turns": [{
                "id": candidates[0]["id"],
                "speaker": "Lieutenant Colonel",
                "addressee": "Valentine",
                "confidence": 0.98,
            }],
            "state_after": {
                "speaker": "Lieutenant Colonel",
                "addressee": "Valentine",
            },
        })
    )
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    sink = {}
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, vampire hunter and imperial officer.\n"
        "- Valentine: Female, protagonist.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )

    lore, dynamic, _ = await update_novel_context_chunk(
        llm_client=client,
        model_name="model",
        current_global_lore=initial_lore,
        current_dynamic_state="",
        source_chunk=source_text,
        translated_chunk=None,
        source_language="English",
        target_language="Vietnamese",
        dialogue_turns=candidates,
        dialogue_attribution_sink=sink,
    )

    assert lore.count("- Eric:") == 1
    assert "- Lieutenant Colonel: Unspecified" not in lore
    assert "- Lieutenant Colonel: Eric" in lore
    assert "- Valentine → Eric:" in dynamic
    assert "- Eric ↔ Valentine:" in dynamic
    assert sink["turns"][0]["speaker"] == "Eric"
    assert sink["state_after"]["speaker"] == "Eric"


@pytest.mark.asyncio
async def test_update_chunk_repairs_model_missed_reincarnation_gender_and_title_alias():
    from src.utils.novel_context import update_novel_context_chunk

    response = MagicMock()
    response.content = (
        "[NEW_CHARACTERS]\n"
        "- Valentine: Male, protagonist, a terminally ill man who "
        "reincarnates as a vampire named Valentine.\n"
        "- Lieutenant Colonel: Male, superior officer to Valentine, "
        "suspicious of her identity.\n\n"
        "[IDENTITY_LINKS]\n\n"
        "[NEW_GLOSSARY]\n\n"
        "[DYNAMIC_STATE]\n"
        "## CURRENT ADDRESSING FORMS\n\n"
        "## RELATIONSHIP EVOLUTION\n"
    )
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    initial_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Valentine: Unspecified, protagonist.\n"
        "- Eric: Male, protagonist of the game.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    source = (
        "[Valentine, starting reincarnation.] When I woke up again, I had "
        "become a girl with a very small build.\n"
        "The Lieutenant Colonel's office was silent. I stared at Eric, who "
        "had dragged me into the room."
    )

    lore, _dynamic, _logs = await update_novel_context_chunk(
        llm_client=client,
        model_name="model",
        current_global_lore=initial_lore,
        current_dynamic_state="",
        source_chunk=source,
        translated_chunk=None,
        source_language="English",
        target_language="Vietnamese",
    )

    assert "- Valentine: Female," in lore
    assert "terminally ill man who reincarnates" not in lore
    assert "- Lieutenant Colonel: Eric" in lore
    assert "- Lieutenant Colonel: Male" not in lore
    assert lore.count("- Eric:") == 1


@pytest.mark.asyncio
async def test_context_llm_delta_cannot_forget_dormant_relationships():
    from unittest.mock import AsyncMock, MagicMock
    from src.utils.novel_context import update_novel_context_chunk

    mock_client = MagicMock()
    mock_client.generate = AsyncMock()
    response = MagicMock()
    response.content = (
        "[NEW_CHARACTERS]\n\n"
        "[NEW_GLOSSARY]\n\n"
        "[DYNAMIC_STATE]\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        "## CURRENT ADDRESSING FORMS\n"
        '- Guard → Captain: source form "Captain" | formal\n'
        "## RELATIONSHIP EVOLUTION\n"
        "- Guard → Captain: Temporary scene relationship.\n"
    )
    mock_client.generate.return_value = response
    current = (
        "## CURRENT ADDRESSING FORMS\n"
        '- Alice → Bob: source form "Bob" | target-language form "anh" | intimate\n\n'
        "## RELATIONSHIP EVOLUTION\n"
        "- Alice ↔ Bob: Established romantic couple."
    )

    _, updated_dynamic, _ = await update_novel_context_chunk(
        llm_client=mock_client,
        model_name="test-model",
        current_global_lore="",
        current_dynamic_state=current,
        source_chunk="The guard saluted the captain.",
        translated_chunk=None,
        source_language="English",
        target_language="Vietnamese",
    )

    assert '- Alice → Bob: source form "Bob"' in updated_dynamic
    assert "- Alice ↔ Bob: Established romantic couple." in updated_dynamic
    assert "- Guard → Captain: Temporary scene relationship." in updated_dynamic


def test_context_prompts_define_durable_dynamic_state_deltas():
    from src.utils.novel_context import (
        SOURCE_ANALYSIS_SYSTEM_PROMPT,
        UPDATE_SYSTEM_PROMPT,
    )

    for prompt in (SOURCE_ANALYSIS_SYSTEM_PROMPT, UPDATE_SYSTEM_PROMPT):
        assert "Your output must follow this strict format" in prompt
        assert "[NEW_CHARACTERS]" in prompt
        assert "[DIALOGUE_ATTRIBUTION]" in prompt
        assert "Omitted entries remain stored indefinitely." in prompt
        assert "Addressee: DELETE" in prompt
        assert "Character A ↔ Character B: DELETE" in prompt
        assert "register, social basis, scope, and reason" in prompt
        assert "direct address vs indirect reference scope" in prompt
        assert "exception to normal age hierarchy" in prompt
        assert "Sino-Vietnamese literary target term" in prompt
        assert "English named skill, ability, technique, spell, combat move, weapon, artifact, or equipment" in prompt


def test_json_dynamic_addressing_preserves_social_basis_and_scope():
    from src.utils.novel_context import _json_dynamic_state

    dynamic = _json_dynamic_state(
        {
            "current_addressing_forms": [
                {
                    "speaker": "Ellen Evans",
                    "addressee": "Frondier De Roach",
                    "source_form": "Frondier",
                    "target_form": "Frondier",
                    "register": "informal peer-like address",
                    "social_basis": (
                        "Ellen is Aster's older sister while Frondier is "
                        "Aster's same-year peer"
                    ),
                    "scope": "direct address and Ellen's indirect thoughts",
                    "details": "academy setting, not an elder-brother form",
                }
            ]
        }
    )

    assert "## CURRENT ADDRESSING FORMS" in dynamic
    assert "Ellen Evans → Frondier De Roach" in dynamic
    assert 'source form "Frondier"' in dynamic
    assert 'target-language form "Frondier"' in dynamic
    assert "informal peer-like address" in dynamic
    assert "same-year peer" in dynamic
    assert "indirect thoughts" in dynamic
    assert "not an elder-brother form" in dynamic


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



def test_novel_context_filename_sanitization_preserves_unicode():
    assert make_novel_context_filename("Vampire.epub") == "Vampire_context.txt"
    assert (
        make_novel_context_filename("Vampire-1.2_3!@#$.epub")
        == "Vampire-1.2_3_____context.txt"
    )
    assert make_novel_context_filename("!!!.epub") == "____context.txt"
    assert make_novel_context_filename("") == "translation_context.txt"
    assert (
        make_novel_context_filename(
            "\u6211\u600e\u4e48\u53ef\u80fd\u4f1a\u53d8\u6210gal"
            "\u5973\u4e3b\u554a.epub"
        )
        == "\u6211\u600e\u4e48\u53ef\u80fd\u4f1a\u53d8\u6210gal"
        "\u5973\u4e3b\u554a_context.txt"
    )


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


def test_snapshot_decode_returns_canonical_lore_for_resume():
    raw = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Emperor: Female, ruler of the Empire.\n"
            "- Serena Augusta: Female, Emperor of the Empire.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        ),
        "",
    )

    _, global_lore, _ = decode_context_snapshot(
        compress_dynamic_state(raw),
        "",
    )

    assert global_lore.count("- Serena Augusta:") == 1
    assert "- Emperor:" not in global_lore


def test_make_novel_context_filename_is_safe_for_every_input_name():
    assert make_novel_context_filename("Book Name.epub") == "Book_Name_context.txt"
    assert (
        make_novel_context_filename("\u65e5\u672c\u8a9e.docx")
        == "\u65e5\u672c\u8a9e_context.txt"
    )
    assert make_novel_context_filename("", "epub") == "epub_context.txt"
    assert is_safe_filename(make_novel_context_filename("\u65e5\u672c\u8a9e.docx"))
    assert (
        normalize_novel_context_filename("\u65e5\u672c\u8a9e_context.txt")
        == "\u65e5\u672c\u8a9e_context.txt"
    )
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
async def test_plain_text_context_update_interval_skips_between_updates(
    monkeypatch,
    tmp_path,
):
    from unittest.mock import MagicMock
    from src.core.common import plain_text_pipeline
    import src.config

    analyzed = []
    translated = []

    async def fake_update(**kwargs):
        analyzed.append(kwargs["source_chunk"])
        return (
            "# GLOBAL LORE\n\n## GLOSSARY & TERMINOLOGY\n- Master: Maître",
            f"Seen {kwargs['source_chunk']}",
            [],
        )

    async def fake_translate(*, main_content, **kwargs):
        translated.append(main_content)
        return f"FR::{main_content}"

    monkeypatch.setattr(src.config, "NOVEL_CONTEXTS_DIR", tmp_path)
    monkeypatch.setattr(
        "src.utils.novel_context.update_novel_context_chunk",
        fake_update,
    )
    monkeypatch.setattr(
        plain_text_pipeline,
        "generate_translation_request",
        fake_translate,
    )
    monkeypatch.setattr(
        plain_text_pipeline,
        "clean_translated_text",
        lambda value: value,
    )

    checkpoint_manager = MagicMock()
    checkpoint_manager.db.get_chunks.return_value = []
    output, _, interrupted = await plain_text_pipeline.translate_paragraphs_plain(
        paragraphs=["Master arrived.", "Master waited.", "Master left."],
        source_language="English",
        target_language="French",
        model_name="model",
        llm_client=object(),
        max_tokens_per_chunk=3,
        prompt_options={
            "auto_update_context": True,
            "input_filename": "novel.txt",
            "novel_context_update_interval": 2,
        },
        checkpoint_manager=checkpoint_manager,
        translation_id="interval-job",
    )

    assert interrupted is False
    assert all(value.startswith("FR::") for value in output)
    assert translated == ["Master arrived.", "Master waited.", "Master left."]
    assert analyzed == ["Master arrived.", "Master left."]


@pytest.mark.asyncio
async def test_plain_text_retry_preserves_failed_chunk_context_snapshot(
    monkeypatch,
    tmp_path,
):
    from unittest.mock import MagicMock
    from src.core.common import plain_text_pipeline
    import src.config

    attempts = {}
    context_updates = {}

    async def fake_update(current_dynamic_state="", source_chunk="", **kwargs):
        context_updates[source_chunk] = context_updates.get(source_chunk, 0) + 1
        lines = [
            line for line in current_dynamic_state.splitlines()
            if line.strip()
        ]
        lines.append(f"- seen {source_chunk}")
        return "# GLOBAL LORE", "\n".join(lines), []

    async def fake_translate(*, main_content, **kwargs):
        attempts[main_content] = attempts.get(main_content, 0) + 1
        if main_content == "source-1" and attempts[main_content] == 1:
            return None
        return f"FR::{main_content}"

    monkeypatch.setattr(src.config, "NOVEL_CONTEXTS_DIR", tmp_path)
    monkeypatch.setattr(
        "src.utils.novel_context.update_novel_context_chunk",
        fake_update,
    )
    monkeypatch.setattr(
        plain_text_pipeline,
        "generate_translation_request",
        fake_translate,
    )
    monkeypatch.setattr(
        plain_text_pipeline,
        "clean_translated_text",
        lambda value: value,
    )

    checkpoint_manager = MagicMock()
    checkpoint_manager.db.get_chunks.return_value = []
    output, stats, interrupted = await plain_text_pipeline.translate_paragraphs_plain(
        paragraphs=["source-0", "source-1", "source-2"],
        source_language="English",
        target_language="French",
        model_name="model",
        llm_client=object(),
        max_tokens_per_chunk=3,
        prompt_options={
            "auto_update_context": True,
            "input_filename": "novel.txt",
        },
        checkpoint_manager=checkpoint_manager,
        translation_id="plain-retry-job",
    )

    assert interrupted is False
    assert output == ["FR::source-0", "FR::source-1", "FR::source-2"]
    assert stats.failed_chunks == 0
    assert attempts["source-1"] == 2
    assert context_updates["source-1"] == 1

    final_source_1_call = [
        call for call in checkpoint_manager.db.save_chunk.call_args_list
        if call.kwargs["original_text"] == "source-1"
    ][-1]
    assert final_source_1_call.kwargs["status"] == "completed"
    decoded, _, _ = decode_context_snapshot(
        final_source_1_call.kwargs["chunk_data"]["context_snapshot"]
    )
    assert "seen source-1" in decoded
    assert "seen source-2" not in decoded


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


# ---------------------------------------------------------------------------
# consolidate_context_lore tests
# ---------------------------------------------------------------------------

import asyncio
from src.utils.novel_context import consolidate_context_lore


@pytest.mark.asyncio
async def test_consolidation_deduplicates_character_descriptions():
    """consolidate_context_lore should merge duplicate character descriptions."""
    global_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        '- Eric: Male, protagonist of the game "Glory of Victory", a soldier who seeks revenge'
        ' against the Vampire Kingdom; protagonist of "Glory of Victory", a soldier seeking'
        " revenge against the Vampire Kingdom, currently a Lieutenant Colonel.\n"
        "- Kim Ji-an: Female, former beta tester, reincarnated as a vampire.\n"
    )
    # LLM returns a clean consolidated list
    clean_response = (
        "- Eric: Male, protagonist of Glory of Victory, a soldier seeking revenge against the"
        " Vampire Kingdom, currently a Lieutenant Colonel.\n"
        "- Kim Ji-an: Female, former beta tester, reincarnated as a vampire."
    )
    mock_response = MagicMock()
    mock_response.content = clean_response
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(return_value=mock_response)

    result_lore, logs = await consolidate_context_lore(
        llm_client=llm_client,
        model_name="test-model",
        global_lore=global_lore,
    )

    # Consolidation should have run and returned a change log
    assert logs, "Expected a change log entry from the consolidation pass"
    # Duplicate phrase should no longer appear in the result
    assert result_lore.count("protagonist of") == 1, (
        "Duplicate 'protagonist of' phrase should be merged to one occurrence"
    )
    # Kim Ji-an entry should still be present
    assert "Kim Ji-an" in result_lore


@pytest.mark.asyncio
async def test_consolidation_prunes_first_pass_non_character_entries():
    """consolidation should remove non-character entries introduced by earlier context passes."""
    global_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, protagonist and soldier.\n"
        "- Hitchcock: Unspecified, company, a high-quality magical product company known for its eagle emblem.\n"
        "- Menosorpo: Unspecified, a mysterious magic circle acquired as a dungeon conquest reward.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- Hitchcock: Hitchcock\n"
        "- Menosorpo: Menosorpo\n"
    )
    llm_response = (
        "- Eric: Male, protagonist and soldier.\n"
        "- Hitchcock: Unspecified, company, a high-quality magical product company known for its eagle emblem.\n"
        "- Menosorpo: Unspecified, a mysterious magic circle acquired as a dungeon conquest reward.\n"
    )
    mock_response = MagicMock()
    mock_response.content = llm_response
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(return_value=mock_response)

    result_lore, logs = await consolidate_context_lore(
        llm_client=llm_client,
        model_name="test-model",
        global_lore=global_lore,
    )

    system_prompt = llm_client.generate.call_args.kwargs["system_prompt"]
    assert "companies" in system_prompt
    assert "magic circles" in system_prompt
    assert "romanization" in system_prompt
    assert logs
    assert "- Eric: Male, protagonist and soldier." in result_lore
    assert "- Hitchcock: Unspecified" not in result_lore
    assert "- Menosorpo: Unspecified" not in result_lore
    assert "- Hitchcock: Hitchcock" in result_lore
    assert "- Menosorpo: Menosorpo" in result_lore


@pytest.mark.asyncio
async def test_consolidation_skips_on_empty_llm_response():
    """consolidate_context_lore should return original lore unchanged on empty LLM response."""
    global_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, protagonist, soldier.\n"
    )
    mock_response = MagicMock()
    mock_response.content = ""
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(return_value=mock_response)

    result_lore, logs = await consolidate_context_lore(
        llm_client=llm_client,
        model_name="test-model",
        global_lore=global_lore,
    )

    assert result_lore == global_lore, "Lore should be unchanged when LLM returns empty"
    assert logs == [], "No change logs expected on empty response"


@pytest.mark.asyncio
async def test_consolidation_skips_on_no_bullet_entries():
    """consolidate_context_lore should return original lore unchanged when LLM outputs no bullets."""
    global_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, protagonist, soldier.\n"
    )
    mock_response = MagicMock()
    mock_response.content = "Here is the cleaned list:"  # no bullet lines
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(return_value=mock_response)

    result_lore, logs = await consolidate_context_lore(
        llm_client=llm_client,
        model_name="test-model",
        global_lore=global_lore,
    )

    assert result_lore == global_lore
    assert logs == []


@pytest.mark.asyncio
async def test_consolidation_with_asterisks_and_numbered_lists():
    """consolidate_context_lore should support other list markers like asterisks or numbers."""
    global_lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, protagonist of the game Glory of Victory.\n"
    )
    asterisk_response = (
        "* Eric: Male, protagonist of Glory of Victory, currently a Lieutenant Colonel.\n"
        "1. Kim Ji-an: Female, reincarnated as a vampire.\n"
    )
    mock_response = MagicMock()
    mock_response.content = asterisk_response
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(return_value=mock_response)

    result_lore, logs = await consolidate_context_lore(
        llm_client=llm_client,
        model_name="test-model",
        global_lore=global_lore,
    )

    assert logs
    assert "- Eric: Male, protagonist of Glory of Victory, currently a Lieutenant Colonel." in result_lore
    assert "- Kim Ji-an: Female, reincarnated as a vampire." in result_lore


@pytest.mark.asyncio
async def test_consolidation_triggered_on_last_chunk():
    """update_novel_context_chunk should run consolidation on the last chunk regardless of interval."""
    # We mock the LLM client behavior to verify if generate is called for consolidation
    llm_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        "[NEW_CHARACTERS]\n- Eric: Male, protagonist.\n"
        "[IDENTITY_LINKS]\n"
        "[NEW_GLOSSARY]\n"
        "[DYNAMIC_STATE]\n"
    )
    
    # Second response is for the consolidation pass
    mock_consolidation = MagicMock()
    mock_consolidation.content = "- Eric: Male, protagonist, soldier."
    
    llm_client.generate = AsyncMock()
    llm_client.generate.side_effect = [mock_response, mock_consolidation]

    from src.utils.novel_context import update_novel_context_chunk
    
    # chunk_index = 3, total_chunks = 3. Since it is the last chunk, it must trigger consolidation.
    updated_lore, _, logs = await update_novel_context_chunk(
        llm_client=llm_client,
        model_name="test-model",
        current_global_lore="# GLOBAL LORE\n\n## CHARACTERS & GENDERS\n- Eric: Male, protagonist.\n",
        current_dynamic_state="",
        source_chunk="Eric did something.",
        translated_chunk=None,
        source_language="en",
        target_language="vi",
        chunk_index=3,
        total_chunks=3,
    )
    
    # We verify that consolidation ran (which means generate was called twice)
    assert llm_client.generate.call_count == 2
    assert any("[Novel Context] Consolidation pass" in log for log in logs)
    assert "soldier" in updated_lore


def test_filter_abstract_concepts_and_spurious_delete():
    from src.utils.novel_context import (
        _is_disposable_unnamed_character,
        _is_non_character_group_entry,
        _is_non_character_work_entry,
        _parse_bullet_entries,
        normalize_global_lore,
    )

    # Test abstract concepts / hallucinations
    assert _is_non_character_work_entry("Death", "Unspecified, personified concept or hallucination experienced by Kim Si-hu") is True
    assert _is_non_character_work_entry("Fear", "Unspecified, abstract concept that governs the character's choices") is True
    assert _is_non_character_work_entry("Sword", "Unspecified, inanimate object used as a weapon") is True
    assert _is_non_character_work_entry("Shadow", "Unspecified, metaphorical representation of inner guilt") is True
    assert _is_non_character_work_entry("Distress Level", "Unspecified, psychological metric of Kim Si-hu that must be managed to prevent his death") is True
    assert _is_non_character_work_entry("Affection Level", "Unspecified, metric representing Kim Si-hu's favorability toward his summon") is True
    assert _is_non_character_work_entry("Evaluation center", "Unspecified, facility within the Academy used to verify summoner abilities") is True
    assert _is_non_character_work_entry("Granzel", "Unspecified, character mentioned in episode title") is True
    assert _is_non_character_work_entry("Menosorpo", "Unspecified, a mysterious magic circle acquired as a dungeon conquest reward") is True
    assert _is_non_character_group_entry("Hitchcock", "Unspecified, company, a high-quality magical product company known for its eagle emblem") is True
    assert _is_non_character_group_entry("Viet family", "Unspecified, noble family known for merchant power") is True
    assert _is_disposable_unnamed_character("Knight", "Unspecified, knight accompanying Lenya Robert, currently observing the interaction") is True
    assert _is_disposable_unnamed_character("Battle referee", "Unspecified, staff overseeing the match between Kim Si-hu and Lenya Robert") is True

    # Valid characters should NOT be filtered
    assert _is_non_character_work_entry("Reaper", "Female, girl in black rags who works for Valentine") is False
    assert _is_non_character_work_entry("Kim Si-hu", "Male, handsome student") is False
    assert _is_non_character_work_entry("Frondier", "Male, student who acquired a mysterious magic circle") is False
    assert _is_non_character_group_entry("Quinie de Viet", "Female, daughter of the Viet family and a merchant student") is False
    assert _is_non_character_group_entry("Hitchcock heir", "Male, heir of the Hitchcock company") is False
    assert _is_disposable_unnamed_character("Butler", "Male, elderly servant working for the Robert family mansion") is False

    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Distress Level: Unspecified, psychological metric of Kim Si-hu that must be managed to prevent his death.\n"
        "- Evaluation center: Unspecified, facility within the Academy used to verify summoner abilities.\n"
        "- Menosorpo: Unspecified, a mysterious magic circle acquired as a dungeon conquest reward.\n"
        "- Hitchcock: Unspecified, company, a high-quality magical product company known for its eagle emblem.\n"
        "- Viet family: Unspecified, noble family known for merchant power.\n"
        "- Knight: Unspecified, knight accompanying Lenya Robert, currently observing the interaction.\n"
        "- Kim Si-hu: Male, 17-year-old Summoner Academy student.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
        "- Distress Level: Mức độ căng thẳng\n"
        "- Menosorpo: Menosorpo\n"
        "- Hitchcock: Hitchcock\n"
    )
    normalized = normalize_global_lore(lore)
    assert "- Distress Level: Unspecified" not in normalized
    assert "- Evaluation center: Unspecified" not in normalized
    assert "- Menosorpo: Unspecified" not in normalized
    assert "- Hitchcock: Unspecified" not in normalized
    assert "- Viet family: Unspecified" not in normalized
    assert "- Knight: Unspecified" not in normalized
    assert "- Kim Si-hu: Male" in normalized
    assert "- Distress Level: Mức độ căng thẳng" in normalized
    assert "- Menosorpo: Menosorpo" in normalized
    assert "- Hitchcock: Hitchcock" in normalized

    # Test that DELETE bullet entries are skipped
    parsed_bullets = _parse_bullet_entries("- DELETE:\n- DELETE: Death\n- Kim Si-hu: Male, student\n- DELETE")
    assert len(parsed_bullets) == 1
    assert parsed_bullets[0] == ("Kim Si-hu", "Male, student")
