import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.prompts.prompts import (
    generate_refinement_prompt,
    generate_subtitle_block_prompt,
    generate_subtitle_refinement_block_prompt,
    generate_translation_prompt,
)
from src.utils.dialogue_attribution import (
    canonicalize_dialogue_attribution,
    detect_dialogue_turns,
    dialogue_attribution_stats,
    format_dialogue_attribution_for_prompt,
    parse_dialogue_attribution,
)
from src.utils.novel_context import (
    RefinementContextTracker,
    build_novel_context,
    character_alias_map,
    map_dialogue_attributions_for_refinement,
    update_novel_context_chunk,
)


def test_dialogue_detection_supports_multilingual_quotes_dashes_and_subtitles():
    source = (
        "“Are you leaving?”\n"
        "「まだだ」\n"
        "— Then wait for me.\n"
        "[3]괜찮아?\n"
        "This is narration."
    )

    turns = detect_dialogue_turns(source)

    assert [turn["cue"] for turn in turns] == [
        "“Are you leaving?”",
        "「まだだ」",
        "— Then wait for me.",
        "[3]괜찮아?",
    ]
    assert len({turn["id"] for turn in turns}) == 4
    assert all(turn["id"].startswith("dlg-") for turn in turns)


def test_stable_dialogue_ids_do_not_depend_on_chunk_position():
    isolated = detect_dialogue_turns("Narration.\n“Stay here.”\nAfterward.")
    shifted = detect_dialogue_turns(
        "Unrelated earlier paragraph.\nNarration.\n“Stay here.”\nAfterward."
    )

    assert isolated[0]["id"] == shifted[0]["id"]


def test_dialogue_detection_supports_unlabelled_conversation_lines():
    turns = detect_dialogue_turns(
        "Are you coming?\n"
        "In a minute.\n"
        "Do not leave without me!"
    )

    assert [turn["cue"] for turn in turns] == [
        "Are you coming?",
        "In a minute.",
        "Do not leave without me!",
    ]


def test_attribution_rejects_unknown_characters_and_low_confidence_guesses():
    candidates = detect_dialogue_turns("“Stay.”\n“Why?”")
    raw = json.dumps(
        {
            "turns": [
                {
                    "id": candidates[0]["id"],
                    "speaker": "Valentine",
                    "addressee": "Invented Person",
                    "confidence": 0.92,
                },
                {
                    "id": candidates[1]["id"],
                    "speaker": "Eric",
                    "addressee": "Valentine",
                    "confidence": 0.4,
                },
            ],
            "state_after": {
                "speaker": "Invented Person",
                "addressee": "Valentine",
            },
        }
    )

    parsed = parse_dialogue_attribution(
        raw,
        candidates,
        {"valentine": "Valentine", "eric": "Eric"},
    )

    assert parsed["turns"][0]["speaker"] == "Valentine"
    assert parsed["turns"][0]["addressee"] == "Unknown"
    assert parsed["state_after"]["speaker"] == "Valentine"
    assert parsed["state_after"]["addressee"] == "Valentine"
    assert dialogue_attribution_stats(parsed) == {
        "identified": 2,
        "assigned": 1,
        "uncertain": 1,
    }


def test_dialogue_attribution_resolves_explicit_title_alias_to_canonical_name():
    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, imperial officer.\n\n"
        "## CHARACTER ALIASES\n"
        "- Lieutenant Colonel: Eric\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    candidates = detect_dialogue_turns("“Stand down.”")
    raw = json.dumps(
        {
            "turns": [{
                "id": candidates[0]["id"],
                "speaker": "Lieutenant Colonel",
                "addressee": "Eric",
                "confidence": 0.97,
            }],
            "state_after": {
                "speaker": "Lieutenant Colonel",
                "addressee": "Eric",
            },
        }
    )

    parsed = parse_dialogue_attribution(
        raw,
        candidates,
        character_alias_map(lore),
    )

    assert parsed["turns"][0]["speaker"] == "Eric"
    assert parsed["turns"][0]["addressee"] == "Eric"
    assert parsed["state_after"] == {
        "speaker": "Eric",
        "addressee": "Eric",
    }


def test_saved_dialogue_map_is_recanonicalized_after_identity_merge():
    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, imperial officer.\n\n"
        "## CHARACTER ALIASES\n"
        "- Lieutenant Colonel: Eric\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    saved = {
        "version": 1,
        "turns": [{
            "id": "dlg-old",
            "cue": "Stand down.",
            "speaker": "Lieutenant Colonel",
            "addressee": "Valentine",
            "confidence": 0.9,
        }],
        "state_after": {
            "speaker": "Lieutenant Colonel",
            "addressee": "Valentine",
        },
    }

    migrated = canonicalize_dialogue_attribution(
        saved,
        {
            **character_alias_map(lore),
            "valentine": "Valentine",
        },
    )

    assert migrated["turns"][0]["speaker"] == "Eric"
    assert migrated["state_after"]["speaker"] == "Eric"


def test_translation_prompt_recanonicalizes_legacy_dialogue_aliases():
    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Eric: Male, imperial officer.\n\n"
        "## CHARACTER ALIASES\n"
        "- Lieutenant Colonel: Eric\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    prompt = generate_translation_prompt(
        main_content="“Stand down.”",
        context_before="",
        context_after="",
        previous_translation_context="",
        prompt_options={
            "novel_context": build_novel_context(lore, ""),
            "dialogue_attribution": {
                "version": 1,
                "turns": [{
                    "id": "dlg-old",
                    "cue": "“Stand down.”",
                    "speaker": "Lieutenant Colonel",
                    "addressee": "Eric",
                    "confidence": 0.95,
                }],
            },
        },
    )

    assert '"speaker":"Eric"' in prompt.user
    assert '"speaker":"Lieutenant Colonel"' not in prompt.user


def test_prompt_metadata_contains_no_output_labels_instruction():
    candidates = detect_dialogue_turns("“Stay.”")
    attribution = {
        "turns": [
            {
                **candidates[0],
                "speaker": "Valentine",
                "addressee": "Eric",
                "confidence": 0.95,
            }
        ]
    }

    section = format_dialogue_attribution_for_prompt(attribution)

    assert "Valentine" in section
    assert "Never print IDs, speaker labels" in section

    prompt_options = {"dialogue_attribution": attribution}
    prompts = [
        generate_translation_prompt(
            main_content="“Stay.”",
            context_before="",
            context_after="",
            previous_translation_context="",
            prompt_options=prompt_options,
        ),
        generate_refinement_prompt(
            draft_translation="“Stay.”",
            prompt_options=prompt_options,
        ),
        generate_subtitle_block_prompt(
            subtitle_blocks=[(0, "Stay.")],
            previous_translation_block="",
            prompt_options=prompt_options,
        ),
        generate_subtitle_refinement_block_prompt(
            subtitle_blocks=[(0, "Stay.")],
            prompt_options=prompt_options,
        ),
    ]
    for prompt in prompts:
        assert "# SCENE-LOCAL DIALOGUE ATTRIBUTION" in prompt.user
        assert "Never print IDs, speaker labels" in prompt.user


@pytest.mark.asyncio
async def test_source_analysis_returns_hidden_attribution_without_polluting_lore():
    candidates = detect_dialogue_turns("“Stay with me,” Valentine said.")
    response = MagicMock()
    response.content = (
        "[NEW_CHARACTERS]\n\n"
        "[IDENTITY_LINKS]\n\n"
        "[NEW_GLOSSARY]\n\n"
        "[DYNAMIC_STATE]\n"
        "## CURRENT ADDRESSING FORMS\n\n"
        "## RELATIONSHIP EVOLUTION\n"
        "- Valentine ↔ Eric: They trust each other.\n\n"
        "[DIALOGUE_ATTRIBUTION]\n"
        + json.dumps(
            {
                "turns": [
                    {
                        "id": candidates[0]["id"],
                        "speaker": "Valentine",
                        "addressee": "Eric",
                        "confidence": 0.98,
                    }
                ],
                "state_after": {
                    "speaker": "Valentine",
                    "addressee": "Eric",
                },
            }
        )
    )
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    sink = {}
    lore = (
        "# GLOBAL LORE\n\n"
        "## CHARACTERS & GENDERS\n"
        "- Valentine: Female, protagonist.\n"
        "- Eric: Male, officer.\n\n"
        "## GLOSSARY & TERMINOLOGY"
    )

    updated_lore, _, _ = await update_novel_context_chunk(
        llm_client=client,
        model_name="model",
        current_global_lore=lore,
        current_dynamic_state="",
        source_chunk="“Stay with me,” Valentine said.",
        translated_chunk=None,
        source_language="English",
        target_language="Vietnamese",
        dialogue_turns=candidates,
        dialogue_attribution_sink=sink,
    )

    assert sink["turns"][0]["speaker"] == "Valentine"
    assert "DIALOGUE_ATTRIBUTION" not in updated_lore
    assert "dlg-" not in updated_lore
    prompt = client.generate.call_args.kwargs["prompt"]
    assert candidates[0]["id"] in prompt


def test_refinement_reuses_dialogue_maps_only_for_exact_unit_alignment():
    attribution = {"version": 1, "turns": [{"id": "dlg-a"}]}
    rows = [
        {
            "chunk_index": 0,
            "status": "completed",
            "translated_text": "One",
            "chunk_data": {"dialogue_attribution": attribution},
        }
    ]

    assert map_dialogue_attributions_for_refinement(1, rows) == [attribution]
    assert map_dialogue_attributions_for_refinement(2, rows) == [None, None]


@pytest.mark.asyncio
async def test_refinement_speaker_state_carries_within_chapter_and_resets_between(
    monkeypatch,
):
    observed_states = []

    async def fake_update(**kwargs):
        observed_states.append(dict(kwargs["current_dialogue_state"]))
        kwargs["dialogue_attribution_sink"].update(
            {
                "version": 1,
                "turns": [],
                "state_after": {"speaker": "Valentine"},
            }
        )
        return kwargs["current_global_lore"], kwargs["current_dynamic_state"], []

    monkeypatch.setattr(
        "src.utils.novel_context.update_novel_context_chunk",
        fake_update,
    )
    tracker = RefinementContextTracker(
        prompt_options={
            "auto_update_context": True,
            "novel_context": build_novel_context("# GLOBAL LORE", ""),
        },
        historical_contexts=[None, None, None],
    )

    for index, scene_key in enumerate((0, 0, 1), start=1):
        await tracker.next_context(
            text="“Stay.”",
            llm_client=object(),
            model_name="model",
            target_language="English",
            display_index=index,
            total_chunks=3,
            scene_key=scene_key,
        )

    assert observed_states == [
        {},
        {"speaker": "Valentine"},
        {},
    ]
