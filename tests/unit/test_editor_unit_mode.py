"""The editor answering by unit id instead of by a span it has to quote.

The two contracts run side by side and the span one is still in charge, so what
these pin is that the new path is complete on its own -- prompt, parse, gate,
patch -- and that turning it on is the only thing that changes.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.config as config
from src.core.editor.prompting import compose_reflection_prompts
from src.core.editor.unit_repair import (
    MAX_REWRITE_CHANGE_RATIO,
    collect_unit_rewrite_patches,
    unit_mode_enabled,
)
from src.core.translator import parse_reflection_result
from src.utils.translation_quality import apply_editor_patches

SOURCE = "Alpha one is here.\n\nBeta two is here.\n\nGamma three is here."
DRAFT = "Alpha un est ici.\n\nBeta deux est ici.\n\nGamma trois est ici."


def _unit_issue(**overrides):
    issue = {
        "issue_id": "issue-1",
        "unit_id": "U-0002",
        "category": "mistranslation",
        "severity": "major",
        "confidence": 0.9,
        "repair_kind": "unit_rewrite",
        "instruction": "Beta was rendered with the wrong verb.",
        "rewritten_unit": "Beta deux se trouve ici.",
    }
    issue.update(overrides)
    return issue


def _compose(options):
    return compose_reflection_prompts(
        source_chunk=SOURCE,
        draft_translation=DRAFT,
        target_language="French",
        novel_context="",
        custom_instructions="",
        glossary_block="",
        deterministic_findings="",
        narrative_voice_context="",
        source_available=True,
        options=options,
    )


def test_unit_mode_is_off_until_something_turns_it_on():
    assert config.EDITOR_UNIT_MODE is False
    assert unit_mode_enabled({}) is False
    assert unit_mode_enabled({"editor_unit_mode": True}) is True


def test_the_unit_prompt_shows_the_source_beside_the_draft():
    prompts = _compose({"editor_unit_mode": True})
    user = prompts.reflection_pair.user
    assert "[U-0001]" in user and "[U-0003]" in user
    assert "  SRC   Alpha one is here." in user
    assert "  DRAFT Alpha un est ici." in user
    # Each paragraph appears once, next to its own translation. The span prompt
    # sends the same two texts and is 0.9% shorter for it; what changes is that
    # the pairing is stated rather than left to the reader.
    assert user.count("Alpha one is here.") == 1
    assert prompts.components["input_mode"] == "unit_bitext"
    assert prompts.components["source_complete"] is True


def test_the_span_contract_is_untouched_when_the_flag_is_off():
    prompts = _compose({})
    assert "[U-0001]" not in prompts.reflection_pair.user
    assert prompts.components["input_mode"] == "complete_audit"
    assert "# RAW SOURCE CHUNK:" in prompts.reflection_pair.user


def test_a_unit_answer_parses_without_a_quote_or_a_replacement():
    result = parse_reflection_result(
        '{"status":"needs_repair","issues":[{"issue_id":"i1",'
        '"unit_id":"u-0002","category":"mistranslation","severity":"major",'
        '"confidence":0.9,"instruction":"Wrong verb.",'
        '"rewritten_unit":"Beta deux se trouve ici.","glossary_update":null}],'
        '"voice_observations":[]}'
    )
    assert result.status == "needs_repair"
    issue = result.issues[0]
    assert issue["unit_id"] == "U-0002"
    assert issue["repair_kind"] == "unit_rewrite"
    assert issue["rewritten_unit"] == "Beta deux se trouve ici."
    assert issue["draft_quote"] == ""
    assert issue["draft_replacement"] is None


def test_a_hedged_unit_finding_is_still_review_only():
    result = parse_reflection_result(
        '{"status":"needs_repair","issues":[{"issue_id":"i1",'
        '"unit_id":"U-0002","category":"style","severity":"minor",'
        '"confidence":0.4,"instruction":"Might read better.",'
        '"rewritten_unit":"Beta deux se trouve ici."}],"voice_observations":[]}'
    )
    assert result.issues[0]["repair_kind"] == "review_only"


def test_a_rewritten_unit_lands_in_the_draft_without_being_searched_for():
    patched, unresolved, errors = apply_editor_patches(
        DRAFT, [_unit_issue()], source_text=SOURCE,
    )
    assert errors == [] and unresolved == []
    assert patched == (
        "Alpha un est ici.\n\nBeta deux se trouve ici.\n\nGamma trois est ici."
    )


def test_a_rewrite_that_touches_more_than_the_defect_is_refused():
    # The measurement the span contract could never make: a span edit only ever
    # showed the span it admitted to, so a model quietly retranslating the
    # paragraph looked exactly like a model fixing one word in it.
    patches, unresolved = collect_unit_rewrite_patches(
        DRAFT,
        SOURCE,
        [_unit_issue(rewritten_unit="Rien de tout cela ne ressemble a l'original.")],
    )
    assert patches == []
    assert unresolved[0]["unresolved_reason"] == "unit_rewrite_drift"


def test_a_repair_just_under_the_drift_limit_is_still_applied():
    patched, unresolved, _ = apply_editor_patches(
        DRAFT, [_unit_issue(rewritten_unit="Beta deux est la.")], source_text=SOURCE,
    )
    assert unresolved == []
    assert "Beta deux est la." in patched
    assert MAX_REWRITE_CHANGE_RATIO < 1.0


def test_an_id_no_unit_carries_is_refused_rather_than_guessed():
    patches, unresolved = collect_unit_rewrite_patches(
        DRAFT, SOURCE, [_unit_issue(unit_id="U-0099")],
    )
    assert patches == []
    assert unresolved[0]["unresolved_reason"] == "unit_unknown"


def test_a_unit_the_draft_never_produced_has_nothing_to_patch():
    # An omission is a real finding and a job for the rewrite pass; there is no
    # draft text to replace, so it must leave here unresolved rather than
    # silently succeed.
    source = "Alpha one is here.\n\nBeta two is entirely missing.\n\nGamma three is here."
    draft = "Alpha one is here.\n\nGamma three is here."
    patches, unresolved = collect_unit_rewrite_patches(
        draft,
        source,
        [_unit_issue(unit_id="U-0002", rewritten_unit="Beta deux est ici.")],
    )
    assert patches == []
    assert unresolved[0]["unresolved_reason"] == "unit_has_no_draft"


def test_two_rewrites_of_one_unit_are_two_answers_to_one_question():
    patches, unresolved = collect_unit_rewrite_patches(
        DRAFT,
        SOURCE,
        [
            _unit_issue(),
            _unit_issue(issue_id="issue-2", rewritten_unit="Beta deux est la."),
        ],
    )
    assert len(patches) == 1
    assert unresolved[0]["unresolved_reason"] == "unit_rewrite_conflict"


def test_a_rewrite_that_changed_nothing_is_not_a_repair():
    patches, unresolved = collect_unit_rewrite_patches(
        DRAFT, SOURCE, [_unit_issue(rewritten_unit="Beta deux est ici.")],
    )
    assert patches == []
    assert unresolved[0]["unresolved_reason"] == "unit_rewrite_no_op"


def test_a_quoted_span_and_a_rewritten_unit_apply_in_the_same_pass():
    # The deterministic validator still reports spans while the editor answers
    # in units, and both are measured against the untouched draft -- applying
    # one kind first would move the ground under the other.
    span_issue = {
        "issue_id": "issue-2",
        "category": "glossary error",
        "severity": "major",
        "confidence": 0.9,
        "repair_kind": "local_replace",
        "draft_quote": "Alpha un est ici.",
        "draft_replacement": {"draft": "Alpha un", "replacement": "Alpha premier"},
    }
    patched, unresolved, errors = apply_editor_patches(
        DRAFT, [span_issue, _unit_issue()], source_text=SOURCE,
    )
    assert errors == [] and unresolved == []
    assert patched == (
        "Alpha premier est ici.\n\nBeta deux se trouve ici.\n\n"
        "Gamma trois est ici."
    )


@pytest.mark.asyncio
async def test_the_whole_pass_repairs_a_chunk_from_a_unit_answer():
    """End to end, with nothing quoted and nothing searched for.

    The pieces above are each pinned on their own; what this pins is that they
    are actually connected -- prompt built from the alignment, answer parsed
    under the unit contract, spans computed from the rewrite, draft repaired.
    """

    prompts = []

    async def generate_async(**kwargs):
        prompts.append(kwargs.get("prompt") or "")
        from src.core.llm_client import LLMResponse

        return LLMResponse(content=json.dumps({
            "status": "needs_repair",
            "issues": [{
                "issue_id": "i1",
                "unit_id": "U-0002",
                "category": "mistranslation",
                "severity": "major",
                "confidence": 0.95,
                "instruction": "Beta was rendered with the wrong verb.",
                "rewritten_unit": "Beta deux se trouve ici.",
                "glossary_update": None,
            }],
            "voice_observations": [],
        }))

    client = MagicMock()
    client.generate_async = AsyncMock(side_effect=generate_async)

    from src.core.translator import run_chunk_reflection_pass

    result = await run_chunk_reflection_pass(
        source_chunk=SOURCE,
        draft_translation=DRAFT,
        target_language="French",
        model_name="editor-model",
        llm_client=client,
        prompt_options={
            "editor_unit_mode": True,
            "editor_provider_resolved": "gemini",
            "editor_model_resolved": "editor-model",
        },
    )
    assert "[U-0002]" in prompts[0]
    assert result == (
        "Alpha un est ici.\n\nBeta deux se trouve ici.\n\nGamma trois est ici."
    )


def test_without_unit_issues_nothing_about_the_old_path_changes():
    span_issue = {
        "issue_id": "issue-1",
        "repair_kind": "local_replace",
        "draft_quote": "Beta deux est ici.",
        "draft_replacement": {"draft": "deux", "replacement": "second"},
    }
    patched, unresolved, errors = apply_editor_patches(DRAFT, [span_issue])
    assert errors == [] and unresolved == []
    assert "Beta second est ici." in patched
