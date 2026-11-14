"""attach_draft_prompt_context: one rule set for every translation pipeline."""

from types import SimpleNamespace

import pytest

from src.core.context import unit_pipeline


@pytest.fixture
def recorded(monkeypatch):
    calls = {}

    def addressing(**kwargs):
        calls["addressing_options"] = dict(kwargs["prompt_options"])
        return "ADDRESSING"

    def relationships(**kwargs):
        calls["relationship_reference"] = kwargs["reference_text"]
        return "RELATIONSHIPS"

    def narrator(translation_id, db, *, chunk_index, target_language):
        calls["narrator_chunk"] = chunk_index
        return "NARRATOR"

    def names(**kwargs):
        calls["names_attribution"] = kwargs["dialogue_attribution"]
        return ["Alice"]

    monkeypatch.setattr(
        "src.utils.db_addressing.build_directed_addressing_prompt_context", addressing
    )
    monkeypatch.setattr(
        "src.utils.relationship_sync.build_relationship_prompt_context", relationships
    )
    monkeypatch.setattr(
        "src.utils.narrator_voice.build_narrator_voice_context", narrator
    )
    monkeypatch.setattr(unit_pipeline, "relevant_character_names", names)
    monkeypatch.setattr(unit_pipeline, "build_unit_prompt_context", lambda **kw: "BUNDLE")
    return calls


def _attach(session, **overrides):
    options = {}
    kwargs = dict(
        translation_id="job",
        db=object(),
        context_session=session,
        source_text="BODY",
        source_language="English",
        target_language="Vietnamese",
        chunk_index=7,
    )
    kwargs.update(overrides)
    return unit_pipeline.attach_draft_prompt_context(options, **kwargs)


def test_stored_attribution_wins_over_the_session(recorded):
    session = SimpleNamespace(global_lore="", dialogue_attribution={"turns": ["session"]})
    options = _attach(session, stored_dialogue_attribution={"turns": ["stored"]})
    assert options["dialogue_attribution"] == {"turns": ["stored"]}
    assert recorded["names_attribution"] == {"turns": ["stored"]}


def test_session_attribution_is_the_fallback(recorded):
    session = SimpleNamespace(global_lore="", dialogue_attribution={"turns": ["session"]})
    options = _attach(session)
    assert options["dialogue_attribution"] == {"turns": ["session"]}


def test_names_are_computed_without_any_attribution(recorded):
    session = SimpleNamespace(global_lore="", dialogue_attribution=None)
    options = _attach(session)
    assert options["active_character_names"] == ["Alice"]
    assert "dialogue_attribution" not in options
    # Addressing selection sees the names computed for this unit.
    assert recorded["addressing_options"]["active_character_names"] == ["Alice"]


def test_no_session_means_no_names(recorded):
    options = _attach(None)
    assert "active_character_names" not in options


def test_relationships_read_the_unit_with_its_nearby_source(recorded):
    _attach(None, nearby_source=("BEFORE", "AFTER"))
    assert recorded["relationship_reference"] == "BEFORE\nBODY\nAFTER"


def test_every_section_and_the_bundle_are_attached(recorded):
    options = _attach(None)
    assert options["directed_addressing_context"] == "ADDRESSING"
    assert options["relationship_context"] == "RELATIONSHIPS"
    assert options["narrative_voice_context"] == "NARRATOR"
    assert options["prompt_context_bundle"] == "BUNDLE"
    assert recorded["narrator_chunk"] == 7
