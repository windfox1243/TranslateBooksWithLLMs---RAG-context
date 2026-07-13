"""Structured, evidence-backed narrator voice regressions."""

import pytest

from src.persistence.database import Database
from src.utils.language_profiles import supported_translation_languages, get_language_profile
from src.utils.narrator_voice import (
    NarratorVoiceProfile,
    build_narrator_voice_context,
    persist_voice_observations,
    profile_from_observations,
    validate_voice_observations,
)


def _apollo_observations():
    source = "I entered the room. I still wanted to become the strongest stayer."
    target = "Tôi bước vào phòng. Tôi vẫn muốn trở thành Stayer mạnh nhất."
    raw = [
        {"segment_id": "SEG-0001", "discourse_mode": "narration",
         "narrator_key": "apollo", "narrator_identity": "Apollo Rainbow",
         "point_of_view": "first", "dimensions": {"self_reference": "tôi"},
         "source_quote": "I entered the room.", "target_quote": "Tôi bước vào phòng.",
         "transition_type": "none", "transition_evidence": "", "confidence": 0.98},
        {"segment_id": "SEG-0002", "discourse_mode": "thought",
         "narrator_key": "apollo", "narrator_identity": "Apollo Rainbow",
         "point_of_view": "first", "dimensions": {"self_reference": "tôi"},
         "source_quote": "I still wanted to become the strongest stayer.",
         "target_quote": "Tôi vẫn muốn trở thành Stayer mạnh nhất.",
         "transition_type": "none", "transition_evidence": "", "confidence": 0.96},
    ]
    return source, target, raw


def test_apollo_profile_uses_exact_narrative_evidence_not_dialogue_counts():
    source, target, raw = _apollo_observations()
    accepted, rejected = validate_voice_observations(
        raw, source_text=source, target_text=target, target_language="Vietnamese"
    )
    assert rejected == []
    profile = profile_from_observations(accepted, chunk_index=3)
    assert profile is not None
    assert profile.self_reference == "tôi"
    assert profile.start_chunk_index == 3


def test_ungrounded_voice_metadata_is_discarded_without_affecting_other_data():
    source, target, raw = _apollo_observations()
    raw[0]["target_quote"] = "tớ"
    accepted, rejected = validate_voice_observations(
        raw, source_text=source, target_text=target, target_language="Vietnamese"
    )
    assert len(accepted) == 1
    assert rejected[0]["reason"] == "target_evidence_not_unique"
    assert profile_from_observations(accepted, chunk_index=0) is None


def test_all_supported_languages_declare_voice_dimensions():
    assert len(supported_translation_languages()) == 17
    for language in supported_translation_languages():
        dimensions = get_language_profile(language).narrator_voice_dimensions
        assert "point_of_view" in dimensions
        assert "style" in dimensions


def test_historical_timeline_and_locked_profile_precedence(tmp_path):
    db = Database(str(tmp_path / "voice.db"))
    db.upsert_narrator_voice_profile("job", NarratorVoiceProfile(
        narrator_key="apollo", narrator_identity="Apollo Rainbow",
        point_of_view="first", self_reference="tôi",
        dimensions={"self_reference": "tôi"}, confidence=1.0,
        provenance="user_manual", status="active", is_locked=True,
        start_chunk_index=0, end_chunk_index=4,
    ).to_dict())
    db.upsert_narrator_voice_profile("job", NarratorVoiceProfile(
        narrator_key="other", point_of_view="third", confidence=0.95,
        provenance="senior_editor", status="active", start_chunk_index=5,
    ).to_dict())
    early = build_narrator_voice_context(
        "job", db, chunk_index=2, target_language="Vietnamese"
    )
    late = build_narrator_voice_context(
        "job", db, chunk_index=6, target_language="Vietnamese"
    )
    assert "self_reference=tôi" in early
    assert "other" not in early
    assert "other" in late
    assert "apollo" not in late


def test_unsupported_transition_creates_conflict_and_resync_preserves_lock(tmp_path):
    db = Database(str(tmp_path / "voice.db"))
    source, target, raw = _apollo_observations()
    accepted, _ = validate_voice_observations(
        raw, source_text=source, target_text=target, target_language="Vietnamese"
    )
    persist_voice_observations(db, "job", 0, accepted)
    changed = []
    for item in accepted:
        item.point_of_view = "third"
        item.dimensions = {"self_reference": "ta"}
        item.segment_id += "-B"
        changed.append(item)
    stats = persist_voice_observations(db, "job", 2, changed)
    assert stats["conflicts"] == 1
    profile = db.get_narrator_voice_profiles("job")[0]
    db.set_narrator_voice_profile_lock("job", profile["id"], True)
    db.quarantine_narrator_voice_after("job", 0)
    assert db.get_narrator_voice_profiles("job")[0]["is_locked"] is True


@pytest.mark.parametrize("language,dimension,value", [
    ("English", "tense", "past"), ("French", "formality", "literary"),
    ("Spanish", "regional_register", "neutral"), ("German", "gender", "female"),
    ("Vietnamese", "self_reference", "tôi"), ("Chinese", "pronoun_omission", "frequent"),
    ("Japanese", "persona", "reserved"), ("Korean", "speech_level", "haeyoche"),
    ("Arabic", "number", "singular"), ("Russian", "gender", "female"),
    ("Hindi", "formality", "neutral"), ("Thai", "politeness", "neutral"),
    ("Italian", "tense", "past"), ("Portuguese", "regional_register", "neutral"),
    ("Dutch", "formality", "neutral"), ("Polish", "gender", "female"),
    ("Turkish", "tense", "past"),
])
def test_language_family_voice_evidence_fixture(language, dimension, value):
    raw = [{
        "segment_id": "SEG-0001", "discourse_mode": "narration",
        "narrator_key": "n", "narrator_identity": "unknown",
        "point_of_view": "first", "dimensions": {dimension: value, "not_valid": "x"},
        "source_quote": "Unique source.", "target_quote": "Unique target.",
        "transition_type": "none", "transition_evidence": "", "confidence": 0.95,
    }]
    accepted, rejected = validate_voice_observations(
        raw, source_text="Unique source.", target_text="Unique target.",
        target_language=language,
    )
    assert rejected == []
    assert accepted[0].dimensions == {dimension: value}


def test_supported_chapter_transition_and_multiple_narrators(tmp_path):
    db = Database(str(tmp_path / "switch.db"))
    source, target, raw = _apollo_observations()
    first, _ = validate_voice_observations(
        raw, source_text=source, target_text=target, target_language="Vietnamese"
    )
    persist_voice_observations(db, "job", 0, first, chapter_index=0)
    switched = []
    for narrator_key, identity, suffix in (("apollo", "Apollo Rainbow", "A"), ("guest", "Guest", "G")):
        for index, item in enumerate(raw):
            candidate = dict(item)
            candidate.update({
                "segment_id": f"SEG-{suffix}{index}",
                "narrator_key": narrator_key,
                "narrator_identity": identity,
                "point_of_view": "third" if narrator_key == "apollo" else "first",
                "dimensions": {"self_reference": "cô ấy" if narrator_key == "apollo" else "tôi"},
                "transition_type": "chapter",
            })
            switched.append(candidate)
    accepted, _ = validate_voice_observations(
        switched, source_text=source, target_text=target, target_language="Vietnamese"
    )
    stats = persist_voice_observations(
        db, "job", 4, accepted, chapter_index=1,
    )
    assert stats["profiles"] == 2
    assert {item["narrator_key"] for item in db.get_narrator_voice_profiles(
        "job", effective_chunk_index=4,
    )} == {"apollo", "guest"}


def test_dialogue_letters_and_malformed_quotes_do_not_establish_narrator():
    source, target, raw = _apollo_observations()
    raw[0]["discourse_mode"] = "dialogue"
    raw[1]["discourse_mode"] = "letter"
    accepted, rejected = validate_voice_observations(
        raw, source_text=source, target_text=target, target_language="Vietnamese"
    )
    assert rejected == []
    assert profile_from_observations(accepted, chunk_index=0) is None
    raw[0]["source_quote"] = '"Unclosed quotation'
    _, rejected = validate_voice_observations(
        raw, source_text=source, target_text=target, target_language="Vietnamese"
    )
    assert rejected[0]["reason"] == "source_evidence_not_unique"
