"""Structured, evidence-backed narrator voice regressions."""

from types import SimpleNamespace

import pytest

from src.persistence.database import Database
from src.utils.language_profiles import supported_translation_languages, get_language_profile
from src.utils.narrator_voice import (
    NarratorVoiceProfile,
    bootstrap_narrator_voice,
    build_narrator_voice_context,
    eligible_bootstrap_boundary,
    narrator_policy_payload,
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


def test_all_supported_languages_declare_safe_narrator_default_policy():
    expected = {
        "English": ("explicit", "I"), "French": ("explicit", "je"),
        "German": ("explicit", "ich"), "Vietnamese": ("explicit", "tôi"),
        "Russian": ("explicit", "я"), "Hindi": ("explicit", "मैं"),
        "Dutch": ("explicit", "ik"), "Spanish": ("pro_drop", ""),
        "Arabic": ("pro_drop", ""), "Italian": ("pro_drop", ""),
        "Portuguese": ("pro_drop", ""), "Polish": ("pro_drop", ""),
        "Turkish": ("pro_drop", ""), "Chinese": ("omission_preferred", ""),
        "Japanese": ("omission_preferred", ""),
        "Korean": ("omission_preferred", ""),
        "Thai": ("persona_required", ""),
    }
    assert set(expected) == set(supported_translation_languages())
    for language, (strategy, self_reference) in expected.items():
        policy = narrator_policy_payload(language)
        assert policy["strategy"] == strategy
        assert policy["self_reference"] == self_reference


@pytest.mark.parametrize(
    "completed,expected", [(0, None), (1, None), (2, 2), (3, 2), (4, 4),
                           (7, 4), (8, 8), (11, 8), (12, 12), (13, 12),
                           (16, 16), (21, 20)],
)
def test_bootstrap_retry_boundaries(completed, expected):
    assert eligible_bootstrap_boundary(completed) == expected


@pytest.mark.asyncio
async def test_preflight_policy_matrix_covers_all_languages_and_formats(
    tmp_path, monkeypatch,
):
    async def generate(**_kwargs):
        return SimpleNamespace(content='''{
          "source_narrator": {
            "narrative_detected": true, "point_of_view": "first",
            "narrator_key": "n", "narrator_identity": "Narrator",
            "chunk_index": 0, "source_quote": "I crossed the station.",
            "confidence": 0.97, "voice_over": true
          }
        }''')

    monkeypatch.setattr("src.core.translator._generate_editor_response", generate)
    db = Database(str(tmp_path / "matrix.db"))
    for language in supported_translation_languages():
        for file_type in ("txt", "epub", "srt", "docx"):
            translation_id = f"{language}-{file_type}"
            result = await bootstrap_narrator_voice(
                db=db, translation_id=translation_id,
                chunks=[{
                    "chunk_index": 0, "status": "pending",
                    "original_text": "I crossed the station.",
                    "translated_text": "",
                }],
                target_language=language, model_name="editor",
                llm_client=object(), file_type=file_type,
            )
            assert result["status"] == "provisional"
            profile = db.get_narrator_voice_profiles(
                translation_id, include_inactive=True,
            )[0]
            assert profile["status"] == "provisional"
            assert profile["dimensions"]["self_reference_strategy"] == (
                narrator_policy_payload(language)["strategy"]
            )


@pytest.mark.asyncio
async def test_srt_dialogue_does_not_create_narrator_without_voice_over(
    tmp_path, monkeypatch,
):
    async def generate(**_kwargs):
        return SimpleNamespace(content='''{"source_narrator": {
          "narrative_detected": true, "point_of_view": "first",
          "narrator_key": "speaker", "narrator_identity": "unknown",
          "chunk_index": 0, "source_quote": "I will be there.",
          "confidence": 0.99, "voice_over": false
        }}''')

    monkeypatch.setattr("src.core.translator._generate_editor_response", generate)
    db = Database(str(tmp_path / "srt.db"))
    result = await bootstrap_narrator_voice(
        db=db, translation_id="dialogue-srt",
        chunks=[{"chunk_index": 0, "status": "pending",
                 "original_text": "I will be there.", "translated_text": ""}],
        target_language="Vietnamese", model_name="editor",
        llm_client=object(), file_type="srt",
    )
    assert result["status"] == "ambiguous"
    assert db.get_narrator_voice_profiles(
        "dialogue-srt", include_inactive=True,
    ) == []


@pytest.mark.asyncio
async def test_ambiguous_two_unit_bootstrap_retries_at_twelve_units(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "retry.db"))
    calls = 0

    async def generate(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(content='{"voice_observations": []}')
        return SimpleNamespace(content='''{
          "voice_observations": [
            {"chunk_index": 0, "segment_id": "SEG-A", "discourse_mode": "narration",
             "narrator_key": "n", "narrator_identity": "Narrator", "point_of_view": "first",
             "dimensions": {"self_reference": "tôi"}, "source_quote": "Source 0.",
             "target_quote": "Target 0.", "transition_type": "none",
             "transition_evidence": "", "confidence": 0.98},
            {"chunk_index": 2, "segment_id": "SEG-B", "discourse_mode": "narration",
             "narrator_key": "n", "narrator_identity": "Narrator", "point_of_view": "first",
             "dimensions": {"self_reference": "tôi"}, "source_quote": "Source 2.",
             "target_quote": "Target 2.", "transition_type": "none",
             "transition_evidence": "", "confidence": 0.97}
          ]
        }''')

    monkeypatch.setattr("src.core.translator._generate_editor_response", generate)
    chunks = [{
        "chunk_index": index, "status": "completed",
        "original_text": f"Source {index}.",
        "translated_text": f"Target {index}.",
    } for index in range(13)]
    first = await bootstrap_narrator_voice(
        db=db, translation_id="job", chunks=chunks[:2],
        target_language="Vietnamese", model_name="editor", llm_client=object(),
        file_type="txt",
    )
    assert first["status"] == "ambiguous"
    second = await bootstrap_narrator_voice(
        db=db, translation_id="job", chunks=chunks,
        target_language="Vietnamese", model_name="editor", llm_client=object(),
        file_type="txt",
    )
    assert second["status"] == "profiled"
    assert db.get_narrator_voice_profiles("job")[0]["self_reference"] == "tôi"
    assert {
        item["boundary_key"] for item in db.get_narrator_bootstrap_attempts("job")
    } == {"v2:completed:2", "v2:completed:12"}


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
