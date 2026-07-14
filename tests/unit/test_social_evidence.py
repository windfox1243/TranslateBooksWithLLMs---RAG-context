import sqlite3

from src.core.context.social_evidence import (
    apply_social_hierarchy_evidence,
    extract_social_hierarchy_evidence,
)
from src.core.context.reconciliation import ContextReconciler
from src.core.context.unit_pipeline import (
    commit_source_social_evidence,
    relevant_character_names,
)
from src.persistence.database import Database
from src.utils.db_addressing import build_directed_addressing_prompt_context
from src.utils.relationship_reasoning_engine import RelationshipReasoningEngine
from src.utils.relationship_schema import RelationshipCandidate


def _dialogue(cue, speaker="Special Week", addressee="Silence Suzuka"):
    return {
        "turns": [{
            "id": "dlg-senior",
            "cue": cue,
            "speaker": speaker,
            "addressee": addressee,
            "confidence": 1.0,
        }],
        "state_after": {"speaker": speaker, "addressee": addressee},
    }


def test_bounded_senpai_scene_generalizes_suzuka_hierarchy():
    cue = '"Suzuka-san, try this dessert! It’s so good! Wanna bite?"'
    source = (
        cue
        + "\nThat lively voice—Special Week—was beside Silence Suzuka.\n"
        + "Spe-chan was casually snacking with her senpai."
    )
    evidence = extract_social_hierarchy_evidence(
        source, _dialogue(cue), source_language="English",
    )
    assert len(evidence) == 1
    assert evidence[0].source == "Special Week"
    assert evidence[0].target == "Silence Suzuka"
    assert evidence[0].hierarchy == "source_junior"
    assert evidence[0].rank_relation == "source_lower"
    assert evidence[0].relative_age == "unknown"
    assert evidence[0].source_form == "Suzuka-san"


def test_bounded_scene_does_not_invent_reverse_hierarchy():
    first = '"Suzuka-san, try this dessert!"'
    reply = '"No thank you, I am watching my weight."'
    source = (
        first + reply
        + "\nThat lively voice—Special Week—was beside Silence Suzuka.\n"
        + "Spe-chan was casually snacking with her senpai."
    )
    dialogue = {
        "turns": [
            {**_dialogue(first)["turns"][0]},
            {
                "id": "reply",
                "cue": reply,
                "speaker": "Silence Suzuka",
                "addressee": "Special Week",
                "confidence": 1.0,
            },
        ]
    }
    evidence = extract_social_hierarchy_evidence(
        source, dialogue, source_language="English",
    )
    assert [(item.source, item.target) for item in evidence] == [
        ("Special Week", "Silence Suzuka")
    ]


def test_neutral_san_without_independent_seniority_is_not_hierarchy():
    cue = '"Suzuka-san, would you like dessert?"'
    source = cue + "\nSpecial Week spoke to Silence Suzuka."
    assert extract_social_hierarchy_evidence(
        source, _dialogue(cue), source_language="English",
    ) == []


def test_direct_senpai_forms_share_one_normalized_rule():
    cases = [
        '"Aster-senpai, wait!"',
        '"Aster-sunbae, wait!"',
        '"Aster-先輩、待って！"',
        '"Aster-선배님, 기다려요!"',
        '"Aster師姐，等等！"',
    ]
    for cue in cases:
        source = f"{cue}\nEllen called to Aster."
        evidence = extract_social_hierarchy_evidence(
            source,
            _dialogue(cue, speaker="Ellen", addressee="Aster"),
            source_language="",
        )
        assert len(evidence) == 1
        assert evidence[0].hierarchy == "source_junior"
        assert evidence[0].relative_age == "unknown"


def test_explicit_kin_age_is_separate_from_social_hierarchy():
    cue = '"Older sister Aster, wait!"'
    evidence = extract_social_hierarchy_evidence(
        cue,
        _dialogue(cue, speaker="Ellen", addressee="Aster"),
        source_language="English",
    )
    assert evidence[0].hierarchy == "source_junior"
    assert evidence[0].relative_age == "source_younger"


def test_social_evidence_persists_relationship_and_safe_vietnamese_pair(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    tx = "social-evidence"
    db.upsert_relationship_node(tx, "Special Week", "special week", gender="female")
    db.upsert_relationship_node(tx, "Silence Suzuka", "silence suzuka", gender="female")
    cue = '"Suzuka-san, try this dessert! It’s so good! Wanna bite?"'
    source = (
        cue
        + "\nThat lively voice—Special Week—was beside Silence Suzuka.\n"
        + "Spe-chan was casually snacking with her senpai."
    )

    applied = apply_social_hierarchy_evidence(
        translation_id=tx,
        db=db,
        source_text=source,
        dialogue_attribution=_dialogue(cue),
        source_language="English",
        target_language="Vietnamese",
        chunk_index=2,
        known_character_names=["Special Week", "Silence Suzuka"],
    )

    assert len(applied) == 1
    edges = db.get_relationship_edges(tx, statuses=["accepted"])
    assert len(edges) == 1
    assert edges[0]["hierarchy"] == "source_junior"
    assert edges[0]["relative_age"] == "unknown"
    rules = db.get_addressing_rules(tx)
    assert len(rules) == 1
    assert rules[0]["self_pronoun"] == "em"
    assert rules[0]["target_pronoun"] == "chị"
    assert rules[0]["vocative"] == "Suzuka-san"
    assert db.get_addressing_rules(tx, "provisional") == []

    # Replaying the same evidence reinforces the same materialized facts.
    apply_social_hierarchy_evidence(
        translation_id=tx,
        db=db,
        source_text=source,
        dialogue_attribution=_dialogue(cue),
        source_language="English",
        target_language="Vietnamese",
        chunk_index=3,
        known_character_names=["Special Week", "Silence Suzuka"],
    )
    assert len(db.get_relationship_edges(tx, statuses=["accepted"])) == 1
    assert len(db.get_addressing_rules(tx)) == 1
    addressing_evidence = db.get_addressing_evidence(
        tx, "Special Week", "Silence Suzuka",
    )
    assert len(addressing_evidence) == 1
    assert addressing_evidence[0]["observation_count"] == 2
    assert addressing_evidence[0]["fingerprint"]


def test_database_exposes_narrow_compatibility_repositories(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    assert db.context.get_addressing_rules("missing") == []
    assert callable(db.editor.create_editor_run)
    assert callable(db.jobs.create_job)
    assert callable(db.narrator.quarantine_narrator_voice_after)


def test_v4_evidence_tables_migrate_additively(tmp_path):
    path = tmp_path / "v4.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE context_addressing_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            translation_id TEXT NOT NULL,
            speaker_name TEXT NOT NULL,
            addressee_name TEXT NOT NULL,
            source_form TEXT NOT NULL,
            usage TEXT NOT NULL DEFAULT 'direct_address',
            source_language TEXT NOT NULL DEFAULT '',
            evidence_quote TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL DEFAULT 'durable',
            confidence REAL NOT NULL DEFAULT 0.5,
            provenance TEXT NOT NULL DEFAULT 'unknown',
            dialogue_turn_id TEXT,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (translation_id, speaker_name, addressee_name,
                    source_form, usage, scope, evidence_quote)
        );
        CREATE TABLE context_relationship_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            translation_id TEXT NOT NULL,
            edge_id INTEGER,
            chunk_index INTEGER NOT NULL,
            file_id TEXT NOT NULL DEFAULT '',
            dialogue_turn_id TEXT NOT NULL DEFAULT '',
            evidence_quote TEXT NOT NULL,
            provenance TEXT NOT NULL,
            parser_status TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.0,
            match_kind TEXT NOT NULL DEFAULT '',
            source_start INTEGER,
            source_end INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO context_addressing_evidence (
            translation_id, speaker_name, addressee_name, source_form,
            evidence_quote
        ) VALUES ('legacy', 'Ellen', 'Aster', 'Aster-senpai', 'Aster-senpai');
        INSERT INTO context_relationship_evidence (
            translation_id, chunk_index, evidence_quote, provenance,
            parser_status
        ) VALUES ('legacy', 1, 'Aster is her senior.', 'source', 'valid');
    """)
    conn.commit()
    conn.close()

    db = Database(str(path))
    addressing = db.get_addressing_evidence("legacy")
    relationships = db.get_relationship_evidence("legacy")
    assert addressing[0]["fingerprint"]
    assert addressing[0]["observation_count"] == 1
    assert addressing[0]["resolution_status"] == "open"
    assert addressing[0]["last_seen_at"]
    assert relationships[0]["fingerprint"]
    assert relationships[0]["last_seen_at"]


def test_retrieval_includes_all_dialogue_turns_and_source_mentions():
    lore = (
        "## CHARACTERS & GENDERS\n"
        "- Aster: Female, team captain.\n"
        "- Ellen: Female, first-year student.\n"
        "- Mara: Female, coach.\n"
        "- Rowan: Male, observer.\n"
        "- Unrelated: Male, absent from this scene.\n\n"
        "## GLOSSARY & TERMINOLOGY\n"
    )
    dialogue = {
        "turns": [
            {"speaker": "Ellen", "addressee": "Aster"},
            {"speaker": "Aster", "addressee": "Mara"},
        ]
    }
    names = relevant_character_names(
        global_lore=lore,
        source_text="Ellen answered Aster while Mara and Rowan watched.",
        dialogue_attribution=dialogue,
        source_language="English",
    )
    assert names == ["Ellen", "Aster", "Mara", "Rowan"]


def test_provisional_evidence_is_retained_but_excluded_from_prompts(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    tx = "unsupported"
    db.upsert_addressing_rule(
        tx, "Ellen", "Aster", "tớ", "bạn",
        contract_version=5,
        validation_status="provisional",
        validation_reason="unsupported_target_pair",
    )
    db.add_addressing_evidence(
        tx, "Ellen", "Aster", "Aster-san",
        evidence_quote='"Aster-san?"',
    )
    assert db.get_addressing_evidence(tx)
    assert build_directed_addressing_prompt_context(
        translation_id=tx,
        db=db,
        target_language="Vietnamese",
        active_character_names=["Ellen", "Aster"],
    ) == ""


def test_current_unit_lore_gender_enables_pre_draft_addressing(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    cue = '"Aster-senpai, wait!"'
    evidence = commit_source_social_evidence(
        translation_id="same-unit",
        db=db,
        global_lore=(
            "## CHARACTERS & GENDERS\n"
            "- Ellen: Female, first-year student.\n"
            "- Aster: Female, team captain.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        ),
        source_text=f"{cue}\nEllen called to Aster.",
        dialogue_attribution=_dialogue(
            cue, speaker="Ellen", addressee="Aster",
        ),
        source_language="English",
        target_language="Vietnamese",
        chunk_index=1,
    )
    assert len(evidence) == 1
    rule = db.get_addressing_rules("same-unit")[0]
    assert (rule["self_pronoun"], rule["target_pronoun"]) == ("em", "chị")


def test_provisional_observation_promotes_when_relationship_arrives(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    tx = "reconcile"
    db.upsert_relationship_node(tx, "Ellen", "ellen", gender="female")
    db.upsert_relationship_node(tx, "Aster", "aster", gender="female")
    db.upsert_addressing_rule(
        tx, "Ellen", "Aster", "tớ", "Aster-senpai",
        vocative="Aster-senpai",
        contract_version=5,
        validation_status="provisional",
        validation_reason="unsupported_vietnamese_target_pair",
    )
    db.add_addressing_evidence(
        tx, "Ellen", "Aster", "Aster-senpai",
        evidence_quote='"Aster-senpai, wait!"',
        chunk_index=1,
    )
    engine = RelationshipReasoningEngine(db=db)
    decision = engine.merge_candidate(
        tx,
        1,
        RelationshipCandidate(
            source="Ellen",
            target="Aster",
            relationship_type="associated",
            direction="directed",
            hierarchy="source_junior",
            rank_relation="source_lower",
            relative_age="unknown",
            evidence_quote='"Aster-senpai, wait!"',
            confidence=1.0,
            provenance="manual_context",
        ),
        source_text='"Aster-senpai, wait!"',
        known_character_names=["Ellen", "Aster"],
    )
    assert decision.status == "accepted"

    assert ContextReconciler(db).reconcile_pair(
        tx, "Ellen", "Aster", target_language="Vietnamese", chunk_index=1,
    )
    rule = db.get_addressing_rules(tx)[0]
    assert (rule["self_pronoun"], rule["target_pronoun"]) == ("em", "chị")
    evidence = db.get_addressing_evidence(tx, "Ellen", "Aster")
    assert evidence[0]["resolution_status"] == "promoted"
