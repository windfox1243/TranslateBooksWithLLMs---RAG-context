"""
Unit tests for DB-backed directed addressing context engine, merge policies, and projections.
"""

import os
import tempfile
import pytest
from src.persistence.database import Database
from src.utils.context_schema import (
    AddressingUpdateDelta,
    AddressingRuleState,
    extract_addressing_deltas_from_text,
)
from src.utils.context_merge_engine import ContextMergeEngine
from src.utils.context_projection import (
    render_addressing_projection,
    render_addressing_markdown,
)


def test_extract_addressing_deltas_from_text():
    llm_output = """
    Below is the translation of the chapter.

    ```json
    {
      "addressing_updates": [
        {
          "speaker": "Nam",
          "addressee": "Lan",
          "self_pronoun": "anh",
          "second_pronoun": "em",
          "vocative": "Lan",
          "register": "intimate",
          "confidence": 0.95,
          "evidence_quote": "Anh yêu em nhiều lắm."
        },
        {
          "speaker": "Lan",
          "addressee": "Mọi người",
          "self_pronoun": "tôi",
          "second_pronoun": "các bạn"
        }
      ]
    }
    ```
    """
    deltas = extract_addressing_deltas_from_text(llm_output)
    assert len(deltas) == 1
    d = deltas[0]
    assert d.speaker == "Nam"
    assert d.addressee == "Lan"
    assert d.self_pronoun == "anh"
    assert d.second_pronoun == "em"
    assert d.register == "intimate"
    assert d.confidence == 0.95


def test_database_addressing_and_audit_logs():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_jobs.db")
        db = Database(db_path=db_path)
        tx_id = "test_tx_001"

        try:
            # Test Upsert
            success = db.upsert_addressing_rule(
                translation_id=tx_id,
                speaker_name="Nam",
                addressee_name="Lan",
                self_pronoun="anh",
                target_pronoun="em",
                vocative="em yêu",
                register="intimate",
                confidence=0.9,
                chunk_index=1,
            )
            assert success is True

            rules = db.get_addressing_rules(tx_id)
            assert len(rules) == 1
            assert rules[0]["speaker_name"] == "Nam"
            assert rules[0]["addressee_name"] == "Lan"
            assert rules[0]["self_pronoun"] == "anh"
            assert rules[0]["target_pronoun"] == "em"
            assert rules[0]["is_locked"] == 0

            # Test Lock
            db.set_addressing_rule_lock(tx_id, "Nam", "Lan", True)
            rules = db.get_addressing_rules(tx_id)
            assert rules[0]["is_locked"] == 1

            # Test Audit Log
            db.add_context_audit_log(
                translation_id=tx_id,
                chunk_index=1,
                speaker_name="Nam",
                addressee_name="Lan",
                old_state=None,
                new_state={"self_pronoun": "anh", "target_pronoun": "em"},
                trigger_source="unit_test",
                evidence_quote="Anh yêu em",
                confidence=0.9,
            )
            logs = db.get_context_audit_logs(tx_id)
            assert len(logs) == 1
            assert logs[0]["speaker_name"] == "Nam"
            assert logs[0]["trigger_source"] == "unit_test"
        finally:
            db.close()


def test_merge_engine_policies():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_merge.db")
        db = Database(db_path=db_path)
        engine = ContextMergeEngine(db=db, confidence_threshold=0.80)
        tx_id = "tx_merge_001"

        try:
            # 1. High confidence delta should apply
            delta1 = AddressingUpdateDelta(
                speaker="Sếp",
                addressee="Cậu",
                self_pronoun="tôi",
                second_pronoun="cậu",
                register="polite",
                confidence=0.9,
            )
            applied = engine.apply_delta(tx_id, chunk_index=0, delta=delta1)
            assert applied is True

            # 2. Low confidence delta should be rejected
            delta_low = AddressingUpdateDelta(
                speaker="Sếp",
                addressee="Cậu",
                self_pronoun="ta",
                second_pronoun="ngươi",
                register="vulgar",
                confidence=0.5,
            )
            applied_low = engine.apply_delta(tx_id, chunk_index=1, delta=delta_low)
            assert applied_low is False

            # 3. User Lock Priority
            db.set_addressing_rule_lock(tx_id, "Sếp", "Cậu", True)
            delta_locked = AddressingUpdateDelta(
                speaker="Sếp",
                addressee="Cậu",
                self_pronoun="anh",
                second_pronoun="em",
                confidence=0.99,
            )
            applied_locked = engine.apply_delta(tx_id, chunk_index=2, delta=delta_locked)
            assert applied_locked is False

            # 4. Roleplay Rejection Priority
            db.set_addressing_rule_lock(tx_id, "Sếp", "Cậu", False)
            delta_roleplay = AddressingUpdateDelta(
                speaker="Sếp",
                addressee="Cậu",
                self_pronoun="tôi",
                second_pronoun="ngài",
                vocative="Master",
                register="butler café roleplay",
                confidence=0.95,
                evidence_quote="Welcome home, Master!",
            )
            applied_roleplay = engine.apply_delta(tx_id, chunk_index=3, delta=delta_roleplay)
            assert applied_roleplay is False
        finally:
            db.close()


def test_context_projection():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_proj.db")
        db = Database(db_path=db_path)
        tx_id = "tx_proj_001"

        try:
            db.upsert_addressing_rule(
                translation_id=tx_id,
                speaker_name="Hoàng",
                addressee_name="Mai",
                self_pronoun="anh",
                target_pronoun="em",
                vocative="Mai",
                register="intimate",
            )

            prompt_str = render_addressing_projection(tx_id, db=db)
            assert "DIRECTED ADDRESSING RULES" in prompt_str
            assert "**Hoàng** addressing **Mai**" in prompt_str
            assert "Self-reference as 'anh'" in prompt_str
            assert "address target as 'em'" in prompt_str

            markdown_str = render_addressing_markdown(tx_id, db=db)
            assert "| Hoàng | Mai | anh | em |" in markdown_str
        finally:
            db.close()


def test_context_projection_active_filter_does_not_match_partial_latin_names():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_proj_partial.db")
        db = Database(db_path=db_path)
        tx_id = "tx_proj_partial_001"

        try:
            db.upsert_addressing_rule(
                translation_id=tx_id,
                speaker_name="Apollo Rainbow",
                addressee_name="Tomio Momozawa",
                self_pronoun="tôi",
                target_pronoun="em",
                vocative="Tomio",
                register="mentor",
            )

            inactive_projection = render_addressing_projection(
                tx_id,
                db=db,
                active_character_names=["Tom"],
            )
            assert inactive_projection == ""

            active_projection = render_addressing_projection(
                tx_id,
                db=db,
                active_character_names=["Tomio"],
            )
            assert "**Apollo Rainbow** addressing **Tomio Momozawa**" in active_projection
        finally:
            db.close()


def test_convert_addressing_text_to_markdown_table():
    from src.utils.context_projection import convert_addressing_text_to_markdown_table
    raw_text = """
    - Apollo Rainbow → Tomio Momozawa: "Trainer" | "self-reference: tôi; second-person pronoun: Trainer; vocative/address form: Trainer" | professional mentor/trainee relationship
    - Tomio Momozawa → Apollo Rainbow: "Apollo" | "self-reference: tôi; second-person pronoun: em; vocative/address form: Apollo" | professional mentor/trainee relationship
    """
    table = convert_addressing_text_to_markdown_table(raw_text)
    assert "| Speaker | Addressee | Tự xưng (Self) | Gọi đối phương (Target) | Danh xưng (Vocative) |" in table
    assert "| Apollo Rainbow | Tomio Momozawa | tôi | Trainer | Trainer |" in table
    assert "| Tomio Momozawa | Apollo Rainbow | tôi | em | Apollo |" in table


def test_db_addressing_markdown_import_export_projection_and_locks():
    from src.utils.db_addressing import (
        build_directed_addressing_prompt_context,
        export_db_addressing_to_markdown,
        sync_markdown_addressing_to_db,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db_bridge.db")
        db = Database(db_path=db_path)
        tx_id = "tx_db_bridge_001"

        try:
            markdown_context = """
            === DYNAMIC STATE ===
            ## CURRENT ADDRESSING FORMS
            - Alice → Bob: source form "Bob" | recommended target-language form "self-reference: tôi; second-person pronoun: em; vocative/address form: Bob" | close friends
            """
            applied = sync_markdown_addressing_to_db(
                translation_id=tx_id,
                db=db,
                context_or_dynamic_state=markdown_context,
                target_language="Vietnamese",
                chunk_index=3,
            )

            assert applied == 1
            projection = build_directed_addressing_prompt_context(
                translation_id=tx_id,
                db=db,
                target_language="Vietnamese",
                active_character_names=["Alice", "Bob"],
            )
            assert "**Alice** addressing **Bob**" in projection
            assert "Self-reference as 'tôi'" in projection
            assert "address target as 'em'" in projection

            exported = export_db_addressing_to_markdown(tx_id, db)
            assert "Alice → Bob" in exported
            assert "second-person pronoun: em" in exported

            db.set_addressing_rule_lock(tx_id, "Alice", "Bob", True)
            changed_markdown = """
            ## CURRENT ADDRESSING FORMS
            - Alice → Bob: source form "Bob" | recommended target-language form "self-reference: tao; second-person pronoun: mày; vocative/address form: Bob" | hostile shift
            """
            locked_applied = sync_markdown_addressing_to_db(
                translation_id=tx_id,
                db=db,
                context_or_dynamic_state=changed_markdown,
                target_language="Vietnamese",
                chunk_index=4,
            )

            assert locked_applied == 0
            rules = db.get_addressing_rules(tx_id)
            assert rules[0]["self_pronoun"] == "tôi"
            assert rules[0]["target_pronoun"] == "em"
        finally:
            db.close()


def test_unknown_language_addressing_profile_is_neutral():
    from src.utils.language_profiles import get_language_profile
    from src.utils.universal_addressing_engine import UniversalAddressingEngine

    profile = get_language_profile("Custom Story Language")
    engine = UniversalAddressingEngine(language="Custom Story Language")

    assert profile.neutral_fallback is True
    assert engine.lang_code == "generic"
    assert engine.validate_and_repair_pair(
        "Trainer",
        "Trainer",
        speaker="Alice",
        addressee="Bob",
        details_context="trainer to trainee",
    ) == ("Trainer", "Trainer", "")
