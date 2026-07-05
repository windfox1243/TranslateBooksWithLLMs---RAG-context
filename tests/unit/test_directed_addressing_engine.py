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
            assert "**Hoàng** khi nói với **Mai**" in prompt_str
            assert "Tự xưng là 'anh'" in prompt_str
            assert "gọi đối phương là 'em'" in prompt_str

            markdown_str = render_addressing_markdown(tx_id, db=db)
            assert "| Hoàng | Mai | anh | em |" in markdown_str
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


