"""Regressions for evidence-backed directed-addressing contract v2."""

import json

import pytest

from src.persistence.database import Database
from src.utils.addressing_schema import AddressingCandidateV2, parse_addressing_candidate_block
from src.utils.context_merge_engine import ContextMergeEngine
from src.utils.db_addressing import export_db_addressing_to_markdown


@pytest.fixture
def addressing_db(tmp_path):
    db = Database(str(tmp_path / "addressing-v2.db"))
    try:
        yield db
    finally:
        db.close()


def _candidate(**overrides):
    values = {
        "speaker": "Alice",
        "addressee": "Bob",
        "target_form": {
            "self_reference": "em",
            "second_person": "anh",
            "vocative": "anh",
        },
        "source_forms": [{
            "text": "Brother",
            "usage": "direct_address",
            "evidence_quote": "Alice called Bob Brother.",
        }],
        "register": "familial",
        "social_basis": ["Bob is older than Alice"],
        "scope": "durable",
        "evidence_quote": "Alice called Bob Brother.",
        "confidence": 0.96,
    }
    values.update(overrides)
    return AddressingCandidateV2.from_dict(values, source_language="English")


def test_v2_parser_separates_register_and_social_basis():
    candidates, status = parse_addressing_candidate_block(json.dumps({
        "updates": [{
            "speaker": "Alice",
            "addressee": "Bob",
            "target_form": {"self_reference": "em", "second_person": "anh"},
            "source_forms": ["Brother"],
            "register": "Bob is Alice's elder brother",
            "social_basis": ["older sibling"],
            "confidence": 0.9,
        }],
    }), source_language="English")
    assert status == "json"
    assert candidates[0].register == "neutral"
    assert candidates[0].social_basis == ["older sibling"]
    assert candidates[0].source_forms[0].text == "Brother"


def test_v2_rejects_missing_source_forms_and_untranslated_vocatives(addressing_db):
    engine = ContextMergeEngine(db=addressing_db)
    missing = _candidate(source_forms=[])
    assert missing is not None
    assert not engine.apply_delta(
        "missing",
        0,
        missing.to_delta(),
        trigger_source="context_update_v2",
        target_language="Vietnamese",
        known_character_names=["Alice", "Bob"],
        source_text="Alice spoke to Bob.",
        source_language="English",
    )
    untranslated = _candidate(target_form={
        "self_reference": "em",
        "second_person": "anh",
        "vocative": "Brother",
    })
    assert not engine.apply_delta(
        "untranslated",
        0,
        untranslated.to_delta(),
        trigger_source="context_update_v2",
        target_language="Vietnamese",
        known_character_names=["Alice", "Bob"],
        source_text="Alice called Bob Brother.",
        source_language="English",
    )
    reasons = [
        item["new_state"]["reason"]
        for job in ("missing", "untranslated")
        for item in addressing_db.get_context_audit_logs(job)
    ]
    assert any("requires at least one exact source form" in reason for reason in reasons)
    assert any("untranslated generic" in reason for reason in reasons)


def test_v3_rejects_indirect_reference_as_directed_addressing(addressing_db):
    candidate = _candidate(source_forms=[{
        "text": "Trainer",
        "usage": "indirect_reference",
        "evidence_quote": "She wanted to find a Trainer.",
    }])
    assert candidate is not None
    applied = ContextMergeEngine(db=addressing_db).apply_delta(
        "indirect", 0, candidate.to_delta(),
        trigger_source="context_update_v2",
        target_language="Vietnamese",
        known_character_names=["Alice", "Bob"],
        source_text="She wanted to find a Trainer.",
        source_language="English",
    )
    assert not applied
    reason = addressing_db.get_context_audit_logs("indirect")[0]["new_state"]["reason"]
    assert "exact spoken direct-address" in reason


def test_existing_indirect_llm_rule_is_quarantined_on_reopen(tmp_path):
    path = str(tmp_path / "migration.db")
    db = Database(path)
    assert db.upsert_addressing_rule(
        "job", "Protagonist", "Trainer", "em", "trainer",
        contract_version=2, provenance="llm_context",
    )
    assert db.add_addressing_evidence(
        "job", "Protagonist", "Trainer", "Trainer",
        usage="indirect_reference",
        evidence_quote="She wanted to find a Trainer.",
        provenance="llm_context",
    )
    db.close()

    reopened = Database(path)
    assert reopened.get_addressing_rules("job") == []
    quarantined = reopened.get_addressing_rules("job", "quarantined")
    assert len(quarantined) == 1
    assert quarantined[0]["validation_reason"] == "missing_direct_dialogue_evidence"
    reopened.close()


def test_v2_import_is_idempotent_and_preserves_source_forms(addressing_db):
    engine = ContextMergeEngine(db=addressing_db)
    candidate = _candidate()
    kwargs = {
        "trigger_source": "context_update_v2",
        "target_language": "Vietnamese",
        "known_character_names": ["Alice", "Bob"],
        "source_text": "Alice called Bob Brother.",
        "source_language": "English",
    }
    assert engine.apply_delta("stable", 0, candidate.to_delta(), **kwargs)
    assert not engine.apply_delta("stable", 1, candidate.to_delta(), **kwargs)
    assert len(addressing_db.get_context_audit_logs("stable")) == 1
    assert len(addressing_db.get_addressing_evidence("stable")) == 1
    rule = addressing_db.get_addressing_rules("stable")[0]
    assert rule["source_forms"] == ["Brother"]
    assert rule["social_basis"] == ["Bob is older than Alice"]
    exported = export_db_addressing_to_markdown("stable", addressing_db)
    assert 'source form "Brother"' in exported
    assert 'source form ""' not in exported


def test_markdown_export_uses_unknown_when_evidence_is_unavailable(addressing_db):
    addressing_db.upsert_addressing_rule(
        "legacy-empty",
        "Alice",
        "Bob",
        "tôi",
        "bạn",
        confidence=1.0,
    )
    exported = export_db_addressing_to_markdown("legacy-empty", addressing_db)
    assert 'source form "unknown"' in exported
    assert 'source form ""' not in exported


def test_context_state_transaction_rolls_back_all_staged_changes(addressing_db):
    with pytest.raises(RuntimeError):
        with addressing_db.context_state_transaction():
            addressing_db.upsert_addressing_rule(
                "rollback",
                "Alice",
                "Bob",
                "em",
                "anh",
                contract_version=2,
                confidence=0.99,
            )
            addressing_db.add_addressing_evidence(
                "rollback",
                "Alice",
                "Bob",
                "Brother",
            )
            raise RuntimeError("simulated failed chunk")
    assert addressing_db.get_addressing_rules("rollback") == []
    assert addressing_db.get_addressing_evidence("rollback") == []
