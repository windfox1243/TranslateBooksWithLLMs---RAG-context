"""Regressions for evidence-backed directed-addressing contract v2."""

import json

import pytest

from src.persistence.database import Database
from src.utils.addressing_schema import AddressingCandidateV2, parse_addressing_candidate_block
from src.utils.context_merge_engine import ContextMergeEngine, _source_form_support_reason
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


def test_v4_import_is_idempotent_and_drops_ungrounded_social_basis(addressing_db):
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
    assert rule["social_basis"] == []
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


@pytest.mark.parametrize(
    ("source_text", "evidence"),
    [
        ('She said, “Thanks, Guri-ko!”', 'She said, "Thanks, Guri-ko!"'),
        ('She paused… then said, “Apollo.”', 'She paused... then said, "Apollo."'),
    ],
)
def test_source_evidence_accepts_typographic_quote_variants(source_text, evidence):
    assert _source_form_support_reason(
        "Guri-ko" if "Guri" in source_text else "Apollo",
        source_text,
        evidence,
        "English",
    ) == ""


def test_source_evidence_rejects_fabricated_dialogue_boundaries():
    source = "She wondered whether Apollo would answer."
    assert _source_form_support_reason(
        "Apollo",
        source,
        '"She wondered whether Apollo would answer."',
        "English",
    ) == "evidence_quote_mismatch"


def test_v4_keeps_partial_suzuka_vocative_rule_provisional(addressing_db):
    source = (
        '"Suzuka-san, try this dessert! It’s so good! Wanna bite?" '
        'Special Week was walking with her senpai.'
    )
    candidate = _candidate(
        speaker="Special Week",
        addressee="Silence Suzuka",
        target_form={
            "self_reference": "tớ",
            "second_person": "Suzuka-san",
            "vocative": "Suzuka-san",
        },
        source_forms=[{
            "text": "Suzuka-san",
            "usage": "direct_address",
            "evidence_quote": "Suzuka-san, try this dessert!",
        }],
        register="polite",
        social_basis=["peer"],
        evidence_quote="Suzuka-san, try this dessert! It’s so good! Wanna bite?",
        confidence=1.0,
    )
    assert candidate is not None

    applied = ContextMergeEngine(db=addressing_db).apply_delta(
        "suzuka", 2, candidate.to_delta(),
        trigger_source="context_update_v2",
        target_language="Vietnamese",
        known_character_names=["Special Week", "Silence Suzuka"],
        source_text=source,
        source_language="English",
    )

    assert not applied
    assert addressing_db.get_addressing_rules("suzuka") == []
    provisional = addressing_db.get_addressing_rules("suzuka", "provisional")
    assert len(provisional) == 1
    assert provisional[0]["validation_reason"] == (
        "unsupported_vietnamese_target_pair"
    )
    assert provisional[0]["confidence"] == pytest.approx(0.79)
    assert provisional[0]["source_forms"] == ["Suzuka-san"]
    assert provisional[0]["social_basis"] == []
    assert export_db_addressing_to_markdown("suzuka", addressing_db).strip() == ""


def test_v4_promotes_peer_pair_after_two_distinct_units(addressing_db):
    engine = ContextMergeEngine(db=addressing_db)
    first = _candidate(
        target_form={
            "self_reference": "tớ",
            "second_person": "cậu",
            "vocative": "Bob",
        },
        source_forms=[{
            "text": "Bob",
            "usage": "direct_address",
            "evidence_quote": "Bob, want some tea?",
        }],
        social_basis=["peer"],
        evidence_quote="Bob, want some tea?",
    )
    second = _candidate(
        target_form={
            "self_reference": "tớ",
            "second_person": "cậu",
            "vocative": "Bob",
        },
        source_forms=[{
            "text": "Bob",
            "usage": "direct_address",
            "evidence_quote": "Bob, let's go together.",
        }],
        social_basis=["peer"],
        evidence_quote="Bob, let's go together.",
    )
    kwargs = {
        "trigger_source": "context_update_v2",
        "target_language": "Vietnamese",
        "known_character_names": ["Alice", "Bob"],
        "source_language": "English",
    }
    assert not engine.apply_delta(
        "peer", 1, first.to_delta(),
        source_text="Bob, want some tea?", **kwargs,
    )
    assert addressing_db.get_addressing_rules("peer") == []
    assert engine.apply_delta(
        "peer", 3, second.to_delta(),
        source_text="Bob, let's go together.", **kwargs,
    )
    active = addressing_db.get_addressing_rules("peer")
    assert len(active) == 1
    assert active[0]["validation_status"] == "active"
    assert {item["chunk_index"] for item in active[0]["evidence"]} == {1, 3}


def test_existing_v3_copied_vocative_is_migrated_to_provisional(tmp_path):
    path = str(tmp_path / "copied-vocative.db")
    db = Database(path)
    assert db.create_job(
        "legacy-suzuka", "txt", {"target_language": "Vietnamese"}
    )
    assert db.upsert_addressing_rule(
        "legacy-suzuka", "Special Week", "Silence Suzuka",
        "tớ", "Suzuka-san", vocative="Suzuka-san",
        social_basis=["peer"], contract_version=3,
        confidence=1.0, provenance="llm_context",
    )
    assert db.add_addressing_evidence(
        "legacy-suzuka", "Special Week", "Silence Suzuka", "Suzuka-san",
        evidence_quote="Suzuka-san, try this dessert!",
        provenance="llm_context", chunk_index=2,
    )
    db.close()

    reopened = Database(path)
    assert reopened.get_addressing_rules("legacy-suzuka") == []
    provisional = reopened.get_addressing_rules(
        "legacy-suzuka", "provisional"
    )
    assert len(provisional) == 1
    assert provisional[0]["validation_reason"] == (
        "unsupported_vietnamese_target_pair"
    )
    assert provisional[0]["social_basis"] == []
    reopened.close()
