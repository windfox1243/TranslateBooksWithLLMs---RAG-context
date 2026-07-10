"""Regression tests for the structured relationship reasoning engine."""

import json
from types import SimpleNamespace

import pytest
from flask import Flask

from src.persistence.database import Database
from src.api.blueprints.translation_routes import (
    _prompt_options_from_start_request,
    create_translation_blueprint,
)
from src.prompts.prompts import generate_translation_prompt
from src.utils.context_merge_engine import ContextMergeEngine
from src.utils.context_schema import AddressingUpdateDelta
from src.utils.relationship_projection import build_relationship_projection
from src.utils.relationship_reasoning_engine import (
    RelationshipReasoningEngine,
    judge_relationship_candidate,
    relationship_support_for_addressing,
)
from src.utils.relationship_schema import (
    RelationshipCandidate,
    parse_relationship_candidate_block,
)
from src.utils.relationship_sync import (
    apply_relationship_graph_to_context,
    build_relationship_prompt_context,
    export_relationship_graph_to_markdown,
    quarantine_incompatible_addressing_rules,
    resolve_relationship_reasoning_mode,
    sync_context_update_relationships_to_db,
    judge_ambiguous_relationship_candidates,
    sync_markdown_relationships_to_db,
)
from src.utils.novel_context import update_novel_context_chunk


@pytest.fixture
def relationship_db(tmp_path):
    db = Database(str(tmp_path / "relationships.db"))
    try:
        yield db
    finally:
        db.close()


def _engine_with_characters(db, translation_id="rel-test", *names):
    engine = RelationshipReasoningEngine(db=db)
    for name in names:
        engine.register_node(translation_id, name)
    return engine


def _manual_candidate(source, target, relationship_type, **overrides):
    values = {
        "source": source,
        "target": target,
        "relationship_type": relationship_type,
        "direction": "directed",
        "scope": "durable",
        "confidence": 1.0,
        "provenance": "manual",
        "details": relationship_type,
    }
    values.update(overrides)
    return RelationshipCandidate(**values)


def test_relationship_schema_migration_is_idempotent(tmp_path):
    path = str(tmp_path / "idempotent.db")
    first = Database(path)
    first.close()
    second = Database(path)
    try:
        tables = {
            row[0]
            for row in second._get_connection().execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "context_relationship_nodes",
            "context_relationship_edges",
            "context_relationship_evidence",
            "context_relationship_conflicts",
        } <= tables
    finally:
        second.close()


def test_project_is_default_and_shadow_never_projects(relationship_db):
    tx = "modes"
    engine = _engine_with_characters(relationship_db, tx, "A", "B")
    decision = engine.merge_candidate(
        tx,
        0,
        _manual_candidate("A", "B", "mentor"),
    )
    assert decision.status == "accepted"
    assert resolve_relationship_reasoning_mode({}) == "project"
    assert build_relationship_prompt_context(
        translation_id=tx,
        db=relationship_db,
        prompt_options={},
        reference_text="A spoke to B.",
    )
    assert build_relationship_prompt_context(
        translation_id=tx,
        db=relationship_db,
        prompt_options={"use_relationship_reasoning": "shadow"},
        reference_text="A spoke to B.",
    ) == ""


def test_normal_web_start_defaults_to_project_and_preserves_explicit_shadow():
    assert _prompt_options_from_start_request({})[
        "use_relationship_reasoning"
    ] == "project"
    assert _prompt_options_from_start_request({
        "prompt_options": {"use_relationship_reasoning": "shadow"},
    })["use_relationship_reasoning"] == "shadow"


def test_source_supported_relationship_is_accepted(relationship_db):
    tx = "evidence"
    engine = _engine_with_characters(relationship_db, tx, "Quinier", "Frondier")
    source = "Quinier told Frondier, his student, to begin the lesson."
    candidate = RelationshipCandidate(
        source="Quinier",
        target="Frondier",
        relationship_type="mentor",
        direction="directed",
        evidence_quote=source,
        confidence=0.94,
        provenance="llm_context",
    )
    decision = engine.merge_candidate(
        tx,
        11,
        candidate,
        source_text=source,
        known_character_names=["Quinier", "Frondier"],
    )
    assert decision.status == "accepted"
    assert relationship_db.get_relationship_edges(tx, statuses=["accepted"])[0][
        "relationship_type"
    ] == "mentor"


def test_durable_llm_relationship_without_exact_evidence_is_quarantined(relationship_db):
    tx = "missing-evidence"
    engine = _engine_with_characters(relationship_db, tx, "Quinier", "Frondier")
    decision = engine.merge_candidate(
        tx,
        11,
        RelationshipCandidate(
            source="Quinier",
            target="Frondier",
            relationship_type="mentor",
            confidence=0.99,
            provenance="llm_context",
        ),
        source_text="Quinier and Frondier entered the room.",
        known_character_names=["Quinier", "Frondier"],
    )
    assert decision.status == "quarantined"
    assert relationship_db.get_relationship_edges(tx, statuses=["accepted"]) == []
    assert relationship_db.get_relationship_conflicts(tx, status="open")[0][
        "validator"
    ] == "source_evidence"


def test_reverse_asymmetric_relationship_must_use_inverse_type(relationship_db):
    tx = "reverse"
    engine = _engine_with_characters(relationship_db, tx, "Parent", "Child")
    assert engine.merge_candidate(
        tx, 0, _manual_candidate("Parent", "Child", "parent")
    ).status == "accepted"
    rejected = engine.merge_candidate(
        tx, 1, _manual_candidate("Child", "Parent", "parent")
    )
    assert rejected.status == "rejected"
    assert rejected.validator == "reverse_semantics"
    accepted = engine.merge_candidate(
        tx, 1, _manual_candidate("Child", "Parent", "child")
    )
    assert accepted.status == "accepted"


def test_locked_relationship_cannot_be_overwritten_or_deleted(relationship_db):
    tx = "locked"
    engine = _engine_with_characters(relationship_db, tx, "Mentor", "Student")
    decision = engine.merge_candidate(
        tx, 0, _manual_candidate("Mentor", "Student", "mentor")
    )
    assert relationship_db.set_relationship_edge_lock(tx, decision.edge_id, True)
    changed = engine.merge_candidate(
        tx,
        1,
        _manual_candidate(
            "Mentor",
            "Student",
            "mentor",
            register="hostile",
        ),
    )
    assert changed.status == "rejected"
    assert changed.validator == "lock"
    assert not relationship_db.delete_relationship_edge(tx, decision.edge_id)


def test_situational_fact_does_not_replace_durable_baseline(relationship_db):
    tx = "situational"
    engine = _engine_with_characters(relationship_db, tx, "A", "B")
    durable = engine.merge_candidate(
        tx, 0, _manual_candidate("A", "B", "friend", direction="symmetric")
    )
    situational = engine.merge_candidate(
        tx,
        1,
        _manual_candidate(
            "A",
            "B",
            "enemy",
            direction="symmetric",
            scope="situational",
            details="acting as enemies during a play",
        ),
    )
    assert durable.status == "accepted"
    assert situational.status == "accepted"
    edges = relationship_db.get_relationship_edges(tx, statuses=["accepted"])
    assert {(edge["relationship_type"], edge["scope"]) for edge in edges} == {
        ("friend", "durable"),
        ("enemy", "situational"),
    }


def test_gram_weapon_cannot_be_imported_as_character_alias(relationship_db):
    tx = "gram"
    context = """# NOVEL CONTEXT

## CHARACTERS & GENDERS
- Sigurd: Male, warrior.

## CHARACTER ALIASES
- Gram: Sigurd

## NAME TRANSLATION MAP

## GLOSSARY & TERMINOLOGY
- Gram: legendary sword weapon

---DYNAMIC_STATE_START---
# DYNAMIC RELATIONSHIP STATE
## CURRENT ADDRESSING FORMS

## RELATIONSHIP EVOLUTION
---DYNAMIC_STATE_END---
"""
    sync_markdown_relationships_to_db(
        translation_id=tx,
        db=relationship_db,
        context_or_dynamic_state=context,
        trigger_source="job_context_load",
    )
    sigurd = relationship_db.get_relationship_node_by_name(tx, "sigurd")
    assert sigurd is not None
    assert "gram" not in {str(alias).casefold() for alias in sigurd["aliases"]}
    conflicts = relationship_db.get_relationship_conflicts(tx, status="open")
    assert any(conflict["validator"] == "entity_type" for conflict in conflicts)


@pytest.mark.parametrize(
    ("language", "source_name", "target_name", "source_text"),
    [
        ("Chinese", "\u674e\u96f7", "\u97e9\u6885", "\u674e\u96f7\u662f\u97e9\u6885\u7684\u5bfc\u5e08\u3002"),
        ("Korean", "\uc9c0\ubbfc", "\uc218\uc9c4", "\uc9c0\ubbfc\uc740 \uc218\uc9c4\uc758 \uba58\ud1a0\uc774\ub2e4."),
        ("Arabic", "\u0639\u0644\u064a", "\u0645\u0631\u064a\u0645", "\u0639\u0644\u064a \u0647\u0648 \u0645\u0639\u0644\u0645 \u0645\u0631\u064a\u0645."),
        ("Thai", "\u0e2a\u0e21\u0e0a\u0e32\u0e22", "\u0e2a\u0e21\u0e2b\u0e0d\u0e34\u0e07", "\u0e2a\u0e21\u0e0a\u0e32\u0e22\u0e40\u0e1b\u0e47\u0e19\u0e04\u0e23\u0e39\u0e02\u0e2d\u0e07\u0e2a\u0e21\u0e2b\u0e0d\u0e34\u0e07"),
    ],
)
def test_non_latin_relationship_evidence_uses_language_neutral_matching(
    relationship_db,
    language,
    source_name,
    target_name,
    source_text,
):
    tx = f"script-{language}"
    engine = _engine_with_characters(relationship_db, tx, source_name, target_name)
    decision = engine.merge_candidate(
        tx,
        0,
        RelationshipCandidate(
            source=source_name,
            target=target_name,
            relationship_type="mentor",
            direction="directed",
            evidence_quote=source_text,
            confidence=0.95,
        ),
        source_text=source_text,
        known_character_names=[source_name, target_name],
        language=language,
    )
    assert decision.status == "accepted"


def test_partial_latin_name_does_not_resolve_tomio(relationship_db):
    tx = "partial-name"
    engine = _engine_with_characters(
        relationship_db,
        tx,
        "Tomio Momozawa",
        "Apollo Rainbow",
    )
    source = "Tom spoke with Apollo Rainbow."
    decision = engine.merge_candidate(
        tx,
        0,
        RelationshipCandidate(
            source="Tom",
            target="Apollo Rainbow",
            relationship_type="friend",
            evidence_quote=source,
            confidence=0.99,
        ),
        source_text=source,
        known_character_names=["Tomio Momozawa", "Apollo Rainbow"],
        language="English",
    )
    assert decision.status == "quarantined"
    assert relationship_db.get_relationship_node_by_name(tx, "tom") is None


def test_relationship_graph_rejects_incompatible_vietnamese_addressing(relationship_db):
    tx = "addressing-cross-check"
    engine = _engine_with_characters(relationship_db, tx, "Parent", "Child")
    assert engine.merge_candidate(
        tx, 0, _manual_candidate("Parent", "Child", "parent")
    ).status == "accepted"
    addressing = ContextMergeEngine(db=relationship_db)
    assert not addressing.apply_delta(
        tx,
        1,
        AddressingUpdateDelta(
            speaker="Parent",
            addressee="Child",
            self_pronoun="em",
            second_pronoun="anh",
            confidence=0.99,
            evidence_quote="Parent addressed Child.",
        ),
        target_language="Vietnamese",
        known_character_names=["Parent", "Child"],
    )
    assert relationship_db.get_addressing_rules(tx) == []


def test_existing_unlocked_addressing_is_quarantined_when_graph_disagrees(relationship_db):
    tx = "addressing-quarantine"
    engine = _engine_with_characters(relationship_db, tx, "Parent", "Child")
    engine.merge_candidate(tx, 0, _manual_candidate("Parent", "Child", "parent"))
    relationship_db.upsert_addressing_rule(
        tx,
        "Parent",
        "Child",
        "em",
        "anh",
        confidence=0.99,
    )
    assert quarantine_incompatible_addressing_rules(
        translation_id=tx,
        db=relationship_db,
        target_language="Vietnamese",
        chunk_index=2,
    ) == 1
    assert relationship_db.get_addressing_rules(tx) == []
    assert relationship_db.get_context_audit_logs(tx)[0]["new_state"]["status"] == "quarantined"


def test_locked_addressing_overrides_relationship_graph(relationship_db):
    tx = "addressing-locked"
    engine = _engine_with_characters(relationship_db, tx, "Parent", "Child")
    engine.merge_candidate(tx, 0, _manual_candidate("Parent", "Child", "parent"))
    relationship_db.upsert_addressing_rule(
        tx,
        "Parent",
        "Child",
        "em",
        "anh",
        confidence=0.99,
        is_locked=1,
    )
    assert quarantine_incompatible_addressing_rules(
        translation_id=tx,
        db=relationship_db,
        target_language="Vietnamese",
    ) == 0
    assert relationship_db.get_addressing_rules(tx)[0]["is_locked"] == 1


def test_relationship_projection_is_before_markdown_and_after_addressing():
    context = "A is a recurring character."
    prompt = generate_translation_prompt(
        main_content="Text",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="French",
        prompt_options={
            "directed_addressing_context": "- A -> B: self=je, target=vous",
            "relationship_context": "- A <-> B: colleague, scope=durable",
            "novel_context": context,
        },
    )
    directed = prompt.user.index("# STRUCTURED DIRECTED ADDRESSING RULES")
    relationship = prompt.user.index("# STRUCTURED RELATIONSHIP CONTEXT")
    markdown = prompt.user.index("# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)")
    assert directed < relationship < markdown


def test_projection_and_markdown_export_include_only_accepted_edges(relationship_db):
    tx = "projection"
    engine = _engine_with_characters(relationship_db, tx, "A", "B", "C")
    engine.merge_candidate(tx, 0, _manual_candidate("A", "B", "friend", direction="symmetric"))
    engine.merge_candidate(
        tx,
        1,
        RelationshipCandidate(
            source="A",
            target="C",
            relationship_type="mentor",
            confidence=0.1,
            evidence_quote="",
        ),
        known_character_names=["A", "C"],
    )
    projection = build_relationship_projection(
        tx,
        relationship_db,
        reference_text="A met B.",
    )
    assert "A <-> B" in projection.prompt_text
    assert "A -> C" not in projection.prompt_text
    exported = export_relationship_graph_to_markdown(tx, relationship_db)
    assert "A \u2194 B" in exported
    assert "A \u2192 C" not in exported


def test_project_staging_keeps_prior_relationships_and_drops_unaccepted_new_edges(
    relationship_db,
):
    tx = "staging"
    fallback = """## CHARACTERS & GENDERS
- A: Unspecified, character.
- B: Unspecified, character.
- C: Unspecified, character.

## CHARACTER ALIASES

## NAME TRANSLATION MAP

## GLOSSARY & TERMINOLOGY

---DYNAMIC_STATE_START---
# DYNAMIC RELATIONSHIP STATE
## CURRENT ADDRESSING FORMS

## RELATIONSHIP EVOLUTION
- A \u2194 B: friends
- source memory marker
---DYNAMIC_STATE_END---"""
    proposed = fallback.replace(
        "- source memory marker",
        "- A \u2192 C: mentor\n- source memory marker\n- latest source marker",
    )
    staged = apply_relationship_graph_to_context(
        proposed,
        tx,
        relationship_db,
        fallback_context=fallback,
    )
    assert "- A \u2194 B: friends" in staged
    assert "- A \u2192 C: mentor" not in staged
    assert "- source memory marker" in staged
    assert "- latest source marker" in staged


def test_context_sync_records_invalid_contract_without_accepting_state(relationship_db):
    tx = "parse-failure"
    sync_context_update_relationships_to_db(
        translation_id=tx,
        db=relationship_db,
        updated_context_or_dynamic_state="",
        source_text="",
        candidates=[],
        parser_status="invalid_json",
        chunk_index=4,
    )
    assert relationship_db.get_relationship_edges(tx, statuses=["accepted"]) == []
    conflict = relationship_db.get_relationship_conflicts(tx, status="open")[0]
    assert conflict["validator"] == "relationship_contract"
    assert conflict["chunk_index"] == 4


def test_relationship_candidate_parser_accepts_json_and_rejects_malformed():
    candidates, status = parse_relationship_candidate_block(json.dumps({
        "relationships": [{
            "source": "A",
            "target": "B",
            "relationship_type": "rival",
            "direction": "symmetric",
            "evidence_quote": "A called B a rival.",
            "confidence": 0.9,
        }],
    }))
    assert status == "json"
    assert candidates[0].relationship_type == "rival"
    assert parse_relationship_candidate_block("{bad json")[1] == "invalid_json"


@pytest.mark.asyncio
async def test_llm_judge_cannot_bypass_missing_source_evidence(relationship_db):
    class Judge:
        async def generate(self, **_kwargs):
            return SimpleNamespace(content=json.dumps({
                "decision": "support",
                "confidence": 0.99,
                "reason": "Likely supported.",
            }))

    tx = "judge"
    engine = _engine_with_characters(relationship_db, tx, "A", "B")
    candidate = RelationshipCandidate(
        source="A",
        target="B",
        relationship_type="mentor",
        confidence=0.4,
    )
    judged, result = await judge_relationship_candidate(
        Judge(),
        candidate,
        "A and B entered the room.",
    )
    assert result.decision == "support"
    assert judged.confidence == 0.99
    decision = engine.merge_candidate(
        tx,
        0,
        judged,
        source_text="A and B entered the room.",
        known_character_names=["A", "B"],
    )
    assert decision.status == "quarantined"
    assert decision.validator == "source_evidence"


@pytest.mark.asyncio
async def test_context_update_exposes_typed_relationship_candidates():
    source = "A called B a trusted rival."
    response_text = f"""[NEW_CHARACTERS]
[IDENTITY_LINKS]
[NEW_GLOSSARY]
[DYNAMIC_STATE]
# DYNAMIC RELATIONSHIP STATE
## CURRENT ADDRESSING FORMS
## RELATIONSHIP EVOLUTION
- A \u2194 B: rivals
[RELATIONSHIP_CANDIDATES]
{{"relationships":[{{"source":"A","target":"B","relationship_type":"rival","direction":"symmetric","scope":"durable","hierarchy":"peer","evidence_quote":"{source}","confidence":0.95,"source_entity_type":"character","target_entity_type":"character","details":"trusted rivals"}}]}}
[DIALOGUE_ATTRIBUTION]
{{"turns":[],"state_after":{{}}}}"""

    class Client:
        async def generate(self, **_kwargs):
            return SimpleNamespace(content=response_text)

    sink = {}
    await update_novel_context_chunk(
        llm_client=Client(),
        model_name="test",
        current_global_lore=(
            "## CHARACTERS & GENDERS\n"
            "- A: Unspecified, character.\n"
            "- B: Unspecified, character.\n\n"
            "## CHARACTER ALIASES\n\n"
            "## NAME TRANSLATION MAP\n\n"
            "## GLOSSARY & TERMINOLOGY"
        ),
        current_dynamic_state="",
        source_chunk=source,
        translated_chunk=None,
        source_language="English",
        target_language="French",
        chunk_index=1,
        total_chunks=2,
        relationship_candidate_sink=sink,
    )
    assert sink["parse_status"] == "json"
    assert sink["candidates"][0]["relationship_type"] == "rival"
    assert sink["candidates"][0]["evidence_quote"] == source


@pytest.mark.asyncio
async def test_context_update_retries_malformed_relationship_json_once():
    first = """[NEW_CHARACTERS]
[IDENTITY_LINKS]
[NEW_GLOSSARY]
[DYNAMIC_STATE]
# DYNAMIC RELATIONSHIP STATE
## CURRENT ADDRESSING FORMS
## RELATIONSHIP EVOLUTION
[RELATIONSHIP_CANDIDATES]
{bad json
[DIALOGUE_ATTRIBUTION]
{"turns":[],"state_after":{}}"""
    second = json.dumps({
        "relationships": [{
            "source": "A",
            "target": "B",
            "relationship_type": "friend",
            "direction": "symmetric",
            "scope": "durable",
            "evidence_quote": "A called B a friend.",
            "confidence": 0.9,
        }],
    })

    class Client:
        def __init__(self):
            self.responses = [first, second]
            self.calls = 0

        async def generate(self, **_kwargs):
            response = self.responses[self.calls]
            self.calls += 1
            return SimpleNamespace(content=response)

    client = Client()
    sink = {}
    await update_novel_context_chunk(
        llm_client=client,
        model_name="test",
        current_global_lore="",
        current_dynamic_state="",
        source_chunk="A called B a friend.",
        translated_chunk=None,
        source_language="English",
        target_language="German",
        chunk_index=1,
        total_chunks=2,
        relationship_candidate_sink=sink,
    )
    assert client.calls == 2
    assert sink["parse_status"] == "json_repaired"
    assert sink["candidates"][0]["relationship_type"] == "friend"


@pytest.mark.asyncio
async def test_ambiguous_relationship_judge_is_batched_once_per_chunk():
    class Judge:
        def __init__(self):
            self.calls = 0

        async def generate(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content=json.dumps({
                "decisions": [
                    {"index": 0, "decision": "support", "confidence": 0.9, "reason": "Explicit."},
                    {"index": 1, "decision": "uncertain", "confidence": 0.4, "reason": "Ambiguous."},
                ],
            }))

    client = Judge()
    candidates = [
        RelationshipCandidate(
            source="A",
            target="B",
            relationship_type="associated",
            evidence_quote="A called B a colleague.",
            confidence=0.6,
        ),
        RelationshipCandidate(
            source="C",
            target="D",
            relationship_type="associated",
            evidence_quote="C stood near D.",
            confidence=0.5,
        ),
    ]
    judged = await judge_ambiguous_relationship_candidates(
        llm_client=client,
        candidates=candidates,
        source_text="A called B a colleague. C stood near D.",
        model_name="test",
        enabled=True,
    )
    assert client.calls == 1
    assert [item["judge_decision"] for item in judged] == ["support", "uncertain"]


def test_friend_and_older_sibling_chain_derives_social_seniority(relationship_db):
    tx = "multi-hop-seniority"
    engine = _engine_with_characters(relationship_db, tx, "A", "B", "C")
    engine.register_node(tx, "A", gender="male")
    engine.register_node(tx, "C", gender="female")
    assert engine.merge_candidate(
        tx,
        0,
        _manual_candidate(
            "A",
            "B",
            "friend",
            direction="symmetric",
            hierarchy="peer",
        ),
    ).status == "accepted"
    assert engine.merge_candidate(
        tx,
        1,
        _manual_candidate(
            "C",
            "B",
            "sibling",
            direction="symmetric",
            hierarchy="source_senior",
            relative_age="source_older",
        ),
    ).status == "accepted"

    support = relationship_support_for_addressing(
        relationship_db,
        tx,
        "A",
        "C",
    )
    assert support["derived"] is True
    assert support["hierarchy"] == "source_junior"
    assert support["confidence"] >= 0.72
    assert len(support["path"]) == 2
    assert relationship_db.get_relationship_derivations(tx)[0]["status"] == "accepted"

    projection = build_relationship_projection(
        tx,
        relationship_db,
        active_character_names=["A", "C"],
        target_language="Vietnamese",
    )
    assert "Derived social seniority" in projection.prompt_text
    assert "A -> C: hierarchy=source_junior" in projection.prompt_text
    assert "default self=em, target=chị" in projection.prompt_text


def test_internally_inconsistent_parent_seniority_is_quarantined(relationship_db):
    tx = "inconsistent-parent"
    engine = _engine_with_characters(relationship_db, tx, "Parent", "Child")
    decision = engine.merge_candidate(
        tx,
        0,
        _manual_candidate(
            "Parent",
            "Child",
            "parent",
            hierarchy="source_junior",
            relative_age="source_younger",
        ),
    )
    assert decision.status == "quarantined"
    assert decision.validator == "semantic_hierarchy"
    assert relationship_db.get_relationship_edges(tx, statuses=["accepted"]) == []


def test_relationship_beta_routes_support_audit_lock_quarantine_and_delete(
    relationship_db,
    tmp_path,
):
    tx = "routes"
    engine = _engine_with_characters(relationship_db, tx, "A", "B")
    decision = engine.merge_candidate(
        tx,
        3,
        _manual_candidate("A", "B", "friend", direction="symmetric"),
    )
    checkpoint_manager = SimpleNamespace(db=relationship_db)
    state_manager = SimpleNamespace(checkpoint_manager=checkpoint_manager)
    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state_manager,
        lambda *_args, **_kwargs: None,
        str(tmp_path),
    ))

    with app.test_client() as client:
        graph = client.get(f"/api/translation/{tx}/relationship-graph")
        assert graph.status_code == 200
        assert graph.get_json()["edge_count"] == 1

        locked = client.post(
            f"/api/translation/{tx}/relationship-edges/{decision.edge_id}/lock",
            json={"is_locked": True},
        )
        assert locked.status_code == 200
        assert client.post(
            f"/api/translation/{tx}/relationship-edges/{decision.edge_id}/quarantine"
        ).status_code == 409

        assert client.post(
            f"/api/translation/{tx}/relationship-edges/{decision.edge_id}/lock",
            json={"is_locked": False},
        ).status_code == 200
        assert client.post(
            f"/api/translation/{tx}/relationship-edges/{decision.edge_id}/quarantine"
        ).status_code == 200

        conflicts = client.get(
            f"/api/translation/{tx}/relationship-conflicts?status=open"
        )
        assert conflicts.status_code == 200
        assert conflicts.get_json()["count"] == 1

        audit = client.get(
            f"/api/translation/{tx}/relationship-audit?source=A&target=B"
        )
        assert audit.status_code == 200
        assert audit.get_json()["edges"][0]["status"] == "quarantined"

        resolution = client.get(
            f"/api/translation/{tx}/addressing-resolution?speaker=A&addressee=B"
        )
        assert resolution.status_code == 200
        assert "resolution" in resolution.get_json()

        deleted = client.delete(
            f"/api/translation/{tx}/relationship-edges/{decision.edge_id}"
        )
        assert deleted.status_code == 200
