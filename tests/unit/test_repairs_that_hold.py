"""One repair that fails must not take the repairs beside it with it."""

from types import SimpleNamespace

import pytest

from src.core.translator import run_chunk_reflection_pass
from src.utils.translation_quality import (
    apply_repairs_that_hold,
    validate_editor_repair,
)


def _issue(issue_id, quote, draft, replacement):
    return {
        "issue_id": issue_id,
        "segment_id": "D0001",
        "category": "wording",
        "severity": "major",
        "confidence": 0.95,
        "repair_kind": "local_replace",
        "source_quote": quote,
        "draft_quote": quote,
        "instruction": "Fix it.",
        "draft_replacement": {"draft": draft, "replacement": replacement},
        "glossary_update": None,
    }


def _apply(applied):
    """A patcher that applies every issue except the ones named."""

    def apply_patches(text, items):
        unresolved = []
        for issue in items:
            if issue["issue_id"] in applied:
                pair = issue["draft_replacement"]
                text = text.replace(pair["draft"], pair["replacement"])
            else:
                unresolved.append(issue)
        return text, unresolved, []

    return apply_patches


def _validate(failing):
    """A validator that reports the named repairs as not having taken."""

    def validate(text, items):
        return [
            f"replacement_not_applied_locally: {issue['draft_replacement']['draft']}"
            for issue in items
            if issue["issue_id"] in failing
        ]

    return validate


def test_the_repairs_that_work_survive_one_that_does_not():
    issues = [
        _issue("ISSUE-001", "alpha", "alpha", "A"),
        _issue("ISSUE-002", "beta", "beta", "B"),
        _issue("ISSUE-003", "gamma", "gamma", "C"),
    ]

    result = apply_repairs_that_hold(
        "alpha beta gamma",
        issues,
        apply_patches=_apply({"ISSUE-001", "ISSUE-002", "ISSUE-003"}),
        validate=_validate({"ISSUE-002"}),
    )

    assert result.text == "A beta C"
    assert [issue["issue_id"] for issue in result.dropped] == ["ISSUE-002"]
    assert [issue["issue_id"] for issue in result.unresolved] == ["ISSUE-002"]
    assert result.errors == []


def test_a_fault_the_draft_already_had_does_not_veto_the_repairs():
    # The measured case: two chunks lost every repair to residue that was in
    # the draft before the editor touched it. Reverting cannot remove such a
    # fault, so vetoing the batch over it gives up the repairs for nothing.
    issues = [_issue("ISSUE-001", "alpha", "alpha", "A")]

    result = apply_repairs_that_hold(
        "alpha beta",
        issues,
        apply_patches=_apply({"ISSUE-001"}),
        validate=lambda text, items: ["source residue remains: senior"],
    )

    assert result.text == "A beta"
    assert result.dropped == []
    assert result.errors == []


def test_a_fault_the_repair_introduced_still_fails_the_batch():
    issues = [_issue("ISSUE-001", "alpha", "alpha", "A")]

    def validate(text, items):
        return ["protected_term_removed: Road Axe"] if "A" in text else []

    result = apply_repairs_that_hold(
        "alpha beta",
        issues,
        apply_patches=_apply({"ISSUE-001"}),
        validate=validate,
    )

    assert result.text == "alpha beta"
    assert result.errors == ["protected_term_removed: Road Axe"]
    assert [issue["issue_id"] for issue in result.unresolved] == ["ISSUE-001"]


def test_every_repair_failing_leaves_the_draft_alone():
    issues = [
        _issue("ISSUE-001", "alpha", "alpha", "A"),
        _issue("ISSUE-002", "beta", "beta", "B"),
    ]

    result = apply_repairs_that_hold(
        "alpha beta",
        issues,
        apply_patches=_apply({"ISSUE-001", "ISSUE-002"}),
        validate=_validate({"ISSUE-001", "ISSUE-002"}),
    )

    assert result.text == "alpha beta"
    assert {issue["issue_id"] for issue in result.dropped} == {
        "ISSUE-001", "ISSUE-002",
    }


def test_colliding_patches_are_left_for_the_editor_to_merge():
    # Two readings of one stretch of text. Dropping either half would pick a
    # winner in a disagreement, so the batch fails and the focused retry asks
    # the editor to reconcile them instead.
    issues = [
        _issue("overlap-1", "alpha beta gamma", "alpha beta", "X"),
        _issue("overlap-2", "alpha beta gamma", "beta gamma", "Y"),
    ]

    def apply_patches(text, items):
        return text, [], ["local_patch_conflict:overlap-1:overlap-2"]

    result = apply_repairs_that_hold(
        "alpha beta gamma",
        issues,
        apply_patches=apply_patches,
        validate=lambda text, items: [],
    )

    assert result.text == "alpha beta gamma"
    assert result.dropped == []
    assert result.errors == ["local_patch_conflict:overlap-1:overlap-2"]


def test_the_real_validator_reports_a_repair_that_did_not_take():
    # Guards the string the attribution matches on: if the validator ever
    # renames this error, dropping stops working and the batch silently goes
    # back to all-or-nothing.
    issues = [_issue("ISSUE-001", "alpha", "alpha", "A")]

    errors = validate_editor_repair(
        "alpha beta",
        issues,
        draft_text="alpha beta",
        source_text="alpha beta",
        source_language="English",
        target_language="Vietnamese",
    )

    assert "replacement_not_applied_locally: alpha" in errors


@pytest.mark.asyncio
async def test_a_chunk_keeps_the_repairs_that_applied_when_one_quote_is_wrong():
    # The measured chunk: six findings, one quoting text that is not in the
    # draft, and all six discarded. The five that locate cleanly now stay.
    reflection = (
        '{"status":"needs_repair","issues":['
        '{"issue_id":"ISSUE-001","segment_id":"D0001","category":"wording",'
        '"severity":"major","confidence":0.95,"repair_kind":"local_replace",'
        '"source_quote":"alpha","draft_quote":"alpha","instruction":"first",'
        '"draft_replacement":{"draft":"alpha","replacement":"AAA"},'
        '"glossary_update":null},'
        '{"issue_id":"ISSUE-002","segment_id":"D0001","category":"wording",'
        '"severity":"major","confidence":0.95,"repair_kind":"local_replace",'
        '"source_quote":"nowhere","draft_quote":"not in the draft at all",'
        '"instruction":"second","draft_replacement":'
        '{"draft":"not in the draft at all","replacement":"BBB"},'
        '"glossary_update":null},'
        '{"issue_id":"ISSUE-003","segment_id":"D0001","category":"wording",'
        '"severity":"major","confidence":0.95,"repair_kind":"local_replace",'
        '"source_quote":"gamma","draft_quote":"gamma","instruction":"third",'
        '"draft_replacement":{"draft":"gamma","replacement":"CCC"},'
        '"glossary_update":null}],"voice_observations":[]}'
    )

    class Client:
        def __init__(self):
            self.responses = [reflection] + ['{"status":"no_issues","issues":[]}'] * 5
            self.requests = []

        async def generate_async(self, **kwargs):
            self.requests.append(kwargs)
            return SimpleNamespace(content=self.responses.pop(0))

    result = await run_chunk_reflection_pass(
        source_chunk="alpha beta gamma",
        draft_translation="alpha beta gamma",
        target_language="English",
        model_name="test",
        llm_client=Client(),
        prompt_options={"context_contract_version": 5, "source_language": "English"},
    )

    assert "AAA" in result and "CCC" in result
