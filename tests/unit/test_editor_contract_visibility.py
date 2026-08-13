"""The editor has to be told what to look for, and be readable when it answers.

Both halves of this file guard the same regression. Between `editor-issue-v6`
and `editor-issue-v7` the contract lost its enumeration of what counts as a
defect, leaving the circular instruction "report issues when issues exist"; at
the same time the critique log was condensed to a three-bullet summary. A small
editor model answered `no_issues` for thirty-one consecutive chunks and nobody
could see it happen.
"""

import src.core.translator as translator
from src.prompts.prompts import (
    REFLECTION_CONTRACT_VERSION,
    _build_reflection_json_contract_section,
)

CHECKLIST_ITEMS = (
    "omission",
    "addition",
    "mistranslation",
    "glossary error",
    "pronoun bleed",
    "gender mismatch",
    "register",
    "consistency",
    "fluency",
    "style",
    "placeholder/format",
)


def test_the_contract_enumerates_every_defect_class():
    contract = _build_reflection_json_contract_section()
    missing = [item for item in CHECKLIST_ITEMS if item not in contract]
    assert not missing, f"contract no longer names: {missing}"


def test_the_native_schema_variant_keeps_the_checklist():
    # The Gemini path drops the tag wrapper and the inline example, but the
    # audit list is the part that makes the task concrete -- it must survive.
    contract = _build_reflection_json_contract_section(native_schema=True)
    missing = [item for item in CHECKLIST_ITEMS if item not in contract]
    assert not missing, f"native contract no longer names: {missing}"


def test_the_contract_version_records_the_enumeration():
    # The version string is stamped on every editor_runs row, so a job's
    # database is enough to tell which contract produced its verdicts.
    assert REFLECTION_CONTRACT_VERSION == "editor-issue-v9-severity-split"


def test_a_clean_verdict_is_still_allowed():
    contract = _build_reflection_json_contract_section()
    assert "no_issues" in contract
    assert "needs_repair" in contract


def _capture(monkeypatch, *, enabled):
    captured = []
    monkeypatch.setattr(translator, "EDITOR_LOG_FULL_RESPONSE", enabled)
    translator.log_full_editor_response(
        lambda event, message, **kwargs: captured.append((event, message)),
        "reflection",
        '{"status":"no_issues","issues":[],"voice_observations":[]}',
        7,
    )
    return captured


def test_the_raw_response_stays_hidden_by_default(monkeypatch):
    assert _capture(monkeypatch, enabled=False) == []


def test_the_raw_response_is_printed_verbatim_when_asked(monkeypatch):
    captured = _capture(monkeypatch, enabled=True)
    assert len(captured) == 1
    event, message = captured[0]
    assert event == "reflection_full_response"
    # Verbatim means verbatim: the empty verdict that hid the regression has to
    # be recognisable in the log, not summarised away.
    assert '{"status":"no_issues","issues":[],"voice_observations":[]}' in message


def test_an_empty_response_is_reported_rather_than_skipped(monkeypatch):
    monkeypatch.setattr(translator, "EDITOR_LOG_FULL_RESPONSE", True)
    captured = []
    translator.log_full_editor_response(
        lambda event, message, **kwargs: captured.append(message), "reflection", "",
    )
    assert captured and "<empty>" in captured[0]


def test_severity_is_separated_from_certainty():
    """The contract must not tell the model to bury minor defects.

    `Uncertain or minor evidence is review_only` conflated two questions: how
    much a defect costs the reader, and how sure the editor is that it is one.
    The second already has its own rule below it, so the first was pure loss --
    a confidently located minor defect was instructed to arrive unrepairable.
    """

    for native in (False, True):
        contract = _build_reflection_json_contract_section(native_schema=native)
        assert "Uncertain evidence is review_only." in contract
        assert "Uncertain or minor evidence" not in contract
        assert "a\n  minor defect you can point at and fix is still local_replace" in contract
        # The uncertainty gate itself stays.
        assert "Confidence below 0.80 is" in contract
