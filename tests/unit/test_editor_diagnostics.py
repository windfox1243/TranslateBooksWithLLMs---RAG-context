"""Persistence and provider plumbing tests for Senior Editor diagnostics."""

import json
from types import SimpleNamespace

import pytest

from src.core.llm.base import LLMResponse
from src.core.llm.exceptions import ProviderRequestError
from src.core.llm_client import LLMClient
from src.core.translator import run_chunk_reflection_pass
from src.persistence.database import Database


@pytest.mark.asyncio
async def test_llm_client_forwards_per_request_generation_options():
    captured = {}

    class Provider:
        async def generate(self, prompt, timeout=None, system_prompt=None, generation_options=None):
            captured["options"] = generation_options
            return LLMResponse(content="ok")

    client = LLMClient(provider_type="ollama", model="test")
    client._provider = Provider()
    await client.generate(
        "prompt",
        temperature=0.0,
        max_output_tokens=777,
        response_schema={"type": "object"},
        stage="reflection_retry",
    )
    options = captured["options"]
    assert options.temperature == 0.0
    assert options.max_output_tokens == 777
    assert options.response_schema == {"type": "object"}
    assert options.stage == "reflection_retry"


def test_editor_diagnostics_are_classified_and_deleted_with_job(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    assert db.create_job("job-1", "txt", {})
    run_id = db.create_editor_run({
        "translation_id": "job-1",
        "chunk_index": 2,
        "phase": "translation",
        "provider": "gemini",
        "model": "editor-model",
        "outcome": "running",
    })
    assert run_id is not None
    assert db.add_editor_attempt(run_id, {
        "attempt_index": 1,
        "stage": "reflection",
        "failure_class": "locator_ambiguous",
        "reason_codes": ["locator_ambiguous:issue-1"],
    })
    assert db.finish_editor_run(run_id, {
        "outcome": "draft_kept_review",
        "failure_class": "locator_ambiguous",
    })
    result = db.get_editor_diagnostics("job-1")
    assert result["classification"] == "classified"
    assert result["summary"]["outcomes"] == {"review_required": 1}
    assert result["runs"][0]["legacy_outcome"] == "draft_kept_review"
    assert db.delete_job("job-1")
    assert db.get_editor_diagnostics("job-1")["classification"] == "legacy_unclassified"


def test_new_editor_outcomes_and_attempts_round_trip_without_legacy_rewrite(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    assert db.create_job("job-2", "txt", {})
    run_id = db.create_editor_run({
        "translation_id": "job-2", "chunk_index": 0,
        "phase": "translation", "outcome": "running",
    })
    assert db.add_editor_attempt(run_id, {
        "attempt_index": 1, "stage": "reflection",
        "finish_reason": "MAX_TOKENS", "was_truncated": True,
        "reason_codes": ["adaptive_output_retry"],
        "prompt_tokens": 12000, "completion_tokens": 500,
        "total_tokens": 12500,
    })
    assert db.add_editor_attempt(run_id, {
        "attempt_index": 2, "stage": "locator_retry",
        "prompt_tokens": 2200, "completion_tokens": 200,
        "total_tokens": 2400,
    })
    assert db.finish_editor_run(run_id, {
        "outcome": "locally_repaired",
        "result_state": "locally_patched",
        "resolved_issue_count": 2,
        "unresolved_issue_count": 0,
        "recovered_truncation": True,
        "prompt_tokens": 14200, "completion_tokens": 700,
        "total_tokens": 14900,
    })
    result = db.get_editor_diagnostics("job-2")
    assert result["summary"]["outcomes"] == {"locally_repaired": 1}
    assert result["summary"]["successful"] == 1
    assert result["summary"]["hard_failed"] == 0
    assert result["summary"]["recovered"] == 1
    assert result["runs"][0]["result_state"] == "locally_patched"
    assert result["runs"][0]["attempts"][0]["reason_codes"] == [
        "adaptive_output_retry"
    ]
    assert result["runs"][0]["request_count"] == 2
    assert result["runs"][0]["max_request_prompt_tokens"] == 12000
    assert result["runs"][0]["max_request_total_tokens"] == 12500
    assert result["runs"][0]["cumulative_prompt_tokens"] == 14200
    assert result["runs"][0]["cumulative_total_tokens"] == 14900
    assert "excerpts" not in result["runs"][0]["attempts"][0]


@pytest.mark.asyncio
async def test_terminal_provider_failure_keeps_draft_and_retains_classification(tmp_path):
    db_path = str(tmp_path / "jobs.db")
    db = Database(db_path)
    assert db.create_job("job-auth", "txt", {})

    class Client:
        async def generate_async(self, **_kwargs):
            raise ProviderRequestError("provider_auth", 401)

    result = await run_chunk_reflection_pass(
        source_chunk="Source.", draft_translation="Valid draft.",
        target_language="English", model_name="editor", llm_client=Client(),
        prompt_options={
            "translation_id": "job-auth", "jobs_db_path": db_path,
            "chunk_index": 0, "source_language": "English",
        },
    )
    assert result == "Valid draft."
    diagnostics = db.get_editor_diagnostics("job-auth")
    assert diagnostics["summary"]["outcomes"] == {"transport_failed": 1}
    assert diagnostics["summary"]["failure_classes"] == {"provider_auth": 1}
    assert diagnostics["summary"]["degraded"] == 1
    assert diagnostics["summary"]["hard_failed"] == 0


@pytest.mark.asyncio
async def test_prompt_composition_records_complete_input_hashes(tmp_path):
    db_path = str(tmp_path / "jobs.db")
    db = Database(db_path)
    assert db.create_job("job-composition", "txt", {})

    class Client:
        async def generate_async(self, **_kwargs):
            return SimpleNamespace(
                content='{"status":"no_issues","issues":[]}',
                prompt_tokens=3210,
                completion_tokens=90,
                total_tokens=3300,
            )

    source = "Complete source sentence."
    draft = "Câu bản thảo hoàn chỉnh."
    result = await run_chunk_reflection_pass(
        source_chunk=source,
        draft_translation=draft,
        target_language="Vietnamese",
        model_name="editor",
        llm_client=Client(),
        prompt_options={
            "translation_id": "job-composition",
            "jobs_db_path": db_path,
            "chunk_index": 0,
            "source_language": "English",
            "context_contract_version": 5,
        },
    )
    assert result == draft
    run = db.get_editor_diagnostics("job-composition")["runs"][0]
    composition = run["diagnostics"]["prompt_composition"]
    assert composition["source_chars"] == len(source)
    assert composition["draft_chars"] == len(draft)
    assert composition["source_complete"] is True
    assert composition["draft_segment_chars"] == len(draft)
    assert len(composition["source_sha256"]) == 64
    assert len(composition["draft_sha256"]) == 64
    assert run["request_count"] == 1
    assert run["max_request_prompt_tokens"] == 3210


def _issue(issue_id, draft_quote, replacement, **overrides):
    """One well-formed local_replace issue, tuned by the caller."""

    issue = {
        "issue_id": issue_id,
        "segment_id": "SEG-0001",
        "category": "mistranslation",
        "severity": "major",
        "confidence": 0.95,
        "repair_kind": "local_replace",
        "source_quote": "Alpha one here.",
        "draft_quote": draft_quote,
        "instruction": "Fix it.",
        "draft_replacement": {"draft": draft_quote, "replacement": replacement},
        "glossary_update": None,
    }
    issue.update(overrides)
    return issue


@pytest.mark.asyncio
async def test_findings_dropped_before_the_actionable_filter_are_still_counted(
    tmp_path,
):
    """A finding discarded upstream must not vanish from the run's record.

    The editor reports two defects. One names a span that is not in the draft,
    so locator validation removes it; the other is too weak to repair
    automatically. Both are findings the reader never sees applied, and the
    warning count is the only place that says so.
    """

    db_path = str(tmp_path / "jobs.db")
    db = Database(db_path)
    assert db.create_job("job-warnings", "txt", {})
    draft = "Alpha one here. Beta two here."

    class Client:
        def __init__(self):
            self.calls = 0

        async def generate_async(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                issues = [
                    _issue("1", "Alpha one here.", "Alpha uno here.",
                           severity="minor", confidence=0.4),
                    _issue("2", "Nowhere in the draft.", "Somewhere else."),
                ]
                content = json.dumps(
                    {"status": "needs_repair", "issues": issues},
                    ensure_ascii=False,
                )
            else:
                # The locator retry cannot ground issue 2 either.
                content = '{"status":"needs_repair","issues":[]}'
            return SimpleNamespace(
                content=content, prompt_tokens=100,
                completion_tokens=50, total_tokens=150,
            )

    result = await run_chunk_reflection_pass(
        source_chunk="Alpha one here. Beta two here.",
        draft_translation=draft,
        target_language="English",
        model_name="editor",
        llm_client=Client(),
        prompt_options={
            "translation_id": "job-warnings",
            "jobs_db_path": db_path,
            "chunk_index": 0,
            "source_language": "English",
        },
    )
    assert result == draft
    run = db.get_editor_diagnostics("job-warnings")["runs"][0]
    assert run["resolved_issue_count"] == 0
    # Two reported, two unapplied: the ungrounded one and the weak one.
    assert run["warning_count"] == 2


@pytest.mark.asyncio
async def test_a_confident_minor_defect_is_repaired_and_a_hedged_one_is_not(
    tmp_path,
):
    """Severity says what a defect costs; confidence says whether to trust it.

    The gate used to demand `major` as well, so a minor defect the editor had
    located exactly and was sure of was discarded with the rest -- one measured
    chunk returned twelve such edits and applied none. Certainty is still the
    gate, and a hedged finding of any severity stays a warning.
    """

    db_path = str(tmp_path / "jobs.db")
    db = Database(db_path)
    assert db.create_job("job-minor", "txt", {})
    draft = "Alpha one here. Beta two here."

    class Client:
        async def generate_async(self, **_kwargs):
            issues = [
                _issue("1", "Alpha one here.", "Alpha uno here.",
                       severity="minor", confidence=0.95),
                _issue("2", "Beta two here.", "Beta dos here.",
                       severity="major", confidence=0.5,
                       source_quote="Beta two here.", segment_id="SEG-0002"),
            ]
            return SimpleNamespace(
                content=json.dumps(
                    {"status": "needs_repair", "issues": issues},
                    ensure_ascii=False,
                ),
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
            )

    result = await run_chunk_reflection_pass(
        source_chunk="Alpha one here. Beta two here.",
        draft_translation=draft,
        target_language="English",
        model_name="editor",
        llm_client=Client(),
        prompt_options={
            "translation_id": "job-minor",
            "jobs_db_path": db_path,
            "chunk_index": 0,
            "source_language": "English",
        },
    )
    assert "Alpha uno here." in result
    assert "Beta two here." in result
    run = db.get_editor_diagnostics("job-minor")["runs"][0]
    assert run["resolved_issue_count"] == 1
    assert run["warning_count"] == 1
    # The stored attempt says which rule decided, not merely that one did.
    excerpt = run["attempts"][0]
    assert excerpt is not None


def test_issue_excerpts_record_what_the_repair_gate_judges():
    from src.utils.editor_diagnostics import issue_excerpts

    excerpts = issue_excerpts([
        {
            "issue_id": "1", "category": "pronoun bleed", "severity": "minor",
            "confidence": 0.9123, "repair_kind": "local_replace",
            "source_quote": "abc", "draft_quote": "abcd",
            "draft_replacement": {"draft": "abcd", "replacement": "ab"},
        },
        {"issue_id": "2", "confidence": "not a number"},
    ])
    assert excerpts[0]["category"] == "pronoun bleed"
    assert excerpts[0]["severity"] == "minor"
    assert excerpts[0]["confidence"] == 0.912
    # A model that answers with the wrong type must not break the record.
    assert excerpts[1]["confidence"] is None
    assert excerpts[1]["severity"] == ""
    assert excerpts[0]["no_op"] is False
    assert issue_excerpts([
        {
            "issue_id": "3", "repair_kind": "local_replace",
            "draft_replacement": {"draft": "same", "replacement": "same"},
        },
    ])[0]["no_op"] is True


def test_a_confident_minor_defect_keeps_the_repair_the_editor_chose():
    """Parsing must not overrule the editor on severity alone.

    Every `minor` issue was rewritten to review_only as it was parsed, before
    any gate saw it, so no downstream policy could have let one through.
    """

    from src.core.translator import _normalize_reflection_issue

    def parsed(**overrides):
        raw = {
            "issue_id": "1", "segment_id": "SEG-0001",
            "category": "register", "severity": "minor", "confidence": 0.95,
            "repair_kind": "local_replace", "source_quote": "a",
            "draft_quote": "b", "instruction": "Fix it.",
            "draft_replacement": {"draft": "b", "replacement": "c"},
        }
        raw.update(overrides)
        return _normalize_reflection_issue(raw)

    assert parsed()["repair_kind"] == "local_replace"
    assert parsed(severity="blocker")["repair_kind"] == "local_replace"
    # Uncertainty is still the thing that withholds an automatic edit.
    assert parsed(confidence=0.5)["repair_kind"] == "review_only"
    assert parsed(severity="major", confidence=0.5)["repair_kind"] == "review_only"


@pytest.mark.asyncio
async def test_an_edit_refused_for_touching_a_protected_name_is_counted(tmp_path):
    """A finding the protected-span filter removes is still a finding.

    The editor proposed translating a proper name the deterministic pass had
    already ruled untouchable. Refusing the edit is right; recording the chunk
    as though the editor had reported nothing is not, and a chunk whose every
    finding landed on a protected entity read as a clean chunk.
    """

    db_path = str(tmp_path / "jobs.db")
    db = Database(db_path)
    assert db.create_job("job-protected", "txt", {})
    draft = "Momozawa Tomio đã đến đây."

    class Client:
        def __init__(self):
            self.calls = 0

        async def generate_async(self, **_kwargs):
            self.calls += 1
            issue = _issue("1", "Momozawa Tomio", "Đào Trạch Phú Hùng", category="name")
            issue["source_quote"] = "Momozawa Tomio arrived here."
            return SimpleNamespace(
                content=json.dumps(
                    {"status": "needs_repair", "issues": [issue]},
                    ensure_ascii=False,
                ),
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
            )

    result = await run_chunk_reflection_pass(
        source_chunk="Momozawa Tomio arrived here.",
        draft_translation=draft,
        target_language="Vietnamese",
        model_name="editor",
        llm_client=Client(),
        prompt_options={
            "translation_id": "job-protected",
            "jobs_db_path": db_path,
            "chunk_index": 0,
            "source_language": "English",
        },
    )
    assert result == draft
    run = db.get_editor_diagnostics("job-protected")["runs"][0]
    assert run["resolved_issue_count"] == 0
    assert run["warning_count"] == 1
