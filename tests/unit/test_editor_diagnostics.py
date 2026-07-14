"""Persistence and provider plumbing tests for Senior Editor diagnostics."""

from types import SimpleNamespace

import pytest

from src.core.llm.base import LLMResponse
from src.core.llm_client import LLMClient
from src.persistence.database import Database
from src.core.llm.exceptions import ProviderRequestError
from src.core.translator import run_chunk_reflection_pass


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
