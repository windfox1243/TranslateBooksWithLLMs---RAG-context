"""Persistence and provider plumbing tests for Senior Editor diagnostics."""

from types import SimpleNamespace

import pytest

from src.core.llm.base import LLMResponse
from src.core.llm_client import LLMClient
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
    assert result["summary"]["outcomes"] == {"draft_kept_review": 1}
    assert db.delete_job("job-1")
    assert db.get_editor_diagnostics("job-1")["classification"] == "legacy_unclassified"
