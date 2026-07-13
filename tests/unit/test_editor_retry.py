"""Tests for immediate checkpoint-backed Senior Editor retries."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.editor_retry import editor_retry_state, run_editor_retry


def test_editor_retry_state_exposes_legacy_pending_checkpoint():
    assert editor_retry_state({"editor_retry_pending": True}) == {
        "status": "ready",
        "legacy_pending": True,
    }


@pytest.mark.asyncio
async def test_editor_retry_runs_immediately_and_saves_result(monkeypatch, tmp_path):
    class DB:
        db_path = str(tmp_path / "jobs.db")

        def get_editor_diagnostics(self, _translation_id):
            return {
                "runs": [
                    {
                        "id": 7,
                        "chunk_index": 0,
                        "outcome": "warnings_only",
                    }
                ]
            }

    class Checkpoints:
        db = DB()

        def __init__(self):
            self.saved = []
            self.checkpoint = {
                "job": {
                    "file_type": "txt",
                    "config": {
                        "llm_provider": "gemini",
                        "model": "draft-model",
                        "source_language": "English",
                        "target_language": "Vietnamese",
                        "output_filename": "translated.txt",
                        "prompt_options": {
                            "editor_provider": "gemini",
                            "editor_model": "editor-model",
                        },
                    },
                },
                "chunks": [
                    {
                        "chunk_index": 0,
                        "original_text": "Hello.",
                        "translated_text": "Xin chào.",
                        "status": "completed",
                        "chunk_data": {},
                    }
                ],
            }

        def load_checkpoint(self, _translation_id):
            return self.checkpoint

        def save_checkpoint(self, **kwargs):
            self.saved.append(kwargs)
            return True

    checkpoints = Checkpoints()
    editor_client = object()
    monkeypatch.setattr(
        "src.core.llm.runtime.build_runtime_spec",
        lambda *_args, **_kwargs: SimpleNamespace(
            provider="gemini",
            model="editor-model",
            api_key="<REDACTED>",
            key_required=True,
        ),
    )
    monkeypatch.setattr(
        "src.core.llm.runtime.create_runtime_client",
        lambda *_args, **_kwargs: editor_client,
    )

    async def fake_reflection(**kwargs):
        assert kwargs["llm_client"] is editor_client
        assert kwargs["prompt_options"]["editor_phase"] == "manual_retry"
        return "Xin chào!"

    async def fake_refresh(*_args, **_kwargs):
        return {"status": "updated", "filename": "translated.txt"}

    monkeypatch.setattr(
        "src.core.translator.run_chunk_reflection_pass", fake_reflection
    )
    monkeypatch.setattr("src.core.editor_retry._refresh_output", fake_refresh)

    state = await run_editor_retry(
        translation_id="job-1",
        chunk_index=0,
        checkpoint_manager=checkpoints,
        output_dir=Path(tmp_path),
    )

    assert state["status"] == "succeeded"
    assert state["outcome"] == "warnings_only"
    assert state["output_sync"]["status"] == "updated"
    assert checkpoints.saved[-1]["translated_text"] == "Xin chào!"
    assert (
        checkpoints.saved[-1]["chunk_data"]["editor_retry"]["status"]
        == "succeeded"
    )
