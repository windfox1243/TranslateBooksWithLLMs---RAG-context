"""API orchestration tests for safe Senior Editor retries."""

import threading
from copy import deepcopy

from flask import Flask

from src.api.blueprints.translation_routes import create_translation_blueprint


class _RetryDB:
    def get_editor_diagnostics(self, _translation_id):
        return {
            "runs": [{
                "id": 9,
                "chunk_index": 0,
                "outcome": "review_required",
            }]
        }


class _RetryCheckpoints:
    def __init__(self, preserved_path):
        self.db = _RetryDB()
        self.checkpoint = {
            "job": {
                "config": {
                    "llm_provider": "ollama",
                    "model": "draft-model",
                    "llm_api_endpoint": "http://127.0.0.1:11434/api/generate",
                    "file_type": "txt",
                    "output_filename": "translated.txt",
                    "preserved_input_path": str(preserved_path),
                },
            },
            "resume_from_index": 1,
            "chunks": [{
                "chunk_index": 0,
                "status": "completed",
                "original_text": "Source",
                "translated_text": "Draft",
                "chunk_data": {
                    "editor_validation": {
                        "unresolved_issues": [{
                            "issue_id": "keep-me",
                            "repair_kind": "local_replace",
                        }],
                    },
                },
            }],
        }
        self.marked_running = False

    def load_checkpoint(self, _translation_id):
        return deepcopy(self.checkpoint)

    def save_checkpoint(self, **kwargs):
        chunk = self.checkpoint["chunks"][0]
        chunk.update({
            "status": kwargs["chunk_status"],
            "original_text": kwargs["original_text"],
            "translated_text": kwargs["translated_text"],
            "chunk_data": deepcopy(kwargs["chunk_data"]),
        })
        return True

    def get_job(self, _translation_id):
        return {
            "status": "interrupted",
            "config": deepcopy(self.checkpoint["job"]["config"]),
        }

    def get_preserved_input_path(self, _translation_id):
        return self.checkpoint["job"]["config"]["preserved_input_path"]

    def mark_running(self, _translation_id):
        self.marked_running = True
        return True


class _RetryState:
    def __init__(self, checkpoints):
        self.checkpoint_manager = checkpoints
        self.job = {
            "status": "running",
            "config": {
                **checkpoints.checkpoint["job"]["config"],
                "request_timeout": 60,
            },
            "interrupted": False,
        }

    def get_translation(self, translation_id):
        return deepcopy(self.job) if translation_id == "job-1" else None

    def get_all_translations(self):
        return {"job-1": deepcopy(self.job)}

    def set_interrupted(self, _translation_id, interrupted=True):
        self.job["interrupted"] = interrupted
        if interrupted:
            self.job["status"] = "interrupted"
        return True

    def exists(self, translation_id):
        return translation_id == "job-1"

    def restore_job_from_checkpoint(self, _translation_id):
        return True

    def get_translation_field(self, _translation_id, field, default=None):
        return self.job.get(field, default)

    def set_translation_field(self, _translation_id, field, value):
        self.job[field] = value
        return True


def test_retry_pauses_same_running_job_then_auto_resumes(
    monkeypatch,
    tmp_path,
):
    preserved = tmp_path / "source.txt"
    preserved.write_text("Source", encoding="utf-8")
    checkpoints = _RetryCheckpoints(preserved)
    state = _RetryState(checkpoints)
    resumed = threading.Event()
    retried = threading.Event()
    started_config = {}

    async def fake_retry(**_kwargs):
        retried.set()
        return {"status": "succeeded"}

    def start_job(_translation_id, config):
        started_config.update(config)
        resumed.set()

    monkeypatch.setattr(
        "src.core.editor_retry.run_editor_retry",
        fake_retry,
    )
    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state,
        start_job,
        str(tmp_path),
    ))

    response = app.test_client().post(
        "/api/translation/job-1/chunks/0/retry-editor",
        json={},
    )

    assert response.status_code == 202
    assert response.get_json()["retry_state"]["pause_requested"] is True
    assert retried.wait(2)
    assert resumed.wait(2)
    assert state.job["interrupted"] is False
    assert state.job["status"] == "running"
    assert started_config["is_resume"] is True
    assert started_config["resume_from_index"] == 1
    assert checkpoints.marked_running is True
    assert checkpoints.checkpoint["chunks"][0]["chunk_data"][
        "editor_validation"
    ]["unresolved_issues"][0]["issue_id"] == "keep-me"


def test_retry_still_rejects_when_another_job_is_active(tmp_path):
    preserved = tmp_path / "source.txt"
    preserved.write_text("Source", encoding="utf-8")
    checkpoints = _RetryCheckpoints(preserved)
    state = _RetryState(checkpoints)
    state.job["status"] = "interrupted"

    def get_all():
        return {
            "job-1": deepcopy(state.job),
            "job-2": {
                "status": "running",
                "config": {"output_filename": "other.txt"},
            },
        }

    state.get_all_translations = get_all
    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state,
        lambda *_args: None,
        str(tmp_path),
    ))

    response = app.test_client().post(
        "/api/translation/job-1/chunks/0/retry-editor",
        json={},
    )

    assert response.status_code == 409
    assert response.get_json()["active_translations"][0]["id"] == "job-2"


def test_retry_failure_still_resumes_the_paused_translation(
    monkeypatch,
    tmp_path,
):
    preserved = tmp_path / "source.txt"
    preserved.write_text("Source", encoding="utf-8")
    checkpoints = _RetryCheckpoints(preserved)
    state = _RetryState(checkpoints)
    resumed = threading.Event()

    async def failed_retry(**_kwargs):
        raise RuntimeError("editor unavailable")

    monkeypatch.setattr(
        "src.core.editor_retry.run_editor_retry",
        failed_retry,
    )
    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state,
        lambda *_args: resumed.set(),
        str(tmp_path),
    ))

    response = app.test_client().post(
        "/api/translation/job-1/chunks/0/retry-editor",
        json={},
    )

    assert response.status_code == 202
    assert resumed.wait(2)
    assert state.job["interrupted"] is False
    assert state.job["status"] == "running"
