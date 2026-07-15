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


class _BatchDB(_RetryDB):
    def __init__(self):
        self.batches = {}
        self.finished = threading.Event()

    def get_editor_diagnostics(self, _translation_id):
        return {
            "classification": "current",
            "summary": {},
            "runs": [{
                "id": 9,
                "chunk_index": 0,
                "outcome": "review_required",
            }],
            "current_review_queue": [{
                "chunk_index": 0,
                "phase": "translation",
                "retryable": True,
                "reason_codes": ["review_required"],
                "outcome": "review_required",
            }],
        }

    def get_active_refinement_results(self, _translation_id):
        return []

    def find_active_editor_repair_batch(self, translation_id):
        for batch in reversed(list(self.batches.values())):
            if (batch["translation_id"] == translation_id
                    and batch["status"] in {"queued", "pausing", "running"}):
                return deepcopy(batch)
        return None

    def create_editor_repair_batch(
        self, batch_id, translation_id, scope, phase, chunk_indices,
        *, stay_paused=True,
    ):
        self.batches[batch_id] = {
            "batch_id": batch_id,
            "translation_id": translation_id,
            "scope": scope,
            "phase": phase,
            "status": "queued",
            "stay_paused": stay_paused,
            "cancel_requested": False,
            "total_items": len(chunk_indices),
            "completed_items": 0,
            "succeeded_items": 0,
            "failed_items": 0,
            "error": None,
            "items": [{
                "chunk_index": index,
                "phase": phase,
                "status": "queued",
            } for index in chunk_indices],
        }
        return True

    def update_editor_repair_batch(self, batch_id, **changes):
        self.batches[batch_id].update(changes)
        if changes.get("status") in {"completed", "failed", "cancelled"}:
            self.finished.set()
        return True

    def update_editor_repair_batch_item(
        self, batch_id, chunk_index, phase, **changes,
    ):
        item = next(
            item for item in self.batches[batch_id]["items"]
            if item["chunk_index"] == chunk_index and item["phase"] == phase
        )
        item.update(changes)
        return True

    def get_editor_repair_batch(self, batch_id):
        batch = self.batches.get(batch_id)
        return deepcopy(batch) if batch else None

    def get_latest_editor_repair_batch(self, translation_id):
        matches = [
            batch for batch in self.batches.values()
            if batch["translation_id"] == translation_id
        ]
        return deepcopy(matches[-1]) if matches else None


class _SocketRecorder:
    def __init__(self):
        self.events = []
        self.resumed = threading.Event()

    def emit(self, event, payload, namespace=None):
        self.events.append((event, deepcopy(payload), namespace))
        if payload.get("editor_repair_batch", {}).get("event") == "resumed":
            self.resumed.set()


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


def test_bulk_repair_reports_live_activity_and_remains_paused(
    monkeypatch,
    tmp_path,
    capsys,
):
    preserved = tmp_path / "source.txt"
    preserved.write_text("Source", encoding="utf-8")
    checkpoints = _RetryCheckpoints(preserved)
    checkpoints.db = _BatchDB()
    state = _RetryState(checkpoints)
    socketio = _SocketRecorder()

    async def fake_retry(**_kwargs):
        return {"status": "succeeded", "outcome": "warnings_only"}

    async def fake_refresh(*_args, **_kwargs):
        return {"status": "updated"}

    monkeypatch.setattr("src.core.editor_retry.run_editor_retry", fake_retry)
    monkeypatch.setattr("src.core.editor_retry._refresh_output", fake_refresh)
    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state,
        lambda *_args: None,
        str(tmp_path),
        socketio=socketio,
    ))

    response = app.test_client().post(
        "/api/translation/job-1/editor-repair-batches",
        json={"scope": "review_required", "stay_paused": True},
    )

    assert response.status_code == 202
    assert checkpoints.db.finished.wait(2)
    batch_id = response.get_json()["batch_id"]
    batch = checkpoints.db.get_editor_repair_batch(batch_id)
    assert batch["status"] == "completed"
    assert batch["succeeded_items"] == 1
    assert state.job["status"] == "interrupted"
    stored_events = [
        entry["data"]["editor_repair_batch"]["event"]
        for entry in state.job.get("logs", [])
        if entry.get("data", {}).get("editor_repair_batch")
    ]
    assert "item_started" in stored_events
    assert "completed" in stored_events
    socket_events = [
        payload["editor_repair_batch"]["event"]
        for _event, payload, _namespace in socketio.events
        if payload.get("editor_repair_batch")
    ]
    assert "item_started" in socket_events
    assert "completed" in socket_events
    assert "is repairing chunk 1" in capsys.readouterr().out

    diagnostics = app.test_client().get(
        "/api/translation/job-1/editor-diagnostics"
    ).get_json()
    assert diagnostics["active_repair_batch"] is None
    assert diagnostics["latest_repair_batch"]["status"] == "completed"


def test_successful_review_batch_auto_resumes_when_requested(
    monkeypatch,
    tmp_path,
):
    preserved = tmp_path / "source.txt"
    preserved.write_text("Source", encoding="utf-8")
    checkpoints = _RetryCheckpoints(preserved)
    checkpoints.db = _BatchDB()
    state = _RetryState(checkpoints)
    socketio = _SocketRecorder()
    resumed = threading.Event()

    async def fake_retry(**_kwargs):
        return {"status": "succeeded", "outcome": "warnings_only"}

    async def fake_refresh(*_args, **_kwargs):
        return {"status": "updated"}

    monkeypatch.setattr("src.core.editor_retry.run_editor_retry", fake_retry)
    monkeypatch.setattr("src.core.editor_retry._refresh_output", fake_refresh)
    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state,
        lambda *_args: resumed.set(),
        str(tmp_path),
        socketio=socketio,
    ))

    response = app.test_client().post(
        "/api/translation/job-1/editor-repair-batches",
        json={
            "scope": "review_required",
            "stay_paused": False,
        },
    )

    assert response.status_code == 202
    assert resumed.wait(2)
    assert socketio.resumed.wait(2)
    assert state.job["status"] == "running"
    batch = checkpoints.db.get_latest_editor_repair_batch("job-1")
    assert batch["status"] == "completed"
    assert batch["stay_paused"] is False
    assert batch["error"] is None
