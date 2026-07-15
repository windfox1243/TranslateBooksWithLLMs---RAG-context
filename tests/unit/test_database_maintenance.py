"""Checkpoint deletion, orphan recovery, and explicit compaction."""

from pathlib import Path
from types import SimpleNamespace
import sqlite3

from flask import Flask

from src.api.blueprints.translation_routes import create_translation_blueprint
from src.persistence.database import Database


def _insert_chunk(db, translation_id, text="source"):
    conn = db._get_connection()
    conn.execute(
        "INSERT INTO checkpoint_chunks "
        "(translation_id,chunk_index,original_text,status) VALUES (?,?,?,?)",
        (translation_id, 0, text, "completed"),
    )
    conn.commit()


def test_delete_job_explicitly_removes_chunks_without_foreign_keys(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    assert db.create_job("job-1", "txt", {})
    _insert_chunk(db, "job-1")
    conn = db._get_connection()
    conn.execute("PRAGMA foreign_keys=OFF")
    assert db.delete_job("job-1")
    assert conn.execute(
        "SELECT count(*) FROM checkpoint_chunks WHERE translation_id='job-1'"
    ).fetchone()[0] == 0


def test_purge_orphans_keeps_current_job(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    assert db.create_job("current", "txt", {})
    _insert_chunk(db, "current", "keep")
    conn = db._get_connection()
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(
        "INSERT INTO checkpoint_chunks "
        "(translation_id,chunk_index,original_text,status) VALUES (?,?,?,?)",
        ("orphan", 0, "remove", "completed"),
    )
    conn.commit()
    stats = db.purge_orphan_rows()
    assert stats["translation_ids"] == 1
    assert conn.execute("SELECT count(*) FROM checkpoint_chunks").fetchone()[0] == 1
    assert conn.execute(
        "SELECT original_text FROM checkpoint_chunks"
    ).fetchone()[0] == "keep"


def test_optimize_creates_backup_and_passes_integrity_check(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(str(db_path))
    assert db.create_job("current", "txt", {})
    _insert_chunk(db, "current")
    result = db.optimize_database()
    assert result["integrity"] == "ok"
    assert Path(result["backup_path"]).is_file()
    assert result["before_bytes"] > 0
    assert result["after_bytes"] > 0


def test_compaction_endpoint_refuses_active_job(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    state = SimpleNamespace(
        checkpoint_manager=SimpleNamespace(db=db),
        get_all_translations=lambda: {"job-1": {"status": "running"}},
    )
    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state, lambda *_args, **_kwargs: None, str(tmp_path),
    ))
    response = app.test_client().post('/api/maintenance/jobs-db/compact', json={})
    assert response.status_code == 409
    assert response.get_json()["active_jobs"] == ["job-1"]


def test_editor_diagnostics_persist_thinking_token_breakdown(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    assert db.create_job("job-1", "txt", {})
    run_id = db.create_editor_run({
        "translation_id": "job-1",
        "chunk_index": 0,
        "phase": "translation",
        "outcome": "running",
    })
    assert db.add_editor_attempt(run_id, {
        "attempt_index": 1,
        "stage": "reflection",
        "prompt_tokens": 100,
        "completion_tokens": 60,
        "thinking_tokens": 3936,
        "total_tokens": 4096,
    })
    assert db.finish_editor_run(run_id, {
        "outcome": "no_issues",
        "prompt_tokens": 100,
        "completion_tokens": 60,
        "thinking_tokens": 3936,
        "total_tokens": 4096,
    })
    conn = db._get_connection()
    attempt = conn.execute(
        "SELECT thinking_tokens,total_tokens FROM editor_attempts"
    ).fetchone()
    run = conn.execute(
        "SELECT thinking_tokens,total_tokens FROM editor_runs"
    ).fetchone()
    assert tuple(attempt) == (3936, 4096)
    assert tuple(run) == (3936, 4096)


def test_editor_token_columns_migrate_existing_database(tmp_path):
    path = tmp_path / "jobs.db"
    db = Database(str(path))
    db.close()
    with sqlite3.connect(path) as conn:
        for table in ("editor_runs", "editor_attempts"):
            conn.execute(f"ALTER TABLE {table} DROP COLUMN thinking_tokens")
            conn.execute(f"ALTER TABLE {table} DROP COLUMN total_tokens")
    migrated = Database(str(path))
    for table in ("editor_runs", "editor_attempts"):
        columns = {
            row[1] for row in migrated._get_connection().execute(
                f"PRAGMA table_info({table})"
            )
        }
        assert {"thinking_tokens", "total_tokens"} <= columns


def test_editor_repair_batch_persists_item_progress(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    assert db.create_job("job", "txt", {})
    assert db.create_editor_repair_batch(
        "batch", "job", "review_required", "effective", [2, 4]
    )
    assert db.update_editor_repair_batch_item(
        "batch", 2, "effective", status="succeeded", outcome="locally_repaired"
    )
    assert db.update_editor_repair_batch(
        "batch", status="running", completed_items=1, succeeded_items=1
    )
    result = db.get_editor_repair_batch("batch")
    assert result["completed_items"] == 1
    assert result["items"][0]["status"] == "succeeded"


def test_latest_editor_repair_batch_returns_completed_batch(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    assert db.create_editor_repair_batch(
        "older", "job", "review_required", "effective", [1]
    )
    assert db.update_editor_repair_batch("older", status="completed")
    assert db.create_editor_repair_batch(
        "newer", "job", "review_required", "effective", [2]
    )

    result = db.get_latest_editor_repair_batch("job")

    assert result["batch_id"] == "newer"
    assert result["items"][0]["chunk_index"] == 2


def test_refinement_pass_promotes_only_complete_exact_results(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    assert db.create_job("job", "txt", {})
    assert db.create_refinement_pass("pass", "job", expected_units=2)
    for index in range(2):
        assert db.save_refinement_chunk_result(
            "pass", "job", index, base_chunk_index=index,
            source_text=f"source {index}", refined_text=f"refined {index}",
            status="completed", quality_status="passed",
        )
    assert db.finish_refinement_pass("pass", successful=True)
    assert [item["refined_text"] for item in db.get_active_refinement_results("job")] == [
        "refined 0", "refined 1"
    ]
