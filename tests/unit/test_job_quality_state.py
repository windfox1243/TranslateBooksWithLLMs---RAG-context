import sqlite3

from src.persistence.database import Database
from src.core.jobs import UnitTranslationOutcome
from src.persistence.checkpoint_manager import CheckpointManager


def test_unit_outcome_separates_execution_failure_from_review_state():
    outcome = UnitTranslationOutcome.review_required(
        "Usable translated draft",
        {"reason_codes": ["editor_transport"]},
    )

    assert outcome.is_completed is True
    assert outcome.execution_status == "completed"
    assert outcome.quality_status == "review_required"
    assert outcome.execution_failure_class is None


def test_checkpoint_aggregates_review_required_without_marking_failure(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.start_job("job", "txt", {}, None)

    assert manager.save_checkpoint(
        translation_id="job",
        chunk_index=0,
        original_text="Source",
        translated_text="Draft",
        chunk_data={
            "quality_status": "review_required",
            "editor_validation": {"outcome": "review_required"},
        },
        total_chunks=1,
        completed_chunks=1,
        failed_chunks=0,
    )

    checkpoint = manager.load_checkpoint("job")
    assert checkpoint["job"]["quality_status"] == "review_required"
    assert checkpoint["job"]["progress"]["review_required_chunks"] == 1
    assert checkpoint["chunks"][0]["status"] == "completed"
    assert checkpoint["chunks"][0]["quality_status"] == "review_required"
    assert checkpoint["chunks"][0]["translated_text"] == "Draft"


def test_job_quality_stays_passed_when_a_later_unit_fails(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.start_job("job", "txt", {}, None)
    manager.db.save_chunk("job", 0, "A", "B", status="completed")
    manager.db.save_chunk(
        "job",
        1,
        "C",
        None,
        status="failed",
        execution_failure_class="provider_rate_limit",
    )

    assert manager.db.get_job("job")["quality_status"] == "passed"
    failed = manager.db.get_chunks("job")[1]
    assert failed["quality_status"] == "not_checked"
    assert failed["execution_failure_class"] == "provider_rate_limit"


def test_existing_database_receives_additive_quality_columns(tmp_path):
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE translation_jobs (
            translation_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            file_type TEXT NOT NULL,
            config JSON NOT NULL,
            progress JSON NOT NULL
        );
        CREATE TABLE checkpoint_chunks (
            translation_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            original_text TEXT NOT NULL,
            translated_text TEXT,
            chunk_data JSON,
            status TEXT NOT NULL,
            completed_at TIMESTAMP,
            PRIMARY KEY (translation_id, chunk_index)
        );
        """
    )
    connection.close()

    database = Database(str(db_path))
    connection = database._get_connection()
    job_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(translation_jobs)")
    }
    chunk_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(checkpoint_chunks)")
    }

    assert "quality_status" in job_columns
    assert {"quality_status", "execution_failure_class"} <= chunk_columns
    database.close()
