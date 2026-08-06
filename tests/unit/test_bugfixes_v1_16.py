"""Regression tests for the correctness fixes shipped in v1.16.0.

Each test pins one previously-broken behaviour:

* Upload directories are removed only for jobs the database actually deleted
  (they used to be derived from a local-time cutoff compared against UTC).
* Database closes every thread's SQLite connection, not just the caller's.
* Completed jobs are evicted from the in-memory state manager.
* Log storage goes through the atomic append.
* Concurrent translation jobs are capped.
"""

import json
import threading

import pytest

from src.persistence.database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(db_path=str(tmp_path / "db" / "test.db"))
    yield database
    database.close_all()


def _make_job(database, translation_id, age_days):
    """Insert a job whose created_at is age_days old in UTC."""
    conn = database._get_connection()
    conn.execute(
        """
        INSERT INTO translation_jobs
            (translation_id, status, file_type, config, progress, created_at)
        VALUES (?, 'completed', 'txt', ?, ?, datetime('now', ? || ' days'))
        """,
        (translation_id, json.dumps({}), json.dumps({}), f"-{age_days}"),
    )
    conn.commit()


class TestOldJobCleanup:
    def test_delete_old_jobs_returns_deleted_ids(self, db):
        _make_job(db, "old_job", age_days=40)
        _make_job(db, "fresh_job", age_days=1)

        assert db.delete_old_jobs(max_age_days=30) == ["old_job"]

    def test_cleanup_old_jobs_still_returns_a_count(self, db):
        _make_job(db, "old_job", age_days=40)

        assert db.cleanup_old_jobs(max_age_days=30) == 1

    def test_uploads_removed_only_for_jobs_the_database_deleted(self, tmp_path, monkeypatch):
        """The upload sweep must follow the database, not a second cutoff.

        The old implementation recomputed the cutoff with datetime.now() (local
        time) and compared it to created_at (UTC). East of UTC that made the
        Python cutoff later than the SQL one, so uploads were destroyed for
        jobs the database kept, and those jobs could never resume.
        """
        import src.config as cfg

        uploads_dir = tmp_path / "uploads"
        monkeypatch.setattr(cfg, "UPLOADS_DIR", uploads_dir)

        from src.persistence.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(db_path=str(tmp_path / "db" / "test.db"))
        database = manager.db
        try:
            assert manager.uploads_dir == uploads_dir

            _make_job(database, "old_job", age_days=40)
            # Recent enough for the database to keep, but old enough that a
            # cutoff skewed by a local UTC offset would have swept it.
            _make_job(database, "borderline_job", age_days=29)
            _make_job(database, "fresh_job", age_days=1)

            for job_id in ("old_job", "borderline_job", "fresh_job"):
                job_dir = manager.uploads_dir / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                (job_dir / "source.txt").write_text("payload", encoding="utf-8")

            jobs_deleted, files_cleaned = manager.cleanup_old_jobs(max_age_days=30)

            assert jobs_deleted == 1
            assert files_cleaned == 1
            assert not (manager.uploads_dir / "old_job").exists()
            # Kept by the database => its uploads must survive.
            assert (manager.uploads_dir / "borderline_job").exists()
            assert (manager.uploads_dir / "fresh_job").exists()
        finally:
            database.close_all()


class TestConnectionLifecycle:
    def test_close_all_reclaims_connections_from_worker_threads(self, db):
        """Worker threads never call close(); close_all() must still reclaim."""
        errors = []
        before = len(db._all_connections)

        def worker():
            try:
                db._get_connection().execute("SELECT 1").fetchone()
            except Exception as exc:  # pragma: no cover - surfaced via assert
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert len(db._all_connections) == before + 5

        db.close_all()

        assert db._all_connections == []

    def test_close_drops_only_the_calling_threads_connection(self, db):
        db._get_connection()
        before = len(db._all_connections)

        db.close()

        assert len(db._all_connections) == before - 1


class TestStateManagerEviction:
    def test_cleanup_completed_job_evicts_in_memory_entry(self):
        from src.api.translation_state import TranslationStateManager

        manager = TranslationStateManager()
        try:
            manager.create_translation("job_1", {"file_path": "x.txt"})
            assert manager.exists("job_1")

            manager.cleanup_completed_job("job_1")

            assert not manager.exists("job_1")
        finally:
            manager.close_database()

    def test_append_log_is_atomic_under_concurrency(self):
        from src.api.translation_state import TranslationStateManager

        manager = TranslationStateManager()
        try:
            manager.create_translation("job_2", {})
            initial = len(manager.get_translation_field("job_2", "logs"))

            def worker(index):
                for i in range(20):
                    manager.append_log("job_2", f"{index}-{i}")

            threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            logs = manager.get_translation_field("job_2", "logs")
            assert len(logs) == initial + 80
        finally:
            manager.close_database()


class TestJobConcurrencyCap:
    def test_job_slots_reflects_configured_maximum(self):
        import src.config as cfg
        from src.api import handlers

        handlers._job_slots = None
        try:
            slots = handlers._get_job_slots()
            acquired = 0
            while slots.acquire(blocking=False):
                acquired += 1
            for _ in range(acquired):
                slots.release()
            assert acquired == cfg.MAX_CONCURRENT_JOBS
        finally:
            handlers._job_slots = None
