"""Gates on the extracted SQLite schema module."""
import sqlite3

from src.persistence.database import Database
from src.persistence.schema import apply_schema


def test_schema_module_owns_the_ddl():
    """database.py delegates; it no longer carries the statements itself."""
    import inspect

    from src.persistence import database

    source = inspect.getsource(database.Database._initialize_schema)
    assert "apply_schema" in source
    assert "CREATE TABLE" not in source


def test_fresh_database_is_fully_built(tmp_path):
    db_path = tmp_path / "nested" / "state.db"
    Database(str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "translation_jobs" in tables
    assert "checkpoint_chunks" in tables
    assert len(tables) > 20


def test_apply_schema_is_idempotent(tmp_path):
    """Reopening an existing database must neither fail nor change its shape."""
    db_path = tmp_path / "state.db"
    Database(str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        before = sorted(
            row[0] for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
            )
        )

    Database(str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        after = sorted(
            row[0] for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
            )
        )
    assert before == after


def test_apply_schema_runs_against_a_plain_connection(tmp_path):
    """The extracted function depends on a lock and a factory, not on Database."""
    import threading

    db_path = tmp_path / "standalone.db"
    with sqlite3.connect(str(db_path)) as conn:
        apply_schema(threading.RLock(), lambda: conn)
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "translation_jobs" in tables


def test_repository_facades_are_cached(tmp_path):
    db = Database(str(tmp_path / "state.db"))
    assert db.jobs is db.jobs
    assert db.editor is db.editor
    assert db.context is db.context
    assert db.narrator is db.narrator
