"""Guards on the PRAGMA user_version stamp that gates the DDL sweep.

Skipping the DDL for a stamped file is only safe while the stamp genuinely
means "this file has every statement the current build creates". Two ways that
breaks, both covered here: a statement added to the DDL without bumping
SCHEMA_VERSION, and a data repair drifting into the skipped half.
"""

import hashlib
import inspect
import sqlite3

from src.persistence import schema
from src.persistence.database import Database

# Recomputed by hand when the DDL legitimately changes -- and the change must
# come with a SCHEMA_VERSION bump in the same commit, or existing databases
# silently skip the new statement.
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_DDL_DIGEST = (
    "abed48a016df9bb0a128d73f7b91cb100ec7eb56fc5ccc8cf9d7ac25baef5285"
)


def _ddl_digest():
    source = inspect.getsource(schema._create_tables_and_migrate_columns)
    return hashlib.sha256(source.encode()).hexdigest()


def test_ddl_changes_require_a_version_bump():
    assert schema.SCHEMA_VERSION == EXPECTED_SCHEMA_VERSION, (
        "SCHEMA_VERSION changed -- update EXPECTED_SCHEMA_VERSION and "
        "EXPECTED_DDL_DIGEST here in the same commit."
    )
    assert _ddl_digest() == EXPECTED_DDL_DIGEST, (
        "_create_tables_and_migrate_columns changed without bumping "
        "SCHEMA_VERSION. Databases stamped with the old version will skip the "
        "new statement and be left missing the table or column."
    )


def test_a_fresh_database_is_stamped(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    try:
        stamp = db._get_connection().execute("PRAGMA user_version").fetchone()[0]
        assert stamp == schema.SCHEMA_VERSION
    finally:
        db.close_all()


def test_the_ddl_sweep_is_skipped_for_a_stamped_file(tmp_path, monkeypatch):
    path = str(tmp_path / "jobs.db")
    Database(path).close_all()

    calls = []
    real = schema._create_tables_and_migrate_columns
    monkeypatch.setattr(
        schema,
        "_create_tables_and_migrate_columns",
        lambda cursor: (calls.append(1), real(cursor))[1],
    )
    Database(path).close_all()
    assert calls == [], "the DDL ran again for an already-stamped file"


def test_the_repairs_run_even_for_a_stamped_file(tmp_path, monkeypatch):
    # The repairs fix rows the current version still writes, so unlike the DDL
    # they must not be gated on the stamp.
    path = str(tmp_path / "jobs.db")
    Database(path).close_all()

    calls = []
    real = schema._repair_data
    monkeypatch.setattr(
        schema, "_repair_data", lambda cursor: (calls.append(1), real(cursor))[1]
    )
    Database(path).close_all()
    assert calls == [1]


def test_an_unstamped_file_still_gets_the_ddl(tmp_path):
    # This is the upgrade path: every database written before the stamp
    # existed reports user_version 0.
    path = tmp_path / "jobs.db"
    Database(str(path)).close_all()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE editor_runs")
        conn.execute("PRAGMA user_version = 0")

    db = Database(str(path))
    try:
        tables = {
            row[0] for row in db._get_connection().execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "editor_runs" in tables
    finally:
        db.close_all()
