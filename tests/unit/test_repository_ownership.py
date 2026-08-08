"""The job and editor repositories must own their SQL, not forward to Database.

`DatabaseRepository.__getattr__` makes forwarding invisible: `db.jobs.get_job`
resolves either way, so a regression that moved a body back onto the facade
would not fail any behavioural test. These assertions pin the direction of the
delegation instead -- the method must be defined on the repository class, and
`Database`'s copy must be the one-line forwarder.
"""

import inspect

from src.persistence.database import Database
from src.persistence.repositories import (
    ContextRepository,
    EditorRepository,
    JobRepository,
    NarratorRepository,
)

JOB_METHODS = sorted(JobRepository.methods)
EDITOR_METHODS = sorted(EditorRepository.methods)


def test_job_repository_defines_its_own_methods():
    for name in JOB_METHODS:
        assert name in vars(JobRepository), f"JobRepository no longer defines {name}"


def test_editor_repository_defines_its_own_methods():
    for name in EDITOR_METHODS:
        assert name in vars(EditorRepository), f"EditorRepository no longer defines {name}"


def test_database_delegates_instead_of_holding_the_sql():
    for name, attr in [(n, "jobs") for n in JOB_METHODS] + [
        (n, "editor") for n in EDITOR_METHODS
    ]:
        source = inspect.getsource(getattr(Database, name))
        assert f"self.{attr}.{name}(" in source, f"Database.{name} stopped delegating"
        assert "cursor" not in source, f"Database.{name} runs SQL again"


def test_owned_methods_are_reached_without_the_forwarder(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    try:
        for name in JOB_METHODS:
            assert getattr(db.jobs, name).__qualname__ == f"JobRepository.{name}"
        for name in EDITOR_METHODS:
            assert getattr(db.editor, name).__qualname__ == f"EditorRepository.{name}"
    finally:
        db.close_all()


def test_context_and_narrator_still_forward():
    # Documented as unmigrated in docs/ARCHITECTURE.md. If one of these grows a
    # real body, move it out of this assertion and into the ownership tests.
    for repo in (ContextRepository, NarratorRepository):
        assert not [
            name for name in repo.methods if name in vars(repo)
        ], f"{repo.__name__} now owns SQL -- update ARCHITECTURE.md and this test"
