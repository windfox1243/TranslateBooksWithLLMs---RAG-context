"""commit_pending_context: the post-translation context commit every pipeline shares."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.core.context.commit import commit_pending_context, save_committed_context


class _FakeDb:
    def __init__(self):
        self.depth = 0
        self.max_depth = 0

    @contextmanager
    def context_state_transaction(self):
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        try:
            yield
        finally:
            self.depth -= 1


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def record(name):
        def fn(*args, **kwargs):
            db = kwargs.get("db") if "db" in kwargs else (args[2] if len(args) > 2 else None)
            seen.append((name, getattr(db, "depth", None)))
        return fn

    monkeypatch.setattr("src.utils.db_addressing.sync_context_update_addressing_to_db", record("addressing"))
    monkeypatch.setattr("src.utils.db_addressing.apply_db_addressing_to_session", record("apply_addressing"))
    monkeypatch.setattr("src.utils.relationship_sync.sync_context_update_relationships_to_db", record("relationships"))
    monkeypatch.setattr("src.utils.relationship_sync.apply_relationship_graph_to_session", record("apply_relationships"))
    return seen


def _commit(fake_db, **overrides):
    kwargs = dict(
        context_session=object(),
        db=fake_db,
        translation_id="job",
        chunk_index=3,
        addressing={"content": "A"},
        relationships={"content": "R", "source_text": "S"},
        target_language="Vietnamese",
        source_language="English",
        relationship_mode="selective",
    )
    kwargs.update(overrides)
    return commit_pending_context(**kwargs)


def test_both_updates_are_written_in_one_transaction_then_applied(calls):
    db = _FakeDb()
    assert _commit(db) is True
    assert calls == [
        ("addressing", 1),
        ("relationships", 1),
        ("apply_addressing", 0),
        ("apply_relationships", 0),
    ]


def test_nests_inside_a_caller_transaction(calls):
    db = _FakeDb()
    with db.context_state_transaction():
        _commit(db)
    assert db.max_depth == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_session": None},
        {"db": None},
        {"translation_id": ""},
        {"addressing": None, "relationships": None},
        {"addressing": None, "relationship_mode": "off"},
        {"relationships": None, "use_db_addressing": False},
    ],
)
def test_nothing_is_written_when_there_is_nothing_to_commit(calls, overrides):
    assert _commit(_FakeDb(), **overrides) is False
    assert calls == []


def test_relationship_mode_off_skips_relationships_only(calls):
    _commit(_FakeDb(), relationship_mode="off")
    assert [name for name, _ in calls] == ["addressing", "apply_addressing"]


def test_save_writes_the_session_and_emits_the_state(tmp_path):
    saved = []
    session = SimpleNamespace(
        save=lambda: saved.append(True),
        content="CTX",
        path=tmp_path / "ctx.md",
    )
    events = []
    save_committed_context(session, log_callback=lambda *a, **k: events.append(a))
    assert saved == [True]
    assert events and events[0][0] == "novel_context_state"
