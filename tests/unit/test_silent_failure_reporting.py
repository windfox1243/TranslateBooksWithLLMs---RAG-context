"""Failures that degrade output must be reported, not swallowed.

Each test here pins one handler that used to be ``except ...: pass``. The
common shape of the bug was that the program kept running with worse data --
stale context, unprotected proper names, dropped aliases -- and nothing in the
logs said so, which is exactly what makes such a regression survive a release.
"""

import asyncio
import logging
import threading

import pytest

from src.core.editor.preflight import run_editor_preflight
from src.persistence.database import Database


def _collect(log):
    """Return a log_callback that appends (event, message) to ``log``."""

    def callback(event, message, *args, **kwargs):
        log.append((event, message))

    return callback


def test_preflight_reports_an_incomplete_protected_term_set(monkeypatch):
    """A broken novel context must not silently unprotect proper names."""
    import src.utils.novel_context as novel_context_module

    def explode(*args, **kwargs):
        raise ValueError("corrupt lore section")

    monkeypatch.setattr(novel_context_module, "extract_global_lore", explode)

    events = []
    result = run_editor_preflight(
        "The Duke greeted Mira.",
        "Le Duc salua Mira.",
        "French",
        "## Global Lore\n- Mira: a scholar\n",
        {"source_language": "English"},
        log_callback=_collect(events),
    )

    assert result is not None, "the pre-flight must still produce a result"
    assert any(
        event == "editor_protected_terms_incomplete" for event, _ in events
    ), f"no warning was emitted; got {[e for e, _ in events]}"


def test_preflight_still_reports_without_a_callback(monkeypatch):
    """The warning path must not itself raise when no callback is wired."""
    import src.utils.novel_context as novel_context_module

    monkeypatch.setattr(
        novel_context_module,
        "extract_global_lore",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")),
    )

    result = run_editor_preflight(
        "Hello.", "Bonjour.", "French", "## Global Lore\n", {}, log_callback=None
    )
    assert result is not None


def test_failed_resync_does_not_run_a_second_time(monkeypatch):
    """A RuntimeError from the resync must propagate, not trigger a re-run.

    The handler used to wrap ``future.result()`` as well as loop acquisition,
    so a RuntimeError raised inside the resync fell through to the
    ``asyncio.run`` fallback and replayed the entire pass -- including its
    database writes.
    """
    from src.core.adapters import generic_translator

    calls = []

    async def failing_resync(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("resync failed")

    monkeypatch.setattr(
        generic_translator,
        "_resync_context_snapshots_async",
        failing_resync,
    )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    # Wait for the loop to actually be running before handing work to it.
    ready = threading.Event()
    loop.call_soon_threadsafe(ready.set)
    assert ready.wait(timeout=5), "the test event loop never started"

    monkeypatch.setattr(asyncio, "get_event_loop", lambda: loop)
    try:
        with pytest.raises(RuntimeError, match="resync failed"):
            generic_translator.resync_context_snapshots_background(
                "job-1", 0, "{}"
            )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert len(calls) == 1, f"the resync ran {len(calls)} times, expected once"


def test_unreadable_stored_aliases_are_reported(tmp_path, caplog):
    """Overwriting unreadable aliases must leave a trace in the log."""
    db = Database(str(tmp_path / "jobs.db"))
    try:
        db.upsert_relationship_node("job-1", "Mira", "mira", aliases=["Mi"])
        with db._get_connection() as conn:
            conn.execute(
                "UPDATE context_relationship_nodes SET aliases = ? "
                "WHERE translation_id = ? AND normalized_name = ?",
                ("{not json", "job-1", "mira"),
            )
            conn.commit()

        with caplog.at_level(logging.WARNING, logger="persistence.database"):
            db.upsert_relationship_node("job-1", "Mira", "mira", aliases=["Mira A."])

        assert any(
            "unreadable stored aliases" in record.getMessage()
            for record in caplog.records
        ), f"no warning logged; got {[r.getMessage() for r in caplog.records]}"
    finally:
        db.close_all()
