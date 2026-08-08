"""Regressions for the whole-job loss two sessions on one context file used to cause.

Making a single write atomic was not enough. A session reads the context file
once when it opens and writes the whole document back at the end of every chunk,
so two jobs sharing a file each replaced the other's entire run -- not one lost
edit, all of them.
"""

from src.utils.novel_context import (
    build_novel_context,
    load_novel_context,
    open_novel_context_session,
    reconcile_context_documents,
    save_novel_context,
    save_novel_context_merged,
)


def _document(*character_lines: str, addressing: str = "") -> str:
    characters = "\n".join(f"- {line}" for line in character_lines)
    return build_novel_context(
        f"# GLOBAL LORE\n\n## CHARACTERS & GENDERS\n{characters}\n",
        f"## CURRENT ADDRESSING FORMS\n{addressing}\n" if addressing else "",
    )


# --- The merge itself ---------------------------------------------------------


def test_reconciling_keeps_characters_from_both_documents():
    merged, change_logs = reconcile_context_documents(
        _document("Alice: female"),
        _document("Bob: male"),
    )
    assert "Alice" in merged
    assert "Bob" in merged
    assert change_logs


def test_reconciling_identical_documents_changes_nothing():
    document = _document("Alice: female")
    merged, change_logs = reconcile_context_documents(document, document)
    assert "Alice" in merged
    assert not change_logs


def test_reconciling_keeps_dynamic_state_from_both_documents():
    merged, _ = reconcile_context_documents(
        _document("Alice: female", addressing="- Alice → Bob: formal"),
        _document("Alice: female", addressing="- Carol → Dave: casual"),
    )
    assert "Alice" in merged and "Bob" in merged
    assert "Carol" in merged and "Dave" in merged


# --- The save that uses it ----------------------------------------------------


def test_a_save_over_an_untouched_file_writes_exactly_what_it_was_given(tmp_path):
    baseline = _document("Alice: female")
    save_novel_context("novel.txt", tmp_path, baseline)

    mine = _document("Alice: female", addressing="- Alice → Bob: formal")
    written, change_logs = save_novel_context_merged(
        "novel.txt", tmp_path, mine, baseline
    )

    # Nothing moved underneath, so this is the plain write it always was.
    assert not change_logs
    assert "Bob" in written
    assert "Bob" in load_novel_context("novel.txt", tmp_path)


def test_a_save_over_another_jobs_work_keeps_both(tmp_path):
    baseline = _document("Alice: female")
    save_novel_context("novel.txt", tmp_path, baseline)

    # Another job wrote to the file after this one read its baseline.
    save_novel_context("novel.txt", tmp_path, _document("Alice: female", "Bob: male"))

    written, change_logs = save_novel_context_merged(
        "novel.txt",
        tmp_path,
        _document("Alice: female", "Carol: female"),
        baseline,
    )

    assert change_logs
    on_disk = load_novel_context("novel.txt", tmp_path)
    for name in ("Alice", "Bob", "Carol"):
        assert name in written, name
        assert name in on_disk, name


def test_an_unreadable_file_still_saves(tmp_path):
    # The read only exists to detect the merge case. Losing it must not lose
    # the write -- that would turn a rare concurrency case into a common one.
    written, change_logs = save_novel_context_merged(
        "novel.txt", tmp_path, _document("Alice: female"), "stale baseline"
    )
    assert not change_logs
    assert "Alice" in load_novel_context("novel.txt", tmp_path)
    assert "Alice" in written


# --- The session that drives it -----------------------------------------------


def _session(tmp_path):
    return open_novel_context_session(
        {"novel_context_file": "novel.txt", "target_language": "English"},
        tmp_path,
    )


def _learn(session, character_line):
    """Add a character the way a chunk update would: inside the right section."""
    session.global_lore = session.global_lore.replace(
        "## CHARACTERS & GENDERS\n",
        f"## CHARACTERS & GENDERS\n- {character_line}\n",
    )


def test_two_sessions_on_one_file_do_not_erase_each_other(tmp_path):
    save_novel_context("novel.txt", tmp_path, _document("Alice: female"))

    first = _session(tmp_path)
    second = _session(tmp_path)
    assert first is not None and second is not None

    _learn(first, "Bob: male")
    _learn(second, "Carol: female")

    # The second session opened before the first one saved, so its in-memory
    # document knows nothing about Bob. It used to overwrite him regardless.
    first.save()
    second.save()

    on_disk = load_novel_context("novel.txt", tmp_path)
    for name in ("Alice", "Bob", "Carol"):
        assert name in on_disk, name


def test_a_session_adopts_what_it_merged(tmp_path):
    save_novel_context("novel.txt", tmp_path, _document("Alice: female"))

    first = _session(tmp_path)
    second = _session(tmp_path)
    _learn(first, "Bob: male")
    first.save()

    _learn(second, "Carol: female")
    second.save()

    # Adopting the merge is what stops the next chunk from proposing state the
    # file has already moved past.
    assert "Bob" in second.global_lore
    assert "Bob" in second.prompt_options["novel_context"]


def test_a_sessions_baseline_tracks_its_own_writes(tmp_path):
    save_novel_context("novel.txt", tmp_path, _document("Alice: female"))
    session = _session(tmp_path)
    assert session is not None

    _learn(session, "Bob: male")
    session.save()
    _learn(session, "Carol: female")
    written, change_logs = save_novel_context_merged(
        "novel.txt",
        tmp_path,
        session.content,
        session.disk_baseline,
        "English",
    )

    # A session that saves twice in a row is not concurrent with itself.
    assert not change_logs
    assert "Carol" in written
