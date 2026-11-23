"""Stored job logs are bounded and compact (ported from upstream ed9a8c14)."""

from src.api import translation_state
from src.api.job_callbacks import _STORED_LOG_FIELD_MAX_CHARS, _compact_log_entry
from src.api.translation_state import MAX_STORED_LOG_ENTRIES, _append_bounded


def test_log_window_keeps_the_newest_entries():
    logs = []
    for i in range(MAX_STORED_LOG_ENTRIES + 7):
        _append_bounded(logs, i)
    assert len(logs) == MAX_STORED_LOG_ENTRIES
    assert logs[0] == 7 and logs[-1] == MAX_STORED_LOG_ENTRIES + 6


def test_state_manager_append_is_bounded():
    manager = translation_state.TranslationStateManager()
    manager.create_translation("job", {})
    for i in range(MAX_STORED_LOG_ENTRIES + 3):
        manager.append_log("job", f"line {i}")
    assert len(manager.get_translation("job")["logs"]) == MAX_STORED_LOG_ENTRIES


def test_long_data_strings_are_truncated_in_the_stored_copy():
    prompt = "x" * (_STORED_LOG_FIELD_MAX_CHARS + 50)
    entry = {"message": "LLM request", "data": {"prompt": prompt, "tokens": 12}}
    stored = _compact_log_entry(entry)
    assert len(stored["data"]["prompt"]) == _STORED_LOG_FIELD_MAX_CHARS + len(" [...]")
    assert stored["data"]["tokens"] == 12
    assert entry["data"]["prompt"] == prompt  # the live entry is untouched
