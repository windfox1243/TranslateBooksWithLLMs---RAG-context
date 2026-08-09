"""The replace that survives a Windows sharing violation."""
import os

import pytest

from src.utils.atomic_replace import replace_atomically


def _sharing_violation(winerror):
    error = OSError(13, "in use by another process")
    error.winerror = winerror
    return error


def test_a_clear_destination_is_replaced_on_the_first_attempt(tmp_path):
    source = tmp_path / "staged.txt"
    destination = tmp_path / "context.txt"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")

    replace_atomically(source, destination)

    assert destination.read_text(encoding="utf-8") == "new"
    assert not source.exists()


@pytest.mark.parametrize("winerror", [5, 32])
def test_a_transient_hold_is_waited_out_rather_than_failing(
    tmp_path, monkeypatch, winerror
):
    """An indexer or a scanner holds the destination for a few milliseconds.

    Before the retry this raised, and the whole save was lost for a hold that
    had already cleared by the time the caller saw the error.
    """
    source = tmp_path / "staged.txt"
    destination = tmp_path / "context.txt"
    source.write_text("new", encoding="utf-8")

    real_replace = os.replace
    attempts = []

    def held_twice(*args, **kwargs):
        attempts.append(1)
        if len(attempts) <= 2:
            raise _sharing_violation(winerror)
        return real_replace(*args, **kwargs)

    monkeypatch.setattr("src.utils.atomic_replace.os.replace", held_twice)
    monkeypatch.setattr("src.utils.atomic_replace.time.sleep", lambda _s: None)

    replace_atomically(source, destination)

    assert len(attempts) == 3
    assert destination.read_text(encoding="utf-8") == "new"


def test_a_destination_held_for_the_whole_window_says_what_to_close(
    tmp_path, monkeypatch
):
    source = tmp_path / "staged.txt"
    source.write_text("new", encoding="utf-8")

    def always_held(*args, **kwargs):
        raise _sharing_violation(32)

    monkeypatch.setattr("src.utils.atomic_replace.os.replace", always_held)
    monkeypatch.setattr("src.utils.atomic_replace.time.sleep", lambda _s: None)

    with pytest.raises(OSError) as raised:
        replace_atomically(source, tmp_path / "context.txt")

    assert "open in another program" in str(raised.value)


def test_an_error_that_is_not_a_hold_is_raised_without_retrying(
    tmp_path, monkeypatch
):
    """A full disk or a missing source must not be waited on six times."""
    source = tmp_path / "staged.txt"
    source.write_text("new", encoding="utf-8")
    attempts = []

    def disk_full(*args, **kwargs):
        attempts.append(1)
        raise OSError(28, "no space left on device")

    monkeypatch.setattr("src.utils.atomic_replace.os.replace", disk_full)

    with pytest.raises(OSError):
        replace_atomically(source, tmp_path / "context.txt")

    assert len(attempts) == 1
