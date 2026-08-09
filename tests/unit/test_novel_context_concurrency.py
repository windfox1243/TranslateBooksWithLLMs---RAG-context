"""Regressions for state that used to leak between concurrent translation jobs.

Three separate bugs, one theme: novel-context state that was scoped to the
process when it should have been scoped to a job.
"""

import asyncio
import threading

import pytest

from src.utils.dialogue_attribution import (
    dialogue_attribution_or_carry,
    empty_dialogue_attribution,
)
from src.utils.novel_context import (
    bypass_context_gating,
    save_novel_context,
    set_bypass_context_gating,
)
from src.utils.novel_context.session import NovelContextSession

# --- The dialogue state a failed update used to discard -----------------------


def test_a_failed_update_carries_the_previous_dialogue_state():
    # Both failure paths in update_novel_context_chunk fill the sink with
    # empty_dialogue_attribution(), which is truthy but says nothing.
    carried = dialogue_attribution_or_carry(
        empty_dialogue_attribution(), {"last_speaker": "Alice"}
    )
    assert carried["state_after"] == {"last_speaker": "Alice"}


def test_a_real_result_replaces_the_previous_state():
    carried = dialogue_attribution_or_carry(
        {"turns": [{"speaker": "Bob"}], "state_after": {"last_speaker": "Bob"}},
        {"last_speaker": "Alice"},
    )
    assert carried["state_after"] == {"last_speaker": "Bob"}


def test_a_result_with_only_state_still_counts_as_real():
    # No turns but a state means the update ran and found the scene unchanged.
    carried = dialogue_attribution_or_carry(
        {"turns": [], "state_after": {"last_speaker": "Bob"}},
        {"last_speaker": "Alice"},
    )
    assert carried["state_after"] == {"last_speaker": "Bob"}


def test_a_missing_sink_carries_forward():
    assert dialogue_attribution_or_carry(None, {"last_speaker": "Alice"})[
        "state_after"
    ] == {"last_speaker": "Alice"}


def _run_analysis(monkeypatch, tmp_path, source_chunk, sink_payload):
    from src.utils import novel_context

    async def fake_update(**kwargs):
        sink = kwargs["dialogue_attribution_sink"]
        sink.clear()
        sink.update(sink_payload)
        return kwargs["current_global_lore"], kwargs["current_dynamic_state"], []

    monkeypatch.setattr(
        novel_context, "update_novel_context_chunk", fake_update, raising=False
    )
    session = NovelContextSession(
        path=tmp_path / "novel.txt",
        prompt_options={},
        global_lore="# GLOBAL LORE\n",
        dynamic_state="",
    )
    session.dialogue_state = {"last_speaker": "Alice"}
    asyncio.run(
        session.analyze_source(
            llm_client=object(),
            model_name="model",
            source_chunk=source_chunk,
            source_language="English",
            target_language="Vietnamese",
            chunk_index=2,
            total_chunks=10,
        )
    )
    return session.dialogue_state


def test_a_dialogue_heavy_chunk_keeps_its_speakers_when_the_update_fails(
    monkeypatch, tmp_path
):
    # This is the case that regressed: the source had dialogue, so the old code
    # trusted the sink -- and the sink was empty because the update had failed.
    assert _run_analysis(
        monkeypatch,
        tmp_path,
        '"Hello," Alice said. "Hi," Bob replied.',
        empty_dialogue_attribution(),
    ) == {"last_speaker": "Alice"}


def test_a_narration_chunk_keeps_its_speakers_too(monkeypatch, tmp_path):
    assert _run_analysis(
        monkeypatch,
        tmp_path,
        "Plain narration with no quoted speech.",
        empty_dialogue_attribution(),
    ) == {"last_speaker": "Alice"}


# --- The gating switch that used to be process-wide ---------------------------


def test_the_gating_override_does_not_escape_its_thread():
    set_bypass_context_gating(None)
    seen = {}
    ready = threading.Event()

    def job(name, value):
        set_bypass_context_gating(value)
        ready.wait(timeout=5)
        seen[name] = bypass_context_gating()

    threads = [
        threading.Thread(target=job, args=("strict", False)),
        threading.Thread(target=job, args=("lenient", True)),
    ]
    for thread in threads:
        thread.start()
    ready.set()
    for thread in threads:
        thread.join(timeout=5)

    # The lenient job used to switch gating off for the strict one mid-run.
    assert seen == {"strict": False, "lenient": True}


def test_without_an_override_the_configured_default_applies(monkeypatch):
    from src import config as _config

    set_bypass_context_gating(None)
    monkeypatch.setattr(_config, "BYPASS_CONTEXT_GATING", False, raising=False)
    assert bypass_context_gating() is False
    monkeypatch.setattr(_config, "BYPASS_CONTEXT_GATING", True, raising=False)
    assert bypass_context_gating() is True


def test_an_override_does_not_write_back_to_the_shared_config(monkeypatch):
    from src import config as _config

    set_bypass_context_gating(None)
    monkeypatch.setattr(_config, "BYPASS_CONTEXT_GATING", True, raising=False)
    set_bypass_context_gating(False)
    try:
        assert bypass_context_gating() is False
        assert _config.BYPASS_CONTEXT_GATING is True
    finally:
        set_bypass_context_gating(None)


def test_opening_a_session_leaves_the_shared_config_alone(monkeypatch, tmp_path):
    # The direct pin on the original bug: this is the call that used to assign
    # the job's option onto src.config for the whole process.
    from src import config as _config
    from src.utils.novel_context import open_novel_context_session

    set_bypass_context_gating(None)
    monkeypatch.setattr(_config, "BYPASS_CONTEXT_GATING", True, raising=False)
    try:
        session = open_novel_context_session(
            {"novel_context_file": "novel.txt", "bypass_context_gating": False},
            tmp_path,
        )
        assert session is not None
        assert _config.BYPASS_CONTEXT_GATING is True
        assert bypass_context_gating() is False
    finally:
        set_bypass_context_gating(None)


# --- The context file two jobs used to corrupt --------------------------------


def test_concurrent_saves_leave_one_whole_document(tmp_path):
    # Same file, same moment, very different sizes -- the shape that used to
    # produce a blend of both through the shared ".tmp" name.
    documents = [
        f"# GLOBAL LORE\n\n## CHARACTERS\n" + f"- Character {index}: role\n" * size
        for index, size in enumerate((400, 5, 900, 20), start=1)
    ]
    barrier = threading.Barrier(len(documents))

    def save(document):
        barrier.wait(timeout=5)
        save_novel_context("shared.txt", tmp_path, document)

    threads = [threading.Thread(target=save, args=(doc,)) for doc in documents]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    written = (tmp_path / "shared.txt").read_text(encoding="utf-8")
    entry_counts = {doc.count("- Character") for doc in documents}
    assert written.count("- Character") in entry_counts


def test_each_writer_stages_through_its_own_temporary_file(tmp_path, monkeypatch):
    # The deterministic half of the fix. The concurrency test above can only
    # catch corruption it happens to provoke; this one pins the property that
    # makes the corruption impossible.
    from pathlib import Path

    staged = []
    real_write_text = Path.write_text

    def record(self, *args, **kwargs):
        staged.append(self.name)
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", record)
    for _ in range(4):
        save_novel_context("shared.txt", tmp_path, "# GLOBAL LORE\n")

    assert len(staged) == 4
    assert len(set(staged)) == 4
    assert all(name != "shared.txt" for name in staged)


def test_no_temporary_files_survive_a_save(tmp_path):
    save_novel_context("novel.txt", tmp_path, "# GLOBAL LORE\n")
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_failed_write_does_not_leave_a_temporary_file(tmp_path, monkeypatch):
    import src.utils.atomic_replace as atomic_replace

    def explode(*args, **kwargs):
        raise OSError("disk full")

    # Patched where the replace actually happens. A bare OSError carries no
    # winerror, so the helper treats it as a real failure and does not retry.
    monkeypatch.setattr(atomic_replace.os, "replace", explode)
    with pytest.raises(OSError):
        save_novel_context("novel.txt", tmp_path, "# GLOBAL LORE\n")
    assert list(tmp_path.glob("*.tmp")) == []
