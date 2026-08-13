"""An editor that stops reading must not look like a clean book."""

import pytest

from src.core.editor.signal_monitor import (
    assess_editor_signal,
    describe_editor_signal,
    warn_if_editor_inert,
)


def _run(index, **overrides):
    values = {
        "id": index,
        "chunk_index": index,
        "model": "some-flash-lite",
        "outcome": "no_issues",
        "response_hash": f"hash-{index}",
        "issue_count": 0,
        "warning_count": 0,
        "deterministic_count": 0,
        "completion_tokens": 27,
        "thinking_tokens": 0,
    }
    values.update(overrides)
    return values


def test_a_working_editor_is_not_reported():
    """Varied responses are a working editor, however few issues it raises."""

    runs = [_run(i) for i in range(20)]
    assert assess_editor_signal(runs).inert is False


def test_one_response_for_many_chunks_is_not_a_review():
    """The cheapest way to satisfy the contract is the same empty envelope."""

    runs = [_run(i, response_hash="same") for i in range(8)]
    verdict = assess_editor_signal(runs)
    assert verdict.inert is True
    assert verdict.reason == "constant_response"
    assert verdict.detail["model"] == "some-flash-lite"
    assert "identical response" in describe_editor_signal(verdict)


def test_a_short_identical_run_is_not_yet_evidence():
    """Two chunks can legitimately produce the same empty answer."""

    runs = [_run(i, response_hash="same") for i in range(3)]
    assert assess_editor_signal(runs).inert is False


def test_silence_beside_deterministic_findings_is_reported():
    """The regex layer finding defects the model never sees is the tell."""

    runs = [_run(i) for i in range(14)]
    runs[4]["deterministic_count"] = 10
    runs[9]["deterministic_count"] = 80
    verdict = assess_editor_signal(runs)
    assert verdict.inert is True
    assert verdict.reason == "no_llm_findings"
    assert verdict.detail["deterministic_count"] == 90


def test_silence_alone_is_not_reported():
    """A genuinely clean text silences both layers; that is not a defect."""

    assert assess_editor_signal([_run(i) for i in range(20)]).inert is False


def test_an_editor_that_finds_anything_is_left_alone():
    """One real finding in the window is proof the editor still reads."""

    runs = [_run(i) for i in range(14)]
    runs[3]["deterministic_count"] = 10
    runs[7]["issue_count"] = 1
    assert assess_editor_signal(runs).inert is False


def test_runs_that_never_reached_the_model_are_ignored():
    """Transport failures say nothing about the editor's judgement."""

    runs = [
        _run(i, outcome="transport_failed", response_hash=None)
        for i in range(10)
    ]
    assert assess_editor_signal(runs).inert is False


class _StubDb:
    def __init__(self, runs):
        self.runs = runs
        self.calls = 0

    def get_recent_editor_runs(self, translation_id, limit=24):
        self.calls += 1
        return self.runs


def test_the_warning_is_said_once_per_job():
    """A per-unit check must not repeat itself for every remaining chunk."""

    db = _StubDb([_run(i, response_hash="same") for i in range(8)])
    logged = []
    for _ in range(5):
        warn_if_editor_inert(
            db, "job-said-once", lambda code, text: logged.append((code, text)),
        )
    assert [code for code, _ in logged] == ["editor_inert_constant_response"]


def test_a_broken_diagnostic_never_interrupts_translation():
    """Diagnostics are best effort; a failing probe must stay silent."""

    class _Exploding:
        def get_recent_editor_runs(self, translation_id, limit=24):
            raise RuntimeError("db is gone")

    logged = []
    verdict = warn_if_editor_inert(
        _Exploding(), "job-broken", lambda code, text: logged.append(code),
    )
    assert verdict.inert is False
    assert logged == []


def test_a_clean_book_that_was_actually_read_is_left_alone():
    """Thinking tokens separate a clean audit from an absent one.

    Measured on a real job: `gemini-3.1-flash-lite` answered one chunk with the
    same empty envelope as every inert run, after spending 7,862 thinking
    tokens on it. The response was identical; the reading was not.
    """

    runs = [
        _run(i, response_hash="same", thinking_tokens=7862) for i in range(8)
    ]
    assert assess_editor_signal(runs).inert is False


def test_one_thinking_run_vindicates_the_window():
    """A model that pays anywhere in the window is not answering blind."""

    runs = [_run(i, response_hash="same") for i in range(8)]
    runs[-2]["thinking_tokens"] = 120
    assert assess_editor_signal(runs).inert is False


def test_silence_beside_deterministic_findings_still_needs_zero_thinking():
    runs = [
        _run(i, deterministic_count=1 if i % 4 == 0 else 0, thinking_tokens=900)
        for i in range(14)
    ]
    assert assess_editor_signal(runs).inert is False


def test_the_starved_editor_is_still_caught():
    """The failure this exists for: no findings, no thinking, defects present."""

    runs = [
        _run(i, deterministic_count=1 if i % 4 == 0 else 0) for i in range(14)
    ]
    verdict = assess_editor_signal(runs)
    assert verdict.inert is True
    assert verdict.reason == "no_llm_findings"
