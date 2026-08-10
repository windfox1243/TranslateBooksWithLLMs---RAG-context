"""Detect a Senior Editor that has stopped contributing signal.

The editor is the only quality gate that reads the source, so a job whose
editor has quietly gone inert looks exactly like a job whose translation is
clean: every run lands on `no_issues`, nothing enters the review queue, and
the progress bar reaches the end without a single complaint. The two states
are indistinguishable from the outside, which is how a weak editor model
reviewed two entire books without ever reporting one finding.

The signals below are deliberately narrow. Each one is something a working
editor cannot plausibly produce, so a warning means the editor is broken
rather than the prose being good.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# A working editor varies its output with its input. Six identical responses
# is already past coincidence for chunks that differ in length and content.
IDENTICAL_RESPONSE_WINDOW = 6
# Silence is only evidence once there has been enough of it, and only when
# something else was finding defects in the same text.
SILENT_WINDOW = 12


@dataclass(frozen=True)
class EditorSignalVerdict:
    """What the recent editor history says about the editor itself."""

    inert: bool = False
    reason: str = ""
    window: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)


def _reached_the_model(run: Dict[str, Any]) -> bool:
    """Report whether this run produced a response worth judging.

    A run that never got an answer says nothing about the editor's judgement,
    only about the transport, which has its own failure classes already.
    """

    if str(run.get("outcome") or "") in {"transport_failed", "blocked", "running"}:
        return False
    return bool(run.get("response_hash"))


def _llm_findings(run: Dict[str, Any]) -> int:
    """Count what the model itself reported, excluding deterministic checks."""

    return (
        int(run.get("issue_count") or 0)
        + int(run.get("warning_count") or 0)
    )


def assess_editor_signal(
    runs: Iterable[Dict[str, Any]],
    *,
    identical_window: int = IDENTICAL_RESPONSE_WINDOW,
    silent_window: int = SILENT_WINDOW,
) -> EditorSignalVerdict:
    """Judge whether the recent editor runs show a working editor.

    `runs` is ordered oldest to newest; only the tail is considered, so a job
    that recovers stops warning on its own.
    """

    judged: List[Dict[str, Any]] = [
        run for run in runs if _reached_the_model(run)
    ]
    if not judged:
        return EditorSignalVerdict()

    # One response for many different chunks is not a review. This catches a
    # model that answers with the same empty envelope every time, which is the
    # cheapest way for a weak model to satisfy the output contract.
    recent = judged[-identical_window:]
    hashes = {str(run.get("response_hash")) for run in recent}
    if len(recent) >= identical_window and len(hashes) == 1:
        return EditorSignalVerdict(
            inert=True,
            reason="constant_response",
            window=len(recent),
            detail={
                "response_hash": next(iter(hashes)),
                "completion_tokens": int(
                    recent[-1].get("completion_tokens") or 0
                ),
                "model": str(recent[-1].get("model") or ""),
                "thinking_tokens": sum(
                    int(run.get("thinking_tokens") or 0) for run in recent
                ),
            },
        )

    # Sustained silence is only damning next to a layer that was finding
    # defects in the same units. A genuinely clean text silences both.
    window = judged[-silent_window:]
    deterministic = sum(
        int(run.get("deterministic_count") or 0) for run in window
    )
    if (
        len(window) >= silent_window
        and deterministic > 0
        and all(_llm_findings(run) == 0 for run in window)
    ):
        return EditorSignalVerdict(
            inert=True,
            reason="no_llm_findings",
            window=len(window),
            detail={
                "deterministic_count": deterministic,
                "model": str(window[-1].get("model") or ""),
                "thinking_tokens": sum(
                    int(run.get("thinking_tokens") or 0) for run in window
                ),
            },
        )

    return EditorSignalVerdict()


def describe_editor_signal(verdict: EditorSignalVerdict) -> str:
    """Render a verdict as one operator-facing line."""

    model = str(verdict.detail.get("model") or "the configured editor model")
    if verdict.reason == "constant_response":
        return (
            f"⚠️ Senior Editor appears inert: the last {verdict.window} reviews "
            f"returned an identical response ({model}). The editor is not "
            "reading the chunks; consider a stronger editor model or a higher "
            "editor thinking level."
        )
    if verdict.reason == "no_llm_findings":
        return (
            f"⚠️ Senior Editor reported nothing across {verdict.window} reviews "
            f"while deterministic checks found "
            f"{verdict.detail.get('deterministic_count', 0)} defect(s) in the "
            f"same units ({model}). The editor is contributing no signal; "
            "consider a stronger editor model or a higher editor thinking level."
        )
    return ""


# The editor pass is entered once per unit and carries no job-scoped object to
# hang this on, so the "already said it" state lives here. Bounded, because a
# long-lived server process translates many books.
_WARNED_JOBS: List[str] = []
_WARNED_LIMIT = 64


def warn_if_editor_inert(
    db: Any,
    translation_id: str,
    log_callback: Optional[Any] = None,
) -> EditorSignalVerdict:
    """Warn at most once per job that the editor has stopped contributing.

    Diagnostics must never interrupt translation, so every failure here is
    swallowed: a job that cannot be assessed simply goes unassessed.
    """

    job = str(translation_id or "")
    if not db or not job or job in _WARNED_JOBS:
        return EditorSignalVerdict()
    try:
        verdict = assess_editor_signal(db.get_recent_editor_runs(job))
    except Exception:
        return EditorSignalVerdict()
    if verdict.inert:
        _WARNED_JOBS.append(job)
        del _WARNED_JOBS[:-_WARNED_LIMIT]
        if log_callback:
            try:
                log_callback(
                    f"editor_inert_{verdict.reason}",
                    describe_editor_signal(verdict),
                )
            except Exception:
                pass
    return verdict
