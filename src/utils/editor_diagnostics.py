"""Local, bounded diagnostics for the Senior Editor state machine."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterable, Iterator, List, Optional

EDITOR_OUTCOMES = {
    "no_issues", "warnings_only", "locally_repaired", "llm_repaired",
    "review_required", "blocked", "transport_failed",
}
EDITOR_FAILURE_CLASSES = {
    "provider_auth", "provider_quota", "provider_rate_limit", "provider_empty",
    "provider_blocked", "provider_truncated", "transport", "schema_rejected",
    "contract_parse", "contract_incomplete", "contract_issue",
    "locator_missing", "locator_ambiguous", "local_patch_conflict",
    "repair_validation", "residue_blocker", "narrator_policy",
    "adapter_invalid", "internal",
}


def bounded_excerpt(value: Any, limit: int = 160) -> str:
    """Return a whitespace-normalized diagnostic excerpt."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def response_hash(value: Any) -> str:
    """Hash a response without persisting the complete model output."""
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def issue_excerpts(issues: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Keep structural issue metadata without persisting book text.

    `category`, `severity` and `confidence` are the three fields the repair
    gate judges an issue by, so without them a run that reported a defect and
    dropped it says only that it happened, never which rule decided. They are
    classifications the model assigns, not book text, so they are safe to keep.
    """
    result = []
    for issue in list(issues or [])[:12]:
        replacement = issue.get("draft_replacement") or {}
        try:
            confidence = round(float(issue.get("confidence")), 3)
        except (TypeError, ValueError):
            confidence = None
        result.append({
            "issue_id": bounded_excerpt(issue.get("issue_id"), 48),
            "category": bounded_excerpt(issue.get("category"), 48),
            "severity": bounded_excerpt(issue.get("severity"), 16),
            "confidence": confidence,
            "repair_kind": bounded_excerpt(issue.get("repair_kind"), 24),
            "source_chars": len(str(issue.get("source_quote") or "")),
            "draft_chars": len(str(issue.get("draft_quote") or "")),
            "replacement_chars": len(str(replacement.get("replacement") or "")),
            # An edit that replaces a span with itself. It reads as a finding
            # everywhere else in the record, and one measured chunk was twelve
            # of them, so the record has to be able to say so.
            "no_op": bool(
                isinstance(issue.get("draft_replacement"), dict)
                and str(replacement.get("draft") or "").strip()
                == str(replacement.get("replacement") or "").strip()
            ),
        })
    return result


def bounded_diagnostics(value: Any) -> Dict[str, Any]:
    """Reduce arbitrary editor diagnostics to bounded, non-book payloads.

    Any key not named below falls through to the scalar branch, and a list that
    lands there used to be discarded without a trace: a run that recorded which
    issue ids it had to give up persisted as one that gave up nothing. So a list
    of scalars is kept, bounded the same way `reason_codes` is.
    """
    if not isinstance(value, dict):
        return {}
    result: Dict[str, Any] = {}
    for key, item in value.items():
        if key == "issues":
            result[key] = issue_excerpts(item or [])
        elif key in {"final_reason_codes", "reason_codes"}:
            result[key] = [bounded_excerpt(entry) for entry in list(item or [])[:12]]
        elif key == "attempts":
            result[key] = [bounded_diagnostics(entry) for entry in list(item or [])[:4]]
        elif key == "automatic_retry" and isinstance(item, dict):
            result[key] = bounded_diagnostics(item)
        elif key == "narrator_conformance" and isinstance(item, dict):
            result[key] = {
                field: item.get(field)
                for field in (
                    "status", "strategy", "policy_source", "enforcement",
                    "profile_revision",
                )
                if isinstance(item.get(field), (str, int, float, bool))
            }
        elif key == "prompt_composition" and isinstance(item, dict):
            safe = {
                field: raw
                for field, raw in item.items()
                if field != "requests"
                and isinstance(raw, (str, int, float, bool))
            }
            safe["requests"] = [
                {
                    field: raw
                    for field, raw in request.items()
                    if isinstance(raw, (str, int, float, bool))
                }
                for request in list(item.get("requests") or [])[:8]
                if isinstance(request, dict)
            ]
            result[key] = safe
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = bounded_excerpt(item) if isinstance(item, str) else item
        elif isinstance(item, (list, tuple)) and all(
            isinstance(entry, (str, int, float, bool)) or entry is None
            for entry in item
        ):
            result[key] = [
                bounded_excerpt(entry) if isinstance(entry, str) else entry
                for entry in list(item)[:12]
            ]
    return result


_OPEN_EDITOR_RUNS: ContextVar[Optional[List["EditorRunRecorder"]]] = ContextVar(
    "open_editor_runs", default=None,
)


@contextmanager
def editor_run_scope() -> Iterator[None]:
    """Close any editor run the code inside this scope left open.

    A run is written as ``running`` and only a finish moves it off that value.
    Every give-up path is meant to record one, but an early return or a raised
    exception parks the row there forever -- and ``running`` belongs to no
    diagnostics bucket, so an abandoned run is not merely mislabelled, it is
    invisible: not successful, not degraded, not queued for review.

    A run salvaged here is recorded as needing review with an ``internal``
    failure class, which is what it is: the editor never reached a verdict for
    a reason on our side.
    """
    open_runs: List["EditorRunRecorder"] = []
    token = _OPEN_EDITOR_RUNS.set(open_runs)
    try:
        yield
    finally:
        _OPEN_EDITOR_RUNS.reset(token)
        for recorder in list(open_runs):
            try:
                recorder.finish("review_required", failure_class="internal")
            except Exception:
                pass


class EditorRunRecorder:
    """Best-effort persistence that never breaks translation work."""

    def __init__(self, options: Optional[Dict[str, Any]], **metadata: Any) -> None:
        self.options = options or {}
        self.run_id: Optional[int] = None
        self.finished = False
        self.db = None
        translation_id = str(self.options.get("translation_id") or "").strip()
        if not translation_id:
            return
        try:
            from src.persistence.database import Database

            self.db = Database(self.options.get("jobs_db_path") or None)
            self.run_id = self.db.create_editor_run({
                "translation_id": translation_id,
                "chunk_index": int(self.options.get("chunk_index", -1)),
                "phase": self.options.get("editor_phase") or "translation",
                "refinement_pass_id": self.options.get("_refinement_pass_id")
                or self.options.get("refinement_pass_id"),
                "provider": self.options.get("editor_provider_resolved")
                or self.options.get("llm_provider"),
                "model": self.options.get("editor_model_resolved")
                or self.options.get("model"),
                "source_language": self.options.get("source_language"),
                "target_language": metadata.get("target_language")
                or self.options.get("target_language"),
                "file_type": self.options.get("file_type"),
                "prompt_version": metadata.get("prompt_version"),
                "contract_version": metadata.get("contract_version"),
                "outcome": "running",
            })
            open_runs = _OPEN_EDITOR_RUNS.get()
            if open_runs is not None:
                open_runs.append(self)
        except Exception:
            self.db = None
            self.run_id = None

    def attempt(self, payload: Dict[str, Any]) -> None:
        if self.db is not None and self.run_id is not None:
            self.db.add_editor_attempt(self.run_id, payload)

    def finish(self, outcome: str, **payload: Any) -> None:
        if outcome not in EDITOR_OUTCOMES:
            outcome = "review_required"
        # The first verdict is the run's verdict; a scope closing behind a path
        # that already recorded one must not overwrite it.
        if self.finished:
            return
        self.finished = True
        open_runs = _OPEN_EDITOR_RUNS.get()
        if open_runs is not None and self in open_runs:
            open_runs.remove(self)
        if self.db is not None and self.run_id is not None:
            payload["diagnostics"] = bounded_diagnostics(
                payload.get("diagnostics")
            )
            self.db.finish_editor_run(self.run_id, {"outcome": outcome, **payload})
