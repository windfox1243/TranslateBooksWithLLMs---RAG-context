"""Local, bounded diagnostics for the Senior Editor state machine."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, Optional


EDITOR_OUTCOMES = {
    "no_issues", "repaired", "draft_kept_review", "blocked",
}
EDITOR_FAILURE_CLASSES = {
    "provider_empty", "provider_blocked", "provider_truncated", "transport",
    "contract_parse", "contract_incomplete", "locator_missing",
    "locator_ambiguous", "local_patch_conflict", "repair_missing",
    "residue_blocker", "adapter_invalid", "internal",
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


def issue_excerpts(issues: Iterable[Dict[str, Any]]) -> list[Dict[str, str]]:
    """Keep only bounded evidence needed to diagnose an issue locator."""
    result = []
    for issue in list(issues or [])[:12]:
        replacement = issue.get("draft_replacement") or {}
        result.append({
            "issue_id": bounded_excerpt(issue.get("issue_id"), 48),
            "source": bounded_excerpt(issue.get("source_quote")),
            "draft": bounded_excerpt(issue.get("draft_quote")),
            "replacement": bounded_excerpt(replacement.get("replacement")),
        })
    return result


def bounded_diagnostics(value: Any) -> Dict[str, Any]:
    """Reduce arbitrary editor diagnostics to bounded, non-book payloads."""
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
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = bounded_excerpt(item) if isinstance(item, str) else item
    return result


class EditorRunRecorder:
    """Best-effort persistence that never breaks translation work."""

    def __init__(self, options: Optional[Dict[str, Any]], **metadata: Any) -> None:
        self.options = options or {}
        self.run_id: Optional[int] = None
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
        except Exception:
            self.db = None
            self.run_id = None

    def attempt(self, payload: Dict[str, Any]) -> None:
        if self.db is not None and self.run_id is not None:
            self.db.add_editor_attempt(self.run_id, payload)

    def finish(self, outcome: str, **payload: Any) -> None:
        if outcome not in EDITOR_OUTCOMES:
            outcome = "draft_kept_review"
        if self.db is not None and self.run_id is not None:
            payload["diagnostics"] = bounded_diagnostics(
                payload.get("diagnostics")
            )
            self.db.finish_editor_run(self.run_id, {"outcome": outcome, **payload})
