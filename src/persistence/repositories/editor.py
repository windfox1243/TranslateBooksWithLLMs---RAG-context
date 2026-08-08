"""Editor-run persistence.

This repository owns the SQL behind the editor pass: run rows, per-unit
attempts and the diagnostics aggregation. As in `JobRepository`, the
connection and the write lock stay on the `Database` facade and are reached
through `self.database`; `Database` keeps a delegating method for each one so
existing call sites are unaffected.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.persistence.repositories.base import DatabaseRepository


class EditorRepository(DatabaseRepository):
    """Owns editor-run SQL."""

    prefixes = ("editor",)
    methods = frozenset({
        "add_editor_attempt",
        "create_editor_run",
        "finish_editor_run",
        "get_editor_diagnostics",
    })

    def create_editor_run(self, payload: Dict[str, Any]) -> Optional[int]:
        """Create one locally persisted Senior Editor run."""
        fields = (
            "translation_id", "chunk_index", "phase", "refinement_pass_id", "provider", "model",
            "source_language", "target_language", "file_type",
            "prompt_version", "contract_version", "outcome",
        )
        values = [payload.get(field) for field in fields]
        with self.database._lock:
            try:
                cursor = self.database._get_connection().cursor()
                cursor.execute(
                    f"INSERT INTO editor_runs ({','.join(fields)}) VALUES "
                    f"({','.join('?' for _ in fields)})",
                    values,
                )
                self.database._commit_connection(self.database._get_connection())
                return int(cursor.lastrowid)
            except Exception as exc:
                print(f"Error creating editor run: {exc}")
                return None

    def add_editor_attempt(self, run_id: int, payload: Dict[str, Any]) -> bool:
        """Append a bounded diagnostic record for one editor request."""
        fields = (
            "run_id", "attempt_index", "stage", "parse_status",
            "failure_class", "reason_codes", "prompt_tokens",
            "completion_tokens", "thinking_tokens", "total_tokens",
            "was_truncated", "finish_reason",
            "blocked_reason", "response_hash", "excerpts",
        )
        values = [
            run_id,
            int(payload.get("attempt_index", 0)),
            payload.get("stage") or "unknown",
            payload.get("parse_status"),
            payload.get("failure_class"),
            json.dumps(payload.get("reason_codes") or [], ensure_ascii=False),
            int(payload.get("prompt_tokens", 0) or 0),
            int(payload.get("completion_tokens", 0) or 0),
            int(payload.get("thinking_tokens", 0) or 0),
            int(payload.get("total_tokens", 0) or 0),
            int(bool(payload.get("was_truncated"))),
            payload.get("finish_reason"),
            payload.get("blocked_reason"),
            payload.get("response_hash"),
            json.dumps(payload.get("excerpts") or [], ensure_ascii=False),
        ]
        with self.database._lock:
            try:
                conn = self.database._get_connection()
                conn.execute(
                    f"INSERT INTO editor_attempts ({','.join(fields)}) VALUES "
                    f"({','.join('?' for _ in fields)})",
                    values,
                )
                self.database._commit_connection(conn)
                return True
            except Exception as exc:
                print(f"Error saving editor attempt: {exc}")
                return False

    def finish_editor_run(self, run_id: int, payload: Dict[str, Any]) -> bool:
        """Finalize one editor run with a classified outcome."""
        with self.database._lock:
            try:
                conn = self.database._get_connection()
                conn.execute(
                    """
                    UPDATE editor_runs SET parse_status = ?, outcome = ?,
                        failure_class = ?, issue_count = ?,
                        warning_count = ?,
                        resolved_issue_count = ?, unresolved_issue_count = ?,
                        result_state = ?, recovered_truncation = ?,
                        deterministic_count = ?, prompt_tokens = ?,
                        completion_tokens = ?, thinking_tokens = ?,
                        total_tokens = ?, was_truncated = ?,
                        finish_reason = ?, blocked_reason = ?, response_hash = ?,
                        diagnostics = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        payload.get("parse_status"),
                        payload.get("outcome") or "review_required",
                        payload.get("failure_class"),
                        int(payload.get("issue_count", 0) or 0),
                        int(payload.get("warning_count", 0) or 0),
                        int(payload.get("resolved_issue_count", 0) or 0),
                        int(payload.get("unresolved_issue_count", 0) or 0),
                        payload.get("result_state") or "unchanged_draft",
                        int(bool(payload.get("recovered_truncation"))),
                        int(payload.get("deterministic_count", 0) or 0),
                        int(payload.get("prompt_tokens", 0) or 0),
                        int(payload.get("completion_tokens", 0) or 0),
                        int(payload.get("thinking_tokens", 0) or 0),
                        int(payload.get("total_tokens", 0) or 0),
                        int(bool(payload.get("was_truncated"))),
                        payload.get("finish_reason"),
                        payload.get("blocked_reason"),
                        payload.get("response_hash"),
                        json.dumps(payload.get("diagnostics") or {}, ensure_ascii=False),
                        int(run_id),
                    ),
                )
                self.database._commit_connection(conn)
                return True
            except Exception as exc:
                print(f"Error finishing editor run: {exc}")
                return False

    def get_editor_diagnostics(self, translation_id: str) -> Dict[str, Any]:
        """Return aggregate and per-run editor diagnostics for a job."""
        with self.database._lock:
            conn = self.database._get_connection()
            rows = [dict(row) for row in conn.execute(
                "SELECT * FROM editor_runs WHERE translation_id = ? ORDER BY id",
                (translation_id,),
            ).fetchall()]
            if not rows:
                return {
                    "translation_id": translation_id,
                    "classification": "legacy_unclassified",
                    "summary": {"total": 0},
                    "runs": [],
                }
            outcomes: Dict[str, int] = {}
            failures: Dict[str, int] = {}
            result_states: Dict[str, int] = {}
            attempts_by_run: Dict[int, list] = {}
            run_ids = [int(row["id"]) for row in rows]
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                for attempt_row in conn.execute(
                    f"SELECT * FROM editor_attempts WHERE run_id IN ({placeholders}) "
                    "ORDER BY run_id, attempt_index",
                    run_ids,
                ).fetchall():
                    attempt = dict(attempt_row)
                    for field in ("reason_codes", "excerpts"):
                        try:
                            attempt[field] = json.loads(attempt.get(field) or "[]")
                        except (TypeError, ValueError):
                            attempt[field] = []
                    # The API exposes classifications, not book excerpts.
                    attempt.pop("excerpts", None)
                    attempts_by_run.setdefault(int(attempt["run_id"]), []).append(
                        attempt
                    )
            for row in rows:
                stored_outcome = row.get("outcome") or "unknown"
                normalized_outcome = {
                    "repaired": "llm_repaired",
                    "draft_kept_review": "review_required",
                }.get(stored_outcome, stored_outcome)
                if normalized_outcome != stored_outcome:
                    row["legacy_outcome"] = stored_outcome
                    row["outcome"] = normalized_outcome
                outcomes[normalized_outcome] = outcomes.get(normalized_outcome, 0) + 1
                result_state = row.get("result_state") or "unchanged_draft"
                result_states[result_state] = result_states.get(result_state, 0) + 1
                if row.get("failure_class"):
                    failures[row["failure_class"]] = failures.get(
                        row["failure_class"], 0
                    ) + 1
                try:
                    row["diagnostics"] = json.loads(row.get("diagnostics") or "{}")
                except (TypeError, ValueError):
                    row["diagnostics"] = {}
                row["diagnostics"].pop("issues", None)
                row["attempts"] = attempts_by_run.get(int(row["id"]), [])
                attempts = row["attempts"]
                row["request_count"] = len([
                    item for item in attempts
                    if int(item.get("prompt_tokens", 0) or 0) > 0
                ])
                row["max_request_prompt_tokens"] = max(
                    (int(item.get("prompt_tokens", 0) or 0) for item in attempts),
                    default=0,
                )
                row["max_request_total_tokens"] = max(
                    (int(item.get("total_tokens", 0) or 0) for item in attempts),
                    default=0,
                )
                row["cumulative_prompt_tokens"] = int(
                    row.get("prompt_tokens", 0) or 0
                )
                row["cumulative_completion_tokens"] = int(
                    row.get("completion_tokens", 0) or 0
                )
                row["cumulative_thinking_tokens"] = int(
                    row.get("thinking_tokens", 0) or 0
                )
                row["cumulative_total_tokens"] = int(
                    row.get("total_tokens", 0) or 0
                )
            successful = sum(
                outcomes.get(name, 0)
                for name in (
                    "no_issues", "warnings_only", "locally_repaired",
                    "llm_repaired",
                )
            )
            review_count = outcomes.get("review_required", 0)
            degraded = outcomes.get("transport_failed", 0)
            hard_failed = outcomes.get("blocked", 0)
            latest_by_unit: Dict[tuple, Dict[str, Any]] = {}
            for row in rows:
                effective_phase = str(row.get("phase") or "translation")
                if effective_phase == "manual_retry":
                    effective_phase = "translation"
                latest_by_unit[(
                    int(row.get("chunk_index", -1)),
                    effective_phase,
                )] = row
            active_refinement_indices = {
                int(item[0]) for item in conn.execute(
                    "SELECT r.base_chunk_index FROM refinement_chunk_results r "
                    "JOIN refinement_passes p ON p.pass_id=r.pass_id "
                    "WHERE p.translation_id=? AND p.promoted=1 "
                    "AND r.base_chunk_index IS NOT NULL",
                    (translation_id,),
                ).fetchall()
            }
            current_runs = [
                row for (chunk_index, phase), row in latest_by_unit.items()
                if not (
                    phase == "translation"
                    and chunk_index in active_refinement_indices
                )
            ]
            current_review_queue = []
            for row in current_runs:
                if row.get("outcome") not in {
                    "review_required", "transport_failed", "blocked",
                }:
                    continue
                reason_codes = []
                for attempt in row.get("attempts") or []:
                    reason_codes.extend(attempt.get("reason_codes") or [])
                reason_codes.extend(
                    (row.get("diagnostics") or {}).get("reason_codes") or []
                )
                current_review_queue.append({
                    "run_id": row.get("id"),
                    "chunk_index": row.get("chunk_index"),
                    "phase": row.get("phase") or "translation",
                    "outcome": row.get("outcome"),
                    "failure_class": row.get("failure_class"),
                    "attempts_used": len(row.get("attempts") or []),
                    "reason_codes": list(dict.fromkeys(
                        str(code) for code in reason_codes if code
                    ))[:12],
                    "retryable": row.get("outcome") in {
                        "review_required", "transport_failed",
                    },
                })
            current_successful = sum(
                1 for row in current_runs
                if row.get("outcome") in {
                    "no_issues", "warnings_only", "locally_repaired", "llm_repaired",
                }
            )
            current_degraded = sum(
                1 for row in current_runs if row.get("outcome") == "transport_failed"
            )
            current_blocked = sum(
                1 for row in current_runs if row.get("outcome") == "blocked"
            )
            return {
                "translation_id": translation_id,
                "classification": "classified",
                "summary": {
                    "total": len(rows),
                    "outcomes": outcomes,
                    "failure_classes": failures,
                    "result_states": result_states,
                    "successful": current_successful,
                    "review_required": len(current_review_queue),
                    "historical_successful": successful,
                    "historical_review_required": review_count,
                    "degraded": current_degraded,
                    "hard_failed": current_blocked,
                    "current_total": len(current_runs),
                    "current_review_required": len(current_review_queue),
                    "warnings": sum(
                        int(row.get("warning_count", 0) or 0) for row in rows
                    ),
                    "recovered": sum(
                        int(bool(row.get("recovered_truncation"))) for row in rows
                    ),
                },
                "current_review_queue": current_review_queue,
                "runs": rows,
            }
