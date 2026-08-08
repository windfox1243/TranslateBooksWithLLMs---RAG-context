"""
In-memory state manager for Sample & Compare runs.

Holds one entry per `sample_id` with the columns, items (source extracts) and
per-cell results streamed as LLM calls complete. Evicted after 1 hour or on
server restart — sample runs are ephemeral by design.
"""
import threading
import time
from typing import Any, Dict, List, Optional

# Sample entries older than this are pruned on every public access.
SAMPLE_TTL_SECONDS = 3600


class SampleStateManager:
    """Thread-safe registry for Sample & Compare runs."""

    def __init__(self) -> None:
        self._samples: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create(
        self,
        sample_id: str,
        items: List[Dict[str, Any]],
        columns: List[Dict[str, Any]],
        mode: str,
    ) -> None:
        """Register a fresh sample run with its initial item/column shape."""
        with self._lock:
            self._prune_expired_locked()
            n_rows = len(items)
            n_cols = len(columns)
            self._samples[sample_id] = {
                "sample_id": sample_id,
                "status": "running",
                "mode": mode,
                "items": items,
                "columns": columns,
                "cells": [
                    {
                        "row": r,
                        "col": c,
                        "phase": "translate",
                        "status": "pending",
                        "output": None,
                        "metrics": None,
                        "error": None,
                    }
                    for r in range(n_rows)
                    for c in range(n_cols)
                ],
                "cancelled": False,
                "created_at": time.time(),
            }

    def get(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Return a deep-ish snapshot of a sample entry, or None if unknown."""
        with self._lock:
            self._prune_expired_locked()
            entry = self._samples.get(sample_id)
            if entry is None:
                return None
            return {
                "sample_id": entry["sample_id"],
                "status": entry["status"],
                "mode": entry["mode"],
                "items": entry["items"],
                "columns": entry["columns"],
                "cells": [dict(cell) for cell in entry["cells"]],
            }

    def exists(self, sample_id: str) -> bool:
        with self._lock:
            return sample_id in self._samples

    def cancel(self, sample_id: str) -> bool:
        """Flag a sample as cancelled. Returns False if the id is unknown."""
        with self._lock:
            entry = self._samples.get(sample_id)
            if entry is None:
                return False
            entry["cancelled"] = True
            return True

    def is_cancelled(self, sample_id: str) -> bool:
        with self._lock:
            entry = self._samples.get(sample_id)
            return bool(entry and entry["cancelled"])

    def update_cell(
        self,
        sample_id: str,
        row: int,
        col: int,
        phase: str,
        *,
        status: str,
        output: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Upsert the cell record for (row, col, phase). The `cells` list is keyed
        by row/col only for `translate`; `refine` cells live as a separate
        record with phase=='refine' so the front-end can render both stacked.
        """
        with self._lock:
            entry = self._samples.get(sample_id)
            if entry is None:
                return
            for cell in entry["cells"]:
                if cell["row"] == row and cell["col"] == col and cell["phase"] == phase:
                    cell["status"] = status
                    cell["output"] = output
                    cell["metrics"] = metrics
                    cell["error"] = error
                    return
            entry["cells"].append({
                "row": row,
                "col": col,
                "phase": phase,
                "status": status,
                "output": output,
                "metrics": metrics,
                "error": error,
            })

    def set_status(self, sample_id: str, status: str) -> None:
        with self._lock:
            entry = self._samples.get(sample_id)
            if entry is not None:
                entry["status"] = status

    def set_run_context(self, sample_id: str, run_ctx: Dict[str, Any]) -> None:
        """Stash the parameters needed to launch the LLM thread later.

        Used by the deferred-dispatch flow: /api/sample/run prepares items and
        stores the context here; /api/sample/<id>/dispatch reads it back to
        spawn `_run_sample_async`.
        """
        with self._lock:
            entry = self._samples.get(sample_id)
            if entry is not None:
                entry["run_context"] = run_ctx

    def get_run_context(self, sample_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._samples.get(sample_id)
            if entry is None:
                return None
            return entry.get("run_context")

    def _prune_expired_locked(self) -> None:
        now = time.time()
        expired = [
            sid for sid, entry in self._samples.items()
            if now - entry["created_at"] > SAMPLE_TTL_SECONDS
        ]
        for sid in expired:
            del self._samples[sid]
