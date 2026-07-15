"""Threshold and phase-boundary repair coordination for review-required units."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class AutoReviewRepairCoordinator:
    """Drain review queues at safe unit boundaries without recursive retries."""

    def __init__(
        self,
        *,
        translation_id: str,
        checkpoint_manager: Any,
        output_dir: Path,
        threshold: int = 3,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.translation_id = translation_id
        self.checkpoint_manager = checkpoint_manager
        self.db = checkpoint_manager.db
        self.output_dir = Path(output_dir)
        self.threshold = max(0, min(int(threshold), 20))
        self.event_callback = event_callback
        self._attempts: Dict[tuple[str, str, int], int] = {}
        self._blocking_lock = threading.Lock()

    def _emit(self, event: str, batch: Dict[str, Any], **details: Any) -> None:
        if self.event_callback:
            self.event_callback(event, {**batch, **details})

    def _translation_candidates(self) -> List[int]:
        diagnostics = self.db.get_editor_diagnostics(self.translation_id)
        return sorted({
            int(item["chunk_index"])
            for item in diagnostics.get("current_review_queue") or []
            if item.get("retryable")
            and str(item.get("phase") or "translation") != "refinement"
        })

    def _refinement_candidates(self) -> tuple[List[int], str]:
        running = self.db.get_running_refinement_pass(self.translation_id)
        pass_id = str((running or {}).get("pass_id") or "")
        if pass_id:
            rows = self.db.get_refinement_results(pass_id)
        else:
            rows = self.db.get_active_refinement_results(self.translation_id)
            pass_id = str(rows[0].get("pass_id") or "") if rows else ""
        candidates = sorted({
            int(row.get("base_chunk_index"))
            for row in rows
            if row.get("base_chunk_index") is not None
            and str(row.get("status") or "completed") == "completed"
            and str(row.get("quality_status") or "") == "review_required"
        })
        return candidates, pass_id

    def _eligible(
        self, phase: str, *, boundary: bool,
    ) -> tuple[List[int], str]:
        if phase == "refinement":
            candidates, pass_id = self._refinement_candidates()
        else:
            candidates, pass_id = self._translation_candidates(), ""
        maximum = 2 if boundary else 1
        return [
            index for index in candidates
            if self._attempts.get((phase, pass_id, index), 0) < maximum
        ], pass_id

    async def repair_if_needed(
        self, phase: str, *, boundary: bool = False,
    ) -> Dict[str, Any]:
        """Repair a threshold-sized queue or drain it at a phase boundary."""
        phase = "refinement" if phase == "refinement" else "translation"
        candidates, pass_id = self._eligible(phase, boundary=boundary)
        if not boundary and (self.threshold == 0 or len(candidates) < self.threshold):
            return {"status": "below_threshold", "queued": len(candidates)}
        if not candidates:
            return {"status": "empty", "queued": 0}
        active = self.db.find_active_editor_repair_batch(self.translation_id)
        if active:
            return {
                "status": "another_batch_active",
                "batch_id": active.get("batch_id"),
            }

        batch_id = f"erb_auto_{uuid.uuid4().hex}"
        scope = "auto_review_boundary" if boundary else "auto_review_threshold"
        self.db.create_editor_repair_batch(
            batch_id,
            self.translation_id,
            scope,
            phase,
            candidates,
            stay_paused=False,
        )
        self.db.update_editor_repair_batch(
            batch_id,
            status="running",
            started_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        )
        batch = self.db.get_editor_repair_batch(batch_id) or {}
        self._emit("auto_running", batch, boundary=boundary)

        completed = succeeded = failed = 0
        from src.core.editor_retry import run_editor_retry, _refresh_output

        for position, chunk_index in enumerate(candidates, start=1):
            key = (phase, pass_id, chunk_index)
            self._attempts[key] = self._attempts.get(key, 0) + 1
            self.db.update_editor_repair_batch_item(
                batch_id, chunk_index, phase,
                status="running",
                started_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            )
            batch = self.db.get_editor_repair_batch(batch_id) or {}
            self._emit(
                "item_started", batch,
                chunk_index=chunk_index, position=position,
            )
            ok = False
            outcome = "failed"
            message = ""
            try:
                result = await run_editor_retry(
                    translation_id=self.translation_id,
                    chunk_index=chunk_index,
                    checkpoint_manager=self.checkpoint_manager,
                    output_dir=self.output_dir,
                    refresh_output=False,
                    phase=phase,
                    refinement_pass_id=pass_id or None,
                )
                outcome = str(result.get("outcome") or result.get("status") or "failed")
                ok = str(result.get("status") or "") == "succeeded"
                message = str(result.get("message") or "")[:500]
            except Exception as exc:  # preserve the remaining pipeline
                outcome = "failed"
                message = type(exc).__name__
            completed += 1
            succeeded += int(ok)
            failed += int(not ok)
            self.db.update_editor_repair_batch_item(
                batch_id, chunk_index, phase,
                status="succeeded" if ok else "failed",
                outcome=outcome,
                message=message,
                completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            )
            self.db.update_editor_repair_batch(
                batch_id,
                completed_items=completed,
                succeeded_items=succeeded,
                failed_items=failed,
            )
            batch = self.db.get_editor_repair_batch(batch_id) or {}
            self._emit(
                "item_succeeded" if ok else "item_failed",
                batch,
                chunk_index=chunk_index, position=position,
            )

        output_sync = {"status": "deferred"}
        if boundary and completed:
            checkpoint = self.checkpoint_manager.load_checkpoint(
                self.translation_id
            ) or {}
            config = dict((checkpoint.get("job") or {}).get("config") or {})
            self._emit(
                "rebuilding", self.db.get_editor_repair_batch(batch_id) or {}
            )
            output_sync = await _refresh_output(
                self.translation_id,
                self.checkpoint_manager,
                self.output_dir,
                str(config.get("output_filename") or ""),
                bool(config.get("bilingual_output")),
            )

        self.db.update_editor_repair_batch(
            batch_id,
            status="completed",
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        )
        batch = self.db.get_editor_repair_batch(batch_id) or {}
        self._emit(
            "auto_completed", batch,
            boundary=boundary, output_status=output_sync.get("status"),
        )
        return {
            "status": "completed",
            "batch_id": batch_id,
            "completed": completed,
            "succeeded": succeeded,
            "failed": failed,
            "output_sync": output_sync,
        }

    def repair_if_needed_blocking(
        self, phase: str, *, boundary: bool = False,
    ) -> Dict[str, Any]:
        """Run the async repair loop from a synchronous unit-stats callback."""
        if not self._blocking_lock.acquire(blocking=False):
            return {"status": "already_running"}
        result: Dict[str, Any] = {}
        error: List[BaseException] = []

        def runner() -> None:
            try:
                result.update(asyncio.run(
                    self.repair_if_needed(phase, boundary=boundary)
                ))
            except BaseException as exc:  # re-raised on the worker thread
                error.append(exc)

        try:
            thread = threading.Thread(
                target=runner,
                name=f"auto-review-repair-{self.translation_id}",
            )
            thread.start()
            thread.join()
            if error:
                raise error[0]
            return result
        finally:
            self._blocking_lock.release()
