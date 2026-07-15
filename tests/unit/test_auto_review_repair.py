"""Automatic threshold and phase-boundary review repair tests."""

from copy import deepcopy

import pytest

from src.core.editor.auto_review_repair import AutoReviewRepairCoordinator


class _RepairDB:
    def __init__(self, translation_candidates=None, refinement_results=None):
        self.translation_candidates = list(translation_candidates or [])
        self.refinement_results = list(refinement_results or [])
        self.batches = {}

    def get_editor_diagnostics(self, _translation_id):
        return {
            "current_review_queue": [{
                "chunk_index": index,
                "phase": "translation",
                "retryable": True,
            } for index in self.translation_candidates]
        }

    def get_running_refinement_pass(self, _translation_id):
        return {"pass_id": "pass-1"} if self.refinement_results else None

    def get_refinement_results(self, _pass_id):
        return deepcopy(self.refinement_results)

    def get_active_refinement_results(self, _translation_id):
        return deepcopy(self.refinement_results)

    def find_active_editor_repair_batch(self, _translation_id):
        return next((
            deepcopy(batch) for batch in reversed(list(self.batches.values()))
            if batch["status"] in {"queued", "pausing", "running"}
        ), None)

    def create_editor_repair_batch(
        self, batch_id, translation_id, scope, phase, chunk_indices,
        *, stay_paused,
    ):
        self.batches[batch_id] = {
            "batch_id": batch_id,
            "translation_id": translation_id,
            "scope": scope,
            "phase": phase,
            "status": "queued",
            "stay_paused": stay_paused,
            "total_items": len(chunk_indices),
            "completed_items": 0,
            "succeeded_items": 0,
            "failed_items": 0,
            "items": [{
                "chunk_index": index,
                "phase": phase,
                "status": "queued",
            } for index in chunk_indices],
        }
        return True

    def update_editor_repair_batch(self, batch_id, **changes):
        self.batches[batch_id].update(changes)
        return True

    def update_editor_repair_batch_item(
        self, batch_id, chunk_index, phase, **changes,
    ):
        item = next(
            item for item in self.batches[batch_id]["items"]
            if item["chunk_index"] == chunk_index and item["phase"] == phase
        )
        item.update(changes)
        return True

    def get_editor_repair_batch(self, batch_id):
        return deepcopy(self.batches[batch_id])


class _Checkpoints:
    def __init__(self, db):
        self.db = db

    def load_checkpoint(self, _translation_id):
        return {
            "job": {
                "config": {
                    "output_filename": "translated.txt",
                    "bilingual_output": False,
                }
            }
        }


@pytest.mark.asyncio
async def test_threshold_waits_until_three_review_chunks(monkeypatch, tmp_path):
    db = _RepairDB([1, 2])
    calls = []

    async def fake_retry(**kwargs):
        calls.append(kwargs["chunk_index"])
        return {"status": "succeeded", "outcome": "locally_repaired"}

    monkeypatch.setattr("src.core.editor_retry.run_editor_retry", fake_retry)
    coordinator = AutoReviewRepairCoordinator(
        translation_id="job",
        checkpoint_manager=_Checkpoints(db),
        output_dir=tmp_path,
        threshold=3,
    )

    assert (await coordinator.repair_if_needed("translation"))["status"] == "below_threshold"
    db.translation_candidates.append(3)
    result = await coordinator.repair_if_needed("translation")

    assert result["status"] == "completed"
    assert calls == [1, 2, 3]
    assert next(iter(db.batches.values()))["succeeded_items"] == 3


@pytest.mark.asyncio
async def test_phase_boundary_repairs_under_threshold_and_rebuilds_once(
    monkeypatch, tmp_path,
):
    db = _RepairDB([7])
    refreshed = []

    async def fake_retry(**_kwargs):
        return {"status": "succeeded", "outcome": "warnings_only"}

    async def fake_refresh(*args, **_kwargs):
        refreshed.append(args)
        return {"status": "updated"}

    monkeypatch.setattr("src.core.editor_retry.run_editor_retry", fake_retry)
    monkeypatch.setattr("src.core.editor_retry._refresh_output", fake_refresh)
    coordinator = AutoReviewRepairCoordinator(
        translation_id="job",
        checkpoint_manager=_Checkpoints(db),
        output_dir=tmp_path,
        threshold=3,
    )

    result = await coordinator.repair_if_needed("translation", boundary=True)

    assert result["succeeded"] == 1
    assert result["output_sync"]["status"] == "updated"
    assert len(refreshed) == 1


@pytest.mark.asyncio
async def test_running_refinement_pass_uses_phase_specific_overlay(
    monkeypatch, tmp_path,
):
    db = _RepairDB(refinement_results=[{
        "pass_id": "pass-1",
        "base_chunk_index": index,
        "status": "completed",
        "quality_status": "review_required",
    } for index in (4, 5, 6)])
    calls = []

    async def fake_retry(**kwargs):
        calls.append((kwargs["chunk_index"], kwargs["refinement_pass_id"]))
        return {"status": "succeeded", "outcome": "llm_repaired"}

    monkeypatch.setattr("src.core.editor_retry.run_editor_retry", fake_retry)
    coordinator = AutoReviewRepairCoordinator(
        translation_id="job",
        checkpoint_manager=_Checkpoints(db),
        output_dir=tmp_path,
        threshold=3,
    )

    result = await coordinator.repair_if_needed("refinement")

    assert result["succeeded"] == 3
    assert calls == [(4, "pass-1"), (5, "pass-1"), (6, "pass-1")]
