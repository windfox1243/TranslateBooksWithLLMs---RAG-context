"""An editor run must never be left parked at 'running'.

A run is written as 'running' up front and moved off it by the state machine.
Every bucket in the diagnostics -- successful, degraded, blocked, review queue
-- is keyed on the other values, so a run abandoned at 'running' does not read
as failed, it reads as nothing at all. These pin the boundary that closes one.
"""

from types import SimpleNamespace

import pytest

from src.core.translator import run_chunk_reflection_pass
from src.persistence.database import Database
from src.utils.editor_diagnostics import EditorRunRecorder, editor_run_scope


@pytest.fixture
def options(tmp_path):
    db = Database(str(tmp_path / "jobs.db"))
    assert db.create_job("job-1", "txt", {})
    db.close()
    return {
        "translation_id": "job-1",
        "chunk_index": 3,
        "jobs_db_path": str(tmp_path / "jobs.db"),
    }


def _runs(options):
    db = Database(options["jobs_db_path"])
    try:
        return db.get_editor_diagnostics("job-1")["runs"]
    finally:
        db.close()


def test_a_run_nobody_finished_is_closed_by_the_scope(options):
    with editor_run_scope():
        EditorRunRecorder(options)

    run = _runs(options)[0]
    assert run["outcome"] == "review_required"
    assert run["failure_class"] == "internal"


def test_a_run_abandoned_by_an_exception_is_closed_and_the_error_still_raises(
    options,
):
    with pytest.raises(RuntimeError, match="editor exploded"):
        with editor_run_scope():
            EditorRunRecorder(options)
            raise RuntimeError("editor exploded")

    assert _runs(options)[0]["outcome"] == "review_required"


def test_the_verdict_a_run_recorded_for_itself_is_not_overwritten(options):
    with editor_run_scope():
        recorder = EditorRunRecorder(options)
        recorder.finish("no_issues")

    run = _runs(options)[0]
    assert run["outcome"] == "no_issues"
    assert not run["failure_class"]


def test_a_second_finish_does_not_replace_the_first(options):
    with editor_run_scope():
        recorder = EditorRunRecorder(options)
        recorder.finish("llm_repaired")
        recorder.finish("blocked", failure_class="provider_blocked")

    assert _runs(options)[0]["outcome"] == "llm_repaired"


def test_a_recorder_outside_any_scope_still_works(options):
    recorder = EditorRunRecorder(options)
    recorder.finish("warnings_only")

    assert _runs(options)[0]["outcome"] == "warnings_only"


def test_runs_from_separate_scopes_do_not_close_each_other(options):
    with editor_run_scope():
        outer = EditorRunRecorder(options)
        with editor_run_scope():
            EditorRunRecorder({**options, "chunk_index": 4})
        # The inner scope closed only its own run.
        assert outer.finished is False
        outer.finish("no_issues")

    by_chunk = {run["chunk_index"]: run["outcome"] for run in _runs(options)}
    assert by_chunk == {3: "no_issues", 4: "review_required"}


@pytest.mark.asyncio
async def test_the_editor_boundary_closes_a_run_the_pass_crashed_on(
    options, monkeypatch,
):
    # Every review goes through this boundary, which is why the scope lives
    # there rather than around each give-up path in the state machine.
    import src.core.translator as translator
    from src.core.editor import EditorService

    async def exploding_pass(*_args, **_kwargs):
        EditorRunRecorder(options)
        raise RuntimeError("editor exploded")

    monkeypatch.setattr(
        translator, "_run_chunk_reflection_pass_impl", exploding_pass
    )

    with pytest.raises(RuntimeError, match="editor exploded"):
        await EditorService().review_chunk()

    assert _runs(options)[0]["outcome"] == "review_required"


@pytest.mark.asyncio
async def test_a_v1_job_that_gives_up_on_the_contract_records_why(tmp_path):
    """The legacy contract keeps the draft, but the run still ended nowhere.

    A v1 job does not wrap the draft for review, so this path used to return
    the text and walk away, leaving the row at 'running'. The draft is still
    kept -- only the record changes.
    """

    db_path = str(tmp_path / "jobs.db")
    db = Database(db_path)
    assert db.create_job("job-v1", "txt", {})

    class Client:
        async def generate_async(self, **_kwargs):
            return SimpleNamespace(
                content='{"status": needs_repair, "issues": }',
                prompt_tokens=100, completion_tokens=20, total_tokens=120,
            )

    result = await run_chunk_reflection_pass(
        source_chunk="Source.", draft_translation="Valid draft.",
        target_language="English", model_name="editor", llm_client=Client(),
        prompt_options={
            "translation_id": "job-v1", "jobs_db_path": db_path,
            "chunk_index": 0, "source_language": "English",
        },
    )

    assert result == "Valid draft."
    run = db.get_editor_diagnostics("job-v1")["runs"][0]
    assert run["outcome"] == "review_required"
    assert run["failure_class"] in {"contract_parse", "contract_incomplete"}
    db.close()
