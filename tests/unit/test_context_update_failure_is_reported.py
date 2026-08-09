"""A context update that produces nothing has to say so where the user looks."""
import pytest

from src.utils.novel_context.session import NovelContextSession


class _NoResponse:
    """An LLM client whose reply carries no content."""

    async def generate(self, **_kwargs):
        return type("Response", (), {"content": ""})()


class _Raises:
    async def generate(self, **_kwargs):
        raise RuntimeError("context window exceeded")


async def _run_update(client, log_callback):
    from src.utils.novel_context.updates import update_novel_context_chunk

    return await update_novel_context_chunk(
        llm_client=client,
        model_name="any-model",
        current_global_lore="# GLOBAL LORE\n",
        current_dynamic_state="",
        source_chunk="Alice steered the ship.",
        translated_chunk=None,
        source_language="English",
        target_language="Vietnamese",
        chunk_index=3,
        total_chunks=9,
        log_callback=log_callback,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client, parse_status",
    [(_NoResponse(), "empty_response"), (_Raises(), "update_failed")],
)
async def test_a_failed_update_reaches_the_job_log(client, parse_status):
    """It used to reach logger.error and stop.

    The job log went on reporting the context as committed, so a context file
    that stayed empty had no stated cause -- which is how a context window too
    small for the update prompt hid for an entire release.
    """
    events = []

    def log_callback(event, message, data=None):
        events.append((event, message, data or {}))

    lore, state, _ = await _run_update(client, log_callback)

    # The caller's state survives the failure untouched.
    assert lore == "# GLOBAL LORE\n"
    assert state == ""

    reported = [entry for entry in events if entry[0] == "novel_context_update_failed"]
    assert len(reported) == 1
    _event, message, data = reported[0]
    assert "unchanged" in message
    assert data["parse_status"] == parse_status
    assert data["chunk_index"] == 3


@pytest.mark.asyncio
async def test_the_raised_cause_is_carried_into_the_report():
    events = []
    await _run_update(_Raises(), lambda e, m, d=None: events.append((e, m, d or {})))

    data = next(d for e, _m, d in events if e == "novel_context_update_failed")
    assert "context window exceeded" in data["detail"]


@pytest.mark.asyncio
async def test_a_session_can_be_handed_a_different_updater(tmp_path):
    """The injectable seam that replaced the stringly-typed package lookup."""
    calls = []

    async def recording_update(**kwargs):
        calls.append(kwargs["chunk_index"])
        return ("# GLOBAL LORE\n- Alice: Female", "Alice is at the helm", [])

    session = NovelContextSession(
        path=tmp_path / "novel.txt",
        prompt_options={},
        global_lore="",
        dynamic_state="",
        update_chunk=recording_update,
    )

    await session.analyze_source(
        llm_client=object(),
        model_name="any-model",
        source_chunk="Alice steered the ship.",
        source_language="English",
        target_language="Vietnamese",
        chunk_index=2,
        total_chunks=5,
    )

    assert calls == [2]
    assert "Alice" in session.global_lore


def test_the_default_updater_is_not_bound_as_a_method(tmp_path):
    """A plain default would arrive as a bound method and eat an argument."""
    session = NovelContextSession(
        path=tmp_path / "novel.txt",
        prompt_options={},
        global_lore="",
        dynamic_state="",
    )

    assert getattr(session.update_chunk, "__self__", None) is None
