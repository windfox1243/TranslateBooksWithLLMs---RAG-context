"""Cost presets and the per-job LLM call ledger."""

import asyncio
import importlib

import pytest

from src import config as _config
from src.core.cost_presets import COST_PRESETS, apply_cost_preset
from src.core.llm.call_ledger import (
    PURPOSE_CONTEXT,
    PURPOSE_EDITOR,
    current_ledger,
    job_ledger,
    llm_purpose,
)
from src.core.llm_client import LLMClient
from src.utils.novel_context.consolidation import _consolidation_interval
from src.utils.relationship_sync import relationship_llm_judge_enabled

translate_file_module = importlib.import_module("src.core.adapters.translate_file")


class _FakeProvider:
    async def generate(self, prompt, timeout=None, system_prompt=None):
        return None


def _client():
    client = LLMClient(provider_type="ollama", model="m", api_endpoint="http://x")
    client._provider = _FakeProvider()
    return client


# --- presets ---------------------------------------------------------------

def test_balanced_changes_nothing_but_records_its_name(monkeypatch):
    monkeypatch.setattr(_config, "COST_PRESET", "balanced", raising=False)
    assert COST_PRESETS["balanced"] == {}
    assert apply_cost_preset({"x": 1}) == {"x": 1, "cost_preset": "balanced"}


def test_economy_fills_only_what_the_job_left_unset():
    options = apply_cost_preset({"cost_preset": "economy", "reflection_mode": True})
    assert options["reflection_mode"] is True
    assert options["novel_context_update_interval"] == 3
    assert options["novel_context_consolidation_interval"] == 0
    assert options["use_relationship_llm_judge"] == "off"


def test_unknown_name_falls_back_to_the_configured_preset(monkeypatch):
    monkeypatch.setattr(_config, "COST_PRESET", "quality", raising=False)
    options = apply_cost_preset({"cost_preset": "cheap"})
    assert options["cost_preset"] == "quality"
    assert options["reflection_mode"] is True


def test_invalid_configured_preset_means_balanced(monkeypatch):
    monkeypatch.setattr(_config, "COST_PRESET", "nonsense", raising=False)
    assert apply_cost_preset(None) == {"cost_preset": "balanced"}


def test_start_request_applies_the_preset_before_its_own_defaults():
    from src.api.blueprints.translation_routes.helpers import (
        _prompt_options_from_start_request,
    )

    options = _prompt_options_from_start_request(
        {"prompt_options": {"cost_preset": "economy", "reflection_mode": True}}
    )
    # The route's own default ("selective") must not undo the preset...
    assert options["use_relationship_llm_judge"] == "off"
    # ...and the editor checkbox the user ticked still wins over the preset.
    assert options["reflection_mode"] is True


@pytest.mark.parametrize(
    "value, enabled",
    [("selective", True), ("always", True), (True, True), ("off", False),
     (False, False), (None, False), ("Selective ", True)],
)
def test_relationship_judge_switch(value, enabled):
    assert relationship_llm_judge_enabled({"use_relationship_llm_judge": value}) is enabled


def test_consolidation_interval_prefers_the_job_value(monkeypatch):
    monkeypatch.setattr(_config, "NOVEL_CONTEXT_CONSOLIDATION_INTERVAL", 5, raising=False)
    assert _consolidation_interval(0) == 0
    assert _consolidation_interval("7") == 7
    assert _consolidation_interval(None) == 5
    assert _consolidation_interval("bad") == 5


# --- ledger ----------------------------------------------------------------

@llm_purpose(PURPOSE_EDITOR)
async def _editor_call(client):
    await client.make_request("p")


@llm_purpose(PURPOSE_CONTEXT)
async def _context_call(client):
    await client.make_request("p")


@pytest.mark.asyncio
async def test_ledger_counts_calls_by_purpose_including_parallel_tasks():
    client = _client()
    with job_ledger() as ledger:
        await client.make_request("draft")
        await asyncio.gather(*(_editor_call(client) for _ in range(3)))
        await _context_call(client)
        # The purpose does not leak out of the decorated call.
        await client.make_request("draft again")
    assert ledger.counts == {"translation": 2, "editor": 3, "context": 1}
    assert ledger.summary(2)["llm_calls_per_chunk"] == 3.0
    assert current_ledger() is None


@pytest.mark.asyncio
async def test_calls_outside_a_job_are_not_recorded_anywhere():
    await _client().make_request("p")
    assert current_ledger() is None


@pytest.mark.asyncio
async def test_nested_job_records_into_the_outer_ledger():
    client = _client()
    with job_ledger() as outer:
        with job_ledger() as inner:
            await client.make_request("p")
        assert inner is outer
    assert outer.total == 1


def test_purpose_decorator_rejects_sync_functions():
    with pytest.raises(TypeError):
        llm_purpose(PURPOSE_EDITOR)(lambda: None)


# --- translate_file wrapper -------------------------------------------------

@pytest.mark.asyncio
async def test_job_wrapper_resolves_preset_in_place_and_reports_calls_per_chunk():
    client = _client()
    seen_options = {}

    async def fake_job(prompt_options=None, stats_callback=None, log_callback=None):
        seen_options.update(prompt_options)
        stats_callback({"completed_chunks": 2})  # restored from a checkpoint
        for _ in range(6):
            await client.make_request("p")
        await _editor_call(client)
        stats_callback({"completed_chunks": 5})
        return True

    job = translate_file_module._with_cost_preset_and_call_report(fake_job)
    logs = []
    options = {"cost_preset": "economy"}
    stats = []
    result = await job(
        prompt_options=options,
        stats_callback=stats.append,
        log_callback=lambda event, message, data=None: logs.append((event, message, data)),
    )

    assert result is True
    # The caller's own dict carries the resolved preset.
    assert options["novel_context_update_interval"] == 3
    assert seen_options["cost_preset"] == "economy"
    assert [s["completed_chunks"] for s in stats] == [2, 5]
    [(event, message, data)] = [entry for entry in logs if entry[0] == "llm_call_summary"]
    assert data["llm_calls_total"] == 7
    assert data["llm_calls_chunks"] == 3
    assert data["llm_calls_per_chunk"] == pytest.approx(2.33)
    assert data["llm_calls_by_purpose"] == {"editor": 1, "translation": 6}
    assert "7 LLM calls for 3 chunks" in message
    assert "economy" in message


@pytest.mark.asyncio
async def test_job_wrapper_reports_even_when_the_job_fails():
    async def failing_job(prompt_options=None, stats_callback=None, log_callback=None):
        await _client().make_request("p")
        raise RuntimeError("boom")

    job = translate_file_module._with_cost_preset_and_call_report(failing_job)
    logs = []
    with pytest.raises(RuntimeError):
        await job(log_callback=lambda event, message, data=None: logs.append(event))
    assert "llm_call_summary" in logs


@pytest.mark.asyncio
async def test_failed_chunks_count_in_the_per_chunk_average():
    client = _client()

    async def job_with_a_failure(prompt_options=None, stats_callback=None, log_callback=None):
        stats_callback({"completed_chunks": 0, "failed_chunks": 0, "processed_chunks": 0})
        for _ in range(8):
            await client.make_request("p")
        # TXT/SRT report failed chunks outside completed_chunks.
        stats_callback({"completed_chunks": 3, "failed_chunks": 1, "processed_chunks": 4})
        return False

    job = translate_file_module._with_cost_preset_and_call_report(job_with_a_failure)
    logs = []
    await job(log_callback=lambda event, message, data=None: logs.append((event, data)))
    [data] = [d for e, d in logs if e == "llm_call_summary"]
    assert data["llm_calls_chunks"] == 4
    assert data["llm_calls_per_chunk"] == 2.0
