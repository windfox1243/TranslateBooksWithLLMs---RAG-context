"""REQUEST_INTERVAL spaces request starts per provider (ported from upstream dc2c76b9)."""

import asyncio

import pytest

from src.core.llm import rate_limit_handler as rlh


@pytest.fixture
def paced(monkeypatch):
    clock = [100.0]
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("src.config.REQUEST_INTERVAL", 6.5)
    monkeypatch.setattr(rlh.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rlh.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(rlh, "_last_request_start", {})
    return clock, slept


@pytest.mark.asyncio
async def test_requests_to_one_provider_are_spaced(paced):
    clock, slept = paced
    await rlh.wait_for_request_slot("gemini")
    await rlh.wait_for_request_slot("gemini")
    assert slept == [6.5]
    clock[0] += 10
    await rlh.wait_for_request_slot("gemini")
    assert slept == [6.5]  # the interval had already passed


@pytest.mark.asyncio
async def test_providers_are_paced_independently(paced):
    _clock, slept = paced
    await rlh.wait_for_request_slot("gemini")
    await rlh.wait_for_request_slot("mistral")
    assert slept == []


@pytest.mark.asyncio
async def test_zero_interval_never_waits(monkeypatch):
    monkeypatch.setattr("src.config.REQUEST_INTERVAL", 0.0)
    monkeypatch.setattr(rlh, "_last_request_start", {})
    await asyncio.wait_for(rlh.wait_for_request_slot("x"), timeout=1)
    await asyncio.wait_for(rlh.wait_for_request_slot("x"), timeout=1)
