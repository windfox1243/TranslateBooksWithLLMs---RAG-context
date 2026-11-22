"""
A transient failure fetching the OpenRouter model catalog must not
disable the reasoning overrides for the life of the server process: the
fetch is skipped for CATALOG_RETRY_SECONDS, then retried.
"""

import asyncio
import sys
import types
import weakref

import httpx
import pytest

FAKE_KEY = "YOUR_API_KEY_HERE"


def _openrouter():
    from src.core.llm.providers.openrouter import OpenRouterProvider
    return (OpenRouterProvider(api_key=FAKE_KEY, model="openai/gpt-4o"),
            OpenRouterProvider, "_load_model_catalog", "_model_catalog")


@pytest.mark.asyncio
@pytest.mark.parametrize("build", [_openrouter], ids=["openrouter"])
async def test_failed_catalog_fetch_is_retried_after_the_ttl(monkeypatch, build):
    provider, cls, loader_name, cache_attr = build()
    # Fresh class-side state for this test only.
    monkeypatch.setattr(cls, cache_attr, None)
    monkeypatch.setattr(cls, f"{cache_attr}_failed_at", None)
    monkeypatch.setattr(cls, f"{cache_attr}_loop_locks", weakref.WeakKeyDictionary())

    clock = [1000.0]
    # Fake clock for the provider module only; the event loop keeps the real one.
    monkeypatch.setattr(sys.modules[cls.__module__], "time",
                        types.SimpleNamespace(monotonic=lambda: clock[0]))

    responses = [httpx.Response(503, text="temporarily unavailable"),
                 httpx.Response(200, json={"data": [{"id": "GPT-4o"}]})]
    requests = []

    def handler(request):
        requests.append(request)
        return responses.pop(0)

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    load = getattr(provider, loader_name)
    try:
        assert await load() is None
        assert len(requests) == 1

        # Within the TTL: no new request, still unknown.
        clock[0] += cls.CATALOG_RETRY_SECONDS - 1
        assert await load() is None
        assert len(requests) == 1

        # Past the TTL: fetched again, and the success is cached.
        clock[0] += 2
        catalog = await load()
        assert catalog is not None and "GPT-4o" in catalog
        assert len(requests) == 2
        assert getattr(cls, f"{cache_attr}_failed_at") is None
    finally:
        await provider.close()


def test_catalog_lock_works_across_job_event_loops(monkeypatch):
    """Each web job runs its own event loop. A class-level asyncio.Lock bound
    to the first job's loop made every later contended fetch raise."""
    from src.core.llm.providers.openrouter import OpenRouterProvider as cls

    monkeypatch.setattr(cls, "_model_catalog_failed_at", None)
    monkeypatch.setattr(cls, "_model_catalog_loop_locks", weakref.WeakKeyDictionary())

    async def handler(request):
        await asyncio.sleep(0.01)  # keep the lock held so the second caller waits
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    async def one_job():
        cls._model_catalog = None  # force a fetch in this job
        provider = cls(api_key=FAKE_KEY, model="openai/gpt-4o")
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            results = await asyncio.gather(
                provider._load_model_catalog(), provider._load_model_catalog(),
            )
        finally:
            await provider.close()
        return results

    monkeypatch.setattr(cls, "_model_catalog", None)
    for _ in range(2):
        first, second = asyncio.run(one_job())
        assert first is not None and "m" in first
        assert second is first
