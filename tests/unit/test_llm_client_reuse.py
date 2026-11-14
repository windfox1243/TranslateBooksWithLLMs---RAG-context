"""The provider keeps one HTTP client per event loop.

LLMProvider._get_client() used asyncio without importing it. The NameError
was swallowed by its own `except Exception`, so the cached client was thrown
away on every call: each request opened a new httpx.AsyncClient, never closed
the previous one, and connection pooling never happened.
"""

import asyncio

import httpx
import pytest

from src.core.llm.providers.gemini import GeminiProvider


@pytest.mark.asyncio
async def test_client_is_reused_within_one_event_loop():
    provider = GeminiProvider(api_key="YOUR_API_KEY_HERE", model="gemini-3.6-flash")
    first = await provider._get_client()
    second = await provider._get_client()
    try:
        assert first is second
    finally:
        await first.aclose()


@pytest.mark.asyncio
async def test_injected_client_is_kept():
    provider = GeminiProvider(api_key="YOUR_API_KEY_HERE", model="gemini-3.6-flash")
    injected = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    provider._client = injected
    try:
        assert await provider._get_client() is injected
    finally:
        await injected.aclose()


def test_client_from_a_closed_loop_is_replaced():
    provider = GeminiProvider(api_key="YOUR_API_KEY_HERE", model="gemini-3.6-flash")

    old_loop = asyncio.new_event_loop()
    first = old_loop.run_until_complete(provider._get_client())
    old_loop.run_until_complete(first.aclose())
    old_loop.close()

    new_loop = asyncio.new_event_loop()
    try:
        second = new_loop.run_until_complete(provider._get_client())
        assert second is not first
        new_loop.run_until_complete(second.aclose())
    finally:
        new_loop.close()
