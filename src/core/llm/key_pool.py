"""
API key pool with throttle tracking and round-robin rotation.

Used by cloud LLM providers to support multiple API keys for the same provider.
On HTTP 429, the failing key is marked throttled and the next available key is
selected on the next request — failover happens without sleeping when possible.

Compatibility:
    - A pool with a single key behaves identically to today's single-key string.
    - Mutating operations share a lock across provider instances with the same
      provider/key set, so cooldowns survive provider re-creation.
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Union


@dataclass
class _KeyState:
    """Internal: throttle state for a single key.

    `throttled_until` is a `time.monotonic()` timestamp; 0 means available.
    Monotonic time avoids wall-clock-drift bugs when comparing expiry against
    the current moment.
    """
    throttled_until: float = 0.0


@dataclass
class _SharedPoolState:
    """Mutable state shared by provider instances with the same key set."""

    states: Dict[str, _KeyState]
    cursor: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


_SHARED_STATES: Dict[Tuple[str, str], _SharedPoolState] = {}


def _shared_state_key(provider_name: str, keys: List[str]) -> Tuple[str, str]:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(hashlib.sha256(key.encode("utf-8")).digest())
        digest.update(b"\0")
    return provider_name, digest.hexdigest()


class KeyPool:
    """Round-robin pool of API keys with per-key throttle tracking.

    Typical usage from a provider (note the two separate counters — rotating
    on 429 must not consume a transient-retry attempt, see issue #217):

        attempt = 0
        rate_limit_events = 0
        while attempt < MAX_TRANSLATION_ATTEMPTS:
            current_key = await self._key_pool.acquire()
            headers = {"Authorization": f"Bearer {current_key}"}
            try:
                response = await client.post(url, headers=headers, ...)
                ...
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    rate_limit_events += 1
                    await handle_rate_limit(
                        self._key_pool, current_key, e.response.headers,
                        rate_limit_events, MAX_TRANSLATION_ATTEMPTS,
                    )
                    continue
                attempt += 1
                ...
    """

    def __init__(
        self,
        keys: Union[str, Iterable[str]],
        provider_name: str = "unknown",
    ):
        """
        Args:
            keys: A single key string or an iterable of keys. Empty strings and
                duplicates are silently dropped (preserving original order).
            provider_name: Logical name used in log messages and RateLimitError.

        Raises:
            ValueError: If no non-empty key is provided.
        """
        if isinstance(keys, str):
            keys = [keys]

        seen = set()
        cleaned: List[str] = []
        for k in keys:
            if k and k not in seen:
                cleaned.append(k)
                seen.add(k)

        if not cleaned:
            raise ValueError(
                f"KeyPool for '{provider_name}' requires at least one non-empty key"
            )

        self._keys: List[str] = cleaned
        self._shared_key = _shared_state_key(provider_name, cleaned)
        shared = _SHARED_STATES.get(self._shared_key)
        if shared is None:
            shared = _SharedPoolState({k: _KeyState() for k in cleaned})
            _SHARED_STATES[self._shared_key] = shared
        else:
            shared.states = {
                key: shared.states.get(key, _KeyState())
                for key in cleaned
            }
            if cleaned:
                shared.cursor %= len(cleaned)
        self._shared = shared
        self._provider_name = provider_name

    @classmethod
    def clear_shared_state(cls) -> None:
        """Clear process-local shared cooldowns.

        This is intended for tests and administrative resets. Production code
        normally keeps the registry so new provider instances do not retry a
        key that a previous instance just saw rate-limited.
        """
        _SHARED_STATES.clear()

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def peek(self) -> str:
        """Return the next key WITHOUT advancing the cursor or locking.

        Cheap, sync. Used by code paths outside translation (e.g. listing
        available models, context detection) that just need *a* valid key.
        """
        now = time.monotonic()
        n = len(self._keys)
        for offset in range(n):
            idx = (self._shared.cursor + offset) % n
            key = self._keys[idx]
            if self._shared.states[key].throttled_until <= now:
                return key
        return min(self._keys, key=lambda k: self._shared.states[k].throttled_until)

    def index_of(self, key: str) -> int:
        """1-based index of `key` for human-readable logging. 0 if unknown."""
        try:
            return self._keys.index(key) + 1
        except ValueError:
            return 0

    async def acquire(self) -> str:
        """Acquire the next key, preferring non-throttled ones.

        Round-robin among non-throttled keys. If every key is throttled,
        returns the one with the earliest expiry — the caller is responsible
        for sleeping or raising via the rate-limit handler.

        Always returns a key; never blocks.
        """
        with self._shared.lock:
            now = time.monotonic()
            n = len(self._keys)
            for offset in range(n):
                idx = (self._shared.cursor + offset) % n
                key = self._keys[idx]
                if self._shared.states[key].throttled_until <= now:
                    self._shared.cursor = (idx + 1) % n
                    return key
            # All throttled — return the one that recovers soonest.
            return min(self._keys, key=lambda k: self._shared.states[k].throttled_until)

    async def mark_throttled(self, key: str, until_monotonic: float) -> None:
        """Mark `key` as throttled until `until_monotonic` (`time.monotonic()`).

        No-op if `key` is not in the pool. We take the max of existing and new
        expiry so a longer existing throttle isn't shortened by a fresh 429.
        """
        with self._shared.lock:
            state = self._shared.states.get(key)
            if state is not None:
                state.throttled_until = max(state.throttled_until, until_monotonic)

    async def has_available(self) -> bool:
        """True if at least one key is currently non-throttled."""
        with self._shared.lock:
            now = time.monotonic()
            return any(s.throttled_until <= now for s in self._shared.states.values())

    async def time_until_next_available(self) -> float:
        """Seconds until the earliest key becomes available.

        Returns 0.0 if any key is currently available.
        """
        with self._shared.lock:
            now = time.monotonic()
            soonest = min(
                (s.throttled_until for s in self._shared.states.values()),
                default=now,
            )
            return max(0.0, soonest - now)
