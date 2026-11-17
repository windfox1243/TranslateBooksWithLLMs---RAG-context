"""Count the LLM requests a translation job makes, by purpose.

A job's cost is mostly the number of model calls per chunk: the draft is one,
and source-context analysis, lore consolidation, relationship judging and the
Senior Editor each add their own. Those calls come from a dozen call sites, so
counting them at the call sites would drift. Every one of them goes through
`LLMClient.make_request`, which records into the ledger of the job that is
running.

The ledger and the current purpose live in context variables. A job opens its
ledger at the start of `translate_file`; asyncio tasks copy the context when
they are created, so parallel chunk workers record into the same ledger, and
two jobs in different threads never see each other's counts. Code that makes
calls for something other than the draft marks itself with `llm_purpose`.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional

PURPOSE_TRANSLATION = "translation"
PURPOSE_CONTEXT = "context"
PURPOSE_EDITOR = "editor"
PURPOSE_REFINEMENT = "refinement"
PURPOSE_OTHER = "other"

_LEDGER: ContextVar[Optional["LLMCallLedger"]] = ContextVar(
    "llm_call_ledger", default=None,
)
_PURPOSE: ContextVar[str] = ContextVar("llm_call_purpose", default=PURPOSE_TRANSLATION)


@dataclass
class LLMCallLedger:
    """Request counts for one job, keyed by purpose."""

    counts: Dict[str, int] = field(default_factory=dict)

    def record(self, purpose: str) -> None:
        self.counts[purpose] = self.counts.get(purpose, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def summary(self, chunks: int) -> Dict[str, Any]:
        """Counts plus the average per chunk (None when no chunk finished)."""
        return {
            "llm_calls_total": self.total,
            "llm_calls_by_purpose": dict(sorted(self.counts.items())),
            "llm_calls_chunks": chunks,
            "llm_calls_per_chunk": (
                round(self.total / chunks, 2) if chunks > 0 else None
            ),
        }

    def describe(self, chunks: int) -> str:
        """One log line, e.g. '42 LLM calls for 12 chunks (3.5 per chunk: ...)'."""
        parts = ", ".join(
            f"{purpose} {count}" for purpose, count in sorted(self.counts.items())
        )
        if chunks > 0:
            head = (
                f"{self.total} LLM calls for {chunks} chunks "
                f"({self.total / chunks:.1f} per chunk"
            )
        else:
            head = f"{self.total} LLM calls (no chunk completed"
        return f"{head}{': ' + parts if parts else ''})"


def current_ledger() -> Optional[LLMCallLedger]:
    return _LEDGER.get()


@contextmanager
def job_ledger() -> Iterator[LLMCallLedger]:
    """Open a ledger for the job running in this context.

    A nested job (a refinement started from a translation, say) keeps
    recording into the outer ledger instead of hiding its calls in its own.
    """
    existing = _LEDGER.get()
    if existing is not None:
        yield existing
        return
    ledger = LLMCallLedger()
    token = _LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        _LEDGER.reset(token)


def record_llm_call() -> None:
    """Called once per request by LLMClient.make_request."""
    ledger = _LEDGER.get()
    if ledger is not None:
        ledger.record(_PURPOSE.get())


@contextmanager
def llm_call_purpose(purpose: str) -> Iterator[None]:
    token = _PURPOSE.set(purpose)
    try:
        yield
    finally:
        _PURPOSE.reset(token)


def llm_purpose(purpose: str) -> Callable:
    """Decorate an async function whose LLM calls serve `purpose`.

    Only async functions are accepted: a sync function that creates a request
    coroutine returns before the request runs, so the purpose would already be
    gone when make_request records it.
    """

    def decorate(fn: Callable) -> Callable:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"llm_purpose needs an async function, got {fn!r}")

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with llm_call_purpose(purpose):
                return await fn(*args, **kwargs)

        return wrapper

    return decorate
