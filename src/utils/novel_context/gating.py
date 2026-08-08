"""Per-job override for the context gating switch.

`BYPASS_CONTEXT_GATING` decides whether a gender the model guessed is allowed
into durable lore without deterministic evidence from the source. It is a
process-wide setting, but every translation job also carries its own
`bypass_context_gating` option -- and jobs run concurrently, one thread each,
up to `MAX_CONCURRENT_JOBS`.

Opening a context session used to apply that option by assigning to
`src.config.BYPASS_CONTEXT_GATING`, which is shared by the whole process. A
second job starting mid-run would flip the switch under the first one, and
because nothing restored it, the last job's choice stayed behind for every job
after it -- including jobs that never asked for it.

A ContextVar keeps the override on the flow that set it. A job thread starts
from an empty context, so it sees the configured default until it sets its own,
and the tasks the job spawns afterwards inherit its value. That is exactly the
scope this setting was always meant to have.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

_BYPASS_OVERRIDE: ContextVar[Optional[bool]] = ContextVar(
    "novel_context_bypass_gating",
    default=None,
)


def set_bypass_context_gating(value: Optional[bool]) -> None:
    """Override the gating switch for the current job flow.

    Passing None drops the override and falls back to the configured default.
    """
    _BYPASS_OVERRIDE.set(None if value is None else bool(value))


def bypass_context_gating() -> bool:
    """Whether gating is bypassed here: the job's choice, else the config default."""
    override = _BYPASS_OVERRIDE.get()
    if override is not None:
        return override

    from src import config as _config

    return bool(getattr(_config, "BYPASS_CONTEXT_GATING", True))
