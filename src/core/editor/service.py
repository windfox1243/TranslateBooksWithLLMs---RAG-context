"""Senior Editor orchestration boundary.

The service is intentionally small while the compatibility implementation is
migrated out of ``src.core.translator`` one cohesive stage at a time. Callers
depend on this service rather than the legacy module layout.
"""

from __future__ import annotations

from typing import Any


class EditorService:
    """Run the editor state machine and return a string-compatible outcome."""

    async def review_chunk(self, *args: Any, **kwargs: Any):
        from src.core.translator import _run_chunk_reflection_pass_impl

        return await _run_chunk_reflection_pass_impl(*args, **kwargs)
