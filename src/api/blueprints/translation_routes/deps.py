"""The dependencies every translation route handler closes over."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class TranslationRouteDeps:
    """Per-blueprint dependencies, passed to each domain's register().

    Frozen because the route handlers read these on every request: a blueprint
    whose dependencies could be swapped after registration would change the
    behaviour of already-registered handlers.
    """

    state_manager: Any
    start_translation_job: Callable[..., Any]
    output_dir: str
    socketio: Optional[Any]
    uploads_dir: Path
