"""Senior-editor diagnostics persistence boundary."""

from .base import DatabaseRepository


class EditorRepository(DatabaseRepository):
    prefixes = ("editor",)
    methods = frozenset({
        "add_editor_attempt", "create_editor_run", "finish_editor_run",
        "get_editor_diagnostics",
    })
