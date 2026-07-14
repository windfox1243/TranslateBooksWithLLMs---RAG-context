"""Small value objects returned by the Senior Editor pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional


class EditorTranslation(str):
    """String-compatible translation carrying an independent quality state."""

    def __new__(
        cls,
        value: str,
        *,
        quality_status: str = "passed",
        diagnostics: Optional[Dict[str, Any]] = None,
    ):
        instance = super().__new__(cls, value)
        instance.quality_status = quality_status
        instance.editor_validation = dict(diagnostics or {})
        return instance


class ReflectionValidationError(RuntimeError):
    """Compatibility error for callers that still use exception-based review flow."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Optional[Dict[str, Any]] = None,
        draft_translation: str = "",
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}
        self.draft_translation = draft_translation


def review_required_translation(
    value: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> EditorTranslation:
    return EditorTranslation(
        value,
        quality_status="review_required",
        diagnostics=diagnostics,
    )
