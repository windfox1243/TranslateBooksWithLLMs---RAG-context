"""Senior Editor public contracts.

The compatibility exports keep existing callers stable while new code imports
editor-specific behavior from a dedicated package instead of the translation
request module.
"""

from .conformance import (
    NARRATOR_CONFORMANCE_VERSION,
    audit_narrator_conformance,
    conformance_editor_issues,
    conformance_fingerprint,
    resolve_narrator_policy,
)
from .contracts import (
    EditorTranslation,
    ReflectionValidationError,
    review_required_translation,
)

__all__ = [
    "NARRATOR_CONFORMANCE_VERSION",
    "audit_narrator_conformance",
    "conformance_editor_issues",
    "conformance_fingerprint",
    "resolve_narrator_policy",
    "EditorTranslation",
    "ReflectionValidationError",
    "review_required_translation",
]
