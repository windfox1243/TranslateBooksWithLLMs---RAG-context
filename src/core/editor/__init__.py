"""Senior Editor public contracts.

The compatibility exports keep existing callers stable while new code imports
editor-specific behavior from a dedicated package instead of the translation
request module.
"""

from .conformance import (
    NARRATOR_CONFORMANCE_VERSION,
    apply_narrator_conformance_patches,
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
from .service import EditorService

__all__ = [
    "NARRATOR_CONFORMANCE_VERSION",
    "apply_narrator_conformance_patches",
    "audit_narrator_conformance",
    "conformance_editor_issues",
    "conformance_fingerprint",
    "resolve_narrator_policy",
    "EditorTranslation",
    "ReflectionValidationError",
    "review_required_translation",
    "EditorService",
]
