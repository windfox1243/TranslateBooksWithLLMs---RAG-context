"""Narrator-conformance boundary for the Senior Editor package."""

from src.utils.narrator_conformance import (
    NARRATOR_CONFORMANCE_VERSION,
    apply_narrator_conformance_patches,
    audit_narrator_conformance,
    conformance_editor_issues,
    conformance_fingerprint,
    resolve_narrator_policy,
)

__all__ = [
    "NARRATOR_CONFORMANCE_VERSION",
    "apply_narrator_conformance_patches",
    "audit_narrator_conformance",
    "conformance_editor_issues",
    "conformance_fingerprint",
    "resolve_narrator_policy",
]
