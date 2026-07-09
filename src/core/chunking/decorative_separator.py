"""Decorative separator detection for chapter-aware chunking."""

from __future__ import annotations

import re


_SEPARATOR_CHARS = set("=-_*~#·•—–")
_SPACED_SEPARATOR_RE = re.compile(r"^(?:[=\-_*~#·•—–]\s*){3,}$")


def is_decorative_separator(text: str) -> bool:
    """Return whether text is only a repeated decorative divider."""
    stripped = (text or "").strip()
    if len(stripped) < 3:
        return False
    if any(char.isalnum() for char in stripped):
        return False

    compact = re.sub(r"\s+", "", stripped)
    if len(compact) < 3:
        return False
    if any(char not in _SEPARATOR_CHARS for char in compact):
        return False
    if len(set(compact)) == 1:
        return True
    return bool(_SPACED_SEPARATOR_RE.fullmatch(stripped))
