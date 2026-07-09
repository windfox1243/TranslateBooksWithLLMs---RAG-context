"""Shared script-aware text matching helpers."""

from __future__ import annotations

import re
from typing import Any

from src.utils.language_profiles import LanguageProfile, get_language_profile


LATIN_BOUNDARY_CHARS = r"A-Za-z0-9À-ÖØ-öø-ÿ_'’-"
RTL_BOUNDARY_CHARS = r"\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff"
CJK_HANGUL_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7a3]")
RTL_RE = re.compile(f"[{RTL_BOUNDARY_CHARS}]")


def plain_key(value: Any) -> str:
    """Casefold and collapse whitespace for stable exact comparisons."""

    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def compact_key(value: Any) -> str:
    """Remove whitespace after casefolding."""

    return re.sub(r"\s+", "", plain_key(value))


def has_cjk_or_hangul(value: Any) -> bool:
    return bool(CJK_HANGUL_RE.search(str(value or "")))


def has_rtl(value: Any) -> bool:
    return bool(RTL_RE.search(str(value or "")))


def _latin_boundary_pattern(label: str) -> str:
    escaped = re.escape(label).replace(r"\ ", r"\s+")
    return rf"(?<![{LATIN_BOUNDARY_CHARS}]){escaped}(?![{LATIN_BOUNDARY_CHARS}])"


def _rtl_boundary_pattern(label: str) -> str:
    escaped = re.escape(label).replace(r"\ ", r"\s+")
    return rf"(?<![{RTL_BOUNDARY_CHARS}]){escaped}(?![{RTL_BOUNDARY_CHARS}])"


def reference_mentions_label(
    label: str,
    reference_text: str,
    language: str | None = None,
) -> bool:
    """Return whether reference text mentions label with script-safe boundaries."""

    clean = str(label or "").strip()
    reference = str(reference_text or "")
    if not clean or not reference.strip():
        return False

    profile = get_language_profile(language)
    if has_cjk_or_hangul(clean):
        return compact_key(clean) in compact_key(reference)
    if has_rtl(clean):
        return bool(re.search(_rtl_boundary_pattern(clean), reference, re.IGNORECASE))
    if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", clean):
        return bool(re.search(_latin_boundary_pattern(clean), reference, re.IGNORECASE))
    if profile.match_policy.no_space_script:
        return compact_key(clean) in compact_key(reference)
    return plain_key(clean) in plain_key(reference)


def active_label_matches_name(
    active_label: str,
    rule_name: str,
    language: str | None = None,
) -> bool:
    """Safe active-character match for projected addressing/profile rows."""

    active = plain_key(active_label)
    name = plain_key(rule_name)
    if not active or not name:
        return False
    if active == name:
        return True
    if has_cjk_or_hangul(active):
        return compact_key(active) in compact_key(name)
    if has_rtl(active):
        return reference_mentions_label(active, name, language)
    active_tokens = active.split()
    name_tokens = name.split()
    return len(active_tokens) == 1 and len(active) >= 3 and active in name_tokens


def glossary_term_occurrences(
    term: str,
    chunk: str,
    *,
    case_sensitive: bool = False,
    language: str | None = None,
) -> int:
    """Count safe glossary term occurrences in a source chunk."""

    if not term or not chunk:
        return 0
    needle = term if case_sensitive else term.casefold()
    haystack = chunk if case_sensitive else chunk.casefold()
    if has_cjk_or_hangul(term) or get_language_profile(language).match_policy.no_space_script:
        return haystack.count(needle)
    if has_rtl(term):
        flags = 0 if case_sensitive else re.IGNORECASE
        return len(re.findall(_rtl_boundary_pattern(term), chunk, flags))
    if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", term):
        flags = 0 if case_sensitive else re.IGNORECASE
        return len(re.findall(_latin_boundary_pattern(term), chunk, flags))
    return haystack.count(needle)


def display_language_profile(language: str | None) -> LanguageProfile:
    """Small readability wrapper for call sites that log profile metadata."""

    return get_language_profile(language)
