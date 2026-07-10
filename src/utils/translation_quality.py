"""Language-aware deterministic validation for translated chunk drafts."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

from src.utils.language_profiles import get_language_profile


_WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)
_PROTECTED_PATTERN = re.compile(
    r"<[^>]+>|\[\[?[^\]\n]+\]?\]|\{\{?[^}\n]+\}?\}|`[^`]+`|"
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_SOURCE_RESIDUE_TERMS = {
    "aunt", "brother", "captain", "chief", "commander", "dad", "daughter",
    "elder", "father", "grandfather", "grandmother", "junior", "lord",
    "master", "mom", "mother", "mister", "mistress", "professor", "senior",
    "sister", "son", "teacher", "uncle", "younger",
}


@dataclass(frozen=True)
class ResidueFinding:
    """One high-confidence source-language span that survived in a draft."""

    source_span: str
    draft_span: str
    confidence: float
    reason: str
    severity: str = "major"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_span": self.source_span,
            "draft_span": self.draft_span,
            "confidence": self.confidence,
            "reason": self.reason,
            "severity": self.severity,
        }


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold().strip()


def _without_protected_markup(text: str) -> str:
    return _PROTECTED_PATTERN.sub(" ", str(text or ""))


def _token_script(token: str) -> str:
    counts = {"latin": 0, "cjk": 0, "hangul": 0, "rtl": 0, "thai": 0, "other": 0}
    for char in token:
        codepoint = ord(char)
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            counts["latin"] += 1
        elif 0x4E00 <= codepoint <= 0x9FFF or 0x3400 <= codepoint <= 0x4DBF:
            counts["cjk"] += 1
        elif 0x3040 <= codepoint <= 0x30FF:
            counts["cjk"] += 1
        elif 0xAC00 <= codepoint <= 0xD7AF or 0x1100 <= codepoint <= 0x11FF:
            counts["hangul"] += 1
        elif 0x0590 <= codepoint <= 0x08FF:
            counts["rtl"] += 1
        elif 0x0E00 <= codepoint <= 0x0E7F:
            counts["thai"] += 1
        elif char.isalpha():
            counts["other"] += 1
    return max(counts, key=counts.get) if any(counts.values()) else "other"


def _protected_term_set(
    protected_terms: Optional[Iterable[str]],
    glossary_terms: Optional[Dict[str, str]],
) -> set[str]:
    protected = {_normalized(item) for item in protected_terms or [] if item}
    for source, target in (glossary_terms or {}).items():
        source_key = _normalized(source)
        target_key = _normalized(target)
        if source_key and source_key == target_key:
            protected.add(source_key)
    return protected


def find_source_residue(
    source_text: str,
    draft_text: str,
    *,
    source_language: str = "",
    target_language: str = "",
    protected_terms: Optional[Iterable[str]] = None,
    glossary_terms: Optional[Dict[str, str]] = None,
    max_findings: int = 8,
) -> List[ResidueFinding]:
    """Return conservative, high-confidence untranslated-source findings."""

    source_profile = get_language_profile(source_language)
    target_profile = get_language_profile(target_language)
    if (
        source_profile.code != "generic"
        and source_profile.code == target_profile.code
    ):
        return []
    source_clean = _without_protected_markup(source_text)
    draft_clean = _without_protected_markup(draft_text)
    if not source_clean.strip() or not draft_clean.strip():
        return []
    draft_normalized = _normalized(draft_clean)
    protected = _protected_term_set(protected_terms, glossary_terms)
    source_tokens = _WORD_RE.findall(source_clean)
    findings: List[ResidueFinding] = []
    seen = set()

    def add(span: str, confidence: float, reason: str) -> None:
        key = _normalized(span)
        if not key or key in seen or key in protected:
            return
        seen.add(key)
        findings.append(ResidueFinding(span, span, confidence, reason))

    script_patterns = {
        "cjk": r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]{2,}",
        "hangul": r"[\u1100-\u11ff\uac00-\ud7af]{2,}",
        "rtl": r"[\u0590-\u08ff]{2,}",
        "thai": r"[\u0e00-\u0e7f]{2,}",
    }
    source_key = _normalized(source_clean)
    for script, pattern in script_patterns.items():
        if script == target_profile.script:
            continue
        for span in re.findall(pattern, draft_clean):
            if _normalized(span) in source_key:
                add(span, 0.97, f"source-script span remained in {target_profile.name} output")

    # Repeated source phrases are high signal even for same-script language pairs.
    for size in (4, 3, 2):
        for index in range(0, max(0, len(source_tokens) - size + 1)):
            phrase_tokens = source_tokens[index:index + size]
            if any(len(token) < 2 for token in phrase_tokens):
                continue
            if all(token[:1].isupper() for token in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            phrase_key = _normalized(phrase)
            if phrase_key in protected or any(
                protected_term and protected_term in phrase_key
                for protected_term in protected
            ):
                continue
            if re.search(rf"(?<!\w){re.escape(phrase_key)}(?!\w)", draft_normalized):
                add(phrase, 0.96, "copied multi-word source span")

    for token in source_tokens:
        key = _normalized(token)
        if len(key) < 4 or key in protected:
            continue
        if not re.search(rf"(?<!\w){re.escape(key)}(?!\w)", draft_normalized):
            continue
        if key in _SOURCE_RESIDUE_TERMS:
            add(token, 0.99, "untranslated kinship, title, or social address term")
            continue
        script = _token_script(token)
        if script != "other" and script != target_profile.script:
            add(token, 0.97, f"source-script token remained in {target_profile.name} output")
            continue
        if token[:1].isupper():
            # Capitalized source labels are usually names, brands, or approved lore.
            continue

    return findings[:max(0, int(max_findings))]


def residue_findings_to_editor_issues(
    findings: Iterable[ResidueFinding],
) -> List[Dict[str, Any]]:
    """Convert deterministic residue findings into mandatory editor issues."""

    return [{
        "category": "untranslated_source",
        "severity": finding.severity,
        "source_quote": finding.source_span,
        "draft_quote": finding.draft_span,
        "instruction": (
            f"Translate the surviving source span '{finding.draft_span}' into the "
            "target language using the active relationship and glossary context."
        ),
        "draft_replacement": None,
        "glossary_update": None,
        "term_replacement": None,
        "deterministic": True,
    } for finding in findings]


def validate_editor_repair(
    repaired_text: str,
    issues: Iterable[Dict[str, Any]],
    *,
    source_text: str,
    source_language: str,
    target_language: str,
    protected_terms: Optional[Iterable[str]] = None,
    glossary_terms: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Validate exact local replacements and deterministic residue removal."""

    errors: List[str] = []
    repaired_key = _normalized(repaired_text)
    for issue in issues or []:
        replacement = issue.get("draft_replacement") if isinstance(issue, dict) else None
        if not isinstance(replacement, dict):
            continue
        draft_span = str(replacement.get("draft") or "").strip()
        target_span = str(replacement.get("replacement") or "").strip()
        if draft_span and _normalized(draft_span) in repaired_key:
            errors.append(f"flagged draft span remains: {draft_span}")
        if target_span and _normalized(target_span) not in repaired_key:
            errors.append(f"required replacement is missing: {target_span}")
    remaining = find_source_residue(
        source_text,
        repaired_text,
        source_language=source_language,
        target_language=target_language,
        protected_terms=protected_terms,
        glossary_terms=glossary_terms,
    )
    errors.extend(
        f"source residue remains: {finding.draft_span}"
        for finding in remaining
    )
    return errors
