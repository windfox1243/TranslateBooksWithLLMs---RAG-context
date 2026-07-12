"""Language-aware deterministic validation for translated chunk drafts."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
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
@dataclass(frozen=True)
class ResidueFinding:
    """One high-confidence source-language span that survived in a draft."""

    source_span: str
    draft_span: str
    confidence: float
    reason: str
    severity: str = "major"
    blocking: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_span": self.source_span,
            "draft_span": self.draft_span,
            "confidence": self.confidence,
            "reason": self.reason,
            "severity": self.severity,
            "blocking": self.blocking,
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


def _matches_protected_term(value: str, protected: Iterable[str]) -> bool:
    """Return whether a residue candidate is or belongs to a protected term."""

    key = _normalized(value)
    if not key:
        return False
    return any(
        term and key == term
        for term in protected
    )


def _mask_protected_intervals(text: str, protected: Iterable[str]) -> str:
    """Mask complete protected spans before residue extraction."""
    result = str(text or "")
    for term in sorted({str(item) for item in protected if item}, key=len, reverse=True):
        pattern = re.escape(term)
        if term[:1].isalnum() and term[-1:].isalnum():
            pattern = rf"(?<!\w){pattern}(?!\w)"
        result = re.sub(
            pattern,
            lambda match: " " * len(match.group(0)),
            result,
            flags=re.IGNORECASE,
        )
    return result


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
    protected = _protected_term_set(protected_terms, glossary_terms)
    source_clean = _mask_protected_intervals(
        _without_protected_markup(source_text), protected,
    )
    draft_clean = _mask_protected_intervals(
        _without_protected_markup(draft_text), protected,
    )
    if not source_clean.strip() or not draft_clean.strip():
        return []
    draft_normalized = _normalized(draft_clean)
    source_tokens = _WORD_RE.findall(source_clean)
    findings: List[ResidueFinding] = []
    seen = set()

    def add(
        span: str,
        confidence: float,
        reason: str,
        *,
        blocking: bool = True,
    ) -> None:
        key = _normalized(span)
        if not key or key in seen or _matches_protected_term(key, protected):
            return
        seen.add(key)
        findings.append(ResidueFinding(
            span, span, confidence, reason,
            severity="major" if blocking else "minor",
            blocking=blocking,
        ))

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

    for term in source_profile.residue_social_terms:
        term_key = _normalized(term)
        if not term_key or _matches_protected_term(term_key, protected):
            continue
        pattern = rf"(?<!\w){re.escape(term_key)}(?!\w)"
        if re.search(pattern, source_key) and re.search(pattern, draft_normalized):
            add(term, 0.99, "untranslated kinship, title, or social address term")

    # Repeated source phrases are high signal even for same-script language pairs.
    phrase_sizes = (4, 3) if source_profile.script == target_profile.script else (4, 3, 2)
    for size in phrase_sizes:
        for index in range(0, max(0, len(source_tokens) - size + 1)):
            phrase_tokens = source_tokens[index:index + size]
            if any(len(token) < 2 for token in phrase_tokens):
                continue
            if all(token[:1].isupper() for token in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            phrase_key = _normalized(phrase)
            if _matches_protected_term(phrase_key, protected):
                continue
            if re.search(rf"(?<!\w){re.escape(phrase_key)}(?!\w)", draft_normalized):
                add(
                    phrase, 0.96, "copied multi-word source span",
                    blocking=source_profile.script != target_profile.script,
                )

    for token in source_tokens:
        key = _normalized(token)
        if len(key) < 4 or _matches_protected_term(key, protected):
            continue
        if not re.search(rf"(?<!\w){re.escape(key)}(?!\w)", draft_normalized):
            continue
        if key in source_profile.residue_social_terms:
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
        "issue_id": f"residue-{index}",
        "category": "untranslated_source",
        "severity": finding.severity,
        "confidence": finding.confidence,
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
    } for index, finding in enumerate(findings, start=1)]


def validate_issue_locators(
    draft_text: str,
    issues: Iterable[Dict[str, Any]],
) -> List[str]:
    """Validate that every direct edit identifies one exact draft location."""
    draft = str(draft_text or "")
    draft_folded = draft.casefold()
    errors: List[str] = []
    for issue in issues or []:
        replacement = issue.get("draft_replacement")
        repair_kind = str(issue.get("repair_kind") or "").casefold()
        if repair_kind and repair_kind != "local_replace":
            continue
        if not isinstance(replacement, dict):
            if repair_kind == "local_replace":
                errors.append(
                    f"replacement_missing:{str(issue.get('issue_id') or 'issue')}"
                )
            continue
        quote = str(issue.get("draft_quote") or "").strip()
        issue_id = str(issue.get("issue_id") or "issue")
        if not quote:
            errors.append(f"locator_missing:{issue_id}")
            continue
        count = draft_folded.count(quote.casefold())
        if count == 0:
            errors.append(f"locator_missing:{issue_id}")
        elif count > 1:
            errors.append(f"locator_ambiguous:{issue_id}")
            continue
        local_quote = quote.casefold()
        old = str(replacement.get("draft") or "").strip().casefold()
        new = str(replacement.get("replacement") or "").strip().casefold()
        if not old or not new:
            errors.append(f"replacement_missing:{issue_id}")
        elif old == new:
            continue
        elif local_quote.count(old) == 0:
            errors.append(f"replacement_outside_locator:{issue_id}")
        elif local_quote.count(old) > 1:
            errors.append(f"replacement_ambiguous_in_locator:{issue_id}")
    return errors


def apply_local_editor_patches(
    draft_text: str,
    issues: Iterable[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]], List[str]]:
    """Apply non-overlapping exact editor substitutions without an LLM rewrite."""
    draft = str(draft_text or "")
    patches = []
    unresolved: List[Dict[str, Any]] = []
    errors: List[str] = []
    for issue in issues or []:
        repair_kind = str(issue.get("repair_kind") or "").casefold()
        if repair_kind and repair_kind != "local_replace":
            unresolved.append(issue)
            continue
        replacement = issue.get("draft_replacement") or {}
        if not isinstance(replacement, dict):
            unresolved.append(issue)
            continue
        quote = str(issue.get("draft_quote") or "").strip()
        old = str(replacement.get("draft") or "").strip()
        new = str(replacement.get("replacement") or "").strip()
        issue_id = str(issue.get("issue_id") or "issue")
        if not quote or not old or not new:
            unresolved.append(issue)
            continue
        folded = draft.casefold()
        quote_folded = quote.casefold()
        if folded.count(quote_folded) != 1:
            unresolved.append(issue)
            continue
        quote_start = folded.index(quote_folded)
        quote_end = quote_start + len(quote)
        local = draft[quote_start:quote_end]
        local_folded = local.casefold()
        if local.count(old) == 1:
            local_offset = local.index(old)
        elif local_folded.count(old.casefold()) == 1:
            local_offset = local_folded.index(old.casefold())
        else:
            unresolved.append(issue)
            continue
        start = quote_start + local_offset
        end = start + len(old)
        patches.append((start, end, new, issue_id))

    patches.sort(key=lambda item: item[0])
    for previous, current in zip(patches, patches[1:]):
        if current[0] < previous[1]:
            errors.append(
                f"local_patch_conflict:{previous[3]}:{current[3]}"
            )
    if errors:
        return draft, list(issues or []), errors
    result = draft
    for start, end, new, _issue_id in reversed(patches):
        result = result[:start] + new + result[end:]
    return result, unresolved, errors


def validate_editor_repair(
    repaired_text: str,
    issues: Iterable[Dict[str, Any]],
    *,
    draft_text: str = "",
    source_text: str,
    source_language: str,
    target_language: str,
    protected_terms: Optional[Iterable[str]] = None,
    glossary_terms: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Validate exact local replacements and deterministic residue removal."""

    errors: List[str] = []
    draft_key = _normalized(draft_text)
    repaired_key = _normalized(repaired_text)
    for issue in issues or []:
        replacement = issue.get("draft_replacement") if isinstance(issue, dict) else None
        if not isinstance(replacement, dict):
            continue
        draft_span = str(replacement.get("draft") or "").strip()
        target_span = str(replacement.get("replacement") or "").strip()
        quote = str(issue.get("draft_quote") or draft_span).strip()
        quote_key = _normalized(quote)
        draft_span_key = _normalized(draft_span)
        target_span_key = _normalized(target_span)

        if not draft_key:
            if draft_span_key and draft_span_key in repaired_key:
                errors.append(f"replacement_span_remaining: {draft_span}")
            if target_span_key and target_span_key not in repaired_key:
                errors.append(f"replacement_missing: {target_span}")
            continue

        quote_count = draft_key.count(quote_key) if quote_key else 0
        if quote_count != 1:
            errors.append(
                f"issue_locator_{'missing' if quote_count == 0 else 'ambiguous'}: {quote}"
            )
            continue

        quote_start = draft_key.index(quote_key)
        quote_end = quote_start + len(quote_key)
        local_quote = draft_key[quote_start:quote_end]
        local_occurrences = [
            match.start()
            for match in re.finditer(re.escape(draft_span_key), local_quote)
        ] if draft_span_key else []
        requested_occurrence = replacement.get("occurrence_index")
        if requested_occurrence is not None:
            try:
                occurrence_index = int(requested_occurrence) - 1
            except (TypeError, ValueError):
                occurrence_index = -1
        elif len(local_occurrences) == 1 or local_quote.startswith(draft_span_key):
            occurrence_index = 0
        elif local_quote.endswith(draft_span_key):
            occurrence_index = len(local_occurrences) - 1
        else:
            occurrence_index = -1
        local_span_offset = (
            local_occurrences[occurrence_index]
            if 0 <= occurrence_index < len(local_occurrences)
            else -1
        )
        if local_span_offset < 0:
            errors.append(f"replacement_not_located_in_issue: {draft_span}")
            continue

        span_start = quote_start + local_span_offset
        span_end = span_start + len(draft_span_key)
        changed_locally = False
        replacement_in_change = False
        matcher = SequenceMatcher(None, draft_key, repaired_key, autojunk=False)
        for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
            if tag == "equal" or old_end <= span_start or old_start >= span_end:
                continue
            changed_locally = True
            nearby_start = max(0, new_start - len(target_span_key) - 8)
            nearby_end = min(
                len(repaired_key),
                new_end + len(target_span_key) + len(quote_key) + 8,
            )
            if target_span_key and target_span_key in repaired_key[nearby_start:nearby_end]:
                replacement_in_change = True

        old_draft_count = draft_key.count(draft_span_key) if draft_span_key else 0
        new_draft_count = repaired_key.count(draft_span_key) if draft_span_key else 0
        old_target_count = draft_key.count(target_span_key) if target_span_key else 0
        new_target_count = repaired_key.count(target_span_key) if target_span_key else 0
        if not changed_locally or new_draft_count >= old_draft_count:
            errors.append(f"replacement_not_applied_locally: {draft_span}")
        if target_span_key and not (
            replacement_in_change or new_target_count > old_target_count
        ):
            errors.append(f"replacement_missing_locally: {target_span}")
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
        if finding.blocking
    )
    return errors


def validate_plain_refinement_structure(draft_text: str, repaired_text: str) -> Optional[str]:
    """Validate non-language structural tokens used by plain-text adapters."""

    placeholder_re = re.compile(r"\[\[?[^\]\n]+\]?\]|\{\{?[^}\n]+\}?\}|<[^>]+>")
    before = placeholder_re.findall(str(draft_text or ""))
    after = placeholder_re.findall(str(repaired_text or ""))
    if before != after:
        return "adapter_placeholder_sequence_changed"
    separator_re = re.compile(r"(?m)^\s*([=*_\-~#])\1{2,}\s*$")
    separators_before = [match.group(0).strip() for match in separator_re.finditer(draft_text)]
    separators_after = [match.group(0).strip() for match in separator_re.finditer(repaired_text)]
    if separators_before != separators_after:
        return "adapter_decorative_separator_sequence_changed"
    return None
