"""Deterministic final-output checks for narrator self-reference policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

from src.utils.language_profiles import get_language_profile
from src.utils.translation_quality import build_editor_segments


NARRATOR_CONFORMANCE_VERSION = 2


@dataclass(frozen=True)
class NarratorConformanceFinding:
    reason_code: str
    segment_id: str
    source_span: str
    target_span: str
    expected_form: str
    observed_form: str
    discourse_mode: str = "narration"
    blocking: bool = True
    target_start: int = -1
    target_end: int = -1
    alignment_confidence: str = "exact"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SOURCE_FIRST_PERSON = {
    "en": ("i", "me", "my", "mine", "myself", "we", "our", "ours"),
    "fr": ("je", "j'", "moi", "mon", "ma", "mes", "nous", "notre"),
    "es": ("yo", "me", "mi", "mis", "nosotros", "nuestro"),
    "de": ("ich", "mich", "mir", "mein", "wir", "unser"),
    "vi": ("tôi", "tớ", "mình", "ta", "tao", "chúng tôi", "chúng ta"),
    "ru": ("я", "мне", "меня", "мой", "мы", "наш"),
    "hi": ("मैं", "मुझे", "मेरा", "हम", "हमारा"),
    "nl": ("ik", "mij", "mijn", "wij", "we", "ons"),
    "it": ("io", "me", "mio", "noi", "nostro"),
    "pt": ("eu", "me", "meu", "nós", "nosso"),
    "pl": ("ja", "mnie", "mój", "my", "nasz"),
    "tr": ("ben", "beni", "benim", "biz", "bizim"),
    "ar": ("أنا", "لي", "نحن", "لنا"),
    "zh": ("我", "我们"),
    "ja": ("私", "僕", "俺", "われわれ"),
    "ko": ("저", "나", "우리"),
    "th": ("ฉัน", "ผม", "ดิฉัน", "เรา"),
}

_SOURCE_THIRD_PERSON = {
    "en": ("he", "him", "his", "she", "her", "hers", "they", "their"),
    "fr": ("il", "elle", "ils", "elles", "son", "sa", "ses", "leur"),
    "es": ("él", "ella", "ellos", "ellas", "su", "sus"),
    "de": ("er", "sie", "ihm", "ihn", "sein", "ihr"),
    "vi": ("anh ấy", "cô ấy", "họ", "của họ"),
    "ru": ("он", "она", "они", "его", "её", "их"),
    "hi": ("वह", "वे", "उसका", "उनका"),
    "nl": ("hij", "zij", "ze", "hem", "haar", "hun"),
}

_QUOTE_OPEN = ('"', "“", "„", "«", "『", "「", "《")
_THOUGHT_OPEN = ("(", "（")
_LETTER_PREFIXES = (
    "to:", "from:", "dear ", "sincerely", "regards", "subject:",
    "gửi:", "từ:", "trân trọng", "liên hệ:",
)


def _norm(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


def _language_code(language: str) -> str:
    return get_language_profile(language).code


def _is_non_narrative_line(line: str) -> bool:
    stripped = str(line or "").strip()
    folded = _norm(stripped)
    return bool(
        not stripped
        or stripped.startswith(_QUOTE_OPEN)
        or stripped.startswith(_THOUGHT_OPEN)
        or stripped.startswith(("—", "–", "- "))
        or folded.startswith(_LETTER_PREFIXES)
        or stripped in {"—", "---", "***"}
    )


def _count_terms(text: str, terms: Iterable[str], code: str) -> int:
    folded = _norm(text)
    count = 0
    for term in terms:
        key = _norm(term)
        if not key:
            continue
        if code in {"zh", "ja", "ko", "th"}:
            count += folded.count(key)
        else:
            count += len(re.findall(
                rf"(?<!\w){re.escape(key)}(?!\w)", folded,
            ))
    return count


def source_has_first_person_narration(
    source_text: str, source_language: str,
) -> bool:
    """Conservatively distinguish first-person narration from embedded thoughts."""

    code = _language_code(source_language)
    narrative = "\n".join(
        line for line in str(source_text or "").splitlines()
        if not _is_non_narrative_line(line)
    )
    first = _count_terms(narrative, _SOURCE_FIRST_PERSON.get(code, ()), code)
    third = _count_terms(narrative, _SOURCE_THIRD_PERSON.get(code, ()), code)
    return first >= 3 and first > third


def resolve_narrator_policy(
    *, target_language: str, db: Any = None, translation_id: str = "",
    chunk_index: int = 0, explicit_override: str = "",
) -> Dict[str, Any]:
    """Resolve locked/evidence profiles before the provisional language policy."""

    if explicit_override:
        return {
            "strategy": "explicit", "self_reference": explicit_override,
            "policy_source": "custom_instruction", "profile_revision": 0,
            "enforcement": "blocking",
        }
    candidates: List[Dict[str, Any]] = []
    if db is not None and translation_id and hasattr(db, "get_narrator_voice_profiles"):
        candidates = [
            item for item in db.get_narrator_voice_profiles(
                translation_id, effective_chunk_index=int(chunk_index),
                include_inactive=True,
            )
            if item.get("status") in {"active", "provisional"}
            and str(item.get("point_of_view") or "").casefold() == "first"
            and str(item.get("self_reference") or "").strip()
        ]
    if candidates:
        candidates.sort(key=lambda item: (
            not bool(item.get("is_locked")),
            item.get("status") != "active",
            -int(item.get("revision") or 0),
        ))
        chosen = candidates[0]
        return {
            "strategy": "explicit",
            "self_reference": str(chosen.get("self_reference") or "").strip(),
            "policy_source": (
                "locked_manual" if chosen.get("is_locked")
                else "accepted_profile" if chosen.get("status") == "active"
                else "provisional_language_profile"
            ),
            "profile_revision": int(chosen.get("revision") or 0),
            "profile_id": chosen.get("id"),
            "enforcement": (
                "blocking"
                if chosen.get("is_locked") or chosen.get("status") == "active"
                else "advisory"
            ),
        }
    default = get_language_profile(target_language).narrator_default_policy
    return {
        "strategy": default.strategy,
        "self_reference": default.self_reference,
        "policy_source": "language_default",
        "profile_revision": 0,
        "enforcement": "advisory",
    }


def _vietnamese_observed_forms(segment: str, expected: str) -> List[str]:
    folded = _norm(segment)
    forms = []
    strong = ("tớ", "tao")
    # These alternatives are unambiguous enough to validate mechanically in
    # narrative text. Kinship and social forms such as ``anh``, ``chị``,
    # ``em``, ``con``, ``mình`` and ``ta`` can be either first-, second-, or
    # third-person Vietnamese references. They require contextual review and
    # must never become deterministic blockers based on sentence position.
    contextual: tuple[str, ...] = ()
    for form in strong:
        if _norm(form) != _norm(expected) and re.search(
            rf"(?<!\w){re.escape(_norm(form))}(?!\w)", folded,
        ):
            forms.append(form)
    for form in contextual:
        if _norm(form) == _norm(expected):
            continue
        if re.search(
            rf"(?:^|[.!?…]\s+){re.escape(_norm(form))}(?!\w)", folded,
        ):
            forms.append(form)
    if _norm(expected) != "tôi" and re.search(r"(?<!\w)tôi(?!\w)", folded):
        forms.append("tôi")
    return list(dict.fromkeys(forms))


_QUOTE_PAIRS = {
    '"': '"', "“": "”", "„": "”", "«": "»", "『": "』", "「": "」",
}
_THOUGHT_PAIRS = {"(": ")", "（": "）"}


def _mask_embedded_discourse(text: str) -> str:
    """Blank quoted dialogue and parenthetical thoughts while preserving offsets."""

    raw = str(text or "")
    chars = list(raw)
    closing = ""
    start = -1
    for index, char in enumerate(raw):
        if not closing:
            if char in _QUOTE_PAIRS:
                closing = _QUOTE_PAIRS[char]
                start = index
            elif char in _THOUGHT_PAIRS:
                closing = _THOUGHT_PAIRS[char]
                start = index
            continue
        if char == closing:
            for offset in range(start, index + 1):
                if chars[offset] not in "\r\n":
                    chars[offset] = " "
            closing = ""
            start = -1
    if closing and start >= 0:
        for offset in range(start, len(chars)):
            if chars[offset] not in "\r\n":
                chars[offset] = " "
    return "".join(chars)


def _line_spans(text: str) -> List[Dict[str, Any]]:
    """Return non-empty line blocks with offsets preserved."""

    spans: List[Dict[str, Any]] = []
    for match in re.finditer(r"[^\r\n]+(?:\r?\n|$)", str(text or "")):
        value = match.group(0)
        if value.strip():
            spans.append({"start": match.start(), "end": match.end(), "text": value})
    if not spans and str(text or "").strip():
        spans.append({"start": 0, "end": len(str(text)), "text": str(text)})
    return spans


def _containing_span(
    spans: List[Dict[str, Any]], offset: int,
) -> tuple[int, Optional[Dict[str, Any]]]:
    for index, span in enumerate(spans):
        if int(span["start"]) <= offset < int(span["end"]):
            return index, span
    return -1, None


def _vietnamese_observed_occurrences(
    segment: str, expected: str,
) -> List[tuple[str, int, int]]:
    """Return exact strong-form offsets without normalizing text length."""

    results: List[tuple[str, int, int]] = []
    for form in ("tớ", "tao"):
        if _norm(form) == _norm(expected):
            continue
        for match in re.finditer(
            rf"(?<!\w){re.escape(form)}(?!\w)", str(segment or ""), re.IGNORECASE,
        ):
            results.append((form, match.start(), match.end()))
    if _norm(expected) != "tôi":
        for match in re.finditer(r"(?<!\w)tôi(?!\w)", str(segment or ""), re.IGNORECASE):
            results.append(("tôi", match.start(), match.end()))
    return results


def audit_narrator_conformance(
    *, source_text: str, target_text: str, source_language: str,
    target_language: str, file_type: str = "txt",
    dialogue_attribution: Optional[Dict[str, Any]] = None,
    db: Any = None, translation_id: str = "", chunk_index: int = 0,
    explicit_override: str = "",
) -> Dict[str, Any]:
    """Audit final narrative spans without changing dialogue or document content."""

    policy = resolve_narrator_policy(
        target_language=target_language, db=db, translation_id=translation_id,
        chunk_index=chunk_index, explicit_override=explicit_override,
    )
    result: Dict[str, Any] = {
        "status": "pass", "validator_version": NARRATOR_CONFORMANCE_VERSION,
        **policy, "reason_codes": [], "violating_segments": [],
    }
    if str(policy.get("strategy")) != "explicit" or not policy.get("self_reference"):
        result["status"] = "not_applicable"
        return result
    if str(file_type or "").casefold() == "srt" and not bool(
        (dialogue_attribution or {}).get("voice_over")
    ):
        result["status"] = "not_applicable"
        return result
    if not source_has_first_person_narration(source_text, source_language):
        result["status"] = "not_applicable"
        return result

    code = _language_code(target_language)
    expected = str(policy.get("self_reference") or "")
    findings: List[NarratorConformanceFinding] = []
    source_masked = _mask_embedded_discourse(source_text)
    target_masked = _mask_embedded_discourse(target_text)
    source_lines = _line_spans(source_masked)
    target_lines = _line_spans(target_masked)
    target_segments = build_editor_segments(target_masked)
    original_target = str(target_text or "")
    source_code = _language_code(source_language)
    enforcement = str(policy.get("enforcement") or "advisory")
    seen_findings = set()
    for segment in target_segments:
        masked_segment = str(segment.get("text") or "")
        masked_text = masked_segment.strip()
        text = original_target[
            int(segment.get("start") or 0):int(segment.get("end") or 0)
        ].strip()
        if not masked_text:
            continue
        if _is_non_narrative_line(text):
            continue
        target_line_index, _target_line = _containing_span(
            target_lines, int(segment.get("start") or 0),
        )
        exact_alignment = (
            target_line_index >= 0
            and len(source_lines) == len(target_lines)
            and target_line_index < len(source_lines)
        )
        source_line = source_lines[target_line_index] if exact_alignment else None
        source_is_first_person = bool(source_line and _count_terms(
            source_line.get("text") or "",
            _SOURCE_FIRST_PERSON.get(source_code, ()),
            source_code,
        ))
        occurrences = (
            _vietnamese_observed_occurrences(masked_segment, expected)
            if code == "vi" else []
        )
        for form, local_start, local_end in occurrences:
            absolute_start = int(segment.get("start") or 0) + local_start
            absolute_end = int(segment.get("start") or 0) + local_end
            key = (absolute_start, absolute_end, _norm(form), _norm(expected))
            if key in seen_findings:
                continue
            seen_findings.add(key)
            is_blocking = (
                enforcement == "blocking"
                and exact_alignment
                and source_is_first_person
            )
            findings.append(NarratorConformanceFinding(
                reason_code="narrator_self_reference_mismatch",
                segment_id=str(segment.get("segment_id") or ""),
                source_span=str((source_line or {}).get("text") or "").strip(),
                target_span=text,
                expected_form=expected,
                observed_form=form,
                blocking=is_blocking,
                target_start=absolute_start,
                target_end=absolute_end,
                alignment_confidence="exact" if exact_alignment else "ambiguous",
            ))
    if findings:
        result["status"] = (
            "fail" if any(item.blocking for item in findings)
            else "review_required"
        )
        result["reason_codes"] = sorted({item.reason_code for item in findings})
        result["violating_segments"] = [item.to_dict() for item in findings]
    return result


def apply_narrator_conformance_patches(
    target_text: str, audit: Dict[str, Any],
) -> tuple[str, List[Dict[str, Any]]]:
    """Apply exact blocking narrator-form substitutions outside embedded discourse."""

    text = str(target_text or "")
    expected = str(audit.get("self_reference") or "").strip()
    if not expected:
        return text, []
    patches: List[Dict[str, Any]] = []
    seen = set()
    for item in audit.get("violating_segments") or []:
        if not item.get("blocking") or item.get("alignment_confidence") != "exact":
            continue
        raw_start = item.get("target_start", -1)
        raw_end = item.get("target_end", -1)
        start = int(-1 if raw_start is None else raw_start)
        end = int(-1 if raw_end is None else raw_end)
        observed = str(item.get("observed_form") or "")
        key = (start, end, _norm(observed), _norm(expected))
        current = text[start:end] if 0 <= start < end <= len(text) else ""
        if key in seen or start < 0 or end <= start or current.casefold() != observed.casefold():
            continue
        seen.add(key)
        replacement = (
            expected[:1].upper() + expected[1:]
            if current[:1].isupper() else expected
        )
        patches.append({"start": start, "end": end, "replacement": replacement})
    for patch in sorted(patches, key=lambda item: item["start"], reverse=True):
        text = text[:patch["start"]] + patch["replacement"] + text[patch["end"]:]
    return text, list(reversed(patches))


def conformance_fingerprint(
    *, source_text: str, target_text: str, policy: Dict[str, Any],
) -> str:
    payload = json.dumps({
        "source": source_text, "target": target_text,
        "policy": {
            "strategy": policy.get("strategy"),
            "self_reference": policy.get("self_reference"),
            "profile_revision": policy.get("profile_revision"),
            "validator_version": NARRATOR_CONFORMANCE_VERSION,
        },
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def conformance_editor_issues(audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert deterministic findings into mandatory full-context repairs."""

    return [{
        "issue_id": f"narrator-{index + 1}",
        "segment_id": item.get("segment_id") or "",
        "category": "narrator_voice",
        "severity": "blocker" if item.get("blocking") else "major",
        "confidence": 1.0,
        "repair_kind": "review_only",
        "deterministic": bool(item.get("blocking")),
        "reason_code": item.get("reason_code"),
        "source_quote": item.get("source_span") or "",
        "draft_quote": item.get("target_span") or "",
        "instruction": (
            "Review the identified narrative span because its exact narrator "
            f"form could not be patched safely to '{item.get('expected_form')}'. "
            "Do not rewrite dialogue, letters, vocatives, or surrounding text."
        ),
        "draft_replacement": None,
        "glossary_update": None,
        "term_replacement": None,
    } for index, item in enumerate(audit.get("violating_segments") or [])]
