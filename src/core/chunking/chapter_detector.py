"""Deterministic chapter-boundary detection for translation chunking.

Structured formats should pass their heading kind (for example ``h1`` or
``heading2``). Plain text falls back to conservative multilingual heading
patterns. The detector intentionally avoids guessing from arbitrary short
lines: a false chapter boundary is more harmful than an undetected one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from src.core.chunking.decorative_separator import is_decorative_separator


_STRUCTURAL_HEADING_RE = re.compile(r"^(?:h|heading)([1-6])$", re.IGNORECASE)
_NUMBER = r"(?:\d+|[ivxlcdm]+|[a-z])"
_LATIN_CHAPTER_RE = re.compile(
    rf"^(?:"
    rf"chapter|chapitre|cap[ií]tulo|kapitel|capitolo|hoofdstuk|"
    rf"rozdzia[lł]|глава|гл\.?|bab|chương"
    rf")\s+{_NUMBER}(?:\s*(?:[:.\-–—]\s*|\s+).*)?$",
    re.IGNORECASE,
)
_LATIN_SECTION_RE = re.compile(
    rf"^(?:"
    rf"part|book|volume|vol\.?|"
    rf"partie|livre|tome|"
    rf"parte|libro|"
    rf"teil|buch|band|"
    rf"parte|libro|volume|"
    rf"часть|том"
    rf")\s+{_NUMBER}(?:\s*(?:[:.\-–—]\s*|\s+).*)?$",
    re.IGNORECASE,
)
_NAMED_BOUNDARY_RE = re.compile(
    r"^(?:"
    r"prologue|epilogue|introduction|interlude|afterword|"
    r"prologue|épilogue|introduction|interlude|"
    r"prólogo|prologo|epílogo|epilogo|introducción|introduccion|interludio|"
    r"prolog|epilog|einleitung|zwischenspiel|"
    r"пролог|эпилог|введение"
    r")(?:\s*(?:[:.\-–—]\s*|\s+).*)?$",
    re.IGNORECASE,
)
_CJK_CHAPTER_RE = re.compile(
    r"^(?:第[0-9０-９一二三四五六七八九十百千万零〇两兩]+[章节章節回卷部篇幕]|"
    r"序章|序幕|終章|终章|最終章|最终章|後日談|后日谈|プロローグ|エピローグ)"
    r"(?:\s*[:：.\-–—]?\s*.*)?$"
)
_KOREAN_CHAPTER_RE = re.compile(
    r"^(?:제\s*[0-9０-９일이삼사오육칠팔구십백천]+\s*[장화부권]|"
    r"서장|서막|종장|최종장|프롤로그|에필로그)"
    r"(?:\s*[:：.\-–—]?\s*.*)?$"
)
_GENERIC_DIGIT_NUMBER = r"(?:\d+|[一二三四五六七八九十百千万零〇两兩]+)"
_GENERIC_ROMAN_NUMBER = r"(?:[ivxlcdm]+)"
_GENERIC_LABEL_NUMBER_RE = re.compile(
    rf"^(?P<label>[^\W\d_][^\d:：.\-–—]{{1,40}}?)\s+"
    rf"(?P<number>{_GENERIC_DIGIT_NUMBER})"
    rf"(?:\s*[:：.\-–—]\s*\S.*)?$",
    re.IGNORECASE,
)
_GENERIC_LABEL_ROMAN_RE = re.compile(
    rf"^(?P<label>[^\W\d_][^\d:：.\-–—]{{1,40}}?)\s+"
    rf"(?P<number>{_GENERIC_ROMAN_NUMBER})"
    rf"(?:\s*[:：.\-–—]\s*\S.*)?$",
    re.IGNORECASE,
)
_GENERIC_NUMBER_TITLE_RE = re.compile(
    rf"^(?P<number>{_GENERIC_DIGIT_NUMBER}|{_GENERIC_ROMAN_NUMBER})"
    rf"\s*[.、:：\-–—]\s*[^\s.。…!?！？,，;；:：\-–—].+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChapterRange:
    """A half-open paragraph range belonging to one semantic chapter."""

    start: int
    end: int
    title: str = ""
    heading_index: Optional[int] = None


def _clean_heading(text: str) -> str:
    return " ".join((text or "").strip().split())


def is_chapter_heading(
    text: str,
    kind: Optional[str] = None,
    llm_client: Optional[Any] = None,
) -> bool:
    """Return whether a paragraph is a reliable chapter boundary.

    Uses fast deterministic regex matching first ($0 cost). If uncertain and an
    llm_client is provided, falls back to lightweight LLM boundary verification.
    """
    cleaned = _clean_heading(text)
    if not cleaned:
        return False
    if is_decorative_separator(cleaned):
        return False

    structural_match = _STRUCTURAL_HEADING_RE.fullmatch((kind or "").strip())
    if structural_match:
        return int(structural_match.group(1)) <= 3

    # Plain-text headings are normally a single short line.
    if "\n" in (text or "") or len(cleaned) > 160:
        return False

    if bool(
        _LATIN_CHAPTER_RE.fullmatch(cleaned)
        or _LATIN_SECTION_RE.fullmatch(cleaned)
        or _NAMED_BOUNDARY_RE.fullmatch(cleaned)
        or _CJK_CHAPTER_RE.fullmatch(cleaned)
        or _KOREAN_CHAPTER_RE.fullmatch(cleaned)
    ):
        return True

    # LLM fallback for ambiguous short standalone lines when regexes miss non-standard titles
    if llm_client and 3 <= len(cleaned) <= 100:
        return verify_chapter_boundary_with_llm(cleaned, llm_client)

    return False


def verify_chapter_boundary_with_llm(heading_candidate: str, llm_client: Any) -> bool:
    """Lightweight LLM fallback to classify non-standard chapter headings (~20 tokens)."""
    if not heading_candidate or not llm_client:
        return False
    try:
        prompt = (
            f"Determine if the following text is a structural chapter heading, section title, or episode header.\n"
            f"Text: \"{heading_candidate}\"\n"
            f"Respond strictly with 'YES' if it is a chapter/section heading, or 'NO' if it is normal narrative text."
        )
        res_call = None
        if hasattr(llm_client, "make_request"):
            res_call = llm_client.make_request(prompt)
        elif hasattr(llm_client, "generate"):
            res_call = llm_client.generate(prompt)
        elif hasattr(llm_client, "translate_text"):
            res_call = llm_client.translate_text(prompt)

        if res_call:
            raw = getattr(res_call, "content", str(res_call)).strip().upper()
            return "YES" in raw
    except Exception:
        pass
    return False


def _generic_heading_family(text: str) -> Optional[str]:
    """Return a language-independent repeated heading family, if any.

    A single generic match is never trusted. ``find_chapter_ranges`` requires
    at least two headings with the same family, which supports labels such as
    Turkish "Bölüm 1" or arbitrary numbered-title conventions without turning
    isolated prose like "Day 1 was difficult" into a chapter boundary.
    """
    cleaned = _clean_heading(text)
    if not cleaned or "\n" in (text or "") or len(cleaned) > 120:
        return None
    if is_decorative_separator(cleaned):
        return None

    label_match = (
        _GENERIC_LABEL_NUMBER_RE.fullmatch(cleaned)
        or _GENERIC_LABEL_ROMAN_RE.fullmatch(cleaned)
    )
    if label_match:
        label = " ".join(label_match.group("label").casefold().split())
        if 1 <= len(label.split()) <= 4:
            return f"label:{label}"

    if _GENERIC_NUMBER_TITLE_RE.fullmatch(cleaned):
        return "numbered-title"
    return None


def find_chapter_ranges(
    paragraphs: Sequence[str],
    kinds: Optional[Sequence[str]] = None,
    llm_client: Optional[Any] = None,
) -> List[ChapterRange]:
    """Split paragraph indices into stable chapter ranges.

    A preface before the first detected heading is kept as its own range.
    When no heading is detected, the entire document is one semantic range;
    the normal token chunker may still split that range if it is oversized.
    """
    if not paragraphs:
        return []

    heading_indices = []
    generic_candidates = {}
    for index, paragraph in enumerate(paragraphs):
        kind = kinds[index] if kinds is not None and index < len(kinds) else None
        if is_chapter_heading(paragraph, kind, llm_client):
            heading_indices.append(index)
            continue
        family = _generic_heading_family(paragraph)
        if family:
            generic_candidates.setdefault(family, []).append(index)

    # Generic patterns are accepted only when the document repeats the same
    # convention. This is language-independent and deliberately conservative.
    for indices in generic_candidates.values():
        if len(indices) >= 2:
            heading_indices.extend(indices)
    heading_indices = sorted(set(heading_indices))

    if not heading_indices:
        return [ChapterRange(0, len(paragraphs))]

    starts = list(heading_indices)
    if starts[0] > 0:
        starts.insert(0, 0)

    ranges: List[ChapterRange] = []
    heading_index_set = set(heading_indices)
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(paragraphs)
        heading_index = start if start in heading_index_set else None
        title = _clean_heading(paragraphs[start]) if heading_index is not None else ""
        ranges.append(ChapterRange(start, end, title, heading_index))
    return ranges
