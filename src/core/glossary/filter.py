"""
Per-chunk glossary filtering.

Latin terms are matched with word boundaries (so "Fan" does not match "Fantasy").
CJK terms are matched as substrings (no word boundary concept in CJK scripts).
The filter returns only the subset of glossary entries that actually appear in
the chunk, sorted by source-term length (longest first) to handle overlaps.

When the per-chunk cap is hit, the kept subset is selected by occurrence count
(most frequent first, length as tiebreaker), then re-sorted by length for
output stability.

Source terms may declare alternative forms separated by '|' to handle inflected
languages (e.g. "Москва|Москве|Москвы|Москвой -> Moscou"). The filter matches
the entry if ANY of the alternatives appears in the chunk; occurrence counts
are summed across alternatives.
"""
import re
from typing import Dict, List, Tuple

from src.core.glossary.models import GlossaryConfig

_CJK_RE = re.compile(r'[぀-ゟ゠-ヿ一-鿿가-힯㐀-䶿]')


def _is_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _has_word_char_at_edge(term: str) -> bool:
    """True if the term starts or ends with a regex \\w character (Latin/digit/underscore)."""
    if not term:
        return False
    return bool(re.match(r'\w', term[0])) or bool(re.match(r'\w', term[-1]))


def _split_alternatives(source: str) -> List[str]:
    """Split a source term on '|' into non-empty stripped alternatives."""
    if "|" not in source:
        stripped = source.strip()
        return [stripped] if stripped else []
    return [a.strip() for a in source.split("|") if a.strip()]


def _max_alt_length(source: str) -> int:
    """Length used for sort: the longest alternative wins (overlap handling)."""
    alts = _split_alternatives(source)
    return max((len(a) for a in alts), default=0)


def _count_alternative(alt: str, chunk: str, haystack: str, flags: int, case_sensitive: bool) -> int:
    """Count occurrences of a single alternative form in the chunk."""
    if _is_cjk(alt) or not _has_word_char_at_edge(alt):
        needle = alt if case_sensitive else alt.lower()
        return haystack.count(needle)
    pattern = r'\b' + re.escape(alt) + r'\b'
    return len(re.findall(pattern, chunk, flags))


def filter_glossary(
    chunk: str,
    glossary_terms: Dict[str, str],
    config: GlossaryConfig = None,
) -> Tuple[Dict[str, str], bool]:
    """
    Return only the glossary entries that appear in the chunk.

    Args:
        chunk: The source text to scan.
        glossary_terms: {source_term: translated_term}. A source_term may
            declare alternative inflected forms separated by '|'.
        config: GlossaryConfig (max_entries cap, case sensitivity).

    Returns:
        (filtered_terms, capped) where filtered_terms preserves order
        (longest source first) and capped is True if the cap was hit.
    """
    if not chunk or not glossary_terms:
        return {}, False

    config = config or GlossaryConfig()
    flags = 0 if config.case_sensitive else re.IGNORECASE

    # Sort by longest alternative descending so longer terms (e.g.
    # "Li Fanqing") are checked before shorter prefixes (e.g. "Li Fan").
    sorted_terms = sorted(
        glossary_terms.items(),
        key=lambda kv: _max_alt_length(kv[0]),
        reverse=True,
    )

    matched: List[Tuple[str, str, int]] = []  # (source, target, occurrence_count)
    haystack = chunk if config.case_sensitive else chunk.lower()

    for source, target in sorted_terms:
        alternatives = _split_alternatives(source)
        if not alternatives:
            continue

        total_count = sum(
            _count_alternative(alt, chunk, haystack, flags, config.case_sensitive)
            for alt in alternatives
        )
        if total_count > 0:
            matched.append((source, target, total_count))

    capped = False
    if config.max_entries and len(matched) > config.max_entries:
        capped = True
        # When capping, keep the most frequent terms first (length as
        # tiebreaker so longer-and-rarer beats shorter-and-rarer). This
        # is more useful than the previous length-only cut, which could
        # drop a high-frequency 2-char CJK name in favor of 50 longer
        # but rarer entries.
        kept = set(
            (s, t) for s, t, _ in
            sorted(matched, key=lambda x: (x[2], _max_alt_length(x[0])), reverse=True)[: config.max_entries]
        )
        # Preserve the original length-desc order in the output so the
        # rendered block stays predictable for the LLM.
        matched = [(s, t, c) for s, t, c in matched if (s, t) in kept]

    return {s: t for s, t, _ in matched}, capped
