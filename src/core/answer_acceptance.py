"""Acceptance rules for a well-tagged answer (TXT / SRT / PGN).

A well-tagged answer is not necessarily a translation: it can be a refusal
wrapped in the tags, a cut-off fragment, a runaway commentary or the source
copied back. ``tagged_answer_problem`` applies the same rules the raw fallback
and the EPUB/DOCX survival guard apply, so the generic translator never saves
such an answer as done. (Ported from upstream fe17ae1c, where it lives in
``request_core``.)
"""
import re
from typing import Optional

from src.core.translator import _REFUSAL_OPENER_RE, _REFUSAL_TOPIC_RE, _weighted_length

_MARKER_RE = re.compile(r"\[(?:id)?\d+\]")
_SOUND_TAG_RE = re.compile(r"\[[^\[\]\n]*\]|\([^()\n]*\)|[♪♫♩♬]")
_SAN_TOKEN_RE = re.compile(
    r"^(?:\d+\.(?:\.\.)?|O-O(?:-O)?|0-0(?:-0)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|"
    r"[KQRBN][a-h1-8]?x?[a-h][1-8]|1-0|0-1|1/2-1/2|\*)[+#]?[!?]{0,2}$"
)
_WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*", re.UNICODE)
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
ECHO_SINGLE_TOKEN_MAX_LENGTH = 24
"""A one-word line up to this long may legitimately come back unchanged."""
ECHO_NAME_MAX_WORDS = 3
"""A line of at most this many capitalized words is treated as a proper name."""
REFUSAL_MAX_SHARE_OF_SOURCE = 0.6
"""A refusal-shaped answer is rejected only when it is this much shorter than
the source: a short dialogue line that happens to read like a refusal ("I'm
sorry, I can't help you.") is a translation of a line of the same length."""


def _normalize_for_echo(text: str) -> str:
    return " ".join((text or "").split()).casefold()


def _line_may_stay_unchanged(line: str) -> bool:
    """True for a line a translator may legitimately return as is."""
    text = _MARKER_RE.sub(" ", line)
    if not _LETTER_RE.search(text):
        return True  # numbers, punctuation, music notes only
    if not _LETTER_RE.search(_SOUND_TAG_RE.sub(" ", text)):
        return True  # pure sound tags: "[MUSIC]", "(laughs)", "♪ ... ♪" without words
    tokens = text.split()
    if tokens and all(_SAN_TOKEN_RE.match(tok.strip(",;:()")) for tok in tokens
                      if tok.strip(",;:()")):
        return True  # chess move text
    words = _WORD_RE.findall(text)
    if len(words) == 1 and len(words[0]) <= ECHO_SINGLE_TOKEN_MAX_LENGTH:
        return True  # a single short token
    if 0 < len(words) <= ECHO_NAME_MAX_WORDS and all(w[:1].isupper() for w in words):
        return True  # a proper name ("John Smith", "Paris")
    return False


def source_may_stay_unchanged(source: str) -> bool:
    """True when every non-empty line of ``source`` may come back unchanged.

    Numbers, punctuation, music notes, pure sound tags ("[MUSIC]",
    "(laughs)"), chess move text, a single short token and a short proper
    name are correct untranslated in any target language, so an answer equal
    to such a source is not an echo.
    """
    lines = [ln for ln in (source or "").splitlines() if ln.strip()]
    return all(_line_may_stay_unchanged(ln) for ln in lines)


def languages_differ(source_language: Optional[str], target_language: Optional[str]) -> bool:
    """True when both languages are known and are not the same."""
    src = (source_language or "").strip().casefold()
    tgt = (target_language or "").strip().casefold()
    return bool(src and tgt and src not in ("auto", "auto-detect") and src != tgt)


def tagged_answer_problem(answer: str, source: str, *, reject_echo: bool) -> Optional[str]:
    """Why an extracted (tagged) answer cannot be saved as a translation, or None.

    Rules, in order:
      * a refusal-shaped answer far shorter than the source;
      * the text-survival bounds shared with EPUB/DOCX (too short, runaway);
      * with ``reject_echo`` (source and target languages differ), an answer
        identical to the source after whitespace and case normalisation,
        unless the source may legitimately stay unchanged
        (:func:`source_may_stay_unchanged`).
    """
    from src.core.epub.text_survival import text_survival_problem

    text = (answer or "").strip()
    head = _MARKER_RE.sub("", text[:300]).strip()
    if (_REFUSAL_OPENER_RE.search(head) and _REFUSAL_TOPIC_RE.search(head)
            and not _REFUSAL_OPENER_RE.search(_MARKER_RE.sub("", source or "").strip())
            and _weighted_length(text) < REFUSAL_MAX_SHARE_OF_SOURCE * _weighted_length(source or "")):
        return "the answer looks like a refusal"
    problem = text_survival_problem(source or "", text)
    if problem:
        return problem
    if (reject_echo and _normalize_for_echo(text) == _normalize_for_echo(source)
            and not source_may_stay_unchanged(source)):
        return "the answer is the source text copied back"
    return None
