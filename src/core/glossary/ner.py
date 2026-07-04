"""
NER-assisted glossary extraction (Phase 2 of the glossary plan).

Given a sample of source text, ask the configured LLM to propose recurring
named entities (characters, locations, sects, items) with a suggested target
translation. The user reviews the candidates before adding them to a glossary
— nothing is auto-applied.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("glossary.ner")

ALLOWED_CATEGORIES = {"character", "location", "organization", "item", "title", "other"}

NER_TAG_IN = "<NER_JSON>"
NER_TAG_OUT = "</NER_JSON>"
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_CJK_RE = re.compile(r"[぀-ゟ゠-ヿ一-鿿가-힯㐀-䶿]")
_RELATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "the", "to", "with",
}


def parse_ner_response(raw: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Permissive parser for NER output.

    Tries, in order:
      1. Content between <NER_JSON>...</NER_JSON> tags.
      2. Content inside the first markdown ```json fence.
      3. The longest balanced [...] JSON array.
      4. The longest balanced {...} JSON object (and pull a list out of any value).

    Returns (candidates, warnings). `candidates` is a list of dicts with at
    least `source` and one of `target`/`category` populated. `warnings` is a
    list of human-readable issues that the caller should surface.
    """
    if not raw:
        return [], ["empty LLM response"]

    text = _strip_thinking_blocks(raw).strip()
    warnings: List[str] = []

    payload = _extract_payload(text, warnings)
    if payload is None:
        return [], warnings + ["could not locate any JSON payload in response"]

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        repaired = _try_repair_json(payload)
        if repaired is None:
            return [], warnings + [f"JSON parse error: {e}"]
        try:
            data = json.loads(repaired)
            warnings.append("JSON was repaired before parsing (trailing comma or similar)")
        except json.JSONDecodeError as e2:
            return [], warnings + [f"JSON parse error after repair: {e2}"]

    items = _coerce_to_list_of_dicts(data, warnings)

    candidates: List[Dict[str, str]] = []
    seen_sources: set[str] = set()
    for entry in items:
        source = _str(entry.get("source") or entry.get("source_term"))
        target = _str(entry.get("target") or entry.get("translated_term") or entry.get("translation"))
        category = _str(entry.get("category") or entry.get("type") or "").lower()

        if not source:
            warnings.append("skipped entry without 'source'")
            continue
        if source in seen_sources:
            continue
        seen_sources.add(source)

        if category and category not in ALLOWED_CATEGORIES:
            warnings.append(f"unknown category '{category}' for '{source}' (kept as-is)")

        candidates.append({
            "source": source,
            "target": target,
            "category": category or "other",
        })

    return candidates, warnings


async def suggest_terms(
    text: str,
    source_language: str,
    target_language: str,
    llm_provider,
    max_chars: int = 6000,
    existing_glossary_terms: Optional[Dict[str, str]] = None,
    max_related_terms: int = 20,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Run the NER prompt against `text` (truncated to `max_chars`) using the
    given provider. Returns (candidates, warnings).
    """
    from src.prompts.prompts import generate_ner_extraction_prompt

    sample = text[:max_chars] if max_chars and len(text) > max_chars else text
    related_terms = related_existing_glossary_terms(
        sample,
        existing_glossary_terms or {},
        max_entries=max_related_terms,
    )
    prompt = generate_ner_extraction_prompt(
        sample,
        source_language,
        target_language,
        related_glossary_terms=related_terms,
    )

    response = await llm_provider.generate(prompt.user, system_prompt=prompt.system)
    if response is None:
        return [], ["LLM returned no response"]

    raw = getattr(response, "content", None) or str(response)
    candidates, warnings = parse_ner_response(raw)

    if not candidates:
        snippet = (raw or '').strip().replace('\n', ' ')[:400]
        logger.info(
            "NER returned 0 candidates (sample_chars=%d, response_chars=%d): %s",
            len(sample), len(raw or ''), snippet,
        )
        if not warnings:
            warnings.append("LLM returned a valid empty list — no recurring entities detected")

    return candidates, warnings


def related_existing_glossary_terms(
    text: str,
    glossary_terms: Dict[str, str],
    max_entries: int = 20,
) -> Dict[str, str]:
    """Return existing glossary entries that are useful hints for this sample.

    Exact source-term occurrences rank highest. For Latin-like terms, a shared
    meaningful token can also include a row, which helps compounds such as
    "zone X" reuse the established "zone" translation without injecting an
    entire large glossary into the NER prompt.
    """
    if not text or not glossary_terms or max_entries <= 0:
        return {}

    text_fold = text.casefold()
    text_tokens = set(_meaningful_tokens(text))
    scored: List[Tuple[int, int, int, str, str]] = []

    for source, target in glossary_terms.items():
        source = str(source or "").strip()
        target = str(target or "").strip()
        if not source or not target:
            continue

        alternatives = [alt.strip() for alt in source.split("|") if alt.strip()]
        if not alternatives:
            continue

        best_exact = 0
        best_overlap = 0
        best_length = 0
        for alt in alternatives:
            alt_fold = alt.casefold()
            if _contains_source_form(text_fold, alt_fold, alt):
                best_exact = 1
            alt_tokens = set(_meaningful_tokens(alt))
            overlap = len(alt_tokens & text_tokens)
            best_overlap = max(best_overlap, overlap)
            best_length = max(best_length, len(alt))

        if best_exact or best_overlap:
            scored.append((best_exact, best_overlap, best_length, source, target))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return {source: target for *_score, source, target in scored[:max_entries]}


def _contains_source_form(text_fold: str, alt_fold: str, original_alt: str) -> bool:
    if not alt_fold:
        return False
    if _CJK_RE.search(original_alt):
        return alt_fold in text_fold
    if _has_word_edge(original_alt):
        return re.search(r"\b" + re.escape(alt_fold) + r"\b", text_fold) is not None
    return alt_fold in text_fold


def _has_word_edge(text: str) -> bool:
    return bool(text and (re.match(r"\w", text[0]) or re.match(r"\w", text[-1])))


def _meaningful_tokens(text: str) -> List[str]:
    tokens = []
    for token in _TOKEN_RE.findall(str(text or "").casefold()):
        if len(token) <= 2 or token in _RELATION_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _strip_thinking_blocks(text: str) -> str:
    """Strip any <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _extract_payload(text: str, warnings: List[str]) -> Optional[str]:
    tag_match = re.search(
        re.escape(NER_TAG_IN) + r"\s*(.*?)\s*" + re.escape(NER_TAG_OUT),
        text,
        flags=re.DOTALL,
    )
    if tag_match:
        return tag_match.group(1).strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        warnings.append("NER tags missing — extracted from markdown code fence")
        return fence_match.group(1).strip()

    # Pick whichever balanced container appears FIRST. If the response is a
    # bare object that wraps the array (e.g. `{"entities": [...]}`), the
    # object's `{` precedes the array's `[`, so the object wins and the
    # `_coerce_to_list_of_dicts` unwrap path can produce the right warning.
    obj_pos = text.find("{")
    arr_pos = text.find("[")

    if obj_pos != -1 and (arr_pos == -1 or obj_pos < arr_pos):
        obj = _find_balanced(text, "{", "}")
        if obj:
            warnings.append("NER tags missing — extracted balanced JSON object")
            return obj

    if arr_pos != -1:
        array = _find_balanced(text, "[", "]")
        if array:
            warnings.append("NER tags missing — extracted balanced JSON array")
            return array

    if obj_pos != -1:
        obj = _find_balanced(text, "{", "}")
        if obj:
            warnings.append("NER tags missing — extracted balanced JSON object")
            return obj

    return None


def _find_balanced(text: str, opener: str, closer: str) -> Optional[str]:
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def _try_repair_json(payload: str) -> Optional[str]:
    """Best-effort repairs: strip trailing commas before } or ]."""
    repaired = re.sub(r",\s*([}\]])", r"\1", payload)
    return repaired if repaired != payload else None


def _coerce_to_list_of_dicts(data: Any, warnings: List[str]) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for key in ("entities", "terms", "candidates", "items", "results"):
            value = data.get(key)
            if isinstance(value, list):
                warnings.append(f"unwrapped list from '{key}' field")
                return [d for d in value if isinstance(d, dict)]
        if "source" in data:
            return [data]
        warnings.append("response was an object without a recognized list field")
        return []
    warnings.append(f"unexpected JSON root type: {type(data).__name__}")
    return []


def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()
