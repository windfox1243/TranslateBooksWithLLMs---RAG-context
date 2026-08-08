"""Compacting the facts recorded about one character.

A character value is a gender followed by a list of facts. This module owns
the fact list: dropping fragments that carry no information, folding a fact
into a more specific one that already covers it, and merging two lists into
one. It reads names only through name_keys, and never looks at the gender
half of the value -- so gender inference can build on it without the two
becoming mutually recursive."""
from __future__ import annotations

import re
from typing import List

from .constants import _NAME_TITLES, _UNIQUE_ROLE_TITLES
from .name_keys import _clean_inline_text, _plain_key

_DETAIL_STOP_WORDS = {
    "a",
    "an",
    "and",
    "is",
    "of",
    "the",
    "who",
    "with",
}

def _detail_key(value: str) -> str:
    return _plain_key(value).rstrip(" .;,:")

def _detail_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", _detail_key(value), flags=re.UNICODE)
        if token not in _DETAIL_STOP_WORDS
    }

def _detail_is_redundant(first: str, second: str) -> bool:
    first_key = _detail_key(first)
    second_key = _detail_key(second)
    if not first_key or not second_key:
        return False
    if first_key in second_key or second_key in first_key:
        return True
    first_tokens = _detail_tokens(first)
    second_tokens = _detail_tokens(second)
    if not first_tokens or not second_tokens:
        return False
    overlap = len(first_tokens & second_tokens) / min(
        len(first_tokens),
        len(second_tokens),
    )
    return overlap >= 0.85

def _is_context_evidence_fact(fact: str) -> bool:
    """Proof labels are internal reasons, not durable character metadata."""
    key = _detail_key(fact)
    if not key:
        return False
    evidence_facts = {
        "explicit source evidence",
        "pronoun evidence",
        "raw source evidence",
        "reincarnated current form",
        "source evidence",
        "source pronoun evidence",
        "source proven correction",
    }
    return key in evidence_facts or key.endswith(" pronoun evidence")

def _strip_low_value_fact_fragments(fact: str) -> str:
    """Remove model-analysis filler while keeping real role facts."""
    clean = _clean_inline_text(fact)
    clean = re.sub(
        r"(?i)^(?:an?\s+)?fictional\s+character\s*,\s*",
        "",
        clean,
    )
    clean = re.sub(
        r"(?i)\s*,?\s*(?:an?\s+)?fictional\s+character\s*$",
        "",
        clean,
    )
    clean = re.sub(
        r"(?i)\s*,\s*character\s+from\s+[\"'“”‘’]?[^;,.\"'“”‘’]+"
        r"[\"'“”‘’]?\s*$",
        "",
        clean,
    )
    clean = re.sub(
        r"(?i)^character\s+from\s+[\"'“”‘’]?[^;,.\"'“”‘’]+"
        r"[\"'“”‘’]?\s*,\s*",
        "",
        clean,
    )
    return _clean_inline_text(clean).strip(" ;,").rstrip(" .")

def _compact_subordinate_facts(facts: List[str]) -> List[str]:
    """Combine repeated English subordinate clauses into one cumulative fact."""
    grouped: Dict[str, Dict[str, Any]] = {}
    untouched: List[Tuple[int, str]] = []
    pattern = re.compile(
        r"^(?P<prefix>.*?)\bsubordinate\s+of\s+(?P<leader>.+?)\.?$",
        flags=re.IGNORECASE,
    )
    for index, fact in enumerate(facts):
        match = pattern.match(fact.strip())
        if not match:
            untouched.append((index, fact))
            continue
        leader = match.group("leader").strip().rstrip(" .")
        leader_key = _plain_key(leader)
        group = grouped.setdefault(
            leader_key,
            {"index": index, "leader": leader, "modifiers": []},
        )
        prefix = re.sub(
            r"^(?:a|an|the)\s+",
            "",
            match.group("prefix").strip().rstrip(" ,"),
            flags=re.IGNORECASE,
        )
        prefix = re.sub(r"(?:,?\s+and)\s*$", "", prefix, flags=re.IGNORECASE)
        for modifier in re.split(r"\s*(?:,|\band\b)\s*", prefix):
            clean = modifier.strip()
            if clean and _plain_key(clean) not in {
                _plain_key(item) for item in group["modifiers"]
            }:
                group["modifiers"].append(clean)

    rendered = list(untouched)
    for group in grouped.values():
        modifiers = group["modifiers"]
        titles = [
            modifier
            for modifier in modifiers
            if _plain_key(modifier).split(maxsplit=1)[0] in _NAME_TITLES
        ]
        descriptors = [
            modifier for modifier in modifiers if modifier not in titles
        ]
        title_prefix = ", ".join(titles)
        descriptor_prefix = " and ".join(descriptors)
        if title_prefix and descriptor_prefix:
            prefix = f"{title_prefix}, {descriptor_prefix}"
        elif title_prefix:
            prefix = f"{title_prefix},"
        else:
            prefix = descriptor_prefix
        fact = (
            f"{prefix} subordinate of {group['leader']}"
            if prefix
            else f"subordinate of {group['leader']}"
        )
        rendered.append((group["index"], fact))
    return [fact for _, fact in sorted(rendered, key=lambda item: item[0])]

def _split_monarch_fact(
    fact: str,
    *,
    role: Optional[str] = None,
    ruler: bool = False,
) -> Optional[Tuple[str, str]]:
    prefix = r"(?:the\s+)?ruler" if ruler else rf"(?:the\s+)?{re.escape(role or '')}"
    match = re.match(
        rf"^{prefix}\s+of\s+(?P<realm>[^,.;]+?)"
        r"(?P<tail>\s*(?:,\s*|\s+(?:with|who|known\s+for|known\s+as)\b).*)?$",
        fact.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return (
        match.group("realm").strip().rstrip(" ."),
        (match.group("tail") or "").strip(" ,").rstrip(" ."),
    )

def _normalize_unique_role_fact(fact: str) -> str:
    match = re.match(
        r"^(?P<role>emperor|empress|king|queen)\s*,\s*"
        r"(?:the\s+)?ruler\s+of\s+(?P<realm>[^,.;]+)"
        r"(?P<tail>\s*,\s*.+)?$",
        fact.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return fact
    role = match.group("role").title()
    realm = match.group("realm").strip().rstrip(" .")
    tail = (match.group("tail") or "").strip(" ,").rstrip(" .")
    return f"{role} of {realm}" + (f", {tail}" if tail else "")

def _compact_unique_role_facts(facts: List[str]) -> List[str]:
    """Prefer a named monarch title over a duplicate generic ruler phrase."""
    compacted = [_normalize_unique_role_fact(fact) for fact in facts]
    for role in _UNIQUE_ROLE_TITLES:
        title_match = next(
            (
                (index, parsed)
                for index, fact in enumerate(compacted)
                if (parsed := _split_monarch_fact(fact, role=role))
            ),
            None,
        )
        ruler_match = next(
            (
                (index, parsed)
                for index, fact in enumerate(compacted)
                if (parsed := _split_monarch_fact(fact, ruler=True))
            ),
            None,
        )
        if not title_match or not ruler_match:
            continue
        title_index, (title_realm, title_tail) = title_match
        ruler_index, (ruler_realm, ruler_tail) = ruler_match
        if _plain_key(title_realm) != _plain_key(ruler_realm):
            continue
        tails = _merge_character_details(title_tail, ruler_tail)
        replacement = f"{role.title()} of {title_realm}"
        if tails:
            replacement = f"{replacement} {tails}"
        first_index = min(title_index, ruler_index)
        compacted = [
            fact
            for index, fact in enumerate(compacted)
            if index not in {title_index, ruler_index}
        ]
        compacted.insert(first_index, replacement)
    return compacted

def _merge_character_details(first: str, second: str) -> str:
    facts: List[str] = []
    for raw_fact in re.split(r"\s*;\s*", f"{first};{second}"):
        fact = _clean_inline_text(raw_fact).strip(" ;").rstrip(" .;,")
        fact = _strip_low_value_fact_fragments(fact)
        if not fact:
            continue
        if _is_context_evidence_fact(fact):
            continue
        redundant_index = next(
            (
                index
                for index, existing in enumerate(facts)
                if _detail_is_redundant(existing, fact)
            ),
            None,
        )
        if redundant_index is None:
            facts.append(fact)
            continue
        existing = facts[redundant_index]
        if len(_detail_tokens(fact)) > len(_detail_tokens(existing)):
            facts[redundant_index] = fact
    facts = _compact_subordinate_facts(facts)
    facts = _compact_unique_role_facts(facts)
    facts = _compact_reincarnation_facts(facts)
    return "; ".join(facts)

def _compact_reincarnation_facts(facts: List[str]) -> List[str]:
    """Merge common split reincarnation facts into one concise description."""
    source_index: Optional[int] = None
    world_index: Optional[int] = None
    source = ""
    world = ""

    for index, fact in enumerate(facts):
        source_match = re.match(
            r"^(?:a\s+|the\s+)?reincarnation\s+of\s+(.+?)\.?$",
            fact,
            flags=re.IGNORECASE,
        )
        if source_match and source_index is None:
            source_index = index
            source = source_match.group(1).strip().rstrip(" .")
            continue

        world_match = re.match(
            r"^(?:a\s+|the\s+)?reincarnation\s+into\s+(.+?)\.?$",
            fact,
            flags=re.IGNORECASE,
        )
        if world_match and world_index is None:
            world_index = index
            world = world_match.group(1).strip().rstrip(" .")

    if source_index is None or world_index is None:
        return facts

    replacement = f"reincarnation of {source} into {world}"
    first_index = min(source_index, world_index)
    compacted = [
        fact
        for index, fact in enumerate(facts)
        if index not in {source_index, world_index}
    ]
    compacted.insert(first_index, replacement)
    return compacted
