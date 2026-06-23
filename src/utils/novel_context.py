"""
Utility functions for managing novel translation context files.

These context files track character genders, relationships (addressing forms),
and key glossary terms across translation segments, ensuring consistency.
"""
from __future__ import annotations

import re
import os
import logging
import base64
import zlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable

logger = logging.getLogger("novel_context")

SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-\.]+\.txt$")
DYNAMIC_STATE_START = "---DYNAMIC_STATE_START---"
DYNAMIC_STATE_END = "---DYNAMIC_STATE_END---"

CHARACTERS_SECTION = "## CHARACTERS & GENDERS"
GLOSSARY_SECTION = "## GLOSSARY & TERMINOLOGY"
ADDRESSING_SECTION = "## CURRENT ADDRESSING FORMS"
RELATIONSHIP_SECTION = "## RELATIONSHIP EVOLUTION"

_INVALID_CONTEXT_KEYS = {
    "",
    "-",
    "n/a",
    "na",
    "name",
    "none",
    "null",
    "unknown",
    "character",
    "character a",
    "character b",
    "canonical name",
    "recommended target term",
    "source term",
    "target term",
}
_GENDER_LABELS = {
    "male",
    "female",
    "non-binary",
    "nonbinary",
    "unknown",
    "unspecified",
}
_SPECIFIC_GENDER_LABELS = {
    "male",
    "female",
    "non-binary",
    "nonbinary",
}
_NAME_TITLES = {
    "captain",
    "commander",
    "count",
    "countess",
    "doctor",
    "dr",
    "duchess",
    "duke",
    "emperor",
    "empress",
    "general",
    "king",
    "lady",
    "lord",
    "major",
    "prince",
    "princess",
    "professor",
    "queen",
    "sergeant",
}
_UNIQUE_ROLE_TITLES = {"emperor", "empress", "king", "queen"}
_RELATIVE_AGE_WORDS = {
    "elder",
    "eldest",
    "older",
    "oldest",
    "younger",
    "youngest",
}
_GENERIC_ROLE_WORDS = {
    "attendant",
    "civilian",
    "commander",
    "doctor",
    "guard",
    "medic",
    "officer",
    "private",
    "sergeant",
    "soldier",
    "victim",
}


def _clean_inline_text(value: str) -> str:
    """Collapse whitespace without changing the language of the content."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _strip_balanced_brackets(value: str) -> str:
    value = _clean_inline_text(value)
    if len(value) >= 2 and value[0] == "[" and value[-1] == "]":
        return value[1:-1].strip()
    return value


def _plain_key(value: str) -> str:
    """Return a stable comparison key for names and placeholders."""
    value = unicodedata.normalize("NFKC", _strip_balanced_brackets(value))
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n.:;,-").casefold()
    return value


def _is_invalid_context_key(value: str) -> bool:
    key = _plain_key(value)
    return key in _INVALID_CONTEXT_KEYS or not re.search(r"\w", key, re.UNICODE)


def _is_disposable_unnamed_character(name: str, value: str) -> bool:
    """Reject explicit one-off unnamed roles that cannot anchor consistency."""
    description = _plain_key(value)
    if not any(marker in description for marker in ("unnamed", "one-scene", "one scene")):
        return False
    name_words = _plain_key(name).replace("'s", "").split()
    return bool(name_words and name_words[-1] in _GENERIC_ROLE_WORDS)


def _strip_trailing_qualifier(name: str) -> str:
    """Treat state/form qualifiers as attributes of the same character."""
    return re.sub(r"\s*\([^()]+\)\s*$", "", name).strip()


def _strip_leading_article(name: str) -> str:
    return re.sub(r"^(?:the)\s+", "", name, flags=re.IGNORECASE).strip()


def _strip_name_title(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2 and parts[0].rstrip(".").casefold() in _NAME_TITLES:
        return " ".join(parts[1:]).strip()
    return name


def _normalize_relative_name_key(name: str) -> str:
    """Collapse age-only relationship aliases while keeping gendered roles distinct."""
    key = _plain_key(name)
    words = key.split()
    if len(words) >= 3 and words[-1] in {"sibling", "brother", "sister"}:
        words = [
            word for index, word in enumerate(words)
            if not (word in _RELATIVE_AGE_WORDS and index < len(words) - 1)
        ]
    return " ".join(words)


def _canonical_display_name(name: str) -> str:
    name = _strip_balanced_brackets(name)
    name = _strip_trailing_qualifier(name)
    name = _strip_leading_article(name)
    titled = _strip_name_title(name)
    return _clean_inline_text(titled or name)


def _character_alias_keys(name: str) -> set[str]:
    """Return deterministic aliases for common state, title, and kinship variants."""
    clean_name = _strip_balanced_brackets(name)
    no_qualifier = _strip_trailing_qualifier(clean_name)
    no_article = _strip_leading_article(no_qualifier)
    no_title = _strip_name_title(no_article)
    aliases = {
        _plain_key(clean_name),
        _plain_key(no_qualifier),
        _plain_key(no_article),
        _plain_key(no_title),
        _normalize_relative_name_key(no_title),
    }
    return {alias for alias in aliases if alias and alias not in _INVALID_CONTEXT_KEYS}


def _monarch_role(name: str) -> str:
    no_article = _strip_leading_article(_strip_trailing_qualifier(
        _strip_balanced_brackets(name)
    ))
    first_word = no_article.split(maxsplit=1)[0].rstrip(".").casefold() if no_article else ""
    return first_word if first_word in _UNIQUE_ROLE_TITLES else ""


def _is_role_only_name(name: str) -> bool:
    return _plain_key(_canonical_display_name(name)) in _UNIQUE_ROLE_TITLES


def _character_names_match(first: str, second: str) -> bool:
    if _character_alias_keys(first) & _character_alias_keys(second):
        return True
    first_role = _monarch_role(first)
    second_role = _monarch_role(second)
    return bool(
        first_role
        and first_role == second_role
        and (_is_role_only_name(first) or _is_role_only_name(second))
    )


def _character_unique_roles(name: str, value: str = "") -> set[str]:
    """Extract identity-bearing unique titles from a name or its own description."""
    roles = set()
    name_role = _monarch_role(name)
    if name_role:
        roles.add(name_role)

    _, details = _split_gender_and_details(value)
    details_key = _plain_key(details)
    for role in _UNIQUE_ROLE_TITLES:
        if re.search(
            rf"(?:^|[;,]\s*)(?:the\s+)?{re.escape(role)}\b",
            details_key,
        ):
            roles.add(role)
    return roles


def _character_identities_match(
    first_name: str,
    first_value: str,
    second_name: str,
    second_value: str,
) -> bool:
    """Match deterministic aliases, including a unique title revealed in lore."""
    if _character_names_match(first_name, second_name):
        return True
    shared_roles = (
        _character_unique_roles(first_name, first_value)
        & _character_unique_roles(second_name, second_value)
    )
    return bool(
        shared_roles
        and (_is_role_only_name(first_name) or _is_role_only_name(second_name))
    )


def _name_specificity(name: str) -> Tuple[int, int, int]:
    canonical = _canonical_display_name(name)
    key = _plain_key(canonical)
    role_only = int(key not in _UNIQUE_ROLE_TITLES)
    no_parenthetical = int("(" not in name and ")" not in name)
    return role_only, len(canonical.split()), no_parenthetical


def _preferred_character_name(first: str, second: str) -> str:
    candidates = [_canonical_display_name(first), _canonical_display_name(second)]
    return max(candidates, key=_name_specificity)


def _split_gender_and_details(value: str) -> Tuple[str, str]:
    clean = _strip_balanced_brackets(value).strip()
    if not clean:
        return "", ""
    first, separator, rest = clean.partition(",")
    gender_candidate = first.strip().rstrip(".")
    if gender_candidate.casefold() in _GENDER_LABELS:
        return gender_candidate, rest.strip()
    return "", clean


def _canonical_gender(gender: str) -> str:
    return {
        "male": "Male",
        "female": "Female",
        "non-binary": "Non-binary",
        "nonbinary": "Non-binary",
        "unknown": "Unspecified",
        "unspecified": "Unspecified",
    }.get(str(gender or "").casefold(), str(gender or "").strip())


def _infer_gender_from_character_details(details: str) -> str:
    """Recover explicit English evidence that a model left after Unspecified.

    Context metadata is required to be English so this conservative repair can
    recognize direct self-references without guessing from names or roles.
    """
    text = _clean_inline_text(details).casefold()
    if not text:
        return ""

    kinship_object = (
        r"(?:own|brother|sister|mother|father|family|wife|husband|son|daughter)"
    )
    male_patterns = (
        r"^(?:an?\s+)?(?:young\s+|old\s+)?(?:male|man|boy)\b",
        r"(?:^|[.;,]\s*)he\b",
        r"\bhimself\b",
        rf"\bwho\b[^.;]{{0,80}}\bhis\s+{kinship_object}\b",
    )
    female_patterns = (
        r"^(?:an?\s+)?(?:young\s+|old\s+)?(?:female|woman|girl)\b",
        r"(?:^|[.;,]\s*)she\b",
        r"\bherself\b",
        rf"\bwho\b[^.;]{{0,80}}\bher\s+{kinship_object}\b",
    )
    has_male = any(re.search(pattern, text) for pattern in male_patterns)
    has_female = any(re.search(pattern, text) for pattern in female_patterns)
    if has_male == has_female:
        return ""
    return "Male" if has_male else "Female"


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
        rf"^{prefix}\s+of\s+(?P<realm>.+?)"
        r"(?P<tail>\s+(?:with|who|known\s+for|known\s+as)\b.*)?$",
        fact.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return (
        match.group("realm").strip().rstrip(" ."),
        (match.group("tail") or "").strip().rstrip(" ."),
    )


def _compact_unique_role_facts(facts: List[str]) -> List[str]:
    """Prefer a named monarch title over a duplicate generic ruler phrase."""
    compacted = list(facts)
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
        if not fact:
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
    return "; ".join(facts)


def _normalize_character_value(value: str) -> str:
    clean = _strip_balanced_brackets(value).strip()
    gender, details = _split_gender_and_details(clean)
    gender = _canonical_gender(gender)
    if gender.casefold() in {"unknown", "unspecified"}:
        gender = _infer_gender_from_character_details(details) or "Unspecified"
    details = _merge_character_details(details, "")
    return f"{gender}, {details}".rstrip(" ,") if gender else details


def _strip_character_correction_marker(value: str) -> Tuple[str, bool]:
    clean = _strip_balanced_brackets(value).strip()
    match = re.match(
        r"(?is)^(?:explicit\s+)?(?:gender\s+)?correction\s*:\s*(.+)$",
        clean,
    )
    if not match:
        trailing_correction = re.search(
            r"""(?is)
            \s*[\(\[]\s*
            correction\s*:\s*
            gender\s+(?:is\s+)?confirmed\s+as\s+
            (?P<gender>male|female|non[- ]?binary)
            \b(?P<evidence>.*?)
            \s*[\)\]]\s*[.;]?\s*$
            """,
            clean,
            flags=re.VERBOSE,
        )
        if not trailing_correction:
            return clean, False

        gender = trailing_correction.group("gender")
        normalized_gender = {
            "male": "Male",
            "female": "Female",
            "non-binary": "Non-binary",
            "non binary": "Non-binary",
            "nonbinary": "Non-binary",
        }[gender.casefold()]
        base = clean[:trailing_correction.start()].strip().rstrip(" .;,")
        _, details = _split_gender_and_details(base)
        corrected = (
            f"{normalized_gender}, {details}"
            if details
            else normalized_gender
        )
        return corrected, True

    return _strip_balanced_brackets(match.group(1)).strip(), True


def _merge_character_values(
    first: str,
    second: str,
    allow_gender_correction: bool = False,
) -> str:
    """Merge descriptions without letting an unsupported guess flip gender."""
    first_clean = _normalize_character_value(first)
    second_clean = _normalize_character_value(second)
    if not first_clean:
        return second_clean
    if not second_clean:
        return first_clean
    first_gender, first_details = _split_gender_and_details(first_clean)
    second_gender, second_details = _split_gender_and_details(second_clean)
    first_specific = first_gender.casefold() in _SPECIFIC_GENDER_LABELS
    second_specific = second_gender.casefold() in _SPECIFIC_GENDER_LABELS
    gender_conflict = bool(
        first_specific
        and second_specific
        and first_gender.casefold() != second_gender.casefold()
    )

    first_folded = first_clean.casefold()
    second_folded = second_clean.casefold()
    if not gender_conflict:
        if first_folded == second_folded or first_folded in second_folded:
            return second_clean
        if second_folded in first_folded:
            return first_clean

    if gender_conflict and allow_gender_correction:
        gender = second_gender
    else:
        if first_specific:
            gender = first_gender
        elif second_specific:
            gender = second_gender
        else:
            gender = first_gender or second_gender

    details = _merge_character_details(first_details, second_details)

    merged = f"{gender}, {details}" if gender and details else (gender or details)
    return merged[:600].rstrip(" ;,")


def _format_character_line(name: str, value: str) -> str:
    normalized = _normalize_character_value(value)
    if normalized and not normalized.endswith((".", "!", "?")):
        normalized = f"{normalized}."
    return f"- {name}: {normalized}"


def _parse_bullet_entries(text: str) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        content = line[1:].strip()
        if ":" not in content:
            if not _is_invalid_context_key(content):
                entries.append((content, ""))
            continue
        key, value = content.split(":", 1)
        key = _strip_balanced_brackets(key)
        if _is_invalid_context_key(key):
            continue
        entries.append((key, value.strip()))
    return entries


def _find_lore_section(lore: str, section_name: str) -> Optional[Tuple[int, int, int]]:
    label = section_name.lstrip("#").strip()
    pattern = re.compile(
        rf"(?im)^#{{1,3}}\s*{re.escape(label)}\s*$"
    )
    match = pattern.search(lore)
    if not match:
        return None
    next_heading = re.search(r"(?m)^#{1,3}\s+\S.*$", lore[match.end():])
    end = match.end() + next_heading.start() if next_heading else len(lore)
    return match.start(), match.end(), end


def _replace_lore_section(lore: str, section_name: str, lines: List[str]) -> str:
    body = "\n".join(lines).strip()
    replacement = section_name + (f"\n{body}" if body else "")
    bounds = _find_lore_section(lore, section_name)
    if bounds is None:
        if section_name == CHARACTERS_SECTION:
            glossary_bounds = _find_lore_section(lore, GLOSSARY_SECTION)
            if glossary_bounds:
                glossary_start = glossary_bounds[0]
                return (
                    f"{lore[:glossary_start].rstrip()}\n\n{replacement}\n\n"
                    f"{lore[glossary_start:].lstrip()}"
                ).strip() + "\n"
        separator = "\n\n" if lore.strip() else ""
        return f"{lore.rstrip()}{separator}{replacement}\n"
    start, _, end = bounds
    suffix = lore[end:].lstrip("\n")
    return f"{lore[:start].rstrip()}\n\n{replacement}\n\n{suffix}".strip() + "\n"


def _deduplicate_character_entries(
    entries: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    normalized: List[Dict[str, Any]] = []

    for raw_name, raw_value in entries:
        if (
            _is_invalid_context_key(raw_name)
            or _is_disposable_unnamed_character(raw_name, raw_value)
        ):
            continue
        aliases = _character_alias_keys(raw_name)
        matching_indices = {
            index
            for index, item in enumerate(normalized)
            if (
                aliases & item["aliases"]
                or _character_identities_match(
                    item["name"],
                    item["value"],
                    raw_name,
                    raw_value,
                )
            )
        }
        if matching_indices:
            index = min(matching_indices)
            item = normalized[index]
            item["name"] = _preferred_character_name(item["name"], raw_name)
            item["value"] = _merge_character_values(item["value"], raw_value)
            item["aliases"].update(aliases)
            for duplicate_index in sorted(
                matching_indices - {index},
                reverse=True,
            ):
                duplicate = normalized.pop(duplicate_index)
                item["name"] = _preferred_character_name(
                    item["name"],
                    duplicate["name"],
                )
                item["value"] = _merge_character_values(
                    item["value"],
                    duplicate["value"],
                )
                item["aliases"].update(duplicate["aliases"])
        else:
            normalized.append({
                "name": _canonical_display_name(raw_name),
                "value": _normalize_character_value(raw_value),
                "aliases": set(aliases),
            })

    alias_map: Dict[str, str] = {}
    result: List[Tuple[str, str]] = []
    for item in normalized:
        name = item["name"]
        result.append((name, item["value"]))
        for alias in item["aliases"] | _character_alias_keys(name):
            alias_map[alias] = name
    return result, alias_map


def _normalize_glossary_entries(entries: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    ordered: Dict[str, Tuple[str, str]] = {}
    for raw_name, raw_value in entries:
        if _is_invalid_context_key(raw_name):
            continue
        key = _plain_key(raw_name)
        ordered[key] = (
            _strip_balanced_brackets(raw_name),
            _strip_balanced_brackets(raw_value),
        )
    return list(ordered.values())


def normalize_global_lore(global_lore: str) -> str:
    """Remove template pollution and merge deterministic character aliases."""
    lore = str(global_lore or "").strip()
    if not lore:
        return ""

    character_bounds = _find_lore_section(lore, CHARACTERS_SECTION)
    if character_bounds:
        _, body_start, body_end = character_bounds
        characters, _ = _deduplicate_character_entries(
            _parse_bullet_entries(lore[body_start:body_end])
        )
        lore = _replace_lore_section(
            lore,
            CHARACTERS_SECTION,
            [_format_character_line(name, value) for name, value in characters],
        )

    glossary_bounds = _find_lore_section(lore, GLOSSARY_SECTION)
    if glossary_bounds:
        _, body_start, body_end = glossary_bounds
        glossary = _normalize_glossary_entries(
            _parse_bullet_entries(lore[body_start:body_end])
        )
        lore = _replace_lore_section(
            lore,
            GLOSSARY_SECTION,
            [f"- {name}: {value}" for name, value in glossary],
        )
    return lore.strip()


def _character_alias_map(global_lore: str) -> Dict[str, str]:
    bounds = _find_lore_section(global_lore, CHARACTERS_SECTION)
    if not bounds:
        return {}
    _, body_start, body_end = bounds
    _, aliases = _deduplicate_character_entries(
        _parse_bullet_entries(global_lore[body_start:body_end])
    )
    return aliases


def _normalize_relationship_notation(text: str) -> str:
    """Convert model-produced LaTeX/ASCII arrows to portable Unicode text."""
    replacements = (
        (r"\leftrightarrow", "↔"),
        (r"\longleftrightarrow", "↔"),
        (r"\rightarrow", "→"),
        (r"\longrightarrow", "→"),
        (r"\leftarrow", "←"),
        (r"\longleftarrow", "←"),
        (r"\to", "→"),
    )
    result = str(text or "")
    for command, arrow in replacements:
        result = re.sub(
            rf"\$?\s*{re.escape(command)}\s*\$?",
            f" {arrow} ",
            result,
            flags=re.IGNORECASE,
        )
    result = re.sub(r"\\\(\s*([↔→←])\s*\\\)", r" \1 ", result)
    result = re.sub(r"\$\s*([↔→←])\s*\$", r" \1 ", result)
    result = result.replace("<=>", " ↔ ").replace("<->", " ↔ ")
    result = result.replace("=>", " → ").replace("->", " → ")
    result = result.replace("<=", " ← ").replace("<-", " ← ")
    result = re.sub(r"[ \t]+", " ", result)
    return result


def _canonical_relationship_party(value: str, alias_map: Dict[str, str]) -> str:
    clean = _strip_balanced_brackets(value).strip()
    for alias in _character_alias_keys(clean):
        if alias in alias_map:
            return alias_map[alias]
    return clean


_DYNAMIC_RELATION_PATTERN = re.compile(
    r"^(?P<prefix>\s*-\s*)?(?P<left>.+?)\s*(?P<arrow>↔|→|←)\s*"
    r"(?P<right>.+?)\s*:\s*(?P<details>.*)$"
)
_DYNAMIC_DELETE_VALUES = {"delete"}


def _parse_dynamic_relation(
    line: str,
    alias_map: Dict[str, str],
) -> Optional[Tuple[Tuple[str, str, str], str, str]]:
    """Parse one canonical addressing/relationship registry entry."""
    match = _DYNAMIC_RELATION_PATTERN.match(line)
    if not match:
        return None

    left = _canonical_relationship_party(match.group("left"), alias_map)
    right = _canonical_relationship_party(match.group("right"), alias_map)
    arrow = match.group("arrow")
    details = _clean_inline_text(match.group("details"))
    if _is_invalid_context_key(left) or _is_invalid_context_key(right):
        return None

    key_left = _plain_key(left)
    key_right = _plain_key(right)
    if arrow == "↔" and key_left > key_right:
        key_left, key_right = key_right, key_left
    relation_key = (key_left, arrow, key_right)
    rendered = f"- {left} {arrow} {right}: {details}".rstrip()
    return relation_key, rendered, details


def _normalize_dynamic_entries(text: str, alias_map: Dict[str, str]) -> str:
    """Normalize one dynamic-state section without adding section headings."""
    output: List[str] = []
    relation_indices: Dict[Tuple[str, str, str], int] = {}

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if output and output[-1] != "":
                output.append("")
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("-"):
            bullet_body = stripped[1:].strip()
            if _is_invalid_context_key(bullet_body) or re.search(
                r"\[(?:character\s+[ab]|old form|new form|form|reason)\]",
                bullet_body,
                flags=re.IGNORECASE,
            ):
                continue

        parsed = _parse_dynamic_relation(line, alias_map)
        if not parsed:
            output.append(line)
            continue

        relation_key, rendered, _ = parsed
        if relation_key in relation_indices:
            previous_index = relation_indices[relation_key]
            output[previous_index] = rendered
        else:
            relation_indices[relation_key] = len(output)
            output.append(rendered)

    while output and output[-1] == "":
        output.pop()
    return "\n".join(output).strip()


def _merge_dynamic_entries(
    current_text: str,
    proposed_text: str,
    alias_map: Dict[str, str],
) -> str:
    """Apply a dynamic-state delta without deleting omitted durable entries."""
    current = _normalize_dynamic_entries(current_text, alias_map)
    proposed = _normalize_dynamic_entries(proposed_text, alias_map)
    if not proposed:
        return current

    output: List[Optional[str]] = []
    relation_indices: Dict[Tuple[str, str, str], int] = {}
    other_lines = set()

    for line in current.splitlines():
        parsed = _parse_dynamic_relation(line, alias_map)
        if parsed:
            relation_key, rendered, _ = parsed
            relation_indices[relation_key] = len(output)
            output.append(rendered)
        else:
            output.append(line)
            if line.strip():
                other_lines.add(_plain_key(line))

    for line in proposed.splitlines():
        parsed = _parse_dynamic_relation(line, alias_map)
        if not parsed:
            line_key = _plain_key(line)
            if line.strip() and line_key not in other_lines:
                output.append(line)
                other_lines.add(line_key)
            continue

        relation_key, rendered, details = parsed
        is_delete = details.strip().rstrip(" .;:").casefold() in (
            _DYNAMIC_DELETE_VALUES
        )
        previous_index = relation_indices.get(relation_key)
        if is_delete:
            if previous_index is not None:
                output[previous_index] = None
                relation_indices.pop(relation_key, None)
            continue
        if previous_index is not None:
            output[previous_index] = rendered
        else:
            relation_indices[relation_key] = len(output)
            output.append(rendered)

    compacted = [line for line in output if line is not None]
    while compacted and not compacted[-1].strip():
        compacted.pop()
    return "\n".join(compacted).strip()


def _split_dynamic_sections(dynamic_state: str) -> Tuple[str, str, bool]:
    addressing_lines: List[str] = []
    relationship_lines: List[str] = []
    destination = relationship_lines
    has_sections = False

    for raw_line in _normalize_relationship_notation(dynamic_state).splitlines():
        heading = _plain_key(raw_line.lstrip("#"))
        if heading == _plain_key(ADDRESSING_SECTION.lstrip("#")):
            destination = addressing_lines
            has_sections = True
            continue
        if heading == _plain_key(RELATIONSHIP_SECTION.lstrip("#")):
            destination = relationship_lines
            has_sections = True
            continue
        if heading == "dynamic relationship state":
            continue
        destination.append(raw_line)

    return (
        "\n".join(addressing_lines).strip(),
        "\n".join(relationship_lines).strip(),
        has_sections,
    )


def _format_dynamic_sections(addressing: str, relationships: str) -> str:
    lines = [ADDRESSING_SECTION]
    if addressing.strip():
        lines.append(addressing.strip())
    lines.extend(["", RELATIONSHIP_SECTION])
    if relationships.strip():
        lines.append(relationships.strip())
    return "\n".join(lines).strip()


def normalize_dynamic_state(
    dynamic_state: str,
    character_aliases: Optional[Dict[str, str]] = None,
) -> str:
    """Normalize dynamic context into stable addressing and relationship sections."""
    alias_map = character_aliases or {}
    addressing, relationships, _ = _split_dynamic_sections(dynamic_state)
    return _format_dynamic_sections(
        _normalize_dynamic_entries(addressing, alias_map),
        _normalize_dynamic_entries(relationships, alias_map),
    )


def merge_dynamic_state(
    current_dynamic_state: str,
    proposed_dynamic_state: str,
    character_aliases: Optional[Dict[str, str]] = None,
) -> str:
    """Merge durable addressing and relationship deltas by participant key.

    Omission never deletes an existing entry, so dormant relationships survive
    an arbitrary number of unrelated chunks. A response updates the matching
    directional pair in place. Deletion requires an explicit ``DELETE`` value.
    """
    aliases = character_aliases or {}
    current = normalize_dynamic_state(current_dynamic_state, aliases)
    if not str(proposed_dynamic_state or "").strip():
        return current

    current_addressing, current_relationships, _ = _split_dynamic_sections(
        current
    )
    proposed_addressing, proposed_relationships, proposed_has_sections = (
        _split_dynamic_sections(proposed_dynamic_state)
    )

    if proposed_has_sections:
        addressing = _merge_dynamic_entries(
            current_addressing,
            proposed_addressing,
            aliases,
        )
        relationships = _merge_dynamic_entries(
            current_relationships,
            proposed_relationships,
            aliases,
        )
    else:
        addressing = current_addressing
        relationships = _merge_dynamic_entries(
            current_relationships,
            proposed_relationships,
            aliases,
        )

    return _format_dynamic_sections(addressing, relationships)


def normalize_novel_context_content(content: str) -> str:
    """Normalize complete or legacy context text at every persistence boundary."""
    text = str(content or "").strip()
    if not text:
        return ""
    if DYNAMIC_STATE_START in text and DYNAMIC_STATE_END in text:
        global_lore = normalize_global_lore(extract_global_lore(text))
        dynamic_state = extract_dynamic_state_from_text(text) or ""
        return build_novel_context(global_lore, dynamic_state)
    return normalize_global_lore(text)


def is_safe_filename(filename: str) -> bool:
    """Whitelist filenames to alphanumerics + `_-.` with .txt extension."""
    return bool(SAFE_FILENAME_RE.match(filename or ""))


def _resolve_inside(directory: Path, filename: str) -> Optional[Path]:
    """Return the file path if it resolves inside `directory`, else None."""
    candidate = directory / filename
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return None
    return candidate


def list_novel_contexts(novel_contexts_dir: Path) -> List[Dict[str, Any]]:
    """List text files in the directory.

    Returns a list of dicts:
        {
            "filename": "my_novel.txt",
            "display_name": "my_novel",
            "format": "txt"
        }
    """
    if not novel_contexts_dir.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for file_path in novel_contexts_dir.glob("*.txt"):
        try:
            file_path.resolve().relative_to(novel_contexts_dir.resolve())
        except ValueError:
            continue

        entries.append(
            {
                "filename": file_path.name,
                "display_name": file_path.stem,
                "format": "txt",
            }
        )

    entries.sort(key=lambda e: e["display_name"].lower())
    return entries


def load_novel_context(filename: str, novel_contexts_dir: Path) -> str:
    """Load context content from a safe filename. Creates empty template if missing."""
    if not is_safe_filename(filename):
        raise ValueError(
            f"Invalid filename '{filename}'. Allowed: alphanumerics, `_`, `-`, `.`; extension must be .txt."
        )

    file_path = _resolve_inside(novel_contexts_dir, filename)
    if file_path is None:
        raise ValueError(
            f"Filename '{filename}' resolves outside Novel_Contexts directory."
        )

    if not file_path.exists():
        # Empty sections are intentional. Example bullets used to leak into
        # real context files as fake characters such as "[Name]" and "[None]".
        template = build_novel_context(
            (
                "# GLOBAL LORE\n"
                "(Characters, genders, and terminology; canonical names only.)\n\n"
                f"{CHARACTERS_SECTION}\n\n"
                f"{GLOSSARY_SECTION}"
            ),
            (
                f"{ADDRESSING_SECTION}\n\n"
                f"{RELATIONSHIP_SECTION}"
            ),
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(template, encoding="utf-8")
        return template

    return normalize_novel_context_content(
        file_path.read_text(encoding="utf-8-sig")
    )


def save_novel_context(filename: str, novel_contexts_dir: Path, content: str) -> None:
    """Atomically save context content to a safe filename."""
    if not is_safe_filename(filename):
        raise ValueError(
            f"Invalid filename '{filename}'. Allowed: alphanumerics, `_`, `-`, `.`; extension must be .txt."
        )

    file_path = _resolve_inside(novel_contexts_dir, filename)
    if file_path is None:
        raise ValueError(
            f"Filename '{filename}' resolves outside Novel_Contexts directory."
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary_path.write_text(
        normalize_novel_context_content(content),
        encoding="utf-8",
    )
    temporary_path.replace(file_path)


def resolve_novel_context_path(filename: str, novel_contexts_dir: Path) -> Path:
    """Resolve novel context file path. Checks directory first, then absolute/relative."""
    if is_safe_filename(filename):
        resolved = _resolve_inside(novel_contexts_dir, filename)
        if resolved:
            return resolved

    # Check if the filename's basename is a safe filename.
    # If the app is frozen, or if the path contains 'Novel_Contexts' / 'TranslateBook_Data',
    # redirect the file resolution to the current local novel_contexts_dir.
    # This prevents absolute paths from old/unbuilt directories leaking in when the executable is moved.
    base_name = os.path.basename(filename)
    if is_safe_filename(base_name):
        import sys
        is_frozen = getattr(sys, 'frozen', False)
        path_str = str(filename).replace('\\', '/')
        if is_frozen or 'novel_contexts' in path_str.lower() or 'translatebook_data' in path_str.lower():
            resolved = _resolve_inside(novel_contexts_dir, base_name)
            if resolved:
                return resolved

    if os.path.isabs(filename):
        return Path(filename).resolve()
    
    # Try relative to current working directory
    return Path(filename).resolve()

def extract_dynamic_state_from_text(context_content: str) -> Optional[str]:
    start_tag = DYNAMIC_STATE_START
    end_tag = DYNAMIC_STATE_END
    start_idx = context_content.find(start_tag)
    end_idx = context_content.find(end_tag)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        state_text = context_content[start_idx + len(start_tag):end_idx].strip()
        # Clean up `# DYNAMIC RELATIONSHIP STATE` headers
        lines = state_text.splitlines()
        cleaned_lines = []
        for line in lines:
            if line.strip().upper().replace(" ", "") == "#DYNAMICRELATIONSHIPSTATE":
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()
    return None

def compress_dynamic_state(dynamic_text: str) -> str:
    compressed = zlib.compress(dynamic_text.encode('utf-8'))
    return base64.b64encode(compressed).decode('ascii')

def extract_global_lore(context_content: str) -> str:
    """Extracts the text before the DYNAMIC_STATE_START tag."""
    start_tag = DYNAMIC_STATE_START
    start_idx = context_content.find(start_tag)
    
    if start_idx != -1:
        return context_content[:start_idx].strip()
    return context_content.strip()

def decompress_dynamic_state(b64_compressed_state: str) -> str:
    """Decompresses a base64 zlib string back to plain text."""
    try:
        compressed = base64.b64decode(b64_compressed_state)
        return zlib.decompress(compressed).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to decompress dynamic state: {e}")
        return ""


def build_novel_context(global_lore: str, dynamic_state: str) -> str:
    """Build the canonical full context representation used by every pipeline."""
    normalized_global = normalize_global_lore(global_lore)
    normalized_dynamic = normalize_dynamic_state(
        dynamic_state,
        _character_alias_map(normalized_global),
    )
    return (
        f"{normalized_global.strip()}\n\n"
        f"{DYNAMIC_STATE_START}\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        f"{normalized_dynamic.strip()}\n"
        f"{DYNAMIC_STATE_END}"
    ).strip()


def make_novel_context_filename(input_filename: str, fallback: str = "translation") -> str:
    """Create a safe, deterministic context filename from an input filename."""
    stem = Path(input_filename or "").stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem).strip(".")
    if not safe_stem:
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", fallback).strip(".") or "translation"
    return f"{safe_stem}_context.txt"


def normalize_novel_context_filename(filename: str) -> str:
    """Return a safe basename for web/API-managed context files."""
    basename = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not is_safe_filename(basename):
        raise ValueError("Invalid novel context filename")
    return basename


def decode_context_snapshot(
    compressed_snapshot: Optional[str],
    fallback_context: str = "",
) -> Tuple[str, str, str]:
    """Decode full snapshots and legacy dynamic-only snapshots.

    Returns ``(full_context, global_lore, dynamic_state)``. New snapshots always
    contain the full canonical context, while old snapshots are combined with
    the supplied fallback global lore.
    """
    fallback_global = extract_global_lore(fallback_context)
    fallback_dynamic = extract_dynamic_state_from_text(fallback_context) or ""
    decoded = decompress_dynamic_state(compressed_snapshot) if compressed_snapshot else ""

    if DYNAMIC_STATE_START in decoded:
        global_lore = extract_global_lore(decoded)
        dynamic_state = extract_dynamic_state_from_text(decoded) or ""
    else:
        global_lore = fallback_global
        dynamic_state = decoded or fallback_dynamic

    full_context = build_novel_context(global_lore, dynamic_state)
    canonical_global = extract_global_lore(full_context)
    return full_context, canonical_global, dynamic_state


def normalize_refinement_context(
    context_content: Optional[str],
    fallback_context: str = "",
) -> str:
    """Overlay final global lore onto a unit's historical dynamic state.

    Characters, proven genders, and glossary terminology are book-wide facts
    that later refinement units may discover after early units were translated.
    Addressing forms and relationship evolution are time-sensitive, so those
    remain sourced from the mapped historical snapshot.

    ``context_content`` may be a full snapshot or a legacy dynamic-only value.
    ``fallback_context`` should be the latest canonical context loaded from the
    context file; when it has no global lore, historical lore is used as a
    compatibility fallback.
    """
    final_global_lore = extract_global_lore(fallback_context)
    if not context_content:
        return build_novel_context(
            final_global_lore,
            extract_dynamic_state_from_text(fallback_context) or "",
        )
    if DYNAMIC_STATE_START in context_content:
        historical_global_lore = extract_global_lore(context_content)
        return build_novel_context(
            final_global_lore or historical_global_lore,
            extract_dynamic_state_from_text(context_content) or "",
        )
    return build_novel_context(
        final_global_lore,
        context_content,
    )


def map_context_snapshots_for_refinement(
    total_chunks: int,
    db_chunks: List[Dict[str, Any]],
    fallback_context: str = "",
    refinement_units: Optional[List[str]] = None,
) -> List[Optional[str]]:
    """Map translation snapshots onto refinement units using output provenance.

    When translation and refinement produce the same number of units, mapping
    is exact. Otherwise translated-text and refinement-unit lengths define a
    cumulative position timeline, which is substantially more accurate than
    mapping by chunk count alone.
    """
    if total_chunks <= 0:
        return []

    timeline_rows: List[Tuple[str, int]] = []
    last_snapshot: Optional[str] = None
    for chunk in sorted(
        db_chunks or [],
        key=lambda item: item.get("chunk_index", -1),
    ):
        snapshot = (chunk.get("chunk_data") or {}).get("context_snapshot")
        if snapshot:
            last_snapshot = snapshot
        if not last_snapshot:
            continue
        translated_text = str(chunk.get("translated_text") or "")
        source_text = str(chunk.get("original_text") or "")
        weight = max(len(translated_text.strip()), len(source_text.strip()), 1)
        timeline_rows.append((last_snapshot, weight))

    if not timeline_rows:
        return [None] * total_chunks

    def decode(snapshot: str) -> str:
        full_context, _, _ = decode_context_snapshot(
            snapshot,
            fallback_context,
        )
        return full_context

    if len(timeline_rows) == total_chunks:
        return [decode(snapshot) for snapshot, _ in timeline_rows]

    contexts: List[Optional[str]] = []
    if refinement_units and len(refinement_units) == total_chunks:
        target_weights = [max(len(str(unit or "").strip()), 1) for unit in refinement_units]
        source_total = sum(weight for _, weight in timeline_rows)
        target_total = sum(target_weights)
        source_boundaries: List[int] = []
        cumulative_source = 0
        for _, weight in timeline_rows:
            cumulative_source += weight
            source_boundaries.append(cumulative_source)

        cumulative_target = 0
        source_index = 0
        for weight in target_weights:
            target_midpoint = cumulative_target + (weight / 2)
            source_position = (target_midpoint / target_total) * source_total
            while (
                source_index < len(source_boundaries) - 1
                and source_position > source_boundaries[source_index]
            ):
                source_index += 1
            contexts.append(decode(timeline_rows[source_index][0]))
            cumulative_target += weight
        return contexts

    for index in range(total_chunks):
        mapped_index = min(
            int(index * len(timeline_rows) / total_chunks),
            len(timeline_rows) - 1,
        )
        contexts.append(decode(timeline_rows[mapped_index][0]))
    return contexts


def map_dialogue_attributions_for_refinement(
    total_chunks: int,
    db_chunks: List[Dict[str, Any]],
) -> List[Optional[Dict[str, Any]]]:
    """Reuse dialogue maps only when translation/refinement units align exactly.

    Unlike cumulative lore snapshots, a speaker map belongs to one local source
    unit. Guessing a proportional mapping after re-chunking could attach the
    wrong speaker to unrelated dialogue, so mismatched layouts deliberately
    fall back to fresh monolingual analysis during refinement.
    """
    if total_chunks <= 0:
        return []
    rows = [
        chunk
        for chunk in sorted(
            db_chunks or [],
            key=lambda item: item.get("chunk_index", -1),
        )
        if chunk.get("status") == "completed"
        and chunk.get("translated_text") is not None
    ]
    if len(rows) != total_chunks:
        return [None] * total_chunks
    return [
        (row.get("chunk_data") or {}).get("dialogue_attribution")
        for row in rows
    ]


@dataclass
class RefinementContextTracker:
    """Resolve historical or source-first context for sequential refinement."""

    prompt_options: Dict[str, Any]
    historical_contexts: List[Optional[str]]
    historical_dialogue_attributions: List[Optional[Dict[str, Any]]] = field(
        default_factory=list
    )
    log_callback: Optional[Callable] = None
    cursor: int = 0

    def __post_init__(self) -> None:
        from src.utils.dialogue_attribution import empty_dialogue_attribution

        base_context = self.prompt_options.get("novel_context", "")
        self.global_lore = extract_global_lore(base_context)
        self.dynamic_state = extract_dynamic_state_from_text(base_context) or ""
        if any(self.historical_contexts):
            # Refinement replays historical states from the beginning. Never
            # seed that replay with the final end-of-book dynamic state.
            self.dynamic_state = ""
        self.auto_analyze = bool(self.prompt_options.get("auto_update_context"))
        self.dialogue_state: Dict[str, str] = {}
        self.dialogue_scene_key: Optional[str] = None
        self.current_dialogue_attribution = empty_dialogue_attribution()

    async def next_context(
        self,
        *,
        text: str,
        llm_client: Any,
        model_name: str,
        target_language: str,
        display_index: int,
        total_chunks: int,
        scene_key: Optional[Any] = None,
    ) -> str:
        """Return context for the next refinement unit without mutating its file."""
        historical = (
            self.historical_contexts[self.cursor]
            if self.cursor < len(self.historical_contexts)
            else None
        )
        historical_dialogue = (
            self.historical_dialogue_attributions[self.cursor]
            if self.cursor < len(self.historical_dialogue_attributions)
            else None
        )
        from src.utils.dialogue_attribution import (
            detect_dialogue_turns,
            dialogue_attribution_stats,
            empty_dialogue_attribution,
        )
        normalized_scene_key = (
            str(scene_key) if scene_key is not None else None
        )
        if (
            normalized_scene_key is not None
            and self.dialogue_scene_key is not None
            and normalized_scene_key != self.dialogue_scene_key
        ):
            self.dialogue_state = {}
        if normalized_scene_key is not None:
            self.dialogue_scene_key = normalized_scene_key
        self.current_dialogue_attribution = (
            historical_dialogue
            or empty_dialogue_attribution(self.dialogue_state)
        )
        if historical_dialogue:
            self.dialogue_state = dict(
                historical_dialogue.get("state_after")
                or self.dialogue_state
            )

        if historical:
            historical_context = normalize_refinement_context(
                historical,
                build_novel_context(self.global_lore, self.dynamic_state),
            )
            self.global_lore = extract_global_lore(historical_context)
            historical_dynamic = (
                extract_dynamic_state_from_text(historical_context) or ""
            )
            self.dynamic_state = merge_dynamic_state(
                self.dynamic_state,
                historical_dynamic,
                _character_alias_map(self.global_lore),
            )
            full_context = build_novel_context(
                self.global_lore,
                self.dynamic_state,
            )
            if self.log_callback:
                self.log_callback(
                    "refinement_context_snapshot",
                    f"📚 Restored historical context for refinement unit {display_index}/{total_chunks}.",
                )
        elif self.auto_analyze and text.strip():
            if self.log_callback:
                self.log_callback(
                    "refinement_context_analyzing",
                    f"🧭 Analyzing context for refinement unit {display_index}/{total_chunks}...",
                )
            dialogue_sink: Dict[str, Any] = {}
            dialogue_turns = detect_dialogue_turns(text)
            self.global_lore, self.dynamic_state, change_logs = await update_novel_context_chunk(
                llm_client=llm_client,
                model_name=model_name,
                current_global_lore=self.global_lore,
                current_dynamic_state=self.dynamic_state,
                source_chunk=text,
                translated_chunk=None,
                source_language=target_language,
                target_language=target_language,
                chunk_index=display_index,
                total_chunks=total_chunks,
                dialogue_turns=dialogue_turns,
                current_dialogue_state=self.dialogue_state,
                dialogue_attribution_sink=dialogue_sink,
            )
            self.current_dialogue_attribution = (
                dialogue_sink
                or empty_dialogue_attribution(self.dialogue_state)
            )
            self.dialogue_state = dict(
                self.current_dialogue_attribution.get("state_after")
                or self.dialogue_state
            )
            full_context = build_novel_context(
                self.global_lore,
                self.dynamic_state,
            )
            if self.log_callback:
                self.log_callback(
                    "refinement_context_ready",
                    f"✅ Context prepared for refinement unit {display_index}/{total_chunks}.",
                )
                for change_log in change_logs:
                    self.log_callback("novel_context_log", change_log)
                if dialogue_turns:
                    stats = dialogue_attribution_stats(
                        self.current_dialogue_attribution
                    )
                    message = (
                        "Dialogue context: "
                        f"{stats['identified']} turns identified, "
                        f"{stats['assigned']} assigned, "
                        f"{stats['uncertain']} uncertain."
                    )
                    logger.info(message)
                    self.log_callback(
                        "dialogue_attribution",
                        message,
                    )
        else:
            full_context = build_novel_context(
                self.global_lore,
                self.dynamic_state,
            )

        self.cursor += 1
        if self.log_callback and (self.global_lore or self.dynamic_state):
            self.log_callback(
                "novel_context_state",
                f"Refinement context ready for unit {display_index}/{total_chunks}",
                {
                    "type": "novel_context_state",
                    "content": full_context,
                    "filename": self.prompt_options.get("novel_context_file", ""),
                    "phase": "refinement",
                    "chunk_index": display_index - 1,
                    "ephemeral": not bool(historical),
                },
            )
        return full_context


@dataclass
class NovelContextSession:
    """Own the mutable context state shared by translation pipelines."""

    path: Path
    prompt_options: Dict[str, Any]
    global_lore: str
    dynamic_state: str
    log_callback: Optional[Callable] = None
    dialogue_state: Dict[str, str] = field(default_factory=dict)
    dialogue_attribution: Dict[str, Any] = field(default_factory=dict)
    dialogue_scene_key: Optional[str] = None

    @property
    def content(self) -> str:
        return build_novel_context(self.global_lore, self.dynamic_state)

    def sync_prompt(self) -> str:
        content = self.content
        self.prompt_options["novel_context"] = content
        return content

    def save(self) -> str:
        content = self.sync_prompt()
        save_novel_context(self.path.name, self.path.parent, content)
        return content

    def snapshot(self) -> str:
        """Return a compressed full-context snapshot."""
        return compress_dynamic_state(self.content)

    async def analyze_source(
        self,
        llm_client: Any,
        model_name: str,
        source_chunk: str,
        source_language: str,
        target_language: str,
        chunk_index: int,
        total_chunks: int,
        scene_key: Optional[Any] = None,
    ) -> List[str]:
        """Analyze source text before translating it and expose the new context."""
        from src.utils.dialogue_attribution import (
            detect_dialogue_turns,
            dialogue_attribution_stats,
            empty_dialogue_attribution,
        )

        normalized_scene_key = (
            str(scene_key) if scene_key is not None else None
        )
        if (
            normalized_scene_key is not None
            and self.dialogue_scene_key is not None
            and normalized_scene_key != self.dialogue_scene_key
        ):
            self.dialogue_state = {}
        if normalized_scene_key is not None:
            self.dialogue_scene_key = normalized_scene_key

        dialogue_turns = detect_dialogue_turns(source_chunk)
        dialogue_sink: Dict[str, Any] = {}
        self.global_lore, self.dynamic_state, change_logs = await update_novel_context_chunk(
            llm_client=llm_client,
            model_name=model_name,
            current_global_lore=self.global_lore,
            current_dynamic_state=self.dynamic_state,
            source_chunk=source_chunk,
            translated_chunk=None,
            source_language=source_language,
            target_language=target_language,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            dialogue_turns=dialogue_turns,
            current_dialogue_state=self.dialogue_state,
            dialogue_attribution_sink=dialogue_sink,
        )
        self.dialogue_attribution = (
            dialogue_sink
            or empty_dialogue_attribution(self.dialogue_state)
        )
        self.dialogue_state = dict(
            self.dialogue_attribution.get("state_after") or self.dialogue_state
        )
        if normalized_scene_key is not None:
            self.dialogue_attribution["scene_key"] = normalized_scene_key
        if self.dialogue_attribution.get("turns"):
            self.prompt_options["dialogue_attribution"] = self.dialogue_attribution
        else:
            self.prompt_options.pop("dialogue_attribution", None)
        if dialogue_turns and self.log_callback:
            stats = dialogue_attribution_stats(self.dialogue_attribution)
            message = (
                "Dialogue context: "
                f"{stats['identified']} turns identified, "
                f"{stats['assigned']} assigned, "
                f"{stats['uncertain']} uncertain."
            )
            logger.info(message)
            self.log_callback(
                "dialogue_attribution",
                message,
            )
        elif dialogue_turns:
            stats = dialogue_attribution_stats(self.dialogue_attribution)
            message = (
                "Dialogue context: "
                f"{stats['identified']} turns identified, "
                f"{stats['assigned']} assigned, "
                f"{stats['uncertain']} uncertain."
            )
            logger.info(message)
            print(message)
        self.save()
        return change_logs


def open_novel_context_session(
    prompt_options: Dict[str, Any],
    novel_contexts_dir: Path,
    input_filename: str = "",
    fallback_name: str = "translation",
    resume_snapshot: Optional[str] = None,
    resume_dialogue_state: Optional[Dict[str, str]] = None,
    resume_dialogue_scene_key: Optional[Any] = None,
    log_callback: Optional[Callable] = None,
) -> Optional[NovelContextSession]:
    """Load/create context state, restore a snapshot, and inject it into prompts."""
    novel_context_file = prompt_options.get("novel_context_file")
    auto_update_context = bool(prompt_options.get("auto_update_context", False))

    if auto_update_context and not novel_context_file:
        novel_context_file = make_novel_context_filename(input_filename, fallback_name)
        prompt_options["novel_context_file"] = novel_context_file
        if log_callback:
            log_callback(
                "novel_context_created",
                f"Auto-created new novel context file: {novel_context_file}",
            )

    if not novel_context_file:
        return None

    path = resolve_novel_context_path(novel_context_file, novel_contexts_dir)
    current_content = load_novel_context(path.name, path.parent)
    if resume_snapshot:
        current_content, global_lore, dynamic_state = decode_context_snapshot(
            resume_snapshot,
            current_content,
        )
    else:
        global_lore = extract_global_lore(current_content)
        dynamic_state = extract_dynamic_state_from_text(current_content) or ""

    session = NovelContextSession(
        path=path,
        prompt_options=prompt_options,
        global_lore=global_lore,
        dynamic_state=dynamic_state,
        log_callback=log_callback,
        dialogue_state=dict(resume_dialogue_state or {}),
        dialogue_scene_key=(
            str(resume_dialogue_scene_key)
            if resume_dialogue_scene_key is not None
            else None
        ),
    )
    content = session.sync_prompt()
    if log_callback:
        log_callback(
            "novel_context_state",
            "Context loaded",
            {
                "type": "novel_context_state",
                "content": content,
                "filename": path.name,
            },
        )
    return session


UPDATE_SYSTEM_PROMPT = """You are an expert novel translation context assistant.
Your task is to analyze the latest source text and its translation, and detect any new characters, glossary terms, or relationship addressing changes.

Identity rules:
- Reuse the exact canonical name already present in CURRENT GLOBAL LORE.
- A title, rank, nickname, transformed state, awakened state, disguise, age qualifier, or relationship label is not a new character when it refers to an existing person. Update the existing canonical entry instead.
- When a title-only entry is later identified by name (for example, "Emperor" = "Serena Augusta"), output only the named canonical character with the title in its concise description.
- Do not add one-scene unnamed soldiers, victims, hallucinations, generic crowds, or incidental job labels unless they recur and their identity/gender is required for translation consistency.
- Never output template entries, "None", "[None]", "Unknown", "N/A", or an empty bullet.
- Record gender only when the source states it or supplies unambiguous grammatical/pronoun evidence. Never guess gender from a name, occupation, rank, appearance, genre convention, or stereotype.
- Before writing "Unspecified", scan the whole latest source for direct evidence such as gendered nouns, pronouns, kinship grammar, or an explicit description. If an existing Unspecified character is now proven Male/Female, output that specific gender directly; this is not a correction.
- An existing specific gender is authoritative. Change it only when the latest source explicitly proves it was wrong; write that rare update as "CORRECTION: [Gender, role, description]".
- Write all character metadata in English, regardless of source and target language.
- For an existing character, output one concise cumulative replacement description containing the important old and new facts. Summarize repeated roles instead of appending duplicate phrases.

Input provided:
1. CURRENT GLOBAL LORE (Characters & Glossary)
2. CURRENT DYNAMIC RELATIONSHIP STATE
3. LATEST SOURCE TEXT
4. LATEST TRANSLATION

Your output must follow this strict format:

[NEW_CHARACTERS]
- Canonical Name: [Gender, role, and concise description]
(Use "Unspecified" rather than guessing when a recurring character must be tracked before gender is explicit. Use "CORRECTION: [Gender, role, description]" only for a source-proven correction. Use "- Canonical Name: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[NEW_GLOSSARY]
- Source Term: [Target Term]
(Only include actual additions or corrections. Use "- Source Term: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[DYNAMIC_STATE]
# DYNAMIC RELATIONSHIP STATE
## CURRENT ADDRESSING FORMS
- Speaker → Addressee: source form "..." | target-language form "..." | register and reason
## RELATIONSHIP EVOLUTION
- Character A ↔ Character B: concise current relationship
(Output both headings every time, but list only additions or changes. Omitted entries remain stored indefinitely. Remove an obsolete entry only with "- Speaker → Addressee: DELETE" or "- Character A ↔ Character B: DELETE". Addressing forms include names, titles, honorifics, pronouns, kinship terms, and formality choices needed in the target language. Use plain Unicode arrows only. Never use LaTeX, backslashes, dollar signs, or ASCII arrows. Do not duplicate these headings.)

[DIALOGUE_ATTRIBUTION]
{"turns":[{"id":"exact candidate id","speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown","confidence":0.0}],"state_after":{"speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown"}}
(Classify only the supplied dialogue candidates. Infer from narration, turn-taking, current scene state, voice, and addressing forms. Use only canonical character names already present in CURRENT GLOBAL LORE, including characters added in this response. Never invent a speaker. Confidence is from 0.0 to 1.0. Return {"turns":[],"state_after":{}} when there are no candidates.)

Do not include any other explanations, markdown fences, or extra text outside these blocks.
"""

SOURCE_ANALYSIS_SYSTEM_PROMPT = """You are an expert novel translation context assistant.
Analyze the latest SOURCE text before it is translated. Detect new or corrected characters, source-proven genders and roles, source terminology that needs a consistent target-language rendering, and relationship/addressing changes that the translator must know now.

Identity rules:
- Reuse the exact canonical name already present in CURRENT GLOBAL LORE.
- Do not create separate characters for ranks, titles, nicknames, transformed/awakened states, disguises, age variants, or relational aliases of an existing person.
- When a title-only entry is later identified by name (for example, "Emperor" = "Serena Augusta"), output only the named canonical character with the title in its concise description.
- Do not add one-scene unnamed soldiers, victims, generic crowds, or incidental roles unless they recur and are necessary for pronoun/address consistency.
- Never output template entries, "None", "[None]", "Unknown", "N/A", or an empty bullet.
- Record gender only when this source text states it or gives unambiguous grammatical/pronoun evidence. Never infer it from names, jobs, ranks, appearance, personality, or genre stereotypes.
- Before writing "Unspecified", scan the entire latest source for direct evidence such as gendered nouns, pronouns, kinship grammar, or an explicit description. If an existing Unspecified character is now proven Male/Female, output that specific gender directly; this is not a correction.
- Treat an existing specific gender as authoritative. Change it only when this source text explicitly proves it wrong, using "CORRECTION: [Gender, role, description]".
- Preserve source-side proper names exactly. Write character metadata descriptions in English so context remains stable when the translation model or target language changes.
- For an existing character, output one concise cumulative replacement description containing the important old and new facts. Summarize repeated roles instead of appending duplicate phrases.

Input provided:
1. CURRENT GLOBAL LORE (Characters & Glossary)
2. CURRENT DYNAMIC RELATIONSHIP STATE
3. LATEST SOURCE TEXT

Your output must follow this strict format:

[NEW_CHARACTERS]
- Canonical Name: [Gender, role, and concise description]
(Use "Unspecified" rather than guessing when a recurring character must be tracked before gender is explicit. Use "CORRECTION: [Gender, role, description]" only for a source-proven correction. Use "- Canonical Name: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[NEW_GLOSSARY]
- Source Term: [Recommended Target Term]
(Only include important recurring terms. Use "- Source Term: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[DYNAMIC_STATE]
# DYNAMIC RELATIONSHIP STATE
## CURRENT ADDRESSING FORMS
- Speaker → Addressee: source form "..." | recommended target-language form "..." | register and reason
## RELATIONSHIP EVOLUTION
- Character A ↔ Character B: concise current relationship
(Output both headings every time, but list only additions or changes. Omitted entries remain stored indefinitely. Remove an obsolete entry only with "- Speaker → Addressee: DELETE" or "- Character A ↔ Character B: DELETE". Addressing forms include names, titles, honorifics, pronouns, kinship terms, and formality choices needed for translation. Use plain Unicode arrows only. Never use LaTeX, backslashes, dollar signs, or ASCII arrows. Keep it concise and do not duplicate headings.)

[DIALOGUE_ATTRIBUTION]
{"turns":[{"id":"exact candidate id","speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown","confidence":0.0}],"state_after":{"speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown"}}
(Classify only the supplied dialogue candidates. Infer from narration, turn-taking, current scene state, voice, and addressing forms. Use only canonical character names already present in CURRENT GLOBAL LORE, including characters added in this response. Never invent a speaker. Confidence is from 0.0 to 1.0. Return {"turns":[],"state_after":{}} when there are no candidates.)

Do not translate the whole passage. Do not include explanations, markdown fences, or text outside these blocks.
"""

UPDATE_USER_PROMPT_TEMPLATE = """### CURRENT GLOBAL LORE:
{current_global_lore}

### CURRENT DYNAMIC RELATIONSHIP STATE:
{current_dynamic_state}

### TRANSLATION PROGRESS: Segment {chunk_index} of {total_chunks}

### LATEST SOURCE TEXT ({source_language}):
{source_chunk}

### LATEST TRANSLATION ({target_language}):
{translated_chunk}

### CURRENT SCENE SPEAKER STATE:
{current_dialogue_state}

### DIALOGUE CANDIDATES:
{dialogue_candidates}

Output the updates now. Output ONLY the strictly formatted blocks."""

SOURCE_ANALYSIS_USER_PROMPT_TEMPLATE = """### CURRENT GLOBAL LORE:
{current_global_lore}

### CURRENT DYNAMIC RELATIONSHIP STATE:
{current_dynamic_state}

### TRANSLATION PROGRESS: Segment {chunk_index} of {total_chunks}

### LATEST SOURCE TEXT ({source_language}):
{source_chunk}

### TARGET LANGUAGE:
{target_language}

### CURRENT SCENE SPEAKER STATE:
{current_dialogue_state}

### DIALOGUE CANDIDATES:
{dialogue_candidates}

Analyze the source for context needed by its translation. Output ONLY the strictly formatted blocks."""


def merge_new_lore(global_lore: str, new_characters: str, new_glossary: str) -> Tuple[str, List[str]]:
    """Merge context updates through canonical character and glossary identities."""
    change_logs: List[str] = []
    lore = normalize_global_lore(global_lore)

    def record(message: str) -> None:
        change_logs.append(message)
        logger.info(message)
        print(message)

    character_bounds = _find_lore_section(lore, CHARACTERS_SECTION)
    if character_bounds:
        _, body_start, body_end = character_bounds
        characters, _ = _deduplicate_character_entries(
            _parse_bullet_entries(lore[body_start:body_end])
        )
    else:
        characters = []

    for raw_name, raw_value in _parse_bullet_entries(new_characters):
        if _is_disposable_unnamed_character(raw_name, raw_value):
            continue
        incoming_aliases = _character_alias_keys(raw_name)
        match_index = None
        for index, (existing_name, existing_value) in enumerate(characters):
            if (
                incoming_aliases & _character_alias_keys(existing_name)
                or _character_identities_match(
                    existing_name,
                    existing_value,
                    raw_name,
                    raw_value,
                )
            ):
                match_index = index
                break

        is_delete = _strip_balanced_brackets(raw_value).casefold() == "delete"
        log_key = _plain_key(raw_name)
        if is_delete:
            if match_index is not None:
                characters.pop(match_index)
                record(f"[Novel Context] Deleted obsolete Character '{log_key}'")
            continue

        canonical_name = _canonical_display_name(raw_name)
        clean_value, explicit_correction = _strip_character_correction_marker(
            raw_value
        )
        clean_value = _normalize_character_value(clean_value)
        if match_index is None:
            characters.append((canonical_name, clean_value))
            record(
                f"[Novel Context] Added Character: "
                f"{_format_character_line(canonical_name, clean_value)}"
            )
            continue

        old_name, old_value = characters[match_index]
        merged_name = _preferred_character_name(old_name, canonical_name)
        merged_value = _merge_character_values(
            old_value,
            clean_value,
            allow_gender_correction=explicit_correction,
        )
        if (old_name, old_value) != (merged_name, merged_value):
            record(
                f"[Novel Context] Corrected/Updated Character '{log_key}': "
                f"{_format_character_line(old_name, old_value)} -> "
                f"{_format_character_line(merged_name, merged_value)}"
            )
        characters[match_index] = (merged_name, merged_value)

    characters, _ = _deduplicate_character_entries(characters)
    lore = _replace_lore_section(
        lore,
        CHARACTERS_SECTION,
        [_format_character_line(name, value) for name, value in characters],
    )

    glossary_bounds = _find_lore_section(lore, GLOSSARY_SECTION)
    if glossary_bounds:
        _, body_start, body_end = glossary_bounds
        glossary = _normalize_glossary_entries(
            _parse_bullet_entries(lore[body_start:body_end])
        )
    else:
        glossary = []
    glossary_index = {
        _plain_key(name): index for index, (name, _) in enumerate(glossary)
    }

    for raw_name, raw_value in _parse_bullet_entries(new_glossary):
        key = _plain_key(raw_name)
        is_delete = _strip_balanced_brackets(raw_value).casefold() == "delete"
        if is_delete:
            if key in glossary_index:
                glossary.pop(glossary_index[key])
                glossary_index = {
                    _plain_key(name): index
                    for index, (name, _) in enumerate(glossary)
                }
                record(f"[Novel Context] Deleted obsolete Glossary Entry '{key}'")
            continue

        clean_name = _strip_balanced_brackets(raw_name)
        clean_value = _strip_balanced_brackets(raw_value)
        if key in glossary_index:
            index = glossary_index[key]
            old_name, old_value = glossary[index]
            if (old_name, old_value) != (clean_name, clean_value):
                record(
                    f"[Novel Context] Corrected/Updated Glossary Entry '{key}': "
                    f"- {old_name}: {old_value} -> - {clean_name}: {clean_value}"
                )
            glossary[index] = (clean_name, clean_value)
        else:
            glossary_index[key] = len(glossary)
            glossary.append((clean_name, clean_value))
            record(
                f"[Novel Context] Added Glossary Entry: "
                f"- {clean_name}: {clean_value}"
            )

    lore = _replace_lore_section(
        lore,
        GLOSSARY_SECTION,
        [f"- {name}: {value}" for name, value in glossary],
    )
    return normalize_global_lore(lore), change_logs


async def update_novel_context_chunk(
    llm_client: Any,
    model_name: str,
    current_global_lore: str,
    current_dynamic_state: str,
    source_chunk: str,
    translated_chunk: Optional[str],
    source_language: str,
    target_language: str,
    chunk_index: int = 0,
    total_chunks: int = 0,
    dialogue_turns: Optional[List[Dict[str, str]]] = None,
    current_dialogue_state: Optional[Dict[str, str]] = None,
    dialogue_attribution_sink: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, List[str]]:
    """Calls the LLM to update global lore and dynamic state incrementally.
    
    Returns:
        Tuple of (updated_global_lore, updated_dynamic_state, change_logs)
    """
    from src.utils.dialogue_attribution import (
        dialogue_candidates_prompt,
        empty_dialogue_attribution,
        parse_dialogue_attribution,
    )

    dialogue_turns = list(dialogue_turns or [])
    current_dialogue_state = dict(current_dialogue_state or {})
    prompt_values = {
        "current_global_lore": current_global_lore,
        "current_dynamic_state": current_dynamic_state,
        "source_language": source_language,
        "target_language": target_language,
        "source_chunk": source_chunk,
        "translated_chunk": translated_chunk or "",
        "chunk_index": chunk_index if chunk_index > 0 else "?",
        "total_chunks": total_chunks if total_chunks > 0 else "?",
        "current_dialogue_state": (
            current_dialogue_state or {"speaker": "Unknown", "addressee": "Unknown"}
        ),
        "dialogue_candidates": dialogue_candidates_prompt(dialogue_turns),
    }
    if translated_chunk is None:
        user_prompt = SOURCE_ANALYSIS_USER_PROMPT_TEMPLATE.format(**prompt_values)
        system_prompt = SOURCE_ANALYSIS_SYSTEM_PROMPT
    else:
        user_prompt = UPDATE_USER_PROMPT_TEMPLATE.format(**prompt_values)
        system_prompt = UPDATE_SYSTEM_PROMPT
    
    try:
        response = await llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )
        
        if not response or not response.content:
            logger.warning("Empty response received from LLM during novel context chunk update. Keeping current state.")
            if dialogue_attribution_sink is not None:
                dialogue_attribution_sink.clear()
                dialogue_attribution_sink.update(
                    empty_dialogue_attribution(current_dialogue_state)
                )
            return current_global_lore, current_dynamic_state, []
            
        content = response.content.strip()
        
        # Parse blocks
        new_chars = ""
        new_glossary = ""
        new_dynamic = current_dynamic_state
        
        import re
        chars_match = re.search(r'\[NEW_CHARACTERS\]\s*(.*?)\s*(?=\[NEW_GLOSSARY\]|\[DYNAMIC_STATE\]|$)', content, re.DOTALL)
        glossary_match = re.search(r'\[NEW_GLOSSARY\]\s*(.*?)\s*(?=\[DYNAMIC_STATE\]|\[NEW_CHARACTERS\]|$)', content, re.DOTALL)
        dynamic_match = re.search(
            r'\[DYNAMIC_STATE\]\s*(.*?)\s*(?=\[DIALOGUE_ATTRIBUTION\]|$)',
            content,
            re.DOTALL,
        )
        dialogue_match = re.search(
            r'\[DIALOGUE_ATTRIBUTION\]\s*(.*?)\s*$',
            content,
            re.DOTALL,
        )
        
        if chars_match:
            new_chars = chars_match.group(1).strip()
        if glossary_match:
            new_glossary = glossary_match.group(1).strip()
        if dynamic_match:
            new_dynamic = dynamic_match.group(1).strip()
            if new_dynamic.startswith("```"):
                lines = new_dynamic.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                new_dynamic = "\n".join(lines).strip()
            
            # Clean dynamic state boundaries if the model generated them by mistake
            new_dynamic = new_dynamic.replace("---DYNAMIC_STATE_START---", "").replace("---DYNAMIC_STATE_END---", "").strip()
            
            # Clean up `# DYNAMIC RELATIONSHIP STATE` headers
            lines = new_dynamic.splitlines()
            cleaned_lines = []
            for line in lines:
                if line.strip().upper().replace(" ", "") == "#DYNAMICRELATIONSHIPSTATE":
                    continue
                cleaned_lines.append(line)
            
            new_dynamic = "\n".join(cleaned_lines).strip()
            
        updated_global_lore, change_logs = merge_new_lore(current_global_lore, new_chars, new_glossary)
        new_dynamic = merge_dynamic_state(
            current_dynamic_state,
            new_dynamic,
            _character_alias_map(updated_global_lore),
        )
        
        # Track relationship logs by comparing old and new dynamic state
        # (This is secondary logging for the terminal, just printing line updates)
        if new_dynamic != current_dynamic_state:
            # We can log that relationship state changed
            change_logs.append("[Novel Context] Dynamic relationship state / addressing forms updated.")
            print("[Novel Context] Dynamic relationship state / addressing forms updated.")

        if dialogue_attribution_sink is not None:
            dialogue_attribution_sink.clear()
            dialogue_attribution_sink.update(
                parse_dialogue_attribution(
                    dialogue_match.group(1).strip() if dialogue_match else "",
                    dialogue_turns,
                    _character_alias_map(updated_global_lore),
                    current_dialogue_state,
                )
            )
            
        return updated_global_lore, new_dynamic, change_logs
        
    except Exception as e:
        logger.error(f"Error in update_novel_context_chunk: {e}")
        if dialogue_attribution_sink is not None:
            dialogue_attribution_sink.clear()
            dialogue_attribution_sink.update(
                empty_dialogue_attribution(current_dialogue_state)
            )
        return current_global_lore, current_dynamic_state, []
