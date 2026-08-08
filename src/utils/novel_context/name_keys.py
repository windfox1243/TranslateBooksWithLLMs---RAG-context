"""Turning a character name into keys, and comparing two names.

Everything here is about the *name* -- normalising it, deriving the keys the
rest of the package matches on, judging whether a name or alias is usable at
all, and deciding whether two names denote one person. None of it reads the
detail half of a character value, which is why this layer can sit underneath
the rest of the character code instead of alongside it.

The lore-section helpers live here for the same reason: they only see section
headings and bullet text."""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

from .constants import (
    _ADDRESS_TERM_SUFFIXES,
    _AMBIGUOUS_SHORT_NAME_KEYS,
    _BARE_NARRATIVE_ROLE_NAMES,
    _CJK_GENERIC_ROLE_NAMES,
    _ENGLISH_GENERIC_ROLE_NAMES,
    _GENERIC_ROLE_WORDS,
    _INVALID_CONTEXT_KEYS,
    _KINSHIP_WORDS,
    _MULTIWORD_NAME_TITLES,
    _NAME_TITLES,
    _NARRATIVE_ROLE_NAME_PATTERN,
    _NARRATIVE_WORK_PATTERN,
    _PHYSICAL_DESCRIPTOR_ANCHORS,
    _PHYSICAL_DESCRIPTOR_RELATIONS,
    _RECURRING_CHARACTER_MARKERS,
    _RELATIVE_AGE_WORDS,
    _ROLE_ONLY_TITLES,
    _ROLE_TITLE_KEYS,
    _ROMANTIC_RELATION_LABELS,
    _TRANSFERABLE_ROLE_ONLY_NAMES,
    _UNIQUE_ROLE_TITLES,
    ALIASES_SECTION,
    CHARACTERS_SECTION,
    GLOSSARY_SECTION,
    NAME_MAP_SECTION,
)

_UNSTABLE_PHYSICAL_WORDS = {
    "boy",
    "girl",
    "man",
    "woman",
    "child",
    "kid",
    "baby",
    "toddler",
    "youth",
    "elder",
    "the boy",
    "the girl",
    "the man",
    "the woman",
    "the child",
    "the kid",
    "the baby",
    "the youth",
    "the elder",
}

_LATIN_BOUNDARY_CHARS = "A-Za-z0-9À-ÖØ-öø-ÿ_'’-"

_LATIN_NAME_PART_STOPWORDS = {
    "al",
    "bin",
    "da",
    "de",
    "del",
    "der",
    "di",
    "du",
    "el",
    "la",
    "le",
    "of",
    "the",
    "van",
    "von",
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

def _relation_label_key(value: str) -> str:
    """Normalize hyphen/space variants of relationship labels."""
    return re.sub(r"\s+", " ", _plain_key(value).replace("-", " ")).strip()

def _generic_role_base_key(name: str) -> str:
    key = _plain_key(name).replace("'s", "")
    key = re.sub(r"\s+(?:#?\d+|[ivxlcdm]+)$", "", key)
    key = re.sub(
        r"\b(?:allied|background|dead|dying|enemy|fallen|female|generic|"
        r"imperial|injured|male|republic|screaming|unnamed|vampire|"
        r"wounded|young|old)\b",
        " ",
        key,
    )
    words = [word for word in key.split() if word]
    return words[-1] if words and words[-1] in _GENERIC_ROLE_WORDS else ""

def _strip_trailing_qualifier(name: str) -> str:
    """Treat state/form qualifiers as attributes of the same character."""
    return re.sub(r"\s*\([^()]+\)\s*$", "", name).strip()

def _strip_leading_article(name: str) -> str:
    return re.sub(r"^(?:the)\s+", "", name, flags=re.IGNORECASE).strip()

def _strip_name_title(name: str) -> str:
    parts = name.split()
    folded_parts = [part.rstrip(".").casefold() for part in parts]
    for title_parts in _MULTIWORD_NAME_TITLES:
        if tuple(folded_parts) == title_parts:
            return name
        if (
            len(parts) > len(title_parts)
            and tuple(folded_parts[:len(title_parts)]) == title_parts
        ):
            return " ".join(parts[len(title_parts):]).strip()
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

def _strip_address_suffix_key(name: str) -> str:
    """Collapse romanized address forms such as Akane-san to Akane."""
    suffix_pattern = "|".join(re.escape(suffix) for suffix in _ADDRESS_TERM_SUFFIXES)
    match = re.match(
        rf"^(?P<base>.+?)(?:[-\s]+)(?:{suffix_pattern})$",
        _strip_balanced_brackets(name).strip(),
        flags=re.IGNORECASE,
    )
    if match:
        return _plain_key(match.group("base"))
    return ""

def _compact_name_key(name: str) -> str:
    key = _plain_key(_canonical_display_name(name))
    compact = re.sub(r"[^0-9a-z]+", "", key)
    return compact if len(compact) >= 12 else ""

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
        _strip_address_suffix_key(no_title),
        _compact_name_key(no_title),
    }
    return {alias for alias in aliases if alias and not _is_invalid_context_key(alias)}

def _monarch_role(name: str) -> str:
    no_article = _strip_leading_article(_strip_trailing_qualifier(
        _strip_balanced_brackets(name)
    ))
    first_word = no_article.split(maxsplit=1)[0].rstrip(".").casefold() if no_article else ""
    return first_word if first_word in _UNIQUE_ROLE_TITLES else ""

def _is_role_only_name(name: str) -> bool:
    return _plain_key(_canonical_display_name(name)) in _UNIQUE_ROLE_TITLES

def _role_title_key_from_name(name: str) -> str:
    key = _plain_key(_canonical_display_name(name))
    return key if key in _ROLE_ONLY_TITLES else ""

def _role_title_keys_from_fact(fact: str) -> set[str]:
    clean = _clean_inline_text(fact)
    if not clean:
        return set()
    keys: set[str] = set()
    for title_key in _ROLE_TITLE_KEYS:
        title_pattern = re.escape(title_key).replace(r"\ ", r"\s+")
        if re.search(
            rf"^(?:the\s+)?{title_pattern}\b"
            r"(?=\s*(?:,|;|and\b|of\b|who\b|with\b|$))",
            clean,
            flags=re.IGNORECASE,
        ):
            keys.add(title_key)
            continue
        if re.search(
            r"\b(?:is|was|becomes|became|serves\s+as|introduced\s+as|"
            r"identified\s+as|revealed\s+as|known\s+as)\s+"
            rf"(?:the\s+)?{title_pattern}\b",
            clean,
            flags=re.IGNORECASE,
        ):
            keys.add(title_key)
    return keys

def _name_specificity(name: str) -> Tuple[int, int, int, int]:
    canonical = _canonical_display_name(name)
    key = _plain_key(canonical)
    role_only = int(key not in _ROLE_ONLY_TITLES)
    has_no_kinship = int(not any(w.lower() in _KINSHIP_WORDS for w in canonical.split()))
    no_parenthetical = int("(" not in name and ")" not in name)
    return role_only, has_no_kinship, len(canonical.split()), no_parenthetical

def _preferred_character_name(first: str, second: str) -> str:
    candidates = [_canonical_display_name(first), _canonical_display_name(second)]
    return max(candidates, key=_name_specificity)

def _singularize_simple_english_token(token: str) -> str:
    """Return a conservative singular form for English metadata comparisons."""
    token = token.casefold()
    if len(token) <= 3 or token.endswith("ss"):
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("es") and re.search(r"(?:ches|shes|xes|zes|ses)$", token):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token

def _simple_singular_name_key(name: str) -> str:
    words = _plain_key(_canonical_display_name(name)).split()
    if len(words) < 2:
        return ""
    words[-1] = _singularize_simple_english_token(words[-1])
    return " ".join(words)

def _simple_plural_name_keys_match(first: str, second: str) -> bool:
    first_key = _plain_key(_canonical_display_name(first))
    second_key = _plain_key(_canonical_display_name(second))
    if not first_key or not second_key or first_key == second_key:
        return False
    return (
        _simple_singular_name_key(first)
        and _simple_singular_name_key(first) == _simple_singular_name_key(second)
    )

def _label_has_cjk_or_hangul(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff\uf900-\ufaff\uac00-\ud7a3]", value))

def _name_reference_pattern(name: str) -> str:
    escaped = re.escape(_canonical_display_name(name))
    return rf"(?<![\w'-]){escaped}(?![\w'-])"

def _text_mentions(value: str, reference_text: str) -> bool:
    return _reference_text_mentions_label(value, reference_text)

def _reference_text_mentions_label(value: str, reference_text: str) -> bool:
    clean = _strip_balanced_brackets(_clean_inline_text(value))
    if _is_invalid_context_key(clean):
        return False
    reference = str(reference_text or "")
    if not reference.strip():
        return False

    if _label_has_cjk_or_hangul(clean):
        compact_value = re.sub(r"\s+", "", clean).casefold()
        compact_reference = re.sub(r"\s+", "", reference).casefold()
        return bool(compact_value and compact_value in compact_reference)

    pattern = re.escape(clean).replace(r"\ ", r"\s+")
    if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", clean):
        return bool(
            re.search(
                rf"(?<![{_LATIN_BOUNDARY_CHARS}]){pattern}(?![{_LATIN_BOUNDARY_CHARS}])",
                reference,
                flags=re.IGNORECASE,
            )
        )

    return clean.casefold() in reference.casefold()

def _reference_mentions_latin_name_part(
    name: str,
    reference_text: str,
) -> bool:
    """Match distinctive short forms for multi-part romanized names.

    Context rows often store full names such as "Frondier De Roach", while
    narration and dialogue use only "Frondier". Treating meaningful name parts
    as selection hints keeps relevant addressing rows in the prompt without
    requiring every short form to be discovered as an explicit alias first.
    """
    folded_reference = str(reference_text or "").casefold()
    if not folded_reference.strip():
        return False

    clean_name = _strip_balanced_brackets(_clean_inline_text(name))
    parts = [
        part
        for part in re.findall(r"[A-Za-z][A-Za-z'_-]{2,}", clean_name)
        if part.casefold() not in _LATIN_NAME_PART_STOPWORDS
    ]
    if len(parts) < 2:
        return False

    for part in parts:
        pattern = rf"(?<![A-Za-z]){re.escape(part.casefold())}(?![A-Za-z])"
        if re.search(pattern, folded_reference):
            return True
    return False

def _is_invalid_context_key(value: str) -> bool:
    key = _plain_key(value)
    return (
        key in _INVALID_CONTEXT_KEYS
        or key in _BARE_NARRATIVE_ROLE_NAMES
        or not re.search(r"\w", key, re.UNICODE)
    )

def _is_quarantined_character_entry(name: str, value: str = "") -> bool:
    """Return whether an entry is usable as terminology, not a character."""
    del value
    return (
        _is_descriptive_role_name(name)
        or _role_title_key_from_name(name) != ""
        or _is_transferable_role_only_name(name)
    )

def _is_unstable_identity_alias(alias: str, allow_physical: bool = False) -> bool:
    """Reject scene-local descriptions that are not stable identity labels."""
    key = _plain_key(alias)
    meta_roles = {
        "hero",
        "protagonist",
        "the hero",
        "the protagonist",
        "the user",
        "user",
    }
    if key in meta_roles or _is_descriptive_role_name(alias):
        return True
    if _relation_label_key(alias) in _ROMANTIC_RELATION_LABELS:
        return True
    if not allow_physical:
        if key in _UNSTABLE_PHYSICAL_WORDS:
            return True
    return False

def _short_name_alias_key(name: str) -> str:
    """Return a safe single-token name that may alias a longer full name."""
    key = _plain_key(_canonical_display_name(name))
    words = key.split()
    if (
        len(words) != 1
        or len(key) < 3
        or key in _AMBIGUOUS_SHORT_NAME_KEYS
        or key in _GENERIC_ROLE_WORDS
        or key in _ROLE_ONLY_TITLES
        or _is_quarantined_character_entry(key)
        or _is_unstable_identity_alias(key, allow_physical=True)
    ):
        return ""
    return key

def _small_typo_distance(first: str, second: str) -> bool:
    if first == second:
        return True
    if abs(len(first) - len(second)) > 1:
        return False
    previous = list(range(len(second) + 1))
    for i, left_char in enumerate(first, 1):
        current = [i]
        for j, right_char in enumerate(second, 1):
            substitution = previous[j - 1] + (left_char != right_char)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] <= 1

def _loose_romanized_token_match(first: str, second: str) -> bool:
    """Match tiny romanization/OCR variants such as Blady/Bladi or Vladi/Bladi."""
    if _small_typo_distance(first, second):
        return True
    first_folded = first.replace("v", "b").replace("y", "i")
    second_folded = second.replace("v", "b").replace("y", "i")
    return _small_typo_distance(first_folded, second_folded)

def _similar_full_names_match(first: str, second: str) -> bool:
    first_parts = _plain_key(_canonical_display_name(first)).split()
    second_parts = _plain_key(_canonical_display_name(second)).split()
    if len(first_parts) < 2 or len(first_parts) != len(second_parts):
        return False
    if first_parts[0] != second_parts[0] or first_parts[-1] != second_parts[-1]:
        return False
    if first_parts == second_parts:
        return False
    return all(
        _loose_romanized_token_match(left, right)
        for left, right in zip(first_parts[1:-1], second_parts[1:-1])
    )

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

def _is_cjk_generic_role_only_name(name: str) -> bool:
    compact = re.sub(r"\s+", "", _strip_balanced_brackets(name))
    return compact in _CJK_GENERIC_ROLE_NAMES

def _is_numbered_generic_role_name(name: str) -> bool:
    key = _plain_key(name)
    return bool(
        re.search(r"(?:^|\s)(?:#?\d+|[ivxlcdm]+)$", key)
        and _generic_role_base_key(name)
    )

def _is_english_generic_role_only_name(name: str) -> bool:
    key = _plain_key(name).replace("'s", "")
    key = re.sub(
        r"\b(?:allied|background|dead|dying|enemy|fallen|female|generic|"
        r"imperial|injured|male|republic|screaming|unnamed|vampire|"
        r"wounded|young|old)\b",
        " ",
        key,
    )
    words = [word for word in key.split() if word]
    if not words:
        return False
    return words[-1] in _ENGLISH_GENERIC_ROLE_NAMES

def _is_descriptive_role_name(name: str) -> bool:
    """Detect analysis labels that describe a role instead of naming a person."""
    key = _plain_key(name)
    return key in _BARE_NARRATIVE_ROLE_NAMES or bool(
        _NARRATIVE_ROLE_NAME_PATTERN.match(key)
    )

def _is_transferable_role_only_name(name: str) -> bool:
    """Detect role/address labels that should not become durable characters."""
    key = _plain_key(name)
    if not key:
        return False
    if key in _TRANSFERABLE_ROLE_ONLY_NAMES:
        return True
    words = key.split()
    if len(words) >= 2 and words[-1] in _ADDRESS_TERM_SUFFIXES:
        return True
    return False

def _has_recurring_character_marker(name: str, value: str) -> bool:
    text = _plain_key(f"{name} {value}")
    if "unnamed" in text:
        return False
    for marker in _RECURRING_CHARACTER_MARKERS:
        if marker == "named":
            if re.search(r"\bnamed\b", text):
                return True
            continue
        if marker in text:
            return True
    return False

def _is_distinctive_physical_descriptor(name: str, value: str = "") -> bool:
    """Allow stable unnamed descriptors such as "boy in the black coat"."""
    key = _plain_key(name)
    text = _plain_key(f"{name} {value}")
    if _has_recurring_character_marker(name, value):
        return True
    if not any(word in key.split() for word in _UNSTABLE_PHYSICAL_WORDS):
        return False
    if not any(
        re.search(rf"\b{re.escape(relation)}\b", key)
        for relation in _PHYSICAL_DESCRIPTOR_RELATIONS
    ):
        return False
    return any(anchor in text for anchor in _PHYSICAL_DESCRIPTOR_ANCHORS)

def _narrative_work_keys(name: str, value: str = "") -> set[str]:
    text = _clean_inline_text(f"{name}; {value}")
    keys = set()
    for match in _NARRATIVE_WORK_PATTERN.finditer(text):
        work = _plain_key(match.group("work"))
        work = re.sub(r"^(?:the\s+)?(?:game|novel|story|series)\s+", "", work)
        if work and work not in _INVALID_CONTEXT_KEYS:
            keys.add(work)
    return keys

def _character_narrative_role_alias_keys(name: str, value: str = "") -> set[str]:
    """Return deterministic aliases such as 'Protagonist of <work>'."""
    keys: set[str] = set()
    works = _narrative_work_keys(name, value)
    for work in works:
        for role in ("protagonist", "main protagonist", "main character", "hero"):
            keys.add(_plain_key(f"{role} of {work}"))
            keys.add(_plain_key(f"the {role} of {work}"))
    return {key for key in keys if key and key not in _INVALID_CONTEXT_KEYS}

def _full_name_contains_short_alias(full_name: str, short_key: str) -> bool:
    parts = _plain_key(_canonical_display_name(full_name)).split()
    return bool(short_key and len(parts) >= 2 and short_key in parts)

def _parse_kinship_name(name: str) -> Optional[Tuple[str, str]]:
    words = _canonical_display_name(name).split()
    if len(words) < 2:
        return None
    last_word = words[-1].lower().rstrip("'s").rstrip("’s")
    if last_word in _KINSHIP_WORDS:
        prefix = " ".join(words[:-1])
        return prefix, last_word
    return None

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
        if section_name in {
            CHARACTERS_SECTION,
            ALIASES_SECTION,
            NAME_MAP_SECTION,
        }:
            following_sections = {
                CHARACTERS_SECTION: (ALIASES_SECTION, NAME_MAP_SECTION, GLOSSARY_SECTION),
                ALIASES_SECTION: (NAME_MAP_SECTION, GLOSSARY_SECTION),
                NAME_MAP_SECTION: (GLOSSARY_SECTION,),
            }[section_name]
            next_section = next(
                (
                    found
                    for candidate in following_sections
                    for found in [_find_lore_section(lore, candidate)]
                    if found
                ),
                None,
            )
            if next_section:
                next_start = next_section[0]
                return (
                    f"{lore[:next_start].rstrip()}\n\n{replacement}\n\n"
                    f"{lore[next_start:].lstrip()}"
                ).strip() + "\n"
        separator = "\n\n" if lore.strip() else ""
        return f"{lore.rstrip()}{separator}{replacement}\n"
    start, _, end = bounds
    suffix = lore[end:].lstrip("\n")
    return f"{lore[:start].rstrip()}\n\n{replacement}\n\n{suffix}".strip() + "\n"

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
