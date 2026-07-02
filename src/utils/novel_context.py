"""
Utility functions for managing novel translation context files.

These context files track character genders, relationships (addressing forms),
and key glossary terms across translation segments, ensuring consistency.
"""
from __future__ import annotations

import json
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

WINDOWS_RESERVED_FILENAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
SAFE_FILENAME_PUNCTUATION = {"_", "-", "."}
DYNAMIC_STATE_START = "---DYNAMIC_STATE_START---"
DYNAMIC_STATE_END = "---DYNAMIC_STATE_END---"

CHARACTERS_SECTION = "## CHARACTERS & GENDERS"
ALIASES_SECTION = "## CHARACTER ALIASES"
NAME_MAP_SECTION = "## NAME TRANSLATION MAP"
GLOSSARY_SECTION = "## GLOSSARY & TERMINOLOGY"
ADDRESSING_SECTION = "## CURRENT ADDRESSING FORMS"
RELATIONSHIP_SECTION = "## RELATIONSHIP EVOLUTION"

_INVALID_CONTEXT_KEYS = {
    "",
    "-",
    "delete",
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
_BARE_NARRATIVE_ROLE_NAMES = {
    "hero",
    "main character",
    "main protagonist",
    "player character",
    "protagonist",
    "the hero",
    "the main character",
    "the main protagonist",
    "the player character",
    "the protagonist",
}
_TRANSFERABLE_ROLE_ONLY_NAMES = {
    "administrator",
    "game administrator",
    "gacha manager",
    "gacha room manager",
    "gacha room npc",
    "main player",
    "non player character",
    "non-player character",
    "npc",
    "player",
    "summon",
    "summoned character",
    "summoner",
    "summoner nim",
    "the summoner",
    "user",
}
_ADDRESS_TERM_SUFFIXES = {
    "nim",
    "sama",
    "san",
    "kun",
    "chan",
    "ssi",
    "sensei",
    "senpai",
    "sunbae",
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
    "lieutenant",
    "lord",
    "major",
    "marshal",
    "prince",
    "princess",
    "professor",
    "queen",
    "sergeant",
}
_MULTIWORD_NAME_TITLES = (
    ("lieutenant", "colonel"),
    ("lieutenant", "commander"),
    ("major", "general"),
)
_ROLE_ONLY_TITLES = _NAME_TITLES | {
    "lieutenant colonel",
    "lieutenant commander",
    "major general",
}
_ROLE_TITLE_KEYS = tuple(
    sorted(_ROLE_ONLY_TITLES, key=lambda item: (-len(item.split()), item))
)
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
    "corporal",
    "doctor",
    "guard",
    "knight",
    "manager",
    "medic",
    "officer",
    "private",
    "referee",
    "sergeant",
    "soldier",
    "soldiers",
    "victim",
    "student",
    "students",
    "classmate",
    "classmates",
    "teacher",
    "teachers",
    "bystander",
    "bystanders",
    "passerby",
    "passersby",
    "pedestrian",
    "pedestrians",
    "crowd",
    "people",
}
_CJK_GENERIC_ROLE_NAMES = {
    "同学",
    "学生",
    "男同学",
    "女同学",
    "男学生",
    "女学生",
    "男生",
    "女生",
    "老师",
    "教师",
    "路人",
    "行人",
}
_ENGLISH_GENERIC_ROLE_NAMES = {
    "student",
    "students",
    "classmate",
    "classmates",
    "teacher",
    "teachers",
    "bystander",
    "bystanders",
    "passerby",
    "passersby",
    "pedestrian",
    "pedestrians",
    "crowd",
    "people",
}
_CJK_NON_NAME_ADDRESS_LABELS = {
    "会长",
    "前辈",
    "后辈",
    "美少女",
    "学生会长",
}
_UNSET_NAME_TRANSLATION = "(not set)"
_AMBIGUOUS_SHORT_NAME_KEYS = {
    "baek",
    "choi",
    "dokgo",
    "han",
    "jang",
    "jeong",
    "jung",
    "kang",
    "kim",
    "kurosaki",
    "lee",
    "lim",
    "namgoong",
    "park",
    "seo",
    "shin",
    "yoon",
}
_SHORT_NAME_EVIDENCE_STOPWORDS = {
    "about",
    "against",
    "along",
    "another",
    "around",
    "character",
    "current",
    "currently",
    "figure",
    "former",
    "hostile",
    "other",
    "person",
    "protagonist",
    "student",
    "teacher",
    "toward",
    "towards",
    "whose",
    "with",
}
_INCIDENTAL_CHARACTER_MARKERS = {
    "abdominal wound",
    "advertisement",
    "attending physician",
    "author of the advertisement",
    "background",
    "body collection",
    "deceased",
    "doctor treating",
    "dying",
    "fallen",
    "generic",
    "incidental",
    "killed",
    "medical professional",
    "missing leg",
    "new recruit",
    "npc entity",
    "observing",
    "oversees",
    "one scene",
    "one-scene",
    "overseeing",
    "physician",
    "screaming in pain",
    "searching for",
    "severed arm",
    "soldier",
    "unnamed",
    "wounded",
    "programmed machine",
    "automated enforcement",
    "background npc",
    "minor npc",
    "unnamed npc",
    "throwaway",
    "one-off",
    "incidental npc",
    "minor role",
}
_EXPLICIT_NPC_MARKERS = {
    "programmed machine",
    "automated enforcement",
    "npc entity",
    "background npc",
    "minor npc",
    "unnamed npc",
    "throwaway",
    "one-off",
    "incidental npc",
}
_PHYSICAL_DESCRIPTOR_ANCHORS = {
    "armor",
    "armour",
    "badge",
    "black coat",
    "cloak",
    "coat",
    "dress",
    "eyepatch",
    "glasses",
    "hat",
    "hood",
    "jacket",
    "mask",
    "robe",
    "scar",
    "sword",
    "uniform",
    "weapon",
}
_PHYSICAL_DESCRIPTOR_RELATIONS = (
    "carrying",
    "holding",
    "in",
    "wearing",
    "with",
)
_GROUP_ENTITY_WORDS = {
    "academy",
    "agency",
    "army",
    "battalion",
    "clan",
    "company",
    "corporation",
    "country",
    "dynasty",
    "empire",
    "faction",
    "family",
    "force",
    "government",
    "guild",
    "house",
    "kingdom",
    "lineage",
    "military",
    "nation",
    "organization",
    "party",
    "school",
    "squad",
    "temple",
    "unit",
}
_RECURRING_CHARACTER_MARKERS = {
    "appears repeatedly",
    "canonical",
    "callsign",
    "code name",
    "codename",
    "important",
    "major character",
    "mentor",
    "named",
    "recurring",
    "repeatedly",
    "returns later",
    "source-named",
}
_NON_CHARACTER_METADATA_NAMES = {
    "author",
    "fan art",
    "hiatus",
    "notice",
    "serialization",
    "serialization time",
}
_NON_CHARACTER_METADATA_DETAIL_PATTERNS = (
    r"\b(?:author|writer|translator|illustrator)\s+of\s+(?:the\s+)?"
    r"(?:current\s+)?(?:work|novel|story|book|series)\b",
    r"\b(?:author|writer)\s*,\s*(?:writer\s+)?of\s+(?:the\s+)?"
    r"(?:current\s+)?(?:work|novel|story|book|series)\b",
    r"\b(?:posted|uploaded|published|serialized)\s+(?:chapter|episode|"
    r"notice|fan\s+art)\b",
)
_NON_CHARACTER_ITEM_DETAIL_PATTERNS = (
    r"\b(?:active|awakening|combat|passive|status|system)\s+skill\b",
    r"\bskill\s+that\b",
    r"\b(?:ability|buff|debuff|quest|stat|title)\s+that\b",
)
_ROMANTIC_RELATION_LABELS = {
    "beloved",
    "boyfriend",
    "ex boyfriend",
    "ex girlfriend",
    "ex lover",
    "ex partner",
    "fiance",
    "fiancee",
    "fiancé",
    "fiancée",
    "former boyfriend",
    "former girlfriend",
    "former lover",
    "former partner",
    "girlfriend",
    "husband",
    "lover",
    "partner",
    "romantic partner",
    "significant other",
    "spouse",
    "wife",
}
_GENDERED_ROMANTIC_RELATION_LABELS = {
    "boyfriend": "Male",
    "ex boyfriend": "Male",
    "fiance": "Male",
    "fiancé": "Male",
    "former boyfriend": "Male",
    "girlfriend": "Female",
    "ex girlfriend": "Female",
    "fiancee": "Female",
    "fiancée": "Female",
    "former girlfriend": "Female",
    "wife": "Female",
    "husband": "Male",
}
_KINSHIP_GENDERS = {
    "father": "Male",
    "mother": "Female",
    "brother": "Male",
    "sister": "Female",
    "son": "Male",
    "daughter": "Female",
    "husband": "Male",
    "wife": "Female",
    "uncle": "Male",
    "aunt": "Female",
    "grandfather": "Male",
    "grandmother": "Female",
    "nephew": "Male",
    "niece": "Female",
}
_KINSHIP_WORDS = set(_KINSHIP_GENDERS.keys())
_DIRECT_GENDER_WORDS = {
    "male": "Male",
    "female": "Female",
    "boy": "Male",
    "girl": "Female",
    "man": "Male",
    "woman": "Female",
    "gentleman": "Male",
    "lady": "Female",
    "guy": "Male",
}
_ROMANTIC_RELATION_PATTERN = (
    r"(?:(?:ex|former)[-\s]+)?(?:girlfriend|boyfriend|lover|partner|"
    r"spouse|wife|husband|romantic\s+partner|significant\s+other|"
    r"beloved|fianc[eé]e?)"
)
_RELATIONSHIP_OBJECT_PRONOUN_VERBS = (
    r"cheated\s+on|abandoned|left|betrayed|dumped|broke\s+up\s+with"
)
_WORK_ENTITY_WORDS = {
    "advertisement",
    "anime",
    "app",
    "book",
    "film",
    "game",
    "manga",
    "movie",
    "novel",
    "series",
    "story",
    "website",
    "webtoon",
}
_WORK_ENTITY_NON_PERSON_ROLES = {
    "administrator",
    "avatar",
    "character",
    "entity",
    "manager",
    "operator",
    "player",
    "protagonist",
    "user",
}
_NARRATIVE_ROLE_NAME_PATTERN = re.compile(
    r"^(?:the\s+)?(?:(?:main\s+)?protagonist|main\s+character|"
    r"hero|player\s+character|fictional\s+character)\s+"
    r"(?:of|in|from)\s+.+$",
    flags=re.IGNORECASE,
)
_NARRATIVE_WORK_PATTERN = re.compile(
    r"\b(?:(?:main\s+)?protagonist|main\s+character|hero|player\s+character)"
    r"\s+(?:of|in|from)\s+(?:the\s+)?(?:(?:game|novel|story|series)\s+)?"
    r"[\"'“”‘’]?(?P<work>[^;,.\"'“”‘’]+)",
    flags=re.IGNORECASE,
)


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


def _is_invalid_context_key(value: str) -> bool:
    key = _plain_key(value)
    return (
        key in _INVALID_CONTEXT_KEYS
        or key in _BARE_NARRATIVE_ROLE_NAMES
        or not re.search(r"\w", key, re.UNICODE)
    )


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


def _is_disposable_unnamed_character(name: str, value: str) -> bool:
    """Reject explicit one-off unnamed roles that cannot anchor consistency."""
    if _is_cjk_generic_role_only_name(name):
        return True
        
    # Proper named characters (e.g. Jenny, Kriha) must NEVER be discarded as unnamed roles
    role_key = _generic_role_base_key(name)
    if not role_key:
        return False
        
    description = _plain_key(value)
    
    # 1. If it has explicit NPC/machine markers, it's disposable (even if recurring is present)
    if any(marker in description for marker in _EXPLICIT_NPC_MARKERS):
        return True
        
    # 2. If it is marked as recurring, it is NOT disposable (saves e.g. crucial recurring teacher)
    if _has_recurring_character_marker(name, value):
        return False
        
    # 3. If it has generic incidental markers, it is disposable
    if any(marker in description for marker in _INCIDENTAL_CHARACTER_MARKERS):
        return True
        
    # 4. Check generic English roles (discard if not recurring)
    if _is_english_generic_role_only_name(name):
        return True
        
    if _is_numbered_generic_role_name(name):
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


def _is_unstable_physical_character_entry(name: str, value: str = "") -> bool:
    """Reject bare physical placeholders while preserving distinctive labels."""
    key = _plain_key(name)
    if not key:
        return False
    words = key.split()
    physical_words = {
        item for item in _UNSTABLE_PHYSICAL_WORDS
        if len(item.split()) == 1
    }
    if key in _UNSTABLE_PHYSICAL_WORDS:
        return not _is_distinctive_physical_descriptor(name, value)
    if not any(word in physical_words for word in words):
        return False
    if _is_distinctive_physical_descriptor(name, value):
        return False
    stripped_words = [
        word for word in words
        if word not in {
            "a",
            "an",
            "the",
            "young",
            "old",
            "older",
            "younger",
            "injured",
            "wounded",
            "dying",
            "screaming",
            "unnamed",
            "unknown",
        }
    ]
    return len(stripped_words) <= 1


def _is_non_character_work_entry(name: str, value: str) -> bool:
    """Reject works/apps or abstract concepts that the model put in the character registry."""
    name_key = _plain_key(name)
    gender, details = _split_gender_and_details(_normalize_character_value(value))
    key = _plain_key(details)
    if not key:
        return False
    gender = _canonical_gender(gender).casefold()

    # Check for abstract concepts, hallucinations, metaphors, or inanimate objects
    if re.search(
        r"\b(?:personified\s+(?:concept|manifestation)|abstract\s+concept|hallucination(?:\s+experienced\s+by)?|metaphor(?:ical)?(?:\s+representation)?|inanimate\s+object|not\s+a\s+character)\b",
        key,
    ):
        return True
    if re.search(
        r"\b(?:level|score|stat|stats|status|points?|grade|metric|"
        r"meter|gauge|window)\b",
        name_key,
    ) and re.search(
        r"\b(?:metric|score|stat|status|level|points?|representing|"
        r"measures?|tracks?|managed|favorability|growth)\b",
        key,
    ):
        return True
    if re.search(
        r"\b(?:center|centre|facility|building|hall|arena|room|office|"
        r"academy|school|association|kingdom|state|world|dimension)\b",
        name_key,
    ) and re.search(
        r"\b(?:facility|location|place|building|room|within|where|used\s+to|"
        r"used\s+for|located)\b",
        key,
    ):
        return True
    if re.search(r"\bmentioned\s+in\s+(?:an?\s+)?(?:episode|chapter)\s+title\b", key):
        return True
    if gender not in _SPECIFIC_GENDER_LABELS and re.search(
        r"\b(?:magic\s+(?:circle|array|formula|formation)|spell\s+circle|"
        r"magic\s+item|artifact|artefact|relic|weapon|sword|bow|shield|"
        r"item|reward|treasure|rune|sigil|spell|skill|ability|technique)\b",
        key,
    ) and re.search(
        r"\b(?:acquired|activated|cast|circle|conquest|drawn|formation|"
        r"granted|item|magic|obtained|reward|spell|stored|summoned|used|"
        r"weapon|wielded)\b",
        key,
    ):
        return True

    words = re.findall(r"\w+", key)
    if not words:
        return False
    if words[0] in _WORK_ENTITY_WORDS:
        if len(words) > 1 and words[1] in _WORK_ENTITY_NON_PERSON_ROLES:
            return False
        return True

    return bool(
        re.search(
            r"\b(?:game|novel|story|series|book|webtoon|manga|anime|"
            r"film|movie|app|website)\s+(?:title|work|setting)\b",
            key,
        )
    )


def _is_non_character_group_entry(name: str, value: str) -> bool:
    """Reject factions, countries, companies, and military units as characters."""
    name_key = _plain_key(name)
    gender, details = _split_gender_and_details(_normalize_character_value(value))
    details_key = _plain_key(details)
    if not name_key or not details_key:
        return False
    if _canonical_gender(gender).casefold() in _SPECIFIC_GENDER_LABELS:
        return False

    name_words = set(re.findall(r"\w+", name_key))
    group_word_pattern = (
        r"(?:academy|agency|army|battalion|clan|company|corporation|country|"
        r"dynasty|empire|faction|family|force|government|guild|house|"
        r"kingdom|lineage|military|nation|organization|party|school|"
        r"squad|temple|unit)"
    )
    details_mentions_group = bool(re.search(rf"\b{group_word_pattern}\b", details_key))
    if name_words & _GROUP_ENTITY_WORDS and details_mentions_group:
        return True
    return bool(
        re.search(rf"^(?:(?:a|an|the)\s+)?{group_word_pattern}\b", details_key)
        or re.search(
            rf"\b(?:known|described|identified|introduced)\s+as\s+(?:(?:a|an|the)\s+)?"
            rf"{group_word_pattern}\b",
            details_key,
        )
    )


def _is_non_character_metadata_or_item_entry(name: str, value: str) -> bool:
    """Reject author notes, publication metadata, skills, and system items."""
    name_key = _plain_key(name)
    _, details = _split_gender_and_details(_normalize_character_value(value))
    details_key = _plain_key(details)
    if name_key in _NON_CHARACTER_METADATA_NAMES:
        return True
    if details_key and any(
        re.search(pattern, details_key, flags=re.IGNORECASE)
        for pattern in _NON_CHARACTER_METADATA_DETAIL_PATTERNS
    ):
        return True
    if details_key and any(
        re.search(pattern, details_key, flags=re.IGNORECASE)
        for pattern in _NON_CHARACTER_ITEM_DETAIL_PATTERNS
    ):
        return True
    return False


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


def _is_quarantined_character_entry(name: str, value: str = "") -> bool:
    """Return whether an entry is usable as terminology, not a character."""
    del value
    return (
        _is_descriptive_role_name(name)
        or _role_title_key_from_name(name) != ""
        or _is_transferable_role_only_name(name)
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


_SOURCE_IDENTITY_ROLE_KEYS = {
    "lieutenant colonel",
    "lieutenant commander",
    "major general",
}


def _character_self_role_title_keys(name: str, value: str = "") -> set[str]:
    """Return role/title labels that the entry applies to itself.

    These are deterministic aliases, but only when the role appears as the
    character's own title. Phrases like "suspicious of the Lieutenant Colonel"
    deliberately do not match.
    """
    keys: set[str] = set()
    role_key = _role_title_key_from_name(name)
    if role_key:
        keys.add(role_key)

    _, details = _split_gender_and_details(_normalize_character_value(value))
    for fact in re.split(r"\s*;\s*", details):
        keys.update(_role_title_keys_from_fact(fact))
    return keys


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


def _singularized_detail_tokens(value: str) -> set[str]:
    return {
        _singularize_simple_english_token(token)
        for token in _detail_tokens(value)
    }


def _character_details_substantially_overlap(
    first_value: str,
    second_value: str,
) -> bool:
    first_gender, first_details = _split_gender_and_details(
        _normalize_character_value(first_value)
    )
    second_gender, second_details = _split_gender_and_details(
        _normalize_character_value(second_value)
    )
    if (
        first_gender.casefold() in _SPECIFIC_GENDER_LABELS
        and second_gender.casefold() in _SPECIFIC_GENDER_LABELS
        and first_gender.casefold() != second_gender.casefold()
    ):
        return False
    first_tokens = _singularized_detail_tokens(first_details)
    second_tokens = _singularized_detail_tokens(second_details)
    if not first_tokens or not second_tokens:
        return False
    shared = first_tokens & second_tokens
    overlap = len(shared) / min(len(first_tokens), len(second_tokens))
    return len(shared) >= 4 and overlap >= 0.70


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


def _parse_kinship_name(name: str) -> Optional[Tuple[str, str]]:
    words = _canonical_display_name(name).split()
    if len(words) < 2:
        return None
    last_word = words[-1].lower().rstrip("'s").rstrip("’s")
    if last_word in _KINSHIP_WORDS:
        prefix = " ".join(words[:-1])
        return prefix, last_word
    return None


def _kinship_identities_match(
    first_name: str,
    first_value: str,
    second_name: str,
    second_value: str,
) -> bool:
    parsed_first = _parse_kinship_name(first_name)
    parsed_second = _parse_kinship_name(second_name)
    if not parsed_first and not parsed_second:
        return False
        
    if parsed_first:
        kinship_name, kinship_val = first_name, first_value
        target_name, target_val = second_name, second_value
        prefix, kinship = parsed_first
    else:
        kinship_name, kinship_val = second_name, second_value
        target_name, target_val = first_name, first_value
        prefix, kinship = parsed_second
        
    prefix_key = _plain_key(prefix).rstrip("'s").rstrip("’s")
    target_name_key = _plain_key(target_name)
    
    # 1. Target name must start with or contain the prefix/family name
    if not (target_name_key.startswith(prefix_key) or prefix_key in target_name_key.split()):
        return False
        
    # 2. Gender compatibility
    kinship_gender = _KINSHIP_GENDERS.get(kinship)
    target_gender, target_details = _split_gender_and_details(_normalize_character_value(target_val))
    if kinship_gender and target_gender:
        if kinship_gender.casefold() != target_gender.casefold():
            return False
            
    # 3. Kinship role check in details
    target_text = _clean_inline_text(target_details).casefold()
    if kinship in target_text:
        return True
    target_words = target_name_key.split()
    if target_words and target_words[-1] in _KINSHIP_WORDS:
        target_kinship = target_words[-1]
        if _KINSHIP_GENDERS.get(target_kinship) == kinship_gender:
            return True
            
    return False


def _character_identities_match(
    first_name: str,
    first_value: str,
    second_name: str,
    second_value: str,
) -> bool:
    """Match deterministic aliases, including a unique title revealed in lore."""
    if _character_names_match(first_name, second_name):
        return True
    if _kinship_identities_match(first_name, first_value, second_name, second_value):
        return True
    if _simple_plural_name_keys_match(
        first_name,
        second_name,
    ) and _character_details_substantially_overlap(first_value, second_value):
        return True
    if _similar_full_names_match(
        first_name,
        second_name,
    ) and _character_details_substantially_overlap(first_value, second_value):
        return True
    first_descriptive = _is_descriptive_role_name(first_name)
    second_descriptive = _is_descriptive_role_name(second_name)
    if first_descriptive != second_descriptive:
        shared_works = (
            _narrative_work_keys(first_name, first_value)
            & _narrative_work_keys(second_name, second_value)
        )
        if shared_works:
            return True
    shared_roles = (
        _character_unique_roles(first_name, first_value)
        & _character_unique_roles(second_name, second_value)
    )
    return bool(
        shared_roles
        and (_is_role_only_name(first_name) or _is_role_only_name(second_name))
    )


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


def _full_name_contains_short_alias(full_name: str, short_key: str) -> bool:
    parts = _plain_key(_canonical_display_name(full_name)).split()
    return bool(short_key and len(parts) >= 2 and short_key in parts)


def _short_name_evidence_terms(value: str) -> set[str]:
    _, details = _split_gender_and_details(_normalize_character_value(value))
    terms: set[str] = set()
    for raw_token in re.findall(r"[a-z][a-z'-]{3,}", _plain_key(details)):
        evidence_term = raw_token.strip("'-")
        if (
            len(evidence_term) < 5
            or evidence_term in _SHORT_NAME_EVIDENCE_STOPWORDS
        ):
            continue
        if evidence_term.endswith("s") and len(evidence_term) > 5:
            evidence_term = evidence_term[:-1]
        if evidence_term and evidence_term not in _SHORT_NAME_EVIDENCE_STOPWORDS:
            terms.add(evidence_term)
    return terms


def _short_full_name_alias_supported(
    short_value: str,
    full_value: str,
) -> bool:
    short_gender, _ = _split_gender_and_details(
        _normalize_character_value(short_value)
    )
    full_gender, _ = _split_gender_and_details(
        _normalize_character_value(full_value)
    )
    if (
        short_gender.casefold() in _SPECIFIC_GENDER_LABELS
        and full_gender.casefold() in _SPECIFIC_GENDER_LABELS
        and short_gender.casefold() != full_gender.casefold()
    ):
        return False
    return bool(
        _short_name_evidence_terms(short_value)
        & _short_name_evidence_terms(full_value)
    )


def _infer_unique_short_name_alias_entries(
    entries: List[Tuple[str, str]],
    explicit_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Infer safe first-name aliases when a unique full-name entry proves it."""
    explicit_aliases = explicit_aliases or {}
    candidates: List[Tuple[str, str, str]] = []
    for raw_name, raw_value in entries:
        name = _canonical_display_name(raw_name)
        if (
            _is_invalid_context_key(name)
            or _is_non_character_work_entry(name, raw_value)
            or _is_non_character_group_entry(name, raw_value)
            or _is_non_character_metadata_or_item_entry(name, raw_value)
            or _is_disposable_unnamed_character(name, raw_value)
            or _is_unstable_physical_character_entry(name, raw_value)
            or _is_quarantined_character_entry(name, raw_value)
        ):
            continue
        candidates.append((name, raw_value, raw_name))

    inferred: Dict[str, str] = {}
    displays: Dict[str, str] = {}
    for short_name, short_value, raw_short_name in candidates:
        short_key = _short_name_alias_key(short_name)
        if not short_key or short_key in explicit_aliases:
            continue
        full_matches = [
            (full_name, full_value)
            for full_name, full_value, _ in candidates
            if (
                _plain_key(full_name) != short_key
                and _full_name_contains_short_alias(full_name, short_key)
            )
        ]
        if len(full_matches) != 1:
            continue
        full_name, full_value = full_matches[0]
        if not _short_full_name_alias_supported(short_value, full_value):
            continue
        inferred[short_key] = _canonical_display_name(full_name)
        displays[short_key] = _canonical_display_name(raw_short_name)
    return inferred, displays


def _infer_singular_plural_alias_entries(
    entries: List[Tuple[str, str]],
    explicit_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Infer aliases for near-identical singular/plural entity names."""
    explicit_aliases = explicit_aliases or {}
    candidates: List[Tuple[str, str, str]] = []
    for raw_name, raw_value in entries:
        name = _canonical_display_name(raw_name)
        if (
            _is_invalid_context_key(name)
            or _is_non_character_work_entry(name, raw_value)
            or _is_non_character_group_entry(name, raw_value)
            or _is_non_character_metadata_or_item_entry(name, raw_value)
            or _is_disposable_unnamed_character(name, raw_value)
            or _is_unstable_physical_character_entry(name, raw_value)
            or _is_quarantined_character_entry(name, raw_value)
        ):
            continue
        candidates.append((name, raw_value, raw_name))

    inferred: Dict[str, str] = {}
    displays: Dict[str, str] = {}
    for index, (left_name, left_value, left_raw) in enumerate(candidates):
        for right_name, right_value, right_raw in candidates[index + 1:]:
            if not (
                _simple_plural_name_keys_match(left_name, right_name)
                and _character_details_substantially_overlap(
                    left_value,
                    right_value,
                )
            ):
                continue
            target = _preferred_character_name(left_name, right_name)
            target_key = _plain_key(target)
            alias = right_raw if _plain_key(left_name) == target_key else left_raw
            for alias_key in _character_alias_keys(alias):
                if alias_key in explicit_aliases:
                    continue
                inferred[alias_key] = _canonical_display_name(target)
                displays[alias_key] = _canonical_display_name(alias)

    return inferred, displays


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


def _infer_gender_from_subject_relationship_label(details: str) -> str:
    """Infer subject gender from a relationship noun used as a descriptor."""
    text = _clean_inline_text(details)
    if not text:
        return ""
    pattern = (
        r"(?:^|[;,]\s*)"
        r"(?:(?:an?|the)\s+)?"
        r"(?:(?:[\w.-]+\s+){0,5}[\w.-]+['’]s\s+|"
        r"(?:his|her|their)\s+)?"
        rf"(?P<label>{_ROMANTIC_RELATION_PATTERN})\b"
        r"(?=\s*(?:,|;|who\b|and\b|$))"
    )
    genders = {
        _GENDERED_ROMANTIC_RELATION_LABELS.get(
            _relation_label_key(match.group("label")),
            "",
        )
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    }
    genders.discard("")
    return next(iter(genders)) if len(genders) == 1 else ""


def _infer_gender_from_character_details(details: str) -> str:
    """Recover explicit English evidence that a model left after Unspecified.

    Context metadata is required to be English so this conservative repair can
    recognize direct self-references without guessing from names or roles.
    """
    text = _clean_inline_text(details).casefold()
    if not text:
        return ""
    relationship_gender = _infer_gender_from_subject_relationship_label(
        details
    )
    if relationship_gender:
        return relationship_gender

    # Infer gender from kinship phrase in details (e.g. "father of...")
    clauses = [c.strip() for c in re.split(r"[,;]", details)]
    allowed_prefixes = {"a", "an", "the", "young", "younger", "old", "older", "eldest", "elder", "former", "deceased", "late", "beloved", "original", "only", "biological"}
    for clause in clauses:
        words = re.findall(r"\w+", clause.casefold())
        if not words:
            continue
        idx = 0
        while idx < len(words) and words[idx] in allowed_prefixes:
            idx += 1
        if idx < len(words) and words[idx] in _KINSHIP_GENDERS:
            if idx + 1 < len(words) and words[idx+1] == "of":
                return _KINSHIP_GENDERS[words[idx]]

    kinship_object = (
        r"(?:own|brother|sister|mother|father|family|wife|husband|son|daughter)"
    )
    male_patterns = (
        r"^(?:an?\s+)?(?:young\s+|old\s+)?(?:male|man|boy)\b",
        r"(?:^|[;,]\s*)(?:(?:an?|the)\s+)?(?:[\w'-]+\s+){0,5}(?:man|boy)\b",
        r"\b(?:described|identified|revealed|introduced|referred\s+to)\s+as\s+"
        r"(?:an?\s+)?(?:[\w'-]+\s+){0,5}(?:man|boy)\b",
        r"(?:^|[.;,]\s*)he\b",
        r"\bhimself\b",
        rf"\bwho\b[^.;]{{0,80}}\bhis\s+{kinship_object}\b",
    )
    female_patterns = (
        r"^(?:an?\s+)?(?:young\s+|old\s+)?(?:female|woman|girl)\b",
        r"(?:^|[;,]\s*)(?:(?:an?|the)\s+)?(?:[\w'-]+\s+){0,5}(?:woman|girl)\b",
        r"\b(?:described|identified|revealed|introduced|referred\s+to)\s+as\s+"
        r"(?:an?\s+)?(?:[\w'-]+\s+){0,5}(?:woman|girl)\b",
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


def _gender_from_evidence_note(note: str) -> str:
    text = _clean_inline_text(note).casefold()
    explicit = re.search(
        r"\b(?:as|to\s+be)\s+(male|female|non[- ]?binary)\b",
        text,
    )
    return (
        _canonical_gender(explicit.group(1).replace(" ", "-"))
        if explicit
        else ""
    )


def _strip_character_evidence_notes(
    value: str,
) -> Tuple[str, str, bool]:
    """Remove model explanations from canonical character metadata.

    Returns ``(clean_value, evidence_gender, correction)``.
    Evidence belongs in the analysis response, never in the durable profile.
    """
    clean = _strip_balanced_brackets(value).strip()
    evidence_gender = ""
    correction = False

    note_pattern = re.compile(
        r"""(?is)
        \s*[\(\[]\s*
        (?P<note>
            (?:(?:explicit\s+)?correction\s*:\s*)?
            (?:
                (?:gender\s+)?
                (?:confirmed|proven|established|determined|inferred|deduced)
                \b
                .*?
            )
        )
        \s*[\)\]]\s*[.;]?
        """
        ,
        flags=re.VERBOSE,
    )

    notes: List[str] = []

    def remove_note(match: re.Match) -> str:
        notes.append(match.group("note"))
        return " "

    clean = note_pattern.sub(remove_note, clean)
    trailing_pattern = re.compile(
        r"""(?is)
        \s*(?:;|\.|\-)\s*
        (?P<note>
            (?:(?:explicit\s+)?correction\s*:\s*)?
            (?:gender\s+)?
            (?:confirmed|proven|established|determined|inferred|deduced)
            \b.*
        )$
        """
        ,
        flags=re.VERBOSE,
    )
    trailing = trailing_pattern.search(clean)
    if trailing:
        notes.append(trailing.group("note"))
        clean = clean[:trailing.start()]

    for note in notes:
        note_gender = _gender_from_evidence_note(note)
        if note_gender:
            evidence_gender = note_gender
        if re.search(r"\bcorrection\s*:", note, flags=re.IGNORECASE):
            correction = True

    return (
        _clean_inline_text(clean).strip(" .;,"),
        evidence_gender,
        correction,
    )


def _split_embedded_character_value_fragments(value: str) -> List[str]:
    """Split malformed one-line entries that contain multiple gender headers."""
    parts = [
        _strip_balanced_brackets(part).strip()
        for part in re.split(r"\s*;\s*", value)
        if part.strip()
    ]
    if not parts:
        return []

    fragments = [parts[0]]
    embedded_header = re.compile(
        r"(?is)^(?:(?:explicit\s+)?(?:gender\s+)?correction\s*:|"
        r"(?:male|female|non[- ]?binary|nonbinary|unknown|unspecified)\s*,)"
    )
    for part in parts[1:]:
        if embedded_header.match(part):
            fragments.append(part)
        else:
            fragments[-1] = f"{fragments[-1]}; {part}"
    return fragments


def _normalize_character_value(value: str) -> str:
    clean, evidence_gender, _ = (
        _strip_character_evidence_notes(value)
    )
    fragments = _split_embedded_character_value_fragments(clean)
    gender = ""
    detail_fragments: List[str] = []

    for index, fragment in enumerate(fragments or [clean]):
        correction_match = re.match(
            r"(?is)^(?:explicit\s+)?(?:gender\s+)?correction\s*:\s*(.+)$",
            fragment,
        )
        is_correction = bool(correction_match)
        if correction_match:
            fragment = _strip_balanced_brackets(correction_match.group(1))

        fragment_gender, fragment_details = _split_gender_and_details(fragment)
        fragment_gender = _canonical_gender(fragment_gender)
        if fragment_gender:
            fragment_key = fragment_gender.casefold()
            current_key = gender.casefold()
            if fragment_key in _SPECIFIC_GENDER_LABELS:
                if (
                    not gender
                    or is_correction
                    or current_key not in _SPECIFIC_GENDER_LABELS
                    or index > 0
                ):
                    gender = fragment_gender
            elif not gender:
                gender = fragment_gender
        else:
            fragment_details = fragment

        if fragment_details:
            detail_fragments.append(fragment_details)

    details = _merge_character_details("; ".join(detail_fragments), "")
    if gender.casefold() in {"unknown", "unspecified"}:
        gender = (
            evidence_gender
            or _infer_gender_from_character_details(details)
            or "Unspecified"
        )
    elif not gender and (evidence_gender or details):
        gender = (
            evidence_gender
            or _infer_gender_from_character_details(details)
        )
    return f"{gender}, {details}".rstrip(" ,") if gender else details


def _remove_self_references_from_details(details: str, name: str) -> str:
    """Remove accidental self-listing from merged role descriptions."""
    canonical_name = _canonical_display_name(name)
    if not canonical_name:
        return details
    clean = details
    escaped = re.escape(canonical_name)
    clean = re.sub(
        rf"\s+\band\s+{escaped}\b(?=\s*(?:who\b|with\b|,|;|\.|$))",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        rf"\b{escaped}\s+and\s+(?=\w)",
        "",
        clean,
        flags=re.IGNORECASE,
    )

    filtered_facts: List[str] = []
    for raw_fact in re.split(r"\s*;\s*", clean):
        fact = _clean_inline_text(raw_fact).strip(" ;,").rstrip(" .")
        if not fact:
            continue
        if _is_character_meta_fact(fact, canonical_name):
            continue
        filtered_facts.append(fact)
    clean = "; ".join(filtered_facts)
    return _clean_inline_text(clean).strip(" ;,")


def _is_character_meta_fact(fact: str, name: str) -> bool:
    """Drop prompt/control descriptions that are not actual character facts."""
    if not fact:
        return False
    if _is_context_evidence_fact(fact):
        return True
    escaped = re.escape(_canonical_display_name(name))
    patterns = (
        rf"^(?:{escaped}'s\s+)?current\s+(?:rank\s+and\s+title|"
        r"title\s+and\s+rank|rank|title|nickname)$",
        rf"^(?:source\s+)?(?:rank|title|nickname)(?:\s*/\s*|"
        rf"\s+or\s+|\s+and\s+)?(?:rank|title|nickname)?\s+for\s+{escaped}$",
        rf"^title\s*/\s*nickname\s+for\s+{escaped}$",
        rf"^title\s+or\s+nickname\s+for\s+{escaped}$",
        rf"^{escaped}'s\s+(?:rank|title|nickname)(?:\s*/\s*|"
        r"\s+or\s+|\s+and\s+)?(?:rank|title|nickname)?$",
    )
    return any(re.search(pattern, fact, flags=re.IGNORECASE) for pattern in patterns)


def _infer_gender_from_kinship(name: str, details: str) -> str:
    # 1. Check name words
    name_words = {w.casefold() for w in re.findall(r"\w+", name)}
    
    # Check direct gender words (e.g. "Female Student", "Shy Boy")
    for word, gender in _DIRECT_GENDER_WORDS.items():
        if word in name_words:
            return gender
            
    # Check kinship words (e.g. "Shigure Father")
    for word, gender in _KINSHIP_GENDERS.items():
        if word in name_words:
            return gender
            
    # 2. Check details clauses
    clauses = [c.strip() for c in re.split(r"[,;]", details)]
    allowed_prefixes = {"a", "an", "the", "young", "younger", "old", "older", "eldest", "elder", "former", "deceased", "late", "beloved", "original", "only", "biological"}
    for clause in clauses:
        words = re.findall(r"\w+", clause.casefold())
        if not words:
            continue
        # Find the first word that is not in allowed_prefixes
        idx = 0
        while idx < len(words) and words[idx] in allowed_prefixes:
            idx += 1
        if idx < len(words) and words[idx] in _KINSHIP_GENDERS:
            # Check if followed by "of"
            if idx + 1 < len(words) and words[idx+1] == "of":
                return _KINSHIP_GENDERS[words[idx]]
    return ""


def _normalize_character_value_for_name(name: str, value: str) -> str:
    normalized = _normalize_character_value(value)
    gender, details = _split_gender_and_details(normalized)
    if _gender_belongs_to_reincarnated_current_body(name, gender, details):
        gender = _infer_previous_reincarnation_identity_gender(
            details
        ) or "Unspecified"
    elif _gender_belongs_to_previous_reincarnation_body(name, gender, details):
        gender = _current_reincarnated_form_gender(name, details) or "Unspecified"
    if gender.casefold() in {"unknown", "unspecified"} or not gender:
        gender = _infer_gender_from_kinship(name, details) or gender
    details = _normalize_reincarnation_details_for_name(name, details)
    details = _remove_self_references_from_details(details, name)
    details = _merge_character_details(details, "")
    return f"{gender}, {details}".rstrip(" ,") if gender else details


def _gender_from_body_word(value: str) -> str:
    key = _plain_key(value)
    if key in {"male", "man", "boy"}:
        return "Male"
    if key in {"female", "woman", "girl"}:
        return "Female"
    return ""


def _gender_belongs_to_previous_reincarnation_body(
    name: str,
    gender: str,
    details: str,
) -> bool:
    """Detect when the gender label describes the pre-reincarnation body.

    A profile keyed by the new/current name must store the current form's
    gender. If the details only say a man/woman reincarnated into this named
    form, that old-body noun is not valid gender evidence for the new identity.
    """
    if gender.casefold() not in _SPECIFIC_GENDER_LABELS:
        return False
    canonical = _canonical_display_name(name)
    if not canonical:
        return False
    name_pattern = re.escape(canonical)
    match = re.search(
        rf"\b(?P<body>male|female|man|woman|boy|girl)\b"
        rf"(?P<middle>[^.;]{{0,120}}?)\breincarnat\w+\s+as\b"
        rf"(?P<form>[^.;]{{0,120}}?)\bnamed\s+{name_pattern}\b",
        details,
        flags=re.IGNORECASE,
    )
    if not match:
        return False
    old_body_gender = _gender_from_body_word(match.group("body"))
    return old_body_gender.casefold() == gender.casefold()


def _looks_like_previous_reincarnation_identity(details: str) -> bool:
    key = _plain_key(details)
    if "reincarnat" not in key and "reborn" not in key and "transmigrat" not in key:
        return False
    previous_identity_markers = {
        "beta tester",
        "blood-related",
        "blood related",
        "blood disease",
        "dying",
        "former self",
        "human host",
        "illness",
        "original self",
        "patient",
        "previous body",
        "terminal",
        "terminally ill",
    }
    return any(marker in key for marker in previous_identity_markers)


def _current_reincarnation_body_matches_gender(
    details: str,
    gender: str,
    name: str,
) -> bool:
    canonical = _canonical_display_name(name)
    name_pattern = re.escape(canonical) if canonical else ""
    body_pattern = r"(?P<body>male|female|man|woman|boy|girl)"
    patterns = (
        r"\b(?:reincarnat\w+|reborn|transmigrat\w+)\s+(?:as|into)\s+"
        r"(?:an?\s+)?(?:[\w'-]+[,\s]+){0,7}"
        rf"{body_pattern}\b(?P<tail>[^.;]{{0,120}})",
        r"\b(?:became|become|becomes|becoming|woke\s+up\s+as|awoke\s+as|"
        r"turns\s+into|turned\s+into)\s+"
        r"(?:an?\s+)?(?:[\w'-]+[,\s]+){0,7}"
        rf"{body_pattern}\b(?P<tail>[^.;]{{0,120}})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, details, flags=re.IGNORECASE):
            body_gender = _gender_from_body_word(match.group("body"))
            if body_gender.casefold() != gender.casefold():
                continue
            tail = match.group("tail") or ""
            if name_pattern and re.search(
                rf"\bnamed\s+{name_pattern}\b",
                tail,
                flags=re.IGNORECASE,
            ):
                continue
            return True
    return False


def _gender_belongs_to_reincarnated_current_body(
    name: str,
    gender: str,
    details: str,
) -> bool:
    """Detect old-identity entries that borrowed the new body's gender.

    If a durable entry is the pre-reincarnation human/profile (for example a
    terminal patient or beta tester), a phrase like "reincarnated as a girl"
    describes the later body, not the original identity keyed by that name.
    """
    if gender.casefold() not in _SPECIFIC_GENDER_LABELS:
        return False
    if not _looks_like_previous_reincarnation_identity(details):
        return False
    return _current_reincarnation_body_matches_gender(details, gender, name)


def _details_without_current_reincarnation_body(details: str) -> str:
    clean = _clean_inline_text(details)
    body_pattern = r"(?:male|female|man|woman|boy|girl)"
    replacements = (
        (
            r"\b(?:reincarnat\w+|reborn|transmigrat\w+)\s+(?:as|into)\s+"
            r"(?:an?\s+)?(?:[\w'-]+[,\s]+){0,7}"
            rf"{body_pattern}\b",
            "reincarnated",
        ),
        (
            r"\b(?:became|become|becomes|becoming|woke\s+up\s+as|awoke\s+as|"
            r"turns\s+into|turned\s+into)\s+"
            r"(?:an?\s+)?(?:[\w'-]+[,\s]+){0,7}"
            rf"{body_pattern}\b",
            "changed form",
        ),
    )
    for pattern, replacement in replacements:
        clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)
    return _clean_inline_text(clean)


def _infer_previous_reincarnation_identity_gender(details: str) -> str:
    return _infer_gender_from_character_details(
        _details_without_current_reincarnation_body(details)
    )


def _current_reincarnated_form_gender(name: str, details: str) -> str:
    canonical = _canonical_display_name(name)
    name_pattern = re.escape(canonical) if canonical else r"\w+"
    patterns = (
        rf"\bnamed\s+{name_pattern}\b[^.;]{{0,120}}?"
        r"\b(?:became|become|becomes|as|into)\s+"
        r"(?:an?\s+)?(?:[\w'-]+[,\s]+){0,6}"
        r"(?P<body>male|female|man|woman|boy|girl)\b",
        r"\b(?:became|become|becomes|reincarnated\s+as|reincarnates\s+as|"
        r"reborn\s+as|turns\s+into|turned\s+into)\s+"
        r"(?:an?\s+)?(?:[\w'-]+[,\s]+){0,6}"
        r"(?P<body>male|female|man|woman|boy|girl)\b",
    )
    genders = {
        _gender_from_body_word(match.group("body"))
        for pattern in patterns
        for match in re.finditer(pattern, details, flags=re.IGNORECASE)
    }
    genders.discard("")
    return next(iter(genders)) if len(genders) == 1 else ""


def _normalize_reincarnation_details_for_name(name: str, details: str) -> str:
    canonical = _canonical_display_name(name)
    if not canonical:
        return details
    name_pattern = re.escape(canonical)

    def replace_old_body(match: re.Match) -> str:
        old_body = _clean_inline_text(match.group("body"))
        form = _clean_inline_text(match.group("form")).strip(" ,")
        article = "an" if old_body[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        form_phrase = f" as {form}" if form else ""
        return f"reincarnated from {article} {old_body}{form_phrase}"

    return re.sub(
        rf"\b(?:an?\s+)?"
        rf"(?P<body>(?:terminally\s+ill\s+)?(?:male|female|man|woman|boy|girl))"
        rf"\s+who\s+reincarnates\s+as\s+"
        rf"(?P<form>[^.;,]{{0,120}}?)\s+named\s+{name_pattern}\b",
        replace_old_body,
        details,
        flags=re.IGNORECASE,
    )


_REFERENCE_PRONOUN_GENDERS = {
    "her": "Female",
    "herself": "Female",
    "his": "Male",
    "himself": "Male",
}
_REFERENCE_PRONOUN_OBJECTS = (
    "identity",
    "true identity",
    "true self",
    "secret",
    "body",
    "appearance",
    "face",
    "life",
    "name",
    "past",
    "condition",
    "status",
)


def _set_character_gender(value: str, gender: str) -> str:
    clean = _normalize_character_value(value)
    _, details = _split_gender_and_details(clean)
    gender = _canonical_gender(gender)
    return f"{gender}, {details}".rstrip(" ,") if details else gender


def _name_reference_pattern(name: str) -> str:
    escaped = re.escape(_canonical_display_name(name))
    return rf"(?<![\w'-]){escaped}(?![\w'-])"


def _infer_gender_reference_to_character(details: str, name: str) -> str:
    """Infer gender from direct pronoun evidence attached to a named target.

    This repairs summaries such as "superior officer to Valentine, suspicious
    of her identity": the pronoun belongs to Valentine, not to the officer.
    """
    if _is_invalid_context_key(name):
        return ""
    text = _clean_inline_text(details)
    if not text:
        return ""
    name_pattern = _name_reference_pattern(name)
    object_pattern = "|".join(
        re.escape(item).replace(r"\ ", r"\s+")
        for item in sorted(
            _REFERENCE_PRONOUN_OBJECTS,
            key=lambda item: (-len(item), item),
        )
    )
    patterns = (
        rf"\b(?:to|of|for|with|about|toward|towards|against|around|"
        rf"regarding)\s+{name_pattern}"
        rf"(?!\s+(?:and|or)\b)[^.;:]{{0,100}}\b"
        rf"(?P<pronoun>her|his)\s+(?:{object_pattern})\b",
        rf"\b(?:suspect(?:s|ed)?|accuse(?:s|d)?|question(?:s|ed)?|"
        rf"doubt(?:s|ed)?|confront(?:s|ed)?|investigate(?:s|d)?|"
        rf"examine(?:s|d)?|interrogate(?:s|d)?|track(?:s|ed)?|"
        rf"watch(?:es|ed)?|recognize(?:s|d)?|identif(?:y|ies|ied)|"
        rf"discover(?:s|ed)?|find(?:s|ing)?|found|protect(?:s|ed)?|"
        rf"rescue(?:s|d)?|help(?:s|ed)?|attack(?:s|ed)?|follow(?:s|ed)?)"
        rf"\s+{name_pattern}"
        rf"(?!\s+(?:and|or)\b)[^.;:]{{0,100}}\b"
        rf"(?P<pronoun>her|his)\s+(?:{object_pattern})\b",
        rf"{name_pattern}(?!\s+(?:and|or)\b)[^.;:]{{0,100}}\b"
        rf"(?P<pronoun>herself|himself)\b",
    )
    genders = {
        _REFERENCE_PRONOUN_GENDERS[match.group("pronoun").casefold()]
        for pattern in patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    }
    return next(iter(genders)) if len(genders) == 1 else ""


def _apply_cross_character_gender_evidence(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    evidence: Dict[str, set[str]] = {}
    canonical_names = [
        item["name"]
        for item in items
        if not _is_invalid_context_key(item.get("name", ""))
    ]

    for source in items:
        _, source_details = _split_gender_and_details(
            _normalize_character_value(source.get("value", ""))
        )
        for target_name in canonical_names:
            gender = _infer_gender_reference_to_character(
                source_details,
                target_name,
            )
            if gender:
                evidence.setdefault(target_name, set()).add(gender)

    for item in items:
        genders = evidence.get(item["name"], set())
        if len(genders) != 1:
            continue
        item["value"] = _set_character_gender(
            item["value"],
            next(iter(genders)),
        )
    return items


def _strip_character_correction_marker(value: str) -> Tuple[str, bool]:
    clean, evidence_gender, evidence_correction = (
        _strip_character_evidence_notes(value)
    )
    match = re.match(
        r"(?is)^(?:explicit\s+)?(?:gender\s+)?correction\s*:\s*(.+)$",
        clean,
    )
    if not match:
        if evidence_gender:
            current_gender, details = _split_gender_and_details(clean)
            if current_gender.casefold() not in _SPECIFIC_GENDER_LABELS:
                clean = (
                    f"{evidence_gender}, {details}"
                    if details
                    else evidence_gender
                )
        return clean, evidence_correction

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
    normalized = _normalize_character_value_for_name(name, value)
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


def _deduplicate_character_entries(
    entries: List[Tuple[str, str]],
    explicit_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    normalized: List[Dict[str, Any]] = []
    explicit_aliases = explicit_aliases or {}
    regular_entries: List[Tuple[str, str]] = []
    descriptive_entries: List[Tuple[str, str]] = []
    for raw_name, raw_value in entries:
        if _is_descriptive_role_name(raw_name):
            descriptive_entries.append((raw_name, raw_value))
        else:
            regular_entries.append((raw_name, raw_value))

    for raw_name, raw_value in regular_entries + descriptive_entries:
        if (
            _is_invalid_context_key(raw_name)
            or _is_non_character_work_entry(raw_name, raw_value)
            or _is_non_character_group_entry(raw_name, raw_value)
            or _is_non_character_metadata_or_item_entry(raw_name, raw_value)
            or _is_disposable_unnamed_character(raw_name, raw_value)
        ):
            continue
        raw_aliases = _character_alias_keys(raw_name)
        forced_name = next(
            (
                explicit_aliases[alias]
                for alias in raw_aliases
                if alias in explicit_aliases
            ),
            None,
        )
        effective_name = forced_name or raw_name
        descriptive_name = bool(
            _is_descriptive_role_name(raw_name)
            and not forced_name
        )
        if (
            not forced_name
            and _is_unstable_physical_character_entry(raw_name, raw_value)
        ):
            continue
        aliases = raw_aliases | _character_alias_keys(effective_name)
        matching_indices = {
            index
            for index, item in enumerate(normalized)
            if (
                aliases & item["aliases"]
                or _character_identities_match(
                    item["name"],
                    item["value"],
                    effective_name,
                    raw_value,
                )
            )
        }
        if matching_indices:
            index = min(matching_indices)
            item = normalized[index]
            item["name"] = (
                _canonical_display_name(forced_name)
                if forced_name
                else item["name"]
                if descriptive_name
                else _preferred_character_name(item["name"], effective_name)
            )
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
        elif descriptive_name:
            continue
        else:
            normalized.append({
                "name": _canonical_display_name(effective_name),
                "value": _normalize_character_value(raw_value),
                "aliases": set(aliases),
            })

    normalized = _merge_role_only_entries_by_unique_self_title(normalized)
    normalized = _apply_cross_character_gender_evidence(normalized)

    alias_map: Dict[str, str] = {}
    result: List[Tuple[str, str]] = []
    for item in normalized:
        name = item["name"]
        value = _normalize_character_value_for_name(name, item["value"])
        result.append((name, value))
        for alias in (
            item["aliases"]
            | _character_alias_keys(name)
            | _character_self_role_title_keys(name, value)
            | _character_narrative_role_alias_keys(name, value)
        ):
            alias_map[alias] = name
    for alias, target in explicit_aliases.items():
        canonical_target = next(
            (
                name
                for name, value in result
                if _character_identities_match(
                    name,
                    value,
                    target,
                    "",
                )
            ),
            None,
        )
        if alias and canonical_target:
            alias_map[alias] = canonical_target
    return result, alias_map


def _merge_role_only_entries_by_unique_self_title(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge a bare role entry into the only named character carrying it.

    This fixes durable context pollution such as a separate "Lieutenant
    Colonel" character after Eric's own entry already says he is the
    Lieutenant Colonel. Ambiguous titles remain separate unless the model or
    user supplies an explicit alias.
    """
    normalized = list(items)
    changed = True
    while changed:
        changed = False
        title_to_named_indices: Dict[str, List[int]] = {}
        role_indices: List[Tuple[int, str]] = []

        for index, item in enumerate(normalized):
            role_key = _role_title_key_from_name(item["name"])
            if role_key:
                role_indices.append((index, role_key))
                continue

            for title_key in _character_self_role_title_keys(
                item["name"],
                item["value"],
            ):
                title_to_named_indices.setdefault(title_key, []).append(index)

        for role_index, role_key in role_indices:
            candidates = sorted(set(title_to_named_indices.get(role_key, [])))
            if len(candidates) != 1:
                continue

            target_index = candidates[0]
            if target_index == role_index:
                continue

            role_item = normalized[role_index]
            target_item = normalized[target_index]
            target_item["value"] = _merge_character_values(
                target_item["value"],
                role_item["value"],
            )
            target_item["aliases"].update(role_item["aliases"])
            target_item["aliases"].update(
                _character_alias_keys(role_item["name"])
            )
            normalized.pop(role_index)
            changed = True
            break

    return normalized


def _parse_alias_entries(text: str) -> List[Tuple[str, str]]:
    return [
        (_strip_balanced_brackets(alias), _strip_balanced_brackets(target))
        for alias, target in _parse_bullet_entries(text)
        if (
            not _is_invalid_context_key(alias)
            and not _is_invalid_context_key(target)
        )
    ]


def _alias_entries_to_map(
    entries: List[Tuple[str, str]],
) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for alias, target in entries:
        target_name = _canonical_display_name(target)
        if (
            _is_invalid_context_key(alias)
            or _is_invalid_context_key(target_name)
            or _character_names_match(alias, target_name)
            or _is_unstable_identity_alias(alias, allow_physical=True)
        ):
            continue
        for alias_key in _character_alias_keys(alias):
            aliases[alias_key] = target_name
    return aliases


def _canonical_alias_entries(
    aliases: Dict[str, str],
    characters: List[Tuple[str, str]],
    display_aliases: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """Render explicit aliases once, pointing at canonical character names."""
    canonical_by_key: Dict[str, str] = {}
    for name, _ in characters:
        for key in _character_alias_keys(name):
            canonical_by_key[key] = name

    output: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for alias_key, target in aliases.items():
        canonical_target = next(
            (
                canonical_by_key[key]
                for key in _character_alias_keys(target)
                if key in canonical_by_key
            ),
            None,
        )
        if (
            not alias_key
            or not canonical_target
            or _is_invalid_context_key(canonical_target)
            or alias_key in _character_alias_keys(canonical_target)
        ):
            continue
        display_alias = (display_aliases or {}).get(alias_key)
        if not display_alias:
            display_alias = " ".join(
                part.capitalize() if part.islower() else part
                for part in alias_key.split()
            )
        output[(_plain_key(display_alias), canonical_target)] = (
            display_alias,
            canonical_target,
        )
    return list(output.values())


def _retain_renderable_aliases(
    aliases: Dict[str, str],
    display_aliases: Dict[str, str],
    deduced_aliases: Dict[str, str],
    source_entries: List[Tuple[str, str]],
) -> None:
    def title_stripped_alias(alias_key: str, target: str) -> bool:
        target_keys = _character_alias_keys(target)
        parts = alias_key.split()
        return bool(
            len(parts) >= 2
            and parts[0] in _NAME_TITLES
            and " ".join(parts[1:]) in target_keys
        )

    for raw_name, _ in source_entries:
        display = _canonical_display_name(raw_name)
        if (
            _is_invalid_context_key(display)
            or _is_descriptive_role_name(display)
            or _role_title_key_from_name(display)
            or _is_quarantined_character_entry(display)
        ):
            continue
        for alias_key in _character_alias_keys(raw_name):
            target = deduced_aliases.get(alias_key)
            if (
                not target
                or alias_key in _character_alias_keys(target)
                or title_stripped_alias(alias_key, target)
            ):
                continue
            display_aliases.setdefault(alias_key, display)
    for alias_key, target in deduced_aliases.items():
        if (
            alias_key in _character_alias_keys(target)
            or title_stripped_alias(alias_key, target)
            or _is_descriptive_role_name(alias_key)
            or _role_title_key_from_name(alias_key)
            or _is_quarantined_character_entry(alias_key)
        ):
            continue
        aliases.setdefault(alias_key, target)


def _source_alias_displays_from_glossary(source: str) -> List[str]:
    clean = _strip_balanced_brackets(source)
    compact = re.sub(r"\s+", "", clean)
    if not re.fullmatch(r"[\u3400-\u9fff\uf900-\ufaff]{3,8}", compact):
        if re.fullmatch(r"[\uac00-\ud7a3]{3}", compact):
            return [compact, compact[-2:]]
        return [clean]
    displays = [compact]
    short = compact[-2:]
    if short != compact:
        displays.append(short)
    return displays


def _character_target_from_glossary_value(
    target: str,
    characters: List[Tuple[str, str]],
) -> str:
    target_name = _canonical_display_name(target)
    if _is_invalid_context_key(target_name):
        return ""
    target_keys = _character_alias_keys(target_name)
    matches = [
        name
        for name, _ in characters
        if (
            _character_names_match(name, target_name)
            or bool(target_keys & _character_alias_keys(name))
        )
    ]
    if len(matches) == 1:
        return matches[0]

    short_key = _short_name_alias_key(target_name)
    if short_key:
        short_matches = [
            name
            for name, _ in characters
            if _full_name_contains_short_alias(name, short_key)
        ]
        if len(short_matches) == 1:
            return short_matches[0]
    return ""


def _add_glossary_character_aliases(
    explicit_aliases: Dict[str, str],
    alias_displays: Dict[str, str],
    glossary_entries: List[Tuple[str, str]],
    characters: List[Tuple[str, str]],
) -> None:
    proposed: Dict[str, Tuple[str, str]] = {}
    ambiguous: set[str] = set()
    for raw_source, raw_target in glossary_entries:
        if _strip_balanced_brackets(raw_target).casefold() == "delete":
            continue
        target = _character_target_from_glossary_value(raw_target, characters)
        if not target:
            continue
        for display in _source_alias_displays_from_glossary(raw_source):
            if (
                _is_invalid_context_key(display)
                or _is_unstable_identity_alias(display, allow_physical=True)
            ):
                continue
            for alias_key in _character_alias_keys(display):
                if alias_key in _character_alias_keys(target):
                    continue
                existing = proposed.get(alias_key)
                if existing and existing[1] != target:
                    ambiguous.add(alias_key)
                    continue
                proposed[alias_key] = (display, target)

    for alias_key, (display, target) in proposed.items():
        if alias_key in ambiguous:
            continue
        existing_target = explicit_aliases.get(alias_key)
        if existing_target and existing_target != target:
            continue
        explicit_aliases[alias_key] = target
        alias_displays.setdefault(alias_key, display)


def _display_role_title(role_key: str) -> str:
    return " ".join(part.capitalize() for part in role_key.split())


def _candidate_named_characters(
    global_lore: str,
    new_characters: str,
) -> List[str]:
    names: Dict[str, str] = {}
    bounds = _find_lore_section(global_lore, CHARACTERS_SECTION)
    entries: List[Tuple[str, str]] = []
    if bounds:
        _, body_start, body_end = bounds
        entries.extend(_parse_bullet_entries(global_lore[body_start:body_end]))
    entries.extend(_parse_bullet_entries(new_characters))
    for raw_name, raw_value in entries:
        name = _canonical_display_name(raw_name)
        if (
            _is_invalid_context_key(name)
            or _is_non_character_work_entry(name, raw_value)
            or _is_non_character_group_entry(name, raw_value)
            or _is_non_character_metadata_or_item_entry(name, raw_value)
            or _is_descriptive_role_name(name)
            or _role_title_key_from_name(name)
            or _is_disposable_unnamed_character(name, raw_value)
            or _is_unstable_physical_character_entry(name, raw_value)
        ):
            continue
        names[_plain_key(name)] = name
    return list(names.values())


def infer_source_identity_links(
    source_text: str,
    current_global_lore: str,
    new_characters: str = "",
) -> str:
    """Extract conservative title-to-character links from direct source coreference."""
    text = _clean_inline_text(source_text)
    if not text:
        return ""
    candidates = _candidate_named_characters(current_global_lore, new_characters)
    if not candidates:
        return ""

    links: List[Tuple[str, str]] = []
    for role_key in sorted(_SOURCE_IDENTITY_ROLE_KEYS):
        role_pattern = re.escape(role_key).replace(r"\ ", r"\s+")
        matched_targets: set[str] = set()
        for name in candidates:
            name_pattern = _name_reference_pattern(name)
            patterns = (
                rf"\b(?:the\s+)?{role_pattern}'s\s+"
                r"(?:office|room|quarters|tent|desk|door|voice|expression|"
                r"face|hand|gaze|order)\b[\s\S]{0,220}"
                rf"{name_pattern}",
                rf"(?:[\"'“”‘’]\s*)?[.…\s]*{role_pattern}"
                rf"[.!?。…]*\s*(?:[\"'“”‘’])[\s\S]{{0,240}}"
                rf"{name_pattern}\s+"
                r"(?:was|were|is|are|said|asked|replied|answered|muttered|"
                r"whispered|looked|stared|gazed|frowned|sighed|smiled|"
                r"continued|spoke)\b",
            )
            if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
                matched_targets.add(name)
        if len(matched_targets) == 1:
            links.append((_display_role_title(role_key), next(iter(matched_targets))))

    return "\n".join(f"- {alias}: {target}" for alias, target in links)


def _source_identity_link_proof_status(
    source_text: str,
    current_global_lore: str,
    new_characters: str,
    alias: str,
    target: str,
) -> Tuple[bool, str]:
    """Return whether raw source text directly proves an alias mapping, with reason.

    A model-proposed identity link can merge two durable character entries, so
    source-analysis updates require direct evidence. Existing/manual context
    edits still enter through paths without source text and remain accepted.
    """
    text = _clean_inline_text(source_text)
    alias = _strip_balanced_brackets(alias)
    target = _canonical_display_name(target)
    if not text or _is_invalid_context_key(alias) or _is_invalid_context_key(target):
        return False, "missing source text or invalid alias/target"

    candidates = _candidate_named_characters(current_global_lore, new_characters)
    candidate_keys = {_plain_key(name): name for name in candidates}
    target_key = _plain_key(target)
    if target_key not in candidate_keys:
        return False, "target is not a current or newly proposed canonical character"

    inferred = {
        _plain_key(inferred_alias): _plain_key(inferred_target)
        for inferred_alias, inferred_target in _parse_bullet_entries(
            infer_source_identity_links(
                text,
                current_global_lore,
                new_characters,
            )
        )
    }
    if inferred.get(_plain_key(alias)) == target_key:
        return True, "source title/coreference backstop proved the link"

    alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
    target_pattern = _name_reference_pattern(candidate_keys[target_key])
    alias_is_named_character = _plain_key(alias) in candidate_keys
    strong_patterns = (
        rf"\b(?:the\s+)?{alias_pattern}\b[\s\S]{{0,80}}"
        rf"\b(?:is|was|becomes|became|named|called|known\s+as|"
        rf"identified\s+as|revealed\s+as|real\s+name\s+is|"
        rf"true\s+name\s+is)\s+{target_pattern}",
        rf"{target_pattern}[\s\S]{{0,80}}\b(?:is|was|serves\s+as|"
        rf"becomes|became|known\s+as|identified\s+as|revealed\s+as)\s+"
        rf"(?:the\s+)?{alias_pattern}\b",
        rf"{target_pattern}\s*,\s*(?:the\s+)?{alias_pattern}\b",
        rf"\b(?:the\s+)?{alias_pattern}\s*,\s*{target_pattern}",
        rf"\b(?:the\s+)?{alias_pattern}\s*\(\s*{target_pattern}\s*\)",
        rf"{target_pattern}\s*\(\s*(?:the\s+)?{alias_pattern}\s*\)",
        rf"\b(?:the\s+)?{alias_pattern}\b[\s\S]{{0,80}}"
        rf"\b(?:also\s+known\s+as|a\.?k\.?a\.?|aka)\s+{target_pattern}",
        rf"{target_pattern}[\s\S]{{0,80}}\b(?:also\s+known\s+as|"
        rf"a\.?k\.?a\.?|aka)\s+(?:the\s+)?{alias_pattern}\b",
        rf"\b(?:the\s+)?{alias_pattern}'s\s+(?:real\s+|true\s+)?name\s+"
        rf"(?:is|was)\s+{target_pattern}",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in strong_patterns):
        return True, "source contains an explicit identity statement or apposition"

    if alias_is_named_character:
        return False, "alias is already a named character and no explicit identity statement was found"

    if re.search(
        rf"\b(?:the\s+)?{alias_pattern}'s\s+"
        r"(?:office|room|quarters|tent|desk|door|voice|expression|"
        r"face|hand|gaze|order)\b[\s\S]{0,220}"
        rf"{target_pattern}",
        text,
        flags=re.IGNORECASE,
    ):
        return True, "source links the title/role to the target by local narration"

    target_words = candidate_keys[target_key].split()
    surname = target_words[-1] if target_words else ""
    if surname and re.search(
        rf"\b(?:the\s+)?{alias_pattern}\s+{re.escape(surname)}\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True, "source uses the alias directly with the target name"

    return False, "source chunk does not directly prove this alias-target mapping"


def _source_proves_identity_link(
    source_text: str,
    current_global_lore: str,
    new_characters: str,
    alias: str,
    target: str,
) -> bool:
    """Return whether raw source text directly proves an alias mapping."""
    proved, _ = _source_identity_link_proof_status(
        source_text,
        current_global_lore,
        new_characters,
        alias,
        target,
    )
    return proved


def _source_reincarnation_gender_for_name(source_text: str, name: str) -> str:
    text = _clean_inline_text(source_text)
    if not text or _is_invalid_context_key(name):
        return ""
    name_pattern = _name_reference_pattern(name)
    body_pattern = r"(?P<body>male|female|man|woman|boy|girl)"
    patterns = (
        rf"{name_pattern}[\s\S]{{0,160}}\breincarnat\w+\b"
        rf"[\s\S]{{0,600}}\b(?:became|become|becomes|becoming|"
        rf"woke\s+up\s+as|awoke\s+as|reincarnated\s+as|reborn\s+as)\s+"
        rf"(?:an?\s+)?(?:[\w'-]+\s+){{0,5}}{body_pattern}\b",
        rf"{name_pattern}[\s\S]{{0,160}}\breincarnat\w+\b"
        rf"[\s\S]{{0,600}}\b(?:very\s+)?(?:cute\s+|small\s+|ragged\s+|"
        rf"young\s+|old\s+){{0,5}}{body_pattern}\b"
        rf"[\s\S]{{0,160}}\b(?:is\s+this\s+me|this\s+is\s+me|my\s+body)\b",
    )
    genders = {
        _gender_from_body_word(match.group("body"))
        for pattern in patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    }
    genders.discard("")
    return next(iter(genders)) if len(genders) == 1 else ""


def _source_direct_gender_for_name(source_text: str, name: str) -> str:
    """Infer a character gender from raw source pronoun evidence."""
    return _infer_gender_reference_to_character(source_text, name)


def _source_relationship_pronoun_gender_for_name(
    source_text: str,
    name: str,
) -> str:
    """Infer gender from source relationship clauses with direct pronouns.

    Relationship labels alone are not gender evidence. This only fires when a
    named person's romantic/partner relation is paired with a pronoun such as
    "him" or "her" in the same local clause.
    """
    text = _clean_inline_text(source_text)
    if not text or _is_invalid_context_key(name):
        return ""
    canonical = _canonical_display_name(name)
    name_pattern = (
        rf"(?<![\w'-]){re.escape(canonical)}(?:['’]s)?(?![\w'-])"
        if canonical
        else ""
    )
    genders: set[str] = set()
    pattern = (
        rf"{name_pattern}[\s\S]{{0,100}}\b"
        rf"{_ROMANTIC_RELATION_PATTERN}\b[\s\S]{{0,160}}\b"
        rf"(?:{_RELATIONSHIP_OBJECT_PRONOUN_VERBS})\s+"
        rf"(?P<pronoun>him|her)\b"
    )
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        pronoun = match.group("pronoun").casefold()
        if pronoun == "him":
            genders.add("Male")
        elif pronoun == "her":
            genders.add("Female")
    return next(iter(genders)) if len(genders) == 1 else ""


def _source_romantic_role_gender_for_name(source_text: str, name: str) -> str:
    """Infer gender for role labels such as Ex-lover from local pronouns."""
    text = _clean_inline_text(source_text)
    key = _plain_key(name)
    if not text or key not in {
        "ex",
        "ex girlfriend",
        "ex lover",
        "ex partner",
        "ex-girlfriend",
        "ex-lover",
        "ex-partner",
        "former girlfriend",
        "former lover",
        "former partner",
        "girlfriend",
        "lover",
        "partner",
    }:
        return ""

    role_pattern = (
        r"(?:ex[-\s]?)?(?:girl\s*friend|boy\s*friend|lover|partner)|"
        r"former\s+(?:girl\s*friend|boy\s*friend|lover|partner)"
    )
    genders: set[str] = set()
    for match in re.finditer(role_pattern, text, flags=re.IGNORECASE):
        window = text[match.start(): match.end() + 220]
        if re.search(r"\b(?:she|her|hers)\b", window, flags=re.IGNORECASE):
            genders.add("Female")
        if re.search(r"\b(?:he|him|his)\b", window, flags=re.IGNORECASE):
            genders.add("Male")
    return next(iter(genders)) if len(genders) == 1 else ""


def infer_source_gender_updates(
    source_text: str,
    current_global_lore: str,
    new_characters: str = "",
) -> str:
    """Extract conservative source-proven gender corrections from raw chunks.

    This complements the LLM response for every file type because all
    pipelines pass plain source text into the shared context updater.
    """
    updates: List[str] = []
    for name in _candidate_named_characters(current_global_lore, new_characters):
        gender = _source_reincarnation_gender_for_name(source_text, name)
        if gender:
            updates.append(f"- {name}: CORRECTION: [{gender}]")
            continue
        gender = _source_relationship_pronoun_gender_for_name(
            source_text,
            name,
        )
        if gender:
            updates.append(f"- {name}: CORRECTION: [{gender}]")
            continue
        gender = _source_romantic_role_gender_for_name(source_text, name)
        if gender:
            updates.append(f"- {name}: CORRECTION: [{gender}]")
            continue
        gender = _source_direct_gender_for_name(source_text, name)
        if gender:
            updates.append(f"- {name}: CORRECTION: [{gender}]")
    return "\n".join(updates)


def _source_proven_gender_for_name(source_text: str, name: str) -> str:
    """Return only deterministic source-backed gender evidence for a name."""
    for detector in (
        _source_reincarnation_gender_for_name,
        _source_relationship_pronoun_gender_for_name,
        _source_romantic_role_gender_for_name,
        _source_direct_gender_for_name,
    ):
        gender = detector(source_text, name)
        if gender:
            return gender
    return ""


def _incoming_detail_gender_for_name(name: str, details: str) -> str:
    """Return self-contained gender evidence already present in character facts."""
    return (
        _current_reincarnated_form_gender(name, details)
        or _infer_gender_from_character_details(details)
    )


def _is_risky_unproven_new_gender_guess(
    name: str,
    details: str,
    source_text: str,
) -> bool:
    """Detect early protagonist guesses that commonly borrow nearby pronouns."""
    if not source_text:
        return False
    key = _plain_key(details)
    if not any(
        marker in key
        for marker in (
            "protagonist",
            "main character",
            "patient",
            "suffering",
            "illness",
            "disease",
            "anemia",
        )
    ):
        return False
    source_key = _plain_key(source_text)
    if _plain_key(name) not in source_key:
        return False
    return any(
        marker in source_key
        for marker in (
            "regular health checkup",
            "blood-related",
            "blood related",
            "anemia",
            "doctor told me",
            "prepare yourself",
            "ex-lover",
            "lover i trusted",
        )
    )


def _gate_unproven_character_gender(
    name: str,
    value: str,
    source_text: str,
    existing_value: str = "",
    explicit_correction: bool = False,
) -> str:
    """Prevent model-guessed genders from entering durable lore.

    Source-analysis prompts already tell the LLM not to guess, but small models
    still borrow nearby pronouns from another character. When raw source text is
    available, a new specific gender must be backed by deterministic evidence.
    Existing specific genders remain authoritative unless the source proves a
    correction; this gate only rejects an incoming unsupported claim.
    """
    try:
        from src import config as _config
        bypass = getattr(_config, "BYPASS_CONTEXT_GATING", True)
    except Exception:
        bypass = True

    if bypass:
        return value

    if not source_text:
        return value

    incoming_gender, incoming_details = _split_gender_and_details(
        _normalize_character_value(value)
    )
    incoming_gender = _canonical_gender(incoming_gender)
    if incoming_gender.casefold() not in _SPECIFIC_GENDER_LABELS:
        return value

    existing_gender, _ = _split_gender_and_details(
        _normalize_character_value(existing_value)
    )
    existing_gender = _canonical_gender(existing_gender)

    source_gender = _source_proven_gender_for_name(source_text, name)
    if source_gender.casefold() == incoming_gender.casefold():
        return value
    if source_gender:
        return (
            f"{source_gender}, {incoming_details}".rstrip(" ,")
            if incoming_details
            else source_gender
        )

    detail_gender = _incoming_detail_gender_for_name(name, incoming_details)
    if detail_gender.casefold() == incoming_gender.casefold():
        return value
    if detail_gender:
        return (
            f"{detail_gender}, {incoming_details}".rstrip(" ,")
            if incoming_details
            else detail_gender
        )

    if existing_gender.casefold() in _SPECIFIC_GENDER_LABELS:
        if explicit_correction:
            return (
                f"{existing_gender}, {incoming_details}".rstrip(" ,")
                if incoming_details
                else existing_gender
            )
        return value
    if not explicit_correction and not _is_risky_unproven_new_gender_guess(
        name,
        incoming_details,
        source_text,
    ):
        return value
    return (
        f"Unspecified, {incoming_details}".rstrip(" ,")
        if incoming_details
        else "Unspecified"
    )


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


def _has_cjk_or_hangul_name_script(value: str) -> bool:
    compact = re.sub(r"\s+", "", _strip_balanced_brackets(value))
    return bool(
        re.search(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7a3]", compact)
    )


def _is_source_name_label(value: str) -> bool:
    compact = re.sub(r"\s+", "", _strip_balanced_brackets(value))
    if not compact or not _has_cjk_or_hangul_name_script(compact):
        return False
    if _is_cjk_generic_role_only_name(compact) or compact in _CJK_NON_NAME_ADDRESS_LABELS:
        return False
    return True


def _is_likely_translated_name(value: str) -> bool:
    clean = _strip_balanced_brackets(value)
    if _is_invalid_context_key(clean) or _has_cjk_or_hangul_name_script(clean):
        return False
    if not re.search(r"[A-Za-z]", clean):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", clean)
    if not words:
        return False
    lowercase_words = {
        "a",
        "an",
        "and",
        "as",
        "for",
        "from",
        "in",
        "of",
        "or",
        "the",
        "to",
        "with",
    }
    return any(
        word[:1].isupper()
        or "-" in word
        or word.casefold() not in lowercase_words and len(words) <= 3
        for word in words
    )


def _build_name_translation_map_lines(
    characters: List[Tuple[str, str]],
    aliases: List[Tuple[str, str]],
    glossary: List[Tuple[str, str]],
) -> List[str]:
    """Render an audit-only source-name-to-translation table."""
    character_keys = {
        alias_key
        for name, _ in characters
        for alias_key in _character_alias_keys(name)
    }
    alias_target_by_key = {
        alias_key: target
        for alias, target in aliases
        for alias_key in _character_alias_keys(alias)
    }
    known_name_keys = set(character_keys) | set(alias_target_by_key)

    glossary_by_key = {
        alias_key: value
        for source, value in glossary
        if _is_likely_translated_name(value)
        for alias_key in _character_alias_keys(source)
    }

    ordered: Dict[str, Tuple[str, str]] = {}

    def add(
        source: str,
        fallback_target: str = "",
        *,
        allow_unset: bool = False,
    ) -> None:
        if not _is_source_name_label(source):
            return
        source_keys = _character_alias_keys(source)
        if not source_keys:
            return
        target = next(
            (glossary_by_key[key] for key in source_keys if key in glossary_by_key),
            fallback_target,
        )
        if not _is_likely_translated_name(target):
            if not allow_unset:
                return
            target = _UNSET_NAME_TRANSLATION
        if target == _UNSET_NAME_TRANSLATION and not allow_unset:
            return
        key = _plain_key(source)
        ordered[key] = (_strip_balanced_brackets(source), _strip_balanced_brackets(target))

    for source, _ in characters:
        source_keys = _character_alias_keys(source)
        if source_keys & known_name_keys:
            add(
                source,
                alias_target_by_key.get(next(iter(source_keys), ""), ""),
                allow_unset=True,
            )
    for source, target in aliases:
        add(source, target, allow_unset=not _is_likely_translated_name(target))
    for source, target in glossary:
        source_keys = _character_alias_keys(source)
        if source_keys & known_name_keys:
            add(source, target)

    return [f"- {source}: {target}" for source, target in ordered.values()]


def _character_alias_keys_from_lore(global_lore: str) -> set[str]:
    bounds = _find_lore_section(global_lore, CHARACTERS_SECTION)
    if not bounds:
        return set()
    _, body_start, body_end = bounds
    keys: set[str] = set()
    for name, value in _parse_bullet_entries(global_lore[body_start:body_end]):
        keys.update(_character_alias_keys(name))
        keys.update(_character_self_role_title_keys(name, value))
        keys.update(_character_narrative_role_alias_keys(name, value))
    alias_bounds = _find_lore_section(global_lore, ALIASES_SECTION)
    if alias_bounds:
        for alias, _ in _parse_alias_entries(
            global_lore[alias_bounds[1]:alias_bounds[2]]
        ):
            keys.update(_character_alias_keys(alias))
    return {key for key in keys if key}


def _discarded_incidental_character_aliases(
    original_global_lore: str,
    normalized_global_lore: str,
) -> set[str]:
    """Return aliases for incidental entries removed during normalization."""
    original_bounds = _find_lore_section(original_global_lore, CHARACTERS_SECTION)
    if not original_bounds:
        return set()
    retained_keys = _character_alias_keys_from_lore(normalized_global_lore)
    _, body_start, body_end = original_bounds
    discarded: set[str] = set()
    for raw_name, raw_value in _parse_bullet_entries(
        original_global_lore[body_start:body_end]
    ):
        aliases = _character_alias_keys(raw_name)
        if aliases & retained_keys:
            continue
        if (
            _is_non_character_work_entry(raw_name, raw_value)
            or _is_non_character_group_entry(raw_name, raw_value)
            or _is_non_character_metadata_or_item_entry(raw_name, raw_value)
            or _is_disposable_unnamed_character(raw_name, raw_value)
            or _is_unstable_physical_character_entry(raw_name, raw_value)
        ):
            discarded.update(aliases)
    return discarded


def normalize_global_lore(global_lore: str) -> str:
    """Remove template pollution and merge deterministic character aliases."""
    lore = str(global_lore or "").strip()
    if not lore:
        return ""

    alias_bounds = _find_lore_section(lore, ALIASES_SECTION)
    alias_entries = (
        _parse_alias_entries(lore[alias_bounds[1]:alias_bounds[2]])
        if alias_bounds
        else []
    )
    explicit_aliases = _alias_entries_to_map(alias_entries)
    alias_displays = {
        alias_key: alias
        for alias, _ in alias_entries
        for alias_key in _character_alias_keys(alias)
    }

    character_bounds = _find_lore_section(lore, CHARACTERS_SECTION)
    characters: List[Tuple[str, str]] = []
    if character_bounds:
        _, body_start, body_end = character_bounds
        raw_character_entries = _parse_bullet_entries(lore[body_start:body_end])
        inferred_aliases, inferred_displays = _infer_unique_short_name_alias_entries(
            raw_character_entries,
            explicit_aliases,
        )
        plural_aliases, plural_displays = _infer_singular_plural_alias_entries(
            raw_character_entries,
            explicit_aliases,
        )
        for alias_key, target in inferred_aliases.items():
            explicit_aliases.setdefault(alias_key, target)
        for alias_key, display in inferred_displays.items():
            alias_displays.setdefault(alias_key, display)
        for alias_key, target in plural_aliases.items():
            explicit_aliases.setdefault(alias_key, target)
        for alias_key, display in plural_displays.items():
            alias_displays.setdefault(alias_key, display)
        characters, deduced_aliases = _deduplicate_character_entries(
            raw_character_entries,
            explicit_aliases,
        )
        _retain_renderable_aliases(
            explicit_aliases,
            alias_displays,
            deduced_aliases,
            raw_character_entries,
        )
        lore = _replace_lore_section(
            lore,
            CHARACTERS_SECTION,
            [_format_character_line(name, value) for name, value in characters],
        )

    if alias_bounds or explicit_aliases:
        aliases = _canonical_alias_entries(
            explicit_aliases,
            characters,
            alias_displays,
        )
        lore = _replace_lore_section(
            lore,
            ALIASES_SECTION,
            [f"- {alias}: {target}" for alias, target in aliases],
        )
    else:
        aliases = []

    glossary_bounds = _find_lore_section(lore, GLOSSARY_SECTION)
    glossary: List[Tuple[str, str]] = []
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
    if characters or aliases or glossary or _find_lore_section(lore, NAME_MAP_SECTION):
        lore = _replace_lore_section(
            lore,
            NAME_MAP_SECTION,
            _build_name_translation_map_lines(characters, aliases, glossary),
        )
    return lore.strip()


def character_alias_map(global_lore: str) -> Dict[str, str]:
    """Return every deterministic and explicit alias for canonical characters."""
    bounds = _find_lore_section(global_lore, CHARACTERS_SECTION)
    if not bounds:
        return {}
    alias_bounds = _find_lore_section(global_lore, ALIASES_SECTION)
    explicit_aliases = _alias_entries_to_map(
        _parse_alias_entries(
            global_lore[alias_bounds[1]:alias_bounds[2]]
        )
        if alias_bounds
        else []
    )
    _, body_start, body_end = bounds
    _, aliases = _deduplicate_character_entries(
        _parse_bullet_entries(global_lore[body_start:body_end]),
        explicit_aliases,
    )
    retained = {}
    for alias, target in aliases.items():
        if _is_quarantined_character_entry(target):
            continue
        retained[alias] = target
    return retained


_character_alias_map = character_alias_map


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


def _within_one_edit(first: str, second: str) -> bool:
    if first == second:
        return True
    if abs(len(first) - len(second)) > 1:
        return False
    if len(first) == len(second):
        return sum(left != right for left, right in zip(first, second)) == 1
    if len(first) > len(second):
        first, second = second, first
    index_first = 0
    index_second = 0
    edits = 0
    while index_first < len(first) and index_second < len(second):
        if first[index_first] == second[index_second]:
            index_first += 1
            index_second += 1
            continue
        edits += 1
        if edits > 1:
            return False
        index_second += 1
    return True


def _fuzzy_canonical_relationship_party(
    value: str,
    alias_map: Dict[str, str],
) -> str:
    compact = _compact_name_key(value)
    if not compact:
        return ""
    matches = {
        target
        for alias, target in alias_map.items()
        if (
            " " not in alias
            and len(alias) >= 12
            and _within_one_edit(compact, alias)
        )
    }
    return next(iter(matches)) if len(matches) == 1 else ""


def _canonical_relationship_party(value: str, alias_map: Dict[str, str]) -> str:
    clean = _strip_balanced_brackets(value).strip()
    if re.search(r"\s&\s", clean):
        parts = re.split(r"\s*&\s*", clean)
        canonical_parts = [
            _canonical_relationship_party(part, alias_map)
            for part in parts
        ]
        if any(
            _plain_key(part) != _plain_key(canonical)
            for part, canonical in zip(parts, canonical_parts)
        ):
            return " & ".join(canonical_parts)
    for alias in _character_alias_keys(clean):
        if alias in alias_map:
            return alias_map[alias]
    fuzzy = _fuzzy_canonical_relationship_party(clean, alias_map)
    if fuzzy:
        return fuzzy
    return clean


def _is_disposable_dynamic_party(value: str, alias_map: Dict[str, str]) -> bool:
    """Drop dynamic rows that point only at filtered generic background roles."""
    clean = _strip_balanced_brackets(value).strip()
    if not clean:
        return False
    if _is_invalid_context_key(clean):
        return True
    for alias in _character_alias_keys(clean):
        if alias in alias_map:
            return False
    key = _plain_key(clean)
    return bool(_is_numbered_generic_role_name(clean) or any(
        marker in key for marker in _INCIDENTAL_CHARACTER_MARKERS
    ) or _is_unstable_identity_alias(clean)
        or _is_transferable_role_only_name(clean)
        or _is_descriptive_role_name(clean))


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
    if key_left == key_right:
        return None
    if arrow == "↔" and key_left > key_right:
        key_left, key_right = key_right, key_left
    relation_key = (key_left, arrow, key_right)
    rendered = f"- {left} {arrow} {right}: {details}".rstrip()
    return relation_key, rendered, details


def _dynamic_relation_has_disposable_party(
    line: str,
    alias_map: Dict[str, str],
    discarded_aliases: Optional[set[str]] = None,
) -> bool:
    match = _DYNAMIC_RELATION_PATTERN.match(line)
    if not match:
        return False
    left = _canonical_relationship_party(match.group("left"), alias_map)
    right = _canonical_relationship_party(match.group("right"), alias_map)
    if _plain_key(left) == _plain_key(right):
        return True
    discarded_aliases = discarded_aliases or set()
    left_aliases = _character_alias_keys(left)
    right_aliases = _character_alias_keys(right)
    if (left_aliases | right_aliases) & discarded_aliases:
        return True
    return (
        _is_disposable_dynamic_party(left, alias_map)
        or _is_disposable_dynamic_party(right, alias_map)
    )


def _normalize_dynamic_entries(
    text: str,
    alias_map: Dict[str, str],
    discarded_aliases: Optional[set[str]] = None,
) -> str:
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

        if _dynamic_relation_has_disposable_party(
            line,
            alias_map,
            discarded_aliases,
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


def _source_address_label_from_details(details: str) -> str:
    clean = _clean_inline_text(details)
    if not clean:
        return ""
    match = re.match(r"^[\"'“”‘’](?P<label>[^\"'“”‘’]{1,16})[\"'“”‘’]", clean)
    if match:
        return _strip_balanced_brackets(match.group("label")).strip()
    first_part = clean.split("|", 1)[0].strip()
    first_part = re.sub(r"\s*\([^)]*\)\s*$", "", first_part).strip()
    return _strip_balanced_brackets(first_part).strip("\"'“”‘’ ")


def _is_source_address_identity_label(label: str) -> bool:
    compact = re.sub(r"\s+", "", _strip_balanced_brackets(label))
    return bool(
        re.fullmatch(r"[\u3400-\u9fff\uf900-\ufaff]{1,8}", compact)
        or re.fullmatch(r"[\uac00-\ud7a3]{1,6}", compact)
    )


def infer_dynamic_address_identity_links(
    dynamic_state: str,
    global_lore: str,
) -> str:
    """Promote stable source-side addressing labels to identity links."""
    if not dynamic_state or not global_lore:
        return ""
    alias_map = character_alias_map(global_lore)
    candidates = {
        _plain_key(name): name
        for name in _candidate_named_characters(global_lore, "")
    }
    addressing, _, _ = _split_dynamic_sections(dynamic_state)
    links: Dict[str, str] = {}
    for line in addressing.splitlines():
        parsed = _parse_dynamic_relation(line, alias_map)
        if not parsed:
            continue
        (_, arrow, target_key), _, details = parsed
        if arrow != "→" or target_key not in candidates:
            continue
        label = _source_address_label_from_details(details)
        if not _is_source_address_identity_label(label):
            continue
        target = candidates[target_key]
        if any(
            alias_key in _character_alias_keys(target)
            for alias_key in _character_alias_keys(label)
        ):
            continue
        links[label] = target
    return "\n".join(f"- {alias}: {target}" for alias, target in links.items())


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
    discarded_character_aliases: Optional[set[str]] = None,
) -> str:
    """Normalize dynamic context into stable addressing and relationship sections."""
    alias_map = character_aliases or {}
    discarded_aliases = discarded_character_aliases or set()
    addressing, relationships, _ = _split_dynamic_sections(dynamic_state)
    return _format_dynamic_sections(
        _normalize_dynamic_entries(addressing, alias_map, discarded_aliases),
        _normalize_dynamic_entries(relationships, alias_map, discarded_aliases),
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
        original_global_lore = extract_global_lore(text)
        global_lore = normalize_global_lore(original_global_lore)
        dynamic_state = extract_dynamic_state_from_text(text) or ""
        discarded_aliases = _discarded_incidental_character_aliases(
            original_global_lore,
            global_lore,
        )
        return build_novel_context(global_lore, dynamic_state, discarded_aliases)
    return normalize_global_lore(text)


def _current_form_fact_from_details(source_name: str, details: str) -> str:
    match = re.search(
        r"\breincarnat\w+\s+as\s+(?P<form>[^.;]+)",
        details,
        flags=re.IGNORECASE,
    )
    form = _clean_inline_text(match.group("form")).strip(" ,") if match else ""
    fact = f"reincarnated form of {_canonical_display_name(source_name)}"
    return f"{fact} as {form}" if form else fact


def _dynamic_links_reincarnated_form(
    dynamic_state: str,
    source_name: str,
    target_name: str,
) -> bool:
    source_pattern = _name_reference_pattern(source_name)
    target_pattern = _name_reference_pattern(target_name)
    for raw_line in _normalize_relationship_notation(dynamic_state).splitlines():
        line = _clean_inline_text(raw_line)
        if not line or "reincarnat" not in line.casefold():
            continue
        if re.search(
            rf"{source_pattern}[\s\S]{{0,180}}\breincarnat\w+"
            rf"[\s\S]{{0,160}}\b(?:into|as)\b[\s\S]{{0,120}}"
            rf"{target_pattern}",
            line,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _apply_dynamic_reincarnation_gender_links(
    global_lore: str,
    dynamic_state: str,
) -> str:
    bounds = _find_lore_section(global_lore, CHARACTERS_SECTION)
    if not bounds or not str(dynamic_state or "").strip():
        return global_lore

    _, body_start, body_end = bounds
    characters = _parse_bullet_entries(global_lore[body_start:body_end])
    if len(characters) < 2:
        return global_lore

    updated = list(characters)
    changed = False
    for source_name, source_value in characters:
        if _is_quarantined_character_entry(source_name, source_value):
            continue
        _, source_details = _split_gender_and_details(
            _normalize_character_value_for_name(source_name, source_value)
        )
        current_form_gender = _current_reincarnated_form_gender(
            source_name,
            source_details,
        )
        if current_form_gender.casefold() not in _SPECIFIC_GENDER_LABELS:
            continue

        for target_index, (target_name, target_value) in enumerate(updated):
            if _character_names_match(source_name, target_name):
                continue
            if _is_quarantined_character_entry(target_name, target_value):
                continue
            if not _dynamic_links_reincarnated_form(
                dynamic_state,
                source_name,
                target_name,
            ):
                continue
            repaired_value = _merge_character_values(
                target_value,
                f"{current_form_gender}, "
                f"{_current_form_fact_from_details(source_name, source_details)}",
                allow_gender_correction=True,
            )
            if repaired_value != target_value:
                updated[target_index] = (target_name, repaired_value)
                changed = True

    if not changed:
        return global_lore
    return _replace_lore_section(
        global_lore,
        CHARACTERS_SECTION,
        [_format_character_line(name, value) for name, value in updated],
    )


def is_safe_filename(filename: str) -> bool:
    """Return whether a context filename is safe while preserving Unicode names."""
    if not filename or filename != filename.strip():
        return False
    if not filename.lower().endswith(".txt"):
        return False
    if any(
        ord(char) < 32 or not (char.isalnum() or char in SAFE_FILENAME_PUNCTUATION)
        for char in filename
    ):
        return False
    if filename in {".", ".."}:
        return False
    stem = filename[:-4].rstrip(". ")
    if not stem or stem in {".", ".."}:
        return False
    if stem.split(".", 1)[0].lower() in WINDOWS_RESERVED_FILENAMES:
        return False
    return True


def _safe_context_filename_stem(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value or "")
    safe_chars = [
        char if char.isalnum() or char in SAFE_FILENAME_PUNCTUATION else "_"
        for char in normalized
    ]
    return "".join(safe_chars).strip(".")


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
            f"Invalid filename '{filename}'. Allowed: Unicode letters/numbers, `_`, `-`, `.`; extension must be .txt."
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
                f"{ALIASES_SECTION}\n\n"
                f"{NAME_MAP_SECTION}\n\n"
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
            f"Invalid filename '{filename}'. Allowed: Unicode letters/numbers, `_`, `-`, `.`; extension must be .txt."
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


def build_novel_context(
    global_lore: str,
    dynamic_state: str,
    discarded_character_aliases: Optional[set[str]] = None,
) -> str:
    """Build the canonical full context representation used by every pipeline."""
    normalized_global = normalize_global_lore(global_lore)
    normalized_global = _apply_dynamic_reincarnation_gender_links(
        normalized_global,
        dynamic_state,
    )
    normalized_dynamic = normalize_dynamic_state(
        dynamic_state,
        _character_alias_map(normalized_global),
        discarded_character_aliases,
    )
    return (
        f"{normalized_global.strip()}\n\n"
        f"{DYNAMIC_STATE_START}\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        f"{normalized_dynamic.strip()}\n"
        f"{DYNAMIC_STATE_END}"
    ).strip()


def _source_memory_budget_chars() -> int:
    try:
        from src import config as _config
        raw_value = getattr(_config, "NOVEL_CONTEXT_SOURCE_MEMORY_CHARS", 6000)
        return max(0, int(raw_value))
    except Exception:
        return 6000


def _clean_source_memory_chunk(source_chunk: str) -> str:
    return _clean_inline_text(source_chunk).strip()


def _bounded_source_memory(chunks: List[str], max_chars: Optional[int] = None) -> str:
    budget = _source_memory_budget_chars() if max_chars is None else int(max_chars)
    if budget <= 0:
        return ""

    separator = "\n\n--- Previous source chunk ---\n\n"
    selected: List[str] = []
    total = 0
    for chunk in reversed(chunks):
        clean = _clean_source_memory_chunk(chunk)
        if not clean:
            continue
        extra = len(clean) + (len(separator) if selected else 0)
        if selected and total + extra > budget:
            break
        if not selected and len(clean) > budget:
            clean = clean[-budget:].lstrip()
            extra = len(clean)
        selected.append(clean)
        total += extra
    return separator.join(reversed(selected))


def _compose_source_analysis_text(source_context: str, source_chunk: str) -> str:
    parts = [
        _clean_source_memory_chunk(source_context),
        _clean_source_memory_chunk(source_chunk),
    ]
    return "\n\n".join(part for part in parts if part)


def _context_prompt_budget_chars(max_tokens: Optional[int]) -> int:
    if max_tokens is None:
        try:
            from src import config as _config
            max_tokens = int(
                getattr(_config, "NOVEL_CONTEXT_PROMPT_MAX_TOKENS", 1800)
            )
        except Exception:
            max_tokens = 1800
    try:
        token_budget = int(max_tokens)
    except (TypeError, ValueError):
        token_budget = 1800
    if token_budget <= 0:
        return 0
    # Conservative tokenizer-free estimate. This renderer runs inside prompt
    # construction, so avoid importing heavier tokenizers or model-specific
    # encoders here. The durable context file remains complete.
    return max(1000, token_budget * 4)


def _section_body(text: str, section_name: str) -> str:
    bounds = _find_lore_section(text, section_name)
    if not bounds:
        return ""
    _, body_start, body_end = bounds
    return text[body_start:body_end].strip()


def _text_mentions(value: str, reference_text: str) -> bool:
    value = _strip_balanced_brackets(value)
    if _is_invalid_context_key(value):
        return False
    folded_reference = str(reference_text or "").casefold()
    if not folded_reference.strip():
        return False
    folded_value = value.casefold()
    return bool(folded_value and folded_value in folded_reference)


def _entry_mentions_reference(
    name: str,
    value: str,
    reference_text: str,
) -> bool:
    return _text_mentions(name, reference_text) or _text_mentions(
        value,
        reference_text,
    )


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


def _split_selected_entries(
    entries: List[Tuple[str, str]],
    reference_text: str,
    selected_names: Optional[set[str]] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    selected: List[Tuple[str, str]] = []
    remaining: List[Tuple[str, str]] = []
    selected_keys = selected_names or set()
    for name, value in entries:
        name_key = _plain_key(name)
        if (
            name_key in selected_keys
            or _entry_mentions_reference(name, value, reference_text)
            or _reference_mentions_latin_name_part(name, reference_text)
        ):
            selected.append((name, value))
        else:
            remaining.append((name, value))
    return selected, remaining


def _append_line_with_budget(
    lines: List[str],
    line: str,
    max_chars: int,
    reserved_chars: int,
) -> bool:
    candidate = lines + [line]
    if len("\n".join(candidate)) + reserved_chars <= max_chars:
        lines.append(line)
        return True
    return False


def _append_section_with_budget(
    lines: List[str],
    section_name: str,
    entries: List[str],
    max_chars: int,
    reserved_chars: int,
) -> None:
    if not entries:
        return
    if not _append_line_with_budget(lines, "", max_chars, reserved_chars):
        return
    if not _append_line_with_budget(lines, section_name, max_chars, reserved_chars):
        return
    for entry in entries:
        _append_line_with_budget(lines, entry, max_chars, reserved_chars)


def _compact_character_gender_line(name: str, value: str) -> str:
    gender, _ = _split_gender_and_details(value)
    gender = _canonical_gender(gender)
    if gender.casefold() not in _SPECIFIC_GENDER_LABELS:
        return ""
    return _format_character_line(name, gender)


def render_novel_context_for_prompt(
    context_content: str,
    reference_text: str = "",
    max_tokens: Optional[int] = None,
    selective: bool = True,
    include_gender_roster: bool = True,
) -> str:
    """Render a prompt-sized view of a durable novel context document.

    The context file remains the complete source of truth. This function only
    selects the text injected into the user prompt. By default, it injects only
    entries mentioned by the current chunk or draft. Callers may set
    ``selective=False`` to preserve the legacy budget-only behavior.
    """
    normalized = normalize_novel_context_content(context_content)
    if not normalized:
        return ""

    max_chars = _context_prompt_budget_chars(max_tokens)
    if not selective:
        if not max_chars or len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip()
    if not max_chars:
        return normalized

    global_lore = extract_global_lore(normalized)
    dynamic_state = extract_dynamic_state_from_text(normalized) or ""

    character_entries = _parse_bullet_entries(
        _section_body(global_lore, CHARACTERS_SECTION)
    )
    character_entries = [
        (name, value)
        for name, value in character_entries
        if not _is_quarantined_character_entry(name, value)
    ]
    alias_entries = _parse_alias_entries(
        _section_body(global_lore, ALIASES_SECTION)
    )
    alias_entries = [
        (alias, target)
        for alias, target in alias_entries
        if not _is_quarantined_character_entry(target)
    ]
    glossary_entries = _parse_bullet_entries(
        _section_body(global_lore, GLOSSARY_SECTION)
    )
    if not any((
        character_entries,
        alias_entries,
        glossary_entries,
        dynamic_state.strip(),
    )):
        return normalized if len(normalized) <= max_chars else normalized[:max_chars].rstrip()

    selected_character_keys: set[str] = set()
    for name, value in character_entries:
        if (
            _entry_mentions_reference(name, value, reference_text)
            or _reference_mentions_latin_name_part(name, reference_text)
        ):
            selected_character_keys.add(_plain_key(name))

    for alias, target in alias_entries:
        if (
            _entry_mentions_reference(alias, target, reference_text)
            or _reference_mentions_latin_name_part(alias, reference_text)
        ):
            selected_character_keys.add(_plain_key(target))

    selected_characters, remaining_characters = _split_selected_entries(
        character_entries,
        reference_text,
        selected_character_keys,
    )
    selected_aliases, remaining_aliases = _split_selected_entries(
        alias_entries,
        reference_text,
        selected_character_keys,
    )
    selected_glossary, remaining_glossary = _split_selected_entries(
        glossary_entries,
        reference_text,
    )

    addressing, relationships, _ = _split_dynamic_sections(dynamic_state)
    dynamic_reference_names = {
        name for name, _ in selected_characters
    } | {
        alias for alias, _ in selected_aliases
    } | {
        target for _, target in selected_aliases
    }
    dynamic_reference_keys = {
        _plain_key(name)
        for name in dynamic_reference_names
    }

    def dynamic_party_is_referenced(party: str) -> bool:
        party = party.strip()
        return (
            _plain_key(party) in dynamic_reference_keys
            or _text_mentions(party, reference_text)
            or _reference_mentions_latin_name_part(party, reference_text)
        )

    def split_dynamic_lines(text: str) -> Tuple[List[str], List[str]]:
        selected_lines: List[str] = []
        remaining_lines: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            relation = _DYNAMIC_RELATION_PATTERN.match(line)
            if relation:
                parties = (relation.group("left"), relation.group("right"))
                should_select = sum(
                    1 for party in parties
                    if dynamic_party_is_referenced(party)
                ) >= 2
            else:
                should_select = any(
                    _text_mentions(token, line)
                    for token in dynamic_reference_names
                ) or _entry_mentions_reference(line, "", reference_text)
            if should_select:
                selected_lines.append(line)
            else:
                remaining_lines.append(line)
        return selected_lines, remaining_lines

    selected_addressing, remaining_addressing = split_dynamic_lines(addressing)
    selected_relationships, remaining_relationships = split_dynamic_lines(
        relationships
    )

    selected_character_lines = [
        _format_character_line(name, value)
        for name, value in selected_characters
    ]
    selected_character_keys = {
        _plain_key(name)
        for name, _ in selected_characters
    }
    pinned_gender_lines: List[str] = []
    if include_gender_roster:
        pinned_gender_lines = [
            line
            for name, value in character_entries
            if _plain_key(name) not in selected_character_keys
            for line in [_compact_character_gender_line(name, value)]
            if line
        ]
    remaining_character_lines = [
        _format_character_line(name, value)
        for name, value in remaining_characters
    ]
    selected_alias_lines = [
        f"- {alias}: {target}"
        for alias, target in selected_aliases
    ]
    remaining_alias_lines = [
        f"- {alias}: {target}"
        for alias, target in remaining_aliases
    ]
    selected_glossary_lines = [
        f"- {name}: {value}"
        for name, value in selected_glossary
    ]
    remaining_glossary_lines = [
        f"- {name}: {value}"
        for name, value in remaining_glossary
    ]

    has_selection = any((
        selected_character_lines,
        pinned_gender_lines,
        selected_alias_lines,
        selected_glossary_lines,
        selected_addressing,
        selected_relationships,
    ))
    if not has_selection:
        return ""

    reserved = len(
        f"\n\n{DYNAMIC_STATE_START}\n# DYNAMIC RELATIONSHIP STATE\n"
        f"{DYNAMIC_STATE_END}"
    )
    rendered_lines: List[str] = ["# GLOBAL LORE"]
    _append_section_with_budget(
        rendered_lines,
        CHARACTERS_SECTION,
        selected_character_lines + pinned_gender_lines,
        max_chars,
        reserved,
    )
    _append_section_with_budget(
        rendered_lines,
        ALIASES_SECTION,
        selected_alias_lines,
        max_chars,
        reserved,
    )
    _append_section_with_budget(
        rendered_lines,
        GLOSSARY_SECTION,
        selected_glossary_lines,
        max_chars,
        reserved,
    )

    rendered_lines.extend(["", DYNAMIC_STATE_START, "# DYNAMIC RELATIONSHIP STATE"])
    _append_section_with_budget(
        rendered_lines,
        ADDRESSING_SECTION,
        selected_addressing,
        max_chars,
        len(DYNAMIC_STATE_END),
    )
    _append_section_with_budget(
        rendered_lines,
        RELATIONSHIP_SECTION,
        selected_relationships,
        max_chars,
        len(DYNAMIC_STATE_END),
    )
    rendered_lines.append(DYNAMIC_STATE_END)
    return "\n".join(rendered_lines).strip()


def render_novel_context_update_view(
    current_global_lore: str,
    current_dynamic_state: str,
    reference_text: str = "",
    max_tokens: Optional[int] = None,
    selective: bool = True,
) -> Tuple[str, str]:
    """Return the lore/dynamic view sent to the context-update LLM.

    The deterministic merge layer still receives the complete stored lore after
    the LLM returns. This view only reduces prompt tokens and unrelated context.
    """
    full_context = build_novel_context(current_global_lore, current_dynamic_state)
    rendered = render_novel_context_for_prompt(
        full_context,
        reference_text=reference_text,
        max_tokens=max_tokens,
        selective=selective,
        include_gender_roster=False,
    )
    if not rendered:
        return "", ""
    return (
        extract_global_lore(rendered),
        extract_dynamic_state_from_text(rendered) or "",
    )


def resolve_novel_context_update_interval(
    prompt_options: Optional[Dict[str, Any]] = None,
) -> int:
    """Return the auto-update cadence for source-derived context analysis."""
    prompt_options = prompt_options or {}
    raw_value = (
        prompt_options.get("novel_context_update_interval")
        or prompt_options.get("context_update_interval")
    )
    if raw_value is None:
        try:
            from src import config as _config
            raw_value = getattr(_config, "NOVEL_CONTEXT_UPDATE_INTERVAL", 1)
        except Exception:
            raw_value = 1
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return 1


def should_update_novel_context_for_index(
    zero_based_index: int,
    prompt_options: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True when auto context analysis should run for this unit.

    Index 0 always updates. With interval N, subsequent updates happen at
    indices N, 2N, ... so the user-facing chunks are 1, N+1, 2N+1, ...
    """
    interval = resolve_novel_context_update_interval(prompt_options)
    return zero_based_index <= 0 or zero_based_index % interval == 0


def make_novel_context_filename(input_filename: str, fallback: str = "translation") -> str:
    """Create a safe, deterministic context filename from an input filename."""
    basename = str(input_filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(basename).stem
    safe_stem = _safe_context_filename_stem(stem)
    if not safe_stem:
        fallback_basename = str(fallback or "").replace("\\", "/").rsplit("/", 1)[-1]
        safe_stem = _safe_context_filename_stem(Path(fallback_basename).stem) or "translation"
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
        status = chunk.get("status")
        if status is not None and status != "completed":
            continue
        if status == "completed" and chunk.get("translated_text") is None:
            continue
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
            canonicalize_dialogue_attribution,
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
        current_aliases = _character_alias_map(self.global_lore)
        self.current_dialogue_attribution = (
            canonicalize_dialogue_attribution(
                historical_dialogue,
                current_aliases,
            )
            if historical_dialogue
            else empty_dialogue_attribution()
        )
        if historical_dialogue:
            self.dialogue_state = dict(
                self.current_dialogue_attribution.get("state_after") or {}
            )

        if historical:
            historical_context = normalize_refinement_context(
                historical,
                build_novel_context(self.global_lore, self.dynamic_state),
            )
            self.global_lore = extract_global_lore(historical_context)
            refreshed_aliases = _character_alias_map(self.global_lore)
            self.current_dialogue_attribution = (
                canonicalize_dialogue_attribution(
                    self.current_dialogue_attribution,
                    refreshed_aliases,
                )
            )
            self.dialogue_state = dict(
                self.current_dialogue_attribution.get("state_after") or {}
            )
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
                selective_context_view=self.prompt_options.get(
                    "novel_context_selective_update",
                    True,
                ),
                context_view_max_tokens=self.prompt_options.get(
                    "novel_context_update_prompt_max_tokens",
                ),
            )
            self.current_dialogue_attribution = (
                dialogue_sink
                or empty_dialogue_attribution()
            )
            self.dialogue_state = dict(
                self.current_dialogue_attribution.get("state_after") or {}
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
    source_memory: List[str] = field(default_factory=list)

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

        source_context = _bounded_source_memory(self.source_memory)
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
            source_context=source_context,
            dialogue_turns=dialogue_turns,
            current_dialogue_state=self.dialogue_state,
            dialogue_attribution_sink=dialogue_sink,
            selective_context_view=self.prompt_options.get(
                "novel_context_selective_update",
                True,
            ),
            context_view_max_tokens=self.prompt_options.get(
                "novel_context_update_prompt_max_tokens",
            ),
        )
        clean_source = _clean_source_memory_chunk(source_chunk)
        if clean_source:
            self.source_memory.append(clean_source)
            self.source_memory = [
                chunk for chunk in self.source_memory
                if chunk.strip()
            ]
            bounded = _bounded_source_memory(self.source_memory)
            self.source_memory = (
                bounded.split("\n\n--- Previous source chunk ---\n\n")
                if bounded
                else []
            )
        self.dialogue_attribution = (
            dialogue_sink
            or empty_dialogue_attribution()
        )
        self.dialogue_state = dict(
            self.dialogue_attribution.get("state_after") or {}
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
    # Override global bypass gating variable dynamically based on job parameters
    if "bypass_context_gating" in prompt_options:
        try:
            from src import config as _config
            _config.BYPASS_CONTEXT_GATING = bool(prompt_options["bypass_context_gating"])
        except Exception:
            pass

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

    from src.utils.dialogue_attribution import canonicalize_dialogue_state
    resume_dialogue_state = canonicalize_dialogue_state(
        resume_dialogue_state,
        _character_alias_map(global_lore),
    )

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


CONSOLIDATION_SYSTEM_PROMPT = """You are a precise novel context editor.
Your only job is to clean up a Characters & Genders list by merging duplicate or redundant entries and removing entries that are not characters.

Rules:
- Each canonical character must appear exactly once with one concise, non-repetitive description.
- Merge all duplicate facts (same idea, different wording) into a single clear phrase.
- Keep the most informative phrasing; drop redundant restatements.
- Keep ALL factual information that is not a pure duplicate: gender, rank, role, key relationships, notable traits.
- Preserve the exact canonical name already shown in the input (do not rename characters).
- Preserve gender labels exactly (Male / Female / Unspecified).
- Identify and remove generic background NPCs (e.g., unnamed students, teachers, guards, bystanders) that do not have named proper counterparts or significant recurring, plot-driving descriptions.
- Identify and remove first-pass non-character mistakes: companies, organizations, factions, families/houses/lineages, countries, facilities, magic circles, artifacts, weapons, skills, systems, metrics, titles, and abstract concepts do not belong in Characters & Genders.
- If a non-character proper noun is useful terminology (e.g., a company name, family/house name, named artifact, named magic circle, or product name), omit it from this output; it belongs in Glossary & Terminology, not Characters & Genders.
- Remove bare relationship labels such as "Brother", "Father", "Lover", "Girlfriend", "Wife", or "Husband" unless the entry clearly names a distinct source-named character; relationships belong in descriptions or dynamic state, not as fake character identities.
- Remove duplicate romanization/source-name variants for Chinese, Japanese, and Korean names when the entries clearly describe the same person. Keep the existing canonical character name from the input and fold the facts into that one entry; do not invent a new pinyin, romaji, or revised-romanization canonical name.
- Output ONLY the cleaned bullet list — one line per character in the format:
  - Canonical Name: Gender, concise description.
- Do NOT add any heading, explanation, markdown fence, or extra text.
- Do NOT invent new facts that are not present in the input.
"""

CONSOLIDATION_USER_PROMPT_TEMPLATE = """Below is the current Characters & Genders list.
Merge duplicate / redundant descriptions so each character has exactly one concise entry.

{characters_section}

Output the cleaned list now (bullet lines only, no other text):"""


async def consolidate_context_lore(
    llm_client: Any,
    model_name: str,
    global_lore: str,
) -> Tuple[str, List[str]]:
    """Run an LLM pass that deduplicates/consolidates the Characters section.

    The deterministic merge catches exact or near-exact fact overlap but misses
    semantically identical descriptions written with different words.  This
    function asks the LLM to rewrite the Characters section into one clean,
    non-redundant entry per character.

    Returns:
        Tuple of (updated_global_lore, change_logs)
    """
    change_logs: List[str] = []

    # Extract the existing Characters section
    char_bounds = _find_lore_section(global_lore, CHARACTERS_SECTION)
    if not char_bounds:
        return global_lore, change_logs

    _, body_start, body_end = char_bounds
    characters_body = global_lore[body_start:body_end].strip()
    if not characters_body:
        return global_lore, change_logs

    user_prompt = CONSOLIDATION_USER_PROMPT_TEMPLATE.format(
        characters_section=characters_body,
    )

    try:
        response = await llm_client.generate(
            prompt=user_prompt,
            system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
        )
        if not response or not response.content:
            logger.warning(
                "Empty response from LLM during context consolidation. Skipping."
            )
            return global_lore, change_logs

        raw = response.content.strip()
        # Clean any markdown code blocks
        if "```" in raw:
            # Try to extract content inside the first code block
            block_match = re.search(r'```(?:[a-zA-Z-]*)\n(.*?)```', raw, re.DOTALL)
            if block_match:
                raw = block_match.group(1).strip()
            else:
                # Fallback: just strip the fence lines
                raw = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("```")).strip()

        # Validate: must have at least one character line (with - or * or number or no prefix but contains ':')
        consolidated_entries = []
        for line in raw.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            
            # Try to match a bullet/list marker: '-', '*', '1.', '2.', etc.
            match = re.match(r'^(?:[-*]|\d+\.)\s*(.*)$', line_str)
            if match:
                content = match.group(1).strip()
            else:
                content = line_str
                
            if ":" in content and not content.startswith("#"):
                parts = content.split(":", 1)
                val = parts[1].strip()
                val_first_word = re.split(r'[\s,.]', val)[0].casefold()
                if val_first_word in _GENDER_LABELS:
                    consolidated_entries.append(f"- {content}")

        if not consolidated_entries:
            logger.warning(
                "Consolidation LLM returned no valid character entries. Skipping."
            )
            return global_lore, change_logs

        consolidated_body = "\n".join(consolidated_entries)
        updated_lore = _replace_lore_section(
            global_lore,
            CHARACTERS_SECTION,
            consolidated_entries,
        )
        updated_lore = normalize_global_lore(updated_lore)

        if updated_lore != global_lore:
            msg = "[Novel Context] Consolidation pass: character list deduped/merged and non-character entries pruned."
            change_logs.append(msg)

        return updated_lore, change_logs

    except Exception as exc:
        logger.error(f"Error during context consolidation LLM call: {exc}")
        return global_lore, change_logs


def _consolidation_interval() -> int:
    """Return the configured consolidation interval (0 = disabled)."""
    try:
        from src import config as _config
        return max(0, int(getattr(_config, "NOVEL_CONTEXT_CONSOLIDATION_INTERVAL", 5)))
    except Exception:
        return 5


UPDATE_SYSTEM_PROMPT = """You are an expert novel translation context assistant.
Your task is to analyze the latest source text and its translation, and detect any new characters, glossary terms, or relationship addressing changes.

Identity rules:
- Reuse the exact canonical name already present in CURRENT GLOBAL LORE.
- A title, rank, nickname, transformed state, awakened state, disguise, age qualifier, or relationship label is not a new character when it refers to an existing person. Update the existing canonical entry instead.
- A descriptor-only label such as "Protagonist", "Hero", "Main Character", "Player Character", "Protagonist of X", "Hero of X", "fictional character", or "character from X" is not a canonical character name or identity link. Use source names such as Kim Ji-an, Valentine, or Eric; otherwise omit it.
- When a title-only entry is later identified by name (for example, "Emperor" = "Serena Augusta"), output only the named canonical character with the title in its concise description.
- When the latest source directly proves that a stable, book-wide title, rank, nickname, or other label is an existing character, record that mapping under IDENTITY_LINKS. Valid proof includes explicit naming, apposition, an identity reveal, or unambiguous same-scene coreference such as a direct address immediately attributed to the named character. Never create an identity link from role similarity alone. Do not persist a bare title that can refer to multiple people or transfer between characters; use the canonical name directly for that scene instead.
- If the source links a role/title to a named person by location or narration (for example, "the Lieutenant Colonel's office" followed by "Eric" as the person in that office), record the role/title under IDENTITY_LINKS instead of creating a separate character.
- For non-English source titles or aliases, preserve the exact source surface label under IDENTITY_LINKS when it is source-proven (for example, "- 중령: Eric"). If the English normalized title also appears in the model's character summary, link that title too.
- For Chinese/Japanese/Korean source names, do not invent new romanized character names such as pinyin, romaji, or revised romanization variants. Reuse an existing canonical romanized name; otherwise record the exact source surface name under NEW_CHARACTERS and put romanization recommendations only under NEW_GLOSSARY.
- When adding or correcting a Chinese/Japanese/Korean source-script character name, also add a NEW_GLOSSARY entry that maps the exact source name to the recommended target-language name rendering. If a short source name or honorific address form is durable, add that exact source form too with the appropriate short or honorific target rendering.
- When the target language is Vietnamese and an English named skill, ability, technique, spell, combat move, weapon, artifact, or equipment needs glossary tracking, prefer a concise Sino-Vietnamese literary target term if it sounds natural in Vietnamese fantasy or game prose. Preserve English only for brands, code labels, UI/system keys, or terms that clearly should remain untranslated.
- Do not add one-scene unnamed soldiers, victims, hallucinations, generic crowds, or incidental job labels unless they recur and their identity/gender is required for translation consistency.
- Do not add numbered/background casualties or generic staff labels such as "Wounded Soldier 1", "Guard 2", "Doctor", or "Private" unless the person is source-named, recurring, or needed for a durable addressing/relationship choice.
- Do not add bare romantic or family relationship labels such as "Lover", "Girlfriend", "Ex-girlfriend", "Boyfriend", "Partner", "Spouse", "Wife", or "Husband" as characters or identity links. If the relationship partner is not source-named, record only the relationship, not a fake identity alias.
- Never output template entries, "None", "[None]", "Unknown", "N/A", or an empty bullet.
- Record gender only when the source states it or supplies unambiguous grammatical/pronoun evidence. Never guess gender from a name, occupation, rank, appearance, genre convention, or stereotype.
- Gender-neutral words such as spouse, partner, lover, parent, child, sibling, officer, captain, commander, major, colonel, and lieutenant colonel never prove gender by themselves.
- A named character having a girlfriend, boyfriend, lover, spouse, or ex-partner does not by itself prove that named character's gender; use pronouns or grammatical evidence such as "him/his" or "her" instead.
- If a character reincarnates, transforms, disguises themselves, or receives a new body, record the gender of the current named form, not the previous body. Keep the previous identity only as a concise description.
- Pronoun evidence attached to another character's relationship with a named person is evidence for that named person. For example, "suspicious of her identity" after naming Valentine proves Valentine is Female, not the suspicious officer.
- If CURRENT GLOBAL LORE has the wrong gender for the current named form and the latest source proves the correction, output that canonical character under NEW_CHARACTERS as "CORRECTION: [Gender, concise role, concise description]". Do not preserve stale gender just because it is already stored.
- Before writing "Unspecified", scan the whole latest source for direct evidence such as gendered nouns, pronouns, kinship grammar, or an explicit description. If an existing Unspecified character is now proven Male/Female, output that specific gender directly; this is not a correction.
- An existing specific gender is authoritative. Change it only when the latest source explicitly proves it was wrong; write that rare update as "CORRECTION: [Gender, role, description]".
- Write all character metadata in English, regardless of source and target language.
- For an existing character, output one concise cumulative replacement description containing the important old and new facts. Summarize repeated roles instead of appending duplicate phrases.
- Character descriptions must contain only the normalized result. Never append evidence notes, quotations, reasoning, confidence, "source pronoun evidence", "reincarnated current form", "Gender confirmed...", "Correction...", parenthetical explanations, or prompt/control labels such as "current rank and title" or "title/nickname for X".

Input provided:
1. CURRENT GLOBAL LORE (Characters & Glossary)
2. CURRENT DYNAMIC RELATIONSHIP STATE
3. RECENT SOURCE MEMORY (bounded previous chunks)
4. LATEST SOURCE TEXT
5. LATEST TRANSLATION

Your output must follow this strict format:

[NEW_CHARACTERS]
- Canonical Name: [Gender, role, and concise description]
(Use "Unspecified" rather than guessing when a recurring character must be tracked before gender is explicit. Use "CORRECTION: [Gender, role, description]" only for a source-proven correction. Use "- Canonical Name: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[IDENTITY_LINKS]
- Source title, rank, nickname, or alias: Canonical Name
(Only include identity links directly established by the latest source. The right side must be one exact canonical character name from CURRENT GLOBAL LORE or NEW_CHARACTERS. Use "- Alias: DELETE" to remove a wrong link. If there are no changes, output no bullet under this header.)

[NEW_GLOSSARY]
- Source Term: [Target Term]
(Only include actual additions or corrections. Use "- Source Term: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[DYNAMIC_STATE]
# DYNAMIC RELATIONSHIP STATE
## CURRENT ADDRESSING FORMS
- Speaker → Addressee: source form "..." | target-language form "..." | register, social basis, scope, and reason
## RELATIONSHIP EVOLUTION
- Character A ↔ Character B: concise current relationship
(Output both headings every time, but list only additions or changes. Omitted entries remain stored indefinitely. Remove an obsolete entry only with "- Speaker → Addressee: DELETE" or "- Character A ↔ Character B: DELETE". Addressing forms include names, titles, honorifics, pronouns, kinship terms, and formality choices needed in the target language. In the final reason field, record the social basis when known: direct address vs indirect reference scope, age/school-year/seniority, family relation, rank/status, setting, intimacy, hostility, deference, or exception to normal age hierarchy. Use plain Unicode arrows only. Never use LaTeX, backslashes, dollar signs, or ASCII arrows. Do not duplicate these headings.)

[DIALOGUE_ATTRIBUTION]
{"turns":[{"id":"exact candidate id","speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown","confidence":0.0}],"state_after":{"speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown"}}
(Classify only the supplied dialogue candidates. Infer from narration, direct dialogue tags, turn-taking, voice, and addressing forms in the latest source. CURRENT SCENE SPEAKER STATE is a weak continuity hint only; never use it as sole proof, and return Unknown when local evidence is unclear. Resolve titles and aliases through CURRENT GLOBAL LORE and IDENTITY_LINKS, but output only canonical character names already present in CURRENT GLOBAL LORE or NEW_CHARACTERS. Never invent a speaker. Confidence is from 0.0 to 1.0. Return {"turns":[],"state_after":{}} when there are no candidates or no current high-confidence speaker.)

Do not include any other explanations, markdown fences, or extra text outside these blocks.
"""

SOURCE_ANALYSIS_SYSTEM_PROMPT = """You are an expert novel translation context assistant.
Analyze the latest SOURCE text before it is translated. Detect new or corrected characters, source-proven genders and roles, source terminology that needs a consistent target-language rendering, and relationship/addressing changes that the translator must know now.

Identity rules:
- Reuse the exact canonical name already present in CURRENT GLOBAL LORE.
- Do not create separate characters for ranks, titles, nicknames, transformed/awakened states, disguises, age variants, or relational aliases of an existing person.
- A descriptor-only label such as "Protagonist", "Hero", "Main Character", "Player Character", "Protagonist of X", "Hero of X", "fictional character", or "character from X" is not a canonical character name or identity link. Use source names such as Kim Ji-an, Valentine, or Eric; otherwise omit it.
- When a title-only entry is later identified by name (for example, "Emperor" = "Serena Augusta"), output only the named canonical character with the title in its concise description.
- When this source directly proves that a stable, book-wide title, rank, nickname, or other label is an existing character, record that mapping under IDENTITY_LINKS. Valid proof includes explicit naming, apposition, an identity reveal, or unambiguous same-scene coreference such as a direct address immediately attributed to the named character. Never create an identity link from role similarity alone. Do not persist a bare title that can refer to multiple people or transfer between characters; use the canonical name directly for that scene instead.
- If the source links a role/title to a named person by location or narration (for example, "the Lieutenant Colonel's office" followed by "Eric" as the person in that office), record the role/title under IDENTITY_LINKS instead of creating a separate character.
- For non-English source titles or aliases, preserve the exact source surface label under IDENTITY_LINKS when it is source-proven (for example, "- 중령: Eric"). If the English normalized title also appears in the model's character summary, link that title too.
- For Chinese/Japanese/Korean source names, do not invent new romanized character names such as pinyin, romaji, or revised romanization variants. Reuse an existing canonical romanized name; otherwise record the exact source surface name under NEW_CHARACTERS and put romanization recommendations only under NEW_GLOSSARY.
- When adding or correcting a Chinese/Japanese/Korean source-script character name, also add a NEW_GLOSSARY entry that maps the exact source name to the recommended target-language name rendering. If a short source name or honorific address form is durable, add that exact source form too with the appropriate short or honorific target rendering.
- When the target language is Vietnamese and an English named skill, ability, technique, spell, combat move, weapon, artifact, or equipment needs glossary tracking, prefer a concise Sino-Vietnamese literary target term if it sounds natural in Vietnamese fantasy or game prose. Preserve English only for brands, code labels, UI/system keys, or terms that clearly should remain untranslated.
- Do not add one-scene unnamed soldiers, victims, generic crowds, or incidental roles unless they recur and are necessary for pronoun/address consistency.
- Do not add numbered/background casualties or generic staff labels such as "Wounded Soldier 1", "Guard 2", "Doctor", or "Private" unless the person is source-named, recurring, or needed for a durable addressing/relationship choice.
- Do not add bare romantic or family relationship labels such as "Lover", "Girlfriend", "Ex-girlfriend", "Boyfriend", "Partner", "Spouse", "Wife", or "Husband" as characters or identity links. If the relationship partner is not source-named, record only the relationship, not a fake identity alias.
- Never output template entries, "None", "[None]", "Unknown", "N/A", or an empty bullet.
- Record gender only when this source text states it or gives unambiguous grammatical/pronoun evidence. Never infer it from names, jobs, ranks, appearance, personality, or genre stereotypes.
- Gender-neutral words such as spouse, partner, lover, parent, child, sibling, officer, captain, commander, major, colonel, and lieutenant colonel never prove gender by themselves.
- A named character having a girlfriend, boyfriend, lover, spouse, or ex-partner does not by itself prove that named character's gender; use pronouns or grammatical evidence such as "him/his" or "her" instead.
- If a character reincarnates, transforms, disguises themselves, or receives a new body, record the gender of the current named form, not the previous body. Keep the previous identity only as a concise description.
- Pronoun evidence attached to another character's relationship with a named person is evidence for that named person. For example, "suspicious of her identity" after naming Valentine proves Valentine is Female, not the suspicious officer.
- If CURRENT GLOBAL LORE has the wrong gender for the current named form and this source proves the correction, output that canonical character under NEW_CHARACTERS as "CORRECTION: [Gender, concise role, concise description]". Do not preserve stale gender just because it is already stored.
- Before writing "Unspecified", scan the entire latest source for direct evidence such as gendered nouns, pronouns, kinship grammar, or an explicit description. If an existing Unspecified character is now proven Male/Female, output that specific gender directly; this is not a correction.
- Treat an existing specific gender as authoritative. Change it only when this source text explicitly proves it wrong, using "CORRECTION: [Gender, role, description]".
- Preserve source-side proper names exactly. Write character metadata descriptions in English so context remains stable when the translation model or target language changes.
- For an existing character, output one concise cumulative replacement description containing the important old and new facts. Summarize repeated roles instead of appending duplicate phrases.
- Character descriptions must contain only the normalized result. Never append evidence notes, quotations, reasoning, confidence, "source pronoun evidence", "reincarnated current form", "Gender confirmed...", "Correction...", parenthetical explanations, or prompt/control labels such as "current rank and title" or "title/nickname for X".

Input provided:
1. CURRENT GLOBAL LORE (Characters & Glossary)
2. CURRENT DYNAMIC RELATIONSHIP STATE
3. RECENT SOURCE MEMORY (bounded previous chunks)
4. LATEST SOURCE TEXT

Your output must follow this strict format:

[NEW_CHARACTERS]
- Canonical Name: [Gender, role, and concise description]
(Use "Unspecified" rather than guessing when a recurring character must be tracked before gender is explicit. Use "CORRECTION: [Gender, role, description]" only for a source-proven correction. Use "- Canonical Name: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[IDENTITY_LINKS]
- Source title, rank, nickname, or alias: Canonical Name
(Only include identity links directly established by this source. The right side must be one exact canonical character name from CURRENT GLOBAL LORE or NEW_CHARACTERS. Use "- Alias: DELETE" to remove a wrong link. If there are no changes, output no bullet under this header.)

[NEW_GLOSSARY]
- Source Term: [Recommended Target Term]
(Only include important recurring terms. Use "- Source Term: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[DYNAMIC_STATE]
# DYNAMIC RELATIONSHIP STATE
## CURRENT ADDRESSING FORMS
- Speaker → Addressee: source form "..." | recommended target-language form "..." | register, social basis, scope, and reason
## RELATIONSHIP EVOLUTION
- Character A ↔ Character B: concise current relationship
(Output both headings every time, but list only additions or changes. Omitted entries remain stored indefinitely. Remove an obsolete entry only with "- Speaker → Addressee: DELETE" or "- Character A ↔ Character B: DELETE". Addressing forms include names, titles, honorifics, pronouns, kinship terms, and formality choices needed for translation. In the final reason field, record the social basis when known: direct address vs indirect reference scope, age/school-year/seniority, family relation, rank/status, setting, intimacy, hostility, deference, or exception to normal age hierarchy. Use plain Unicode arrows only. Never use LaTeX, backslashes, dollar signs, or ASCII arrows. Keep it concise and do not duplicate headings.)

[DIALOGUE_ATTRIBUTION]
{"turns":[{"id":"exact candidate id","speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown","confidence":0.0}],"state_after":{"speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown"}}
(Classify only the supplied dialogue candidates. Infer from narration, direct dialogue tags, turn-taking, voice, and addressing forms in the latest source. CURRENT SCENE SPEAKER STATE is a weak continuity hint only; never use it as sole proof, and return Unknown when local evidence is unclear. Resolve titles and aliases through CURRENT GLOBAL LORE and IDENTITY_LINKS, but output only canonical character names already present in CURRENT GLOBAL LORE or NEW_CHARACTERS. Never invent a speaker. Confidence is from 0.0 to 1.0. Return {"turns":[],"state_after":{}} when there are no candidates or no current high-confidence speaker.)

Do not translate the whole passage. Do not include explanations, markdown fences, or text outside these blocks.
"""

UPDATE_USER_PROMPT_TEMPLATE = """### CURRENT GLOBAL LORE:
{current_global_lore}

### CURRENT DYNAMIC RELATIONSHIP STATE:
{current_dynamic_state}

### TRANSLATION PROGRESS: Segment {chunk_index} of {total_chunks}

### RECENT SOURCE MEMORY ({source_language}, previous chunks only):
{source_context}

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

### RECENT SOURCE MEMORY ({source_language}, previous chunks only):
{source_context}

### LATEST SOURCE TEXT ({source_language}):
{source_chunk}

### TARGET LANGUAGE:
{target_language}

### CURRENT SCENE SPEAKER STATE:
{current_dialogue_state}

### DIALOGUE CANDIDATES:
{dialogue_candidates}

Analyze the source for context needed by its translation. Output ONLY the strictly formatted blocks."""


def merge_new_lore(
    global_lore: str,
    new_characters: str,
    new_glossary: str,
    new_aliases: str = "",
    source_text: str = "",
    trusted_aliases: str = "",
) -> Tuple[str, List[str]]:
    """Merge context updates through canonical character and glossary identities."""
    change_logs: List[str] = []
    raw_global_lore = str(global_lore or "")
    lore = normalize_global_lore(global_lore)

    def record(message: str) -> None:
        change_logs.append(message)

    alias_bounds = _find_lore_section(lore, ALIASES_SECTION)
    alias_entries = (
        _parse_alias_entries(lore[alias_bounds[1]:alias_bounds[2]])
        if alias_bounds
        else []
    )
    explicit_aliases = _alias_entries_to_map(alias_entries)
    alias_displays = {
        alias_key: alias
        for alias, _ in alias_entries
        for alias_key in _character_alias_keys(alias)
    }

    alias_updates = [
        (raw_alias, raw_target, True)
        for raw_alias, raw_target in _parse_bullet_entries(trusted_aliases)
    ] + [
        (raw_alias, raw_target, False)
        for raw_alias, raw_target in _parse_bullet_entries(new_aliases)
    ]

    for raw_alias, raw_target, is_trusted_alias in alias_updates:
        if (
            _is_invalid_context_key(raw_alias)
            or _is_unstable_identity_alias(raw_alias, allow_physical=True)
        ):
            continue
        alias_keys = _character_alias_keys(raw_alias)
        if not alias_keys:
            continue
        is_delete = _strip_balanced_brackets(raw_target).casefold() == "delete"
        if is_delete:
            removed = False
            for alias_key in alias_keys:
                if alias_key in explicit_aliases:
                    explicit_aliases.pop(alias_key, None)
                    alias_displays.pop(alias_key, None)
                    removed = True
            if removed:
                record(
                    "[Novel Context] Deleted identity link "
                    f"'{_plain_key(raw_alias)}'."
                )
            continue

        target = _canonical_display_name(raw_target)
        if (
            _is_invalid_context_key(target)
            or _character_names_match(raw_alias, target)
        ):
            continue
        already_linked = any(
            explicit_aliases.get(alias_key) == target
            for alias_key in alias_keys
        )
        identity_link_proved = True
        identity_link_skip_reason = ""
        if source_text and not is_trusted_alias and not already_linked:
            identity_link_proved, identity_link_skip_reason = _source_identity_link_proof_status(
                source_text,
                lore,
                new_characters,
                raw_alias,
                target,
            )
        if not identity_link_proved:
            logger.warning(
                "[Novel Context] Skipped unsafe identity link '%s' -> '%s': %s.",
                _plain_key(raw_alias),
                target,
                identity_link_skip_reason,
            )
            continue
        changed = any(
            explicit_aliases.get(alias_key) != target
            for alias_key in alias_keys
        )
        for alias_key in alias_keys:
            explicit_aliases[alias_key] = target
            alias_displays[alias_key] = _strip_balanced_brackets(raw_alias)
        if changed:
            record(
                "[Novel Context] Linked identity "
                f"'{_plain_key(raw_alias)}' -> '{target}'."
            )

    character_bounds = _find_lore_section(lore, CHARACTERS_SECTION)
    current_character_entries: List[Tuple[str, str]] = []
    if character_bounds:
        _, body_start, body_end = character_bounds
        current_character_entries = _parse_bullet_entries(
            lore[body_start:body_end]
        )
        characters, current_deduced_aliases = _deduplicate_character_entries(
            current_character_entries,
            explicit_aliases,
        )
        _retain_renderable_aliases(
            explicit_aliases,
            alias_displays,
            current_deduced_aliases,
            current_character_entries,
        )
    else:
        characters = []

    glossary_alias_entries: List[Tuple[str, str]] = []
    glossary_alias_bounds = _find_lore_section(lore, GLOSSARY_SECTION)
    if glossary_alias_bounds:
        _, body_start, body_end = glossary_alias_bounds
        glossary_alias_entries.extend(
            _parse_bullet_entries(lore[body_start:body_end])
        )
    glossary_alias_entries.extend(_parse_bullet_entries(new_glossary))
    _add_glossary_character_aliases(
        explicit_aliases,
        alias_displays,
        glossary_alias_entries,
        characters,
    )

    incoming_character_entries = _parse_bullet_entries(new_characters)
    raw_character_bounds = _find_lore_section(
        raw_global_lore,
        CHARACTERS_SECTION,
    )
    if raw_character_bounds and explicit_aliases:
        _, raw_body_start, raw_body_end = raw_character_bounds
        retained_alias_keys = {
            alias
            for name, _ in characters
            for alias in _character_alias_keys(name)
        }
        for raw_name, raw_value in _parse_bullet_entries(
            raw_global_lore[raw_body_start:raw_body_end]
        ):
            raw_aliases = _character_alias_keys(raw_name)
            if not raw_aliases or not (raw_aliases & set(explicit_aliases)):
                continue
            if raw_aliases & retained_alias_keys:
                continue
            incoming_character_entries.append((raw_name, raw_value))
    inferred_aliases, inferred_displays = _infer_unique_short_name_alias_entries(
        characters + incoming_character_entries,
        explicit_aliases,
    )
    if inferred_aliases:
        changed_aliases = []
        for alias_key, target in inferred_aliases.items():
            if explicit_aliases.get(alias_key) == target:
                continue
            explicit_aliases[alias_key] = target
            if alias_key in inferred_displays:
                alias_displays[alias_key] = inferred_displays[alias_key]
            changed_aliases.append((alias_key, target))
        if changed_aliases:
            characters, changed_deduced_aliases = _deduplicate_character_entries(
                characters,
                explicit_aliases,
            )
            _retain_renderable_aliases(
                explicit_aliases,
                alias_displays,
                changed_deduced_aliases,
                characters,
            )
            for alias_key, target in changed_aliases:
                record(
                    "[Novel Context] Linked identity "
                    f"'{alias_displays.get(alias_key, alias_key)}' -> '{target}'."
                )
    regular_incoming = [
        entry for entry in incoming_character_entries
        if not _is_descriptive_role_name(entry[0])
    ]
    descriptive_incoming = [
        entry for entry in incoming_character_entries
        if _is_descriptive_role_name(entry[0])
    ]

    for raw_name, raw_value in regular_incoming + descriptive_incoming:
        if (
            _is_non_character_work_entry(raw_name, raw_value)
            or _is_non_character_group_entry(raw_name, raw_value)
            or _is_non_character_metadata_or_item_entry(raw_name, raw_value)
            or _is_disposable_unnamed_character(raw_name, raw_value)
        ):
            continue
        incoming_aliases = _character_alias_keys(raw_name)
        forced_name = next(
            (
                explicit_aliases[alias]
                for alias in incoming_aliases
                if alias in explicit_aliases
            ),
            None,
        )
        effective_name = forced_name or raw_name
        descriptive_name = bool(
            _is_descriptive_role_name(raw_name)
            and not forced_name
        )
        if (
            not forced_name
            and _is_quarantined_character_entry(raw_name, raw_value)
        ):
            record(
                "[Novel Context] Quarantined role-like character "
                f"'{_plain_key(raw_name)}'."
            )
            continue
        if (
            not forced_name
            and _is_unstable_physical_character_entry(raw_name, raw_value)
        ):
            record(
                "[Novel Context] Quarantined physical placeholder "
                f"'{_plain_key(raw_name)}'."
            )
            continue
        incoming_aliases |= _character_alias_keys(effective_name)
        match_index = None
        for index, (existing_name, existing_value) in enumerate(characters):
            if (
                incoming_aliases & _character_alias_keys(existing_name)
                or _character_identities_match(
                    existing_name,
                    existing_value,
                    effective_name,
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
                record(f"[Novel Context] Deleted character '{log_key}'.")
            continue

        canonical_name = _canonical_display_name(effective_name)
        clean_value, explicit_correction = _strip_character_correction_marker(
            raw_value
        )
        clean_value = _normalize_character_value(clean_value)
        clean_value = _gate_unproven_character_gender(
            canonical_name,
            clean_value,
            source_text,
            characters[match_index][1] if match_index is not None else "",
            explicit_correction,
        )
        if match_index is None:
            if descriptive_name:
                continue
            characters.append((canonical_name, clean_value))
            record(
                f"[Novel Context] Added character '{_plain_key(canonical_name)}'."
            )
            continue

        old_name, old_value = characters[match_index]
        merged_name = (
            canonical_name
            if forced_name
            else old_name
            if descriptive_name
            else _preferred_character_name(old_name, canonical_name)
        )
        merged_value = _merge_character_values(
            old_value,
            clean_value,
            allow_gender_correction=explicit_correction,
        )
        if (old_name, old_value) != (merged_name, merged_value):
            record(
                f"[Novel Context] Updated character '{log_key}'."
            )
        characters[match_index] = (merged_name, merged_value)

    final_character_entries = list(characters)
    characters, final_deduced_aliases = _deduplicate_character_entries(
        characters,
        explicit_aliases,
    )
    _retain_renderable_aliases(
        explicit_aliases,
        alias_displays,
        final_deduced_aliases,
        final_character_entries + incoming_character_entries,
    )
    lore = _replace_lore_section(
        lore,
        CHARACTERS_SECTION,
        [_format_character_line(name, value) for name, value in characters],
    )
    canonical_aliases = _canonical_alias_entries(
        explicit_aliases,
        characters,
        alias_displays,
    )
    if alias_bounds or canonical_aliases:
        lore = _replace_lore_section(
            lore,
            ALIASES_SECTION,
            [
                f"- {alias}: {target}"
                for alias, target in canonical_aliases
            ],
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
                record(f"[Novel Context] Deleted glossary term '{key}'.")
            continue

        clean_name = _strip_balanced_brackets(raw_name)
        clean_value = _strip_balanced_brackets(raw_value)
        if key in glossary_index:
            index = glossary_index[key]
            old_name, old_value = glossary[index]
            if (old_name, old_value) != (clean_name, clean_value):
                record(
                    f"[Novel Context] Updated glossary term '{key}'."
                )
            glossary[index] = (clean_name, clean_value)
        else:
            glossary_index[key] = len(glossary)
            glossary.append((clean_name, clean_value))
            record(
                f"[Novel Context] Added glossary term '{key}'."
            )

    lore = _replace_lore_section(
        lore,
        GLOSSARY_SECTION,
        [f"- {name}: {value}" for name, value in glossary],
    )
    return normalize_global_lore(lore), change_logs


_CONTEXT_UPDATE_JSON_KEYS = {
    "characters",
    "new_characters",
    "identity_links",
    "aliases",
    "new_glossary",
    "glossary",
    "dynamic_state",
    "dialogue_attribution",
}


def _json_key_id(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key or "").casefold())


_CONTEXT_UPDATE_JSON_KEY_IDS = {
    _json_key_id(key)
    for key in _CONTEXT_UPDATE_JSON_KEYS
}


def _json_get(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    wanted = {_json_key_id(key) for key in keys}
    for key, value in mapping.items():
        if _json_key_id(key) in wanted:
            return value
    return None


def _strip_markdown_fence(text: str) -> str:
    text = str(text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _find_balanced_container(
    text: str,
    start: int,
    opener: str = "{",
    closer: str = "}",
) -> Optional[str]:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _parse_context_update_json(raw: str) -> Optional[Dict[str, Any]]:
    """Extract a structured context update from a model response.

    Providers vary in how strictly they honor JSON-only instructions, so this
    accepts bare JSON, fenced JSON, or a balanced JSON object embedded in text.
    The object must contain at least one recognized context-update key to avoid
    confusing legacy dialogue-only JSON blocks with a full update.
    """
    text = re.sub(
        r"<think>.*?</think>",
        "",
        str(raw or ""),
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    if not text:
        return None

    candidates = [_strip_markdown_fence(text)]
    fence_match = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    for match in re.finditer(r"{", text):
        balanced = _find_balanced_container(text, match.start())
        if balanced:
            candidates.append(balanced)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            repaired = re.sub(r",\s*([}\]])", r"\1", str(candidate))
            if repaired == candidate:
                continue
            try:
                payload = json.loads(repaired)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        if not isinstance(payload, dict):
            continue
        normalized_keys = {_json_key_id(key) for key in payload}
        if normalized_keys & _CONTEXT_UPDATE_JSON_KEY_IDS:
            return payload
    return None


def _coerce_json_items(value: Any, item_keys: set) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        lowered = {str(key).casefold() for key in value}
        if lowered & item_keys:
            return [value]
        items: List[Any] = []
        for key, nested_value in value.items():
            if isinstance(nested_value, dict):
                merged = dict(nested_value)
                merged.setdefault("name", key)
                items.append(merged)
            else:
                items.append({"name": key, "value": nested_value})
        return items
    if isinstance(value, str):
        return [value]
    return []


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _json_action(value: Dict[str, Any]) -> str:
    action = _json_text(_json_get(value, "action")).casefold()
    if action:
        return action
    if _json_get(value, "delete") is True:
        return "delete"
    return ""


def _is_json_delete(value: Dict[str, Any]) -> bool:
    action = _json_action(value)
    if action in {"delete", "remove", "deleted"}:
        return True
    raw_value = _json_text(_json_get(value, "value", "target"))
    return raw_value.casefold() == "delete"


def _is_json_correction(value: Dict[str, Any]) -> bool:
    return _json_action(value) in {"correction", "correct", "corrected"}


def _preformatted_update_line(value: str) -> str:
    line = value.strip()
    if not line:
        return ""
    return line if line.startswith("-") else f"- {line}"


def _json_character_lines(value: Any) -> str:
    item_keys = {
        "name",
        "canonical_name",
        "character",
        "gender",
        "role",
        "description",
        "details",
        "summary",
        "value",
        "action",
    }
    lines: List[str] = []
    for item in _coerce_json_items(value, item_keys):
        if isinstance(item, str):
            line = _preformatted_update_line(item)
            if line:
                lines.append(line)
            continue
        if not isinstance(item, dict):
            continue
        name = _json_text(
            _json_get(item, "name", "canonical_name", "canonical", "character")
        )
        if not name:
            continue
        if _is_json_delete(item):
            lines.append(f"- {name}: DELETE")
            continue
        value_text = _json_text(
            _json_get(item, "value", "description", "details", "summary")
        )
        role = _json_text(_json_get(item, "role"))
        if role and value_text and role.casefold() not in value_text.casefold():
            value_text = f"{role}, {value_text}"
        elif role and not value_text:
            value_text = role
        gender = _json_text(_json_get(item, "gender"))
        if gender and value_text:
            value_text = f"{gender}, {value_text}"
        elif gender and not value_text:
            value_text = gender
        if not value_text:
            continue
        if _is_json_correction(item) and not value_text.casefold().startswith(
            "correction:"
        ):
            value_text = f"CORRECTION: [{value_text.strip('[]')}]"
        lines.append(f"- {name}: {value_text}")
    return "\n".join(lines)


def _json_alias_lines(value: Any) -> str:
    item_keys = {
        "alias",
        "source",
        "label",
        "title",
        "canonical",
        "canonical_name",
        "target",
        "action",
    }
    lines: List[str] = []
    for item in _coerce_json_items(value, item_keys):
        if isinstance(item, str):
            line = _preformatted_update_line(item)
            if line:
                lines.append(line)
            continue
        if not isinstance(item, dict):
            continue
        alias = _json_text(
            _json_get(item, "alias", "source", "label", "title", "name")
        )
        canonical = _json_text(
            _json_get(item, "canonical", "canonical_name", "target", "value")
        )
        if not alias:
            continue
        lines.append(f"- {alias}: {'DELETE' if _is_json_delete(item) else canonical}")
    return "\n".join(line for line in lines if not line.endswith(": "))


def _json_glossary_lines(value: Any) -> str:
    item_keys = {
        "source",
        "source_term",
        "term",
        "target",
        "target_term",
        "translation",
        "recommended_target_term",
        "action",
    }
    lines: List[str] = []
    for item in _coerce_json_items(value, item_keys):
        if isinstance(item, str):
            line = _preformatted_update_line(item)
            if line:
                lines.append(line)
            continue
        if not isinstance(item, dict):
            continue
        source = _json_text(
            _json_get(item, "source", "source_term", "term", "name")
        )
        target = _json_text(
            _json_get(
                item,
                "target",
                "target_term",
                "recommended_target_term",
                "translation",
                "value",
            )
        )
        if not source:
            continue
        lines.append(f"- {source}: {'DELETE' if _is_json_delete(item) else target}")
    return "\n".join(line for line in lines if not line.endswith(": "))


def _json_dynamic_line(item: Any, relationship: bool = False) -> str:
    if isinstance(item, str):
        return _preformatted_update_line(item)
    if not isinstance(item, dict):
        return ""
    line = _json_text(_json_get(item, "line", "text"))
    if line:
        return _preformatted_update_line(line)

    if relationship:
        left = _json_text(
            _json_get(item, "character_a", "left", "speaker", "source")
        )
        right = _json_text(
            _json_get(item, "character_b", "right", "addressee", "target")
        )
        details = _json_text(
            _json_get(item, "relationship", "details", "description", "value")
        )
        arrow = _json_text(_json_get(item, "arrow")) or "↔"
    else:
        left = _json_text(_json_get(item, "speaker", "character_a"))
        right = _json_text(_json_get(item, "addressee", "character_b"))
        parts = []
        source_form = _json_text(_json_get(item, "source_form"))
        target_form = _json_text(
            _json_get(item, "target_form", "recommended_target_form")
        )
        register = _json_text(_json_get(item, "register", "reason"))
        social_basis = _json_text(
            _json_get(
                item,
                "social_basis",
                "basis",
                "relationship_basis",
                "social_context",
            )
        )
        scope = _json_text(
            _json_get(item, "scope", "usage_scope", "addressing_scope")
        )
        details = _json_text(_json_get(item, "details", "value"))
        if source_form:
            parts.append(f'source form "{source_form}"')
        if target_form:
            parts.append(f'target-language form "{target_form}"')
        if register:
            parts.append(register)
        if social_basis:
            parts.append(social_basis)
        if scope:
            parts.append(scope)
        if details:
            parts.append(details)
        details = " | ".join(parts)
        arrow = "→"

    if not left or not right:
        return ""
    if _is_json_delete(item):
        details = "DELETE"
    if not details:
        return ""
    return f"- {left} {arrow} {right}: {details}"


def _json_dynamic_state(value: Any) -> str:
    if isinstance(value, str):
        return _strip_markdown_fence(value)
    if isinstance(value, list):
        relationship_lines = [
            _json_dynamic_line(item, relationship=True)
            for item in value
        ]
        relationship_lines = [line for line in relationship_lines if line]
        return _format_dynamic_sections("", "\n".join(relationship_lines))
    if not isinstance(value, dict):
        return ""

    addressing_items = _json_get(
        value,
        "current_addressing_forms",
        "addressing_forms",
        "addressing",
    ) or []
    relationship_items = _json_get(
        value,
        "relationship_evolution",
        "relationships",
        "relationship_changes",
    ) or []
    addressing_lines = [
        _json_dynamic_line(item, relationship=False)
        for item in _coerce_json_items(addressing_items, {"speaker", "addressee"})
    ]
    relationship_lines = [
        _json_dynamic_line(item, relationship=True)
        for item in _coerce_json_items(
            relationship_items,
            {"character_a", "character_b", "relationship"},
        )
    ]
    addressing = "\n".join(line for line in addressing_lines if line)
    relationships = "\n".join(line for line in relationship_lines if line)
    if not addressing and not relationships:
        return ""
    return _format_dynamic_sections(addressing, relationships)


def _context_update_sections_from_json(
    payload: Dict[str, Any],
) -> Tuple[str, str, str, str, str]:
    characters = _json_character_lines(
        _json_get(payload, "new_characters", "characters")
    )
    aliases = _json_alias_lines(
        _json_get(payload, "identity_links", "aliases")
    )
    glossary = _json_glossary_lines(
        _json_get(payload, "new_glossary", "glossary")
    )
    dynamic = _json_dynamic_state(_json_get(payload, "dynamic_state"))
    dialogue = _json_get(payload, "dialogue_attribution")
    dialogue_raw = (
        json.dumps(dialogue, ensure_ascii=False)
        if dialogue is not None
        else ""
    )
    return characters, aliases, glossary, dynamic, dialogue_raw


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
    source_context: str = "",
    dialogue_turns: Optional[List[Dict[str, str]]] = None,
    current_dialogue_state: Optional[Dict[str, str]] = None,
    dialogue_attribution_sink: Optional[Dict[str, Any]] = None,
    selective_context_view: bool = True,
    context_view_max_tokens: Optional[int] = None,
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
    source_analysis_text = _compose_source_analysis_text(
        source_context,
        source_chunk,
    )
    prompt_reference_text = "\n\n".join(
        part
        for part in (
            source_analysis_text,
            translated_chunk or "",
        )
        if part
    )
    prompt_global_lore, prompt_dynamic_state = render_novel_context_update_view(
        current_global_lore,
        current_dynamic_state,
        reference_text=prompt_reference_text,
        max_tokens=context_view_max_tokens,
        selective=selective_context_view,
    )
    prompt_values = {
        "current_global_lore": prompt_global_lore,
        "current_dynamic_state": prompt_dynamic_state,
        "source_language": source_language,
        "target_language": target_language,
        "source_context": source_context or "(none)",
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
                    empty_dialogue_attribution()
                )
            return current_global_lore, current_dynamic_state, []
            
        content = response.content.strip()
        
        # Parse blocks
        new_chars = ""
        new_aliases = ""
        new_glossary = ""
        new_dynamic = current_dynamic_state
        dialogue_raw = ""
        
        import re
        chars_match = re.search(
            r'\[NEW_CHARACTERS\]\s*(.*?)\s*'
            r'(?=\[IDENTITY_LINKS\]|\[NEW_GLOSSARY\]|\[DYNAMIC_STATE\]|$)',
            content,
            re.DOTALL,
        )
        aliases_match = re.search(
            r'\[IDENTITY_LINKS\]\s*(.*?)\s*'
            r'(?=\[NEW_GLOSSARY\]|\[DYNAMIC_STATE\]|\[NEW_CHARACTERS\]|$)',
            content,
            re.DOTALL,
        )
        glossary_match = re.search(
            r'\[NEW_GLOSSARY\]\s*(.*?)\s*'
            r'(?=\[DYNAMIC_STATE\]|\[IDENTITY_LINKS\]|\[NEW_CHARACTERS\]|$)',
            content,
            re.DOTALL,
        )
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
        if aliases_match:
            new_aliases = aliases_match.group(1).strip()
        if glossary_match:
            new_glossary = glossary_match.group(1).strip()
        if dynamic_match:
            new_dynamic = dynamic_match.group(1).strip()
        if dialogue_match:
            dialogue_raw = dialogue_match.group(1).strip()

        source_backstop_gender_updates = infer_source_gender_updates(
            source_analysis_text,
            current_global_lore,
            new_chars,
        )
        if source_backstop_gender_updates:
            new_chars = "\n".join(
                part
                for part in (new_chars, source_backstop_gender_updates)
                if part.strip()
            )
        source_backstop_aliases = infer_source_identity_links(
            source_analysis_text,
            current_global_lore,
            new_chars,
        )
        if source_backstop_aliases:
            new_aliases = "\n".join(
                part
                for part in (new_aliases, source_backstop_aliases)
                if part.strip()
            )
        trusted_dynamic_aliases = infer_dynamic_address_identity_links(
            current_dynamic_state,
            current_global_lore,
        )
        if new_dynamic.strip():
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
        else:
            new_dynamic = current_dynamic_state
            
        updated_global_lore, change_logs = merge_new_lore(
            current_global_lore,
            new_chars,
            new_glossary,
            new_aliases,
            source_analysis_text,
            trusted_dynamic_aliases,
        )
        new_dynamic = merge_dynamic_state(
            current_dynamic_state,
            new_dynamic,
            _character_alias_map(updated_global_lore),
        )

        # LLM consolidation pass: periodically deduplicate character descriptions
        # that the deterministic merge layer missed (semantically similar rephrasing).
        consolidation_interval = _consolidation_interval()
        is_last_chunk = (total_chunks > 0 and chunk_index == total_chunks)
        if (
            consolidation_interval > 0
            and chunk_index > 0
            and (chunk_index % consolidation_interval == 0 or is_last_chunk)
        ):
            logger.info(
                f"[Novel Context] Running consolidation pass at chunk {chunk_index}/{total_chunks}."
            )
            consolidated_lore, consolidation_logs = await consolidate_context_lore(
                llm_client=llm_client,
                model_name=model_name,
                global_lore=updated_global_lore,
            )
            updated_global_lore = consolidated_lore
            change_logs.extend(consolidation_logs)

        # Track relationship logs by comparing old and new dynamic state
        # (This is secondary logging for the terminal, just printing line updates)
        if new_dynamic != current_dynamic_state:
            # We can log that relationship state changed
            change_logs.append("[Novel Context] Dynamic state updated.")

        if dialogue_attribution_sink is not None:
            dialogue_attribution_sink.clear()
            dialogue_attribution_sink.update(
                parse_dialogue_attribution(
                    dialogue_raw,
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
                empty_dialogue_attribution()
            )
        return current_global_lore, current_dynamic_state, []
