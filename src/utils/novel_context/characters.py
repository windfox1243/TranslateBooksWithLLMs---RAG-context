"""Character entry semantics: classification, gender inference and entry merging.

What is left here really is mutually recursive. A character value is a gender
followed by a list of facts, and `_normalize_character_value` is the hub every
other concern goes through: classification normalises a value before judging
it, gender inference reads and rewrites the same string, and deduplication
consults both. Splitting those apart would only move the cycle into the import
graph.

The two layers that were *not* part of that cycle have been lifted out, into
`name_keys` (naming, keys and name comparison) and `character_facts` (the fact
list a value carries). Neither reaches back into this module."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Moved out of this module; re-exported so that every existing
# `from .characters import ...` keeps resolving.
from .character_facts import (  # noqa: E402,F401
    _DETAIL_STOP_WORDS,
    _compact_reincarnation_facts,
    _compact_subordinate_facts,
    _compact_unique_role_facts,
    _detail_is_redundant,
    _detail_key,
    _detail_tokens,
    _is_context_evidence_fact,
    _merge_character_details,
    _normalize_unique_role_fact,
    _split_monarch_fact,
    _strip_low_value_fact_fragments,
)
from .constants import (
    _DIRECT_GENDER_WORDS,
    _EXPLICIT_NPC_MARKERS,
    _GENDER_LABELS,
    _GENDERED_ROMANTIC_RELATION_LABELS,
    _GROUP_ENTITY_WORDS,
    _INCIDENTAL_CHARACTER_MARKERS,
    _KINSHIP_GENDERS,
    _KINSHIP_WORDS,
    _NAME_TITLES,
    _NON_CHARACTER_ITEM_DETAIL_PATTERNS,
    _NON_CHARACTER_METADATA_DETAIL_PATTERNS,
    _NON_CHARACTER_METADATA_NAMES,
    _ROMANTIC_RELATION_PATTERN,
    _SHORT_NAME_EVIDENCE_STOPWORDS,
    _SPECIFIC_GENDER_LABELS,
    _UNIQUE_ROLE_TITLES,
    _WORK_ENTITY_NON_PERSON_ROLES,
    _WORK_ENTITY_WORDS,
    CHARACTERS_SECTION,
    GLOSSARY_SECTION,
)
from .name_keys import (  # noqa: E402,F401
    _LATIN_BOUNDARY_CHARS,
    _LATIN_NAME_PART_STOPWORDS,
    _UNSTABLE_PHYSICAL_WORDS,
    _canonical_display_name,
    _character_alias_keys,
    _character_names_match,
    _character_narrative_role_alias_keys,
    _clean_inline_text,
    _compact_name_key,
    _find_lore_section,
    _full_name_contains_short_alias,
    _generic_role_base_key,
    _has_recurring_character_marker,
    _is_cjk_generic_role_only_name,
    _is_descriptive_role_name,
    _is_distinctive_physical_descriptor,
    _is_english_generic_role_only_name,
    _is_invalid_context_key,
    _is_numbered_generic_role_name,
    _is_quarantined_character_entry,
    _is_role_only_name,
    _is_transferable_role_only_name,
    _is_unstable_identity_alias,
    _label_has_cjk_or_hangul,
    _loose_romanized_token_match,
    _monarch_role,
    _name_reference_pattern,
    _name_specificity,
    _narrative_work_keys,
    _normalize_relative_name_key,
    _parse_bullet_entries,
    _parse_kinship_name,
    _plain_key,
    _preferred_character_name,
    _reference_mentions_latin_name_part,
    _reference_text_mentions_label,
    _relation_label_key,
    _replace_lore_section,
    _role_title_key_from_name,
    _role_title_keys_from_fact,
    _short_name_alias_key,
    _similar_full_names_match,
    _simple_plural_name_keys_match,
    _simple_singular_name_key,
    _singularize_simple_english_token,
    _small_typo_distance,
    _strip_address_suffix_key,
    _strip_balanced_brackets,
    _strip_leading_article,
    _strip_name_title,
    _strip_trailing_qualifier,
    _text_mentions,
)


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
def _glossary_entry_is_item_or_terminology(name: str, value: str) -> bool:
    """Return whether a glossary entry is object/skill terminology, not a person."""
    text = _plain_key(f"{name} {value}")
    if not text:
        return False
    return bool(
        re.search(
            r"\b(?:sword|weapon|blade|artifact|artefact|relic|item|equipment|"
            r"spell|skill|ability|technique|title|quest|stat|system|buff|"
            r"debuff|magic\s+circle|formation|rune|sigil)\b",
            text,
        )
    )
def _identity_endpoint_non_character_reason(
    label: str,
    global_lore: str,
    new_characters: str = "",
    new_glossary: str = "",
) -> str:
    """Return a reason when an identity-link endpoint is known non-character terminology."""
    label_key = _plain_key(label)
    if not label_key:
        return "empty identity label"

    character_entries: List[Tuple[str, str]] = []
    bounds = _find_lore_section(global_lore, CHARACTERS_SECTION)
    if bounds:
        _, body_start, body_end = bounds
        character_entries.extend(_parse_bullet_entries(global_lore[body_start:body_end]))
    character_entries.extend(_parse_bullet_entries(new_characters))
    for name, value in character_entries:
        if _plain_key(name) != label_key:
            continue
        if (
            _is_non_character_work_entry(name, value)
            or _is_non_character_group_entry(name, value)
            or _is_non_character_metadata_or_item_entry(name, value)
        ):
            return "identity label is classified as non-character terminology"

    glossary_entries: List[Tuple[str, str]] = []
    glossary_bounds = _find_lore_section(global_lore, GLOSSARY_SECTION)
    if glossary_bounds:
        _, body_start, body_end = glossary_bounds
        glossary_entries.extend(_parse_bullet_entries(global_lore[body_start:body_end]))
    glossary_entries.extend(_parse_bullet_entries(new_glossary))
    for source, target in glossary_entries:
        if _plain_key(source) != label_key and _plain_key(target) != label_key:
            continue
        if _glossary_entry_is_item_or_terminology(source, target):
            return "identity label is glossary/object terminology"

    return ""
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
    canonical_name = _canonical_display_name(name)
    if not canonical_name:
        return ""
    if canonical_name.casefold() not in text.casefold():
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
def _character_gender_map(global_lore: str) -> Dict[str, str]:
    return {
        key: str(profile["gender"])
        for key, profile in _character_profile_map(global_lore).items()
        if str(profile.get("gender", "")).casefold() in _SPECIFIC_GENDER_LABELS
    }
def _character_profile_map(global_lore: str) -> Dict[str, Dict[str, str]]:
    bounds = _find_lore_section(global_lore, CHARACTERS_SECTION)
    if not bounds:
        return {}
    _, body_start, body_end = bounds
    profiles: Dict[str, Dict[str, str]] = {}
    for name, value in _parse_bullet_entries(global_lore[body_start:body_end]):
        gender, _ = _split_gender_and_details(
            _normalize_character_value_for_name(name, value)
        )
        _raw_gender, details = _split_gender_and_details(value)
        gender = _canonical_gender(gender)
        profiles[_plain_key(name)] = {
            "name": _canonical_display_name(name),
            "gender": gender if gender.casefold() in _SPECIFIC_GENDER_LABELS else "",
            "details": _clean_inline_text(details),
        }
    return profiles
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
def _entry_mentions_reference(
    name: str,
    value: str,
    reference_text: str,
) -> bool:
    return _text_mentions(name, reference_text) or _text_mentions(
        value,
        reference_text,
    )
