"""Infer identity links and gender updates proven by the source text."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .characters import (
    _SOURCE_IDENTITY_ROLE_KEYS,
    _canonical_display_name,
    _canonical_gender,
    _clean_inline_text,
    _current_reincarnated_form_gender,
    _find_lore_section,
    _gender_from_body_word,
    _infer_gender_from_character_details,
    _infer_gender_reference_to_character,
    _is_descriptive_role_name,
    _is_disposable_unnamed_character,
    _is_invalid_context_key,
    _is_non_character_group_entry,
    _is_non_character_metadata_or_item_entry,
    _is_non_character_work_entry,
    _is_unstable_physical_character_entry,
    _name_reference_pattern,
    _normalize_character_value,
    _parse_bullet_entries,
    _plain_key,
    _reference_text_mentions_label,
    _role_title_key_from_name,
    _split_gender_and_details,
    _strip_balanced_brackets,
)
from .constants import (
    _RELATIONSHIP_OBJECT_PRONOUN_VERBS,
    _ROMANTIC_RELATION_PATTERN,
    _SPECIFIC_GENDER_LABELS,
    CHARACTERS_SECTION,
)


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

    alias_tokens = {_plain_key(w) for w in alias.split() if len(w) > 1}
    target_tokens = {_plain_key(w) for w in candidate_keys[target_key].split() if len(w) > 1}
    shared_tokens = alias_tokens & target_tokens
    if shared_tokens:
        matching_targets = [
            ck for ck, cname in candidate_keys.items()
            if shared_tokens & {_plain_key(w) for w in cname.split() if len(w) > 1}
        ]
        if len(matching_targets) == 1 and matching_targets[0] == target_key:
            if _reference_text_mentions_label(alias, text):
                return True, "source uses the alias label with the target character's name"
            return False, "alias contains the target name but is absent from the source chunk"

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
