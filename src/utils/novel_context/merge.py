"""Merge and normalize dynamic state into a coherent context document."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .characters import (
    _canonical_display_name,
    _character_names_match,
    _clean_inline_text,
    _current_reincarnated_form_gender,
    _find_lore_section,
    _format_character_line,
    _is_quarantined_character_entry,
    _merge_character_values,
    _name_reference_pattern,
    _normalize_character_value_for_name,
    _parse_bullet_entries,
    _plain_key,
    _reference_mentions_latin_name_part,
    _reference_text_mentions_label,
    _replace_lore_section,
    _split_gender_and_details,
)
from .constants import (
    _SPECIFIC_GENDER_LABELS,
    CHARACTERS_SECTION,
    DYNAMIC_STATE_END,
    DYNAMIC_STATE_START,
)
from .document import extract_dynamic_state_from_text, extract_global_lore
from .dynamic_state import (
    _format_dynamic_sections,
    _merge_dynamic_entries,
    _normalize_dynamic_entries,
    _normalize_relationship_notation,
    _parse_dynamic_relation,
    _split_dynamic_sections,
)
from .glossary import (
    _discarded_incidental_character_aliases,
    _normalized_character_alias_map,
    normalize_global_lore,
)
from .vietnamese import (
    _filter_vietnamese_addressing_delta,
    _is_vietnamese_target_language,
    _remove_vietnamese_addressing_mismatches,
    _repair_vietnamese_addressing_block,
    _replace_vietnamese_addressing_field,
    _seed_vietnamese_addressing_from_relationships,
    _vietnamese_addressing_field,
)


def _sanitize_vietnamese_dynamic_state(
    dynamic_state: str,
    alias_map: Dict[str, str],
    discarded_character_aliases: Optional[set[str]] = None,
    character_genders: Optional[Dict[str, str]] = None,
    character_profiles: Optional[Dict[str, Dict[str, str]]] = None,
    translated_chunk: Optional[str] = None,
    dialogue_attribution: Optional[Dict[str, Any]] = None,
    target_language: Optional[str] = None,
) -> str:
    """Remove stored addressing rows whose paired-address data conflicts (Vietnamese target language only)."""
    if target_language and not _is_vietnamese_target_language(target_language):
        return dynamic_state

    normalized = normalize_dynamic_state(
        dynamic_state,
        alias_map,
        discarded_character_aliases,
    )
    addressing, relationships, _ = _split_dynamic_sections(normalized)
    addressing = _seed_vietnamese_addressing_from_relationships(
        addressing,
        relationships,
        alias_map,
        character_profiles,
    )
    if translated_chunk:
        addressing = cross_check_addressing_with_dialogue(
            addressing_block=addressing,
            translated_chunk=translated_chunk,
            alias_map=alias_map,
            dialogue_attribution=dialogue_attribution,
            character_genders=character_genders,
        )
    filtered_addressing = _remove_vietnamese_addressing_mismatches(
        addressing,
        alias_map,
    )
    if addressing.strip() and not filtered_addressing.strip():
        filtered_addressing = addressing
    return _format_dynamic_sections(filtered_addressing, relationships)
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
def _select_relevant_character_profiles(
    reference_text: str,
    character_profiles: Optional[Dict[str, Dict[str, str]]],
    max_profiles: int = 15,
) -> Dict[str, Dict[str, str]]:
    """Select ONLY character profiles that appear in reference text (matching names, source names, or aliases)."""
    if not character_profiles or not reference_text:
        return {}

    relevant: Dict[str, Dict[str, str]] = {}

    for key, info in character_profiles.items():
        if not isinstance(info, dict):
            continue

        char_name = info.get("name", "")
        source_name = info.get("source_name") or info.get("original_name") or ""
        aliases = info.get("aliases") or info.get("alias") or []
        if isinstance(aliases, str):
            aliases = [a.strip() for a in aliases.split(",") if a.strip()]

        matched = _reference_text_mentions_label(str(key), reference_text)

        if not matched and char_name:
            matched = (
                _reference_text_mentions_label(char_name, reference_text)
                or _reference_mentions_latin_name_part(char_name, reference_text)
            )

        if not matched and source_name:
            matched = _reference_text_mentions_label(source_name, reference_text)

        if not matched and aliases:
            for alias in aliases:
                alias_str = str(alias).strip()
                if len(alias_str) >= 2 and _reference_text_mentions_label(
                    alias_str,
                    reference_text,
                ):
                    matched = True
                    break

        if matched:
            relevant[key] = info
            if len(relevant) >= max_profiles:
                break

    return relevant
def cross_check_addressing_with_dialogue(
    addressing_block: str,
    translated_chunk: str,
    alias_map: Optional[Dict[str, str]] = None,
    dialogue_attribution: Optional[Dict[str, Any]] = None,
    character_genders: Optional[Dict[str, str]] = None,
) -> str:
    """
    Cross-checks extracted addressing rules against actual spoken dialogue in translated text.
    High-Confidence Safety Gate: ONLY auto-corrects if speaker attribution confidence is high (>= 0.8)
    AND the spoken quote contains an explicit, unambiguous pronoun.
    """
    if not addressing_block or not translated_chunk:
        return addressing_block

    aliases = alias_map or {}
    from src.utils.dialogue_attribution import detect_dialogue_turns
    turns = detect_dialogue_turns(translated_chunk)
    if not turns:
        return addressing_block

    attributed_turns = (dialogue_attribution or {}).get("turns", [])
    turn_speaker_map: Dict[str, Tuple[str, str, float]] = {}
    for turn in attributed_turns:
        turn_id = turn.get("id")
        spk = turn.get("speaker", "")
        adr = turn.get("addressee", "")
        conf = float(turn.get("confidence", 0.0))
        if turn_id and spk and conf >= 0.8:
            turn_speaker_map[turn_id] = (spk, adr, conf)

    lines = addressing_block.splitlines()
    repaired_lines = []

    for raw_line in lines:
        parsed = _parse_dynamic_relation(raw_line, aliases)
        if not parsed:
            repaired_lines.append(raw_line)
            continue

        relation_key, _, details = parsed
        spk, adr, _ = relation_key
        spk_key, adr_key = _plain_key(spk), _plain_key(adr)

        self_ref = _vietnamese_addressing_field(details, "self-reference")

        matched_quote = ""
        for t in turns:
            t_id = t.get("id", "")
            if t_id in turn_speaker_map:
                turn_spk, turn_adr, conf = turn_speaker_map[t_id]
                if _plain_key(turn_spk) == spk_key and (not turn_adr or _plain_key(turn_adr) == adr_key):
                    matched_quote = t.get("cue", "")
                    break

        if matched_quote:
            cue_lower = matched_quote.lower()
            if self_ref in {"ta", "tao"} and re.search(r"\b(anh|chị|tớ|tôi|em)\b", cue_lower):
                found_self = re.search(r"\b(anh|chị|tớ|tôi|em)\b", cue_lower).group(1)
                details = _replace_vietnamese_addressing_field(details, "self-reference", found_self)
                vocative = _vietnamese_addressing_field(details, "vocative/address form") or adr
                raw_line = f"- {spk} → {adr}: \"{vocative}\" | \"{details}\""

        repaired_lines.append(raw_line)

    return "\n".join(repaired_lines)
def merge_dynamic_state(
    current_dynamic_state: str,
    proposed_dynamic_state: str,
    character_aliases: Optional[Dict[str, str]] = None,
    target_language: Optional[str] = None,
    character_genders: Optional[Dict[str, str]] = None,
    character_profiles: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """Merge durable addressing and relationship deltas by participant key.

    Omission never deletes an existing entry, so dormant relationships survive
    an arbitrary number of unrelated chunks. A response updates the matching
    directional pair in place. Deletion requires an explicit ``DELETE`` value.
    """
    aliases = character_aliases or {}
    current = normalize_dynamic_state(current_dynamic_state, aliases)
    if _is_vietnamese_target_language(target_language):
        current = _sanitize_vietnamese_dynamic_state(
            current,
            aliases,
            character_genders=character_genders,
            character_profiles=character_profiles,
        )
    if not str(proposed_dynamic_state or "").strip():
        return current

    current_addressing, current_relationships, _ = _split_dynamic_sections(
        current
    )
    proposed_addressing, proposed_relationships, proposed_has_sections = (
        _split_dynamic_sections(proposed_dynamic_state)
    )

    if proposed_has_sections:
        if _is_vietnamese_target_language(target_language):
            current_addressing = _repair_vietnamese_addressing_block(
                current_addressing,
                aliases,
                character_genders,
            )
            proposed_addressing = _filter_vietnamese_addressing_delta(
                proposed_addressing,
                aliases,
                current_addressing,
                character_genders,
            )
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

    if _is_vietnamese_target_language(target_language):
        addressing = _seed_vietnamese_addressing_from_relationships(
            addressing,
            relationships,
            aliases,
            character_profiles,
        )
        addressing = _repair_vietnamese_addressing_block(
            addressing,
            aliases,
            character_genders,
        )
        filtered_addressing = _remove_vietnamese_addressing_mismatches(
            addressing,
            aliases,
        )
        if filtered_addressing.strip() or not addressing.strip():
            addressing = filtered_addressing

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
        _normalized_character_alias_map(normalized_global),
        discarded_character_aliases,
    )
    return (
        f"{normalized_global.strip()}\n\n"
        f"{DYNAMIC_STATE_START}\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        f"{normalized_dynamic.strip()}\n"
        f"{DYNAMIC_STATE_END}"
    ).strip()
def _compose_novel_context_from_parts(global_lore: str, dynamic_state: str) -> str:
    return (
        f"{str(global_lore or '').strip()}\n\n"
        f"{DYNAMIC_STATE_START}\n"
        "# DYNAMIC RELATIONSHIP STATE\n"
        f"{str(dynamic_state or '').strip()}\n"
        f"{DYNAMIC_STATE_END}"
    ).strip()
