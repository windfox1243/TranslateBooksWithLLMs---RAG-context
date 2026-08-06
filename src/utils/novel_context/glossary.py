"""Glossary normalization, name-translation maps and global lore normalization."""
from __future__ import annotations

import re
from typing import Optional, List, Dict, Any, Tuple, Callable

from .constants import (
    ALIASES_SECTION,
    CHARACTERS_SECTION,
    GLOSSARY_SECTION,
    NAME_MAP_SECTION,
    _CJK_NON_NAME_ADDRESS_LABELS,
    _UNSET_NAME_TRANSLATION,
)
from .characters import (
    _alias_entries_to_map,
    _canonical_alias_entries,
    _canonical_display_name,
    _character_alias_keys,
    _character_narrative_role_alias_keys,
    _character_self_role_title_keys,
    _deduplicate_character_entries,
    _find_lore_section,
    _format_character_line,
    _infer_singular_plural_alias_entries,
    _infer_unique_short_name_alias_entries,
    _is_cjk_generic_role_only_name,
    _is_disposable_unnamed_character,
    _is_invalid_context_key,
    _is_non_character_group_entry,
    _is_non_character_metadata_or_item_entry,
    _is_non_character_work_entry,
    _is_quarantined_character_entry,
    _is_unstable_physical_character_entry,
    _parse_alias_entries,
    _parse_bullet_entries,
    _plain_key,
    _replace_lore_section,
    _retain_renderable_aliases,
    _strip_balanced_brackets,
)

def _has_non_ascii_target_script(term: str) -> bool:
    clean = _strip_balanced_brackets(term)
    if _has_cjk_or_hangul_name_script(clean):
        return False
    return bool(re.search(r"[^\x00-\x7F]", clean))
def _is_inverted_target_to_source_glossary_pair(name: str, value: str) -> bool:
    clean_name = _strip_balanced_brackets(name)
    clean_val = _strip_balanced_brackets(value)
    if not clean_name or not clean_val:
        return False
    name_has_target = _has_non_ascii_target_script(clean_name)
    val_has_target = _has_non_ascii_target_script(clean_val)
    if name_has_target and not val_has_target:
        return True
    return False
def _normalize_glossary_entries(entries: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    ordered: Dict[str, Tuple[str, str]] = {}
    stored_pairs: set[Tuple[str, str]] = set()
    for raw_name, raw_value in entries:
        clean_name = _strip_balanced_brackets(raw_name)
        clean_value = _strip_balanced_brackets(raw_value)
        if not clean_name or _is_invalid_context_key(clean_name):
            continue

        if _is_inverted_target_to_source_glossary_pair(clean_name, clean_value):
            clean_name, clean_value = clean_value, clean_name

        key = _plain_key(clean_name)
        val_key = _plain_key(clean_value)

        if (val_key, key) in stored_pairs:
            continue

        if key in ordered:
            old_name, old_val = ordered[key]
            if old_name.casefold() == clean_name.casefold() and old_name[0:1].isupper():
                clean_name = old_name
            if old_val.casefold() == clean_value.casefold() and old_val[0:1].isupper():
                clean_value = old_val

        ordered[key] = (clean_name, clean_value)
        stored_pairs.add((key, val_key))
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
def _normalized_character_alias_map(global_lore: str) -> Dict[str, str]:
    """Build aliases from already-normalized lore without re-deduplicating it."""
    bounds = _find_lore_section(global_lore, CHARACTERS_SECTION)
    if not bounds:
        return {}

    alias_bounds = _find_lore_section(global_lore, ALIASES_SECTION)
    aliases = _alias_entries_to_map(
        _parse_alias_entries(
            global_lore[alias_bounds[1]:alias_bounds[2]]
        )
        if alias_bounds
        else []
    )

    _, body_start, body_end = bounds
    for name, _ in _parse_bullet_entries(global_lore[body_start:body_end]):
        canonical = _canonical_display_name(name)
        if (
            _is_invalid_context_key(canonical)
            or _is_quarantined_character_entry(canonical)
        ):
            continue
        for alias_key in _character_alias_keys(canonical):
            aliases.setdefault(alias_key, canonical)

    return {
        alias: target
        for alias, target in aliases.items()
        if not _is_quarantined_character_entry(target)
    }
