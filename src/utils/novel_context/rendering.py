"""Render novel context into prompt blocks and human-readable views."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .characters import (
    _canonical_gender,
    _clean_inline_text,
    _entry_mentions_reference,
    _find_lore_section,
    _format_character_line,
    _is_quarantined_character_entry,
    _parse_alias_entries,
    _parse_bullet_entries,
    _plain_key,
    _reference_mentions_latin_name_part,
    _split_gender_and_details,
    _text_mentions,
)
from .constants import (
    _SPECIFIC_GENDER_LABELS,
    ADDRESSING_SECTION,
    ALIASES_SECTION,
    CHARACTERS_SECTION,
    DYNAMIC_STATE_END,
    DYNAMIC_STATE_START,
    GLOSSARY_SECTION,
    RELATIONSHIP_SECTION,
    logger,
)
from .document import extract_dynamic_state_from_text, extract_global_lore
from .dynamic_state import _DYNAMIC_RELATION_PATTERN, _split_dynamic_sections
from .merge import build_novel_context, normalize_novel_context_content
from .retrieval import cap_roster, rank, score_entry, split_budget


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
def _default_prompt_max_tokens() -> int:
    try:
        from src import config as _config

        return max(0, int(getattr(_config, "NOVEL_CONTEXT_PROMPT_MAX_TOKENS", 3000)))
    except Exception:
        return 3000
def _context_prompt_budget_chars(max_tokens: Optional[int]) -> int:
    """Characters the rendered context may occupy (0 = no ceiling).

    A caller that says nothing gets the configured default rather than an
    unlimited prompt. Ranking and the dynamic-state reserve only bite once
    there is a ceiling, and a context with no ceiling grows with the book --
    which is the state this used to ship in. Passing 0 explicitly still means
    unlimited, so the ceiling can be turned off deliberately.
    """
    if max_tokens is None:
        max_tokens = _default_prompt_max_tokens()
    try:
        token_budget = int(max_tokens)
    except (TypeError, ValueError):
        token_budget = _default_prompt_max_tokens()
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
    if max_chars <= 0:
        lines.append(line)
        return True
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
) -> int:
    """Append what fits, returning how many entries had to be left out."""
    if not entries:
        return 0
    if not _append_line_with_budget(lines, "", max_chars, reserved_chars):
        return len(entries)
    if not _append_line_with_budget(lines, section_name, max_chars, reserved_chars):
        return len(entries)
    dropped = 0
    for entry in entries:
        if not _append_line_with_budget(lines, entry, max_chars, reserved_chars):
            dropped += 1
    return dropped
def _dynamic_line_name_keys(lines: List[str]) -> set:
    """Names on both sides of the dynamic lines this chunk already selected.

    A character can be central to a scene without the chunk writing their name
    -- dialogue does this constantly. If an addressing or relationship line for
    them was selected, they are on stage, and their recorded gender is exactly
    what the model needs to pick pronouns for the next "she said".
    """
    keys = set()
    for line in lines:
        relation = _DYNAMIC_RELATION_PATTERN.match(line.strip())
        if not relation:
            continue
        keys.add(_plain_key(relation.group("left").strip()))
        keys.add(_plain_key(relation.group("right").strip()))
    keys.discard("")
    return keys
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
    active_speaker: Optional[str] = None,
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
        if not max_chars or len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip()

    has_first_person = bool(re.search(r"\b(i|me|my|myself|mine)\b", reference_text, re.IGNORECASE))
    selected_character_keys: set[str] = set()
    active_speaker_key = (
        _plain_key(active_speaker)
        if active_speaker and _plain_key(active_speaker) not in {"unknown", "unspecified", "none", "null", "n/a", "na", "?"}
        else None
    )

    for name, value in character_entries:
        name_key = _plain_key(name)
        is_active_pov = bool(active_speaker_key and name_key == active_speaker_key)
        is_protagonist = not active_speaker_key and any(cue in value.casefold() for cue in ("protagonist", "main character", "narrator", "reincarnated individual"))
        if (
            _entry_mentions_reference(name, value, reference_text)
            or _reference_mentions_latin_name_part(name, reference_text)
            or (has_first_person and (is_active_pov or is_protagonist))
        ):
            selected_character_keys.add(name_key)

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
    dynamic_reference_keys = set(selected_character_keys) | {
        _plain_key(name)
        for name in dynamic_reference_names
    }

    pov_speaker_key = active_speaker_key
    if not pov_speaker_key:
        for name, value in character_entries:
            if any(cue in value.casefold() for cue in ("protagonist", "main character", "narrator", "reincarnated individual")):
                pov_speaker_key = _plain_key(name)
                break

    def dynamic_party_is_referenced(party: str) -> bool:
        party = party.strip()
        return (
            _plain_key(party) in dynamic_reference_keys
            or _text_mentions(party, reference_text)
            or _reference_mentions_latin_name_part(party, reference_text)
        )

    def dynamic_line_address_form_is_referenced(line: str) -> bool:
        for quoted in re.findall(r'"([^"\n]{1,80})"', line):
            if _text_mentions(quoted, reference_text):
                return True
        return False

    def split_dynamic_lines(text: str) -> Tuple[List[str], List[str]]:
        selected_lines: List[str] = []
        remaining_lines: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            relation = _DYNAMIC_RELATION_PATTERN.match(line)
            if relation:
                left, right = relation.group("left").strip(), relation.group("right").strip()
                left_key, right_key = _plain_key(left), _plain_key(right)
                left_in_text = dynamic_party_is_referenced(left)
                right_in_text = dynamic_party_is_referenced(right)

                is_pov_involved = left_key == pov_speaker_key or right_key == pov_speaker_key

                should_select = (
                    (left_in_text and right_in_text)
                    or dynamic_line_address_form_is_referenced(line)
                    or (
                        has_first_person
                        and pov_speaker_key
                        and is_pov_involved
                        and (left_in_text or right_in_text)
                    )
                )
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

    # Everything below competes for one budget, so order it by how hard this
    # chunk points at it. Without this the cast discovered first wins whatever
    # the chapter is actually about.
    on_stage_keys = _dynamic_line_name_keys(
        selected_addressing + selected_relationships
    )

    def character_score(entry: Tuple[str, str]) -> int:
        name, value = entry
        return score_entry(
            name,
            value,
            reference_text,
            on_stage_keys=on_stage_keys,
            is_pov=bool(has_first_person and _plain_key(name) == pov_speaker_key),
        )

    selected_characters = rank(selected_characters, character_score)
    selected_aliases = rank(
        selected_aliases,
        lambda entry: score_entry(entry[0], entry[1], reference_text, on_stage_keys=on_stage_keys),
    )
    selected_glossary = rank(
        selected_glossary,
        lambda entry: score_entry(entry[0], entry[1], reference_text),
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
    roster_cut = 0
    if include_gender_roster:
        # The roster names every character in the book, so it is the one part
        # of the prompt that grows with the book rather than with the chunk.
        roster_entries = rank(
            [
                (name, value)
                for name, value in character_entries
                if _plain_key(name) not in selected_character_keys
                and _compact_character_gender_line(name, value)
            ],
            character_score,
        )
        pinned_gender_lines, roster_cut = cap_roster(
            [
                _compact_character_gender_line(name, value)
                for name, value in roster_entries
            ]
        )
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
    # The lore is rendered first and the dynamic state last, so without a floor
    # a long cast simply eats the budget and the addressing rules -- the reason
    # this whole system exists for a language with gendered address -- never
    # reach the prompt at all. Whatever the lore leaves rolls over.
    lore_budget, _dynamic_reserve = split_budget(max_chars)
    dropped: Dict[str, int] = {"gender roster": roster_cut}
    rendered_lines: List[str] = ["# GLOBAL LORE"]
    dropped["characters"] = _append_section_with_budget(
        rendered_lines,
        CHARACTERS_SECTION,
        selected_character_lines + pinned_gender_lines,
        lore_budget,
        reserved,
    )
    dropped["aliases"] = _append_section_with_budget(
        rendered_lines,
        ALIASES_SECTION,
        selected_alias_lines,
        lore_budget,
        reserved,
    )
    dropped["glossary"] = _append_section_with_budget(
        rendered_lines,
        GLOSSARY_SECTION,
        selected_glossary_lines,
        lore_budget,
        reserved,
    )

    rendered_lines.extend(["", DYNAMIC_STATE_START, "# DYNAMIC RELATIONSHIP STATE"])
    dropped["addressing"] = _append_section_with_budget(
        rendered_lines,
        ADDRESSING_SECTION,
        selected_addressing,
        max_chars,
        len(DYNAMIC_STATE_END),
    )
    dropped["relationships"] = _append_section_with_budget(
        rendered_lines,
        RELATIONSHIP_SECTION,
        selected_relationships,
        max_chars,
        len(DYNAMIC_STATE_END),
    )
    rendered_lines.append(DYNAMIC_STATE_END)
    _report_dropped(dropped)
    return "\n".join(rendered_lines).strip()
def _report_dropped(dropped: Dict[str, int]) -> None:
    """Say what the budget cut. Silent truncation reads as a complete prompt."""
    cut = {name: count for name, count in dropped.items() if count > 0}
    if cut:
        logger.info(
            "Novel context prompt budget left out: %s.",
            ", ".join(f"{count} {name}" for name, count in sorted(cut.items())),
        )
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
