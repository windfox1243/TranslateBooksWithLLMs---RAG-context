"""Parsing and merging of the dynamic relationship/addressing state block."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .characters import (
    _character_alias_keys,
    _clean_inline_text,
    _compact_name_key,
    _is_descriptive_role_name,
    _is_invalid_context_key,
    _is_numbered_generic_role_name,
    _is_transferable_role_only_name,
    _is_unstable_identity_alias,
    _plain_key,
    _strip_balanced_brackets,
)
from .constants import (
    _INCIDENTAL_CHARACTER_MARKERS,
    ADDRESSING_SECTION,
    RELATIONSHIP_SECTION,
)
from .glossary import character_alias_map
from .identity_links import _candidate_named_characters


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

    # Auto-format unquoted/2-part Vietnamese addressing lines into canonical 3-part format
    if "self-reference:" in details and not details.startswith('"'):
        pipe_parts = [p.strip() for p in details.split("|")]
        if len(pipe_parts) == 2:
            details_body = pipe_parts[0].strip().strip('"')
            reason = pipe_parts[1].strip()
            details = f'"{right}" | "{details_body}" | {reason}'

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
        re.fullmatch(r"[\u3400-\u9fff\uf900-\ufaff]{2,8}", compact)
        or re.fullmatch(r"[\uac00-\ud7a3]{2,6}", compact)
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
    proposed: Dict[str, Tuple[str, str]] = {}
    ambiguous: set[str] = set()
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
        label_keys = _character_alias_keys(label)
        if any(alias_key in _character_alias_keys(target) for alias_key in label_keys):
            continue
        existing_targets = {
            alias_map[alias_key]
            for alias_key in label_keys
            if alias_key in alias_map
        }
        if existing_targets and existing_targets != {target}:
            ambiguous.update(label_keys)
            continue
        for alias_key in label_keys:
            existing = proposed.get(alias_key)
            if existing and existing[1] != target:
                ambiguous.add(alias_key)
                continue
            proposed[alias_key] = (label, target)

    unique_links: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for alias_key, (label, target) in proposed.items():
        if alias_key not in ambiguous:
            unique_links[(_plain_key(label), target)] = (label, target)
    return "\n".join(
        f"- {alias}: {target}"
        for alias, target in unique_links.values()
    )
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
