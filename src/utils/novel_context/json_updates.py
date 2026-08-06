"""Parse structured JSON context updates into context document lines."""
from __future__ import annotations

import json
import re
from typing import Optional, List, Dict, Any, Tuple, Callable

from .dynamic_state import _format_dynamic_sections

_CONTEXT_UPDATE_JSON_KEYS = {
    "characters",
    "new_characters",
    "identity_links",
    "aliases",
    "new_glossary",
    "glossary",
    "dynamic_state",
    "dialogue_attribution",
    "relationship_candidates",
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
        self_reference = _json_text(
            _json_get(
                item,
                "self_reference",
                "speaker_self_reference",
                "target_self_reference",
            )
        )
        addressee_form = _json_text(
            _json_get(
                item,
                "addressee_form",
                "target_addressee_form",
                "reference_form",
            )
        )
        second_person_pronoun = _json_text(
            _json_get(
                item,
                "second_person_pronoun",
                "target_second_person_pronoun",
                "you_pronoun",
            )
        )
        vocative_nickname = _json_text(
            _json_get(
                item,
                "vocative_nickname",
                "vocative_form",
                "address_form",
                "target_address_form",
                "nickname",
                "target_vocative",
            )
        )
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
        if not target_form and (
            self_reference
            or addressee_form
            or second_person_pronoun
            or vocative_nickname
        ):
            paired = []
            if self_reference:
                paired.append(f"self-reference: {self_reference}")
            if second_person_pronoun:
                paired.append(f"second-person pronoun: {second_person_pronoun}")
            if vocative_nickname:
                paired.append(f"vocative/address form: {vocative_nickname}")
            if addressee_form:
                paired.append(f"addressee form: {addressee_form}")
            target_form = "; ".join(paired)
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
