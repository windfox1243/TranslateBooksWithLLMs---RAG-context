"""Shared source-unit preparation used by text and XHTML pipelines."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.core.context.contracts import PromptContextBundle
from src.core.context.social_evidence import apply_social_hierarchy_evidence
from src.utils.text_matching import reference_mentions_label


def dialogue_participants(dialogue_attribution: Optional[Dict[str, Any]]) -> List[str]:
    """Return every grounded dialogue participant in stable turn order."""

    result: List[str] = []
    attribution = dialogue_attribution or {}
    for turn in attribution.get("turns") or []:
        for value in (turn.get("speaker"), turn.get("addressee")):
            name = str(value or "").strip()
            if name and name.casefold() not in {"unknown", "none", "null"}:
                result.append(name)
    return list(dict.fromkeys(result))


def relevant_character_names(
    *,
    global_lore: str,
    source_text: str,
    dialogue_attribution: Optional[Dict[str, Any]],
    source_language: str = "",
) -> List[str]:
    """Combine dialogue participants with known characters mentioned in source."""

    from src.utils.novel_context import _character_profile_map

    names = dialogue_participants(dialogue_attribution)
    for profile in _character_profile_map(global_lore).values():
        name = str(profile.get("name") or "").strip()
        if name and reference_mentions_label(name, source_text, source_language):
            names.append(name)
    return list(dict.fromkeys(names))


def commit_source_social_evidence(
    *,
    translation_id: str,
    db: Any,
    global_lore: str,
    source_text: str,
    dialogue_attribution: Optional[Dict[str, Any]],
    source_language: str,
    target_language: str,
    chunk_index: int,
    log_callback=None,
):
    """Commit deterministic source hierarchy before the current draft request."""

    from src.utils.novel_context import _character_profile_map
    from src.utils.relationship_schema import normalize_relationship_name

    profiles = _character_profile_map(global_lore)
    active = relevant_character_names(
        global_lore=global_lore,
        source_text=source_text,
        dialogue_attribution=dialogue_attribution,
        source_language=source_language,
    )
    active_keys = {
        normalize_relationship_name(name) for name in active if name
    }
    # The relationship graph is the authoritative source for gender used by
    # target-language reconciliation. Materialize only scene-relevant profiles
    # before resolving the current source unit so a newly learned gender can
    # guide the same unit's draft.
    for profile in profiles.values():
        name = str(profile.get("name") or "").strip()
        normalized = normalize_relationship_name(name)
        if name and normalized in active_keys:
            db.upsert_relationship_node(
                translation_id,
                name,
                normalized,
                entity_type="character",
                gender=str(profile.get("gender") or "unknown"),
            )
    return apply_social_hierarchy_evidence(
        translation_id=translation_id,
        db=db,
        source_text=source_text,
        dialogue_attribution=dialogue_attribution,
        source_language=source_language,
        target_language=target_language,
        chunk_index=chunk_index,
        known_character_names=active,
        log_callback=log_callback,
    )


def build_unit_prompt_context(
    *,
    addressing: str = "",
    relationships: str = "",
    narrator: str = "",
    nearby_source: Iterable[str] = (),
) -> str:
    """Render the typed prompt-context bundle without accumulated candidates."""

    return PromptContextBundle(
        addressing=[addressing] if addressing else [],
        relationships=[relationships] if relationships else [],
        narrator=narrator,
        nearby_source="\n".join(str(item) for item in nearby_source if item),
    ).render()
