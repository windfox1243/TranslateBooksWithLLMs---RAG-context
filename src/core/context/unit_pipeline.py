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
    global_lore: str = "",
    reference_text: str = "",
    addressing: str = "",
    relationships: str = "",
    narrator: str = "",
    nearby_source: Iterable[str] = (),
) -> str:
    """Render the typed prompt-context bundle without accumulated candidates."""

    entities: List[str] = []
    glossary: List[str] = []
    if global_lore.strip() and reference_text.strip():
        from src.utils.novel_context import (
            ALIASES_SECTION,
            CHARACTERS_SECTION,
            GLOSSARY_SECTION,
            _parse_bullet_entries,
            _section_body,
            render_novel_context_for_prompt,
        )

        selected = render_novel_context_for_prompt(
            global_lore,
            reference_text=reference_text,
            max_tokens=0,
            selective=True,
            include_gender_roster=False,
        )
        for section in (CHARACTERS_SECTION, ALIASES_SECTION):
            entities.extend(
                f"- {name}: {value}"
                for name, value in _parse_bullet_entries(
                    _section_body(selected, section)
                )
            )
        glossary.extend(
            f"- {name}: {value}"
            for name, value in _parse_bullet_entries(
                _section_body(selected, GLOSSARY_SECTION)
            )
        )

    return PromptContextBundle(
        entities=entities,
        glossary=glossary,
        addressing=[addressing] if addressing else [],
        relationships=[relationships] if relationships else [],
        narrator=narrator,
        nearby_source="\n".join(str(item) for item in nearby_source if item),
    ).render()


def prepare_unit_prompt_options(
    base_options: Optional[Dict[str, Any]], *, unit_index: int, phase: str,
    file_type: str, source_text: str, target_language: str,
    dialogue_attribution: Optional[Dict[str, Any]] = None,
    chapter_index: Any = None, scene_key: str = "",
    nearby_source: Iterable[str] = (),
) -> Dict[str, Any]:
    """Build the common phase-aware options contract for one source unit."""
    options = dict(base_options or {})
    options.update({
        "chunk_index": int(unit_index),
        "editor_phase": str(phase),
        "file_type": str(file_type or options.get("file_type") or ""),
        "chapter_index": chapter_index,
        "scene_key": str(scene_key or ""),
        "dialogue_attribution": dict(dialogue_attribution or {}),
    })
    translation_id = str(options.get("translation_id") or "")
    db = options.get("_checkpoint_db")
    narrator = ""
    if translation_id and db is not None:
        from src.utils.narrator_voice import build_narrator_voice_context
        narrator = build_narrator_voice_context(
            translation_id, db, chunk_index=int(unit_index),
            target_language=target_language,
        )
        if narrator:
            options["narrative_voice_context"] = narrator
    novel_context = str(options.get("novel_context") or "")
    global_lore = novel_context
    if novel_context:
        from src.utils.novel_context import extract_global_lore
        global_lore = extract_global_lore(novel_context)
    options["prompt_context_bundle"] = build_unit_prompt_context(
        global_lore=global_lore,
        reference_text=source_text,
        addressing=str(options.get("selected_addressing_context") or ""),
        relationships=str(options.get("selected_relationship_context") or ""),
        narrator=narrator,
        nearby_source=nearby_source,
    )
    return options
