"""Prompt projection for accepted relationship graph state."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.persistence.database import Database
from src.utils.relationship_schema import RelationshipProjection, clean_relationship_text
from src.utils.text_matching import active_label_matches_name, reference_mentions_label


def _name_is_active(name: str, active_names: Iterable[str], language: str) -> bool:
    return any(
        active_label_matches_name(active, name, language)
        or active_label_matches_name(name, active, language)
        for active in active_names
        if active
    )


def _edge_relevance(
    edge: Dict[str, Any],
    active_names: List[str],
    reference_text: str,
    language: str,
) -> tuple[int, str]:
    source = str(edge.get("source_name") or "")
    target = str(edge.get("target_name") or "")
    if active_names and (
        _name_is_active(source, active_names, language)
        or _name_is_active(target, active_names, language)
    ):
        return 3, "active_character"
    if reference_text and (
        reference_mentions_label(source, reference_text, language)
        or reference_mentions_label(target, reference_text, language)
    ):
        return 2, "source_reference"
    if edge.get("is_locked"):
        return 1, "locked_fact"
    if edge.get("scope") == "durable":
        return 0, "durable_fallback"
    return -1, "situational_not_active"


def _render_edge(edge: Dict[str, Any]) -> str:
    arrow = "<->" if edge.get("direction") == "symmetric" else "->"
    metadata = [str(edge.get("relationship_type") or "associated")]
    hierarchy = str(edge.get("hierarchy") or "unknown")
    if hierarchy != "unknown":
        metadata.append(f"hierarchy={hierarchy}")
    scope = str(edge.get("scope") or "durable")
    metadata.append(f"scope={scope}")
    register = str(edge.get("register") or "neutral")
    if register != "neutral":
        metadata.append(f"register={register}")
    intimacy = str(edge.get("intimacy") or "unknown")
    if intimacy != "unknown":
        metadata.append(f"intimacy={intimacy}")
    details = clean_relationship_text(edge.get("details"), 220)
    suffix = f"; {details}" if details else ""
    return (
        f"- {edge.get('source_name')} {arrow} {edge.get('target_name')}: "
        f"{', '.join(metadata)}{suffix}"
    )


def _render_addressing_rule(rule: Dict[str, Any]) -> str:
    parts = [
        f"self={clean_relationship_text(rule.get('self_pronoun'), 80)}",
        f"target={clean_relationship_text(rule.get('target_pronoun'), 80)}",
    ]
    vocative = clean_relationship_text(rule.get("vocative"), 100)
    if vocative:
        parts.append(f"vocative={vocative}")
    register = clean_relationship_text(rule.get("register"), 80)
    if register:
        parts.append(f"register={register}")
    return (
        f"- {rule.get('speaker_name')} -> {rule.get('addressee_name')}: "
        f"{', '.join(parts)}"
    )


def build_relationship_projection(
    translation_id: str,
    db: Optional[Database],
    *,
    reference_text: str = "",
    active_character_names: Optional[Iterable[str]] = None,
    target_language: str = "",
    max_edges: int = 16,
    max_addressing_rules: int = 10,
) -> RelationshipProjection:
    """Select and render scene-relevant accepted graph facts for a prompt."""

    if not translation_id or db is None:
        return RelationshipProjection(fallback_reasons=["relationship_database_unavailable"])
    active_names = [str(name) for name in active_character_names or [] if name]
    edges = db.get_relationship_edges(translation_id, statuses=["accepted"])
    ranked = []
    for edge in edges:
        score, reason = _edge_relevance(
            edge,
            active_names,
            reference_text,
            target_language,
        )
        if score < 0:
            continue
        ranked.append((score, int(bool(edge.get("is_locked"))), edge, reason))
    ranked.sort(
        key=lambda item: (
            item[0], item[1], item[2].get("confidence", 0.0),
            int(item[2].get("last_chunk_index", 0)),
        ),
        reverse=True,
    )
    selected = ranked[:max(0, int(max_edges))]

    addressing_rules = []
    for rule in db.get_addressing_rules(translation_id):
        if active_names and not (
            _name_is_active(str(rule.get("speaker_name") or ""), active_names, target_language)
            or _name_is_active(str(rule.get("addressee_name") or ""), active_names, target_language)
        ):
            continue
        if not active_names and reference_text and not (
            reference_mentions_label(str(rule.get("speaker_name") or ""), reference_text, target_language)
            or reference_mentions_label(str(rule.get("addressee_name") or ""), reference_text, target_language)
        ):
            continue
        addressing_rules.append(rule)
    addressing_rules = addressing_rules[:max(0, int(max_addressing_rules))]

    object_conflicts = [
        conflict for conflict in db.get_relationship_conflicts(
            translation_id,
            status="open",
            limit=100,
        )
        if conflict.get("validator") == "entity_type"
    ]
    if active_names or reference_text:
        object_conflicts = [
            conflict for conflict in object_conflicts
            if _name_is_active(str(conflict.get("source_name") or ""), active_names, target_language)
            or _name_is_active(str(conflict.get("target_name") or ""), active_names, target_language)
            or reference_mentions_label(
                str(conflict.get("source_name") or ""),
                reference_text,
                target_language,
            )
            or reference_mentions_label(
                str(conflict.get("target_name") or ""),
                reference_text,
                target_language,
            )
        ]
    object_conflicts = object_conflicts[:5]

    lines: List[str] = []
    if selected:
        lines.append("Accepted relationship facts:")
        lines.extend(_render_edge(edge) for _score, _locked, edge, _reason in selected)
    if addressing_rules:
        lines.extend(["", "Active directed addressing projections:"])
        lines.extend(_render_addressing_rule(rule) for rule in addressing_rules)
    if object_conflicts:
        lines.extend(["", "Forbidden identity conflations:"])
        for conflict in object_conflicts:
            lines.append(
                f"- Do not treat {conflict.get('source_name')} as a character identity "
                f"for {conflict.get('target_name')}."
            )

    reasons = [
        {
            "entry_type": "relationship_edge",
            "label": f"{edge.get('source_name')} -> {edge.get('target_name')}",
            "reason": reason,
        }
        for _score, _locked, edge, reason in selected
    ]
    reasons.extend({
        "entry_type": "addressing_edge",
        "label": f"{rule.get('speaker_name')} -> {rule.get('addressee_name')}",
        "reason": "active_pair" if active_names else "source_reference",
    } for rule in addressing_rules)
    fallback_reasons = []
    if not lines:
        fallback_reasons.append(
            "relationship_graph_empty" if not edges else "no_scene_relevant_relationships"
        )
    return RelationshipProjection(
        prompt_text="\n".join(lines).strip(),
        selection_reasons=reasons,
        fallback_reasons=fallback_reasons,
        edge_count=len(selected),
        addressing_count=len(addressing_rules),
        conflict_count=len(object_conflicts),
    )
