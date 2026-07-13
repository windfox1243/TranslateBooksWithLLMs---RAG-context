"""Prompt projection for accepted relationship graph state."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.persistence.database import Database
from src.utils.language_profiles import get_language_profile
from src.utils.relationship_reasoning_engine import (
    migrate_relationship_reasoning_v2,
    relationship_support_for_addressing,
)
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


def _vietnamese_seniority_hint(support: Dict[str, Any]) -> str:
    hierarchy = str(support.get("hierarchy") or "unknown")
    source_gender = str(support.get("source_gender") or "unknown").casefold()
    target_gender = str(support.get("target_gender") or "unknown").casefold()
    if hierarchy == "source_junior":
        target_form = "anh" if target_gender == "male" else "chị" if target_gender == "female" else "a senior kinship pronoun"
        return f"default self=em, target={target_form}"
    if hierarchy == "source_senior":
        self_form = "anh" if source_gender == "male" else "chị" if source_gender == "female" else "a senior kinship pronoun"
        return f"default self={self_form}, target=em"
    return ""


def _render_seniority_path(
    source: str,
    target: str,
    support: Dict[str, Any],
    target_language: str,
) -> str:
    path_types = [
        str(item.get("relationship_type") or "associated")
        for item in support.get("path") or []
    ]
    metadata = [
        f"hierarchy={support.get('hierarchy')}",
        f"confidence={float(support.get('confidence') or 0.0):.2f}",
        f"path={' + '.join(path_types)}",
    ]
    if get_language_profile(target_language).addressing_family == "vietnamese":
        hint = _vietnamese_seniority_hint(support)
        if hint:
            metadata.append(hint)
    return f"- {source} -> {target}: {', '.join(metadata)}"


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
    migrate_relationship_reasoning_v2(db, translation_id)
    active_names = [str(name) for name in active_character_names or [] if name]
    # Addressing edges mirror the directed-addressing table. Render only the
    # validated active rules below so a quarantined legacy mirror cannot leak
    # back into prompts as an accepted relationship fact.
    edges = [
        edge for edge in db.get_relationship_edges(
            translation_id, statuses=["accepted"],
        )
        if str(edge.get("relationship_type") or "").casefold() != "addressing"
    ]
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

    relevant_names: List[str] = []
    for name in active_names:
        node = db.get_relationship_node_by_name(translation_id, name)
        if node and node.get("entity_type") == "character":
            relevant_names.append(str(node.get("canonical_name") or name))
    if not relevant_names:
        for _score, _locked, edge, _reason in selected:
            for key in ("source_name", "target_name"):
                name = str(edge.get(key) or "")
                if name and reference_mentions_label(name, reference_text, target_language):
                    relevant_names.append(name)
    relevant_names = list(dict.fromkeys(relevant_names))[:8]
    derived_seniority = []
    for source in relevant_names:
        for target in relevant_names:
            if source == target:
                continue
            support = relationship_support_for_addressing(
                db,
                translation_id,
                source,
                target,
            )
            if (
                support.get("derived")
                and not support.get("conflict")
                and support.get("hierarchy") in {"source_senior", "source_junior"}
                and float(support.get("confidence") or 0.0) >= 0.72
            ):
                derived_seniority.append((source, target, support))

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
    if derived_seniority:
        lines.extend(["", "Derived social seniority (use unless explicit scene evidence overrides it):"])
        lines.extend(
            _render_seniority_path(source, target, support, target_language)
            for source, target, support in derived_seniority
        )
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
    reasons.extend({
        "entry_type": "derived_seniority",
        "label": f"{source} -> {target}",
        "reason": str(support.get("basis") or "derived_relationship_chain"),
    } for source, target, support in derived_seniority)
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
