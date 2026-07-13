"""Sync relationship graph state with markdown context and translation pipelines."""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional

from src.persistence.database import Database
from src.utils.progress_logging import emit_progress_log
from src.utils.relationship_projection import build_relationship_projection
from src.utils.relationship_reasoning_engine import (
    RelationshipReasoningEngine,
    relationship_candidate_needs_llm_judge,
    relationship_support_for_addressing,
)
from src.utils.relationship_schema import (
    RelationshipCandidate,
    RelationshipMergeDecision,
    classify_entity_type,
    normalize_relationship_name,
)


_RELATION_LINE_RE = re.compile(
    r"^\s*-\s*(?P<source>.+?)\s*(?P<arrow>\u2194|\u2192|\u2190|<->|->|<-)\s*"
    r"(?P<target>.+?)\s*:\s*(?P<details>.*?)\s*$"
)
_SITUATIONAL_CUES = (
    "acting",
    "disguise",
    "dream",
    "hallucination",
    "pretend",
    "quoted history",
    "roleplay",
    "temporary",
)


def resolve_relationship_reasoning_mode(prompt_options: Optional[Dict[str, Any]]) -> str:
    """Resolve off/shadow/project mode, defaulting live jobs to project mode."""

    value = (prompt_options or {}).get("use_relationship_reasoning", "project")
    if value is True:
        return "project"
    if value is False or value is None:
        return "off"
    mode = str(value).strip().casefold()
    return mode if mode in {"off", "shadow", "project"} else "project"


def _active_names_from_options(prompt_options: Optional[Dict[str, Any]]) -> List[str]:
    options = prompt_options or {}
    active = [str(item) for item in options.get("active_character_names") or [] if item]
    attribution = options.get("dialogue_attribution") or {}
    if isinstance(attribution, dict):
        state_after = attribution.get("state_after") or {}
        if isinstance(state_after, dict):
            for key in ("speaker", "addressee"):
                value = state_after.get(key)
                if value and str(value).casefold() not in {"unknown", "none", "null"}:
                    active.append(str(value))
    return active


def _context_parts(context_or_dynamic_state: str) -> tuple[str, str, str]:
    from src.utils.novel_context import (
        _split_dynamic_sections,
        extract_dynamic_state_from_text,
        extract_global_lore,
    )

    text = str(context_or_dynamic_state or "")
    global_lore = extract_global_lore(text)
    dynamic_state = extract_dynamic_state_from_text(text) or text
    addressing, relationships, _has_sections = _split_dynamic_sections(dynamic_state)
    return global_lore, addressing, relationships


def _register_context_nodes(
    engine: RelationshipReasoningEngine,
    translation_id: str,
    global_lore: str,
    *,
    chunk_index: int,
    log_callback=None,
) -> List[str]:
    """Import canonical characters, aliases, and classified glossary entities."""

    if not global_lore.strip():
        return []
    from src.utils.novel_context import (
        ALIASES_SECTION,
        GLOSSARY_SECTION,
        _character_profile_map,
        _find_lore_section,
        _parse_alias_entries,
        _parse_bullet_entries,
    )

    profiles = _character_profile_map(global_lore)
    known_names = []
    for profile in profiles.values():
        name = str(profile.get("name") or "").strip()
        if not name:
            continue
        known_names.append(name)
        engine.register_node(
            translation_id,
            name,
            entity_type="character",
            gender=str(profile.get("gender") or "unknown"),
        )

    object_terms: Dict[str, str] = {}
    glossary_bounds = _find_lore_section(global_lore, GLOSSARY_SECTION)
    if glossary_bounds:
        for term, details in _parse_bullet_entries(
            global_lore[glossary_bounds[1]:glossary_bounds[2]]
        ):
            entity_type = classify_entity_type("", f"{term} {details}")
            if entity_type == "character":
                continue
            object_terms[normalize_relationship_name(term)] = entity_type
            engine.register_node(
                translation_id,
                term,
                entity_type=entity_type,
            )

    alias_bounds = _find_lore_section(global_lore, ALIASES_SECTION)
    if alias_bounds:
        for alias, canonical in _parse_alias_entries(
            global_lore[alias_bounds[1]:alias_bounds[2]]
        ):
            alias_type = object_terms.get(
                normalize_relationship_name(alias),
                "character",
            )
            engine.register_alias(
                translation_id,
                canonical,
                alias,
                alias_entity_type=alias_type,
                chunk_index=chunk_index,
                log_callback=log_callback,
            )
    return known_names


def parse_markdown_relationship_candidates(
    relationship_markdown: str,
    *,
    provenance: str = "relationship_markdown",
    confidence: float = 0.95,
) -> List[RelationshipCandidate]:
    """Parse relationship-evolution markdown into typed candidates."""

    candidates = []
    for raw_line in str(relationship_markdown or "").splitlines():
        match = _RELATION_LINE_RE.match(raw_line)
        if not match:
            continue
        source = match.group("source").strip()
        target = match.group("target").strip()
        arrow = match.group("arrow")
        details = match.group("details").strip()
        if arrow in {"\u2190", "<-"}:
            source, target = target, source
            arrow = "\u2192"
        scope = (
            "situational"
            if any(cue in details.casefold() for cue in _SITUATIONAL_CUES)
            else "durable"
        )
        candidates.append(RelationshipCandidate(
            source=source,
            target=target,
            relationship_type=details,
            direction="symmetric" if arrow in {"\u2194", "<->"} else "directed",
            scope=scope,
            evidence_quote="" if provenance.startswith("llm") or provenance == "context_update" else raw_line.strip(),
            confidence=confidence,
            provenance=provenance,
            details=details,
            parser_status="markdown",
            action="delete" if details.casefold() == "delete" else "upsert",
        ))
    return candidates


def sync_db_addressing_to_relationship_graph(
    *,
    translation_id: str,
    db: Optional[Database],
    chunk_index: int = 0,
    log_callback=None,
) -> List[RelationshipMergeDecision]:
    """Compatibility no-op: addressing has its own directed projection table."""

    # Older releases mirrored addressing rules into the semantic graph. Those
    # edges polluted inverse-role validation. Existing mirrors remain readable
    # for audit compatibility but every semantic consumer filters them out.
    return []


def quarantine_incompatible_addressing_rules(
    *,
    translation_id: str,
    db: Optional[Database],
    target_language: str,
    chunk_index: int = 0,
    log_callback=None,
) -> int:
    """Remove unlocked Vietnamese address rules that contradict graph hierarchy."""

    if not translation_id or db is None:
        return 0
    from src.utils.language_profiles import get_language_profile

    if get_language_profile(target_language).addressing_family != "vietnamese":
        return 0
    from src.utils.context_merge_engine import _vi_pair_direction

    quarantined = 0
    for rule in list(db.get_addressing_rules(translation_id)):
        if str(rule.get("scope") or "durable").casefold() == "situational":
            continue
        support = relationship_support_for_addressing(
            db,
            translation_id,
            str(rule.get("speaker_name") or ""),
            str(rule.get("addressee_name") or ""),
        )
        expected = {
            "source_senior": "senior_to_junior",
            "source_junior": "junior_to_senior",
            "peer": "peer",
        }.get(support.get("hierarchy"), "")
        actual = _vi_pair_direction(
            str(rule.get("self_pronoun") or ""),
            str(rule.get("target_pronoun") or ""),
        )
        if not expected or not actual or expected == actual:
            continue
        pair = f"{rule.get('speaker_name')} -> {rule.get('addressee_name')}"
        if rule.get("is_locked"):
            emit_progress_log(
                log_callback,
                "relationship_addressing_conflict_locked",
                f"Locked addressing rule {pair} overrides conflicting relationship hierarchy.",
                level="warning",
                layer="relationship_reasoning",
                data={
                    "pair": pair,
                    "expected_hierarchy": expected,
                    "actual_hierarchy": actual,
                },
            )
            continue
        if not db.delete_addressing_rule(
            translation_id,
            str(rule.get("speaker_name") or ""),
            str(rule.get("addressee_name") or ""),
        ):
            continue
        quarantined += 1
        db.add_context_audit_log(
            translation_id=translation_id,
            chunk_index=chunk_index,
            speaker_name=str(rule.get("speaker_name") or ""),
            addressee_name=str(rule.get("addressee_name") or ""),
            old_state=rule,
            new_state={
                "status": "quarantined",
                "reason": "relationship graph hierarchy contradiction",
                "expected_hierarchy": expected,
                "actual_hierarchy": actual,
            },
            trigger_source="relationship_graph:quarantine",
            confidence=float(rule.get("confidence") or 1.0),
        )
        for edge in db.get_relationship_edges_for_pair(
            translation_id,
            normalize_relationship_name(rule.get("speaker_name")),
            normalize_relationship_name(rule.get("addressee_name")),
            statuses=["accepted"],
        ):
            if edge.get("relationship_type") == "addressing":
                db.set_relationship_edge_status(
                    translation_id,
                    edge["id"],
                    "quarantined",
                )
        emit_progress_log(
            log_callback,
            "relationship_addressing_quarantined",
            f"Quarantined addressing rule {pair}: relationship hierarchy expects {expected}.",
            level="warning",
            layer="relationship_reasoning",
            data={
                "pair": pair,
                "expected_hierarchy": expected,
                "actual_hierarchy": actual,
                "chunk_index": chunk_index,
            },
        )
    return quarantined


def sync_markdown_relationships_to_db(
    *,
    translation_id: str,
    db: Optional[Database],
    context_or_dynamic_state: str,
    target_language: str = "",
    chunk_index: int = 0,
    trigger_source: str = "relationship_markdown",
    log_callback=None,
) -> List[RelationshipMergeDecision]:
    """Import human-editable markdown relationship state through the engine."""

    if not translation_id or db is None:
        return []
    global_lore, _addressing, relationships = _context_parts(context_or_dynamic_state)
    engine = RelationshipReasoningEngine(db=db)
    known_names = _register_context_nodes(
        engine,
        translation_id,
        global_lore,
        chunk_index=chunk_index,
        log_callback=log_callback,
    )
    candidates = parse_markdown_relationship_candidates(
        relationships,
        provenance=trigger_source,
        confidence=1.0 if trigger_source in {"manual", "rest_api", "user_manual"} else 0.95,
    )
    decisions = engine.merge_candidates(
        translation_id,
        chunk_index,
        candidates,
        known_character_names=known_names,
        language=target_language,
        log_callback=log_callback,
    )
    quarantine_incompatible_addressing_rules(
        translation_id=translation_id,
        db=db,
        target_language=target_language,
        chunk_index=chunk_index,
        log_callback=log_callback,
    )
    sync_db_addressing_to_relationship_graph(
        translation_id=translation_id,
        db=db,
        chunk_index=chunk_index,
        log_callback=log_callback,
    )
    if candidates:
        emit_progress_log(
            log_callback,
            "relationship_markdown_synced",
            f"Relationship graph imported {sum(d.status == 'accepted' for d in decisions)}/"
            f"{len(candidates)} markdown candidate(s).",
            layer="relationship_reasoning",
            data={
                "mode": "import",
                "candidate_count": len(candidates),
                "accepted_count": sum(d.status == "accepted" for d in decisions),
                "quarantined_count": sum(d.status == "quarantined" for d in decisions),
                "rejected_count": sum(d.status == "rejected" for d in decisions),
            },
        )
    return decisions


def sync_context_update_relationships_to_db(
    *,
    translation_id: str,
    db: Optional[Database],
    updated_context_or_dynamic_state: str,
    source_text: str,
    candidates: Optional[Iterable[Any]] = None,
    parser_status: str = "absent",
    target_language: str = "",
    chunk_index: int = 0,
    active_character_names: Optional[Iterable[str]] = None,
    log_callback=None,
) -> List[RelationshipMergeDecision]:
    """Commit one successful chunk's relationship candidates to graph state."""

    if not translation_id or db is None:
        return []
    global_lore, _addressing, relationships = _context_parts(
        updated_context_or_dynamic_state
    )
    engine = RelationshipReasoningEngine(db=db)
    known_names = _register_context_nodes(
        engine,
        translation_id,
        global_lore,
        chunk_index=chunk_index,
        log_callback=log_callback,
    )
    parsed_candidates: List[RelationshipCandidate] = []
    for item in candidates or []:
        if isinstance(item, RelationshipCandidate):
            parsed_candidates.append(item)
        elif isinstance(item, dict):
            parsed = RelationshipCandidate.from_dict(
                item,
                default_provenance="llm_context",
                parser_status=parser_status,
            )
            if parsed:
                parsed_candidates.append(parsed)
    if not parsed_candidates and parser_status in {"absent", "legacy", "markdown"}:
        parsed_candidates = parse_markdown_relationship_candidates(
            relationships,
            provenance="llm_legacy_context",
            confidence=0.78,
        )
    if parser_status in {"invalid_json", "invalid_contract"}:
        engine.record_parse_failure(
            translation_id,
            chunk_index,
            parser_status,
            log_callback=log_callback,
        )
    decisions = engine.merge_candidates(
        translation_id,
        chunk_index,
        parsed_candidates,
        source_text=source_text,
        known_character_names=known_names,
        active_character_names=active_character_names,
        language=target_language,
        log_callback=log_callback,
    )
    quarantine_incompatible_addressing_rules(
        translation_id=translation_id,
        db=db,
        target_language=target_language,
        chunk_index=chunk_index,
        log_callback=log_callback,
    )
    sync_db_addressing_to_relationship_graph(
        translation_id=translation_id,
        db=db,
        chunk_index=chunk_index,
        log_callback=log_callback,
    )
    return decisions


async def judge_ambiguous_relationship_candidates(
    *,
    llm_client: Any,
    candidates: Iterable[Any],
    source_text: str,
    model_name: str,
    enabled: bool,
    locked_facts: Optional[List[Dict[str, Any]]] = None,
    log_callback=None,
) -> List[Dict[str, Any]]:
    """Classify all ambiguous candidates with at most one LLM call per chunk."""

    normalized: List[RelationshipCandidate] = []
    for item in candidates or []:
        candidate = item if isinstance(item, RelationshipCandidate) else RelationshipCandidate.from_dict(item)
        if candidate:
            normalized.append(candidate)
    if not enabled or not llm_client:
        return [candidate.to_dict() for candidate in normalized]

    ambiguous_indexes = [
        index for index, candidate in enumerate(normalized)
        if relationship_candidate_needs_llm_judge(candidate)
    ]
    if not ambiguous_indexes:
        return [candidate.to_dict() for candidate in normalized]

    contract = {
        "decisions": [{
            "index": 0,
            "decision": "support | reject | uncertain",
            "confidence": 0.0,
            "reason": "brief evidence classification",
        }],
    }
    candidate_payload = [
        {"index": index, "candidate": normalized[index].to_dict()}
        for index in ambiguous_indexes
    ]
    prompt = (
        "Classify every proposed relationship against the source excerpt. "
        "Do not infer from genre stereotypes, names, or world knowledge. Do not "
        "change endpoints, relationship types, direction, or evidence. Return "
        "only one JSON object matching this contract:\n"
        f"{json.dumps(contract)}\n\n"
        f"Candidates: {json.dumps(candidate_payload, ensure_ascii=False)}\n"
        f"Locked facts: {json.dumps(locked_facts or [], ensure_ascii=False)}\n"
        f"Source excerpt: {str(source_text or '')[:6000]}"
    )
    system_prompt = (
        "You are a conservative relationship evidence classifier. Locked facts "
        "are immutable. Use uncertain whenever the named participants, direction, "
        "or social relationship are not explicit in the supplied source."
    )
    decisions: Dict[int, Dict[str, Any]] = {}
    parse_status = "json"
    try:
        if hasattr(llm_client, "generate_async"):
            response = llm_client.generate_async(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model_name,
                temperature=0.0,
            )
        else:
            response = llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model_name,
                temperature=0.0,
            )
        if inspect.isawaitable(response):
            response = await response
        raw = str(getattr(response, "content", "") or "").strip()
        fence = re.search(
            r"```(?:json)?\s*(.*?)```",
            raw,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fence:
            raw = fence.group(1).strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        payload = json.loads(match.group(0) if match else raw)
        rows = payload.get("decisions") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("Missing decisions list")
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                index = int(row.get("index"))
            except (TypeError, ValueError):
                continue
            if index not in ambiguous_indexes or index in decisions:
                continue
            decision = str(row.get("decision") or "").strip().casefold()
            if decision not in {"support", "reject", "uncertain"}:
                decision = "uncertain"
            try:
                confidence = max(0.0, min(1.0, float(row.get("confidence") or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            decisions[index] = {
                "decision": decision,
                "confidence": confidence,
                "reason": str(row.get("reason") or "").strip()[:300],
            }
        if any(index not in decisions for index in ambiguous_indexes):
            parse_status = "incomplete_json"
    except Exception as exc:
        parse_status = f"call_or_parse_failed:{type(exc).__name__}"

    judged: List[RelationshipCandidate] = []
    for index, candidate in enumerate(normalized):
        if index not in ambiguous_indexes:
            judged.append(candidate)
            continue
        decision = decisions.get(index, {
            "decision": "uncertain",
            "confidence": 0.0,
            "reason": "Batched judge did not return a valid decision.",
        })
        updated = replace(
            candidate,
            confidence=(
                max(candidate.confidence, float(decision["confidence"]))
                if decision["decision"] == "support"
                else candidate.confidence
            ),
            judge_decision=str(decision["decision"]),
            judge_reason=str(decision["reason"]),
        )
        judged.append(updated)
        emit_progress_log(
            log_callback,
            "relationship_llm_judge",
            f"Relationship judge classified {candidate.source} -> {candidate.target} "
            f"as {updated.judge_decision or 'uncertain'}.",
            layer="relationship_reasoning",
            data={
                "source": candidate.source,
                "target": candidate.target,
                "decision": updated.judge_decision or "uncertain",
                "parse_status": parse_status,
            },
        )
    return [candidate.to_dict() for candidate in judged]


def build_relationship_prompt_context(
    *,
    translation_id: str,
    db: Optional[Database],
    target_language: str = "",
    prompt_options: Optional[Dict[str, Any]] = None,
    reference_text: str = "",
    active_character_names: Optional[Iterable[str]] = None,
    log_callback=None,
) -> str:
    """Build project-mode graph context; shadow mode never changes prompts."""

    mode = resolve_relationship_reasoning_mode(prompt_options)
    if mode != "project" or not translation_id or db is None:
        return ""
    active = list(active_character_names or []) or _active_names_from_options(prompt_options)
    projection = build_relationship_projection(
        translation_id,
        db,
        reference_text=reference_text,
        active_character_names=active,
        target_language=target_language,
        max_edges=int((prompt_options or {}).get("relationship_prompt_max_edges", 16)),
    )
    if prompt_options is not None:
        prompt_options["relationship_projection_metadata"] = {
            "mode": mode,
            "contract_version": "1.0",
            "edge_count": projection.edge_count,
            "addressing_count": projection.addressing_count,
            "conflict_count": projection.conflict_count,
            "selection_reasons": projection.selection_reasons,
            "fallback_reasons": projection.fallback_reasons,
        }
    if projection.prompt_text:
        emit_progress_log(
            log_callback,
            "relationship_projected",
            "Structured relationship graph injected into the prompt.",
            layer="relationship_reasoning",
            data={
                "mode": mode,
                "edge_count": projection.edge_count,
                "addressing_count": projection.addressing_count,
                "conflict_count": projection.conflict_count,
                "selection_reasons": projection.selection_reasons,
            },
        )
    return projection.prompt_text


def attach_relationship_context_to_prompt_options(
    prompt_options: Optional[Dict[str, Any]],
    *,
    translation_id: str,
    db: Optional[Database],
    target_language: str = "",
    reference_text: str = "",
    log_callback=None,
) -> Dict[str, Any]:
    """Attach project-mode graph context to a mutable adapter option mapping."""

    options = prompt_options if isinstance(prompt_options, dict) else {}
    options.setdefault("use_relationship_reasoning", "project")
    relationship_context = build_relationship_prompt_context(
        translation_id=translation_id,
        db=db,
        target_language=target_language,
        prompt_options=options,
        reference_text=reference_text,
        log_callback=log_callback,
    )
    if relationship_context:
        options["relationship_context"] = relationship_context
    else:
        options.pop("relationship_context", None)
    return options


def export_relationship_graph_to_markdown(
    translation_id: str,
    db: Optional[Database],
) -> str:
    """Export accepted non-addressing graph edges as relationship markdown."""

    if not translation_id or db is None:
        return ""
    lines = []
    for edge in db.get_relationship_edges(translation_id, statuses=["accepted"]):
        if edge.get("relationship_type") == "addressing":
            continue
        arrow = "\u2194" if edge.get("direction") == "symmetric" else "\u2192"
        details = str(edge.get("details") or edge.get("relationship_type") or "associated").strip()
        lines.append(
            f"- {edge.get('source_name')} {arrow} {edge.get('target_name')}: {details}"
        )
    return "\n".join(lines)


def apply_relationship_graph_to_context(
    context_content: str,
    translation_id: str,
    db: Optional[Database],
    fallback_context: str = "",
) -> str:
    """Return context content with accepted graph relationships exported."""

    global_lore, addressing, current_relationships = _context_parts(
        context_content
    )
    unmanaged_lines = [
        line.rstrip()
        for line in current_relationships.splitlines()
        if line.strip() and not _RELATION_LINE_RE.match(line)
    ]
    exported = export_relationship_graph_to_markdown(translation_id, db)
    if not exported:
        if not fallback_context:
            return context_content
        _fallback_global, _fallback_addressing, exported = _context_parts(
            fallback_context
        )
    relationship_lines = [line for line in exported.splitlines() if line.strip()]
    existing_keys = {line.casefold().strip() for line in relationship_lines}
    relationship_lines.extend(
        line for line in unmanaged_lines
        if line.casefold().strip() not in existing_keys
    )
    from src.utils.novel_context import (
        _format_dynamic_sections,
        build_novel_context,
    )

    return build_novel_context(
        global_lore,
        _format_dynamic_sections(addressing, "\n".join(relationship_lines)),
    )


def apply_relationship_graph_to_session(
    context_session: Any,
    translation_id: str,
    db: Optional[Database],
    fallback_context: str = "",
) -> bool:
    """Refresh a NovelContextSession from accepted graph relationship state."""

    if not context_session or not translation_id or db is None:
        return False
    updated = apply_relationship_graph_to_context(
        context_session.content,
        translation_id,
        db,
        fallback_context=fallback_context,
    )
    if updated == context_session.content:
        return False
    from src.utils.novel_context import extract_dynamic_state_from_text

    context_session.dynamic_state = extract_dynamic_state_from_text(updated) or ""
    context_session.sync_prompt()
    return True
