"""Bridge between markdown novel context and DB-backed directed addressing."""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.persistence.database import Database
from src.utils.context_schema import AddressingUpdateDelta
from src.utils.context_merge_engine import ContextMergeEngine
from src.utils.progress_logging import emit_progress_log


_PAIR_RE = re.compile(r"^\s*-\s*(?P<speaker>.+?)\s*(?:→|->)\s*(?P<addressee>.+?)\s*:\s*(?P<details>.+?)\s*$")
_QUOTED_RE = re.compile(r'"([^"\n]{1,160})"')
_REGISTER_VALUES = {
    "neutral", "formal", "polite", "casual", "intimate", "hostile",
    "vulgar", "archaic", "familial",
}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _field(details: str, *names: str) -> str:
    for name in names:
        pattern = re.escape(name).replace(r"\ ", r"[-_\s]*")
        match = re.search(
            rf"(?:^|[;|]\s*){pattern}\s*[:=]\s*(?P<value>[^;|\n]+)",
            details,
            flags=re.IGNORECASE,
        )
        if match:
            return _clean(match.group("value")).strip(" \"'")
    return ""


def _quoted_field(details: str, *names: str) -> str:
    """Read a quoted named field without relying on quote position."""

    for name in names:
        pattern = re.escape(name).replace(r"\ ", r"[-_\s]*")
        match = re.search(
            rf"(?:^|[|;]\s*){pattern}\s*[:=]?\s*\"(?P<value>[^\"\n]*)\"",
            details,
            flags=re.IGNORECASE,
        )
        if match:
            return _clean(match.group("value"))
    return ""


def _target_form_from_details(details: str) -> str:
    quoted_explicit = _quoted_field(
        details,
        "recommended target-language form",
        "target-language form",
        "target form",
    )
    if quoted_explicit:
        return quoted_explicit
    explicit = _field(
        details,
        "recommended target-language form",
        "target-language form",
        "target form",
    )
    quoted = _QUOTED_RE.findall(details)
    if explicit and not any(
        marker in explicit.casefold()
        for marker in ("self-reference", "second-person", "vocative")
    ):
        return explicit
    if len(quoted) >= 2:
        return _clean(quoted[1])
    if quoted:
        return _clean(quoted[0])
    if explicit:
        return explicit
    return ""


def parse_markdown_addressing_lines(markdown_text: str) -> List[AddressingUpdateDelta]:
    """Parse markdown CURRENT ADDRESSING FORMS lines into directed deltas."""

    deltas: List[AddressingUpdateDelta] = []
    for raw_line in str(markdown_text or "").splitlines():
        match = _PAIR_RE.match(raw_line)
        if not match:
            continue
        speaker = _clean(match.group("speaker"))
        addressee = _clean(match.group("addressee"))
        details = match.group("details")
        target_form_details = _target_form_from_details(details)
        field_source = f"{details}; {target_form_details}"
        self_ref = _field(field_source, "self-reference", "self pronoun", "self")
        second = _field(
            field_source,
            "second-person pronoun",
            "target pronoun",
            "second pronoun",
            "target",
        )
        vocative = _field(
            field_source,
            "vocative/address form",
            "vocative",
            "address form",
        )
        if not second:
            second = target_form_details
        if not self_ref:
            self_ref = "unspecified"
        register = "neutral"
        social_basis = []
        for part in [p.strip() for p in details.split("|") if p.strip()]:
            lowered = part.casefold()
            if not any(key in lowered for key in ("source form", "target", "self-reference", "second-person", "vocative")):
                clean_part = _clean(part).strip(" \"'")
                if clean_part.casefold() in _REGISTER_VALUES:
                    register = clean_part.casefold()
                elif clean_part:
                    social_basis.extend(
                        item.strip()
                        for item in re.split(r"[,;]", clean_part)
                        if item.strip()
                    )
        source_form = _quoted_field(details, "source form")
        if not source_form:
            compact_quotes = _QUOTED_RE.findall(details)
            if (
                compact_quotes
                and "source form" not in details.casefold()
                and "target-language form" not in details.casefold()
            ):
                source_form = _clean(compact_quotes[0])
        if speaker and addressee and second:
            deltas.append(
                AddressingUpdateDelta(
                    speaker=speaker,
                    addressee=addressee,
                    self_pronoun=self_ref,
                    second_pronoun=second,
                    vocative=vocative,
                    register=register,
                    confidence=0.95,
                    evidence_quote=raw_line.strip(),
                    source_forms=(
                        [{
                            "text": source_form,
                            "usage": "direct_address",
                            "evidence_quote": raw_line.strip(),
                            "scope": "durable",
                            "confidence": 0.95,
                        }]
                        if source_form and source_form.casefold() != "unknown"
                        else []
                    ),
                    social_basis=social_basis,
                    scope="durable",
                    provenance="markdown_context",
                )
            )
    return deltas


def parse_markdown_addressing_deletes(markdown_text: str) -> List[Tuple[str, str]]:
    """Parse CURRENT ADDRESSING FORMS delete directives."""

    deletes: List[Tuple[str, str]] = []
    for raw_line in str(markdown_text or "").splitlines():
        match = _PAIR_RE.match(raw_line)
        if not match:
            continue
        if _clean(match.group("details")).casefold() != "delete":
            continue
        speaker = _clean(match.group("speaker"))
        addressee = _clean(match.group("addressee"))
        if speaker and addressee:
            deletes.append((speaker, addressee))
    return deletes


def _extract_addressing_markdown(context_or_dynamic_state: str) -> str:
    from src.utils.novel_context import (
        ADDRESSING_SECTION,
        extract_dynamic_state_from_text,
        _split_dynamic_sections,
    )

    text = textwrap.dedent(str(context_or_dynamic_state or ""))
    dynamic_state = extract_dynamic_state_from_text(text) or text
    addressing, _relationships, _has_sections = _split_dynamic_sections(dynamic_state)
    if ADDRESSING_SECTION in addressing:
        return addressing
    return addressing


def _known_character_names_from_context(context_or_dynamic_state: str) -> List[str]:
    """Return canonical character names and aliases available in a context blob."""

    try:
        from src.utils.novel_context import (
            _character_profile_map,
            character_alias_map,
            extract_global_lore,
        )

        text = textwrap.dedent(str(context_or_dynamic_state or ""))
        global_lore = extract_global_lore(text) or text
        profiles = _character_profile_map(global_lore)
        names = [str(info.get("name") or key) for key, info in profiles.items()]
        aliases = character_alias_map(global_lore)
        names.extend(aliases.keys())
        names.extend(aliases.values())
        return [name for name in names if str(name or "").strip()]
    except Exception:
        return []


def _active_names_from_dialogue(dialogue_attribution: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(dialogue_attribution, dict):
        return []
    state_after = dialogue_attribution.get("state_after") or {}
    if not isinstance(state_after, dict):
        return []
    active: List[str] = []
    for key in ("speaker", "addressee"):
        value = state_after.get(key)
        if value and str(value).casefold() not in {"unknown", "none", "null"}:
            active.append(str(value))
    return active


def sync_markdown_addressing_to_db(
    *,
    translation_id: str,
    db: Database,
    context_or_dynamic_state: str,
    target_language: str = "",
    chunk_index: int = 0,
    trigger_source: str = "markdown_context",
    log_callback=None,
    known_character_names: Optional[Iterable[str]] = None,
    active_character_names: Optional[Iterable[str]] = None,
    dialogue_attribution: Optional[Dict[str, Any]] = None,
) -> int:
    """Import markdown addressing rules into DB through the merge engine."""

    if not translation_id or db is None:
        return 0
    addressing = _extract_addressing_markdown(context_or_dynamic_state)
    deleted = 0
    for speaker, addressee in parse_markdown_addressing_deletes(addressing):
        existing = next(
            (
                rule for rule in db.get_addressing_rules(translation_id)
                if _clean(rule.get("speaker_name")).casefold() == speaker.casefold()
                and _clean(rule.get("addressee_name")).casefold() == addressee.casefold()
            ),
            None,
        )
        if existing and existing.get("is_locked"):
            if log_callback:
                log_callback(
                    "addressing_delete_rejected",
                    f"Rejected addressing delete for {speaker} -> {addressee}: locked by user.",
                )
            continue
        if db.delete_addressing_rule(translation_id, speaker, addressee):
            deleted += 1
            db.add_context_audit_log(
                translation_id=translation_id,
                chunk_index=chunk_index,
                speaker_name=speaker,
                addressee_name=addressee,
                old_state=existing,
                new_state={"status": "deleted", "reason": "delete directive"},
                trigger_source=f"{trigger_source}:delete",
                evidence_quote=f"- {speaker} -> {addressee}: DELETE",
                confidence=1.0,
            )
            if log_callback:
                log_callback(
                    "addressing_deleted",
                    f"Deleted addressing rule: {speaker} -> {addressee} [chunk {chunk_index + 1}]",
                )
    deltas = parse_markdown_addressing_lines(addressing)
    if not deltas:
        return deleted
    engine = ContextMergeEngine(db=db)
    known_names = list(known_character_names or []) or _known_character_names_from_context(
        context_or_dynamic_state
    )
    active_names = list(active_character_names or []) + _active_names_from_dialogue(
        dialogue_attribution
    )
    applied = engine.apply_batch_deltas(
        translation_id=translation_id,
        chunk_index=chunk_index,
        deltas=deltas,
        trigger_source=trigger_source,
        log_callback=log_callback,
        target_language=target_language,
        known_character_names=known_names,
        active_character_names=active_names,
        dialogue_attribution=dialogue_attribution,
    )
    emit_progress_log(
        log_callback,
        "db_addressing_imported",
        f"Imported {applied}/{len(deltas)} directed addressing rule(s) into DB.",
        layer="db_addressing",
        data={
            "candidate_count": len(deltas),
            "applied_count": applied,
            "deleted_count": deleted,
        },
    )
    return applied


def sync_context_update_addressing_to_db(
    *,
    translation_id: str,
    db: Database,
    updated_context_or_dynamic_state: str,
    target_language: str = "",
    chunk_index: int = 0,
    log_callback=None,
    known_character_names: Optional[Iterable[str]] = None,
    active_character_names: Optional[Iterable[str]] = None,
    dialogue_attribution: Optional[Dict[str, Any]] = None,
    candidates: Optional[Iterable[Any]] = None,
    source_text: str = "",
    source_language: str = "",
    contract_version: int = 1,
) -> int:
    """Import addressing rules discovered by a context update into DB."""

    if contract_version >= 2:
        from src.utils.addressing_schema import AddressingCandidateV2

        parsed = []
        for item in candidates or []:
            candidate = (
                item
                if isinstance(item, AddressingCandidateV2)
                else AddressingCandidateV2.from_dict(
                    item,
                    source_language=source_language,
                )
            )
            if candidate:
                parsed.append(candidate)
        if not parsed:
            return 0
        engine = ContextMergeEngine(db=db)
        known_names = list(known_character_names or []) or (
            _known_character_names_from_context(updated_context_or_dynamic_state)
        )
        active_names = list(active_character_names or []) + (
            _active_names_from_dialogue(dialogue_attribution)
        )
        applied = 0
        for candidate in parsed:
            if candidate.action == "delete":
                existing = next((
                    rule for rule in db.get_addressing_rules(translation_id)
                    if _clean(rule.get("speaker_name")).casefold()
                    == candidate.speaker.casefold()
                    and _clean(rule.get("addressee_name")).casefold()
                    == candidate.addressee.casefold()
                ), None)
                if existing and not existing.get("is_locked"):
                    applied += int(db.delete_addressing_rule(
                        translation_id,
                        candidate.speaker,
                        candidate.addressee,
                    ))
                continue
            applied += int(engine.apply_delta(
                translation_id=translation_id,
                chunk_index=chunk_index,
                delta=candidate.to_delta(),
                trigger_source="context_update_v2",
                log_callback=log_callback,
                target_language=target_language,
                known_character_names=known_names,
                active_character_names=active_names,
                dialogue_attribution=dialogue_attribution,
                source_text=source_text,
                source_language=source_language,
            ))
        emit_progress_log(
            log_callback,
            "db_addressing_v2_imported",
            f"Imported {applied}/{len(parsed)} structured addressing update(s).",
            layer="db_addressing",
            data={
                "candidate_count": len(parsed),
                "applied_count": applied,
                "contract_version": 2,
            },
        )
        return applied

    return sync_markdown_addressing_to_db(
        translation_id=translation_id,
        db=db,
        context_or_dynamic_state=updated_context_or_dynamic_state,
        target_language=target_language,
        chunk_index=chunk_index,
        trigger_source="context_update",
        log_callback=log_callback,
        known_character_names=known_character_names,
        active_character_names=active_character_names,
        dialogue_attribution=dialogue_attribution,
    )


def _active_names_from_prompt_options(prompt_options: Optional[Dict[str, Any]]) -> List[str]:
    options = prompt_options or {}
    active: List[str] = []
    for item in options.get("active_character_names") or []:
        if item:
            active.append(str(item))
    attribution = options.get("dialogue_attribution") or {}
    if isinstance(attribution, dict):
        state_after = attribution.get("state_after") or {}
        if isinstance(state_after, dict):
            for key in ("speaker", "addressee"):
                value = state_after.get(key)
                if value and str(value).casefold() not in {"unknown", "none", "null"}:
                    active.append(str(value))
    return active


def build_directed_addressing_prompt_context(
    *,
    translation_id: str,
    db: Optional[Database],
    target_language: str = "",
    prompt_options: Optional[Dict[str, Any]] = None,
    active_character_names: Optional[Iterable[str]] = None,
    log_callback=None,
) -> str:
    """Render active DB-directed addressing rules for prompt injection."""

    if not translation_id or db is None:
        return ""
    if prompt_options and prompt_options.get("use_db_directed_addressing") is False:
        return ""
    active = list(active_character_names or []) or _active_names_from_prompt_options(prompt_options)
    try:
        from src.utils.context_projection import render_addressing_projection

        projection = render_addressing_projection(
            translation_id,
            db=db,
            target_language=target_language,
            active_character_names=active or None,
        )
    except Exception as exc:
        emit_progress_log(
            log_callback,
            "db_addressing_projection_failed",
            f"DB directed addressing projection failed: {exc}",
            level="warning",
            layer="db_addressing",
        )
        return ""
    if projection.strip():
        emit_progress_log(
            log_callback,
            "db_addressing_projected",
            "DB directed addressing projection injected into prompt.",
            layer="db_addressing",
            data={"active_character_count": len(active)},
        )
    return projection.strip()


def export_db_addressing_to_markdown(
    translation_id: str,
    db: Optional[Database],
) -> str:
    """Export DB addressing rules into markdown CURRENT ADDRESSING FORMS lines."""

    if not translation_id or db is None:
        return ""
    rules = db.get_addressing_rules(translation_id)
    lines = []
    for rule in rules:
        speaker = _clean(rule.get("speaker_name"))
        addressee = _clean(rule.get("addressee_name"))
        second = _clean(rule.get("target_pronoun"))
        if not speaker or not addressee or not second:
            continue
        self_ref = _clean(rule.get("self_pronoun"))
        vocative = _clean(rule.get("vocative")) or "none"
        register = _clean(rule.get("register")) or "polite"
        source_forms = [
            _clean(value)
            for value in rule.get("source_forms") or []
            if _clean(value)
        ]
        source_form = source_forms[0] if source_forms else "unknown"
        social_basis = [
            _clean(value)
            for value in rule.get("social_basis") or []
            if _clean(value)
        ]
        if self_ref and self_ref != "unspecified":
            target_form = (
                f"self-reference: {self_ref}; second-person pronoun: {second}; "
                f"vocative/address form: {vocative}"
            )
        else:
            target_form = second
        lines.append(
            f'- {speaker} → {addressee}: source form "{source_form}" | '
            f'recommended target-language form "{target_form}" | '
            f'{register}'
            + (f"; {', '.join(social_basis)}" if social_basis else "")
        )
    return "\n".join(lines)


def apply_db_addressing_to_context(
    context_content: str,
    translation_id: str,
    db: Optional[Database],
    fallback_context: str = "",
) -> str:
    """Return context content with DB addressing exported into dynamic state."""

    exported = export_db_addressing_to_markdown(translation_id, db)
    if not exported:
        if not fallback_context:
            return context_content
        exported = _extract_addressing_markdown(fallback_context)
    from src.utils.novel_context import (
        build_novel_context,
        extract_dynamic_state_from_text,
        extract_global_lore,
        _format_dynamic_sections,
        _split_dynamic_sections,
    )

    global_lore = extract_global_lore(context_content)
    dynamic_state = extract_dynamic_state_from_text(context_content) or ""
    _old_addressing, relationships, _has_sections = _split_dynamic_sections(dynamic_state)
    return build_novel_context(global_lore, _format_dynamic_sections(exported, relationships))


def apply_db_addressing_to_session(
    context_session: Any,
    translation_id: str,
    db: Optional[Database],
    fallback_context: str = "",
) -> bool:
    """Refresh a NovelContextSession dynamic state from DB addressing rules."""

    if not context_session or not translation_id or db is None:
        return False
    updated = apply_db_addressing_to_context(
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
