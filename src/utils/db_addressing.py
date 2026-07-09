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


def _target_form_from_details(details: str) -> str:
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
        register = "polite"
        for part in [p.strip() for p in details.split("|") if p.strip()]:
            lowered = part.casefold()
            if not any(key in lowered for key in ("source form", "target", "self-reference", "second-person", "vocative")):
                register = _clean(part).lower() or register
                break
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
                )
            )
    return deltas


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


def sync_markdown_addressing_to_db(
    *,
    translation_id: str,
    db: Database,
    context_or_dynamic_state: str,
    target_language: str = "",
    chunk_index: int = 0,
    trigger_source: str = "markdown_context",
    log_callback=None,
) -> int:
    """Import markdown addressing rules into DB through the merge engine."""

    del target_language
    if not translation_id or db is None:
        return 0
    addressing = _extract_addressing_markdown(context_or_dynamic_state)
    deltas = parse_markdown_addressing_lines(addressing)
    if not deltas:
        return 0
    engine = ContextMergeEngine(db=db)
    applied = engine.apply_batch_deltas(
        translation_id=translation_id,
        chunk_index=chunk_index,
        deltas=deltas,
        trigger_source=trigger_source,
        log_callback=log_callback,
    )
    emit_progress_log(
        log_callback,
        "db_addressing_imported",
        f"Imported {applied}/{len(deltas)} directed addressing rule(s) into DB.",
        layer="db_addressing",
        data={"candidate_count": len(deltas), "applied_count": applied},
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
) -> int:
    """Import addressing rules discovered by a context update into DB."""

    return sync_markdown_addressing_to_db(
        translation_id=translation_id,
        db=db,
        context_or_dynamic_state=updated_context_or_dynamic_state,
        target_language=target_language,
        chunk_index=chunk_index,
        trigger_source="context_update",
        log_callback=log_callback,
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
        if self_ref and self_ref != "unspecified":
            target_form = (
                f"self-reference: {self_ref}; second-person pronoun: {second}; "
                f"vocative/address form: {vocative}"
            )
        else:
            target_form = second
        lines.append(
            f'- {speaker} → {addressee}: source form "" | '
            f'recommended target-language form "{target_form}" | {register}'
        )
    return "\n".join(lines)


def apply_db_addressing_to_context(
    context_content: str,
    translation_id: str,
    db: Optional[Database],
) -> str:
    """Return context content with DB addressing exported into dynamic state."""

    exported = export_db_addressing_to_markdown(translation_id, db)
    if not exported:
        return context_content
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
) -> bool:
    """Refresh a NovelContextSession dynamic state from DB addressing rules."""

    if not context_session or not translation_id or db is None:
        return False
    updated = apply_db_addressing_to_context(context_session.content, translation_id, db)
    if updated == context_session.content:
        return False
    from src.utils.novel_context import extract_dynamic_state_from_text

    context_session.dynamic_state = extract_dynamic_state_from_text(updated) or ""
    context_session.sync_prompt()
    return True
