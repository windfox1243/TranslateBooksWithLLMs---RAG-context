"""Commit a translated unit's pending context state, shared by all pipelines.

Context analysis runs before a unit is translated, but what it learned about
addressing and relationships is only written to the database once the
translation succeeded, so a failed unit leaves no trace. The TXT/SRT, Plain
Text and EPUB pipelines each carried a copy of that commit; this is the one
implementation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def commit_pending_context(
    *,
    context_session: Any,
    db: Any,
    translation_id: str,
    chunk_index: int,
    addressing: Optional[Dict[str, Any]],
    relationships: Optional[Dict[str, Any]],
    target_language: str,
    source_language: str,
    contract_version: int,
    relationship_mode: str,
    use_db_addressing: bool = True,
    log_callback: Optional[Callable] = None,
) -> bool:
    """Write pending addressing and relationship updates atomically.

    `addressing` and `relationships` are the records the pipeline stored for
    this unit before translating it (or None). Both are written in one
    context-state transaction, which nests inside a caller's transaction,
    and the session is then refreshed from the database so the next unit's
    prompt reads what was committed. Returns True when anything was written.
    """
    from src.utils.db_addressing import (
        apply_db_addressing_to_session,
        sync_context_update_addressing_to_db,
    )
    from src.utils.relationship_sync import (
        apply_relationship_graph_to_session,
        sync_context_update_relationships_to_db,
    )

    if not (context_session and translation_id and db is not None):
        return False
    commit_addressing = bool(addressing) and use_db_addressing
    commit_relationships = bool(relationships) and relationship_mode != "off"
    if not (commit_addressing or commit_relationships):
        return False

    with db.context_state_transaction():
        if commit_addressing:
            sync_context_update_addressing_to_db(
                translation_id=translation_id,
                db=db,
                updated_context_or_dynamic_state=addressing["content"],
                target_language=target_language,
                chunk_index=chunk_index,
                log_callback=log_callback,
                dialogue_attribution=addressing.get("dialogue_attribution"),
                candidates=addressing.get("candidates"),
                source_text=addressing.get("source_text", ""),
                source_language=source_language,
                contract_version=contract_version,
            )
        if commit_relationships:
            sync_context_update_relationships_to_db(
                translation_id=translation_id,
                db=db,
                updated_context_or_dynamic_state=relationships["content"],
                source_text=relationships["source_text"],
                candidates=relationships.get("candidates"),
                parser_status=relationships.get("parser_status", "absent"),
                target_language=target_language,
                chunk_index=chunk_index,
                active_character_names=relationships.get("active_character_names"),
                log_callback=log_callback,
            )
    if commit_addressing:
        apply_db_addressing_to_session(context_session, translation_id, db)
    if commit_relationships:
        apply_relationship_graph_to_session(context_session, translation_id, db)
    return True


def save_committed_context(
    context_session: Any,
    *,
    contract_version: int,
    log_callback: Optional[Callable] = None,
) -> None:
    """Persist the session file and tell the UI the context changed."""
    from src.utils.progress_logging import emit_progress_log

    context_session.save()
    if log_callback and contract_version >= 2:
        emit_progress_log(
            log_callback,
            "novel_context_state",
            "Context committed",
            layer="novel_context",
            data={
                "type": "novel_context_state",
                "content": context_session.content,
                "filename": context_session.path.name,
            },
        )
