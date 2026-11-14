"""Analyze a source unit's context before it is translated, for every pipeline.

The TXT/SRT, Plain Text and EPUB pipelines each ran the same sequence before
translating a unit: update the novel context from the source, commit the
deterministic social evidence it contains, record what the analysis
proposed for addressing and relationships (to be committed only once the
translation succeeds, see commit.py), and in "project" mode refresh the
session from the database so this unit's own prompt already reads it. This
is the one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class UnitContextAnalysis:
    """What analyzing one unit produced.

    `analyzed` is True once the source analysis itself succeeded, even if a
    later step failed; `addressing` and `relationships` are the pending
    records for commit_pending_context (None when not produced).
    """

    analyzed: bool = False
    addressing: Optional[Dict[str, Any]] = None
    relationships: Optional[Dict[str, Any]] = None


async def analyze_unit_context(
    *,
    context_session: Any,
    db: Any,
    translation_id: str,
    llm_client: Any,
    model_name: str,
    source_text: str,
    source_language: str,
    target_language: str,
    local_index: int,
    total_units: int,
    global_index: int,
    scene_key: Any,
    relationship_mode: str,
    prompt_options: Dict[str, Any],
    unit_label: str = "chunk",
    log_callback: Optional[Callable] = None,
) -> UnitContextAnalysis:
    """Run the pre-translation context analysis for one unit.

    Failures are logged as ``novel_context_update_failed`` and never raised:
    the unit is still translated, with whatever context is available.
    """
    from src.core.context.unit_pipeline import (
        commit_source_social_evidence,
        relevant_character_names,
    )
    from src.utils.db_addressing import apply_db_addressing_to_session
    from src.utils.relationship_sync import (
        apply_relationship_graph_to_session,
        judge_ambiguous_relationship_candidates,
    )

    result = UnitContextAnalysis()
    has_db = bool(translation_id) and db is not None
    use_db_addressing = prompt_options.get("use_db_directed_addressing", True)

    if log_callback:
        log_callback(
            "novel_context_updating",
            f"Analyzing source context for {unit_label} {local_index + 1} "
            "before translation...",
        )
    try:
        fallback_context = context_session.content
        change_logs = await context_session.analyze_source(
            llm_client=llm_client,
            model_name=model_name,
            source_chunk=source_text,
            source_language=source_language,
            target_language=target_language,
            chunk_index=local_index + 1,
            total_chunks=total_units,
            scene_key=scene_key,
        )
        result.analyzed = True

        # Deterministic social evidence is committed before the draft request
        # so an honorific or explicit hierarchy in this unit can guide its own
        # translation. LLM-derived candidates wait for commit_pending_context.
        if has_db:
            commit_source_social_evidence(
                translation_id=translation_id,
                db=db,
                global_lore=context_session.global_lore,
                source_text=source_text,
                dialogue_attribution=context_session.dialogue_attribution,
                source_language=source_language,
                target_language=target_language,
                chunk_index=global_index,
                log_callback=log_callback,
            )
        if log_callback:
            log_callback(
                "novel_context_updated",
                f"Novel context prepared for {unit_label} {local_index + 1}.",
            )
            for change_log in change_logs:
                log_callback("novel_context_log", change_log)

        if has_db and use_db_addressing:
            result.addressing = {
                "content": context_session.content,
                "dialogue_attribution": context_session.dialogue_attribution,
                "candidates": list(context_session.addressing_candidates),
                "parser_status": context_session.addressing_parse_status,
                "source_text": source_text,
            }
        if has_db and relationship_mode != "off":
            locked_facts = [
                edge for edge in db.get_relationship_edges(
                    translation_id,
                    statuses=["accepted"],
                )
                if edge.get("is_locked")
            ]
            judged = await judge_ambiguous_relationship_candidates(
                llm_client=llm_client,
                candidates=context_session.relationship_candidates,
                source_text=source_text,
                model_name=model_name,
                enabled=(
                    prompt_options.get("use_relationship_llm_judge")
                    in {True, "selective", "always"}
                ),
                locked_facts=locked_facts,
                log_callback=log_callback,
            )
            result.relationships = {
                "content": context_session.content,
                "source_text": source_text,
                "candidates": judged,
                "parser_status": context_session.relationship_parse_status,
                "active_character_names": relevant_character_names(
                    global_lore=context_session.global_lore,
                    source_text=source_text,
                    dialogue_attribution=context_session.dialogue_attribution,
                    source_language=source_language,
                ),
            }
            # "shadow" computes the graph without letting it reach prompts;
            # only "project" refreshes the session from the database here.
            if relationship_mode == "project":
                if use_db_addressing:
                    apply_db_addressing_to_session(
                        context_session,
                        translation_id,
                        db,
                        fallback_context=fallback_context,
                    )
                apply_relationship_graph_to_session(
                    context_session,
                    translation_id,
                    db,
                    fallback_context=fallback_context,
                )
                context_session.save()
    except Exception as e:
        if log_callback:
            log_callback(
                "novel_context_update_failed",
                f"Failed to prepare novel context: {str(e)}",
            )
    return result
