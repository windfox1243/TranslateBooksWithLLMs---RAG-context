"""Drive a single per-chunk novel context update against the LLM."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .addressing_requirements import (
    _inject_addressing_candidate_contract,
    _missing_addressing_requirements,
    _retry_missing_addressing_candidates,
)
from .characters import _character_gender_map, _character_profile_map, _plain_key
from .consolidation import _consolidation_interval, consolidate_context_lore
from .constants import logger
from .dynamic_state import infer_dynamic_address_identity_links
from .glossary import _character_alias_map
from .identity_links import infer_source_gender_updates, infer_source_identity_links
from .lore_merge import (
    SOURCE_ANALYSIS_SYSTEM_PROMPT,
    SOURCE_ANALYSIS_USER_PROMPT_TEMPLATE,
    UPDATE_SYSTEM_PROMPT,
    UPDATE_USER_PROMPT_TEMPLATE,
    merge_new_lore,
)
from .merge import merge_dynamic_state
from .rendering import _compose_source_analysis_text, render_novel_context_update_view


@dataclass(frozen=True)
class ContextUpdateOutcome:
    """Everything one context update produced, as one value.

    The three candidate sinks used to be filled at twelve separate points spread
    across this function's three exits: twelve chances to leave one stale, to
    forget one, or -- as happened -- to fill one on a path where its emptiness
    would be read as a finding rather than as a failure. Both failure exits fill
    the dialogue sink with `empty_dialogue_attribution()`, which is truthy and
    carries nothing, and a caller that read it as "this chunk had no speakers"
    cleared speaker state that was still current.

    One value built per exit and published in one place makes those twelve
    points one, and makes `succeeded` something the code states outright instead
    of something a caller has to infer from the shape of a dict.
    """

    global_lore: str
    dynamic_state: str
    change_logs: List[str] = field(default_factory=list)
    dialogue_attribution: Dict[str, Any] = field(default_factory=dict)
    relationship: Dict[str, Any] = field(default_factory=dict)
    addressing: Dict[str, Any] = field(default_factory=dict)
    succeeded: bool = True

    def publish(
        self,
        dialogue_attribution_sink: Optional[Dict[str, Any]],
        relationship_candidate_sink: Optional[Dict[str, Any]],
        addressing_candidate_sink: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, List[str]]:
        """Fill the caller's sinks and return the triple they unpack."""
        for sink, payload in (
            (dialogue_attribution_sink, self.dialogue_attribution),
            (relationship_candidate_sink, self.relationship),
            (addressing_candidate_sink, self.addressing),
        ):
            if sink is not None:
                sink.clear()
                sink.update(payload)
        return self.global_lore, self.dynamic_state, self.change_logs


def _update_failed(
    global_lore: str,
    dynamic_state: str,
    parse_status: str,
) -> ContextUpdateOutcome:
    """The outcome of an update that produced nothing, keeping the caller's state."""
    from src.utils.dialogue_attribution import empty_dialogue_attribution

    empty = {"candidates": [], "parse_status": parse_status}
    return ContextUpdateOutcome(
        global_lore=global_lore,
        dynamic_state=dynamic_state,
        dialogue_attribution=empty_dialogue_attribution(),
        relationship=dict(empty),
        addressing=dict(empty),
        succeeded=False,
    )


async def update_novel_context_chunk(
    llm_client: Any,
    model_name: str,
    current_global_lore: str,
    current_dynamic_state: str,
    source_chunk: str,
    translated_chunk: Optional[str],
    source_language: str,
    target_language: str,
    chunk_index: int = 0,
    total_chunks: int = 0,
    source_context: str = "",
    dialogue_turns: Optional[List[Dict[str, str]]] = None,
    current_dialogue_state: Optional[Dict[str, str]] = None,
    dialogue_attribution_sink: Optional[Dict[str, Any]] = None,
    relationship_candidate_sink: Optional[Dict[str, Any]] = None,
    addressing_candidate_sink: Optional[Dict[str, Any]] = None,
    context_contract_version: int = 1,
    selective_context_view: bool = True,
    context_view_max_tokens: Optional[int] = None,
    custom_instructions: str = "",
    glossary_block: str = "",
    log_callback: Optional[Callable] = None,
) -> Tuple[str, str, List[str]]:
    """Calls the LLM to update global lore and dynamic state incrementally.

    Returns:
        Tuple of (updated_global_lore, updated_dynamic_state, change_logs)
    """
    from src.utils.dialogue_attribution import (
        dialogue_candidates_prompt,
        parse_dialogue_attribution,
    )

    dialogue_turns = list(dialogue_turns or [])
    current_dialogue_state = dict(current_dialogue_state or {})
    source_analysis_text = _compose_source_analysis_text(
        source_context,
        source_chunk,
    )
    prompt_reference_text = "\n\n".join(
        part
        for part in (
            source_analysis_text,
            translated_chunk or "",
        )
        if part
    )
    prompt_global_lore, prompt_dynamic_state = render_novel_context_update_view(
        current_global_lore,
        current_dynamic_state,
        reference_text=prompt_reference_text,
        max_tokens=context_view_max_tokens,
        selective=selective_context_view,
    )
    custom_instructions_section = (
        f"\n### CUSTOM INSTRUCTIONS & STYLE GUIDELINES:\n{custom_instructions.strip()}\n"
        if custom_instructions and custom_instructions.strip()
        else ""
    )
    glossary_block_section = (
        f"\n### ACTIVE PROJECT GLOSSARY:\n{glossary_block.strip()}\n"
        if glossary_block and glossary_block.strip()
        else ""
    )
    prompt_values = {
        "current_global_lore": prompt_global_lore,
        "current_dynamic_state": prompt_dynamic_state,
        "custom_instructions_section": custom_instructions_section,
        "glossary_block_section": glossary_block_section,
        "source_language": source_language,
        "target_language": target_language,
        "source_context": source_context or "(none)",
        "source_chunk": source_chunk,
        "translated_chunk": translated_chunk or "",
        "chunk_index": chunk_index if chunk_index > 0 else "?",
        "total_chunks": total_chunks if total_chunks > 0 else "?",
        "current_dialogue_state": (
            current_dialogue_state or {"speaker": "Unknown", "addressee": "Unknown"}
        ),
        "dialogue_candidates": dialogue_candidates_prompt(dialogue_turns),
    }
    if translated_chunk is None:
        user_prompt = SOURCE_ANALYSIS_USER_PROMPT_TEMPLATE.format(**prompt_values)
        system_prompt = SOURCE_ANALYSIS_SYSTEM_PROMPT
    else:
        user_prompt = UPDATE_USER_PROMPT_TEMPLATE.format(**prompt_values)
        system_prompt = UPDATE_SYSTEM_PROMPT
    if context_contract_version >= 2:
        system_prompt = _inject_addressing_candidate_contract(system_prompt)

    try:
        response = await llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        if not response or not response.content:
            logger.warning("Empty response received from LLM during novel context chunk update. Keeping current state.")
            return _update_failed(
                current_global_lore, current_dynamic_state, "empty_response"
            ).publish(
                dialogue_attribution_sink,
                relationship_candidate_sink,
                addressing_candidate_sink,
            )

        content = response.content.strip()

        # Parse blocks
        new_chars = ""
        new_aliases = ""
        new_glossary = ""
        new_dynamic = current_dynamic_state
        dialogue_raw = ""
        relationship_candidate_raw = ""
        addressing_candidate_raw = ""

        import re
        chars_match = re.search(
            r'\[NEW_CHARACTERS\]\s*(.*?)\s*'
            r'(?=\[IDENTITY_LINKS\]|\[NEW_GLOSSARY\]|\[DYNAMIC_STATE\]|$)',
            content,
            re.DOTALL,
        )
        aliases_match = re.search(
            r'\[IDENTITY_LINKS\]\s*(.*?)\s*'
            r'(?=\[NEW_GLOSSARY\]|\[DYNAMIC_STATE\]|\[NEW_CHARACTERS\]|$)',
            content,
            re.DOTALL,
        )
        glossary_match = re.search(
            r'\[NEW_GLOSSARY\]\s*(.*?)\s*'
            r'(?=\[DYNAMIC_STATE\]|\[IDENTITY_LINKS\]|\[NEW_CHARACTERS\]|$)',
            content,
            re.DOTALL,
        )
        dynamic_match = re.search(
            r'\[DYNAMIC_STATE\]\s*(.*?)\s*'
            r'(?=\[ADDRESSING_CANDIDATES\]|\[RELATIONSHIP_CANDIDATES\]|\[DIALOGUE_ATTRIBUTION\]|$)',
            content,
            re.DOTALL,
        )
        addressing_candidate_match = re.search(
            r'\[ADDRESSING_CANDIDATES\]\s*(.*?)\s*'
            r'(?=\[RELATIONSHIP_CANDIDATES\]|\[DIALOGUE_ATTRIBUTION\]|$)',
            content,
            re.DOTALL,
        )
        relationship_candidate_match = re.search(
            r'\[RELATIONSHIP_CANDIDATES\]\s*(.*?)\s*'
            r'(?=\[DIALOGUE_ATTRIBUTION\]|$)',
            content,
            re.DOTALL,
        )
        dialogue_match = re.search(
            r'\[DIALOGUE_ATTRIBUTION\]\s*(.*?)\s*$',
            content,
            re.DOTALL,
        )

        if chars_match:
            new_chars = chars_match.group(1).strip()
        if aliases_match:
            new_aliases = aliases_match.group(1).strip()
        if glossary_match:
            new_glossary = glossary_match.group(1).strip()
        if dynamic_match:
            new_dynamic = dynamic_match.group(1).strip()
        if relationship_candidate_match:
            relationship_candidate_raw = relationship_candidate_match.group(1).strip()
        if addressing_candidate_match:
            addressing_candidate_raw = addressing_candidate_match.group(1).strip()
        if dialogue_match:
            dialogue_raw = dialogue_match.group(1).strip()

        addressing_candidates = []
        addressing_parse_status = "absent"
        if context_contract_version >= 2 and addressing_candidate_raw:
            from src.utils.addressing_schema import parse_addressing_candidate_block
            from src.utils.progress_logging import emit_progress_log

            addressing_candidates, addressing_parse_status = (
                parse_addressing_candidate_block(
                    addressing_candidate_raw,
                    source_language=source_language,
                )
            )
            if addressing_parse_status in {"invalid_json", "invalid_contract"}:
                retry_prompt = (
                    "Repair only the JSON syntax of this addressing candidate payload. "
                    "Return one object with an updates array and no prose. Do not add "
                    "facts or evidence that were absent.\n\n"
                    f"Malformed payload:\n{addressing_candidate_raw[:6000]}"
                )
                try:
                    retry_response = await llm_client.generate(
                        prompt=retry_prompt,
                        system_prompt=(
                            "You repair structured addressing JSON syntax only. "
                            "Never invent source forms, translations, or evidence."
                        ),
                    )
                    repaired_raw = str(
                        getattr(retry_response, "content", "") or ""
                    ).strip()
                    repaired, repaired_status = parse_addressing_candidate_block(
                        repaired_raw,
                        source_language=source_language,
                    )
                    if repaired_status == "json":
                        addressing_candidates = repaired
                        addressing_parse_status = "json_repaired"
                except Exception:
                    addressing_parse_status = "invalid_json"
            emit_progress_log(
                log_callback,
                "addressing_contract_parse",
                f"Addressing candidate contract parsed with {addressing_parse_status}.",
                level=(
                    "warning"
                    if addressing_parse_status in {"invalid_json", "invalid_contract"}
                    else "info"
                ),
                layer="db_addressing",
                data={
                    "parse_status": addressing_parse_status,
                    "candidate_count": len(addressing_candidates),
                    "contract_version": 2,
                },
            )
        # The addressing payload used to be published here as well as at the end
        # of the successful path. There is no return between the two, so this one
        # was always overwritten before any caller could see it.

        relationship_candidates = []
        relationship_parse_status = "absent"
        if relationship_candidate_raw:
            from src.utils.progress_logging import emit_progress_log
            from src.utils.relationship_schema import parse_relationship_candidate_block

            relationship_candidates, relationship_parse_status = (
                parse_relationship_candidate_block(relationship_candidate_raw)
            )
            if relationship_parse_status in {"invalid_json", "invalid_contract"}:
                emit_progress_log(
                    log_callback,
                    "relationship_contract_retry",
                    "Relationship candidate JSON was malformed; retrying the contract once.",
                    level="warning",
                    layer="relationship_reasoning",
                    data={"parse_status": relationship_parse_status},
                )
                retry_prompt = (
                    "Repair the relationship candidate payload below. Return only one "
                    "valid JSON object with the shape "
                    "{\"relationships\":[{\"source\":\"...\",\"target\":\"...\","
                    "\"relationship_type\":\"...\",\"direction\":\"directed | symmetric\","
                    "\"scope\":\"durable | situational\",\"hierarchy\":\"source_senior | "
                    "source_junior | peer | unknown\",\"evidence_quote\":\"exact source quote\","
                    "\"confidence\":0.0}]}. Do not add evidence or facts that were absent.\n\n"
                    f"Malformed payload:\n{relationship_candidate_raw[:6000]}"
                )
                try:
                    retry_response = await llm_client.generate(
                        prompt=retry_prompt,
                        system_prompt=(
                            "You repair JSON syntax only. Preserve the proposed facts and "
                            "return no prose or markdown."
                        ),
                    )
                    retry_content = str(
                        getattr(retry_response, "content", "") or ""
                    ).strip()
                    retry_candidates, retry_status = parse_relationship_candidate_block(
                        retry_content
                    )
                    if retry_status == "json":
                        relationship_candidates = retry_candidates
                        relationship_parse_status = "json_repaired"
                except Exception:
                    relationship_parse_status = "invalid_json"
            emit_progress_log(
                log_callback,
                "relationship_contract_parse",
                "Relationship candidate contract "
                f"({ 'source analysis' if translated_chunk is None else 'translation sync' }) "
                f"parsed with {relationship_parse_status}.",
                level=(
                    "warning"
                    if relationship_parse_status in {"invalid_json", "invalid_contract"}
                    else "info"
                ),
                layer="relationship_reasoning",
                data={
                    "parse_status": relationship_parse_status,
                    "candidate_count": len(relationship_candidates),
                    "contract_version": "1.0",
                    "phase": (
                        "source_analysis"
                        if translated_chunk is None
                        else "translation_sync"
                    ),
                },
            )
            for candidate in (
                relationship_candidates if translated_chunk is None else []
            ):
                emit_progress_log(
                    log_callback,
                    "relationship_candidate_extracted",
                    f"Relationship candidate extracted: {candidate.source} -> "
                    f"{candidate.target} [{candidate.relationship_type}].",
                    layer="relationship_reasoning",
                    data={
                        "chunk_index": chunk_index - 1 if chunk_index > 0 else 0,
                        "source": candidate.source,
                        "target": candidate.target,
                        "relationship_type": candidate.relationship_type,
                        "scope": candidate.scope,
                        "confidence": candidate.confidence,
                        "parser_status": relationship_parse_status,
                    },
                )
        source_backstop_gender_updates = infer_source_gender_updates(
            source_analysis_text,
            current_global_lore,
            new_chars,
        )
        if source_backstop_gender_updates:
            new_chars = "\n".join(
                part
                for part in (new_chars, source_backstop_gender_updates)
                if part.strip()
            )
        source_backstop_aliases = infer_source_identity_links(
            source_analysis_text,
            current_global_lore,
            new_chars,
        )
        if source_backstop_aliases:
            new_aliases = "\n".join(
                part
                for part in (new_aliases, source_backstop_aliases)
                if part.strip()
            )
        trusted_dynamic_aliases = infer_dynamic_address_identity_links(
            current_dynamic_state,
            current_global_lore,
        )
        if new_dynamic.strip():
            if new_dynamic.startswith("```"):
                lines = new_dynamic.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                new_dynamic = "\n".join(lines).strip()

            # Clean dynamic state boundaries if the model generated them by mistake
            new_dynamic = new_dynamic.replace("---DYNAMIC_STATE_START---", "").replace("---DYNAMIC_STATE_END---", "").strip()

            # Clean up `# DYNAMIC RELATIONSHIP STATE` headers
            lines = new_dynamic.splitlines()
            cleaned_lines = []
            for line in lines:
                if line.strip().upper().replace(" ", "") == "#DYNAMICRELATIONSHIPSTATE":
                    continue
                cleaned_lines.append(line)

            new_dynamic = "\n".join(cleaned_lines).strip()
        else:
            new_dynamic = current_dynamic_state

        updated_global_lore, change_logs = merge_new_lore(
            current_global_lore,
            new_chars,
            new_glossary,
            new_aliases,
            source_analysis_text,
            trusted_dynamic_aliases,
        )
        new_dynamic = merge_dynamic_state(
            current_dynamic_state,
            new_dynamic,
            _character_alias_map(updated_global_lore),
            target_language=target_language,
            character_genders=_character_gender_map(updated_global_lore),
            character_profiles=_character_profile_map(updated_global_lore),
        )

        # LLM consolidation pass: periodically deduplicate character descriptions
        # that the deterministic merge layer missed (semantically similar rephrasing).
        consolidation_interval = _consolidation_interval()
        is_last_chunk = (total_chunks > 0 and chunk_index == total_chunks)
        if (
            consolidation_interval > 0
            and chunk_index > 0
            and (chunk_index % consolidation_interval == 0 or is_last_chunk)
        ):
            logger.info(
                f"[Novel Context] Running consolidation pass at chunk {chunk_index}/{total_chunks}."
            )
            consolidated_lore, consolidation_logs = await consolidate_context_lore(
                llm_client=llm_client,
                model_name=model_name,
                global_lore=updated_global_lore,
                dynamic_state=new_dynamic,
            )
            updated_global_lore = consolidated_lore
            change_logs.extend(consolidation_logs)

        # Track relationship logs by comparing old and new dynamic state
        # (This is secondary logging for the terminal, just printing line updates)
        if new_dynamic != current_dynamic_state:
            # We can log that relationship state changed
            change_logs.append("[Novel Context] Dynamic state updated.")

        dialogue_attribution = parse_dialogue_attribution(
            dialogue_raw,
            dialogue_turns,
            _character_alias_map(updated_global_lore),
            current_dialogue_state,
        )
        coverage_gaps: List[Dict[str, Any]] = []
        if context_contract_version >= 2:
            from src.utils.progress_logging import emit_progress_log

            requirements = _missing_addressing_requirements(
                dialogue_attribution,
                addressing_candidates,
                current_dynamic_state,
                source_language,
            )
            if requirements:
                emit_progress_log(
                    log_callback,
                    "addressing_coverage_retry",
                    "High-confidence dialogue addressing coverage was incomplete; "
                    "retrying the missing pairs once.",
                    level="warning",
                    layer="db_addressing",
                    data={"missing_pair_count": len(requirements)},
                )
                try:
                    retry_candidates, retry_status = (
                        await _retry_missing_addressing_candidates(
                            llm_client,
                            requirements,
                            source_language,
                            target_language,
                        )
                    )
                    existing_pairs = {
                        (_plain_key(item.speaker), _plain_key(item.addressee))
                        for item in addressing_candidates
                    }
                    for candidate in retry_candidates:
                        pair = (
                            _plain_key(candidate.speaker),
                            _plain_key(candidate.addressee),
                        )
                        if pair not in existing_pairs:
                            addressing_candidates.append(candidate)
                            existing_pairs.add(pair)
                    if retry_status == "json":
                        addressing_parse_status = "json_coverage_retried"
                except Exception as coverage_error:
                    logger.warning(
                        "Addressing coverage retry failed for chunk %s: %s",
                        chunk_index,
                        coverage_error,
                    )
                coverage_gaps = _missing_addressing_requirements(
                    dialogue_attribution,
                    addressing_candidates,
                    current_dynamic_state,
                    source_language,
                )
                if coverage_gaps:
                    addressing_parse_status = "coverage_gap"
                    emit_progress_log(
                        log_callback,
                        "addressing_coverage_gap",
                        "Addressing candidates still omit one or more "
                        "high-confidence directed dialogue pairs.",
                        level="warning",
                        layer="db_addressing",
                        data={
                            "missing_pair_count": len(coverage_gaps),
                            "pairs": [
                                {
                                    "speaker": item["speaker"],
                                    "addressee": item["addressee"],
                                    "dialogue_turn_id": item["dialogue_turn_id"],
                                }
                                for item in coverage_gaps
                            ],
                        },
                    )

        return ContextUpdateOutcome(
            global_lore=updated_global_lore,
            dynamic_state=new_dynamic,
            change_logs=change_logs,
            dialogue_attribution=dialogue_attribution,
            relationship={
                "candidates": [
                    {
                        **candidate.to_dict(),
                        "parser_status": relationship_parse_status,
                    }
                    for candidate in relationship_candidates
                ],
                "parse_status": relationship_parse_status,
            },
            addressing={
                "candidates": [
                    candidate.to_dict() for candidate in addressing_candidates
                ],
                "parse_status": addressing_parse_status,
                "coverage_gaps": coverage_gaps,
            },
        ).publish(
            dialogue_attribution_sink,
            relationship_candidate_sink,
            addressing_candidate_sink,
        )

    except Exception as e:
        logger.error(f"Error in update_novel_context_chunk: {e}")
        return _update_failed(
            current_global_lore, current_dynamic_state, "update_failed"
        ).publish(
            dialogue_attribution_sink,
            relationship_candidate_sink,
            addressing_candidate_sink,
        )
