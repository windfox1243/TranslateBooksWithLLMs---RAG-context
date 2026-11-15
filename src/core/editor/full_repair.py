"""Bounded full-chunk repair: the editor's last resort after local patches.

When exact-span edits could not resolve the editor's findings, the
reflection pass asks the editor to rewrite the whole chunk, validates the
rewrite deterministically, and retries with the validation failures as
feedback, a bounded number of times. A failure that repeats unchanged
either schedules the one configured escalation or stops the loop.

Moved out of src/core/translator.py's reflection pass; the conditions and
the order of every step are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class FullRepairOutcome:
    """What the repair loop ended with.

    `repaired_text` is set only when a rewrite passed every validation; it
    is then already recorded and announced, and the caller finishes the run
    with it. Otherwise `validation_errors` holds the last attempt's reasons.
    `narrator_conformance` is the last audit of a rewrite, or None when no
    rewrite got that far.
    """

    attempts: int = 0
    repaired_text: str = ""
    raw_content: str = ""
    validation_errors: List[str] = field(default_factory=list)
    attempt_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    narrator_conformance: Optional[Dict[str, Any]] = None


async def run_full_chunk_repair(
    *,
    session: Any,
    source_chunk: str,
    draft_translation: str,
    critique_feedback: str,
    reflection_issues: List[Dict[str, Any]],
    term_pairs: Any,
    context_session: Any,
    target_language: str,
    source_language: str,
    custom_instructions: str,
    glossary_block: str,
    active_novel_context: str,
    source_available: bool,
    narrative_voice_context: str,
    protected_terms: List[str],
    glossary_terms: Dict[str, Any],
    options: Dict[str, Any],
    repair_validator: Optional[Callable[[str], Optional[str]]],
    max_attempts: int,
    log_callback: Optional[Callable] = None,
) -> FullRepairOutcome:
    """Ask for a validated full rewrite, up to `max_attempts` times."""
    from src.core import translator as _translator
    from src.core.editor import audit_narrator_conformance
    from src.core.llm import TranslationExtractor
    from src.prompts.prompts import generate_chunk_repair_prompt
    from src.utils.progress_logging import emit_progress_log
    from src.utils.translation_quality import (
        describe_repair_failures,
        validate_editor_repair,
    )

    outcome = FullRepairOutcome()
    validation_errors: List[str] = []
    previous_failure_signature: Optional[tuple[str, ...]] = None

    for repair_attempt in range(max_attempts):
        outcome.attempts = repair_attempt + 1
        repair_feedback = critique_feedback
        if validation_errors:
            repair_feedback += (
                "\n\nPREVIOUS REPAIR VALIDATION FAILURES:\n- "
                + "\n- ".join(describe_repair_failures(
                    validation_errors, draft_text=draft_translation,
                ))
                + "\nReturn a corrected complete translation."
            )
        repair_pair = generate_chunk_repair_prompt(
            source_chunk=source_chunk,
            draft_translation=draft_translation,
            critique_feedback=repair_feedback,
            target_language=target_language,
            custom_instructions=custom_instructions,
            glossary_block=glossary_block,
            novel_context=active_novel_context,
            source_available=source_available,
            narrative_voice_context=narrative_voice_context,
        )
        validation_errors = []
        repair_response = None
        raw_content = ""
        try:
            repair_response = await session.generate(
                prompt=repair_pair.user,
                system_prompt=repair_pair.system,
                temperature=0.3 if repair_attempt == 0 else 0.0,
                max_output_tokens=max(
                    session.output_tokens,
                    min(16384, len(draft_translation) // 3 + 1024),
                ),
                stage=f"repair_{repair_attempt + 1}",
            )
            raw_content = (
                repair_response.content
                if repair_response and getattr(repair_response, "content", None)
                else ""
            )
            editor_client = session.client
            repaired_text = (
                editor_client.extract_translation(raw_content)
                if hasattr(editor_client, "extract_translation")
                else None
            )
            if not repaired_text:
                extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
                repaired_text = extractor.extract(raw_content)
            repaired_text = str(repaired_text or "").strip()
            if not repaired_text:
                validation_errors.append("repair output was empty")
            if repaired_text and repair_validator:
                try:
                    adapter_feedback = repair_validator(repaired_text)
                except Exception as validation_error:
                    adapter_feedback = f"repair validator raised {validation_error}"
                if adapter_feedback:
                    validation_errors.append(str(adapter_feedback))
            if repaired_text:
                validation_errors.extend(validate_editor_repair(
                    repaired_text,
                    reflection_issues,
                    draft_text=draft_translation,
                    source_text=source_chunk,
                    source_language=source_language,
                    target_language=target_language,
                    protected_terms=protected_terms,
                    glossary_terms=glossary_terms,
                ))
            if repaired_text:
                repaired_conformance = audit_narrator_conformance(
                    source_text=source_chunk,
                    target_text=repaired_text,
                    source_language=source_language,
                    target_language=target_language,
                    file_type=str(options.get("file_type") or "txt"),
                    dialogue_attribution=options.get("dialogue_attribution") or {},
                    db=options.get("_checkpoint_db"),
                    translation_id=str(options.get("translation_id") or ""),
                    chunk_index=int(options.get("chunk_index", 0) or 0),
                    explicit_override=str(
                        options.get("narrator_self_reference_override") or ""
                    ),
                )
                outcome.narrator_conformance = repaired_conformance
                if repaired_conformance.get("status") == "fail":
                    validation_errors.extend(
                        f"{item.get('reason_code')}:{item.get('segment_id')}"
                        for item in repaired_conformance.get("violating_segments") or []
                        if item.get("blocking", True)
                    )
            if repaired_text and not validation_errors:
                session.record_attempt(
                    f"repair_{repair_attempt + 1}", repair_response, raw_content,
                    parse_status="translation_tags",
                    issues=reflection_issues,
                )
                if term_pairs and context_session and hasattr(
                    context_session,
                    "register_editor_terms",
                ):
                    context_session.register_editor_terms(term_pairs)
                if log_callback:
                    emit_progress_log(
                        log_callback,
                        "editor_state_fallback_applied",
                        "Senior Editor state: full-chunk fallback validated and applied.",
                        layer="senior_editor_repair",
                    )
                outcome.repaired_text = repaired_text
                outcome.raw_content = raw_content
                return outcome
        except Exception as e:
            validation_errors.append(
                f"repair_call_failed:{_translator._classify_editor_exception(e)}"
            )

        failure_class = (
            "provider_empty" if "empty" in " ".join(validation_errors).casefold()
            else "adapter_invalid" if any(
                str(reason).startswith("adapter_")
                for reason in validation_errors
            )
            else "repair_validation"
        )
        session.record_attempt(
            f"repair_{repair_attempt + 1}",
            repair_response,
            str(raw_content or ""),
            failure_class=failure_class,
            reason_codes=validation_errors,
            issues=reflection_issues,
        )

        if log_callback:
            emit_progress_log(
                log_callback,
                "repair_validation_failed",
                "Senior Editor state: full-chunk fallback was rejected; preserving the best valid draft for review.",
                level="warning",
                layer="senior_editor_repair",
                data={"attempt": repair_attempt + 1, "reasons": validation_errors},
            )
        outcome.attempt_diagnostics.append({
            "attempt": repair_attempt + 1,
            "reason_codes": list(validation_errors),
            "adapter_valid": not any(
                str(reason).startswith("adapter_")
                for reason in validation_errors
            ),
        })
        failure_signature = tuple(sorted(set(validation_errors)))
        if failure_signature and failure_signature == previous_failure_signature:
            if (
                session.escalation_enabled and not session.escalation_used
                and repair_attempt + 1 < max_attempts
            ):
                session.escalation_pending = True
                if log_callback:
                    emit_progress_log(
                        log_callback,
                        "editor_escalation_scheduled",
                        "Senior Editor state: one configured escalation will review the unchanged validation failure.",
                        level="warning",
                        layer="senior_editor_repair",
                    )
                previous_failure_signature = None
                continue
            if log_callback:
                emit_progress_log(
                    log_callback,
                    "repair_no_progress",
                    "Senior Editor state: stopping automatic repair because the validation findings did not improve.",
                    level="warning",
                    layer="senior_editor_repair",
                    data={"attempt": repair_attempt + 1},
                )
            break
        previous_failure_signature = failure_signature

    outcome.validation_errors = validation_errors
    return outcome
