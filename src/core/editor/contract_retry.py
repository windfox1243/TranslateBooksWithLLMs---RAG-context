"""Recover a Senior Editor answer that broke the reflection contract.

Two steps, run in this order by the reflection pass when the editor's
answer is malformed or incomplete:

1. `retry_invalid_contract` asks once more. Issues that located their span
   but omitted the replacement are asked only for that missing text;
   otherwise the whole reflection is requested again under the strict
   JSON contract.
2. `salvage_partial_contract` keeps whatever is still usable: valid issues
   stay, and issues without a replacement become review evidence instead
   of a reason to resend the whole chunk.

Moved out of src/core/translator.py's reflection pass; the conditions and
the order of every step are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

_MISSING_REPLACEMENT_SYSTEM_PROMPT = (
    "You supply the missing replacement text for editor issues. "
    "Return one JSON object only. Use only the supplied issues and "
    "candidate draft segments; never rewrite or quote the complete "
    "chunk."
)


@dataclass
class ContractRetryOutcome:
    """The reflection result and critique after the retry.

    `replaced` is True when the retry answered in place of the first
    attempt (rather than correcting part of it), so it is now what the
    editor reported and the caller recounts the reported issues.
    """

    reflection_result: Any
    critique: str
    replaced: bool = False


def _requires_missing_replacement(issue: Dict[str, Any]) -> bool:
    from src.core import translator as _translator

    return (
        _translator._issue_requires_draft_replacement(issue)
        and not issue.get("draft_replacement")
    )


async def retry_invalid_contract(
    *,
    session: Any,
    reflection_result: Any,
    critique: str,
    draft_translation: str,
    reflection_pair: Any,
    options: Dict[str, Any],
    log_callback: Optional[Callable] = None,
) -> ContractRetryOutcome:
    """Ask the editor once more for a usable reflection contract."""
    from src.core import translator as _translator
    from src.prompts.prompts import (
        REFLECTION_CONTRACT_VERSION,
        REFLECTION_PROMPT_VERSION,
    )
    from src.utils.progress_logging import emit_progress_log

    ReflectionResult = _translator.ReflectionResult
    outcome = ContractRetryOutcome(reflection_result, critique)

    malformed_issue_ids = {
        str(issue.get("issue_id") or "")
        for issue in reflection_result.issues
        if _requires_missing_replacement(issue)
    }
    if log_callback:
        emit_progress_log(
            log_callback,
            "reflection_parse_retry",
            "Senior Editor returned malformed or incomplete JSON; retrying once with the strict editor contract.",
            layer="senior_editor_reflection",
            data={
                "prompt_version": REFLECTION_PROMPT_VERSION,
                "contract_version": REFLECTION_CONTRACT_VERSION,
            },
        )
    if malformed_issue_ids:
        # These issues located their span and stopped short of naming the
        # repair, so asking them to correct a locator asks the wrong
        # question and gets the same answer back.
        retry_user_prompt = _translator._build_missing_replacement_retry_prompt(
            draft_translation,
            reflection_result.issues,
            malformed_issue_ids,
        )
        retry_system_prompt = _MISSING_REPLACEMENT_SYSTEM_PROMPT
    else:
        retry_user_prompt = (
            f"{reflection_pair.user}\n\n"
            "The previous response was not a usable JSON contract. Return "
            "one valid JSON object only, without prose or markdown."
        )
        retry_system_prompt = reflection_pair.system
    try:
        retry_response = await session.generate(
            prompt=retry_user_prompt,
            system_prompt=retry_system_prompt,
            temperature=0.0,
            max_output_tokens=session.output_tokens,
            stage="reflection_contract_retry",
            structured=True,
        )
        retry_critique = (
            (retry_response.content or "").strip()
            if retry_response and getattr(retry_response, "content", None)
            else ""
        )
        retry_result = _translator.parse_reflection_result(retry_critique)
        session.record_attempt(
            "reflection_contract_retry", retry_response, retry_critique,
            parse_status=retry_result.parse_status,
            failure_class=(
                "contract_incomplete"
                if _translator._reflection_contract_incomplete(retry_result)
                else "contract_parse"
                if retry_result.parse_status in {"invalid_json", "empty"}
                else ""
            ),
            issues=retry_result.issues,
        )
        _translator.log_full_editor_response(
            log_callback, "reflection_contract_retry", retry_critique,
            int(options.get("chunk_index", 0) or 0),
        )
        if log_callback:
            emit_progress_log(
                log_callback,
                "reflection_parse_retry_status",
                f"Senior Editor reflection retry parsed with {retry_result.parse_status}.",
                layer="senior_editor_reflection",
                data={
                    "prompt_version": REFLECTION_PROMPT_VERSION,
                    "contract_version": REFLECTION_CONTRACT_VERSION,
                    "parse_status": retry_result.parse_status,
                },
            )
        if retry_result.parse_status == "json" and retry_result.issues:
            if malformed_issue_ids:
                corrected_by_id = {
                    str(issue.get("issue_id") or ""): issue
                    for issue in retry_result.issues
                    if str(issue.get("issue_id") or "") in malformed_issue_ids
                }
                merged = [
                    corrected_by_id.get(str(issue.get("issue_id") or ""), issue)
                    for issue in reflection_result.issues
                ]
                outcome.reflection_result = ReflectionResult(
                    "needs_repair", merged, retry_critique, "json",
                    retry_result.voice_observations
                    or reflection_result.voice_observations,
                )
            elif not _translator._reflection_contract_invalid(retry_result):
                # The retry answered in place of the first attempt rather than
                # correcting part of it, so it is now what the editor reported.
                outcome.reflection_result = retry_result
                outcome.replaced = True
            outcome.critique = retry_critique
    except Exception as e:
        if log_callback:
            emit_progress_log(
                log_callback,
                "reflection_parse_retry_failed",
                f"Senior Editor reflection JSON retry failed: {e}",
                layer="senior_editor_reflection",
            )
    return outcome


@dataclass
class ContractSalvage:
    """What salvaging a still-invalid contract changed.

    `review_required` is None when nothing was salvaged (the caller keeps
    its flag); otherwise it replaces the caller's contract_review_required.
    The two increments are added to the caller's review and warning counts.
    """

    reflection_result: Any
    review_required: Optional[bool] = None
    review_increment: int = 0
    warning_increment: int = 0


def salvage_partial_contract(reflection_result: Any, *, log_callback=None) -> ContractSalvage:
    """Keep the usable part of a reflection whose contract is still invalid."""
    from src.core import translator as _translator
    from src.utils.progress_logging import emit_progress_log

    ReflectionResult = _translator.ReflectionResult
    invalid_issues = [
        issue for issue in reflection_result.issues
        if _requires_missing_replacement(issue)
    ]
    valid_issues = [
        issue for issue in reflection_result.issues
        if not _requires_missing_replacement(issue)
    ]
    invalid_issue_count = len(invalid_issues)
    deterministic_invalid = sum(
        1 for issue in invalid_issues if issue.get("deterministic")
    )
    if valid_issues and reflection_result.parse_status == "json":
        salvage = ContractSalvage(
            ReflectionResult(
                "needs_repair",
                valid_issues,
                reflection_result.raw_text,
                reflection_result.parse_status,
                reflection_result.voice_observations,
            ),
            review_required=deterministic_invalid > 0,
            review_increment=deterministic_invalid,
            warning_increment=invalid_issue_count - deterministic_invalid,
        )
        if log_callback:
            emit_progress_log(
                log_callback,
                "reflection_partial_contract",
                (
                    "Senior Editor kept valid issues and moved "
                    f"{invalid_issue_count} incomplete issue(s) to review."
                ),
                level="warning",
                layer="senior_editor_reflection",
            )
        return salvage
    if invalid_issues and reflection_result.parse_status == "json":
        # Preserve a semantically useful concern when the model cannot
        # provide a safe exact edit after the focused correction. It is
        # review evidence, not a reason to resend the complete chunk.
        # No warning increment: these issues stay in the result as
        # review_only and are counted once by the actionable filter.
        return ContractSalvage(
            ReflectionResult(
                "needs_repair",
                [
                    {
                        **issue,
                        "repair_kind": "review_only",
                        "draft_replacement": None,
                    }
                    for issue in invalid_issues
                ],
                reflection_result.raw_text,
                reflection_result.parse_status,
                reflection_result.voice_observations,
            ),
            review_required=deterministic_invalid > 0,
            review_increment=deterministic_invalid,
        )
    return ContractSalvage(reflection_result)
