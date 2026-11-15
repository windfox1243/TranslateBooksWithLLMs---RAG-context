"""Focused retry for local editor patches that could not be located.

When the editor's exact-span edits do not land (a quote that is not in the
draft, or occurs twice), the reflection pass asks again for just those
issues, showing the candidate neighborhoods, up to three times. Each attempt
is told what the previous one was rejected for, and a byte-identical answer
ends the loop.

Moved out of src/core/translator.py's reflection pass; the conditions and
the order of every step are unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

_SYSTEM_PROMPT = (
    "You correct one bounded set of unresolved local translation "
    "edits. Use only the supplied candidate neighborhoods. Return "
    "canonical reflection JSON and never rewrite the complete chunk."
)
MAX_FOCUSED_ATTEMPTS = 3


@dataclass
class LocatorRetryOutcome:
    """What the focused retry ended with.

    On success `patched_draft` is the draft with every retried issue
    applied, `resolved_issues` the issues that account for it (corrected
    ones plus those the draft already read), and `raw` / `parse_status` /
    `voice_observations` describe the answer that succeeded. `patch_errors`
    is the caller's list as it stands after the retry (empty on success).
    """

    succeeded: bool = False
    patched_draft: str = ""
    resolved_issues: List[Dict[str, Any]] = field(default_factory=list)
    raw: str = ""
    parse_status: str = ""
    voice_observations: Any = None
    patch_errors: List[str] = field(default_factory=list)
    already_read_ids: List[str] = field(default_factory=list)


def _stage(attempt: int) -> str:
    return "local_patch_retry" if attempt == 1 else f"local_patch_retry_{attempt}"


async def run_focused_locator_retry(
    *,
    session: Any,
    followup_base: str,
    unresolved_issues: List[Dict[str, Any]],
    patch_errors: List[str],
    source_chunk: str,
    source_language: str,
    target_language: str,
    protected_terms: List[str],
    glossary_terms: Dict[str, Any],
) -> LocatorRetryOutcome:
    """Ask again for the unresolved local edits, at most three times."""
    from src.core import translator as _translator
    from src.utils.translation_quality import (
        apply_local_editor_patches,
        normalize_unique_issue_locators,
        replacement_already_in_draft,
        validate_editor_repair,
        validate_issue_locators,
    )

    ReflectionResult = _translator.ReflectionResult
    patch_errors = list(patch_errors)
    outcome = LocatorRetryOutcome()

    followup_ids = {
        str(issue.get("issue_id") or "") for issue in unresolved_issues
        if str(issue.get("issue_id") or "")
    }
    followup_reasons = [
        *patch_errors,
        *[
            f"local_patch_unresolved:{issue.get('issue_id') or 'unknown'}"
            for issue in unresolved_issues
        ],
    ]
    previous_response_fingerprint = None
    identical_no_progress = 0
    # What the last attempt was rejected for. Without it the three attempts
    # are one question asked three times: every input to the prompt is
    # fixed before the loop, so the second and third calls resample the
    # same request at temperature zero and learn nothing from the answer
    # that just failed.
    previous_attempt_errors: List[str] = []
    for focused_attempt in range(1, MAX_FOCUSED_ATTEMPTS + 1):
        followup_prompt = _translator._build_focused_locator_retry_prompt(
            followup_base,
            unresolved_issues,
            followup_ids,
            followup_reasons,
            previous_attempt_errors=previous_attempt_errors,
            previous_attempt_index=focused_attempt - 1,
        )
        followup_errors: List[str] = []
        followup_response = None
        followup_raw = ""
        followup_result = ReflectionResult("no_issues", [], "", "empty")
        try:
            followup_response = await session.generate(
                prompt=followup_prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=session.output_tokens,
                stage=_stage(focused_attempt),
                structured=True,
            )
            followup_raw = str(
                getattr(followup_response, "content", "") or ""
            ).strip()
            response_fingerprint = hashlib.sha256(
                followup_raw.encode("utf-8")
            ).hexdigest()
            if response_fingerprint == previous_response_fingerprint:
                identical_no_progress += 1
            else:
                identical_no_progress = 0
            previous_response_fingerprint = response_fingerprint
            followup_result = _translator.parse_reflection_result(followup_raw)
            corrected_issues = [
                issue for issue in followup_result.issues
                if not followup_ids
                or str(issue.get("issue_id") or "") in followup_ids
            ]
            corrected_issues, _normalized = normalize_unique_issue_locators(
                followup_base, corrected_issues,
            )
            # The audit screens these out, but a retry asked to relocate an
            # edit can answer with one the draft already reads. It asks for
            # nothing, so it is satisfied rather than unresolved: held out of
            # the batch, counted as done, and named in the record.
            already_read = [
                issue for issue in corrected_issues
                if replacement_already_in_draft(followup_base, issue)
            ]
            outcome.already_read_ids.extend(
                str(issue.get("issue_id") or "unknown") for issue in already_read
            )
            corrected_issues = [
                issue for issue in corrected_issues if issue not in already_read
            ]
            if _translator._reflection_contract_invalid(followup_result) or not (
                corrected_issues or already_read
            ):
                followup_errors.append("local_patch_retry_contract_invalid")
            else:
                followup_errors.extend(
                    validate_issue_locators(followup_base, corrected_issues)
                )
            retry_patched = followup_base
            retry_unresolved = list(corrected_issues)
            if not followup_errors:
                retry_patched, retry_unresolved, retry_patch_errors = (
                    apply_local_editor_patches(followup_base, corrected_issues)
                )
                followup_errors.extend(retry_patch_errors)
            if not followup_errors and not retry_unresolved:
                followup_errors.extend(validate_editor_repair(
                    retry_patched,
                    corrected_issues,
                    draft_text=followup_base,
                    source_text=source_chunk,
                    source_language=source_language,
                    target_language=target_language,
                    protected_terms=protected_terms,
                    glossary_terms=glossary_terms,
                ))
            session.record_attempt(
                _stage(focused_attempt),
                followup_response,
                followup_raw,
                parse_status=followup_result.parse_status,
                failure_class=("local_patch_conflict" if followup_errors else ""),
                reason_codes=followup_errors,
                issues=corrected_issues + already_read,
            )
            if not followup_errors and not retry_unresolved:
                outcome.succeeded = True
                outcome.patched_draft = retry_patched
                outcome.resolved_issues = corrected_issues + already_read
                outcome.raw = followup_raw
                outcome.parse_status = followup_result.parse_status
                outcome.voice_observations = followup_result.voice_observations
                outcome.patch_errors = []
                return outcome
            patch_errors.extend(
                error for error in followup_errors if error not in patch_errors
            )
            # An attempt can fail without erroring: it answers cleanly and
            # leaves issues unresolved. Measured, that is the case that put
            # the third attempt back on the first attempt's exact prompt.
            previous_attempt_errors = list(followup_errors) or [
                f"local_patch_unresolved:{issue.get('issue_id') or 'unknown'}"
                for issue in retry_unresolved
            ]
            if identical_no_progress >= 1:
                patch_errors.append("local_patch_retry_no_progress")
                break
        except Exception as exc:
            followup_errors.append(
                f"local_patch_retry_failed:{type(exc).__name__}"
            )
            patch_errors.extend(
                error for error in followup_errors if error not in patch_errors
            )
            session.record_attempt(
                _stage(focused_attempt),
                followup_response,
                followup_raw,
                parse_status=followup_result.parse_status,
                failure_class="local_patch_conflict",
                reason_codes=followup_errors,
                issues=followup_result.issues,
            )
            break

    outcome.patch_errors = patch_errors
    return outcome
