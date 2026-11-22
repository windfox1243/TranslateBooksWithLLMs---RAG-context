"""One Senior Editor run's model requests and their accounting.

`EditorRequestSession` owns everything about talking to the editor model
for one chunk: which client, model and thinking controls to use, the
request budget, the native-schema / tagged fallback, the truncation retry,
the one-shot escalation to a stronger model, and the token and diagnostics
accounting that ends in `finish`. The reflection pass in
src/core/translator.py decides *what* to ask; this class decides *how*,
and keeps the counters out of that function.

The provider call itself and the schema-capability caches stay in
src/core/translator.py (`_generate_editor_response`,
`_EDITOR_SCHEMA_UNSUPPORTED`, `_EDITOR_SCHEMA_GENERIC_REJECTIONS`) and are
looked up there at call time, so tests and narrator_voice that patch or
clear them keep acting on the same objects.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional

from src.core.llm.call_ledger import PURPOSE_EDITOR, llm_purpose

# Requests one run may send, retries and fallbacks included.
EDITOR_REQUEST_BUDGET = 6

_THINKING_KEYS = (
    "thinking_level", "thinking_budget", "thinking_enabled", "reasoning_effort",
)


def _thinking_kwargs(controls: Dict[str, Any]) -> Dict[str, Any]:
    return {key: controls.get(key) for key in _THINKING_KEYS}


class EditorRequestSession:
    """Client selection, request budget, fallbacks and accounting for one run."""

    def __init__(
        self,
        *,
        options: Dict[str, Any],
        llm_client: Any,
        model_name: str,
        target_language: str,
        log_callback: Optional[Callable] = None,
    ) -> None:
        from src.core.editor.unit_repair import unit_mode_enabled
        from src.core.llm.generation_controls import (
            resolve_editor_output_tokens,
            resolve_thinking_controls,
        )
        from src.prompts.prompts import (
            REFLECTION_CONTRACT_VERSION,
            REFLECTION_PROMPT_VERSION,
            UNIT_CONTRACT_VERSION,
            UNIT_PROMPT_VERSION,
        )
        from src.utils.editor_diagnostics import EditorRunRecorder

        self.options = options
        self.log_callback = log_callback

        self.client = options.get("_editor_llm_client") or llm_client
        self.model = str(options.get("editor_model_resolved") or model_name)
        self.provider = str(
            options.get("editor_provider_resolved")
            or options.get("llm_provider")
            or "unknown"
        ).casefold()
        self.endpoint = str(options.get("editor_api_endpoint") or "")
        self.controls = resolve_thinking_controls(
            self.provider,
            self.model,
            options.get("editor_thinking_level") or "auto",
            role="editor",
            endpoint=self.endpoint,
            reasoning_supported=bool(options.get("editor_reasoning_supported")),
        )
        self.output_limit = options.get("editor_model_output_limit")
        self.output_tokens = resolve_editor_output_tokens(
            self.provider,
            self.model,
            options.get("editor_max_output_tokens"),
            self.controls.get("mode"),
            reported_limit=self.output_limit,
        )

        self.escalation_client = options.get("_editor_escalation_llm_client")
        self.escalation_model = str(options.get("editor_escalation_model") or "").strip()
        self.escalation_enabled = bool(
            options.get("editor_escalation_enabled", False)
            and self.escalation_client
            and self.escalation_model
        )
        # Set by the reflection pass when a repair failed the same way twice;
        # the next repair request then goes to the escalation model once.
        self.escalation_pending = False
        self.escalation_used = False

        self.schema_capability_key = (
            self.provider,
            self.model.casefold(),
            self.endpoint.rstrip("/").casefold(),
        )
        # Which contract the editor was actually asked to answer. The
        # diagnostics compare runs against each other, and two contracts
        # recorded under one version would make that comparison meaningless.
        self.unit_mode = unit_mode_enabled(options)
        self.recorder = EditorRunRecorder(
            options,
            target_language=target_language,
            prompt_version=(
                UNIT_PROMPT_VERSION if self.unit_mode else REFLECTION_PROMPT_VERSION
            ),
            contract_version=(
                UNIT_CONTRACT_VERSION if self.unit_mode else REFLECTION_CONTRACT_VERSION
            ),
        )

        self.request_index = 0
        self.request_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_thinking_tokens = 0
        self.total_tokens = 0
        self.truncation_retry_stages: set[str] = set()
        self.any_truncation = False
        self.recovered_truncation = False
        self.last_finish_reason = ""
        self.last_blocked_reason = ""
        self.request_compositions: List[Dict[str, Any]] = []
        self.reflection_components: Dict[str, Any] = {}

    # --- accounting -------------------------------------------------------

    def capture_request_composition(
        self, stage: str, prompt: str, system_prompt: str,
    ) -> None:
        self.request_compositions.append({
            "stage": stage,
            "system_chars": len(system_prompt),
            "user_chars": len(prompt),
            "estimated_input_tokens": (len(system_prompt) + len(prompt) + 3) // 4,
            "prompt_hash": hashlib.sha256(
                (system_prompt + "\x1f" + prompt).encode("utf-8")
            ).hexdigest(),
        })

    def record_attempt(
        self,
        stage: str,
        response: Any,
        raw: str,
        *,
        parse_status: str = "",
        failure_class: str = "",
        reason_codes: Optional[List[str]] = None,
        issues: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        from src.utils.editor_diagnostics import issue_excerpts, response_hash

        self.request_index += 1
        prompt_tokens = int(getattr(response, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response, "completion_tokens", 0) or 0)
        thinking_tokens = int(getattr(response, "thinking_tokens", 0) or 0)
        attempt_total = int(
            getattr(response, "total_tokens", 0)
            or prompt_tokens + completion_tokens + thinking_tokens
        )
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_thinking_tokens += thinking_tokens
        self.total_tokens += attempt_total
        attempt_truncated = getattr(response, "was_truncated", False) is True
        self.any_truncation = self.any_truncation or attempt_truncated
        finish_reason = str(getattr(response, "finish_reason", "") or "")
        blocked_reason = str(getattr(response, "blocked_reason", "") or "")
        if finish_reason:
            self.last_finish_reason = finish_reason
        if blocked_reason:
            self.last_blocked_reason = blocked_reason
        self.recorder.attempt({
            "attempt_index": self.request_index,
            "stage": stage,
            "parse_status": parse_status,
            "failure_class": failure_class,
            "reason_codes": reason_codes or [],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": attempt_total,
            "was_truncated": attempt_truncated,
            "finish_reason": finish_reason,
            "blocked_reason": blocked_reason,
            "response_hash": response_hash(raw),
            "excerpts": issue_excerpts(issues or []),
        })

    def finish(
        self,
        outcome: str,
        *,
        defaults: Dict[str, Any],
        narrator_conformance: Dict[str, Any],
        automatic_repair_attempts: int,
        max_automatic_repair_attempts: int,
        deterministic_count: int,
        payload: Dict[str, Any],
    ) -> None:
        """Record the run's verdict; `defaults` fill whatever `payload` omits."""
        from src.core.editor.signal_monitor import warn_if_editor_inert

        for key, value in defaults.items():
            payload.setdefault(key, value)
        diagnostics = dict(payload.get("diagnostics") or {})
        diagnostics.setdefault("narrator_conformance", narrator_conformance)
        diagnostics.setdefault("prompt_composition", {
            **self.reflection_components,
            "requests": self.request_compositions,
        })
        diagnostics.setdefault("automatic_retry", {
            "trigger": (
                "deterministic_validation"
                if automatic_repair_attempts else "none"
            ),
            "current_attempt": automatic_repair_attempts,
            "maximum_attempts": max_automatic_repair_attempts,
            "terminal_reason": outcome,
        })
        payload["diagnostics"] = diagnostics
        self.recorder.finish(
            outcome,
            prompt_tokens=self.total_prompt_tokens,
            completion_tokens=self.total_completion_tokens,
            thinking_tokens=self.total_thinking_tokens,
            total_tokens=self.total_tokens,
            was_truncated=self.any_truncation,
            recovered_truncation=self.recovered_truncation,
            finish_reason=self.last_finish_reason,
            blocked_reason=self.last_blocked_reason,
            deterministic_count=deterministic_count,
            **payload,
        )
        # An editor that has gone inert reports `no_issues` forever, which is
        # indistinguishable from a clean book unless someone says so.
        warn_if_editor_inert(
            self.recorder.db,
            str(self.options.get("translation_id") or ""),
            self.log_callback,
        )

    # --- requests ---------------------------------------------------------

    @llm_purpose(PURPOSE_EDITOR)
    async def generate(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_output_tokens: int,
        stage: str,
        structured: bool = False,
        fallback_prompt: Optional[str] = None,
        fallback_system_prompt: Optional[str] = None,
    ) -> Any:
        """Send one editor request with its schema fallback and truncation retry."""
        from src.core import translator as _translator
        from src.core.llm.exceptions import StructuredOutputSchemaError
        from src.core.llm.generation_controls import resolve_thinking_controls
        from src.prompts.prompts import REFLECTION_RESPONSE_SCHEMA, UNIT_RESPONSE_SCHEMA

        options = self.options
        active_client = self.client
        active_model = self.model
        active_controls = self.controls
        if self.escalation_pending and stage.startswith("repair_"):
            active_client = self.escalation_client
            active_model = self.escalation_model
            active_controls = resolve_thinking_controls(
                str(options.get("editor_escalation_provider") or self.provider),
                active_model,
                options.get("editor_escalation_thinking_level") or "minimal",
                role="editor",
                endpoint=str(options.get("editor_escalation_endpoint") or ""),
            )
            self.escalation_pending = False
            self.escalation_used = True
        schema = (
            (UNIT_RESPONSE_SCHEMA if self.unit_mode else REFLECTION_RESPONSE_SCHEMA)
            if structured and options.get("editor_native_schema", True)
            and self.schema_capability_key not in _translator._EDITOR_SCHEMA_UNSUPPORTED
            else None
        )
        disable_native_schema = False
        active_prompt = prompt if schema is not None else (fallback_prompt or prompt)
        active_system_prompt = (
            system_prompt
            if schema is not None
            else (fallback_system_prompt or system_prompt)
        )

        try:
            if self.request_count >= EDITOR_REQUEST_BUDGET:
                raise RuntimeError("Senior Editor request budget exhausted")
            self.request_count += 1
            self.capture_request_composition(stage, active_prompt, active_system_prompt)
            response = await _translator._generate_editor_response(
                llm_client=active_client,
                prompt=active_prompt,
                system_prompt=active_system_prompt,
                model_name=active_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_schema=schema,
                stage=stage,
                **_thinking_kwargs(active_controls),
            )
            if response is None:
                raise RuntimeError("Senior Editor provider returned no response")
            return await self._retry_truncated(
                response,
                schema,
                stage=stage,
                structured=structured,
                max_output_tokens=max_output_tokens,
                active_client=active_client,
                active_model=active_model,
                active_prompt=active_prompt,
                active_system_prompt=active_system_prompt,
            )
        except StructuredOutputSchemaError as exc:
            if schema is None:
                raise
            if getattr(exc, "identifies_schema", True):
                disable_native_schema = True
            else:
                seen = _translator._EDITOR_SCHEMA_GENERIC_REJECTIONS.get(
                    self.schema_capability_key, 0
                ) + 1
                _translator._EDITOR_SCHEMA_GENERIC_REJECTIONS[
                    self.schema_capability_key
                ] = seen
                disable_native_schema = (
                    seen >= _translator._EDITOR_SCHEMA_GENERIC_LIMIT
                )
            self.record_attempt(
                f"{stage}_schema_fallback", None, "",
                failure_class="schema_rejected",
                reason_codes=[
                    f"native_schema_rejected:{type(exc).__name__}",
                    *(["native_schema_disabled"] if disable_native_schema else []),
                ],
            )
        if disable_native_schema:
            _translator._EDITOR_SCHEMA_UNSUPPORTED.add(self.schema_capability_key)
        if self.request_count >= EDITOR_REQUEST_BUDGET:
            raise RuntimeError("Senior Editor request budget exhausted")
        self.request_count += 1
        self.capture_request_composition(
            f"{stage}_tagged_fallback",
            fallback_prompt or prompt,
            fallback_system_prompt or system_prompt,
        )
        response = await _translator._generate_editor_response(
            llm_client=active_client,
            prompt=fallback_prompt or prompt,
            system_prompt=fallback_system_prompt or system_prompt,
            model_name=active_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_schema=None,
            stage=f"{stage}_tagged_fallback",
            **_thinking_kwargs(active_controls),
        )
        if response is not None:
            try:
                response.structured_output_fallback = True
            except Exception:
                pass
        # The retry repeats the request that was cut off, which here is the
        # tagged fallback: the schema-mode prompt without its schema would
        # ask for a contract nothing enforces.
        return await self._retry_truncated(
            response,
            None,
            stage=stage,
            structured=structured,
            max_output_tokens=max_output_tokens,
            active_client=active_client,
            active_model=active_model,
            active_prompt=fallback_prompt or prompt,
            active_system_prompt=fallback_system_prompt or system_prompt,
        )

    async def _retry_truncated(
        self,
        response: Any,
        active_schema: Any,
        *,
        stage: str,
        structured: bool,
        max_output_tokens: int,
        active_client: Any,
        active_model: str,
        active_prompt: str,
        active_system_prompt: str,
    ) -> Any:
        """Retry once, with a larger allowance and less reasoning, if cut off."""
        from src.core import translator as _translator
        from src.core.llm.generation_controls import (
            adaptive_retry_output_tokens,
            resolve_thinking_controls,
        )
        from src.utils.progress_logging import emit_progress_log

        if (
            response is None
            or getattr(response, "was_truncated", False) is not True
            or stage in self.truncation_retry_stages
        ):
            return response
        self.truncation_retry_stages.add(stage)
        raw = str(getattr(response, "content", "") or "")
        self.record_attempt(
            f"{stage}_max_tokens",
            response,
            raw,
            parse_status=(
                _translator.parse_reflection_result(raw).parse_status
                if structured else ""
            ),
            failure_class="provider_truncated",
            reason_codes=["adaptive_output_retry"],
        )
        retry_tokens = adaptive_retry_output_tokens(
            max_output_tokens,
            self.output_limit,
        )
        retry_controls = resolve_thinking_controls(
            self.provider,
            self.model,
            "minimal",
            role="editor",
            endpoint=self.endpoint,
            reasoning_supported=bool(self.options.get("editor_reasoning_supported")),
        )
        if self.log_callback:
            emit_progress_log(
                self.log_callback,
                "editor_truncation_retry",
                (
                    "Senior Editor exhausted its output allowance; retrying "
                    f"with {retry_tokens} tokens and reduced reasoning."
                ),
                level="warning",
                layer="senior_editor_reflection",
                data={
                    "previous_output_tokens": max_output_tokens,
                    "retry_output_tokens": retry_tokens,
                    "thinking_mode": retry_controls.get("mode"),
                },
            )
        if self.request_count >= EDITOR_REQUEST_BUDGET:
            return response
        self.request_count += 1
        self.capture_request_composition(
            f"{stage}_max_tokens_retry", active_prompt, active_system_prompt,
        )
        retried = await _translator._generate_editor_response(
            llm_client=active_client,
            prompt=active_prompt,
            system_prompt=active_system_prompt,
            model_name=active_model,
            temperature=0.0,
            max_output_tokens=retry_tokens,
            response_schema=active_schema,
            stage=f"{stage}_max_tokens_retry",
            **_thinking_kwargs(retry_controls),
        )
        if retried is not None and getattr(retried, "was_truncated", False) is not True:
            self.recovered_truncation = True
        return retried
