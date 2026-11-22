"""
OpenRouter provider implementation.

This module provides the OpenRouterProvider class for interacting with
OpenRouter's API, which provides access to 200+ models.

Features:
    - Access to 200+ models (Claude, GPT-4, Llama, Mistral, etc.)
    - Built-in cost tracking
    - Model validation
    - Automatic context size detection
    - Reasoning disabled by default (translation-friendly), per model metadata
"""

import asyncio
import json
import threading
import time
import weakref
from typing import Any, Callable, Dict, List, Optional, Union

import httpx

from src.config import (
    MAX_TRANSLATION_ATTEMPTS,
    OPENROUTER_DISABLE_THINKING,
    REQUEST_TIMEOUT,
)

from ..base import (
    LLMGenerationOptions,
    LLMProvider,
    LLMResponse,
    terminal_provider_failure,
)
from ..exceptions import (
    ContextOverflowError,
    RateLimitError,
    StructuredOutputSchemaError,
)
from ..rate_limit_handler import handle_rate_limit, is_retryable_http_status
from ..utils.error_classification import is_context_overflow_error


def _status_from_error_code(code) -> Optional[int]:
    """The HTTP-style status in an OpenRouter body error's ``code``, if any."""
    status: Optional[int] = None
    if isinstance(code, int) and not isinstance(code, bool):
        status = code
    elif isinstance(code, str) and code.strip().isdigit():
        status = int(code.strip())
    if status is not None and not 400 <= status <= 599:
        return None
    return status


class OpenRouterProvider(LLMProvider):
    """
    Provider for OpenRouter API.

    OpenRouter provides unified access to multiple LLM providers including:
        - Anthropic (Claude)
        - OpenAI (GPT-4, GPT-3.5)
        - Meta (Llama)
        - Google (Gemini, PaLM)
        - Mistral AI
        - And 200+ more models

    Features:
        - Automatic model validation
        - Per-request cost tracking
        - Session cost accumulation
        - Context size detection

    Configuration:
        endpoint: https://openrouter.ai/api/v1/chat/completions
        model: Model identifier (e.g., "anthropic/claude-3-opus")
        api_key: OpenRouter API key

    Example:
        >>> provider = OpenRouterProvider(
        ...     api_key="sk-or-...",
        ...     model="anthropic/claude-3-opus"
        ... )
        >>> response = await provider.generate("Translate: Hello")
        >>> print(f"Cost: ${provider.get_session_cost()}")
    """

    # OpenRouter API endpoints
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODELS_URL = "https://openrouter.ai/api/v1/models"

    # Session cost tracking (class-level)
    _session_cost = 0.0
    _session_tokens = {"prompt": 0, "completion": 0}
    _cost_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    # Fallback text-only models (sorted by cost, cheapest first)
    FALLBACK_MODELS = [
        # === CHEAP MODELS ===
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "mistralai/mistral-small-24b-instruct-2501",
        # === MID-TIER MODELS ===
        "anthropic/claude-3-5-haiku-20241022",
        "openai/gpt-4o-mini",
        "google/gemini-1.5-pro",
        "deepseek/deepseek-chat",
        # === PREMIUM MODELS ===
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet-20241022",
    ]

    # --- Reasoning override ------------------------------------------------
    # OpenRouter exposes one unified `reasoning` request parameter, and
    # /api/v1/models describes each model's reasoning support:
    #   "reasoning": {"mandatory": bool, "default_enabled": bool,
    #                 "supported_efforts": [...], "default_effort": "..."}
    # plus "reasoning" in `supported_parameters`. Many models reason by default
    # (DeepSeek V4.x at effort "high", Qwen 3.x), which multiplies billed output
    # tokens for no gain on translation. Mandatory-reasoning models reject a
    # disable request, so they get their lowest supported effort instead.

    # Effort levels ordered least-thinking first.
    REASONING_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

    # Model catalog entries, fetched once per process: {model_id: model_dict}
    # A failed fetch is retried after this many seconds.
    CATALOG_RETRY_SECONDS = 300
    _model_catalog: Optional[Dict[str, Dict[str, Any]]] = None
    # monotonic time of the last failed fetch; retried after CATALOG_RETRY_SECONDS
    _model_catalog_failed_at: Optional[float] = None
    # One asyncio.Lock per event loop: each web job runs on its own loop, and a
    # lock is bound to the loop that first waits on it, so a single class-level
    # lock failed for every job after the first. Guarded by a thread lock.
    _model_catalog_loop_locks: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
    _model_catalog_state_lock = threading.Lock()
    # Resolved override per model: {model_id: reasoning dict, or {} for none}
    _reasoning_overrides: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "anthropic/claude-sonnet-4",
        disable_thinking: bool = OPENROUTER_DISABLE_THINKING
    ):
        """
        Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (default: anthropic/claude-sonnet-4)
            disable_thinking: Turn reasoning off (or down to the lowest effort
                on models where it is mandatory). Reasoning tokens are billed
                as output: DeepSeek V4.1 Flash was seen producing 4.7k-31k
                output tokens for ~3k-token translation chunks with it on.
        """
        super().__init__(model, api_keys=api_key, provider_name="openrouter")
        self.disable_thinking = disable_thinking
        self._warned_reasoning_tokens = False

    @staticmethod
    def _catalog_failed_recently() -> bool:
        """True while a failed catalog fetch is too recent to retry.

        A transient failure must not disable the overrides for the life of
        the process (reasoning would stay on, and billed, for every job).
        """
        failed_at = OpenRouterProvider._model_catalog_failed_at
        return (failed_at is not None
                and time.monotonic() - failed_at < OpenRouterProvider.CATALOG_RETRY_SECONDS)

    async def _load_model_catalog(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Fetch OpenRouter's model catalog (reasoning metadata included).

        Fetched once per process and cached class-side. Returns None when the
        catalog is unreachable, which means "unknown" and not "no models";
        the fetch is retried once CATALOG_RETRY_SECONDS have passed.
        """
        if OpenRouterProvider._model_catalog is not None:
            return OpenRouterProvider._model_catalog
        if OpenRouterProvider._catalog_failed_recently():
            return None

        async with OpenRouterProvider._catalog_lock_for_running_loop():
            if OpenRouterProvider._model_catalog is not None:
                return OpenRouterProvider._model_catalog
            if OpenRouterProvider._catalog_failed_recently():
                return None
            try:
                client = await self._get_client()
                response = await client.get(
                    self.MODELS_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30
                )
                response.raise_for_status()
                OpenRouterProvider._model_catalog_failed_at = None
                OpenRouterProvider._model_catalog = {
                    m["id"]: m for m in response.json().get("data", []) if m.get("id")
                }
                return OpenRouterProvider._model_catalog
            except Exception as e:
                print(f"[OpenRouter] WARN: could not read the model catalog ({e}); "
                      f"reasoning stays at the model default")
                OpenRouterProvider._model_catalog_failed_at = time.monotonic()
                return None

    @staticmethod
    def _catalog_lock_for_running_loop() -> asyncio.Lock:
        """The catalog fetch lock of the event loop running this coroutine."""
        loop = asyncio.get_running_loop()
        with OpenRouterProvider._model_catalog_state_lock:
            lock = OpenRouterProvider._model_catalog_loop_locks.get(loop)
            if lock is None:
                lock = asyncio.Lock()
                OpenRouterProvider._model_catalog_loop_locks[loop] = lock
            return lock

    def _pick_reasoning_override(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Map a model's catalog entry to the `reasoning` value translation wants."""
        if "reasoning" not in (model_info.get("supported_parameters") or []):
            return {}

        reasoning = model_info.get("reasoning") or {}
        if not reasoning.get("mandatory"):
            return {"enabled": False}

        # Mandatory reasoning: disabling is rejected, so ask for the lowest
        # effort the model advertises. Without an effort list, leave its default.
        supported = reasoning.get("supported_efforts") or []
        for effort in self.REASONING_EFFORT_ORDER[1:]:
            if effort in supported:
                return {"effort": effort}
        return {}

    async def _get_reasoning_override(self, force: bool = False) -> Dict[str, Any]:
        """Return the `reasoning` request value for this model ({} = send none).

        `force` asks for the disabled (or lowest mandatory) setting even when
        OPENROUTER_DISABLE_THINKING is off, for a caller that explicitly chose
        to switch reasoning off.
        """
        if not self.disable_thinking and not force:
            return {}

        cached = OpenRouterProvider._reasoning_overrides.get(self.model)
        if cached is not None:
            return dict(cached)

        catalog = await self._load_model_catalog()
        if catalog is None:
            return {}

        model_info = catalog.get(self.model)
        if model_info is None:
            print(f"[OpenRouter] WARN: '{self.model}' is not in the model catalog; "
                  f"reasoning stays at the model default")
            override = {}
        else:
            override = self._pick_reasoning_override(model_info)
            if (model_info.get("reasoning") or {}).get("mandatory"):
                print(f"[OpenRouter] WARN: '{self.model}' always reasons and cannot "
                      f"turn it off; reasoning tokens are billed as output "
                      f"(requested: {override or 'model default'})")
            elif override:
                print(f"[OpenRouter] {self.model}: reasoning disabled")

        OpenRouterProvider._reasoning_overrides[self.model] = override
        return dict(override)

    @classmethod
    def get_session_cost(cls) -> tuple:
        """
        Get the current session cost and token usage.

        Returns:
            Tuple of (total_cost_usd, token_counts_dict)
        """
        return cls._session_cost, cls._session_tokens.copy()

    @classmethod
    def reset_session_cost(cls) -> None:
        """Reset the session cost tracking."""
        cls._session_cost = 0.0
        cls._session_tokens = {"prompt": 0, "completion": 0}

    @classmethod
    def set_cost_callback(cls, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """
        Set a callback to receive cost updates after each API call.

        Args:
            callback: Function that receives a dict with:
                - request_cost: Cost of this specific request (USD)
                - session_cost: Cumulative session cost (USD)
                - prompt_tokens: Tokens used for this request's prompt
                - completion_tokens: Tokens generated in this request
                - total_prompt_tokens: Cumulative prompt tokens
                - total_completion_tokens: Cumulative completion tokens
        """
        cls._cost_callback = callback

    async def get_available_models(self, text_only: bool = True) -> list:
        """
        Fetch available OpenRouter models from API.

        Args:
            text_only: If True, filter out vision/multimodal models (default: True)

        Returns:
            List of model dicts with id, name, pricing info, sorted by price
        """
        if not self.api_key:
            return self._get_fallback_models()

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            client = await self._get_client()

            response = await client.get(
                self.MODELS_URL,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()

            models_data = response.json().get("data", [])
            filtered_models = []

            for model in models_data:
                model_id = model.get("id", "")
                architecture = model.get("architecture", {})
                modality = architecture.get("modality", "")

                if text_only:
                    if modality == "multimodal":
                        continue
                    model_id_lower = model_id.lower()
                    vision_keywords = ["vision", "vl", "-v-", "image"]
                    if any(kw in model_id_lower for kw in vision_keywords):
                        continue

                pricing = model.get("pricing", {})
                prompt_price = float(pricing.get("prompt", "0") or "0")
                completion_price = float(pricing.get("completion", "0") or "0")

                prompt_per_million = prompt_price * 1_000_000
                completion_per_million = completion_price * 1_000_000

                is_free = ":free" in model_id
                display_name = model.get("name", model_id)
                if is_free and "(free" not in display_name.lower():
                    display_name = f"{display_name} (free, 20 req/min)"

                filtered_models.append({
                    "id": model_id,
                    "name": display_name,
                    "context_length": model.get("context_length", 0),
                    "output_token_limit": (
                        model.get("top_provider", {})
                        .get("max_completion_tokens")
                    ),
                    "pricing": {
                        "prompt": prompt_price,
                        "completion": completion_price,
                        "prompt_per_million": prompt_per_million,
                        "completion_per_million": completion_per_million,
                    },
                    "total_price": prompt_price + completion_price,
                    "is_free": is_free,
                    "reasoning": model.get("reasoning"),
                })

            # Sort: paid models by price ascending, free models last (shared 20 req/min limit)
            filtered_models.sort(key=lambda x: (x["is_free"], x["total_price"]))

            if len(filtered_models) < 5:
                return self._get_fallback_models()

            return filtered_models

        except Exception as e:
            print(f"[OpenRouter] WARN: Failed to fetch models: {e}")
            return self._get_fallback_models()

    def _get_fallback_models(self) -> list:
        """Return fallback models list when API fetch fails."""
        return [{"id": m, "name": m, "pricing": {"prompt": 0, "completion": 0}}
                for m in self.FALLBACK_MODELS]

    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT,
                      system_prompt: Optional[str] = None,
                      generation_options: Optional[LLMGenerationOptions] = None
                      ) -> Optional[LLMResponse]:
        """
        Generate text using OpenRouter API with cost tracking.

        Args:
            prompt: The user prompt (content to translate)
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt (role/instructions)

        Returns:
            LLMResponse with content and token usage info, or None if failed

        Raises:
            ContextOverflowError: If input exceeds model's context window
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if generation_options and generation_options.temperature is not None:
            payload["temperature"] = generation_options.temperature
        if generation_options and generation_options.max_output_tokens:
            payload["max_tokens"] = int(generation_options.max_output_tokens)
        if generation_options and generation_options.response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "senior_editor_result",
                    "strict": True,
                    "schema": generation_options.response_schema,
                },
            }
        # An explicit effort chosen by the model-aware thinking controls wins.
        # Otherwise (no choice, or an explicit "none") reasoning is switched off
        # the way the model's catalog entry allows: {"enabled": false}, or the
        # lowest effort for models whose reasoning is mandatory.
        effort = generation_options.reasoning_effort if generation_options else None
        reasoning_override: Dict[str, Any] = {}
        if effort and effort != "none":
            payload["reasoning"] = {"effort": effort, "exclude": True}
        else:
            reasoning_override = await self._get_reasoning_override(
                force=effort == "none"
            )
            if reasoning_override:
                payload["reasoning"] = reasoning_override

        client = await self._get_client()
        # 429s have their own budget (rate_limit_events): rotating to a spare
        # key must not consume a transient-retry attempt (issue #217).
        attempt = 0
        rate_limit_events = 0
        while attempt < MAX_TRANSLATION_ATTEMPTS:
            current_key = await self._key_pool.acquire()
            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/windfox1243/TranslateBooksWithLLMs---RAG-context",
                "X-Title": "TranslateBookWithLLM",
            }
            try:
                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()

                result = response.json()

                # OpenRouter answers 200 as soon as the upstream accepts the
                # request, so a later failure arrives in the body: an error
                # object with no choices, or a choice that ended with
                # finish_reason "error". error.code carries the HTTP-style
                # status; classify like that status and never return the
                # partial text. (Ported from upstream ede8d0b0.)
                body_error = result.get("error") if isinstance(result, dict) else None
                choices = (result.get("choices") or []) if isinstance(result, dict) else []
                if isinstance(body_error, dict):
                    finish = (choices[0] or {}).get("finish_reason") if choices else None
                    if not choices or finish == "error":
                        status = _status_from_error_code(body_error.get("code"))
                        message = body_error.get("message") or str(body_error)[:300]
                        print(f"[OpenRouter] WARN: provider error in an HTTP 200 answer "
                              f"(code {status}): {message}")
                        if status == 429:
                            rate_limit_events += 1
                            await handle_rate_limit(
                                self._key_pool, current_key, {},
                                rate_limit_events, MAX_TRANSLATION_ATTEMPTS,
                            )
                            continue
                        if status is not None and not is_retryable_http_status(status):
                            return terminal_provider_failure(
                                generation_options,
                                "provider_auth" if status in {401, 403}
                                else "provider_quota" if status == 402
                                else "transport",
                                status,
                            )
                        attempt += 1
                        if attempt < MAX_TRANSLATION_ATTEMPTS:
                            await asyncio.sleep(2)
                            continue
                        return terminal_provider_failure(
                            generation_options, "transport", status,
                        )

                if "choices" not in result or len(result["choices"]) == 0:
                    print(f"[OpenRouter] WARN: Unexpected response format: {result}")
                    return terminal_provider_failure(
                        generation_options, "provider_empty"
                    )

                # NOTE: a present-but-null "content" makes .get(..., "") return None,
                # so coalesce with `or ""` to guarantee a string downstream.
                response_text = result["choices"][0].get("message", {}).get("content") or ""

                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                completion_details = usage.get("completion_tokens_details") or {}
                thinking_tokens = int(
                    completion_details.get("reasoning_tokens", 0) or 0
                )
                total_tokens = int(
                    usage.get(
                        "total_tokens", prompt_tokens + completion_tokens
                    ) or 0
                )

                if (thinking_tokens and reasoning_override.get("enabled") is False
                        and not self._warned_reasoning_tokens):
                    self._warned_reasoning_tokens = True
                    print(f"[OpenRouter] WARN: '{self.model}' still produced "
                          f"{thinking_tokens} reasoning tokens with reasoning disabled; "
                          f"the upstream provider may ignore the setting")

                if not response_text.strip():
                    print(f"[OpenRouter] WARN: Empty response from model '{self.model}' "
                          f"({prompt_tokens}+{completion_tokens} tokens). The model likely "
                          f"refused or filtered this chunk (sensitive/policy-flagged content "
                          f"or provider-side moderation). Try a different model.")

                if "cost" in result:
                    cost = float(result.get("cost", 0))
                else:
                    # Fallback estimate when OpenRouter omits cost (typical rates in USD)
                    cost = (prompt_tokens * 0.50 / 1_000_000) + (completion_tokens * 1.50 / 1_000_000)

                OpenRouterProvider._session_cost += cost
                OpenRouterProvider._session_tokens["prompt"] += prompt_tokens
                OpenRouterProvider._session_tokens["completion"] += completion_tokens

                reasoning_note = f" ({thinking_tokens} reasoning)" if thinking_tokens else ""
                print(f"[OpenRouter] {prompt_tokens}+{completion_tokens} tokens{reasoning_note} | "
                      f"Cost: ${cost:.6f} (session: ${OpenRouterProvider._session_cost:.4f})")

                if OpenRouterProvider._cost_callback:
                    try:
                        OpenRouterProvider._cost_callback({
                            "request_cost": cost,
                            "session_cost": OpenRouterProvider._session_cost,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_prompt_tokens": OpenRouterProvider._session_tokens["prompt"],
                            "total_completion_tokens": OpenRouterProvider._session_tokens["completion"],
                        })
                    except Exception as cb_err:
                        print(f"[OpenRouter] WARN: Cost callback error: {cb_err}")

                return LLMResponse(
                    content=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=total_tokens,
                    context_limit=0,  # OpenRouter manages context internally
                    was_truncated=False,
                    finish_reason=str(
                        result.get("choices", [{}])[0].get("finish_reason") or ""
                    ),
                    request_id=str(getattr(response, "headers", {}).get("x-request-id") or ""),
                    thinking_tokens=thinking_tokens,
                    total_tokens=total_tokens,
                )

            except httpx.TimeoutException as e:
                print(f"OpenRouter API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                attempt += 1
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    await asyncio.sleep(2)
                    continue
                return terminal_provider_failure(generation_options, "transport")
            except httpx.HTTPStatusError as e:
                error_body = ""
                error_message = str(e)
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    error_body = e.response.text[:500]
                    error_message = f"{e} - {error_body}"

                if e.response.status_code == 429:
                    rate_limit_events += 1
                    await handle_rate_limit(
                        self._key_pool, current_key, e.response.headers,
                        rate_limit_events, MAX_TRANSLATION_ATTEMPTS,
                    )
                    continue

                if (
                    e.response.status_code in {400, 422}
                    and generation_options
                    and generation_options.response_schema
                    and any(marker in error_message.casefold() for marker in (
                        "response_format", "json_schema", "schema",
                    ))
                ):
                    raise StructuredOutputSchemaError(
                        "OpenRouter endpoint rejected the output schema."
                    ) from e

                if e.response.status_code == 404:
                    print(f"[OpenRouter] ERROR: Model '{self.model}' not found!")
                    print(f"   Check available models at https://openrouter.ai/models")
                    print(f"   Response: {error_body}")
                elif e.response.status_code == 401:
                    print(f"[OpenRouter] ERROR: Invalid API key!")
                elif e.response.status_code == 402:
                    print(f"[OpenRouter] ERROR: Insufficient credits!")
                else:
                    print(f"OpenRouter API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    print(f"Response details: Status {e.response.status_code}, Body: {error_body}...")

                status_code = e.response.status_code if e.response is not None else None
                if is_context_overflow_error(error_message, status_code):
                    raise ContextOverflowError(f"OpenRouter context overflow: {error_message}")

                # Client errors (404 model, 401 key, 402 credits, 400) won't
                # recover on retry — fail fast instead of retrying 3x.
                if not is_retryable_http_status(e.response.status_code):
                    status = e.response.status_code
                    failure_class = (
                        "provider_auth" if status in {401, 403}
                        else "provider_quota" if status == 402
                        else "transport"
                    )
                    return terminal_provider_failure(
                        generation_options, failure_class, status
                    )

                attempt += 1
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    await asyncio.sleep(2)
                    continue
                return terminal_provider_failure(
                    generation_options, "transport", e.response.status_code
                )
            except json.JSONDecodeError as e:
                print(f"OpenRouter API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                attempt += 1
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    await asyncio.sleep(2)
                    continue
                return terminal_provider_failure(generation_options, "transport")
            except (RateLimitError, ContextOverflowError):
                # Must reach the pipeline: a rate-limit pause or a chunk split,
                # never a retried-then-failed chunk. (Ported from upstream 2b0a431f.)
                raise
            except Exception as e:
                print(f"OpenRouter API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                attempt += 1
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    await asyncio.sleep(2)
                    continue
                return terminal_provider_failure(generation_options, "transport")

        return terminal_provider_failure(generation_options, "transport")
