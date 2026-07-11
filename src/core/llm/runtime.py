"""Provider-specific runtime configuration for draft and editor clients."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping, Optional

import src.config as config
from src.common.provider_metadata import KEY_REQUIRED_PROVIDERS, provider_env_var
from src.core.llm_client import LLMClient
from src.core.llm import LLMGenerationOptions
from src.core.llm.generation_controls import resolve_thinking_controls


@dataclass(frozen=True)
class ProviderRuntimeSpec:
    """Resolved, secret-safe description of one provider client."""

    provider: str
    model: str
    api_endpoint: str
    api_key: str
    env_var: str
    key_required: bool

    def same_model_as(self, other: "ProviderRuntimeSpec") -> bool:
        return self.provider == other.provider and self.model == other.model


def provider_endpoint(provider: str, requested: Optional[str] = None) -> str:
    """Resolve an endpoint without leaking a draft provider's endpoint."""

    normalized = str(provider or "ollama").strip().casefold()
    requested = str(requested or "").strip()
    if requested:
        return requested
    if normalized == "ollama":
        return str(config.OLLAMA_API_ENDPOINT)
    if normalized == "openai":
        return str(config.OPENAI_API_ENDPOINT)
    endpoint_name = f"{normalized.upper()}_API_ENDPOINT"
    return str(getattr(config, endpoint_name, "") or "")


def build_runtime_spec(
    provider: str,
    model: str,
    *,
    api_endpoint: Optional[str] = None,
    credentials: Optional[Mapping[str, Any]] = None,
) -> ProviderRuntimeSpec:
    """Build one provider spec from request values with `.env` fallback."""

    normalized = str(provider or "ollama").strip().casefold()
    env_var = provider_env_var(normalized)
    values = credentials or {}
    raw_key = values.get(f"{normalized}_api_key") or values.get("api_key")
    api_key = str(
        raw_key
        or (os.getenv(env_var, getattr(config, env_var, "")) if env_var else "")
        or ""
    )
    return ProviderRuntimeSpec(
        provider=normalized,
        model=str(model or "").strip(),
        api_endpoint=provider_endpoint(normalized, api_endpoint),
        api_key=api_key,
        env_var=env_var,
        key_required=normalized in KEY_REQUIRED_PROVIDERS,
    )


def create_runtime_client(
    spec: ProviderRuntimeSpec,
    *,
    context_window: Optional[int] = None,
    log_callback: Optional[Any] = None,
    default_generation_options: Optional[LLMGenerationOptions] = None,
) -> LLMClient:
    """Create an isolated client from a resolved provider specification."""

    kwargs = {
        "model": spec.model,
        "api_endpoint": spec.api_endpoint,
        "context_window": context_window,
        "log_callback": log_callback,
        "default_generation_options": default_generation_options,
    }
    if spec.api_key:
        kwargs["api_key"] = spec.api_key
    return LLMClient(provider_type=spec.provider, **kwargs)


def build_draft_and_editor_clients(
    *,
    draft_provider: str,
    draft_model: str,
    draft_endpoint: Optional[str],
    prompt_options: Optional[Mapping[str, Any]],
    credentials: Optional[Mapping[str, Any]],
    context_window: Optional[int] = None,
    log_callback: Optional[Any] = None,
) -> tuple[LLMClient, LLMClient, ProviderRuntimeSpec, ProviderRuntimeSpec]:
    """Create draft/editor clients, reusing only an identical provider/model."""

    options = prompt_options or {}
    draft_spec = build_runtime_spec(
        draft_provider,
        draft_model,
        api_endpoint=draft_endpoint,
        credentials=credentials,
    )
    editor_provider = str(options.get("editor_provider") or draft_provider)
    editor_model = str(options.get("editor_model") or draft_model)
    editor_endpoint = options.get("editor_api_endpoint")
    if not editor_endpoint and editor_provider.strip().casefold() == draft_spec.provider:
        editor_endpoint = draft_spec.api_endpoint
    editor_spec = build_runtime_spec(
        editor_provider,
        editor_model,
        api_endpoint=editor_endpoint,
        credentials=credentials,
    )
    draft_client = create_runtime_client(
        draft_spec,
        context_window=context_window,
        log_callback=log_callback,
        default_generation_options=LLMGenerationOptions(**{
            key: value for key, value in resolve_thinking_controls(
                draft_spec.provider,
                draft_spec.model,
                options.get("draft_thinking_level"),
                role="draft",
                endpoint=draft_spec.api_endpoint,
                reasoning_supported=bool(options.get("draft_reasoning_supported")),
            ).items() if key in {
                "thinking_level", "thinking_budget", "thinking_enabled",
                "reasoning_effort",
            }
        }),
    )
    editor_client = (
        draft_client
        if editor_spec.same_model_as(draft_spec)
        else create_runtime_client(
            editor_spec,
            context_window=context_window,
            log_callback=log_callback,
            default_generation_options=LLMGenerationOptions(**{
                key: value for key, value in resolve_thinking_controls(
                    editor_spec.provider,
                    editor_spec.model,
                    options.get("editor_thinking_level"),
                    role="editor",
                    endpoint=editor_spec.api_endpoint,
                    reasoning_supported=bool(options.get("editor_reasoning_supported")),
                ).items() if key in {
                    "thinking_level", "thinking_budget", "thinking_enabled",
                    "reasoning_effort",
                }
            }),
        )
    )
    return draft_client, editor_client, draft_spec, editor_spec
