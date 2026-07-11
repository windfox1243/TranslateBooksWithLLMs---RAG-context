"""Model-aware generation controls shared by the API, UI, and providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, Optional


GENERATION_MODES = (
    "auto", "off", "on", "minimal", "low", "medium", "high", "dynamic",
)
MIN_EDITOR_OUTPUT_TOKENS = 1024
DEFAULT_EDITOR_OUTPUT_LIMIT = 65536


@dataclass(frozen=True)
class GenerationCapabilities:
    """Safe model metadata used to render generation controls."""

    provider: str
    model: str
    thinking_supported: bool = False
    thinking_control: str = "none"
    thinking_modes: tuple[str, ...] = ("auto",)
    default_thinking_mode: str = "auto"
    can_disable_thinking: bool = False
    output_token_limit: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["thinking_modes"] = list(self.thinking_modes)
        return payload


def _is_gemini_3(model: str) -> bool:
    return bool(re.search(r"(?:^|/)gemini-3(?:[.\-]|$)", model.casefold()))


def _is_gemini_25(model: str) -> bool:
    return "gemini-2.5" in model.casefold()


def generation_capabilities(
    provider: str,
    model: str,
    *,
    endpoint: str = "",
) -> GenerationCapabilities:
    """Classify supported controls without making a billable model request."""

    normalized_provider = str(provider or "").strip().casefold()
    normalized_model = str(model or "").strip()
    lowered = normalized_model.casefold()
    base = {
        "provider": normalized_provider,
        "model": normalized_model,
    }
    if normalized_provider == "gemini" and _is_gemini_3(lowered):
        if "pro" in lowered:
            modes = (
                ("auto", "low", "medium", "high")
                if "3.1" in lowered
                else ("auto", "low", "high")
            )
            default_mode = "high"
        else:
            modes = ("auto", "minimal", "low", "medium", "high")
            default_mode = (
                "minimal" if "flash-lite" in lowered
                else "medium" if "3.5-flash" in lowered
                else "high"
            )
        return GenerationCapabilities(
            **base,
            thinking_supported=True,
            thinking_control="level",
            thinking_modes=modes,
            default_thinking_mode=default_mode,
            can_disable_thinking=False,
            output_token_limit=DEFAULT_EDITOR_OUTPUT_LIMIT,
        )

    if normalized_provider == "gemini" and _is_gemini_25(lowered):
        can_disable = "pro" not in lowered
        modes = ["auto"]
        if can_disable:
            modes.append("off")
        modes.extend(("minimal", "low", "medium", "high", "dynamic"))
        return GenerationCapabilities(
            **base,
            thinking_supported=True,
            thinking_control="budget",
            thinking_modes=tuple(modes),
            default_thinking_mode="off" if "flash-lite" in lowered else "dynamic",
            can_disable_thinking=can_disable,
            output_token_limit=DEFAULT_EDITOR_OUTPUT_LIMIT,
        )

    if normalized_provider == "ollama":
        if "gpt-oss" in lowered:
            return GenerationCapabilities(
                **base,
                thinking_supported=True,
                thinking_control="level",
                thinking_modes=("auto", "low", "medium", "high"),
                default_thinking_mode="medium",
                can_disable_thinking=False,
            )
        thinking_markers = (
            "qwen3", "deepseek-r1", "deepseek-v3.1", "qwq",
            "phi4-reasoning", "marco-o1", "exaone-deep",
        )
        if any(marker in lowered for marker in thinking_markers):
            uncontrollable = any(
                marker in lowered
                for marker in (
                    "qwen3:30b", "deepseek-r1", "qwq",
                    "phi4-reasoning", "marco-o1", "exaone-deep",
                )
            )
            modes = ("auto", "on") if uncontrollable else ("auto", "off", "on")
            return GenerationCapabilities(
                **base,
                thinking_supported=True,
                thinking_control="boolean",
                thinking_modes=modes,
                default_thinking_mode="on",
                can_disable_thinking=not uncontrollable,
            )

    official_openai = (
        not endpoint or "api.openai.com" in str(endpoint).casefold()
    )
    if normalized_provider == "openai" and official_openai and re.search(
        r"(?:^|/)(?:gpt-5|o1|o3|o4)(?:[.\-]|$)", lowered
    ):
        if "pro" in lowered:
            modes = ("auto", "high")
        elif "gpt-5.1" in lowered:
            modes = ("auto", "off", "low", "medium", "high")
        else:
            modes = ("auto", "minimal", "low", "medium", "high")
        return GenerationCapabilities(
            **base,
            thinking_supported=True,
            thinking_control="effort",
            thinking_modes=modes,
            default_thinking_mode="high" if "pro" in lowered else "medium",
            can_disable_thinking="off" in modes,
        )

    if normalized_provider == "openrouter":
        reasoning_markers = (
            "openai/gpt-5", "openai/o1", "openai/o3", "openai/o4",
            "google/gemini-2.5", "google/gemini-3", "anthropic/claude",
            "deepseek/deepseek-r1", "qwen/qwq", "qwen/qwen3",
        )
        if any(marker in lowered for marker in reasoning_markers):
            return GenerationCapabilities(
                **base,
                thinking_supported=True,
                thinking_control="effort",
                thinking_modes=(
                    "auto", "off", "minimal", "low", "medium", "high",
                ),
                default_thinking_mode="medium",
                can_disable_thinking=True,
            )

    return GenerationCapabilities(**base)


def normalize_thinking_mode(value: Any) -> str:
    mode = str(value or "auto").strip().casefold()
    return mode if mode in GENERATION_MODES else "auto"


def resolve_thinking_controls(
    provider: str,
    model: str,
    requested_mode: Any,
    *,
    role: str,
    endpoint: str = "",
    reasoning_supported: bool = False,
) -> Dict[str, Any]:
    """Resolve a UI mode into provider-native generation option fields."""

    normalized_provider = str(provider or "").strip().casefold()
    capabilities = generation_capabilities(provider, model, endpoint=endpoint)
    if normalized_provider == "openrouter" and reasoning_supported:
        capabilities = GenerationCapabilities(
            provider="openrouter",
            model=model,
            thinking_supported=True,
            thinking_control="effort",
            thinking_modes=(
                "auto", "off", "minimal", "low", "medium", "high",
            ),
            default_thinking_mode="medium",
            can_disable_thinking=True,
        )
    if not capabilities.thinking_supported:
        return {"mode": "auto", "thinking_level": None, "thinking_budget": None}

    mode = normalize_thinking_mode(requested_mode)
    if mode not in capabilities.thinking_modes:
        mode = "auto"

    if mode == "auto":
        if role == "editor":
            if capabilities.thinking_control in {"level", "effort"}:
                mode = "minimal" if "minimal" in capabilities.thinking_modes else "low"
            elif capabilities.thinking_control == "boolean":
                mode = "off" if capabilities.can_disable_thinking else "on"
            else:
                mode = "off" if capabilities.can_disable_thinking else "minimal"
        elif capabilities.thinking_control == "budget":
            # Preserve the pre-beta.34 Gemini 2.5 behavior where possible.
            mode = "off" if capabilities.can_disable_thinking else "dynamic"
        else:
            return {"mode": "auto", "thinking_level": None, "thinking_budget": None}

    if capabilities.thinking_control == "effort":
        return {
            "mode": mode,
            "thinking_level": None,
            "thinking_budget": None,
            "thinking_enabled": None,
            "reasoning_effort": "none" if mode == "off" else mode,
        }

    if capabilities.thinking_control == "boolean":
        return {
            "mode": mode,
            "thinking_level": None,
            "thinking_budget": None,
            "thinking_enabled": mode != "off",
            "reasoning_effort": None,
        }

    if capabilities.thinking_control == "level":
        return {"mode": mode, "thinking_level": mode, "thinking_budget": None}

    budget_by_mode = {
        "off": 0,
        "minimal": 128 if "pro" in model.casefold() else 512,
        "low": 1024,
        "medium": 4096,
        "high": 8192,
        "dynamic": -1,
    }
    return {
        "mode": mode,
        "thinking_level": None,
        "thinking_budget": budget_by_mode.get(mode, -1),
        "thinking_enabled": None,
        "reasoning_effort": None,
    }


def resolve_editor_output_tokens(
    provider: str,
    model: str,
    requested: Any,
    thinking_mode: Any,
    *,
    reported_limit: Any = None,
) -> int:
    """Resolve Auto/model-maximum/custom output settings to a bounded integer."""

    normalized_provider = str(provider or "").strip().casefold()
    capabilities = generation_capabilities(provider, model)
    try:
        supplied_limit = int(reported_limit or 0)
    except (TypeError, ValueError):
        supplied_limit = 0
    maximum = supplied_limit or capabilities.output_token_limit or 16384
    maximum = max(MIN_EDITOR_OUTPUT_TOKENS, min(maximum, DEFAULT_EDITOR_OUTPUT_LIMIT))

    value = str(requested or "auto").strip().casefold()
    if value == "model_max":
        return maximum
    if value not in {"", "auto"}:
        try:
            return max(MIN_EDITOR_OUTPUT_TOKENS, min(int(value), maximum))
        except (TypeError, ValueError):
            pass

    mode = normalize_thinking_mode(thinking_mode)
    automatic = {
        "off": 4096,
        "minimal": 4096,
        "low": 8192,
        "medium": 8192,
        "high": 16384,
        "dynamic": 16384,
        "auto": 4096 if normalized_provider == "gemini" else 2048,
    }[mode]
    return min(automatic, maximum)


def adaptive_retry_output_tokens(current: int, maximum: Any = None) -> int:
    """Double a truncated editor allowance without exceeding provider limits."""

    try:
        limit = int(maximum or DEFAULT_EDITOR_OUTPUT_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_EDITOR_OUTPUT_LIMIT
    limit = max(MIN_EDITOR_OUTPUT_TOKENS, min(limit, DEFAULT_EDITOR_OUTPUT_LIMIT))
    return min(max(8192, int(current or 0) * 2), limit)
