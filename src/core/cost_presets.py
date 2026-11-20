"""Cost presets: one setting that trades LLM calls per chunk for quality.

The draft costs one call per chunk. What multiplies that is everything the
fork adds around it: source-context analysis before a chunk, the periodic
lore consolidation pass, the LLM judge for ambiguous relationships, and the
Senior Editor's review and repairs. Each has its own setting; a preset sets
them together.

- ``economy`` analyzes context every third chunk instead of every chunk,
  turns off consolidation and the relationship judge, and leaves the Senior
  Editor off.
- ``balanced`` changes nothing: every setting keeps its own value.
- ``quality`` analyzes context before every chunk and turns the Senior Editor
  on.

Precedence, highest first: an option set on the job itself (a web UI control,
an API field), then the preset, then the individual .env setting, then the
built-in default. The preset sits above .env because a generated .env always
writes the individual values out, which would otherwise make the preset
unreachable.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

DEFAULT_COST_PRESET = "balanced"

COST_PRESETS: Dict[str, Dict[str, Any]] = {
    "economy": {
        "novel_context_update_interval": 3,
        "novel_context_consolidation_interval": 0,
        "use_relationship_llm_judge": "off",
        "reflection_mode": False,
    },
    "balanced": {},
    "quality": {
        "novel_context_update_interval": 1,
        "reflection_mode": True,
    },
}


def normalize_cost_preset(value: Any) -> Optional[str]:
    """Return the preset name for `value`, or None when it names none."""
    name = str(value or "").strip().casefold()
    return name if name in COST_PRESETS else None


def _configured_preset() -> str:
    try:
        from src import config as _config

        return normalize_cost_preset(getattr(_config, "COST_PRESET", "")) or DEFAULT_COST_PRESET
    except Exception:
        return DEFAULT_COST_PRESET


def apply_cost_preset(prompt_options: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a copy of `prompt_options` with the job's preset filled in.

    The preset comes from ``prompt_options["cost_preset"]`` or, failing that,
    the COST_PRESET setting; an unknown name falls back to balanced. Keys the
    job already sets are left alone. The resolved name is stored back under
    ``cost_preset`` so a resumed job keeps the same preset.
    """
    options = dict(prompt_options or {})
    name = normalize_cost_preset(options.get("cost_preset")) or _configured_preset()
    for key, value in COST_PRESETS[name].items():
        options.setdefault(key, value)
    options["cost_preset"] = name
    return options
