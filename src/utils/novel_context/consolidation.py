"""Periodic consolidation of accumulated lore into a compact form."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .characters import (
    _alias_entries_to_map,
    _canonical_display_name,
    _character_alias_keys,
    _character_names_match,
    _find_lore_section,
    _is_invalid_context_key,
    _is_unstable_identity_alias,
    _parse_alias_entries,
    _parse_bullet_entries,
    _plain_key,
    _replace_lore_section,
    _strip_balanced_brackets,
)
from .constants import _GENDER_LABELS, ALIASES_SECTION, CHARACTERS_SECTION, logger
from .glossary import normalize_global_lore
from .lore_merge import merge_new_lore

CONSOLIDATION_SYSTEM_PROMPT = """You are a precise novel context editor.
Your job is to clean up a Characters & Genders list by merging duplicate or redundant entries, removing entries that are not characters, and preserving source-proven identity links for merged labels.

Rules:
- Each canonical character must appear exactly once with one concise, non-repetitive description.
- Merge all duplicate facts (same idea, different wording) into a single clear phrase.
- Keep the most informative phrasing; drop redundant restatements.
- Keep ALL factual information that is not a pure duplicate: gender, rank, role, key relationships, notable traits.
- Preserve the exact canonical name already shown in the input (do not rename characters).
- Preserve gender labels exactly (Male / Female / Unspecified).
- Identify and remove generic background NPCs (e.g., unnamed students, teachers, guards, bystanders) that do not have named proper counterparts or significant recurring, plot-driving descriptions.
- Identify and remove first-pass non-character mistakes: companies, organizations, factions, families/houses/lineages, countries, facilities, magic circles, artifacts, weapons, skills, systems, metrics, titles, and abstract concepts do not belong in Characters & Genders.
- If a non-character proper noun is useful terminology (e.g., a company name, family/house name, named artifact, named magic circle, or product name), omit it from this output; it belongs in Glossary & Terminology, not Characters & Genders.
- Remove bare relationship labels such as "Brother", "Father", "Lover", "Girlfriend", "Wife", or "Husband" unless the entry clearly names a distinct source-named character; relationships belong in descriptions or dynamic state, not as fake character identities.
- Remove duplicate romanization/source-name variants for Chinese, Japanese, and Korean names when the entries clearly describe the same person. Keep the existing canonical character name from the input and fold the facts into that one entry; do not invent a new pinyin, romaji, or revised-romanization canonical name.
- Use IDENTITY_LINKS when a removed character entry is actually a stable label, physical form, transformed state, disguise, codename, masked identity, experimental designation, or other source-side alias for a retained canonical character. This includes nonhuman entities described under both a body/type label and a later canonical name.
- Only add an identity link when the Characters list or Dynamic Relationship State clearly proves both labels are the same entity. Never merge from role similarity alone, and never merge merely because two entities share the same ally, enemy, species, origin, or broad function.
- **SIMILAR NAMES WARNING (CRITICAL):** Never merge or link similar-looking character names (e.g. Alex vs. Alles, or Marc vs. Mark) unless there is absolute, direct proof in the text that they are the same person. Keep them strictly separate. Even if a correction or nickname mapping is defined for one character (e.g., 'Canonical name is Alles, source text uses Alex'), DO NOT apply that translation/remapping to a different character who is actually named Alex. They are separate individuals.
- Output ONLY these blocks:
[CHARACTERS]
- Canonical Name: Gender, concise description.
[IDENTITY_LINKS]
- Removed Alias: Canonical Name
- If there are no identity links, leave the IDENTITY_LINKS block empty.
- Character lines must use this format:
  - Canonical Name: Gender, concise description.
- Do NOT add any heading, explanation, markdown fence, or extra text.
- Do NOT invent new facts that are not present in the input.
"""
CONSOLIDATION_USER_PROMPT_TEMPLATE = """Below is the current Characters & Genders list.
Merge duplicate / redundant descriptions so each character has exactly one concise entry.

You may use the Dynamic Identity Evidence only as supporting evidence for duplicate identity links; do not rewrite relationships.

### CHARACTERS & GENDERS
{characters_section}

### DYNAMIC IDENTITY EVIDENCE
{identity_evidence}

Output the cleaned context now:"""
_CONSOLIDATION_IDENTITY_EVIDENCE_TERMS = {
    "alias",
    "avatar",
    "calamity",
    "clone",
    "codename",
    "creature",
    "devil",
    "disguise",
    "disguised",
    "doppelganger",
    "doppelgänger",
    "entity",
    "experimental",
    "form",
    "gelatinous",
    "golem",
    "masked",
    "monster",
    "named form",
    "nickname",
    "prototype",
    "replicate",
    "replica",
    "shapeshift",
    "slime",
    "spirit",
    "subject",
    "transformed",
    "true identity",
    "undercover",
}
def _line_mentions_character(line_key: str, name_key: str) -> bool:
    if not line_key or not name_key:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(name_key)}(?!\w)", line_key))
def _consolidation_identity_evidence(
    characters_body: str,
    dynamic_state: str,
    max_lines: int = 80,
) -> str:
    """Return only dynamic lines likely to help identity consolidation."""
    dynamic_lines = [
        line.strip()
        for line in str(dynamic_state or "").splitlines()
        if line.strip().startswith(("-", "*"))
    ]
    if not dynamic_lines:
        return "(none)"

    evidence_character_keys: set[str] = set()
    for name, details in _parse_bullet_entries(characters_body):
        folded = _plain_key(f"{name} {details}")
        if any(term in folded for term in _CONSOLIDATION_IDENTITY_EVIDENCE_TERMS):
            evidence_character_keys.update(_character_alias_keys(name))

    if not evidence_character_keys:
        return "(none)"

    evidence_lines = []
    seen: set[str] = set()
    for line in dynamic_lines:
        line_key = _plain_key(line)
        if not any(
            _line_mentions_character(line_key, name_key)
            for name_key in evidence_character_keys
        ):
            continue
        if line_key in seen:
            continue
        evidence_lines.append(line)
        seen.add(line_key)
        if len(evidence_lines) >= max_lines:
            break

    return "\n".join(evidence_lines) if evidence_lines else "(none)"
async def consolidate_context_lore(
    llm_client: Any,
    model_name: str,
    global_lore: str,
    dynamic_state: str = "",
) -> Tuple[str, List[str]]:
    """Run an LLM pass that deduplicates/consolidates the Characters section.

    The deterministic merge catches exact or near-exact fact overlap but misses
    semantically identical descriptions written with different words.  This
    function asks the LLM to rewrite the Characters section into one clean,
    non-redundant entry per character.

    Returns:
        Tuple of (updated_global_lore, change_logs)
    """
    change_logs: List[str] = []

    # Extract the existing Characters section
    char_bounds = _find_lore_section(global_lore, CHARACTERS_SECTION)
    if not char_bounds:
        return global_lore, change_logs

    _, body_start, body_end = char_bounds
    characters_body = global_lore[body_start:body_end].strip()
    if not characters_body:
        return global_lore, change_logs

    user_prompt = CONSOLIDATION_USER_PROMPT_TEMPLATE.format(
        characters_section=characters_body,
        identity_evidence=_consolidation_identity_evidence(
            characters_body,
            dynamic_state,
        ),
    )

    try:
        response = await llm_client.generate(
            prompt=user_prompt,
            system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
        )
        if not response or not response.content:
            logger.warning(
                "Empty response from LLM during context consolidation. Skipping."
            )
            return global_lore, change_logs

        raw = response.content.strip()
        # Clean any markdown code blocks
        if "```" in raw:
            # Try to extract content inside the first code block
            block_match = re.search(r'```(?:[a-zA-Z-]*)\n(.*?)```', raw, re.DOTALL)
            if block_match:
                raw = block_match.group(1).strip()
            else:
                # Fallback: just strip the fence lines
                raw = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("```")).strip()

        original_character_entries = _parse_bullet_entries(characters_body)
        original_character_keys = {
            _plain_key(name)
            for name, _ in original_character_entries
            if not _is_invalid_context_key(name)
        }
        existing_alias_bounds = _find_lore_section(global_lore, ALIASES_SECTION)
        existing_alias_map = _alias_entries_to_map(
            _parse_alias_entries(
                global_lore[existing_alias_bounds[1]:existing_alias_bounds[2]]
            ) if existing_alias_bounds else []
        )

        # Validate: must have at least one character line (with - or * or number or no prefix but contains ':')
        consolidated_entries = []
        identity_link_entries: List[Tuple[str, str]] = []
        current_output_section = "characters"
        for line in raw.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            section_match = re.match(r"^\[([A-Za-z_ ]+)\]\s*$", line_str)
            if section_match:
                section_name = section_match.group(1).strip().casefold().replace(" ", "_")
                if section_name in {"characters", "characters_&_genders"}:
                    current_output_section = "characters"
                elif section_name == "identity_links":
                    current_output_section = "identity_links"
                else:
                    current_output_section = "other"
                continue
            
            # Try to match a bullet/list marker: '-', '*', '1.', '2.', etc.
            match = re.match(r'^(?:[-*]|\d+\.)\s*(.*)$', line_str)
            if match:
                content = match.group(1).strip()
            else:
                content = line_str

            if current_output_section == "identity_links":
                if ":" not in content or content.startswith("#"):
                    continue
                raw_alias, raw_target = content.split(":", 1)
                alias = _strip_balanced_brackets(raw_alias)
                target = _canonical_display_name(raw_target)
                if alias and target:
                    identity_link_entries.append((alias, target))
                continue

            if current_output_section == "characters" and ":" in content and not content.startswith("#"):
                parts = content.split(":", 1)
                val = parts[1].strip()
                val_first_word = re.split(r'[\s,.]', val)[0].casefold()
                if val_first_word in _GENDER_LABELS:
                    consolidated_entries.append(f"- {content}")

        if not consolidated_entries:
            logger.warning(
                "Consolidation LLM returned no valid character entries. Skipping."
            )
            return global_lore, change_logs

        consolidated_body = "\n".join(consolidated_entries)
        updated_lore = _replace_lore_section(
            global_lore,
            CHARACTERS_SECTION,
            consolidated_entries,
        )

        consolidated_character_entries = _parse_bullet_entries(consolidated_body)
        consolidated_character_keys = {
            _plain_key(name)
            for name, _ in consolidated_character_entries
            if not _is_invalid_context_key(name)
        }
        retained_target_by_key = {
            _plain_key(name): name
            for name, _ in consolidated_character_entries
            if not _is_invalid_context_key(name)
        }
        accepted_identity_lines = []
        for alias, target in identity_link_entries:
            alias_key = _plain_key(alias)
            target_key = _plain_key(target)
            canonical_target = retained_target_by_key.get(target_key)
            if (
                not alias_key
                or not canonical_target
                or alias_key not in original_character_keys
                or alias_key in consolidated_character_keys
                or _character_names_match(alias, canonical_target)
                or _is_unstable_identity_alias(alias, allow_physical=True)
                or existing_alias_map.get(alias_key) != canonical_target
            ):
                continue
            accepted_identity_lines.append(f"- {alias}: {canonical_target}")
        if accepted_identity_lines:
            updated_lore, alias_logs = merge_new_lore(
                updated_lore,
                "",
                "",
                "\n".join(accepted_identity_lines),
            )
            change_logs.extend(alias_logs)

        updated_lore = normalize_global_lore(updated_lore)

        if updated_lore != global_lore:
            msg = "[Novel Context] Consolidation pass: character list deduped/merged and non-character entries pruned."
            change_logs.append(msg)

        return updated_lore, change_logs

    except Exception as exc:
        logger.error(f"Error during context consolidation LLM call: {exc}")
        return global_lore, change_logs
def _consolidation_interval() -> int:
    """Return the configured consolidation interval (0 = disabled)."""
    try:
        from src import config as _config
        return max(0, int(getattr(_config, "NOVEL_CONTEXT_CONSOLIDATION_INTERVAL", 5)))
    except Exception:
        return 5
