"""
Build the glossary block to inject into the user prompt.

The block style mirrors the existing prompt voice in src/prompts/prompts.py
(numbered priorities, MANDATORY phrasing) so the LLM treats glossary entries
with the same weight as the rest of the instructions.
"""
from typing import Dict, Optional


def build_glossary_block(
    filtered_terms: Dict[str, str],
    target_language: str = "",
    term_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """
    Render the glossary block. Empty string if no terms.

    Args:
        filtered_terms: {source: target} of terms that match the current chunk.
        target_language: present for symmetry with the rest of the prompt API.
        term_metadata: optional {source: {category}} mapping. When a term
            has a category, it is rendered as a bracketed hint after the
            arrow so the LLM can disambiguate homonyms (e.g. a name vs a
            place sharing the same spelling).

    The block lives in the user prompt near the source segment. This keeps the
    system prompt cacheable while still placing required translations close
    enough to the input text that the model will not forget them.
    """
    if not filtered_terms:
        return ""

    metadata = term_metadata or {}

    lines = [
        "# GLOSSARY - REQUIRED TRANSLATIONS",
        "",
        "MANDATORY: use these EXACT translations whenever the source term appears.",
        "Do NOT paraphrase, transliterate differently, or invent alternatives.",
        "Apply each rule consistently every time the term occurs.",
        "When several source forms are listed before the arrow (comma-separated), they are inflected variants of the same entity — translate any of them with the single target on the right.",
        "For compound source phrases that contain a required glossary term plus extra meaningful words, keep the required glossary translation exact and translate the remaining words naturally, unless a longer glossary entry gives a specific translation.",
        "Bracketed hints after the arrow (e.g. [character]) describe the entity type — use them only to disambiguate, not as part of the translation.",
        "",
    ]

    for source, target in filtered_terms.items():
        meta = metadata.get(source) or {}
        category = (meta.get("category") or "").strip()
        # Render alternatives (declined forms separated by '|' in storage) as
        # a comma-separated list so the LLM reads them as a natural set.
        display_source = ", ".join(
            a.strip() for a in source.split("|") if a.strip()
        ) or source
        if category:
            lines.append(f"  - {display_source} -> {target}  [{category}]")
        else:
            lines.append(f"  - {display_source} -> {target}")

    return "\n".join(lines) + "\n"


def build_refinement_glossary_block(
    filtered_terms: Dict[str, str],
    target_language: str = "",
    term_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """
    Render the glossary block for the refinement pass. Empty string if no terms.

    The refinement input is the target-language draft, so the terms were
    matched on their target forms, and the instruction is to keep those
    renderings rather than to translate a source term.
    (Ported from upstream df0de763.)

    Args:
        filtered_terms: {source: target} of the terms whose target form
            appears in the draft.
        target_language: present for symmetry with build_glossary_block.
        term_metadata: optional {source: {category}} hints.
    """
    if not filtered_terms:
        return ""

    metadata = term_metadata or {}

    lines = [
        "# GLOSSARY - KEEP THESE EXACT FORMS",
        "",
        "MANDATORY: the draft uses the required glossary renderings listed below "
        "(source term -> required rendering). Keep these exact forms while refining.",
        "Do NOT replace them with a synonym, a different transliteration, or a "
        "more familiar variant.",
        "Bracketed hints after a rendering (e.g. [character]) describe the entity "
        "type — never part of the text.",
        "",
    ]

    for source, target in filtered_terms.items():
        meta = metadata.get(source) or {}
        category = (meta.get("category") or "").strip()
        display_source = ", ".join(
            a.strip() for a in source.split("|") if a.strip()
        ) or source
        if category:
            lines.append(f"  - {display_source} -> {target}  [{category}]")
        else:
            lines.append(f"  - {display_source} -> {target}")

    return "\n".join(lines) + "\n"
