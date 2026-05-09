"""
Build the glossary block to inject into the system prompt.

The block style mirrors the existing prompt voice in prompts/prompts.py
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

    The block lives between the optional sections and the placeholder section
    in the system prompt — close enough to the input text that the model will
    not forget it, but not after the FINAL REMINDER so the output-language
    reminder stays last.
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
