"""
Projection engine to render persistent database addressing state into prompt contexts.
"""

from typing import Any, Dict, List, Optional

from src.persistence.database import Database
from src.utils.text_matching import active_label_matches_name


def render_addressing_projection(
    translation_id: str,
    db: Optional[Database] = None,
    target_language: Optional[str] = None,
    active_character_names: Optional[Any] = None,
) -> str:
    """
    Render active directed addressing rules from DB into a clear, concise instruction block
    for LLM translation system prompt.
    """
    if db is None:
        db = Database()

    rules = db.get_addressing_rules(translation_id)
    if not rules:
        return ""

    # Filter rules to active characters in current scene when active_character_names is provided
    if active_character_names:
        active_set = set()
        if isinstance(active_character_names, dict):
            for k, info in active_character_names.items():
                active_set.add(str(k).casefold().strip())
                if isinstance(info, dict):
                    if info.get("name"):
                        active_set.add(str(info["name"]).casefold().strip())
                    if info.get("source_name"):
                        active_set.add(str(info["source_name"]).casefold().strip())
                    aliases = info.get("aliases") or info.get("alias") or []
                    if isinstance(aliases, str):
                        aliases = [a.strip() for a in aliases.split(",") if a.strip()]
                    for a in aliases:
                        if len(str(a).strip()) >= 2:
                            active_set.add(str(a).casefold().strip())
        elif isinstance(active_character_names, (list, set, tuple)):
            for item in active_character_names:
                if item:
                    active_set.add(str(item).casefold().strip())

        if active_set:
            filtered_rules = []
            for r in rules:
                spk = str(r.get("speaker_name") or "")
                adr = str(r.get("addressee_name") or "")
                if (
                    r.get("is_locked")
                    or any(
                        active_label_matches_name(active, spk, target_language)
                        or active_label_matches_name(active, adr, target_language)
                        for active in active_set
                    )
                ):
                    filtered_rules.append(r)
            rules = filtered_rules

    if not rules:
        return ""

    lines = [
        "### DIRECTED ADDRESSING RULES:",
        "Strictly adhere to character-to-character addressing forms below:",
    ]

    from src.utils.universal_addressing_engine import UniversalAddressingEngine
    engine = UniversalAddressingEngine(language=target_language or "generic")

    from src.utils.context_schema import is_situational_context

    for r in rules:
        speaker = r.get("speaker_name")
        addressee = r.get("addressee_name")
        self_p = r.get("self_pronoun")
        target_p = r.get("target_pronoun")
        vocative = r.get("vocative")
        register = r.get("register", "polite")

        f_self, f_target = engine.get_forbidden_pronouns(self_p or "", target_p or "")
        forbidden_list = sorted(list(f_self | f_target))

        forbidden_str = (
            f" [FORBIDDEN: {', '.join(repr(t) for t in forbidden_list)}]"
            if forbidden_list
            else ""
        )
        vocative_str = f" (vocative '{vocative}')" if vocative else ""
        situational_note = " [SITUATIONAL CONTEXT: Active in this scene only]" if is_situational_context(r) else ""
        if self_p and str(self_p).strip().casefold() != "unspecified":
            instruction = (
                f"Self-reference as '{self_p}', address target as '{target_p}'"
            )
        else:
            instruction = f"Address target as '{target_p}'"
        lines.append(
            f"- **{speaker}** addressing **{addressee}**: "
            f"{instruction}{vocative_str} "
            f"[Tone: {register}]{situational_note}{forbidden_str}."
        )

    return "\n".join(lines)
