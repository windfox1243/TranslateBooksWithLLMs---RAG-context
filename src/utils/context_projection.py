"""
Projection engine to render persistent database addressing state into prompt contexts.
"""

from typing import Optional, List, Dict, Any
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


def render_addressing_markdown(translation_id: str, db: Optional[Database] = None) -> str:
    """
    Render full addressing state into a formatted Markdown view artifact.
    """
    if db is None:
        db = Database()

    rules = db.get_addressing_rules(translation_id)
    if not rules:
        return "# Dynamic Character Addressing Rules\n\nNo active addressing rules recorded.\n"

    lines = [
        "# Dynamic Character Addressing Rules",
        "",
        "| Speaker | Addressee | Self Pronoun | Target Pronoun | Vocative | Register | Locked | Last Chunk |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |",
    ]

    for r in rules:
        locked_badge = "🔒 Yes" if r.get("is_locked") else "No"
        lines.append(
            f"| {r.get('speaker_name')} | {r.get('addressee_name')} | {r.get('self_pronoun')} | "
            f"{r.get('target_pronoun')} | {r.get('vocative') or '-'} | {r.get('register') or 'polite'} | "
            f"{locked_badge} | {r.get('last_chunk_index', 0)} |"
        )

    lines.append("")
    return "\n".join(lines)


def convert_addressing_text_to_markdown_table(text: str) -> str:
    """
    Convert raw/legacy pipe-delimited addressing text entries into a clean Markdown table.
    """
    if not text or not text.strip():
        return ""

    table_lines = [
        "| Speaker | Addressee | Tự xưng (Self) | Gọi đối phương (Target) | Danh xưng (Vocative) | Sắc thái / Ghi chú |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    has_entries = False
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("-"):
            continue
        # Extract "- Speaker → Addressee: details"
        parts = line.lstrip("- ").split(":", 1)
        if len(parts) < 2:
            continue
        pair_part = parts[0].strip()
        details_part = parts[1].strip()

        if "→" in pair_part:
            pair = [p.strip() for p in pair_part.split("→", 1)]
        elif "->" in pair_part:
            pair = [p.strip() for p in pair_part.split("->", 1)]
        else:
            continue

        speaker, addressee = pair[0], pair[1]

        # Extract fields from details_part
        self_p = "-"
        target_p = "-"
        vocative = "-"
        notes = "-"

        # Parse pipe separated parts if present
        pipe_chunks = [c.strip() for c in details_part.split("|")]
        if len(pipe_chunks) >= 1 and pipe_chunks[0].startswith('"') and pipe_chunks[0].endswith('"'):
            vocative = pipe_chunks[0].strip('" ')

        for chunk in pipe_chunks:
            chunk_clean = chunk.strip('" ')
            if "self-reference:" in chunk:
                # Extract self reference
                for sub in chunk.split(";"):
                    sub_clean = sub.strip('" ')
                    if "self-reference:" in sub_clean:
                        self_p = sub_clean.split(":", 1)[1].strip('" ')
                    elif "second-person pronoun:" in sub_clean:
                        target_p = sub_clean.split(":", 1)[1].strip('" ')
                    elif "vocative/address form:" in sub_clean:
                        vocative = sub_clean.split(":", 1)[1].strip('" ')
            elif chunk_clean != vocative and not chunk_clean.startswith("self-reference"):
                notes = chunk_clean

        table_lines.append(f"| {speaker} | {addressee} | {self_p} | {target_p} | {vocative} | {notes} |")
        has_entries = True

    if not has_entries:
        return text

    return "\n".join(table_lines)
