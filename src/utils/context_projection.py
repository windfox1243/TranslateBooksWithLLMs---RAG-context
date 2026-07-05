"""
Projection engine to render persistent database addressing state into prompt contexts.
"""

from typing import Optional, List, Dict, Any
from src.persistence.database import Database


def render_addressing_projection(translation_id: str, db: Optional[Database] = None) -> str:
    """
    Render active directed addressing rules from DB into a clear, concise instruction block
    for LLM translation system prompt.
    """
    if db is None:
        db = Database()

    rules = db.get_addressing_rules(translation_id)
    if not rules:
        return ""

    lines = [
        "### QUY TẮC XƯNG HÔ CÓ HƯỚNG (DIRECTED ADDRESSING RULES):",
        "Chú ý tuân thủ tuyệt đối cách xưng hô và xưng xưng/gọi ngôi của từng nhân vật dưới đây:",
    ]

    for r in rules:
        speaker = r.get("speaker_name")
        addressee = r.get("addressee_name")
        self_p = r.get("self_pronoun")
        target_p = r.get("target_pronoun")
        vocative = r.get("vocative")
        register = r.get("register", "polite")

        vocative_str = f" (gọi là '{vocative}')" if vocative else ""
        lines.append(
            f"- **{speaker}** khi nói với **{addressee}**: "
            f"Tự xưng là '{self_p}', gọi đối phương là '{target_p}'{vocative_str} [Sắc thái: {register}]."
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
