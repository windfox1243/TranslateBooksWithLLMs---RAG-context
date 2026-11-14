"""Prompt text used when the target language is Vietnamese.

Kept out of src/prompts/prompts.py so the prompt builders hold no
Vietnamese-specific wording; they look this module up by profile code.
"""

# Appended to the translation system prompt (see
# prompts._build_target_language_style_section).
STYLE_GUARDRAILS = """
# VIETNAMESE STYLE GUARDRAILS

- Maintain consistent Vietnamese pronouns and register across adjacent paragraphs.
- For serious literary first-person narration, prefer "tôi" unless the source or established character voice clearly requires an intimate/casual "mình".
- Do not switch the same narrator from "tôi" to "mình" within the same scene or reflective passage unless the relationship/register intentionally changes.
- Treat `## CURRENT ADDRESSING FORMS` as authoritative for direct address. If it gives a target-language form that is a name or title, use that form; do not replace it with Vietnamese kinship pronouns such as "anh", "chị", or "em", and do not call or refer to that character with an unrelated kinship pronoun.
- When a matching addressing entry gives or implies a paired Vietnamese register, apply both sides of the pair: the speaker's self-reference and the addressee/reference form. Do not combine an intimate/casual addressee form or nickname with a formal/neutral self-reference unless the context explicitly says the speaker is keeping distance.
- If a Vietnamese addressing entry spells out `self-reference`, `second-person pronoun`, `third-person pronoun`, or `vocative/address form`, obey each part exactly: use the self-reference for the speaker, the second-person pronoun for "you", the third-person pronoun when referring to that character to others (e.g. "cậu" -> "cậu ấy"/"cậu ta"), and the vocative/address form only when directly calling or naming the addressee.
- Do not infer "anh", "chị", or "em" from gender, status, affection, or politeness alone. Use them only when the source/context establishes the age, kinship, or seniority relationship, or when the addressing entry explicitly requires that form.
- Vietnamese address is a paired social system, not a single pronoun. Choose both the speaker's self-reference and the addressee/reference form from the relationship and setting. Common pairs include, but are not limited to, "em-cô", "em-thầy", "con-mẹ/cha", "bố/mẹ-con", "tớ-cậu", "tôi-cậu", "mình-cậu", "tao-mày", name-only, title-only, or neutral forms when appropriate.
- For casual peer, classmate, roommate, and friendly rival dialogue, prefer "tớ-cậu" as the default youthful peer pair unless the established character voice, narration, or stored context specifically supports the softer/more self-reflective "mình-cậu".
- Apply age, family, school-year, seniority, and relationship facts to both direct address and indirect references in dialogue, thoughts, and narration. When a speaker or narrator refers to a character in third person (e.g. 'he', 'him', 'his'), the indirect pronoun MUST strictly reflect the speaker/narrator's seniority to that character. If the speaker is senior or older (or addresses that character as 'cậu', 'em', or 'cháu' in direct address, such as an older mentor speaking about a younger student), NEVER use 'anh' or 'anh ấy' when talking about them to third parties; use 'cậu ấy', 'cậu ta', or 'cậu [Name]' instead.
- For English named skills, abilities, techniques, spells, combat moves, weapons, artifacts, and equipment, prefer concise Sino-Vietnamese literary renderings when that sounds natural in Vietnamese fantasy or game prose.
- Do not leave an English skill, technique, or named item untranslated merely because it is capitalized. Preserve the English form only when it is a brand, code label, UI/system key, or a required glossary entry says to keep it.
- Keep character names exact; the Sino-Vietnamese rendering preference applies to named powers, weapons, items, and terminology, not people.
- For Korean source texts, do not leak raw Korean honorific suffixes (such as "-ssi" / -씨, "-nim" / -님) directly into the Vietnamese translated text (e.g. do not write "Aten-ssi" or "Renzo-ssi"). Convert them into natural Vietnamese address forms, pronouns, or standard character names according to context and established addressing forms.
- For Japanese source texts, familiar Japanese honorific suffixes (such as "-sama", "-senpai", "-san", "-kun", "-chan", "-dono") may be preserved in Vietnamese light novel/manga prose when consistent with established character voice, context, or stored addressing forms.
- Do not default to generic fantasy or archaic address pairs (such as "ta-nàng" or "ta-ngươi") merely because a character holds a title like princess, prince, or noble. Addressing MUST strictly conform to the established context of their relationship, setting, intimacy, and character dynamics (e.g. classmates, peers, colleagues, or formal court etiquette when explicitly required).
- Preserve established addressing forms from the glossary, dialogue context, and previous paragraph."""

# Extra rule for glossary NER suggestions (see prompts.generate_ner_prompt).
NER_SKILL_GUIDANCE = """
8. For English named skills, abilities, techniques, spells, combat moves, weapons, artifacts, and equipment, prefer concise Sino-Vietnamese literary target renderings when natural for Vietnamese fantasy or game prose. Preserve English only for brands, code labels, UI/system keys, or terms that clearly should remain untranslated."""

# Examples substituted into the Senior Editor reflection prompt (see
# prompts._get_language_specific_prompt_guidance).
REFLECTION_GUIDANCE = {
    "loc_example": ' (e.g. in Vietnamese: translate English "Momozawa Trainer" -> "Huấn luyện viên Momozawa", NOT literal unlocalized "Momozawa Trainer")',
    "pronoun_examples": " (e.g. 'em', 'chú', 'con', 'tới', 'tớ')",
    "monologue_guidance": ' (e.g. "Liệu mình/tôi có thắng...", or emotional mental thoughts addressing someone internally like "Em yêu anh..."). In Vietnamese internal monologue, \'mình\'/\'tôi\' are standard for self-reflection, while intimate address pairs (\'em-anh\', \'tớ-cậu\') are completely valid for emotional thoughts directed at a character.',
    "gender_examples": " (e.g., using male pronouns 'anh' for a female character, or incorrect seniority/kinship terms)",
    "proper_name_example": ' (e.g., preserve concise proper names like "Học viện Tracen", do NOT over-expand into long descriptive phrases like "Học viện Đào tạo Mã nương Nhật Bản")',
    "stutter_example": ' (e.g., "Wh-what..." -> "Gì-gì...", "S-sorry..." -> "X-xin lỗi...")',
    "addressee_examples": ' (e.g. "cô nương", "tiểu thư", "thiếu gia", "anh", "chị", "em", "bác sĩ")',
}
