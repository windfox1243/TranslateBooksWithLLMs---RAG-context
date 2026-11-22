from typing import Dict, List, NamedTuple, Optional, Tuple

from src.config import (
    INPUT_TAG_IN,
    INPUT_TAG_OUT,
    TRANSLATE_TAG_IN,
    TRANSLATE_TAG_OUT,
    create_placeholder,
)
from src.languages.vietnamese import prompting as vi_prompting
from src.prompts.examples import (
    TAG0,
    build_placeholder_section,
    get_output_format_example,
    get_subtitle_example,
)
from src.utils.language_profiles import get_language_profile, is_target_language

REFLECTION_JSON_TAG_IN = "<REFLECTION_JSON>"
REFLECTION_JSON_TAG_OUT = "</REFLECTION_JSON>"
REFLECTION_PROMPT_VERSION = "senior-editor-reflection-v8"
REFLECTION_CONTRACT_VERSION = "editor-issue-v9-severity-split"

UNIT_PROMPT_VERSION = "senior-editor-unit-v1"
UNIT_CONTRACT_VERSION = "editor-unit-v10-rewrite"

# Voice evidence is orthogonal to how a repair is addressed, and the narrator
# timeline is built from it, so both contracts carry the identical section
# rather than two copies free to drift apart.
_VOICE_OBSERVATIONS_SCHEMA = {
    "type": "array",
    "maxItems": 12,
    "items": {
        "type": "object",
        "properties": {
            "segment_id": {"type": "string"},
            "discourse_mode": {"type": "string", "enum": [
                "narration", "dialogue", "thought", "letter", "embedded_story"
            ]},
            "narrator_key": {"type": "string"},
            "narrator_identity": {"type": "string"},
            "point_of_view": {"type": "string"},
            "dimensions": {"type": "object"},
            "source_quote": {"type": "string"},
            "target_quote": {"type": "string"},
            "transition_type": {"type": "string", "enum": [
                "none", "chapter", "scene", "explicit"
            ]},
            "transition_evidence": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": [
            "segment_id", "discourse_mode", "narrator_key",
            "narrator_identity", "point_of_view", "dimensions",
            "source_quote", "target_quote", "transition_type",
            "transition_evidence", "confidence"
        ],
        "additionalProperties": False,
    },
}

# The unit contract asks for a judgement and a rewritten unit, and nothing else.
# Where the text sits is answered by the id we issued, so the fields the old
# contract spends most of its failures on -- draft_quote, draft_replacement,
# segment_id, repair_kind -- are simply absent.
UNIT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["no_issues", "needs_repair"]},
        "issues": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                    "unit_id": {"type": "string"},
                    "category": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["blocker", "major", "minor"],
                    },
                    "confidence": {"type": "number"},
                    "instruction": {"type": "string"},
                    "rewritten_unit": {"type": "string"},
                    "glossary_update": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "target": {"type": "string"},
                                },
                                "required": ["source", "target"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                },
                "required": [
                    "issue_id", "unit_id", "category", "severity", "confidence",
                    "instruction", "rewritten_unit", "glossary_update",
                ],
                "additionalProperties": False,
            },
        },
        "voice_observations": _VOICE_OBSERVATIONS_SCHEMA,
    },
    "required": ["status", "issues", "voice_observations"],
    "additionalProperties": False,
}

REFLECTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["no_issues", "needs_repair"]},
        "issues": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                    "segment_id": {"type": "string"},
                    "category": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["blocker", "major", "minor"],
                    },
                    "confidence": {"type": "number"},
                    "repair_kind": {
                        "type": "string",
                        "enum": ["local_replace", "rewrite", "review_only"],
                    },
                    "source_quote": {"type": "string"},
                    "draft_quote": {"type": "string"},
                    "instruction": {"type": "string"},
                    "draft_replacement": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {
                                    "draft": {"type": "string"},
                                    "replacement": {"type": "string"},
                                },
                                "required": ["draft", "replacement"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                    "glossary_update": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "target": {"type": "string"},
                                },
                                "required": ["source", "target"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                },
                "required": [
                    "issue_id", "segment_id", "category", "severity", "confidence",
                    "repair_kind",
                    "source_quote", "draft_quote", "instruction",
                    "draft_replacement", "glossary_update",
                ],
                "additionalProperties": False,
            },
        },
        "voice_observations": _VOICE_OBSERVATIONS_SCHEMA,
    },
    "required": ["status", "issues", "voice_observations"],
    "additionalProperties": False,
}


class PromptPair(NamedTuple):
    """A pair of system and user prompts for LLM translation."""
    system: str
    user: str


# ============================================================================
# SHARED PROMPT SECTIONS
# ============================================================================

def _build_custom_instructions_section(
    custom_instructions: str,
    apply_to: str,
) -> str:
    """Build scoped custom-instruction guidance without weakening hard contracts."""
    if not custom_instructions or not custom_instructions.strip():
        return ""

    return f"""# MANDATORY CUSTOM STYLE INSTRUCTIONS

These instructions have highest priority for style, tone, terminology, naming, and project-specific preferences.
They cannot override output format, tag/placeholder preservation, subtitle markers, source completeness, or security constraints.

{custom_instructions.strip()}

Apply these instructions to {apply_to} wherever they affect style, tone, terminology, or naming.

"""


# What the editor is asked to look for. Shared by both output contracts: the
# audit is the same job whether a repair is addressed by a quoted span or by a
# unit id, and only one of the two should ever have to be kept current.
_AUDIT_CHECKLIST = """AUDIT CHECKLIST -- walk every item against the source before answering:
- omission: source content missing from the draft.
- addition: draft content with no basis in the source.
- mistranslation: the draft states something the source does not.
- glossary error: an established term rendered inconsistently.
- pronoun bleed: an address term that contradicts the established relationship.
- gender mismatch: a referent's gender flipped against the source or context.
- register: honorifics, formality, or politeness level wrong for the speakers.
- consistency: a name, title, or fact contradicting an earlier chunk.
- fluency: target-language phrasing a native reader would not write.
- style: narrator voice or discourse mode drifting from the established profile.
- placeholder/format: a tag, marker, number, or layout element damaged.
Report every material defect you find. Reaching the end of the checklist with
nothing to report is a valid answer only when you actually checked each item."""


def _build_reflection_json_contract_section(*, native_schema: bool = False) -> str:
    """Return the structured Senior Editor output contract."""
    wrapper = (
        "Return exactly one JSON object."
        if native_schema else
        f"Return exactly one JSON object inside {REFLECTION_JSON_TAG_IN} and "
        f"{REFLECTION_JSON_TAG_OUT}."
    )
    example = "" if native_schema else f"""
{REFLECTION_JSON_TAG_IN}
{{"status":"no_issues","issues":[],"voice_observations":[]}}
{REFLECTION_JSON_TAG_OUT}"""
    return f"""{_AUDIT_CHECKLIST}

STRICT OUTPUT CONTRACT:
- {wrapper} No prose or markdown.
- Return at most 12 material issues, ordered by severity and confidence.
- Use no_issues only when no repair is needed; otherwise use needs_repair.
- category must name the checklist item the issue came from.
- local_replace requires one exact numbered draft segment, a unique draft_quote,
  and draft_replacement with exact current and replacement spans.
- rewrite is only for structural, cross-cutting, or completeness defects that
  cannot be repaired locally. Uncertain evidence is review_only.
- severity says how much the defect costs the reader, not how sure you are; a
  minor defect you can point at and fix is still local_replace.
- Omit preferences and equally valid alternatives. Confidence below 0.80 is
  review_only and must not trigger automatic repair.
- glossary_update is only for durable terminology; otherwise use null.
- voice_observations are evidence, not edits. Use exact source/target spans,
  keep discourse modes separate, and return [] when evidence is ambiguous.
- Issue fields are exactly: issue_id, segment_id, category, severity,
  confidence, repair_kind, source_quote, draft_quote, instruction,
  draft_replacement, glossary_update. Do not rename or omit them.
- Follow the supplied response schema exactly.{example}"""

def _get_output_format_section(
    translate_tag_in: str,
    translate_tag_out: str,
    input_tag_in: str,
    input_tag_out: str,
    additional_rules: str = "",
    example_format: str = "Your translated text here"
) -> str:
    """
    Generate standardized output format instructions.

    Args:
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        input_tag_in: Opening tag for input text
        input_tag_out: Closing tag for input text
        additional_rules: Optional additional formatting rules
        example_format: Example text to show in correct format

    Returns:
        str: Formatted output format instructions
    """
    additional_rules_text = f"\n{additional_rules}" if additional_rules else ""

    return f"""# OUTPUT FORMAT

**CRITICAL OUTPUT RULES:**
1. Translate ONLY the text between "{input_tag_in}" and "{input_tag_out}" tags
2. Your response MUST start with {translate_tag_in} (first characters, no text before)
3. Your response MUST end with {translate_tag_out} (last characters, no text after)
4. Include NOTHING before {translate_tag_in} and NOTHING after {translate_tag_out}
5. Do NOT add explanations, comments, notes, or greetings{additional_rules_text}

**INCORRECT examples (DO NOT do this):**
❌ "Here is the translation: {translate_tag_in}Text...{translate_tag_out}"
❌ "{translate_tag_in}Text...{translate_tag_out} (Additional comment)"
❌ "Sure! {translate_tag_in}Text...{translate_tag_out}"
❌ "Text..." (missing tags entirely)
❌ "{translate_tag_in}Text..." (missing closing tag)

**CORRECT format (ONLY this):**
✅ {translate_tag_in}
{example_format}
{translate_tag_out}
"""


# Numbering starts at 6 because rules 1-5 are emitted by _get_output_format_section.
_SUBTITLE_FORMAT_RULES = (
    "\n6. Each subtitle has an index marker: [index]text - PRESERVE these markers exactly"
    "\n7. Keep ONE [index] per subtitle - do NOT merge or split subtitles"
    "\n8. Maintain line breaks between indexed subtitles"
    "\n9. Preserve inline tags (<i>, <b>, <u>, <font ...>, {\\an8}, etc.) and any \\n line breaks INSIDE a subtitle exactly as in the source"
)


# ============================================================================
# OPTIONAL PROMPT SECTIONS
# ============================================================================

# Technical content preservation section (for technical documents)
TECHNICAL_CONTENT_SECTION = """
**Technical Content (DO NOT TRANSLATE):**
- Code snippets and syntax: `function()`, `variable_name`, `class MyClass`
- Command lines: `npm install`, `git commit -m "message"`
- File paths: `/usr/bin/`, `C:/Users/Documents/`
- URLs: `https://example.com`, `www.site.org`
- Programming identifiers, API names, and technical terms"""

# Text cleanup section (for OCR or poorly formatted source texts)
TEXT_CLEANUP_SECTION = """
# TEXT CLEANUP (Source Defects Correction)

The source text may contain OCR errors, formatting artifacts, or typographic defects.
**CORRECT THESE ISSUES during translation:**

- **Line breaks**: Fix broken words (e.g., "trans-\\nlation" → "translation")
- **Spacing**: Remove double spaces, fix missing spaces after punctuation
- **Punctuation**: Correct misplaced or missing punctuation marks
- **Paragraph flow**: Merge incorrectly split paragraphs, preserve intentional breaks

**DO NOT** add content, remove meaningful text, or alter the author's style."""


def _build_optional_prompt_sections(prompt_options: dict) -> str:
    """
    Build optional prompt sections based on the provided options.

    Args:
        prompt_options: Dictionary containing prompt customization flags:
            - preserve_technical_content: DEPRECATED - Technical content is now protected
              via placeholder system (no prompt section needed)
            - text_cleanup: Include OCR/typographic defect correction instructions

    Returns:
        str: Concatenated optional sections to include in the system prompt
    """
    if prompt_options is None:
        prompt_options = {}

    sections = []

    # Technical content preservation is now handled by the placeholder system
    # (TagPreserver with protect_technical=True), so no prompt instructions are needed.
    # The LLM never sees technical content - it's hidden in placeholders like [id0], [id1].
    # Leaving this commented for reference:
    # if prompt_options.get('preserve_technical_content', False):
    #     sections.append(TECHNICAL_CONTENT_SECTION)

    # Text cleanup for OCR or poorly formatted sources
    if prompt_options.get('text_cleanup', False):
        sections.append(TEXT_CLEANUP_SECTION)

    # Join sections with double newline for proper separation
    return '\n\n'.join(sections)


def _build_target_language_style_section(target_language: str) -> str:
    """Return target-language-specific style guardrails."""
    profile = get_language_profile(target_language)
    if profile.prompt_style == "vietnamese":
        return _build_vietnamese_style_section()
    if profile.prompt_style == "japanese":
        return _build_japanese_style_section()
    if profile.prompt_style == "korean":
        return _build_korean_style_section()
    if profile.prompt_style == "chinese":
        return """
# CHINESE STYLE GUARDRAILS

- Use natural Chinese word order for titles, kinship terms, and honorific address.
- Preserve established character names and relationship-based address forms.
- Do not invent kinship or status markers without source/context evidence."""
    if profile.prompt_style == "romance":
        return f"""
# {profile.name.upper()} STYLE GUARDRAILS

- Maintain consistent formal/informal address when the source or context establishes it.
- Preserve gender and number agreement for names, titles, occupations, and relationship terms.
- Do not change a character's gender, formality level, or title from unsupported assumptions."""
    if profile.prompt_style == "germanic":
        return f"""
# {profile.name.upper()} STYLE GUARDRAILS

- Maintain consistent formal/informal address when the source or context establishes it.
- Preserve natural capitalization, title placement, and grammatical agreement.
- Do not change a character's gender, formality level, or title from unsupported assumptions."""
    if profile.prompt_style == "rtl":
        return """
# RTL LANGUAGE STYLE GUARDRAILS

- Use natural right-to-left target-language phrasing and punctuation conventions.
- Preserve names, placeholders, and inline markers exactly.
- Do not invent gendered or honorific address without source/context evidence."""
    return f"""
# TARGET LANGUAGE STYLE GUARDRAILS

- Use natural {target_language or 'target-language'} grammar, punctuation, and register.
- Preserve established names, titles, relationships, and terminology consistently.
- Do not invent gender, kinship, seniority, or formality markers without source/context evidence."""


def _build_vietnamese_style_section() -> str:
    return vi_prompting.STYLE_GUARDRAILS


def _build_japanese_style_section() -> str:
    return """
# JAPANESE STYLE GUARDRAILS

- Treat `## CURRENT ADDRESSING FORMS` as authoritative for direct address, speech levels, and honorific suffixes.
- Preserve character honorific suffixes (e.g. "-sama" / 様, "-senpai" / 先輩, "-san" / さん, "-kun" / 君, "-chan" / ちゃん, "-dono" / 殿) when established by character relationship context, source text, or stored addressing forms.
- Do not drop or replace honorific suffixes with Western neutral pronouns unless context explicitly indicates a change in intimacy or register.
- Maintain consistent first-person (watashi, boku, ore, etc.) and second-person (anata, kimi, omae, etc.) personal pronouns matching each character's established persona across adjacent paragraphs.
- Apply appropriate politeness and speech levels (Desu/Masu vs Tameguchi / formal vs casual register) consistently within each scene."""


def _build_korean_style_section() -> str:
    return """
# KOREAN STYLE GUARDRAILS

- Treat `## CURRENT ADDRESSING FORMS` as authoritative for direct address, speech levels, and honorific titles.
- Preserve character honorific titles and suffixes (e.g. "-ssi" / -씨, "-nim" / -님, "sunbae" / 선배, "oppa" / 오빠, "unnie" / 언니, "hyung" / 형, "noona" / 누나) when established by character relationship context or stored addressing forms.
- Maintain consistent honorific registers (Jondaetmal / 존댓말 vs Banmal / 반말) matching each speaker's seniority, rank, and intimacy level across adjacent paragraphs.
- Do not drop honorific suffixes or switch speech levels abruptly within the same scene unless the relationship or context explicitly changes."""


def _build_relationship_addressing_section() -> str:
    """Return language-neutral guardrails for social address consistency."""
    return """
# RELATIONSHIP AND ADDRESSING GUARDRAILS

- Treat `## CURRENT ADDRESSING FORMS` as authoritative for direct address when a matching speaker/addressee pair is present.
- SITUATIONAL AND TRANSIENT ADDRESS ISOLATION: Treat situational context notes in `## CURRENT ADDRESSING FORMS` carefully. If an addressing entry explicitly specifies a temporary, situational, disguised, sarcastic, or scenario-bound mode (e.g. butler café roleplay, undercover disguise, stage play, acting, temporary game role, mock interaction), apply those address forms ONLY within that specific scene setting; do NOT bleed temporary or situational address forms (such as "Master", "Chủ nhân", or fake title pronouns) into standard narration, general conversations, or non-situational scenes.
- DIALOGUE PRONOUN ISOLATION (CRITICAL): Paired addressing rules (`Speaker → Addressee: self-reference; second-person pronoun`) apply ONLY when speaking to or directly addressing that specific person in dialogue or direct thoughts. Do NOT bleed dialogue self-references (e.g. `em`, `chú`, `con`) into general non-dialogue story narration.
- For languages with pronouns, kinship terms, honorifics, particles, titles, or speech levels, choose them from proven relationship facts: age, family relationship, school year, rank, social status, intimacy, and seniority.
- Apply those relationship facts to both direct address and indirect references in dialogue, thoughts, and narration.
- Do not invent kinship, seniority, or respect markers from gender, affection, politeness, or genre expectation alone. If relationship facts are insufficient, prefer the stored name/title or a neutral form natural for the target language."""


def _build_dialogue_attribution_section(prompt_options: dict) -> str:
    """Build hidden scene-local speaker metadata for the current unit."""
    if not prompt_options:
        return ""
    from src.utils.dialogue_attribution import (
        canonicalize_dialogue_attribution,
        format_dialogue_attribution_for_prompt,
    )
    from src.utils.novel_context import character_alias_map

    attribution = prompt_options.get("dialogue_attribution")
    aliases = character_alias_map(
        prompt_options.get("novel_context", "")
    )
    if aliases:
        attribution = canonicalize_dialogue_attribution(
            attribution,
            aliases,
        )
    return format_dialogue_attribution_for_prompt(
        attribution
    )


def _canonicalized_dialogue_attribution(prompt_options: dict) -> dict:
    """Return dialogue attribution resolved against current novel-context aliases."""
    if not prompt_options:
        return {}
    from src.utils.dialogue_attribution import (
        canonicalize_dialogue_attribution,
        canonicalize_dialogue_state,
    )
    from src.utils.novel_context import character_alias_map

    attribution = prompt_options.get("dialogue_attribution") or {}
    aliases = character_alias_map(
        prompt_options.get("novel_context", "")
    )
    if aliases:
        canonical = canonicalize_dialogue_attribution(
            attribution,
            aliases,
        )
        if not canonical.get("state_after") and attribution.get("state_after"):
            canonical["state_after"] = canonicalize_dialogue_state(
                attribution.get("state_after"),
                aliases,
            )
        return canonical
    return attribution


def _build_novel_context_section(
    prompt_options: dict,
    reference_text: str = "",
    refinement: bool = False,
) -> str:
    """Build the dynamic novel context block for the user prompt.

    Novel context changes per chunk, so it belongs with user content rather
    than the system prompt. This keeps provider-side system prompt caching
    effective while still giving each request the relevant book memory.
    """
    if not prompt_options:
        return ""

    prompt_bundle = str(prompt_options.get("prompt_context_bundle") or "").strip()
    novel_context = prompt_options.get('novel_context')
    if prompt_bundle:
        rendered_context = prompt_bundle
    else:
        if not novel_context or not str(novel_context).strip():
            return ""

        attribution = _canonicalized_dialogue_attribution(prompt_options)
        state_after = attribution.get("state_after") or {}
        active_speaker = state_after.get("speaker")

        from src.utils.novel_context import render_novel_context_for_prompt

        rendered_context = render_novel_context_for_prompt(
            novel_context,
            reference_text=reference_text,
            max_tokens=prompt_options.get('novel_context_prompt_max_tokens'),
            selective=prompt_options.get('novel_context_selective_injection', True),
            active_speaker=active_speaker,
        )
    if not rendered_context.strip():
        return ""

    guidance = """

Use this context for narrative consistency: character identity, proven gender,
aliases, relationships, addressing, and discovered terminology hints.
Treat stated character genders as binding continuity facts. Do not reinterpret
or change a character's gender from a title, role, old body, disguise,
relationship label, or model prior.
If this context conflicts with `# GLOSSARY - REQUIRED TRANSLATIONS`, the
required glossary wins."""
    if refinement:
        guidance += """

Use book-wide character identity, proven gender, and terminology for consistency.
The global lore may contain facts discovered later in the book: never add,
foreshadow, or reveal any fact that is not already present in the draft or its
immediate local context."""

    return f"""# NOVEL CONTEXT (CHARACTERS, RELATIONSHIPS & GLOSSARY)

{rendered_context.strip()}{guidance}"""


def _build_directed_addressing_section(prompt_options: dict) -> str:
    """Build a structured DB-backed directed-addressing prompt block."""
    if not prompt_options:
        return ""
    if str(prompt_options.get("prompt_context_bundle") or "").strip():
        return ""
    directed_context = str(
        prompt_options.get("directed_addressing_context") or ""
    ).strip()
    if not directed_context:
        return ""
    return f"""# STRUCTURED DIRECTED ADDRESSING RULES

{directed_context}

These pair-specific rules override markdown `## CURRENT ADDRESSING FORMS` for the same speaker/addressee pair. Use markdown novel context only when no structured rule matches the current pair."""


def _build_relationship_context_section(prompt_options: dict) -> str:
    """Build the accepted relationship graph prompt block in project mode."""

    if not prompt_options:
        return ""
    if str(prompt_options.get("prompt_context_bundle") or "").strip():
        return ""
    relationship_context = str(
        prompt_options.get("relationship_context") or ""
    ).strip()
    if not relationship_context:
        return ""
    return f"""# STRUCTURED RELATIONSHIP CONTEXT

{relationship_context}

Use only accepted graph facts. Situational facts apply only to the current scene. Locked facts are immutable. This block overrides conflicting markdown relationship claims; pair-specific structured addressing rules still have higher priority for actual address forms."""


# ============================================================================
# TRANSLATION PROMPT FUNCTIONS
# ============================================================================

def generate_translation_prompt(
    main_content: str,
    context_before: str,
    context_after: str,
    previous_translation_context: str,
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    glossary_block: str = "",
) -> PromptPair:
    """
    Generate the translation prompt with all contextual elements.

    Args:
        main_content: The text to translate
        context_before: Text appearing before main_content for context
        context_after: Text appearing after main_content for context
        previous_translation_context: Previously translated text for consistency
        source_language: Source language name
        target_language: Target language name
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        has_placeholders: If True, includes placeholder preservation instructions (for EPUB HTML tags)
        prompt_options: Optional dict with prompt customization options:
            - preserve_technical_content: If True, includes instructions to NOT translate
              code, paths, URLs, etc. (for technical documents)
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format.
            If None, uses default [[0]] format

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    # Default None languages to empty strings
    source_language = source_language or ""
    target_language = target_language or ""

    # Initialize prompt_options if not provided
    if prompt_options is None:
        prompt_options = {}

    # Extract custom instructions if provided
    custom_instructions = prompt_options.get('custom_instructions', '')

    # Get target-language-specific example text for output format
    example_texts = {
        "chinese": "您翻译的文本在这里" if not has_placeholders else f"您翻译的文本在这里，所有{TAG0}标记都精确保留",
        "french": "Votre texte traduit ici" if not has_placeholders else f"Votre texte traduit ici, tous les marqueurs {TAG0} sont préservés exactement",
        "spanish": "Su texto traducido aquí" if not has_placeholders else f"Su texto traducido aquí, todos los marcadores {TAG0} se preservan exactamente",
        "german": "Ihr übersetzter Text hier" if not has_placeholders else f"Ihr übersetzter Text hier, alle {TAG0}-Markierungen werden genau beibehalten",
        "japanese": "翻訳されたテキストはこちら" if not has_placeholders else f"翻訳されたテキストはこちら、すべての{TAG0}マーカーは正確に保持されます",
        "italian": "Il tuo testo tradotto qui" if not has_placeholders else f"Il tuo testo tradotto qui, tutti i marcatori {TAG0} sono conservati esattamente",
        "portuguese": "Seu texto traduzido aqui" if not has_placeholders else f"Seu texto traduzido aqui, todos os marcadores {TAG0} são preservados exatamente",
        "russian": "Ваш переведенный текст здесь" if not has_placeholders else f"Ваш переведенный текст здесь, все маркеры {TAG0} сохранены точно",
        "korean": "번역된 텍스트는 여기에" if not has_placeholders else f"번역된 텍스트는 여기에, 모든 {TAG0} 마커는 정확히 보존됩니다",
    }

    # Try to match target language to get appropriate example
    target_lang_lower = (target_language or "").lower()
    example_format_text = example_texts.get(target_lang_lower, "Your translated text here")

    # Build the output format section outside the f-string to avoid backslash issues in Python 3.11
    output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules="",
        example_format=example_format_text
    )

    # Build placeholder preservation section dynamically based on languages
    if has_placeholders:
        placeholder_section = build_placeholder_section(source_language, target_language, placeholder_format)
    else:
        placeholder_section = ""

    # Build optional prompt sections based on prompt_options
    optional_sections = _build_optional_prompt_sections(prompt_options)
    relationship_addressing_section = _build_relationship_addressing_section()
    target_language_style_section = _build_target_language_style_section(
        target_language
    )

    custom_instructions_section = _build_custom_instructions_section(
        custom_instructions,
        "every word you translate",
    )

    # SYSTEM PROMPT - Role and instructions (stable across requests)
    system_prompt = f"""You are a professional {target_language} translator and writer.

{custom_instructions_section}# TRANSLATION PRINCIPLES

Translate {source_language} to {target_language}. Output only the translation.

**PRIORITY ORDER:**
1. Preserve exact names and concise proper nouns (keep concise academy/school/institution names intact rather than over-expanding them into long, wordy descriptive paraphrases).
   - **SIMILAR NAMES WARNING:** Be extremely careful with similar proper names. If two names look alike (e.g., 'Alex' and 'Alles'), treat them as distinct, separate entities unless explicitly told they are aliases. Never apply a correction or translation meant for one character name to a different character name just because they sound or look similar.
2. Match original tone and formality
3. Use natural {target_language} phrasing - never word-for-word
4. Fix grammar/spelling errors in output
5. Translate idioms to {target_language} equivalents

**QUALITY CHECK:**
- Does it sound natural to a native {target_language} speaker?
- Are all details from the original included?
- Does punctuation follow {target_language} conventions?

If unsure between literal and natural phrasing: **choose natural**.

**LAYOUT PRESERVATION:**
- Keep the exact text layout, spacing, line breaks, and indentation
- **WRITE YOUR TRANSLATION IN {target_language.upper()} - THIS IS MANDATORY**
{optional_sections}
{relationship_addressing_section}
{target_language_style_section}
{placeholder_section}

# FINAL REMINDER: YOUR OUTPUT LANGUAGE

**YOU MUST TRANSLATE INTO {target_language.upper()}.**
Your entire translation output must be written in {target_language}.
Do NOT write in {source_language} or any other language - ONLY {target_language.upper()}.

{output_format_section}"""

    # USER PROMPT - Context and content to translate (varies per request)
    previous_translation_block_text = ""
    if previous_translation_context and previous_translation_context.strip():
        previous_translation_block_text = f"""# CONTEXT - Previous Paragraph

For consistency and natural flow, here's what came immediately before:

{previous_translation_context}

"""

    # Glossary block lives in the user prompt: it changes per chunk, so
    # keeping it out of the system prompt lets the system prompt stay
    # stable and cacheable across chunks.
    directed_addressing_section = _build_directed_addressing_section(prompt_options)
    relationship_context_section = _build_relationship_context_section(prompt_options)
    dialogue_section = _build_dialogue_attribution_section(prompt_options)
    narrator_voice_context = str(
        prompt_options.get("narrative_voice_context") or ""
    ).strip()
    bundle_has_narrator = bool(str(prompt_options.get("prompt_context_bundle") or "").strip())
    narrator_voice_section = (
        "# ESTABLISHED NARRATOR VOICE (HISTORICAL TIMELINE)\n"
        + narrator_voice_context
        + "\nApply it only to the matching narrator and discourse scope; dialogue addressing cannot override it.\n\n"
        if narrator_voice_context and not bundle_has_narrator else ""
    )
    novel_context_section = _build_novel_context_section(
        prompt_options,
        reference_text="\n".join(
            part for part in (
                context_before,
                main_content,
                context_after,
                previous_translation_context,
                dialogue_section,
            )
            if part
        ),
    )
    if novel_context_section:
        novel_context_section = f"{novel_context_section}\n\n"
    if directed_addressing_section:
        directed_addressing_section = f"{directed_addressing_section}\n\n"
    if relationship_context_section:
        relationship_context_section = f"{relationship_context_section}\n\n"
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""
    if dialogue_section:
        dialogue_section = f"{dialogue_section}\n\n"

    # Retry after an answer that left source text behind (issue #270). Kept in
    # the user prompt so the cacheable system prompt never changes.
    residue_retry_section = ""
    if prompt_options.get('source_residue_retry'):
        residue_retry_section = f"""# IMPORTANT - PREVIOUS ATTEMPT INCOMPLETE

A previous attempt left part of this text in {source_language}. Translate EVERY sentence into {target_language}, including short, repeated or quoted ones. Do not skip, summarize or copy any {source_language} passage.

"""

    user_prompt = f"""{previous_translation_block_text}{directed_addressing_section}{relationship_context_section}{narrator_voice_section}{novel_context_section}{glossary_section}{dialogue_section}{residue_retry_section}# TEXT TO TRANSLATE

{INPUT_TAG_IN}
{main_content}
{INPUT_TAG_OUT}

REMINDER: Output ONLY your translation in this exact format:
{translate_tag_in}
your translation here
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your translation now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


# ============================================================================
# GLOSSARY: NER EXTRACTION PROMPT (Phase 2)
# ============================================================================

NER_TAG_IN = "<NER_JSON>"
NER_TAG_OUT = "</NER_JSON>"


def generate_ner_extraction_prompt(
    text: str,
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    related_glossary_terms: Optional[Dict[str, str]] = None,
) -> PromptPair:
    """
    Build a prompt that asks the LLM to extract recurring proper-noun entities
    (characters, locations, organizations/sects, items) from a sample of source
    text, along with a suggested target-language translation for each.

    Output is wrapped in <NER_JSON>...</NER_JSON> with a strict schema. The
    parser is permissive (handles markdown fences, missing tags, partial JSON).
    """
    vietnamese_skill_guidance = ""
    if is_target_language(target_language, "vi"):
        vietnamese_skill_guidance = vi_prompting.NER_SKILL_GUIDANCE

    system_prompt = f"""You are a literary entity extractor. Your job is to read a passage written in {source_language} and identify recurring proper nouns that a translator would want to keep consistent across an entire book.

# CATEGORIES (use exactly these labels)

- "character"     — named persons (李凡, Li Fan, Captain Ahab)
- "location"      — places, regions, buildings (青玄宗大殿, Mount Tai)
- "organization"  — sects, schools, clans, factions, companies (青玄宗, Heavenly Sword Gate)
- "item"          — named artifacts, weapons, treasures, techniques (混沌珠, Excalibur)
- "title"         — honorifics or named ranks tied to a person (Elder, 长老, Master)
- "other"         — anything else worth keeping consistent (events, magical formulas)

# RULES

1. Extract ONLY proper nouns or named concepts that look likely to recur. Skip generic words.
2. Do NOT translate common nouns or descriptive phrases — only named entities.
3. For each entity, propose ONE canonical {target_language} translation. Use the standard romanization or the most natural literary rendering. Keep the proposal concise.
4. Deduplicate: if the same entity appears multiple times in the passage, list it once.
5. If you are unsure about an entry, omit it rather than guessing.
6. Preserve the original {source_language} form exactly as it appears in the text (no extra spaces, no normalization).
7. If the related existing glossary contains part of a longer candidate, reuse the established target for that part when it is semantically the same. Prefer a longer existing glossary entry over composing a new one from shorter parts.
{vietnamese_skill_guidance}

# OUTPUT FORMAT

Return ONLY a JSON array wrapped between {NER_TAG_IN} and {NER_TAG_OUT}. No prose, no explanations.

Each array element MUST be an object with these keys:
  - "source"   (string, required) — the entity in {source_language}
  - "target"   (string, required) — the proposed {target_language} translation
  - "category" (string, required) — one of the labels listed above

Example:
{NER_TAG_IN}
[
  {{"source": "李凡", "target": "Li Fan", "category": "character"}},
  {{"source": "青玄宗", "target": "Qingxuan Sect", "category": "organization"}}
]
{NER_TAG_OUT}

If no entities are found, return an empty array: {NER_TAG_IN}[]{NER_TAG_OUT}.

Do NOT wrap the JSON in markdown code fences. Do NOT add commentary before or after the tags."""

    related_glossary_block = ""
    if related_glossary_terms:
        related_lines = [
            "# RELATED EXISTING GLOSSARY",
            "",
            "Use these existing translations as consistency hints for overlapping names or compound terms. Do not repeat an entry that is already complete; suggest only genuinely new longer terms or missing related entities.",
            "",
        ]
        for source, target in related_glossary_terms.items():
            related_lines.append(f"  - {source} -> {target}")
        related_glossary_block = "\n".join(related_lines) + "\n\n"

    user_prompt = f"""# SOURCE TEXT ({source_language})

{related_glossary_block}
{INPUT_TAG_IN}
{text}
{INPUT_TAG_OUT}

Extract the recurring named entities now. Output the JSON array between {NER_TAG_IN} and {NER_TAG_OUT}, nothing else."""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


def generate_refinement_prompt(
    draft_translation: str,
    context_before: str = "",
    context_after: str = "",
    previous_refined_context: str = "",
    target_language: Optional[str] = None,
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    additional_instructions: str = "",
    glossary_block: str = "",
) -> PromptPair:
    """
    Generate a refinement prompt to polish a draft translation.

    This is used for a second pass where the LLM improves a first-pass translation,
    focusing on literary quality, natural flow, and stylistic excellence.

    Args:
        draft_translation: The first-pass translation to refine
        context_before: Previously refined text for context (default: "")
        context_after: Text appearing after for context (default: "")
        previous_refined_context: Last refined text for consistency (default: "")
        target_language: Target language name
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        has_placeholders: If True, includes placeholder preservation instructions
        prompt_options: Optional dict with prompt customization options
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format.
            If None, uses default [[0]] format
        additional_instructions: Additional refinement instructions to include in the prompt (default: "")

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    target_language = target_language or ""

    if prompt_options is None:
        prompt_options = {}

    # Get target-language-specific example text for output format
    example_texts = {
        "chinese": "您润色后的文本在这里",
        "french": "Votre texte affiné ici",
        "spanish": "Su texto refinado aquí",
        "german": "Ihr verfeinerter Text hier",
        "japanese": "洗練されたテキストはこちら",
        "italian": "Il tuo testo raffinato qui",
        "portuguese": "Seu texto refinado aqui",
        "russian": "Ваш улучшенный текст здесь",
        "korean": "다듬어진 텍스트는 여기에",
    }

    target_lang_lower = target_language.lower()
    example_format_text = example_texts.get(target_lang_lower, "Your refined text here")

    output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules="",
        example_format=example_format_text
    )

    # Build placeholder preservation section if needed
    if has_placeholders:
        placeholder_section = build_placeholder_section(target_language, target_language, placeholder_format)
    else:
        placeholder_section = ""

    # Build optional prompt sections
    optional_sections = _build_optional_prompt_sections(prompt_options)
    relationship_addressing_section = _build_relationship_addressing_section()
    target_language_style_section = _build_target_language_style_section(
        target_language
    )

    # Add additional instructions section if provided
    additional_instructions_section = ""
    if additional_instructions and additional_instructions.strip():
        additional_instructions_section = f"""

# ADDITIONAL REFINEMENT INSTRUCTIONS

{additional_instructions.strip()}"""

    # SYSTEM PROMPT for refinement
    system_prompt = f"""You are an elite {target_language} literary editor and prose stylist.

# YOUR TASK: REFINE AND POLISH

You will receive a DRAFT {target_language} translation that needs significant improvement.
Your job is to REWRITE it with perfect literary {target_language} style.

**THE INPUT IS:**
- A rough, literal, or awkward {target_language} translation
- It may have unnatural phrasing, stilted expressions, or poor flow
- Consider it a "bad" first draft that probably needs substantial reworking

**YOUR OUTPUT MUST BE:**
- Fluent, natural {target_language} prose
- Stylistically excellent - as if written by a skilled {target_language} author

# REFINEMENT PRINCIPLES

**PRIORITY ORDER:**
1. **Natural flow** - Sentences should flow beautifully in {target_language}
2. **Idiomatic expressions** - Use natural {target_language} idioms and phrasings
3. **Elegant word choice** - Select the most appropriate and refined vocabulary
4. **Rhythm and cadence** - The text should have pleasant reading rhythm
5. **Preserve meaning** - Keep the original meaning intact while improving style

**WHAT TO FIX:**
- Awkward literal translations → Natural {target_language} expressions
- Repetitive or dull vocabulary → Rich, varied word choices
- Unnatural word order → Proper {target_language} syntax
- **Passages left in another language** → Translate them into {target_language}; the draft must not keep any sentence in its source language
- **Lexical repetitions and cacophony** → Use synonyms to avoid same-root word repetition
  (e.g., "the singer sang a song" → "the singer performed a song" or "the vocalist sang a melody")

**WHAT TO PRESERVE:**
- All factual content and meaning
- Character names and proper nouns
- Technical terms (if any)
{optional_sections}
{relationship_addressing_section}
{target_language_style_section}
{placeholder_section}
{additional_instructions_section}

# CRITICAL REMINDER

You are REWRITING in {target_language.upper()}, not translating from scratch.
The input is meant to be in {target_language}, but poorly written. If a passage
was left in another language, translate it into {target_language} as part of the rewrite.
Your output must be polished, literary-quality {target_language}.

**⚠️ PLACEHOLDER PRESERVATION IS ABSOLUTELY CRITICAL:**
If the input contains ANY placeholders (like [id0], [id1], etc.), you MUST preserve them EXACTLY.
Removing or corrupting placeholders will corrupt the document structure.
Your refinement MUST maintain the exact same placeholders in the exact same positions.

{output_format_section}"""

    # USER PROMPT
    previous_context_block = ""
    if previous_refined_context and previous_refined_context.strip():
        previous_context_block = f"""# CONTEXT - Previous Refined Paragraph

For consistency and natural flow, here's what came immediately before:

{previous_refined_context}

"""

    directed_addressing_section = _build_directed_addressing_section(prompt_options)
    relationship_context_section = _build_relationship_context_section(prompt_options)
    dialogue_section = _build_dialogue_attribution_section(prompt_options)
    narrator_voice_context = str(
        prompt_options.get("narrative_voice_context") or ""
    ).strip()
    bundle_has_narrator = bool(str(prompt_options.get("prompt_context_bundle") or "").strip())
    narrator_voice_section = (
        "# ESTABLISHED NARRATOR VOICE (HISTORICAL, READ-ONLY)\n"
        + narrator_voice_context
        + "\nRefinement must preserve this profile and must not infer or rewrite narrator history.\n\n"
        if narrator_voice_context and not bundle_has_narrator else ""
    )
    # Glossary block injected here (per-chunk dynamic) so the system prompt
    # stays cacheable across chunks.
    novel_context_section = _build_novel_context_section(
        prompt_options,
        reference_text="\n".join(
            part for part in (
                context_before,
                draft_translation,
                context_after,
                previous_refined_context,
                dialogue_section,
            )
            if part
        ),
        refinement=True,
    )
    if novel_context_section:
        novel_context_section = f"{novel_context_section}\n\n"
    if directed_addressing_section:
        directed_addressing_section = f"{directed_addressing_section}\n\n"
    if relationship_context_section:
        relationship_context_section = f"{relationship_context_section}\n\n"
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""
    if dialogue_section:
        dialogue_section = f"{dialogue_section}\n\n"

    user_prompt = f"""{previous_context_block}{directed_addressing_section}{relationship_context_section}{narrator_voice_section}{novel_context_section}{glossary_section}{dialogue_section}# DRAFT TO REFINE

The following is a rough {target_language} translation that needs significant improvement.
Rewrite it with elegant, literary-quality {target_language} prose:

{INPUT_TAG_IN}
{draft_translation}
{INPUT_TAG_OUT}

REMINDER: Output ONLY your refined text in this exact format:
{translate_tag_in}
your refined text here
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your refined version now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


def generate_subtitle_refinement_block_prompt(
    subtitle_blocks: List[Tuple[int, str]],
    previous_refined_block: str = "",
    target_language: Optional[str] = None,
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    additional_instructions: str = "",
    glossary_block: str = "",
    prompt_options: dict = None,
) -> PromptPair:
    """
    Generate a refinement prompt for multiple subtitles in a single LLM call.

    Mirrors generate_subtitle_block_prompt but rewrites each draft subtitle into
    polished target-language prose while preserving the [index] markers.

    Args:
        subtitle_blocks: List of tuples (local_index, draft_translated_text)
        previous_refined_block: Last refined block for continuity
        target_language: Target language
        translate_tag_in: Opening tag for refinement output
        translate_tag_out: Closing tag for refinement output
        additional_instructions: Extra refinement guidance
        glossary_block: Optional glossary block
        prompt_options: Optional dict with prompt customization options

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    target_language = target_language or ""

    if prompt_options is None:
        prompt_options = {}

    subtitle_additional_rules = _SUBTITLE_FORMAT_RULES
    subtitle_example_format = "[0]Première ligne affinée\n[1]Deuxième ligne affinée"
    subtitle_output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules=subtitle_additional_rules,
        example_format=subtitle_example_format,
    )
    relationship_addressing_section = _build_relationship_addressing_section()
    target_language_style_section = _build_target_language_style_section(
        target_language
    )

    additional_instructions_section = ""
    if additional_instructions and additional_instructions.strip():
        additional_instructions_section = f"""

# ADDITIONAL REFINEMENT INSTRUCTIONS

{additional_instructions.strip()}"""

    system_prompt = f"""You are an elite {target_language} subtitle editor and dialogue stylist.

# YOUR TASK: REFINE A BLOCK OF SUBTITLES

You will receive a block of DRAFT {target_language} subtitles, each prefixed with an [index] marker.
Your job is to REWRITE each subtitle with natural, idiomatic {target_language} dialogue while
preserving the index markers and the one-subtitle-per-marker structure.

**THE INPUT IS:**
- A block of draft {target_language} subtitles, possibly literal or awkward
- Each subtitle is prefixed with [N] where N is its local index

**YOUR OUTPUT MUST BE:**
- The same number of subtitles, each prefixed with the SAME [N] marker
- Fluent, natural spoken {target_language} suited to subtitling

# REFINEMENT PRINCIPLES

**PRIORITY ORDER:**
1. **Natural dialogue** - sound like real {target_language} speech, not translation
2. **Reading speed** - keep subtitle length viewer-friendly
3. **Continuity** - terminology and tone consistent across the block
4. **Preserve meaning** - keep the original meaning intact while improving style

**WHAT TO FIX:**
- Awkward literal phrasing -> natural {target_language} expressions
- Repetitive vocabulary that is clearly an artefact of literal translation -> varied word choices
- Unnatural word order -> proper {target_language} syntax

**WHAT TO PRESERVE:**
- The [index] markers exactly as given
- All factual content and meaning
- Character names and proper nouns
- The one-subtitle-per-[index] structure (no merging, no splitting)
- Intentional repetitions (e.g. "No. No. No.") and dialogue dashes ("- ...\\n- ...") when present in the draft
- Inline formatting tags and any \\n line breaks inside a subtitle{additional_instructions_section}
{relationship_addressing_section}
{target_language_style_section}

# CRITICAL REMINDERS

You are NOT translating - you are REWRITING in {target_language.upper()}.
The input is already in {target_language}, but possibly poorly written.
Your output must be polished, natural {target_language} dialogue.

**Index markers are MANDATORY:** every input [N] must appear exactly once in the output,
in the same order, followed by the refined text for that subtitle.

{subtitle_output_format_section}"""

    previous_refined_block_text = ""
    if previous_refined_block and previous_refined_block.strip():
        previous_refined_block_text = f"""# CONTEXT - Previous Refined Block

For continuity and consistency, here's the previous refined block:

{previous_refined_block}

"""

    formatted_subtitles = [f"[{idx}]{text}" for idx, text in subtitle_blocks]
    formatted_subtitles_text = "\n".join(formatted_subtitles)

    directed_addressing_section = _build_directed_addressing_section(prompt_options)
    relationship_context_section = _build_relationship_context_section(prompt_options)
    dialogue_section = _build_dialogue_attribution_section(prompt_options)
    novel_context_section = _build_novel_context_section(
        prompt_options,
        reference_text="\n".join(
            part for part in (
                previous_refined_block,
                formatted_subtitles_text,
                dialogue_section,
            )
            if part
        ),
        refinement=True,
    )
    if novel_context_section:
        novel_context_section = f"{novel_context_section}\n\n"
    if directed_addressing_section:
        directed_addressing_section = f"{directed_addressing_section}\n\n"
    if relationship_context_section:
        relationship_context_section = f"{relationship_context_section}\n\n"
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""
    if dialogue_section:
        dialogue_section = f"{dialogue_section}\n\n"

    user_prompt = f"""{previous_refined_block_text}{directed_addressing_section}{relationship_context_section}{novel_context_section}{glossary_section}{dialogue_section}# SUBTITLES TO REFINE

{INPUT_TAG_IN}
{formatted_subtitles_text}
{INPUT_TAG_OUT}

REMINDER: Output format must be:
{translate_tag_in}
[0]refined subtitle 0
[1]refined subtitle 1
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your refined block now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


def generate_subtitle_block_prompt(
    subtitle_blocks: List[Tuple[int, str]],
    previous_translation_block: str,
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    custom_instructions: str = "",
    glossary_block: str = "",
    prompt_options: dict = None,
) -> PromptPair:
    """
    Generate translation prompt for multiple subtitle blocks with index markers.

    Args:
        subtitle_blocks: List of tuples (index, text) for subtitles to translate
        previous_translation_block: Previous translated block for context
        source_language: Source language
        target_language: Target language
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        custom_instructions: Additional custom translation instructions
        glossary_block: Optional glossary block
        prompt_options: Optional dict with prompt customization options

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    source_language = source_language or ""
    target_language = target_language or ""

    if prompt_options is None:
        prompt_options = {}

    # Extract custom instructions from prompt_options if not explicitly provided
    if not custom_instructions:
        custom_instructions = prompt_options.get('custom_instructions', '')

    # Build the output format section outside the f-string to avoid backslash issues in Python 3.11
    subtitle_additional_rules = _SUBTITLE_FORMAT_RULES
    subtitle_example_format = "[1]第一行翻译文本\n[2]第二行翻译文本"
    subtitle_output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules=subtitle_additional_rules,
        example_format=subtitle_example_format
    )
    relationship_addressing_section = _build_relationship_addressing_section()
    target_language_style_section = _build_target_language_style_section(
        target_language
    )

    custom_instructions_section = _build_custom_instructions_section(
        custom_instructions,
        "every subtitle",
    )

    # SYSTEM PROMPT - Role and instructions for subtitle translation
    system_prompt = f"""You are a professional {target_language} subtitle translator and dialogue adaptation specialist.

# CRITICAL: TARGET LANGUAGE IS {target_language.upper()}

**YOUR SUBTITLE TRANSLATION MUST BE WRITTEN ENTIRELY IN {target_language.upper()}.**

You are translating subtitles FROM {source_language} TO {target_language}.
Your output must be in {target_language} ONLY - do NOT use any other language.

# SUBTITLE TRANSLATION PRINCIPLES

**Quality Standards:**
- Translate dialogues naturally and conversationally for {target_language} viewers
- Adapt expressions, slang, and cultural references appropriately
- Keep subtitle length readable (typically 40-42 characters per line)
- Restructure sentences naturally (avoid word-by-word translation)
- Maintain speaker's tone, personality, and emotion
- **WRITE YOUR TRANSLATION IN {target_language.upper()} - THIS IS MANDATORY**

**Subtitle-Specific Rules:**
- Prioritize clarity and reading speed over literal accuracy
- Condense when necessary without losing meaning
- Use natural, spoken {target_language} (not formal written style)
- Preserve intentional repetitions (e.g. "No. No. No.") and dialogue dashes ("- ...\\n- ...") from the source
- Preserve inline formatting tags (<i>, <b>, <font ...>, {{\\an8}}, etc.) and any \\n line breaks inside a subtitle
{custom_instructions_section}
{relationship_addressing_section}
{target_language_style_section}

# FINAL REMINDER: YOUR OUTPUT LANGUAGE

**YOU MUST TRANSLATE INTO {target_language.upper()}.**
Your entire subtitle translation must be written in {target_language}.
Do NOT write in {source_language} or any other language - ONLY {target_language.upper()}.

{subtitle_output_format_section}"""

    # USER PROMPT - Context and subtitles to translate
    previous_translation_block_text = ""
    if previous_translation_block and previous_translation_block.strip():
        previous_translation_block_text = f"""# CONTEXT - Previous Subtitle Block

For continuity and consistency, here's the previous subtitle block:

{previous_translation_block}

"""

    # Format subtitle blocks with indices
    formatted_subtitles = [f"[{idx}]{text}" for idx, text in subtitle_blocks]

    # Join subtitles outside f-string to avoid Python 3.11 backslash issues
    formatted_subtitles_text = "\n".join(formatted_subtitles)

    directed_addressing_section = _build_directed_addressing_section(prompt_options)
    relationship_context_section = _build_relationship_context_section(prompt_options)
    dialogue_section = _build_dialogue_attribution_section(prompt_options)
    # Novel context and glossary blocks live in the user prompt because they
    # vary per chunk/block.
    novel_context_section = _build_novel_context_section(
        prompt_options,
        reference_text="\n".join(
            part for part in (
                previous_translation_block,
                formatted_subtitles_text,
                dialogue_section,
            )
            if part
        ),
    )
    if novel_context_section:
        novel_context_section = f"{novel_context_section}\n\n"
    if directed_addressing_section:
        directed_addressing_section = f"{directed_addressing_section}\n\n"
    if relationship_context_section:
        relationship_context_section = f"{relationship_context_section}\n\n"
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""
    if dialogue_section:
        dialogue_section = f"{dialogue_section}\n\n"

    user_prompt = f"""{previous_translation_block_text}{directed_addressing_section}{relationship_context_section}{novel_context_section}{glossary_section}{dialogue_section}# SUBTITLES TO TRANSLATE

{INPUT_TAG_IN}
{formatted_subtitles_text}
{INPUT_TAG_OUT}

REMINDER: Output format must be:
{translate_tag_in}
[1]translated subtitle 1
[2]translated subtitle 2
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your translation now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


# ============================================================================
# PLACEHOLDER CORRECTION PROMPT
# ============================================================================

# ============================================================================
# ALIAS FOR BACKWARDS COMPATIBILITY
# ============================================================================

def generate_post_processing_prompt(
    translated_text: str,
    target_language: Optional[str] = None,
    context_before: str = "",
    context_after: str = "",
    additional_instructions: str = "",
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    glossary_block: str = "",
) -> PromptPair:
    """
    Alias for generate_refinement_prompt with parameter name mapping.

    This function exists for backwards compatibility and to provide a more intuitive
    API for post-processing/refinement use cases.

    Args:
        translated_text: The draft translation to refine (mapped to draft_translation)
        target_language: Target language name
        context_before: Previously refined text for context
        context_after: Text appearing after for context
        additional_instructions: Additional refinement instructions
        has_placeholders: If True, includes placeholder preservation instructions
        prompt_options: Optional dict with prompt customization options
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    return generate_refinement_prompt(
        draft_translation=translated_text,
        context_before=context_before,
        context_after=context_after,
        previous_refined_context="",  # Not used in post-processing calls
        target_language=target_language,
        has_placeholders=has_placeholders,
        prompt_options=prompt_options,
        placeholder_format=placeholder_format,
        additional_instructions=additional_instructions,
        glossary_block=glossary_block,
    )


def generate_chunk_reflection_prompt(
    source_chunk: str,
    draft_translation: str,
    target_language: Optional[str] = None,
    novel_context: str = "",
    custom_instructions: str = "",
    glossary_block: str = "",
    deterministic_findings: str = "",
    source_available: bool = True,
    native_schema: bool = False,
    narrative_voice_context: str = "",
) -> PromptPair:
    """
    Generate a chunk reflection prompt acting as a Senior Translation Editor / Critic.
    Evaluates line-by-line fidelity, narrative vs. dialogue pronoun isolation, and term accuracy.
    """
    target_lang = target_language or "target language"
    from src.utils.translation_quality import format_editor_segments

    reflection_contract = _build_reflection_json_contract_section(
        native_schema=native_schema,
    )

    source_mode = (
        "Perform bilingual source-fidelity and target-language quality review."
        if source_available
        else "Perform monolingual target-language style, fluency, consistency, glossary, relationship, and formatting review. Do not claim omissions, mistranslations, or source residue because no aligned source is available."
    )
    system_prompt = f"""You are an evidence-driven literary translation editor for {target_lang}.
{source_mode}

AUDIT POLICY:
- Compare every source segment with the numbered draft for completeness,
  meaning, terminology, proper names, grammar, register, and structure.
- Report only material defects supported by exact source and draft evidence.
- Explicit source names, nicknames, titles, honorifics, stuttering, and
  intentional register changes override background defaults.
- Use context only when the source is implicit. Never invent relationships,
  gender, hierarchy, or reverse addressing from absent evidence.
- Keep narration, dialogue, thought, letters, and embedded stories separate.
  Pair-specific dialogue forms never define narrator voice.
- Apply an established narrator policy across matching narrative scope and
  chapters; do not extend it into dialogue or a different narrator.
- A gendered vocative describes its addressee, not automatically its speaker.
- Preserve concise identity-bearing names as complete spans. Do not classify or
  translate a token inside a protected name independently.
- Prefer no issue over speculative polishing. Preserve all valid wording.

{reflection_contract}"""

    user_sections = [
        f"# RAW SOURCE CHUNK:\n{source_chunk.strip()}"
        if source_available
        else "# SOURCE CAPABILITY:\nMonolingual review only; no aligned source chunk is available."
    ]
    if custom_instructions and custom_instructions.strip():
        user_sections.append(f"# CUSTOM INSTRUCTIONS & STYLE GUIDELINES:\n{custom_instructions.strip()}")
    if glossary_block and glossary_block.strip():
        user_sections.append(f"# GLOSSARY & TERM MAPPING:\n{glossary_block.strip()}")
    if deterministic_findings and deterministic_findings.strip():
        user_sections.append(
            "# DETERMINISTIC FINAL-OUTPUT FINDINGS (MANDATORY)\n"
            "These findings were produced by a non-LLM validator. Resolve each "
            "high-confidence span. Narrator findings must be reported as blocker "
            "rewrite issues and may never be downgraded to review_only. Report an "
            "exact draft_replacement when a safe local edit is possible:\n"
            f"{deterministic_findings.strip()}"
        )
    if narrative_voice_context and narrative_voice_context.strip():
        user_sections.append(
            "# ESTABLISHED NARRATOR VOICE (EVIDENCE-BACKED HISTORICAL TIMELINE)\n"
            + narrative_voice_context.strip()
        )
    user_sections.append(
        "# RELEVANT CONTEXT (use only when the source is implicit):\n"
        + (novel_context.strip() if novel_context.strip() else "None")
    )
    user_sections.append(
        "# NUMBERED DRAFT TRANSLATION TO AUDIT:\n"
        + format_editor_segments(draft_translation)
    )
    user_sections.append("Perform your rigorous Senior Editor audit now:")

    return PromptPair(system=system_prompt.strip(), user="\n\n".join(user_sections))


def _build_unit_json_contract_section(*, native_schema: bool = False) -> str:
    """Return the unit-rewrite Senior Editor output contract."""
    wrapper = (
        "Return exactly one JSON object."
        if native_schema else
        f"Return exactly one JSON object inside {REFLECTION_JSON_TAG_IN} and "
        f"{REFLECTION_JSON_TAG_OUT}."
    )
    example = "" if native_schema else f"""
{REFLECTION_JSON_TAG_IN}
{{"status":"no_issues","issues":[],"voice_observations":[]}}
{REFLECTION_JSON_TAG_OUT}"""
    return f"""{_AUDIT_CHECKLIST}

STRICT OUTPUT CONTRACT:
- {wrapper} No prose or markdown.
- Return at most 12 material issues, ordered by severity and confidence.
- Use no_issues only when no repair is needed; otherwise use needs_repair.
- category must name the checklist item the issue came from.
- unit_id is one of the ids above, copied exactly. It is the only way to say
  where a defect is: never quote the draft and never describe a position.
- rewritten_unit is the complete DRAFT text of that unit as it should read
  after the repair. Copy every part you are not repairing character for
  character, including punctuation, spacing and any markers.
- Repair only the defect you reported. A rewrite that also rephrases wording no
  issue was raised about is rejected whole, so keep the edit as small as the
  defect. Preferences and equally valid alternatives are not defects.
- One issue per unit. Report the whole repair of a unit as one issue.
- A unit whose DRAFT reads (missing) has nothing to rewrite; report it with an
  empty rewritten_unit and it will be raised for review.
- severity says how much the defect costs the reader, not how sure you are.
  Confidence below 0.80 is advisory and will not be applied.
- glossary_update is only for durable terminology; otherwise use null.
- voice_observations are evidence, not edits. Use exact source/target spans,
  keep discourse modes separate, and return [] when evidence is ambiguous.
- Issue fields are exactly: issue_id, unit_id, category, severity, confidence,
  instruction, rewritten_unit, glossary_update. Do not rename or omit them.
- Follow the supplied response schema exactly.{example}"""


def generate_unit_reflection_prompt(
    source_chunk: str,
    draft_translation: str,
    target_language: Optional[str] = None,
    novel_context: str = "",
    custom_instructions: str = "",
    glossary_block: str = "",
    deterministic_findings: str = "",
    source_available: bool = True,
    native_schema: bool = False,
    narrative_voice_context: str = "",
) -> PromptPair:
    """Generate the Senior Editor prompt for the unit-rewrite contract.

    The source and the draft arrive as one aligned bitext rather than two walls
    of text, so the editor is not also being asked to work out which paragraph
    corresponds to which before it can judge anything.

    This buys no tokens back: measured over 28 chunks of one real book the unit
    prompt is 0.9% larger than the span prompt, because the same two texts are
    being sent either way and the ids and labels are added on top. What it buys
    is that the alignment is done once, correctly, instead of being redone in
    the model's head for every finding.
    """

    from src.core.editor.units import align_units, format_bitext

    target_lang = target_language or "target language"
    units = align_units(source_chunk if source_available else "", draft_translation)
    source_mode = (
        "Perform bilingual source-fidelity and target-language quality review."
        if source_available
        else "Perform monolingual target-language style, fluency, consistency, glossary, relationship, and formatting review. Do not claim omissions, mistranslations, or source residue because no aligned source is available."
    )
    system_prompt = f"""You are an evidence-driven literary translation editor for {target_lang}.
{source_mode}

AUDIT POLICY:
- Each unit holds one source passage and the draft produced for it. Judge every
  unit for completeness, meaning, terminology, proper names, grammar, register,
  and structure.
- Report only material defects supported by the unit's own evidence.
- Explicit source names, nicknames, titles, honorifics, stuttering, and
  intentional register changes override background defaults.
- Use context only when the source is implicit. Never invent relationships,
  gender, hierarchy, or reverse addressing from absent evidence.
- Keep narration, dialogue, thought, letters, and embedded stories separate.
  Pair-specific dialogue forms never define narrator voice.
- Apply an established narrator policy across matching narrative scope and
  chapters; do not extend it into dialogue or a different narrator.
- A gendered vocative describes its addressee, not automatically its speaker.
- Preserve concise identity-bearing names as complete spans. Do not classify or
  translate a token inside a protected name independently.
- Prefer no issue over speculative polishing. Preserve all valid wording.

{_build_unit_json_contract_section(native_schema=native_schema)}"""

    user_sections = []
    if custom_instructions and custom_instructions.strip():
        user_sections.append(f"# CUSTOM INSTRUCTIONS & STYLE GUIDELINES:\n{custom_instructions.strip()}")
    if glossary_block and glossary_block.strip():
        user_sections.append(f"# GLOSSARY & TERM MAPPING:\n{glossary_block.strip()}")
    if deterministic_findings and deterministic_findings.strip():
        user_sections.append(
            "# DETERMINISTIC FINAL-OUTPUT FINDINGS (MANDATORY)\n"
            "These findings were produced by a non-LLM validator. Resolve each "
            "one in the unit that contains it, and report it under that unit's "
            "id:\n"
            f"{deterministic_findings.strip()}"
        )
    if narrative_voice_context and narrative_voice_context.strip():
        user_sections.append(
            "# ESTABLISHED NARRATOR VOICE (EVIDENCE-BACKED HISTORICAL TIMELINE)\n"
            + narrative_voice_context.strip()
        )
    user_sections.append(
        "# RELEVANT CONTEXT (use only when the source is implicit):\n"
        + (novel_context.strip() if novel_context.strip() else "None")
    )
    user_sections.append(
        "# ALIGNED UNITS TO AUDIT:\n"
        + format_bitext(units, source_available=source_available)
    )
    user_sections.append("Perform your rigorous Senior Editor audit now:")

    return PromptPair(system=system_prompt.strip(), user="\n\n".join(user_sections))


def generate_chunk_repair_prompt(
    source_chunk: str,
    draft_translation: str,
    critique_feedback: str,
    target_language: Optional[str] = None,
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    custom_instructions: str = "",
    glossary_block: str = "",
    novel_context: str = "",
    source_available: bool = True,
    narrative_voice_context: str = "",
) -> PromptPair:
    """
    Generate a chunk repair prompt applying critique feedback to produce the final pristine chunk.
    """
    target_lang = target_language or "target language"
    system_prompt = f"""You are an Expert Translation Refiner for {target_lang}.
Apply the Senior Editor's structured critique feedback to repair the draft translation.
Fix only flagged actionable issues: pronoun leaks, missing lines, mistranslations, glossary errors, gender mismatches, register flaws, placeholder issues, or format issues.
Preserve unflagged wording wherever possible.
Preserve every tag, placeholder, subtitle marker, line marker, and structural marker exactly unless the critique explicitly identifies it as wrong.
Output ONLY the final repaired translation text inside {translate_tag_in} and {translate_tag_out} tags."""

    user_sections = []
    if source_available:
        user_sections.append(f"# ORIGINAL SOURCE CHUNK:\n{source_chunk.strip()}")
    else:
        user_sections.append(
            "# SOURCE CAPABILITY:\nMonolingual repair only; preserve meaning and do not invent source-fidelity corrections."
        )
    user_sections.append(f"# DRAFT TRANSLATION:\n{draft_translation.strip()}")
    if custom_instructions and custom_instructions.strip():
        user_sections.append(f"# CUSTOM INSTRUCTIONS & STYLE GUIDELINES:\n{custom_instructions.strip()}")
    if glossary_block and glossary_block.strip():
        user_sections.append(f"# GLOSSARY & TERM MAPPING:\n{glossary_block.strip()}")
    if narrative_voice_context and narrative_voice_context.strip():
        user_sections.append(
            "# ESTABLISHED NARRATOR VOICE (EVIDENCE-BACKED HISTORICAL TIMELINE)\n"
            + narrative_voice_context.strip()
        )
    user_sections.append(f"# ACTIVE NOVEL LORE & ADDRESSING RULES:\n{novel_context.strip() if novel_context.strip() else 'None'}")
    user_sections.append(f"# SENIOR EDITOR CRITIQUE & REQUIRED FIXES:\n{critique_feedback.strip()}")
    user_sections.append(f"Output your pristine repaired translation in {translate_tag_in}...{translate_tag_out} now:")

    return PromptPair(system=system_prompt.strip(), user="\n\n".join(user_sections))
