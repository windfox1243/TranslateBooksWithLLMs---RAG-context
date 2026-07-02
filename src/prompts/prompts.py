from typing import List, NamedTuple, Tuple, Optional

from src.prompts.examples import (build_placeholder_section,
                              get_output_format_example, get_subtitle_example,
                              TAG0)
from src.config import (INPUT_TAG_IN, INPUT_TAG_OUT, TRANSLATE_TAG_IN,
                        TRANSLATE_TAG_OUT, PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX,
                        create_placeholder)

# Tags for placeholder correction responses
CORRECTED_TAG_IN = "<CORRECTED_TAG_IN>"
CORRECTED_TAG_OUT = "<CORRECTED_TAG_OUT>"


class PromptPair(NamedTuple):
    """A pair of system and user prompts for LLM translation."""
    system: str
    user: str


# ============================================================================
# SHARED PROMPT SECTIONS
# ============================================================================

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
    target = (target_language or "").strip().casefold()
    if target not in {"vietnamese", "tiếng việt", "tieng viet", "vi"}:
        return ""
    return """
# VIETNAMESE STYLE GUARDRAILS

- Maintain consistent Vietnamese pronouns and register across adjacent paragraphs.
- For serious literary first-person narration, prefer "tôi" unless the source or established character voice clearly requires an intimate/casual "mình".
- Do not switch the same narrator from "tôi" to "mình" within the same scene or reflective passage unless the relationship/register intentionally changes.
- Treat `## CURRENT ADDRESSING FORMS` as authoritative for direct address. If it gives a target-language form that is a name or title, use that form; do not replace it with Vietnamese kinship pronouns such as "anh", "chị", or "em".
- Do not infer "anh", "chị", or "em" from gender, status, affection, or politeness alone. Use them only when the source/context establishes the age, kinship, or seniority relationship, or when the addressing entry explicitly requires that form.
- Apply age, family, school-year, seniority, and relationship facts to both direct address and indirect references in dialogue, thoughts, and narration. If the viewpoint speaker/thinker is older or senior to the referenced character, or the referenced character is a same-age peer of the speaker's younger sibling, do not call or refer to that character as "anh" or "chị"; use the stored name/title or a neutral peer form instead.
- Preserve established addressing forms from the glossary, dialogue context, and previous paragraph."""


def _build_relationship_addressing_section() -> str:
    """Return language-neutral guardrails for social address consistency."""
    return """
# RELATIONSHIP AND ADDRESSING GUARDRAILS

- Treat `## CURRENT ADDRESSING FORMS` as authoritative for direct address when a matching speaker/addressee pair is present.
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

    novel_context = prompt_options.get('novel_context')
    if not novel_context or not str(novel_context).strip():
        return ""

    from src.utils.novel_context import render_novel_context_for_prompt

    rendered_context = render_novel_context_for_prompt(
        novel_context,
        reference_text=reference_text,
        max_tokens=prompt_options.get('novel_context_prompt_max_tokens'),
        selective=prompt_options.get('novel_context_selective_injection', True),
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


# ============================================================================
# TRANSLATION PROMPT FUNCTIONS
# ============================================================================

def generate_translation_prompt(
    main_content: str,
    context_before: str,
    context_after: str,
    previous_translation_context: str,
    source_language: str = "English",
    target_language: str = "English",
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
    target_lang_lower = target_language.lower()
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

    # Build custom instructions section
    custom_instructions_section = ""
    if custom_instructions and custom_instructions.strip():
        custom_instructions_section = f"""# ⚠️ MANDATORY STYLE INSTRUCTIONS - ABSOLUTE PRIORITY ⚠️

**These instructions override ALL other guidelines. Non-compliance = FAILURE.**

{custom_instructions.strip()}

⚠️ Apply to EVERY word you translate. Zero exceptions. ⚠️

"""

    # SYSTEM PROMPT - Role and instructions (stable across requests)
    system_prompt = f"""You are a professional {target_language} translator and writer.

{custom_instructions_section}# TRANSLATION PRINCIPLES

Translate {source_language} to {target_language}. Output only the translation.

**PRIORITY ORDER:**
1. Preserve exact names
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
    novel_context_section = _build_novel_context_section(
        prompt_options,
        reference_text="\n".join(
            part for part in (
                context_before,
                main_content,
                context_after,
                previous_translation_context,
            )
            if part
        ),
    )
    if novel_context_section:
        novel_context_section = f"{novel_context_section}\n\n"
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""
    dialogue_section = _build_dialogue_attribution_section(prompt_options)
    if dialogue_section:
        dialogue_section = f"{dialogue_section}\n\n"

    user_prompt = f"""{previous_translation_block_text}{novel_context_section}{glossary_section}{dialogue_section}# TEXT TO TRANSLATE

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
    source_language: str = "Chinese",
    target_language: str = "English",
) -> PromptPair:
    """
    Build a prompt that asks the LLM to extract recurring proper-noun entities
    (characters, locations, organizations/sects, items) from a sample of source
    text, along with a suggested target-language translation for each.

    Output is wrapped in <NER_JSON>...</NER_JSON> with a strict schema. The
    parser is permissive (handles markdown fences, missing tags, partial JSON).
    """
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

    user_prompt = f"""# SOURCE TEXT ({source_language})

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
    target_language: str = "English",
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
- A amator, literal, or awkward {target_language} translation
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

You are NOT translating - you are REWRITING in {target_language.upper()}.
The input is already in {target_language}, but poorly written.
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
            )
            if part
        ),
        refinement=True,
    )
    if novel_context_section:
        novel_context_section = f"{novel_context_section}\n\n"
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""
    dialogue_section = _build_dialogue_attribution_section(prompt_options)
    if dialogue_section:
        dialogue_section = f"{dialogue_section}\n\n"

    user_prompt = f"""{previous_context_block}{novel_context_section}{glossary_section}{dialogue_section}# DRAFT TO REFINE

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
    target_language: str = "English",
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

    novel_context_section = _build_novel_context_section(
        prompt_options,
        reference_text="\n".join(
            part for part in (
                previous_refined_block,
                formatted_subtitles_text,
            )
            if part
        ),
        refinement=True,
    )
    if novel_context_section:
        novel_context_section = f"{novel_context_section}\n\n"
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""
    dialogue_section = _build_dialogue_attribution_section(prompt_options)
    if dialogue_section:
        dialogue_section = f"{dialogue_section}\n\n"

    user_prompt = f"""{previous_refined_block_text}{novel_context_section}{glossary_section}{dialogue_section}# SUBTITLES TO REFINE

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
    source_language: str = "English",
    target_language: str = "English",
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

    # Build custom instructions section if provided
    custom_instructions_section = ""
    if custom_instructions and custom_instructions.strip():
        custom_instructions_section = f"""

# ⚠️ MANDATORY STYLE INSTRUCTIONS - ABSOLUTE PRIORITY ⚠️

**These instructions override ALL other guidelines. Non-compliance = FAILURE.**

{custom_instructions.strip()}

⚠️ Apply to EVERY subtitle. Zero exceptions. ⚠️
"""

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
- Preserve inline formatting tags (<i>, <b>, <font ...>, {{\\an8}}, etc.) and any \\n line breaks inside a subtitle{custom_instructions_section}

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

    # Novel context and glossary blocks live in the user prompt because they
    # vary per chunk/block.
    novel_context_section = _build_novel_context_section(
        prompt_options,
        reference_text="\n".join(
            part for part in (
                previous_translation_block,
                formatted_subtitles_text,
            )
            if part
        ),
    )
    if novel_context_section:
        novel_context_section = f"{novel_context_section}\n\n"
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""
    dialogue_section = _build_dialogue_attribution_section(prompt_options)
    if dialogue_section:
        dialogue_section = f"{dialogue_section}\n\n"

    user_prompt = f"""{previous_translation_block_text}{novel_context_section}{glossary_section}{dialogue_section}# SUBTITLES TO TRANSLATE

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

def generate_placeholder_correction_prompt(
    original_text: str,
    translated_text: str,
    specific_errors: str,
    source_language: str,
    target_language: str,
    expected_count: int,
    placeholder_format: Optional[Tuple[str, str]] = None
) -> PromptPair:
    """
    Generate a prompt for correcting placeholder errors in a translation.

    This prompt is used when a translation has placeholder issues (missing,
    duplicated, mutated, or out of order). It asks the LLM to fix ONLY the
    placeholder positions without modifying the translated text.

    Args:
        original_text: Source text with correct placeholders
        translated_text: Translation with placeholder errors
        specific_errors: Detailed error description (generated by build_specific_error_details)
        source_language: Source language name (e.g., "English")
        target_language: Target language name (e.g., "French")
        expected_count: Number of placeholders expected (0 to expected_count-1)
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format.
            If None, uses default [[0]] format

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    # Use custom format if provided, otherwise use defaults
    if placeholder_format:
        prefix, suffix = placeholder_format
    else:
        prefix, suffix = PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX

    # Generate dynamic placeholder examples using the correct format
    def make_placeholder(idx: int) -> str:
        return f"{prefix}{idx}{suffix}"

    max_index = expected_count - 1 if expected_count > 0 else 0
    placeholder_format_str = f"{prefix}N{suffix}"
    example_range = f"{make_placeholder(0)} to {make_placeholder(max_index)}"
    placeholder_list = ", ".join(make_placeholder(i) for i in range(min(3, expected_count)))
    if expected_count > 3:
        placeholder_list += ", etc."

    # SYSTEM PROMPT
    system_prompt = f"""You are a technical placeholder correction specialist.

## YOUR TASK

A {source_language} to {target_language} translation was performed, but the placeholders were corrupted.
You must fix the placeholder positions to match the original text structure.

## PLACEHOLDER FORMAT

**CORRECT format:** {make_placeholder(0)}, {make_placeholder(1)}, {make_placeholder(2)}, etc.
- Brackets: {prefix} and {suffix}
- Sequential numbering starting from 0
- Expected range for this text: {example_range}

**FORMAT VARIATIONS:**
The system uses different placeholder formats based on text content:
- [id0], [id1], [id2]... (default - semantic markers, highest accuracy)
- /0, /1, /2... (when text contains brackets)
- $0$, $1$, $2$... (when text contains brackets and slashes)
- [[0]], [[1]], [[2]]... (legacy format)

All formats follow the same rules: preserve exact format, maintain sequential order, keep position.

## HOW TO POSITION PLACEHOLDERS

Placeholders represent HTML/XML tags. To position them correctly:

1. **Look at the ORIGINAL text** to see what content each placeholder surrounds
2. **Find the equivalent content** in the translation
3. **Place the placeholder at the same logical position** around that content

**Example:**
- Original: "{make_placeholder(0)}Hello{make_placeholder(1)} world"
- If translation is "Bonjour monde", the placeholders mark "Hello"
- Correct: "{make_placeholder(0)}Bonjour{make_placeholder(1)} monde"

## VALIDATION RULES

1. **EXACT COUNT**: Must contain exactly {expected_count} placeholders
2. **SEQUENTIAL ORDER**: Placeholders must appear in order: {placeholder_list}
3. **NO DUPLICATES**: Each placeholder must appear exactly once
4. **NO MUTATIONS**: Use ONLY the {placeholder_format_str} format
5. **POSITION MATCHING**: Each placeholder must surround the translated equivalent of what it surrounded in the original

## CRITICAL INSTRUCTIONS

- Analyze the ORIGINAL to understand what each placeholder marks
- Position placeholders around the SAME semantic content in the translation
- Do NOT add or remove words from the translation
- Keep the {target_language} text intact, only fix placeholder positions

## OUTPUT FORMAT

Your response MUST start with {CORRECTED_TAG_IN} and end with {CORRECTED_TAG_OUT}.
Include NOTHING before or after these tags."""

    # USER PROMPT
    user_prompt = f"""## ORIGINAL TEXT ({source_language}) - Reference for placeholder positions:

<ORIGINAL_TAG_IN>
{original_text}
<ORIGINAL_TAG_OUT>

## TRANSLATION WITH ERRORS ({target_language}):

<TRANSLATION_TAG_IN>
{translated_text}
<TRANSLATION_TAG_OUT>

## DETECTED ERRORS:

{specific_errors}

## YOUR TASK:

Reposition the placeholders {example_range} in the translation above.
Keep the translated text unchanged - only fix placeholder positions.

Provide your corrected version now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


# ============================================================================
# ALIAS FOR BACKWARDS COMPATIBILITY
# ============================================================================

def generate_post_processing_prompt(
    translated_text: str,
    target_language: str = "English",
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
