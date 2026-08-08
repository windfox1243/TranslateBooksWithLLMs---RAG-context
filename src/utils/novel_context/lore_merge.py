"""Prompts and merge logic for folding an LLM context update into lore."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .characters import (
    _add_glossary_character_aliases,
    _alias_entries_to_map,
    _canonical_alias_entries,
    _canonical_display_name,
    _character_alias_keys,
    _character_identities_match,
    _character_names_match,
    _deduplicate_character_entries,
    _find_lore_section,
    _format_character_line,
    _identity_endpoint_non_character_reason,
    _infer_unique_short_name_alias_entries,
    _is_descriptive_role_name,
    _is_disposable_unnamed_character,
    _is_invalid_context_key,
    _is_non_character_group_entry,
    _is_non_character_metadata_or_item_entry,
    _is_non_character_work_entry,
    _is_quarantined_character_entry,
    _is_unstable_identity_alias,
    _is_unstable_physical_character_entry,
    _merge_character_values,
    _normalize_character_value,
    _parse_alias_entries,
    _parse_bullet_entries,
    _plain_key,
    _preferred_character_name,
    _replace_lore_section,
    _retain_renderable_aliases,
    _strip_balanced_brackets,
    _strip_character_correction_marker,
)
from .constants import ALIASES_SECTION, CHARACTERS_SECTION, GLOSSARY_SECTION, logger
from .glossary import (
    _is_inverted_target_to_source_glossary_pair,
    _normalize_glossary_entries,
    normalize_global_lore,
)
from .identity_links import (
    _candidate_named_characters,
    _gate_unproven_character_gender,
    _source_identity_link_proof_status,
)

UPDATE_SYSTEM_PROMPT = """You are an expert novel translation context assistant.
Your task is to analyze the latest source text and its translation, and detect any new characters, glossary terms, or relationship addressing changes.

Identity rules:
- Reuse the exact canonical name already present in CURRENT GLOBAL LORE.
- A title, rank, nickname, transformed state, awakened state, disguise, age qualifier, or relationship label is not a new character when it refers to an existing person. Update the existing canonical entry instead.
- A descriptor-only label such as "Protagonist", "Hero", "Main Character", "Player Character", "Protagonist of X", "Hero of X", "fictional character", or "character from X" is not a canonical character name or identity link. Use source names such as Kim Ji-an, Valentine, or Eric; otherwise omit it.
- When a title-only entry is later identified by name (for example, "Emperor" = "Serena Augusta"), output only the named canonical character with the title in its concise description.
- When the latest source directly proves that a stable, book-wide title, rank, nickname, or other label is an existing character, record that mapping under IDENTITY_LINKS. Valid proof includes explicit naming, apposition, an identity reveal, or unambiguous same-scene coreference such as a direct address immediately attributed to the named character. Never create an identity link from role similarity alone. Do not persist a bare title that can refer to multiple people or transfer between characters; use the canonical name directly for that scene instead.
- **SIMILAR NAMES WARNING (CRITICAL):** Do not merge, conflate, or link similar-looking character names (e.g. Alex vs. Alles) under NEW_CHARACTERS, IDENTITY_LINKS, or DYNAMIC_STATE. Treat them as completely separate individuals unless there is absolute, direct, source-proven evidence that they are the same person. If the glossary or instructions direct you to map a specific source name to a canonical name (e.g. source 'Alex' -> canonical 'Alles'), apply this mapping ONLY to that specific character, and never to another character who is actually named Alex.
- If the source links a role/title to a named person by location or narration (for example, "the Lieutenant Colonel's office" followed by "Eric" as the person in that office), record the role/title under IDENTITY_LINKS instead of creating a separate character.
- For non-English source titles or aliases, preserve the exact source surface label under IDENTITY_LINKS when it is source-proven (for example, "- 중령: Eric"). If the English normalized title also appears in the model's character summary, link that title too.
- For Chinese/Japanese/Korean source names, do not invent new romanized character names such as pinyin, romaji, or revised romanization variants. Reuse an existing canonical romanized name; otherwise record the exact source surface name under NEW_CHARACTERS and put romanization recommendations only under NEW_GLOSSARY.
- When adding or correcting a Chinese/Japanese/Korean source-script character name, also add a NEW_GLOSSARY entry that maps the exact source name to the recommended target-language name rendering. If a short source name or honorific address form is durable, add that exact source form too with the appropriate short or honorific target rendering.
- When the target language is Vietnamese and an English named skill, ability, technique, spell, combat move, weapon, artifact, or equipment needs glossary tracking, prefer a concise Sino-Vietnamese literary target term if it sounds natural in Vietnamese fantasy or game prose. Preserve English only for brands, code labels, UI/system keys, or terms that clearly should remain untranslated.
- Do not add one-scene unnamed soldiers, victims, hallucinations, generic crowds, or incidental job labels unless they recur and their identity/gender is required for translation consistency.
- Do not add numbered/background casualties or generic staff labels such as "Wounded Soldier 1", "Guard 2", "Doctor", or "Private" unless the person is source-named, recurring, or needed for a durable addressing/relationship choice.
- Do not add bare romantic or family relationship labels such as "Lover", "Girlfriend", "Ex-girlfriend", "Boyfriend", "Partner", "Spouse", "Wife", or "Husband" as characters or identity links. If the relationship partner is not source-named, record only the relationship, not a fake identity alias.
- Never output template entries, "None", "[None]", "Unknown", "N/A", or an empty bullet.
- Record gender only when the source states it or supplies unambiguous grammatical/pronoun evidence. Never guess gender from a name, occupation, rank, appearance, genre convention, or stereotype.
- Gender-neutral words such as spouse, partner, lover, parent, child, sibling, officer, captain, commander, major, colonel, and lieutenant colonel never prove gender by themselves.
- A named character having a girlfriend, boyfriend, lover, spouse, or ex-partner does not by itself prove that named character's gender; use pronouns or grammatical evidence such as "him/his" or "her" instead.
- If a character reincarnates, transforms, disguises themselves, or receives a new body, record the gender of the current named form, not the previous body. Keep the previous identity only as a concise description.
- Pronoun evidence attached to another character's relationship with a named person is evidence for that named person. For example, "suspicious of her identity" after naming Valentine proves Valentine is Female, not the suspicious officer.
- If CURRENT GLOBAL LORE has the wrong gender for the current named form and the latest source proves the correction, output that canonical character under NEW_CHARACTERS as "CORRECTION: [Gender, concise role, concise description]". Do not preserve stale gender just because it is already stored.
- Before writing "Unspecified", scan the whole latest source for direct evidence such as gendered nouns, pronouns, kinship grammar, or an explicit description. If an existing Unspecified character is now proven Male/Female, output that specific gender directly; this is not a correction.
- An existing specific gender is authoritative. Change it only when the latest source explicitly proves it was wrong; write that rare update as "CORRECTION: [Gender, role, description]".
- Write all character metadata in English, regardless of source and target language.
- For an existing character, output one concise cumulative replacement description containing the important old and new facts. Summarize repeated roles instead of appending duplicate phrases.
- Character descriptions must contain only the normalized result. Never append evidence notes, quotations, reasoning, confidence, "source pronoun evidence", "reincarnated current form", "Gender confirmed...", "Correction...", parenthetical explanations, or prompt/control labels such as "current rank and title" or "title/nickname for X".

Input provided:
1. CURRENT GLOBAL LORE (Characters & Glossary)
2. CURRENT DYNAMIC RELATIONSHIP STATE
3. RECENT SOURCE MEMORY (bounded previous chunks)
4. LATEST SOURCE TEXT
5. LATEST TRANSLATION

Your output must follow this strict format:

[NEW_CHARACTERS]
- Canonical Name: [Gender, role, and concise description]
(Use "Unspecified" rather than guessing when a recurring character must be tracked before gender is explicit. Use "CORRECTION: [Gender, role, description]" only for a source-proven correction. Use "- Canonical Name: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[IDENTITY_LINKS]
- Source title, rank, nickname, or alias: Canonical Name
(Only include identity links directly established by the latest source. The right side must be one exact canonical character name from CURRENT GLOBAL LORE or NEW_CHARACTERS. Use "- Alias: DELETE" to remove a wrong link. If there are no changes, output no bullet under this header.)

[NEW_GLOSSARY]
- Source Term: [Target Term]
(Only include actual additions or corrections. Use "- Source Term: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[DYNAMIC_STATE]
# DYNAMIC RELATIONSHIP STATE
## CURRENT ADDRESSING FORMS
- Speaker → Addressee: source form "..." | target-language form "..." | register, social basis, scope, and reason
## RELATIONSHIP EVOLUTION
- Character A ↔ Character B: concise current relationship
(Output both headings every time, but list only additions or changes. Omitted entries remain stored indefinitely. Remove an obsolete entry only with "- Speaker → Addressee: DELETE" or "- Character A ↔ Character B: DELETE". Addressing forms include names, titles, honorifics, pronouns, kinship terms, and formality choices needed in the target language. For Vietnamese, every addressing entry must make the target-language form a complete paired-address record with all applicable parts: `self-reference: ...; second-person pronoun: ...; vocative/address form: ...`. Use `none` only for a part that truly does not apply. Prefer a true Vietnamese paired-address choice in `self-reference` and `second-person pronoun` (e.g. tôi/cô, em/anh, chị/em, tớ/cậu, ta/ngươi) when relationship facts support one. Use a proper name or short title in `second-person pronoun` (e.g. Apollo, Tomio, Spe, huấn luyện viên, bác sĩ) only when no stable pronoun/kinship pair is known, or when the source consistently uses that name/title as the direct address; otherwise put names, titles, and longer calls in `vocative/address form`. Proven kinship, age, seniority, family, rank, or teacher/student hierarchy overrides name/title fallback. When a speaker naturally alternates or mixes address forms (e.g. calling someone both "anh" and "huấn luyện viên", or "cậu" and "Tomio"), list the acceptable variants separated by slashes only if both variants are actually supported by the source/context. Never put long descriptive background phrases or multi-word role explanations (e.g. "huấn luyện viên của Apollo Rainbow") in `second-person pronoun` - put long descriptive calls in `vocative/address form`. Do not store only the addressee nickname if the speaker's self-reference should also change. If age, school-year, seniority, family, rank, or teacher/student hierarchy is known, choose Vietnamese pronouns that reflect that hierarchy; do not use peer `tớ/cậu` for senior/junior or older/younger relationships. Do not mix modern polite/casual Vietnamese self-references such as `tôi`, `tớ`, `mình`, `em`, `anh`, or `chị` with the contemptuous/archaic second-person pronoun `ngươi`; for hostile archaic contempt, use a coherent pair such as `ta/ngươi`. For Vietnamese, the final reason field must explicitly state any known age, school-year, seniority, family, rank, or teacher/student hierarchy; do not summarize a hierarchical pair merely as peers/classmates. In the final reason field, record the social basis when known: direct address vs indirect reference scope, age/school-year/seniority, family relation, rank/status, setting, intimacy, hostility, deference, or exception to normal age hierarchy. Use plain Unicode arrows only. Never use LaTeX, backslashes, dollar signs, or ASCII arrows. Do not duplicate these headings.)

[RELATIONSHIP_CANDIDATES]
{"relationships":[{"source":"canonical character","target":"canonical character","relationship_type":"parent | child | mentor | student | superior | subordinate | master | servant | sibling | ally | rival | spouse | peer | friend | enemy | colleague | romantic_partner | family | associated","direction":"directed | symmetric","scope":"durable | situational","hierarchy":"source_senior | source_junior | peer | unknown","relative_age":"source_older | source_younger | same_age | unknown","rank_relation":"source_higher | source_lower | equal | unknown","intimacy":"brief label","register":"brief label","evidence_quote":"first exact quote from LATEST SOURCE TEXT","evidence_spans":[{"quote":"exact source span","role":"source | target | relationship","dialogue_turn_id":"optional turn id"}],"confidence":0.0,"source_entity_type":"character","target_entity_type":"character","dialogue_turn_id":"optional candidate id","details":"concise current fact"}]}
(Output valid JSON on one line. Include every new or changed relationship from DYNAMIC_STATE. Durable candidates require one or more exact source spans from one bounded scene. Every evidence_spans quote must appear verbatim in LATEST SOURCE TEXT; collectively they must identify both participants and support the direction. Never stitch separate lines into a fabricated quote. Use situational scope for disguise, acting, roleplay, dreams, or quoted history. Weapons, artifacts, items, skills, spells, abilities, UI labels, and system terms are not characters and must not receive relationship edges. Return {"relationships":[]} when there are no relationship candidates.)

[DIALOGUE_ATTRIBUTION]
{"turns":[{"id":"exact candidate id","speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown","confidence":0.0}],"state_after":{"speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown"}}
(Classify only the supplied dialogue candidates. Infer from narration, direct dialogue tags, turn-taking, voice, and addressing forms in the latest source. CURRENT SCENE SPEAKER STATE is a weak continuity hint only; never use it as sole proof, and return Unknown when local evidence is unclear. Resolve titles and aliases through CURRENT GLOBAL LORE and IDENTITY_LINKS, but output only canonical character names already present in CURRENT GLOBAL LORE or NEW_CHARACTERS. Never invent a speaker. Confidence is from 0.0 to 1.0. Return {"turns":[],"state_after":{}} when there are no candidates or no current high-confidence speaker.)

Do not include any other explanations, markdown fences, or extra text outside these blocks.
"""
SOURCE_ANALYSIS_SYSTEM_PROMPT = """You are an expert novel translation context assistant.
Analyze the latest SOURCE text before it is translated. Detect new or corrected characters, source-proven genders and roles, source terminology that needs a consistent target-language rendering, and relationship/addressing changes that the translator must know now.

Identity rules:
- Reuse the exact canonical name already present in CURRENT GLOBAL LORE.
- Do not create separate characters for ranks, titles, nicknames, transformed/awakened states, disguises, age variants, or relational aliases of an existing person.
- A descriptor-only label such as "Protagonist", "Hero", "Main Character", "Player Character", "Protagonist of X", "Hero of X", "fictional character", or "character from X" is not a canonical character name or identity link. Use source names such as Kim Ji-an, Valentine, or Eric; otherwise omit it.
- When a title-only entry is later identified by name (for example, "Emperor" = "Serena Augusta"), output only the named canonical character with the title in its concise description.
- When this source directly proves that a stable, book-wide title, rank, nickname, or other label is an existing character, record that mapping under IDENTITY_LINKS. Valid proof includes explicit naming, apposition, an identity reveal, or unambiguous same-scene coreference such as a direct address immediately attributed to the named character. Never create an identity link from role similarity alone. Do not persist a bare title that can refer to multiple people or transfer between characters; use the canonical name directly for that scene instead.
- **SIMILAR NAMES WARNING (CRITICAL):** Do not merge, conflate, or link similar-looking character names (e.g. Alex vs. Alles) under NEW_CHARACTERS, IDENTITY_LINKS, or DYNAMIC_STATE. Treat them as completely separate individuals unless there is absolute, direct, source-proven evidence that they are the same person. If the glossary or instructions direct you to map a specific source name to a canonical name (e.g. source 'Alex' -> canonical 'Alles'), apply this mapping ONLY to that specific character, and never to another character who is actually named Alex.
- If the source links a role/title to a named person by location or narration (for example, "the Lieutenant Colonel's office" followed by "Eric" as the person in that office), record the role/title under IDENTITY_LINKS instead of creating a separate character.
- For non-English source titles or aliases, preserve the exact source surface label under IDENTITY_LINKS when it is source-proven (for example, "- 중령: Eric"). If the English normalized title also appears in the model's character summary, link that title too.
- For Chinese/Japanese/Korean source names, do not invent new romanized character names such as pinyin, romaji, or revised romanization variants. Reuse an existing canonical romanized name; otherwise record the exact source surface name under NEW_CHARACTERS and put romanization recommendations only under NEW_GLOSSARY.
- When adding or correcting a Chinese/Japanese/Korean source-script character name, also add a NEW_GLOSSARY entry that maps the exact source name to the recommended target-language name rendering. If a short source name or honorific address form is durable, add that exact source form too with the appropriate short or honorific target rendering.
- When the target language is Vietnamese and an English named skill, ability, technique, spell, combat move, weapon, artifact, or equipment needs glossary tracking, prefer a concise Sino-Vietnamese literary target term if it sounds natural in Vietnamese fantasy or game prose. Preserve English only for brands, code labels, UI/system keys, or terms that clearly should remain untranslated.
- Do not add one-scene unnamed soldiers, victims, generic crowds, or incidental roles unless they recur and are necessary for pronoun/address consistency.
- Do not add numbered/background casualties or generic staff labels such as "Wounded Soldier 1", "Guard 2", "Doctor", or "Private" unless the person is source-named, recurring, or needed for a durable addressing/relationship choice.
- Do not add bare romantic or family relationship labels such as "Lover", "Girlfriend", "Ex-girlfriend", "Boyfriend", "Partner", "Spouse", "Wife", or "Husband" as characters or identity links. If the relationship partner is not source-named, record only the relationship, not a fake identity alias.
- Never output template entries, "None", "[None]", "Unknown", "N/A", or an empty bullet.
- Record gender only when this source text states it or gives unambiguous grammatical/pronoun evidence. Never infer it from names, jobs, ranks, appearance, personality, or genre stereotypes.
- Gender-neutral words such as spouse, partner, lover, parent, child, sibling, officer, captain, commander, major, colonel, and lieutenant colonel never prove gender by themselves.
- A named character having a girlfriend, boyfriend, lover, spouse, or ex-partner does not by itself prove that named character's gender; use pronouns or grammatical evidence such as "him/his" or "her" instead.
- If a character reincarnates, transforms, disguises themselves, or receives a new body, record the gender of the current named form, not the previous body. Keep the previous identity only as a concise description.
- Pronoun evidence attached to another character's relationship with a named person is evidence for that named person. For example, "suspicious of her identity" after naming Valentine proves Valentine is Female, not the suspicious officer.
- If CURRENT GLOBAL LORE has the wrong gender for the current named form and this source proves the correction, output that canonical character under NEW_CHARACTERS as "CORRECTION: [Gender, concise role, concise description]". Do not preserve stale gender just because it is already stored.
- Before writing "Unspecified", scan the entire latest source for direct evidence such as gendered nouns, pronouns, kinship grammar, or an explicit description. If an existing Unspecified character is now proven Male/Female, output that specific gender directly; this is not a correction.
- Treat an existing specific gender as authoritative. Change it only when this source text explicitly proves it wrong, using "CORRECTION: [Gender, role, description]".
- Preserve source-side proper names exactly. Write character metadata descriptions in English so context remains stable when the translation model or target language changes.
- For an existing character, output one concise cumulative replacement description containing the important old and new facts. Summarize repeated roles instead of appending duplicate phrases.
- Character descriptions must contain only the normalized result. Never append evidence notes, quotations, reasoning, confidence, "source pronoun evidence", "reincarnated current form", "Gender confirmed...", "Correction...", parenthetical explanations, or prompt/control labels such as "current rank and title" or "title/nickname for X".

Input provided:
1. CURRENT GLOBAL LORE (Characters & Glossary)
2. CURRENT DYNAMIC RELATIONSHIP STATE
3. RECENT SOURCE MEMORY (bounded previous chunks)
4. LATEST SOURCE TEXT

Your output must follow this strict format:

[NEW_CHARACTERS]
- Canonical Name: [Gender, role, and concise description]
(Use "Unspecified" rather than guessing when a recurring character must be tracked before gender is explicit. Use "CORRECTION: [Gender, role, description]" only for a source-proven correction. Use "- Canonical Name: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[IDENTITY_LINKS]
- Source title, rank, nickname, or alias: Canonical Name
(Only include identity links directly established by this source. The right side must be one exact canonical character name from CURRENT GLOBAL LORE or NEW_CHARACTERS. Use "- Alias: DELETE" to remove a wrong link. If there are no changes, output no bullet under this header.)

[NEW_GLOSSARY]
- Source Term: [Recommended Target Term]
(Only include important recurring terms. Use "- Source Term: DELETE" to delete an obsolete entry. If there are no changes, output no bullet under this header.)

[DYNAMIC_STATE]
# DYNAMIC RELATIONSHIP STATE
## CURRENT ADDRESSING FORMS
- Speaker → Addressee: source form "..." | recommended target-language form "..." | register, social basis, scope, and reason
## RELATIONSHIP EVOLUTION
- Character A ↔ Character B: concise current relationship
(Output both headings every time, but list only additions or changes. Omitted entries remain stored indefinitely. Remove an obsolete entry only with "- Speaker → Addressee: DELETE" or "- Character A ↔ Character B: DELETE". Addressing forms include names, titles, honorifics, pronouns, kinship terms, and formality choices needed for translation. For Vietnamese, every addressing entry must make the recommended target-language form a complete paired-address record with all applicable parts: `self-reference: ...; second-person pronoun: ...; vocative/address form: ...`. Use `none` only for a part that truly does not apply. Prefer a true Vietnamese paired-address choice in `self-reference` and `second-person pronoun` (e.g. tôi/cô, em/anh, chị/em, tớ/cậu, ta/ngươi) when relationship facts support one. Use a proper name or short title in `second-person pronoun` (e.g. Apollo, Tomio, Spe, huấn luyện viên, bác sĩ) only when no stable pronoun/kinship pair is known, or when the source consistently uses that name/title as the direct address; otherwise put names, titles, and longer calls in `vocative/address form`. Proven kinship, age, seniority, family, rank, or teacher/student hierarchy overrides name/title fallback. When a speaker naturally alternates or mixes address forms (e.g. calling someone both "anh" and "huấn luyện viên", or "cậu" and "Tomio"), list the acceptable variants separated by slashes only if both variants are actually supported by the source/context. Never put long descriptive background phrases or multi-word role explanations (e.g. "huấn luyện viên của Apollo Rainbow") in `second-person pronoun` - put long descriptive calls in `vocative/address form`. Do not store only the addressee nickname if the speaker's self-reference should also change. If age, school-year, seniority, family, rank, or teacher/student hierarchy is known, choose Vietnamese pronouns that reflect that hierarchy; do not use peer `tớ/cậu` for senior/junior or older/younger relationships. Do not mix modern polite/casual Vietnamese self-references such as `tôi`, `tớ`, `mình`, `em`, `anh`, or `chị` with the contemptuous/archaic second-person pronoun `ngươi`; for hostile archaic contempt, use a coherent pair such as `ta/ngươi`. For Vietnamese, the final reason field must explicitly state any known age, school-year, seniority, family, rank, or teacher/student hierarchy; do not summarize a hierarchical pair merely as peers/classmates. In the final reason field, record the social basis when known: direct address vs indirect reference scope, age/school-year/seniority, family relation, rank/status, setting, intimacy, hostility, deference, or exception to normal age hierarchy. Use plain Unicode arrows only. Never use LaTeX, backslashes, dollar signs, or ASCII arrows. Keep it concise and do not duplicate headings.)

[RELATIONSHIP_CANDIDATES]
{"relationships":[{"source":"canonical character","target":"canonical character","relationship_type":"parent | child | mentor | student | superior | subordinate | master | servant | sibling | ally | rival | spouse | peer | friend | enemy | colleague | romantic_partner | family | associated","direction":"directed | symmetric","scope":"durable | situational","hierarchy":"source_senior | source_junior | peer | unknown","relative_age":"source_older | source_younger | same_age | unknown","rank_relation":"source_higher | source_lower | equal | unknown","intimacy":"brief label","register":"brief label","evidence_quote":"first exact quote from LATEST SOURCE TEXT","evidence_spans":[{"quote":"exact source span","role":"source | target | relationship","dialogue_turn_id":"optional turn id"}],"confidence":0.0,"source_entity_type":"character","target_entity_type":"character","dialogue_turn_id":"optional candidate id","details":"concise current fact"}]}
(Output valid JSON on one line. Include every new or changed relationship from DYNAMIC_STATE. Durable candidates require one or more exact source spans from one bounded scene. Every evidence_spans quote must appear verbatim in LATEST SOURCE TEXT; collectively they must identify both participants and support the direction. Never stitch separate lines into a fabricated quote. Use situational scope for disguise, acting, roleplay, dreams, or quoted history. Weapons, artifacts, items, skills, spells, abilities, UI labels, and system terms are not characters and must not receive relationship edges. Return {"relationships":[]} when there are no relationship candidates.)

[DIALOGUE_ATTRIBUTION]
{"turns":[{"id":"exact candidate id","speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown","confidence":0.0}],"state_after":{"speaker":"canonical character name or Unknown","addressee":"canonical character name or Unknown"}}
(Classify only the supplied dialogue candidates. Infer from narration, direct dialogue tags, turn-taking, voice, and addressing forms in the latest source. CURRENT SCENE SPEAKER STATE is a weak continuity hint only; never use it as sole proof, and return Unknown when local evidence is unclear. Resolve titles and aliases through CURRENT GLOBAL LORE and IDENTITY_LINKS, but output only canonical character names already present in CURRENT GLOBAL LORE or NEW_CHARACTERS. Never invent a speaker. Confidence is from 0.0 to 1.0. Return {"turns":[],"state_after":{}} when there are no candidates or no current high-confidence speaker.)

Do not translate the whole passage. Do not include explanations, markdown fences, or text outside these blocks.
"""
UPDATE_USER_PROMPT_TEMPLATE = """### CURRENT GLOBAL LORE:
{current_global_lore}

### CURRENT DYNAMIC RELATIONSHIP STATE:
{current_dynamic_state}
{custom_instructions_section}{glossary_block_section}
### TRANSLATION PROGRESS: Segment {chunk_index} of {total_chunks}

### RECENT SOURCE MEMORY ({source_language}, previous chunks only):
{source_context}

### LATEST SOURCE TEXT ({source_language}):
{source_chunk}

### LATEST TRANSLATION ({target_language}):
{translated_chunk}

### CURRENT SCENE SPEAKER STATE:
{current_dialogue_state}

### DIALOGUE CANDIDATES:
{dialogue_candidates}

Output the updates now. Output ONLY the strictly formatted blocks."""
SOURCE_ANALYSIS_USER_PROMPT_TEMPLATE = """### CURRENT GLOBAL LORE:
{current_global_lore}

### CURRENT DYNAMIC RELATIONSHIP STATE:
{current_dynamic_state}
{custom_instructions_section}{glossary_block_section}
### TRANSLATION PROGRESS: Segment {chunk_index} of {total_chunks}

### RECENT SOURCE MEMORY ({source_language}, previous chunks only):
{source_context}

### LATEST SOURCE TEXT ({source_language}):
{source_chunk}

### TARGET LANGUAGE:
{target_language}

### CURRENT SCENE SPEAKER STATE:
{current_dialogue_state}

### DIALOGUE CANDIDATES:
{dialogue_candidates}

Analyze the source for context needed by its translation. Output ONLY the strictly formatted blocks."""
def merge_new_lore(
    global_lore: str,
    new_characters: str,
    new_glossary: str,
    new_aliases: str = "",
    source_text: str = "",
    trusted_aliases: str = "",
) -> Tuple[str, List[str]]:
    """Merge context updates through canonical character and glossary identities."""
    change_logs: List[str] = []
    raw_global_lore = str(global_lore or "")
    lore = normalize_global_lore(global_lore)

    def record(message: str) -> None:
        change_logs.append(message)

    alias_bounds = _find_lore_section(lore, ALIASES_SECTION)
    alias_entries = (
        _parse_alias_entries(lore[alias_bounds[1]:alias_bounds[2]])
        if alias_bounds
        else []
    )
    explicit_aliases = _alias_entries_to_map(alias_entries)
    alias_displays = {
        alias_key: alias
        for alias, _ in alias_entries
        for alias_key in _character_alias_keys(alias)
    }
    valid_character_targets = {
        _plain_key(name): _canonical_display_name(name)
        for name in _candidate_named_characters(lore, new_characters)
    }
    for alias_key, target in list(explicit_aliases.items()):
        if _plain_key(target) not in valid_character_targets:
            explicit_aliases.pop(alias_key, None)
            alias_displays.pop(alias_key, None)

    alias_updates = [
        (raw_alias, raw_target, True)
        for raw_alias, raw_target in _parse_bullet_entries(trusted_aliases)
    ] + [
        (raw_alias, raw_target, False)
        for raw_alias, raw_target in _parse_bullet_entries(new_aliases)
    ]

    for raw_alias, raw_target, is_trusted_alias in alias_updates:
        if (
            _is_invalid_context_key(raw_alias)
            or _is_unstable_identity_alias(raw_alias, allow_physical=True)
        ):
            continue
        alias_keys = _character_alias_keys(raw_alias)
        if not alias_keys:
            continue
        is_delete = _strip_balanced_brackets(raw_target).casefold() == "delete"
        if is_delete:
            removed = False
            for alias_key in alias_keys:
                if alias_key in explicit_aliases:
                    explicit_aliases.pop(alias_key, None)
                    alias_displays.pop(alias_key, None)
                    removed = True
            if removed:
                record(
                    "[Novel Context] Deleted identity link "
                    f"'{_plain_key(raw_alias)}'."
                )
            continue

        target = _canonical_display_name(raw_target)
        if (
            _is_invalid_context_key(target)
            or _character_names_match(raw_alias, target)
        ):
            continue
        if _plain_key(target) not in valid_character_targets:
            logger.warning(
                "[Novel Context] Skipped unsafe identity link '%s' -> '%s': "
                "target is not a canonical character.",
                _plain_key(raw_alias),
                target,
            )
            continue
        non_character_reason = _identity_endpoint_non_character_reason(
            raw_alias,
            lore,
            new_characters,
            new_glossary,
        )
        if non_character_reason:
            logger.warning(
                "[Novel Context] Skipped unsafe identity link '%s' -> '%s': %s.",
                _plain_key(raw_alias),
                target,
                non_character_reason,
            )
            continue
        already_linked = any(
            explicit_aliases.get(alias_key) == target
            for alias_key in alias_keys
        )
        identity_link_proved = True
        identity_link_skip_reason = ""
        if source_text and not is_trusted_alias and not already_linked:
            identity_link_proved, identity_link_skip_reason = _source_identity_link_proof_status(
                source_text,
                lore,
                new_characters,
                raw_alias,
                target,
            )
        if not identity_link_proved:
            logger.warning(
                "[Novel Context] Skipped unsafe identity link '%s' -> '%s': %s.",
                _plain_key(raw_alias),
                target,
                identity_link_skip_reason,
            )
            continue
        changed = any(
            explicit_aliases.get(alias_key) != target
            for alias_key in alias_keys
        )
        for alias_key in alias_keys:
            explicit_aliases[alias_key] = target
            alias_displays[alias_key] = _strip_balanced_brackets(raw_alias)
        if changed:
            record(
                "[Novel Context] Linked identity "
                f"'{_plain_key(raw_alias)}' -> '{target}'."
            )

    character_bounds = _find_lore_section(lore, CHARACTERS_SECTION)
    current_character_entries: List[Tuple[str, str]] = []
    if character_bounds:
        _, body_start, body_end = character_bounds
        current_character_entries = _parse_bullet_entries(
            lore[body_start:body_end]
        )
        characters, current_deduced_aliases = _deduplicate_character_entries(
            current_character_entries,
            explicit_aliases,
        )
        _retain_renderable_aliases(
            explicit_aliases,
            alias_displays,
            current_deduced_aliases,
            current_character_entries,
        )
    else:
        characters = []

    glossary_alias_entries: List[Tuple[str, str]] = []
    glossary_alias_bounds = _find_lore_section(lore, GLOSSARY_SECTION)
    if glossary_alias_bounds:
        _, body_start, body_end = glossary_alias_bounds
        glossary_alias_entries.extend(
            _parse_bullet_entries(lore[body_start:body_end])
        )
    glossary_alias_entries.extend(_parse_bullet_entries(new_glossary))
    _add_glossary_character_aliases(
        explicit_aliases,
        alias_displays,
        glossary_alias_entries,
        characters,
    )

    incoming_character_entries = _parse_bullet_entries(new_characters)
    raw_character_bounds = _find_lore_section(
        raw_global_lore,
        CHARACTERS_SECTION,
    )
    if raw_character_bounds and explicit_aliases:
        _, raw_body_start, raw_body_end = raw_character_bounds
        retained_alias_keys = {
            alias
            for name, _ in characters
            for alias in _character_alias_keys(name)
        }
        for raw_name, raw_value in _parse_bullet_entries(
            raw_global_lore[raw_body_start:raw_body_end]
        ):
            raw_aliases = _character_alias_keys(raw_name)
            if not raw_aliases or not (raw_aliases & set(explicit_aliases)):
                continue
            if raw_aliases & retained_alias_keys:
                continue
            incoming_character_entries.append((raw_name, raw_value))
    inferred_aliases, inferred_displays = _infer_unique_short_name_alias_entries(
        characters + incoming_character_entries,
        explicit_aliases,
    )
    if inferred_aliases:
        changed_aliases = []
        for alias_key, target in inferred_aliases.items():
            if explicit_aliases.get(alias_key) == target:
                continue
            explicit_aliases[alias_key] = target
            if alias_key in inferred_displays:
                alias_displays[alias_key] = inferred_displays[alias_key]
            changed_aliases.append((alias_key, target))
        if changed_aliases:
            characters, changed_deduced_aliases = _deduplicate_character_entries(
                characters,
                explicit_aliases,
            )
            _retain_renderable_aliases(
                explicit_aliases,
                alias_displays,
                changed_deduced_aliases,
                characters,
            )
            for alias_key, target in changed_aliases:
                record(
                    "[Novel Context] Linked identity "
                    f"'{alias_displays.get(alias_key, alias_key)}' -> '{target}'."
                )
    regular_incoming = [
        entry for entry in incoming_character_entries
        if not _is_descriptive_role_name(entry[0])
    ]
    descriptive_incoming = [
        entry for entry in incoming_character_entries
        if _is_descriptive_role_name(entry[0])
    ]

    for raw_name, raw_value in regular_incoming + descriptive_incoming:
        if (
            _is_non_character_work_entry(raw_name, raw_value)
            or _is_non_character_group_entry(raw_name, raw_value)
            or _is_non_character_metadata_or_item_entry(raw_name, raw_value)
            or _is_disposable_unnamed_character(raw_name, raw_value)
        ):
            continue
        incoming_aliases = _character_alias_keys(raw_name)
        forced_name = next(
            (
                explicit_aliases[alias]
                for alias in incoming_aliases
                if alias in explicit_aliases
            ),
            None,
        )
        effective_name = forced_name or raw_name
        descriptive_name = bool(
            _is_descriptive_role_name(raw_name)
            and not forced_name
        )
        if (
            not forced_name
            and _is_quarantined_character_entry(raw_name, raw_value)
        ):
            record(
                "[Novel Context] Quarantined role-like character "
                f"'{_plain_key(raw_name)}'."
            )
            continue
        if (
            not forced_name
            and _is_unstable_physical_character_entry(raw_name, raw_value)
        ):
            record(
                "[Novel Context] Quarantined physical placeholder "
                f"'{_plain_key(raw_name)}'."
            )
            continue
        incoming_aliases |= _character_alias_keys(effective_name)
        match_index = None
        for index, (existing_name, existing_value) in enumerate(characters):
            if (
                incoming_aliases & _character_alias_keys(existing_name)
                or _character_identities_match(
                    existing_name,
                    existing_value,
                    effective_name,
                    raw_value,
                )
            ):
                match_index = index
                break

        is_delete = _strip_balanced_brackets(raw_value).casefold() == "delete"
        log_key = _plain_key(raw_name)
        if is_delete:
            if match_index is not None:
                characters.pop(match_index)
                record(f"[Novel Context] Deleted character '{log_key}'.")
            continue

        canonical_name = _canonical_display_name(effective_name)
        clean_value, explicit_correction = _strip_character_correction_marker(
            raw_value
        )
        clean_value = _normalize_character_value(clean_value)
        clean_value = _gate_unproven_character_gender(
            canonical_name,
            clean_value,
            source_text,
            characters[match_index][1] if match_index is not None else "",
            explicit_correction,
        )
        if match_index is None:
            if descriptive_name:
                continue
            characters.append((canonical_name, clean_value))
            record(
                f"[Novel Context] Added character '{_plain_key(canonical_name)}'."
            )
            continue

        old_name, old_value = characters[match_index]
        merged_name = (
            canonical_name
            if forced_name
            else old_name
            if descriptive_name
            else _preferred_character_name(old_name, canonical_name)
        )
        merged_value = _merge_character_values(
            old_value,
            clean_value,
            allow_gender_correction=explicit_correction,
        )
        if (old_name, old_value) != (merged_name, merged_value):
            record(
                f"[Novel Context] Updated character '{log_key}'."
            )
        characters[match_index] = (merged_name, merged_value)

    final_character_entries = list(characters)
    characters, final_deduced_aliases = _deduplicate_character_entries(
        characters,
        explicit_aliases,
    )
    _retain_renderable_aliases(
        explicit_aliases,
        alias_displays,
        final_deduced_aliases,
        final_character_entries + incoming_character_entries,
    )
    lore = _replace_lore_section(
        lore,
        CHARACTERS_SECTION,
        [_format_character_line(name, value) for name, value in characters],
    )
    canonical_aliases = _canonical_alias_entries(
        explicit_aliases,
        characters,
        alias_displays,
    )
    if alias_bounds or canonical_aliases:
        lore = _replace_lore_section(
            lore,
            ALIASES_SECTION,
            [
                f"- {alias}: {target}"
                for alias, target in canonical_aliases
            ],
        )

    glossary_bounds = _find_lore_section(lore, GLOSSARY_SECTION)
    if glossary_bounds:
        _, body_start, body_end = glossary_bounds
        glossary = _normalize_glossary_entries(
            _parse_bullet_entries(lore[body_start:body_end])
        )
    else:
        glossary = []
    glossary_index = {
        _plain_key(name): index for index, (name, _) in enumerate(glossary)
    }

    for raw_name, raw_value in _parse_bullet_entries(new_glossary):
        clean_name = _strip_balanced_brackets(raw_name)
        clean_value = _strip_balanced_brackets(raw_value)
        if not clean_name or _is_invalid_context_key(clean_name):
            continue

        if _is_inverted_target_to_source_glossary_pair(clean_name, clean_value):
            clean_name, clean_value = clean_value, clean_name

        key = _plain_key(clean_name)
        val_key = _plain_key(clean_value)

        # Layer 1 inverse pair check: if key is missing but val_key is stored in glossary
        if key not in glossary_index and val_key in glossary_index:
            existing_name, existing_val = glossary[glossary_index[val_key]]
            if _plain_key(existing_val) == key:
                key = val_key
                clean_name, clean_value = existing_name, existing_val

        is_delete = clean_value.casefold() == "delete"

        if is_delete:
            target_idx = glossary_index.get(key)
            if target_idx is None:
                for idx, (g_name, g_val) in enumerate(glossary):
                    if _plain_key(g_val) == key or _plain_key(g_name) == key:
                        target_idx = idx
                        break
            if target_idx is not None:
                glossary.pop(target_idx)
                glossary_index = {
                    _plain_key(name): index
                    for index, (name, _) in enumerate(glossary)
                }
                record(f"[Novel Context] Deleted glossary term '{key}'.")
            continue

        if key in glossary_index:
            index = glossary_index[key]
            old_name, old_value = glossary[index]
            if old_value.casefold() == clean_value.casefold() and old_value[0:1].isupper():
                clean_value = old_value
            if old_name.casefold() == clean_name.casefold() and old_name[0:1].isupper():
                clean_name = old_name
            if (old_name, old_value) != (clean_name, clean_value):
                record(
                    f"[Novel Context] Updated glossary term '{key}'."
                )
            glossary[index] = (clean_name, clean_value)
        else:
            glossary_index[key] = len(glossary)
            glossary.append((clean_name, clean_value))
            record(
                f"[Novel Context] Added glossary term '{key}'."
            )

    lore = _replace_lore_section(
        lore,
        GLOSSARY_SECTION,
        [f"- {name}: {value}" for name, value in glossary],
    )
    return normalize_global_lore(lore), change_logs
