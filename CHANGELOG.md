# Changelog

## 1.14.54 - 2026-07-05

### Fixed

- **Character Gender Cross-Validation**: `UniversalAddressingEngine.validate_and_repair_pair()` now cross-validates target pronouns against character profile gender metadata (`character_genders`), preventing false male pronoun defaults (`anh`) for female addressees and repairing gender pronoun mismatches across all gender pairings (Female-Female, Male-Male, Female-Male, Male-Female).
- **Addressing Operator & Hierarchy Guardrails**: Enhanced `_repair_vietnamese_addressing_line` and `_repair_vietnamese_addressing_block` in `novel_context.py` to convert invalid bidirectional (`↔`) asymmetric pronoun rules into directional (`→`) entries, clean stray markdown formatting, and prevent generic senior/junior relationships from falsely triggering trainer-trainee hierarchy logic.

## 1.14.53 - 2026-07-05

### Fixed

- **Senior/Junior Academy Addressing Repairs**: Added `senior`, `senpai`, `tiền bối`, and `senior/junior` hierarchy context cues to `_VIETNAMESE_INCOMPATIBLE_ADDRESSING_REPAIRS` in `novel_context.py`, repairing incompatible `tôi/tớ` self-references into `em` when a junior student addresses older female/male seniors (`chị`/`anh`).
- **Context File Data Cleanup**: Cleaned up addressing entries in `F:/TranslateBook_Data/Novel_Contexts/I_Want_to_Be_a_Fluffy__Airheaded_Gray_Umamusume_and_Trick_My_Trainer___Vietnamese__context.txt` (Apollo Rainbow addressing senior girls Maruzensky, Silence Suzuka, Symboli Rudolf now correctly uses `em` - `chị` and Tomio Momozawa uses `em` - `anh`).

## 1.14.52 - 2026-07-05

### Added

- **Addressing Negative Constraints Directive**: Added `get_forbidden_pronouns` in `UniversalAddressingEngine` and updated `context_projection.py` to inject explicit negative constraints (`[CẤM DÙNG: ...]`) into translation system prompts.
- **Post-Translation Addressing Auditor**: Added `audit_addressing_violations` in `UniversalAddressingEngine` to detect dialogue turns violating active pair pronoun rules without extra LLM API overhead.

### Fixed

- **Cross-Directional Pair Hierarchy Repair**: Enhanced `_repair_vietnamese_addressing_block` in `novel_context.py` to cross-reference reverse addressing pairs (e.g. Trainee addressing Trainer as peer `cậu` while Trainer addresses Trainee as `em`), automatically repairing hierarchy mismatches across context blocks.

### Tests

- Added unit test coverage `test_forbidden_pronouns_and_auditing` in `test_universal_addressing_engine.py` and `test_cross_directional_addressing_block_repair` in `test_novel_context.py`.

## 1.14.51 - 2026-07-05

### Added

- **Directed Addressing Context Architecture**: Re-architected Vietnamese addressing context tracking into a DB-backed **Directed Addressing State Machine** with SQLite relational tables (`context_entities`, `context_addressing_rules`, `context_audit_logs`).
- **Structured LLM Addressing Extraction**: Added `extract_addressing_deltas_from_text` to parse structured JSON deltas directly from LLM output while filtering out group/crowd dialogue.
- **Deterministic Merge Policy Engine**: Implemented `ContextMergeEngine` enforcing User Lock overrides, configurable confidence thresholding (`ADDRESSING_MERGE_CONFIDENCE_THRESHOLD`, default `0.80`), register stability rules, and real-time SocketIO log events (`addressing_merged`, `addressing_rejected`).
- **Read-Only Context Projection & Markdown Tables**: Created `context_projection.py` to render active DB rules into clean, non-mutable system prompt guidelines and formatted Markdown table views (`convert_addressing_text_to_markdown_table`).
- **Universal Multi-Language Addressing Constraint Engine**: Introduced `UniversalAddressingEngine` (`src/utils/universal_addressing_engine.py`) providing high-performance O(1) table-driven intra-pair incompatibility filtering, register alignment, and social hierarchy constraint solving across Vietnamese, Japanese, Korean, Chinese, French, and English.
- **Trainee-to-Trainer Hierarchy Repair**: Added automatic repair in `_repair_vietnamese_addressing_details` to catch and correct peer pronoun mismatches when a trainee addresses a senior trainer (replacing invalid `cậu` with senior title/vocative).
- **Addressing Audit Trail & REST APIs**: Exposed `/api/translations/<id>/addressing-rules`, `/api/translations/<id>/addressing-audit-log`, and `/api/translations/<id>/addressing-rules/lock` endpoints for UI integration.

### Tests

- Added comprehensive unit test coverage (`test_directed_addressing_engine.py`) and novel integration test (`test_novel_addressing_integration.py`).

## 1.14.50 - 2026-07-04


### Fixed

- **POV-aware novel context selection:** Canonicalized dialogue speaker aliases before selecting POV context and stopped first-person POV filtering from hiding explicit character-pair or quoted-address matches.
- **Vietnamese addressing cleanup:** Narrowed possessive `second-person pronoun` cleanup so only known title heads such as `huấn luyện viên của Apollo Rainbow` are shortened, while uncertain possessive phrases remain unchanged. Vietnamese prompts now state that proven kinship, age, seniority, family, rank, and teacher/student hierarchy override name/title fallback.
- **Vietnamese second-person addressing:** Repaired source names, nicknames, and honorific calls that leaked into `second-person pronoun` fields by converting them to Vietnamese address-pair terms when supported by relationship/gender evidence, while preserving the source call in `vocative/address form`.
- **Vietnamese addressing persistence:** Added a relationship-driven fallback that seeds Vietnamese addressing forms from stored relationship/profile evidence when the addressing section is empty, preventing relationship-only context saves from leaving `## CURRENT ADDRESSING FORMS` blank.
- **Vietnamese peer register default:** Adjusted peer/classmate/roommate/friendly-rival addressing fallback and prompt guidance to prefer `tớ/cậu` by default, while preserving `mình/cậu` when established character voice or stored context supports it.
- **Vietnamese non-peer intimacy repair:** Prevented high-intimacy trainer, mentor, senior, and professional relationships from using peer self-references such as `mình` or `tớ`; these are repaired to `tôi` while preserving the appropriate second-person form and vocative, without forcing reciprocal addressing from the reverse direction alone.
- **Vietnamese kinship addressing stability:** Prevented established directional kinship/hierarchy pairs such as `em/anh` from being overwritten by weaker neutralized updates such as `tôi/anh` when no attitude-shift evidence is present.
- **Vietnamese peer and trainer addressing:** Repaired default peer-level `mình/cậu` updates to `tớ/cậu` unless an established soft/self-reflective voice is explicit, and prevented trainer-to-trainee entries from addressing the junior trainee as a senior form such as `chị`.
- **Context resume source-of-truth:** Normal translation resume now keeps the edited context file as the source of truth when it already contains dynamic state entries, instead of letting an older checkpoint snapshot overwrite manual context fixes. Snapshot restore remains available for empty contexts and explicit snapshot-first flows.
- **External context files:** Preserved existing absolute `.txt` context paths so contexts kept outside the default `Novel_Contexts` directory are not silently redirected to a same-named managed file.
- **Glossary compound handling:** Added glossary prompt guidance for compound source phrases, preserving required glossary translations exactly while translating surrounding meaningful words naturally unless a longer glossary entry overrides them.
- **NER glossary suggestions:** NER term suggestions now receive a small related subset of the existing glossary, selected by exact source matches or shared meaningful keywords, so new compound terms can reuse established translations without injecting the full glossary. Obvious untranslated fragments in suggested compound targets are repaired only when the raw source component is still present.
- **Save & Re-sync feedback:** The context editor Save & Re-sync button now remains clickable while editing and reports when there are no context changes to save, instead of becoming a silent disabled button.

### Tests

- Added regression coverage for canonical dialogue speaker aliases, first-person context selection, explicit relationship matches, conservative Vietnamese possessive addressing cleanup, Vietnamese second-person source-name repair, non-peer intimacy repair, directional kinship-pair stability, peer `tớ/cậu` default repair, trainer-to-trainee hierarchy repair, edited-context-over-stale-snapshot resume behavior, relationship-seeded Vietnamese addressing with the default `tớ/cậu` peer register, external context path preservation, compound glossary guidance, related-glossary NER hints and repair, and Save & Re-sync no-change feedback.

## 1.14.48 - 2026-07-03

### Fixed

- **Context-driven Vietnamese addressing guardrails:** Refined prompt style guardrails for Vietnamese translation targets to enforce strict adherence to actual relationship context, intimacy, and character dynamics rather than defaulting to generic fantasy/archaic address pairs (`ta-nàng`, `ta-ngươi`). Separated Korean vs Japanese honorific suffix rules (converting raw Korean `-ssi`/`-nim` into natural Vietnamese while allowing Japanese light novel suffixes like `-sama`/`-senpai`).
- **Addressing line auto-formatting & persistence:** Enhanced novel context parsing (`_repair_vietnamese_addressing_line` and `_has_complete_vietnamese_addressing_details`) to auto-format simple 2-part addressing lines (such as `tớ - cậu`) into canonical 3-part format, preventing the `## CURRENT ADDRESSING FORMS` section from being cleared during context updates.
- **Live context reload & UI button disabled states:** Added real-time disk context re-loading in `xhtml_translator` and `generic_translator` before translating each chunk/unit to immediately reflect edited Global Lore/Glossary entries without requiring job restarts. Added disabled styling and interaction locks for context re-sync UI buttons.
- **Immediate Save & Re-sync interrupt visual feedback:** Added immediate UI status badge updates (`Re-sync: pausing`), instant button loading/disable locks, and active WebSocket log emissions when requesting context re-sync during an active translation job so the user is immediately informed that the system is waiting for the current chunk to finish before interrupting and propagating updated context.

## 1.14.46 - 2026-07-03

### Added

- **Japanese & Korean target language style guardrails:** Added dedicated style guardrails to prompt generation for Japanese (`JAPANESE STYLE GUARDRAILS`) and Korean (`KOREAN STYLE GUARDRAILS`) target languages, enforcing preservation of honorific suffixes/titles (`-sama`, `-senpai`, `-san`, `-kun`, `-chan`, `-dono`, `-ssi`, `-nim`, `sunbae`, `oppa`, `unnie`, `hyung`, `noona`) and speech/politeness registers (`Desu/Masu` vs `Tameguchi`, `Jondaetmal` vs `Banmal`).

## 1.14.45 - 2026-07-03

### Fixed

- **Vietnamese addressing filter resilience:** Added flexible Regex field extraction for LLM key variations (`self-reference`, `xưng`, `second-person pronoun`, `gọi`, `vocative/address form`, `danh xưng`).
- **Expanded Eastern & extended family cues:** Expanded kinship and status cue dictionaries with Eastern fantasy, xianxia, wuxia, and extended family terms (`sư huynh`, `sư tỷ`, `sư đệ`, `sư phụ`, `sư tôn`, `tông chủ`, `trưởng lão`, `thiếu gia`, `tiểu thư`, `bác`, `chú`, `dì`, `thím`, `dượng`).
- **Attitude shift & workplace title exemptions:** Preserved intentional sarcasm / attitude shift registers (`mỉa mai`, `châm biếm`, `thù địch`) without forcing kinship repairs, and exempted formal workplace/professional titles (`chủ tịch`, `giám đốc`, `quản gia`, `bác sĩ`, `luật sư`, `thanh tra`, `giáo sư`) from generic neutral downgrade filters.

## 1.14.44 - 2026-07-03

### Fixed

- **Save & Re-sync button reactivity:** Fixed issue where context snapshot re-sync could only be triggered once due to chunk selection value resets, missing global anchor fallbacks, and unreleased button lock states on error.
- **Context Preview header layout:** Added flex wrapping to the Context Preview header flex containers so action buttons and controls wrap cleanly instead of overflowing off-screen on smaller viewports.

## 1.14.43 - 2026-07-03

### Fixed

- **Vietnamese novel-context addressing:** Repaired incompatible kinship pairs such as `tớ/chị` into coherent family-register forms and blocked generic downgrades like `tôi/cô` when existing context proves a sibling, seniority, royal, or title-based relationship.
- **Selective context injection:** Prompt context selection now includes addressing rows when the source form or dialogue metadata references the pair, so entries such as `Aster Evans → Ellen Evans: "Sis"` are not dropped from translation/refinement prompts.
- **Skipped-chunk source memory:** Novel context sessions now retain source text from chunks skipped by the update interval, giving later context updates the evidence needed for relationship and addressing corrections without extra LLM calls.
- **Dialogue continuity:** Scene-local dialogue state now survives narration-only chunks and context re-sync passes instead of being cleared when no new dialogue turns are detected.
- **Novel context snapshot resume:** Decoded resume snapshots now return canonical dynamic state text, preventing legacy dynamic-only snapshots from leaking malformed sections into later context views.
- **Sample prompt context loading:** Sample generation now normalizes novel context filenames before loading them, matching the safer web/API-managed context-file behavior used by translation routes.

### Tests

- Added regression coverage for Vietnamese addressing repair, dialogue-driven context selection, skipped source memory, narration-only dialogue carryover, and canonical snapshot decoding.

## 1.14.42 - 2026-07-03

### Fixed

- **Novel context addressing auto-formatting:** Added auto-sanitization for unquoted/2-part dynamic addressing entries into canonical 3-part format (`- Speaker → Addressee: "Alias" | "details" | reason`).
- **Addressing field completeness validation:** Strengthened `_has_complete_vietnamese_addressing_details` to verify that all three addressing fields (`self-reference`, `second-person pronoun`, `vocative/address form`) are non-empty.

## 1.14.41 - 2026-07-03

### Fixed

- **Vietnamese register addressing:** Enforced coherent register rules for Vietnamese dynamic addressing deltas, rejecting mismatched pronoun pairs such as `tôi` paired with contemptuous `ngươi`.
- **Release title naming:** Release workflows now set GitHub Release titles directly to tag names (`v1.14.41`), omitting the `TranslateBooksWithLLMs` prefix.

## 1.14.40 - 2026-07-03


### Fixed

- **Context identity consolidation:** Character consolidation can now preserve
  source-proven identity links when it merges stable labels, forms, disguises,
  codenames, or masked identities into a retained canonical character.
- **Focused identity evidence:** Consolidation receives only identity-relevant
  dynamic relationship lines instead of the full relationship state, reducing
  prompt noise while still catching far-apart alias evidence.
- **Context snapshot identity repair:** Resume/refine snapshots now keep merged
  identity aliases such as transformed forms and source-side labels from
  reappearing as separate canonical characters.

## 1.14.39 - 2026-07-02

### Fixed

- **Paused context re-sync recovery:** Saved translation cards now expose unfinished context re-sync state, convert stale background runs back to a resumable paused state, and route recovery through Resume Re-sync instead of normal translation resume.
- **Context re-sync safety guard:** Normal translation resume now refuses to start while a context re-sync is still running, pausing, or paused, preventing stale context snapshots from being skipped.
- **Context re-sync progress logs:** Checkpoint-only Resume Re-sync jobs now restore enough in-memory state to stream progress logs and status updates while saved snapshots are being propagated.
- **Saved-card context re-sync refresh:** Saved translation cards now keep polling while a context re-sync is running, so the badge and action button update when the background job completes.
- **Saved-card context re-sync auto-resume:** Resume Re-sync now preserves the pending translation auto-resume follow-up after the re-sync completes.
- **Saved-job failure badges:** Saved translation cards now recompute failed chunk counts from checkpoint rows, so the warning badge disappears after a previously failed chunk is retried successfully.

## 1.14.38 - 2026-07-02

### Fixed

- **Relationship-based addressing:** Novel context prompts now store richer social basis, scope, and reason details for address forms, and translation prompts use those facts for direct address and indirect references across target languages.
- **Short-name context selection:** Prompt context selection now matches distinctive short forms of romanized full names and avoids leaking unrelated one-sided relationship rows.
- **Vietnamese address pairs:** Vietnamese prompt guidance now treats address as a paired social system, covering examples such as `em-cô`, `em-thầy`, `tớ-cậu`, `bố/mẹ-con`, title-only, name-only, and neutral forms without treating the list as exhaustive.
- **Non-character magic terms:** Named magic circles, artifacts, rewards, and similar terms are filtered out of Characters & Genders while preserved as glossary terminology.
- **Non-character organization terms:** Companies, organizations, factions, family houses, and similar proper nouns are pruned from Characters & Genders and left as glossary terminology when useful.
- **Identity-link diagnostics:** Unsafe identity-link skips now log the exact rejection reason, while explicit appositive and `aka`-style source evidence can safely prove stable aliases.

## 1.14.37 - 2026-07-01

### Added

- **Add New Content jobs:** Saved translations can now start a separate continuation job from an updated source file, reusing only the matching completed prefix and translating unfinished, changed, or newly added content into a new output file.
- **Continuation progress logs:** Continuation jobs now log when prefix comparison starts and how many existing units were reused before translation begins.
- **New-chapter-only context carryover:** When an Add New Content source contains only new chapters and no old prefix matches, the job now starts its novel context from the previous job's latest saved context snapshot.

### Fixed

- **Context timeline leakage:** Historical chunk context views now show their exact saved snapshot instead of mixing in future global lore, while explicit Global context edits still use the latest book-wide lore.
- **Completed-job continuation bases:** Completed checkpoints now remain visible in the saved translations list as Add New Content bases, with normal Resume disabled for already-complete jobs.
- **Windows venv setup:** `setup-and-update.bat` now uses an existing `venv\Scripts\python.exe` when available and falls back to `python` or `py -3` for first-time setup.

## 1.14.36 - 2026-07-01

### Fixed

- **Kinship-based gender inference:** Added automatic gender inference from kinship nouns (e.g. "father of Shigure Aya" -> Male; "mother of Toda Hitona" -> Female) in character details and names during context normalization.
- **Kinship-based consolidation merging:** Generic kinship role characters (like "Shigure Father") are now correctly merged into matching proper-named characters (like "Shigure Soichiro") during the consolidation pass, preferring proper names as the canonical keys.

## 1.14.35 - 2026-07-01

### Fixed

- **Source-name identity memory:** Stable Chinese, Japanese, and Korean source addressing forms already present in dynamic context now become trusted character aliases during context merging, preventing later chunks from reintroducing pinyin, romaji, or revised-romanization duplicate characters such as `Tuo Ping`.
- **CJK/Korean romanization guardrails:** Context-update prompts now explicitly reuse existing canonical romanized character names instead of inventing new romanized variants, while keeping romanization suggestions in glossary entries.

## 1.14.34 - 2026-07-01

### Fixed

- **Unicode novel-context filenames:** Auto-created novel context filenames now preserve Unicode letters and numbers, so source titles in Chinese, Japanese, Korean, and other scripts no longer collapse into underscores.
- **Name alias consistency:** Novel-context glossary entries that point to an existing character now also act as identity aliases during context merging, preventing source-name variants from becoming duplicate character entries.
- **CJK/Korean source-name variants:** Han full-name glossary entries now infer the common final two-character short form, and three-syllable Hangul names infer the two-syllable given-name form when the glossary target resolves to one known character.
- **Address suffix aliases:** Romanized address forms such as `Akane-san`, `Takuhei-kun`, and `Kim Min-su-ssi` now merge into the base character entry instead of creating separate durable characters.

## 1.14.33 - 2026-07-01

### Fixed

- **EPUB refine-after topology:** Chapter-aware EPUB refine-after now uses EPUB spine/content files as refinement units, preserving the stable chapter/file topology instead of re-chunking the translated EPUB into a different number of refinement requests.
- **Legacy checkpoint compatibility:** This avoids trusting old over-split EPUB checkpoint rows for refinement call count, so projects completed before the chapter-splitting fix do not inherit the old `1020`-chunk topology during refinement.

## 1.14.32 - 2026-07-01

### Fixed

- **Refinement phase start freeze:** Refinement no longer streams the full novel context payload to the browser when loading global lore for phase 2, preventing large context files from freezing the UI/socket before the first refinement request.
- **EPUB refinement progress:** EPUB refine-after now emits progress per refinement chunk instead of only after each XHTML file, preventing phase 2 from looking frozen during slow or multi-chunk files.

## 1.14.31 - 2026-06-30

### Fixed

- **Global context edit cost:** Saving changes from the Global context tab now propagates the edited global lore through saved snapshots without replaying every later chunk through the LLM, as long as the dynamic state was unchanged.
- **Timeline safety:** Dynamic-state edits still use the existing forward LLM replay, and global-only propagation falls back to LLM replay when a later chunk has no saved dynamic snapshot to preserve.
- **Pause/resume consistency:** Context resync checkpoints now remember whether the run is global-lore propagation or timeline replay, so paused jobs resume with the same behavior.

## 1.14.30 - 2026-06-30

### Fixed

- **Chapter-aware chunking false positives:** Generic chapter detection no longer treats English prose such as `That's why I...`, `But I...`, `If I...`, `I...`, or `It's been 1 year.` as repeated generic chapter headings. This prevents chapter-aware mode from splitting normal prose into many tiny translation units.
- **Shared file-type coverage:** The fix applies through the shared chapter detector used by TXT, EPUB Plain Text Mode, and DOCX Plain Text Mode, while preserving known multilingual headings and safe generic numbered headings.

## 1.14.29 - 2026-06-30

### Fixed

- **Live chunk-budget refresh:** New web translation jobs now reload `.env` before resolving `MAX_TOKENS_PER_CHUNK` when the UI does not send an explicit per-job value, preventing stale server defaults from silently controlling chunk size.
- **Chunk-budget observability:** EPUB HTML and plain-text translation logs now include the source-token budget used when creating translation units, making unexpected chunk counts easier to diagnose immediately.

## 1.14.28 - 2026-06-30

### Fixed

- **Seria name variant canonicalization:** Relationship and addressing context now resolves compact spacing variants such as `Seria Bladi Demonkill` and one-edit typo variants such as `Seria Blady Demonkill` to `Seria Bladi Demon Kill` when the match is unique.
- **Composite relationship party normalization:** Dynamic context now canonicalizes character names inside composite parties such as `Kim Si-hu & Seria Bladi Demonkill`.
- **Vietnamese first-person consistency:** Translation and refinement prompts now include Vietnamese style guardrails that prefer consistent serious-literary `tôi` narration over accidental switches to casual `mình`.

## 1.14.27 - 2026-06-30

### Fixed

- **Singular/plural character alias merge:** Near-identical singular/plural entity names such as `Death God` and `Death Gods` now merge only when their descriptions substantially overlap, and the plural form is persisted as an alias.
- **Plural group safety:** Weakly related plural groups remain separate, preventing broad plural normalization from merging background groups into named individuals.

## 1.14.26 - 2026-06-30

### Fixed

- **Deduplicated context logs:** Removed duplicate context-change emissions caused by the core context logger/print path and the UI progress callback both reporting the same event.
- **Safer dialogue carry-over:** Dialogue speaker state now persists only from current high-confidence dialogue turns. Empty, malformed, or uncertain dialogue attribution clears the carried state instead of silently reusing a previous chunk's speaker.
- **Weaker dialogue continuity prompt:** Context-update prompts now treat previous scene speaker state as a weak hint only, requiring local source evidence before assigning a speaker.

## 1.14.25 - 2026-06-29

### Fixed

- **Restored legacy novel-context update prompts:** Reverted lore updates from full JSON objects back to the stricter legacy `[NEW_CHARACTERS]`, `[IDENTITY_LINKS]`, `[NEW_GLOSSARY]`, and `[DYNAMIC_STATE]` blocks because JSON-shaped character objects made bad semantic facts easier to merge.
- **Role-only character quarantine:** Prevented transferable roles and address terms such as `Summoner`, `Summoner-nim`, `NPC`, and `Player Character` from being admitted or updated as durable characters unless a source-proven identity link maps them to a canonical person.
- **Safer prompt and dynamic context views:** Existing role-like character pollution is no longer injected into selective context prompts or used as a dynamic-state relationship party, without silently deleting user context files.
- **Preserved durable resync controls:** Kept the v1.4.25 pause/resume/status UI and checkpoint persistence for context resync.

## 1.4.25 - 2026-06-29

### Added

- **Durable context resync controls:** Added background API routes and UI buttons to manually trigger and inspect progress of context resynchronization across translation checkpoints.
- **Structured JSON-based context updates:** Replaced plaintext key-value prompts with structured JSON instructions for `new_characters`, `identity_links`, `new_glossary`, and `dynamic_state`, supporting both camelCase/snake_case and map/list output formats from the LLM.
- **Safeguard for unproven character merges:** Added checks to reject identity links attempting to merge distinct named characters (e.g., merging Alice and Bob) if both already exist with detailed lore descriptions, protecting against hallucinated aliases.
- **Durable dynamic state deltas:** Enabled partial update support for relationships and addressing forms, where omitted fields remain stored indefinitely, and explicit `DELETE` operations can prune stale entries.

## 1.4.24 - 2026-06-29

### Fixed

- **Context resync scene key reset:** Fixed a bug during context resync where chapter boundaries were not detected due to missing `chapter_index` in EPUB database checkpoints. We added a fallback check for `scene_key` inside the `dialogue_attribution` dictionary to correctly reset dialogue speaker tracking at chapter transitions across all formats.
- **EPUB chapter index persistence:** Updated the EPUB translation pipeline to write `chapter_index` to the root of `chunk_data` in the SQLite checkpoint database, aligning it with other formats.

## 1.4.23 - 2026-06-29

### Added

- **Multi-language explicit identity links:** Explicit `[IDENTITY_LINKS]` and `[CHARACTER ALIASES]` now accept physical aliases like "boy", "girl", "child", "man", "woman" and their equivalents in any language (e.g. "소년", "garçon", "cậu bé"). When a link is explicitly established, it bypasses the unstable-alias safeguard to merge the character cards. The background safeguard still protects against automatic/incidental mapping of these words.

## 1.4.22 - 2026-06-29

### Fixed

- **Model/Provider override on resume:** Fixed a bug where switching models or providers when resuming a paused translation did not persist to the checkpoint database and in-memory translation status. This caused the UI to show the original model and reverted the job back to the original model if paused/resumed again.

## 1.4.21 - 2026-06-29

### Fixed

- **Concept/Hallucination filtering in Novel Context:** Extended character recognition logic to automatically detect and discard abstract concepts, personifications, hallucinations, metaphors, or inanimate objects (e.g. "Death" as a personified concept) from the character list based on their description.
- **Spurious delete key handling:** Fixed parsing of `DELETE` bullet points when the LLM outputs command bullet lines out of sequence (e.g., `- DELETE:` or `- DELETE: <name>`), preventing them from being added to the character list as a character named "DELETE".

## 1.4.20 - 2026-06-28


### Added

- **LLM context consolidation pass**: after every N context chunk updates (default: 5, configurable via `NOVEL_CONTEXT_CONSOLIDATION_INTERVAL` in `.env`), an LLM call rewrites the `## CHARACTERS & GENDERS` section to merge duplicate or redundant character descriptions that the deterministic heuristic merge layer misses. This fixes issues such as the same character receiving near-identical descriptions with slightly different wording across chunks (e.g., "protagonist of the game Glory of Victory, a soldier who seeks revenge" appended after an identical entry). Set `NOVEL_CONTEXT_CONSOLIDATION_INTERVAL=0` to disable.
- Added `BYPASS_CONTEXT_GATING` checkbox to the Web UI Novel Context settings panel. The toggle lets users enable or disable context validation per-job from the UI. The state is persisted to profiles and `.env` via the Settings panel and updates reactively across all 7 supported UI languages.

## 1.4.19 - 2026-06-28


### Added

- New configuration setting `BYPASS_CONTEXT_GATING` (default: `True`) to bypass deterministic English pronoun regex checks on source text. This allows the translation pipeline to directly trust LLM-identified character genders, aliases, and corrections, resolving context degradation issues during translation of non-English novels.

## 1.4.18 - 2026-06-25

### Fixed

- Novel context now gates new or corrective character genders against
  deterministic source evidence before merging them into durable lore. This
  prevents early chunks from saving nearby pronouns from another character as a
  character's gender, such as storing Kim Ji-an as female when chunk 1 does not
  prove that.

## 1.4.17 - 2026-06-25

### Fixed

- Novel context no longer accepts bare narrative-role labels such as
  `Protagonist`, `Hero`, or `Main Character` as canonical character names or
  identity links. This prevents in-story uses of "protagonist" from merging the
  main character into a different named character.
- Factions, nations, companies, and one-off advertisement/job labels are now
  filtered out of `CHARACTERS & GENDERS` instead of being stored as people.
- Novel context now rejects bare romantic/family labels such as `Lover` or
  `Ex-girlfriend` as durable identity aliases, and drops relationship rows that
  collapse to the same character after alias resolution.
- Source-side relationship clauses such as `Kim Ji-an's ex-girlfriend cheated
  on him` can repair Kim Ji-an's gender from direct pronoun evidence without
  treating the existence of an ex-girlfriend as gender proof by itself.
- Reincarnation cleanup now prevents the original human identity from absorbing
  the gender of the current reincarnated body while still allowing the current
  named form to be repaired from source-proven body evidence.

## 1.4.16 - 2026-06-25

### Fixed

- Generated compact `.env` files now include
  `NOVEL_CONTEXT_SOURCE_MEMORY_CHARS`, matching `.env.example`, the docs, and
  the runtime default.

## 1.4.15 - 2026-06-25

### Improved

- Novel context source analysis now receives a bounded rolling memory of
  previous source chunks via `NOVEL_CONTEXT_SOURCE_MEMORY_CHARS`, helping it
  resolve identities and genders that are split across chunk boundaries.

### Fixed

- Work titles such as games, novels, or series are no longer persisted in
  `CHARACTERS & GENDERS` when the model describes them as non-character works.
- Reincarnation context can now repair a stale target identity gender when one
  context entry stores the current-form gender and the dynamic state links that
  source identity to the current named form.

## 1.4.14 - 2026-06-24

This stable release hardens the source-derived novel context architecture
around failed chunks, resume, re-sync, refinement, and prompt assembly.

### Improved

- Novel context is now injected into the dynamic user prompt instead of the
  system prompt. This keeps provider-side system-prompt caching stable while
  still giving each translation/refinement request the relevant book memory.
- Oversized novel context is rendered through a relevance selector that keeps
  matching characters, aliases, glossary terms, addressing forms, and dormant
  relationships for the current unit instead of blindly sending the whole file.
- New `.env` knobs control context prompt size and source-context update
  cadence: `NOVEL_CONTEXT_PROMPT_MAX_TOKENS` and
  `NOVEL_CONTEXT_UPDATE_INTERVAL`.
- Original glossary rules remain hard requirements. Novel-context glossary or
  terminology entries are treated as discovered hints, and the prompt now
  explicitly says `# GLOSSARY - REQUIRED TRANSLATIONS` wins on conflict.
- The glossary preview UI now describes the block as being inserted into the
  translation prompt, matching the actual architecture.
- Context analysis prompts now explicitly tell the LLM to correct stale stored
  gender when the latest source proves the current named form's gender, and to
  avoid saving prompt/control labels as character facts.
- Context analysis now treats descriptor-only labels such as `Protagonist of X`
  or `character from X` as merge-only hints instead of durable character names.
- Narrative-role phrases such as `Protagonist of X` can still resolve
  addressing and relationship rows to the named character without being saved
  as noisy aliases.
- Context analysis now tells the LLM to omit numbered/background casualties and
  generic staff labels unless they are source-named, recurring, or needed for a
  durable addressing/relationship choice.
- Chapter-aware translation help now explains that short chapters are not
  merged together, so request count can be higher than normal chunking while
  preserving chapter/refinement alignment.

### Fixed

- Deferred failed-chunk retries no longer re-run source context analysis for
  the same chunk, preventing duplicated character, glossary, dialogue, or
  relationship updates.
- Failed-chunk retries now preserve the original per-chunk source-context
  snapshot instead of overwriting it with later/final context.
- Resume now restores the latest available source-context snapshot, including
  snapshots from failed or partial chunks, while keeping failed translation
  output retryable.
- Failed chunks with source-context snapshots are now visible in the context
  selector and can be edited/re-synced like completed source snapshots.
- EPUB resume now restores source context from the latest processed snapshot
  instead of rewinding to the chunk before the first failed translation.
- Plain-text mode now carries source-context snapshots from failed or partial
  previous units into later files/chapters.
- Generic TXT/SRT retry paths preserve the failed unit's original
  source-context snapshot when a same-run deferred retry succeeds.
- Generated `.env` files are now compact and practical instead of copying the
  full commented `.env.example` reference. New private configs still include
  current knobs such as `NOVEL_CONTEXT_PROMPT_MAX_TOKENS` and
  `NOVEL_CONTEXT_UPDATE_INTERVAL`.
- GitHub release workflow metadata now uses the current tag name and publishes
  stable releases instead of the stale `1.4.12` prerelease label.
- GitHub release notes now use only the current changelog section and unwrap
  source hard line breaks so the release page does not show awkward manual
  newlines.
- Source-proven pronoun evidence such as `Eric suspected Valentine ... her
  identity` can now repair a stale character gender without incorrectly
  flipping the sentence subject.
- Character profiles now discard context prompt/control fragments such as
  `current rank and title` or `title/nickname for Eric` instead of persisting
  them as lore.
- Internal correction reasons such as `source pronoun evidence` and
  `reincarnated current form` are no longer saved as character descriptions.
- Descriptor-only role entries now merge into the named character when a shared
  source-work role proves the identity, without renaming the character to the
  descriptor.
- Scene-local aliases such as `the girl`, `the protagonist`, or `user` are no
  longer persisted as character aliases.
- Ruler/title descriptions such as `ruler of the Empire; Emperor, ruler of the
  Empire` now compact into one concise title fact.
- Incidental numbered/background roles such as wounded soldiers and generic
  one-off doctors are filtered out of durable global lore, while recurring or
  source-named generic roles remain preserved.
- Legacy context normalization now also removes relationship/addressing rows
  that point only at discarded background characters, while preserving arbitrary
  dormant relationships for any number of chunks.

### Safety and compatibility

- Refinement still ignores failed or partial translated-output rows, so source
  facts can help later chapters without treating untranslated fallback text as
  final translation.
- Context re-sync walks failed or partial source chunks for global context, but
  their translation status remains unchanged and retryable.
- The hard glossary remains independent from source-derived novel context, so
  existing glossary databases and NER/imported glossary terms keep their
  priority.

### Validation

- Full automated validation passed: 1,422 passed, 1 skipped, 10 deselected.
- Windows executable built and smoke-tested locally: `/`, English settings
  locale JSON, and translation batch JavaScript all returned HTTP 200. A fresh
  first-run folder generated the compact `.env` with the current context knobs.

## 1.4.13 - 2026-06-23

This stable release makes refinement consume the final canonical lore while
preserving scene-local history, and turns context re-sync into an actionable
correction workflow instead of a context-only update.

### Improved

- Character identity is now represented by a shared canonical alias registry.
  Source-proven titles, ranks, nicknames, and multilingual aliases resolve to
  the same character across global lore, relationship/addressing state,
  dialogue attribution, translation, re-sync, resume, and refinement.
- Context analysis now emits explicit identity links only when the source
  establishes stable identity; ambiguous or transferable bare ranks remain
  separate instead of being merged by role similarity.
- Saved dialogue maps and scene speaker state are re-canonicalized against the
  latest identity registry before prompt injection.
- Every refinement unit now receives the final book-wide character identities,
  source-proven genders, and glossary terminology.
- Historical addressing forms, relationship evolution, and dialogue
  attribution remain mapped to the appropriate translation unit, preventing
  later relationship state from leaking into early chapters.
- Refinement prompts explicitly prevent later-discovered global facts from
  being added, foreshadowed, or revealed before they appear locally.
- Background context re-sync can reuse API credentials from the live job and
  waits according to the configured request timeout when pausing active work.
- Addressing forms and relationship evolution now use a durable keyed registry:
  LLM responses provide only deltas, omitted pairs remain stored for any number
  of chunks, and matching participant pairs update deterministically.

### Fixed

- Character profiles no longer retain explanations such as `Gender confirmed
  by source text ...`; durable profiles store only the normalized result.
- Context-analysis prompts explicitly reject gender-neutral words such as
  `spouse`, military rank, occupation, or role context as new gender evidence.
- Direct gendered descriptions retained in metadata, such as `a blonde-haired
  girl`, now promote an earlier `Unspecified` entry deterministically.
- A source-proven title identity such as `Lieutenant Colonel` → `Eric` now
  merges duplicate character entries and rewrites relationship, addressing,
  and dialogue references to the canonical name.
- Malformed character entries with embedded contradictory gender fragments,
  such as `Male, ...; Female, ...`, are collapsed into one canonical gender
  and concise cumulative description.
- Bare role entries are now folded into a named character when exactly one
  named entry already carries that role as its own title; ambiguous shared
  roles remain separate unless there is an explicit identity link.
- Reincarnated/transformed characters now keep the gender of the current
  named form rather than the previous body, and raw source evidence can repair
  this before context is saved for any supported file type.
- Raw source coreference such as `the Lieutenant Colonel's office` followed by
  `Eric` now supplements missed model identity links in the shared context
  updater, so the fix applies to TXT, EPUB, SRT, DOCX, and chapter mode.
- Direct-address title dialogue such as `"Lieutenant Colonel."` followed by
  Eric replying or acting now also supplements missed identity links.
- Context-analysis prompts now explicitly preserve source-language title or
  alias surface forms in identity links when non-English source text proves
  the mapping.
- Auto-updated source context now remains available even when the chunk's
  translation output fails, so later chapters can still use discovered
  character, gender, glossary, relationship, and dialogue facts.
- Failed TXT/SRT units, plain-text chunks, and EPUB/XHTML chunks now receive a
  deferred automatic retry before final output is reconstructed.
- Failed chunks remain marked retryable until their translation succeeds;
  source-context snapshots may advance, but source-language fallback text is
  never treated as a completed translation.
- Refinement context mapping ignores failed or partial translated-output rows,
  while background re-sync still walks their source text so global context is
  not weakened by a translation failure.
- The context editor now reports a visible localized error when Save &
  Re-sync cannot start because no editable snapshot, job, or global anchor is
  available.
- Context re-sync no longer crashes with `AttributeError: 'int' object has no
  attribute 'value'` when recording a numeric context revision.
- The context editor now exposes a dedicated book-wide `Global Context` view.
  Editing global characters, genders, glossary, or terminology automatically
  anchors re-sync at the first available historical snapshot; users no longer
  need to select an arbitrary chunk for global corrections.
- Early refinement units could retain incomplete character or glossary lore
  even after later source text established the canonical facts.
- Re-syncing context during or after refinement could update snapshots without
  correcting the already-produced final translation.
- Selecting an early context snapshot in the editor displayed its obsolete
  character, gender, and glossary lore instead of combining current global
  lore with that chunk's historical relationship state.
- Saving an edited historical snapshot temporarily replaced the browser's
  cached `Latest state` view with the selected chunk.
- Re-sync during refinement now invalidates the old result, repairs subsequent
  snapshots, and runs one fresh corrective refinement from the preserved
  first-pass translation.
- Corrective refinement replaces the intended final file across TXT, SRT,
  EPUB, and DOCX instead of creating a duplicate or refining an already-refined
  file.
- Characters disappearing for many unrelated chunks could lose their saved
  addressing forms when the context model returned a complete-looking dynamic
  section containing only currently active relationships.

### Safety and compatibility

- The first-pass translation is retained with the checkpoint only for jobs
  using refine-after mode and is removed when the checkpoint is deleted.
- Context and refinement revisions are persisted so failed or interrupted
  corrective passes remain visibly stale rather than being marked current.
- Durable addressing or relationship entries can be removed only through an
  explicit `DELETE` delta; model omission is never interpreted as deletion.
- Refinement replays legacy snapshots cumulatively, recovering dormant
  relationships omitted by old batch snapshots without leaking final
  end-of-book dynamic state into early chapters.
- Gender remains `Unspecified` without source evidence. Character names are not
  treated as calibrated gender probabilities because multilingual,
  transliterated, unisex, title-based, and fictional names make that inference
  unsafe as global lore.

### Validation

- 1,405 selected automated tests passed.
- Windows executable startup and local UI smoke tests passed.

## 1.4.12 - 2026-06-23

This stable release promotes the global-context experiment after hardening
translation, refinement, resume, profile, and context behavior across TXT,
SRT, EPUB, and DOCX workflows.

### Added

- Source-first global novel context with canonical characters, genders,
  glossary terms, addressing forms, and relationship evolution.
- Chapter-aware translation mode with language-independent structural
  detection and safe subdivision of oversized chapters.
- Hidden dialogue speaker/addressee attribution for ambiguous conversations,
  including checkpoint persistence and refinement reuse when units align.
- Save/load translation profiles and detailed per-step context, translation,
  re-sync, and refinement logs.

### Improved

- Character identity consolidation now merges title-only aliases into named
  characters while preserving distinct named monarchs.
- `Unspecified` genders are promoted when later source evidence directly proves
  a specific gender, without weakening protection against unsupported flips.
- Character descriptions are normalized into concise cumulative facts instead
  of repeatedly appending duplicate roles.
- Context metadata remains English and canonical when switching translation
  models, providers, target languages, or importing a profile for continuation
  chapters.
- Profiles restore glossary, custom-instruction, and novel-context selections
  only after their asynchronous option lists are ready.
- Stable releases are compared correctly against prerelease versions, and the
  update checker now targets this repository.

### Fixed

- Resuming with another model could feed stale, unnormalized snapshot lore to
  context analysis.
- Existing `Unspecified` entries could reject later `Male` or `Female` evidence
  unless the model emitted an exact correction marker.
- Title/name aliases such as `Emperor` and `Serena Augusta` could remain
  duplicated.
- Direct evidence retained inside a description, such as a character mourning
  `his brother`, was not recovered when the model still returned
  `Unspecified`.
- Profile loading could silently lose the selected context, custom
  instructions, or glossary because of startup timing.
- Failed EPUB and DOCX chunks are retained as failed and retried rather than
  appearing as completed untranslated content.
- EPUB resume accounting, cloud API resume, context re-sync, locale-reactive
  context controls, and refine-after context alignment regressions.

### Safety and compatibility

- Profiles never import old translation IDs, resume indices, context
  snapshots, or dialogue state into a new file.
- New jobs continue to read the current `MAX_TOKENS_PER_CHUNK` value from
  `.env`; profiles do not freeze the previous job's chunk budget.
- Speaker attribution remains hidden metadata and is never written into the
  translated output or editable novel-context file.

### Validation

- 1,368 selected automated tests passed.
- Windows executable startup and local UI smoke tests passed.

## 1.4.12-context-experiment.2 - 2026-06-23

This prerelease adds hidden, scene-local dialogue speaker attribution to the
source-first context architecture.

### Added

- Dialogue-turn detection for multilingual quotation marks, dialogue dashes,
  subtitle markers, and short unlabelled conversational line sequences.
- Speaker and addressee inference inside the existing source-context LLM call,
  avoiding a second API request per translation unit.
- Compact progress logs showing identified, assigned, and uncertain dialogue
  turns after context analysis.
- Stable dialogue-turn identities and checkpoint persistence for resume,
  context re-sync, and refine-after workflows.

### Improved

- Translation and refinement prompts can use high-confidence speaker metadata
  for pronouns, addressing forms, register, and character voice.
- Speaker state carries across adjacent units and resets at chapter boundaries.
- TXT, SRT, EPUB, and DOCX refinement reuse speaker maps only when translation
  and refinement units align exactly; otherwise refinement performs fresh
  monolingual context analysis.

### Safety

- Speaker attribution remains hidden working metadata and is never written to
  novel-context files, the context editor, profiles, or translated output.
- Only canonical characters already present in novel lore are accepted.
- Unknown names and assignments below the confidence threshold remain
  uncertain and are not injected into translation prompts.

### Validation

- 1,356 selected automated tests passed.
- Windows executable startup smoke test passed and reported the expected
  prerelease version.

## 1.4.12-context-experiment.1 - 2026-06-23

This prerelease contains the global-context architecture experiment and its
resume, refinement, and checkpoint reliability fixes.

### Added

- Source-first novel context analysis for TXT, SRT, EPUB, and DOCX workflows.
- Structured global lore, glossary, current addressing forms, and relationship
  evolution sections.
- Chapter-aware translation mode with language-independent structural
  detection and safe subdivision of oversized chapters.
- Save/load translation profiles.
- Per-step UI logs for translation, context updates, context re-sync, and
  refinement.
- Exact translation-unit reuse for refine-after workflows when checkpoint
  drafts match the translated output.

### Improved

- Character alias normalization and duplicate removal.
- Gender handling now avoids unsupported guesses, preserves established facts,
  and accepts explicit source-backed corrections.
- Relationship arrows are normalized to plain Unicode instead of LaTeX.
- Context snapshot selection uses actual persisted checkpoint indices.
- Cloud-provider credentials are restored safely for manual resume and
  context-triggered auto-resume.
- `MAX_TOKENS_PER_CHUNK` values above 1,000 are honored, including values
  loaded from `.env`.
- EPUB resume progress is counted from untouched source XHTML, preventing
  translated-language tokenization from changing totals or shifting context
  indices.

### Fixed

- EPUB client-creation failures could be reported as successful translations.
- EPUB and DOCX Phase-3 fallbacks could preserve source-language chunks while
  marking them completed.
- Failed EPUB and DOCX chunks are now stored as failed and retried on resume,
  including non-contiguous failures before later successful chunks.
- EPUB processing no longer advances beyond an incomplete chapter/file, so the
  resume pointer cannot skip untranslated content.
- Partial-file progress is restored without counting failed chunks as complete.
- Context re-sync no longer loses cloud API credentials during auto-resume.
- Locale-sensitive context controls refresh immediately when the UI language
  changes.

### Validation

- 1,348 selected automated tests passed.
- Windows executable startup smoke test passed.

### Known limitation

- Jobs falsely marked completed by an older build can only be resumed if their
  checkpoint was retained. Otherwise, start a new translation for the affected
  file.
