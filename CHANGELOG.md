# Changelog

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
- Auto-updated context is now transactional: source facts are staged for the
  current chunk's prompt, then committed only after that chunk translates and
  saves successfully.
- Failed, partial, rate-limited, or interrupted chunks roll back their staged
  context so later chunks, re-sync, and refinement do not learn from
  uncommitted work.
- Refinement context mapping and background re-sync now ignore failed or
  partial chunk snapshots, keeping those chunks retryable instead of treating
  source-language fallback text as translated history.
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
