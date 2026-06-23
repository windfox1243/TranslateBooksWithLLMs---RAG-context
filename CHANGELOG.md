# Changelog

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
