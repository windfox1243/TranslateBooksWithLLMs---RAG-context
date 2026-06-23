# Changelog

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
