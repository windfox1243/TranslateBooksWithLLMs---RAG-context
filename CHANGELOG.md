# Changelog

User-visible changes only, one short entry each. Entries for 1.x releases are
in [docs/changelog-archive.md](docs/changelog-archive.md).

## 2.0.0 - Unreleased

The fork now versions independently of upstream. Its git history was
rewritten to drop committed binaries, so clones made before 2.0.0 must be
cloned again.

### Added

- Cost presets: `economy`, `balanced` (default) and `quality` trade LLM calls
  per chunk for quality. Choose one in the web settings panel, with
  `--cost-preset` on the CLI, or with `COST_PRESET` in `.env`.
- Every job ends with a log line giving its LLM calls per chunk, split into
  translation, context, editor and refinement.
- The CLI can run the Senior Editor (through the `quality` preset).
- `.env.example` now documents every setting the application reads.

### Changed

- Vietnamese addressing and pronoun rules apply only when the target language
  is Vietnamese; other targets no longer receive them.
- Jobs started with an older context contract run on the current one.
- Links in the app, release notes and translated-file attribution point to
  this fork; bug reports and feature requests go to the fork's issue tracker.
- Webhook URLs and headers are masked in the settings panel, and custom
  endpoints are checked before an API key is sent to them.

### Fixed

- Pausing an EPUB job mid-file now marks it as interrupted, so it resumes.
- Plain Text Mode no longer fails with a `TypeError` on start.
- Reloading the novel context during a job no longer crashes it.
- Context resync now imports addressing rules, and setting the relationship
  judge to "off" now turns it off during resync as well.
- EPUB: drop caps, ruby text, covers and text-free pages survive translation;
  CJK typography and metadata are handled correctly.
- Providers: Anthropic headers, invalid-key detection, Ollama rate limits and
  OpenRouter reasoning settings now behave correctly; tokenizer data works
  offline.
- Uploaded archives can no longer write outside their target folder.

### Removed

- The anonymous installation fingerprint.
- Rewriting of `.env` and `.env.example` on import, and the unused
  `ENABLE_STRUCTURED_ADDRESSING` setting.
