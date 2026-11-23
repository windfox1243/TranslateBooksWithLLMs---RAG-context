# Changelog

User-visible changes only, one short entry each. Entries for 1.x releases are
in [docs/changelog-archive.md](docs/changelog-archive.md).

## 2.0.0 - 2026-11-23

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
- `REQUEST_INTERVAL` spaces requests to a provider, to stay under a
  requests-per-minute quota.

### Changed

- Vietnamese addressing and pronoun rules apply only when the target language
  is Vietnamese; other targets no longer receive them.
- Jobs started with an older context contract run on the current one.
- Links in the app, release notes and translated-file attribution point to
  this fork; bug reports and feature requests go to the fork's issue tracker.
- Webhook URLs and headers are masked in the settings panel, and custom
  endpoints are checked before an API key is sent to them.
- A job in which no unit could be translated ends as failed instead of
  partial, and stays resumable.
- The CLI exits with status 1 when a job fails or finishes partially.
- Chunks the model refused, cut short, echoed back or left in the source
  language are retried and, if still bad, marked failed for a resume
  instead of being saved as translated.
- An EPUB file whose translation fails is written back unchanged.
- The refinement pass matches the glossary on its target forms, so it gets
  a glossary again.
- The web interface is translated throughout (audiobook dialog, page title,
  provider labels, file statuses) and follows a language switch at once.

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
- Security: output names are confined to the output folder, only documents
  and audio files can be opened from the web UI, request bodies are capped,
  and the server rejects foreign Host headers on a loopback bind.
- Rate limits pause the job instead of failing chunks; the OpenRouter model
  list works after the first job; no request silently falls back to Ollama.
- Resuming a job twice no longer runs two workers on it, and a server
  restart no longer interrupts a CLI job that is still running.
- An interrupted EPUB file resumes mid-file instead of from its first chunk.
- EPUB: sentences no longer run together where a long paragraph is split;
  TOC entries keep their own heading; mis-nested markup no longer erases a
  chapter; RTL output keeps valid XHTML.
- DOCX Plain Text Mode keeps tables and each image's size; SRT files with a
  byte-order mark keep their first subtitle.
- French output keeps the space before `! ? ; :`.
- Restoring a running job after a page reload works with Windows paths, and
  closing the page interrupts the job as intended.
- Startup no longer loads the GPU text-to-speech stack.

### Removed

- The anonymous installation fingerprint.
- Rewriting of `.env` and `.env.example` on import, and the unused
  `ENABLE_STRUCTURED_ADDRESSING` setting.
