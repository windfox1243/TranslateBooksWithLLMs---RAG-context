# Architecture

TranslateBook is a checkpointed translation pipeline with a Flask API and a
browser-based client. Format adapters normalize input into translation units,
the translation service obtains a draft, optional editor stages assess or
repair it, and persistence records enough state to resume without rebuilding
already completed work.

## Runtime flow

1. API handlers create or resume a translation job and preserve one canonical
   output path for that job.
2. A format pipeline (`plain_text_pipeline`, `xhtml_translator`, subtitle, or
   a generic adapter) produces independent translation units.
3. `src.core.translator` communicates with the configured LLM and returns a
   draft. The Senior Editor can refine it, but editor availability does not
   determine whether a structurally valid draft exists.
4. Format validation decides whether the draft can be rebuilt into the output.
5. The checkpoint manager stores each unit and updates aggregate job progress.
6. The API exposes execution and quality state independently; the frontend
   renders completed jobs with editorial findings as review recommended.

## State model

Execution and quality are separate dimensions:

| Dimension | Values | Meaning |
| --- | --- | --- |
| Unit execution | `completed`, `failed` | Whether usable translated content exists |
| Unit quality | `not_checked`, `passed`, `review_required` | Result of optional quality assessment |
| Job execution | Existing job lifecycle states | Scheduling, interruption, and completion |
| Job quality | `not_checked`, `passed`, `review_required` | Aggregate of completed units |

An editor transport error, inconclusive advisory policy, or unresolved
non-structural finding preserves a valid draft as `completed` with
`review_required`. Provider failure, empty output, or structural corruption is
an execution failure. This distinction prevents expensive editor retries from
discarding useful translations while retaining an explicit review queue.

## Package boundaries

- `src/core/jobs/`: format-independent outcome contracts.
- `src/core/editor/`: public editor contracts and narrator conformance policy.
  Callers should import editor types from this package rather than depending on
  internal translator implementation details.
- `src/core/adapters/`: converts format-specific content to and from units.
- `src/core/common/`, `src/core/epub/`, `src/core/subtitle_translator.py`:
  orchestration and structural validation for each format.
- `src/persistence/`: SQLite storage and checkpoint aggregation. Migrations are
  additive so an existing jobs database remains readable.
- `src/api/`: request lifecycle and serialized job state.
- `src/web/static/js/translation/`: client orchestration plus focused view
  helpers such as `job-quality.js`.

## Resume and output identity

A new job chooses its unique output path once and persists it in job config.
Every resume reuses that path. Resume code may reclassify an older failed row as
completed/review-required only when a draft is present and current structural
validation accepts it. It must not silently create a second output filename.

## Extension guidance

- Add cross-format result semantics to `src/core/jobs`, not individual loops.
- Add editor policy and diagnostics contracts to `src/core/editor`.
- Keep provider requests and response parsing out of API handlers.
- Persist new checkpoint fields with additive migrations and backward-compatible
  defaults.
- Keep dynamic frontend labels in locale files for all seven locales and make
  their render path react to `localeChanged`.
- Put release verification in CI before packaging; source archives should be
  generated from Git so tracked files cannot be omitted by an allowlist.

`src/core/translator.py` remains the compatibility surface for translation,
reflection parsing, and refinement. Future refactors should extract one
cohesive service at a time (request execution, editor parsing, then refinement)
behind the contracts above, keeping its public imports stable until downstream
callers and tests have migrated.
