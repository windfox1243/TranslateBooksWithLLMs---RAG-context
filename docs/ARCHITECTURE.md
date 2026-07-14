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
3. The shared unit-context pipeline extracts dialogue and social cues, commits
   validated source facts, and retrieves only scene-relevant accepted state.
4. `src.core.translator` communicates with the configured LLM and returns a
   draft. The Senior Editor can refine it, but editor availability does not
   determine whether a structurally valid draft exists.
5. Format validation decides whether the draft can be rebuilt into the output.
6. The checkpoint manager stores each unit and updates aggregate job progress.
7. The API exposes execution and quality state independently; the frontend
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
- `src/core/context/`: typed source-analysis results, relevant prompt bundles,
  generalized social-evidence normalization, and pair-level reconciliation.
- `src/core/adapters/`: converts format-specific content to and from units.
- `src/core/common/`, `src/core/epub/`, `src/core/subtitle_translator.py`:
  orchestration and structural validation for each format.
- `src/persistence/repositories/`: narrow job, editor, structured-context, and
  narrator interfaces. `Database` remains the compatibility facade while SQL
  ownership moves behind these boundaries incrementally.
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

## Context authority and evidence lifecycle

SQLite is authoritative for active addressing and relationship state. Markdown
dynamic sections remain import/export surfaces for older jobs and human review.
Contract-v5 source analysis stores accepted facts before drafting the current
unit. Evidence that is ambiguous, incomplete, or unsupported remains retained
with an open resolution state, but prompt projection reads only accepted edges
and active rules. Pair-level reconciliation runs after new evidence, accepted
relationship changes, and manual locks; it promotes a provisional target pair
only when the language profile's required hierarchy and gender inputs are known.

Social hierarchy, institutional rank, and chronological age are independent.
For example, an explicit senior title can prove conversational seniority while
age remains unknown. Production resolvers operate on grounded spans and
normalized language-profile cues and never contain title-specific character
names.

## Editor request boundaries

The editor compatibility entrypoint delegates to `EditorService`. Deterministic
adapter, narrator, glossary, proper-name, and phrase-aware residue checks run
before model repair. Full-unit review is the default after relevant-context
retrieval, with no application-level token ceiling or default windowing. The
initial audit always receives the complete filtered source and draft; hashes in
the request diagnostics prove that neither was truncated. Selective retrieval
is always active for contract v5, even when no optional provider-capacity budget
is configured, and the typed prompt bundle is the only structured-context
injection for that contract.

Narrator validation uses paragraph and line offsets plus dialogue, thought, and
letter masking instead of proportional sentence alignment. Conclusive locked or
active narration mismatches are patched at their exact target offsets before a
model call; ambiguous discourse evidence remains advisory. Exact local issues
receive one bounded locator correction containing only invalid issue IDs and
candidate neighborhoods. An unsupported model-only locator remains a diagnostic
warning, while a failed deterministic locator requires review. Only structural,
completeness, or cross-cutting consistency defects permit a full rewrite.
Repeated validation fingerprints stop the loop. A separately configured
escalation client may receive one final attempt, but escalation is disabled by
default.

Editor usage is reported on two independent scales. Request telemetry exposes
the largest individual request and its stage, while compatibility totals remain
cumulative across reflection, locator correction, local repair, and any full
rewrite. Prompt-composition diagnostics separate fixed instructions, source,
draft, selected context, glossary, deterministic findings, and critique so a
large cumulative total cannot be mistaken for one oversized request.

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
reflection parsing, and refinement. Editor execution now enters through
`EditorService`, context preparation through `src/core/context`, and persistence
through narrow repositories. Further extraction should keep these public
imports stable until downstream callers and tests have migrated.
