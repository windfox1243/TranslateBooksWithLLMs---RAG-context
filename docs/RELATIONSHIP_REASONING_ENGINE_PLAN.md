# Full Relationship Reasoning Engine Plan

## Implementation Status

The beta engine is implemented. `project` mode is the default so accepted,
scene-relevant graph facts are injected into live translation, reflection, and
repair prompts. `shadow` remains available for audit-only runs, and `off`
disables graph extraction and projection. Chunk-derived candidates are staged
until translation and adapter validation succeed.

## Goal

Build a structured relationship reasoning engine that can explain, validate,
and project character relationships across translation chunks without relying
on brittle markdown-only heuristics or unchecked LLM claims. The engine should
be useful for every supported language while still allowing language-specific
addressing behavior where the target language needs it.

## Scope

- Canonicalize relationship facts between characters, groups, and named roles.
- Keep directed speaker-addressee addressing as a specialized relationship
  view, backed by the existing DB-addressing tables.
- Separate character identity aliases from object, weapon, item, skill, spell,
  title, and glossary terminology.
- Track evidence, source chunk, confidence, provenance, locks, and rejection
  reasons for every inferred relationship.
- Project only scene-relevant relationship facts into prompts with explicit
  selection reasons.
- Provide deterministic validation first, with optional LLM assistance only
  for ambiguous evidence classification.

## Non-Goals

- Do not replace the human-editable markdown novel context.
- Do not make LLM inference the final authority over locked user rules.
- Do not make Vietnamese relationship rules the fallback for unknown or custom
  target languages.
- Do not infer private or unsupported relationships from name similarity alone.

## Core Data Model

### RelationshipGraph

Create an internal graph keyed by canonical character ids:

- `CharacterNode`: canonical name, aliases, source labels, language/script
  profile, entity type, lock state.
- `RelationshipEdge`: source id, target id, relationship type, direction,
  polarity, register/formality, hierarchy, intimacy, temporal scope, evidence,
  confidence, provenance, and status.
- `AddressingEdge`: speaker id, addressee id, self-reference, second-person
  form, vocative, register, language profile, scope, evidence, confidence,
  lock state, and audit metadata.
- `EvidenceSpan`: chunk index, source quote, translated quote when available,
  dialogue turn id, extraction layer, parser status, and source file id.
- `ConflictRecord`: conflicting edge ids, validator name, severity, decision,
  and remediation hint.

### Relationship Types

Start with a closed internal enum, then allow controlled extension:

- family: parent, child, sibling, spouse, partner, relative.
- social: friend, rival, enemy, ally, mentor, student, superior, subordinate.
- institutional: commander, subordinate, colleague, client, servant, master.
- narrative: protagonist, narrator, point-of-view, disguise, temporary role.
- uncertain: candidate relationship that requires more evidence.

## Evidence Pipeline

1. Collect candidate facts from markdown context, DB addressing, dialogue
   attribution, glossary/object classifiers, source text, and LLM context
   updates.
2. Normalize names through the shared text matching policy.
3. Reject non-character endpoints before relationship merge.
4. Require evidence for new edges unless they come from trusted manual context
   or locked DB state.
5. Attach every accepted, rejected, or quarantined fact to an audit record.
6. Defer persistence for chunk-derived updates until the translation unit
   succeeds and adapter validation passes.

## Merge Policy

- Manual locks always win.
- Existing high-confidence durable facts win over temporary scene facts.
- New durable facts require direct source evidence or trusted markdown/manual
  provenance.
- Symmetric relationships must be type-compatible on both directions.
- Asymmetric relationships must preserve directionality. For example, parent
  and child, superior and subordinate, mentor and student, and speaker and
  addressee cannot be silently swapped.
- Register or hierarchy jumps require strong evidence and should be
  quarantined when they contradict recent dialogue attribution.
- Unknown/custom target languages use neutral merge rules, then project
  language-specific behavior only when a profile explicitly supports it.

## Validators

Add deterministic validators in this order:

1. Entity type validator: characters only for relationship and identity edges.
2. Alias validator: no substring activation, no similar-name conflation, no
   object or weapon identity aliases.
3. Evidence validator: quote, apposition, explicit relationship wording, or
   trusted manual source is required for new durable edges.
4. Direction validator: checks reverse pair compatibility and speaker-addressee
   alignment against dialogue attribution.
5. Language profile validator: applies target-language requirements such as
   paired address forms, honorific compatibility, formality, and RTL/no-space
   matching boundaries.
6. Temporal scope validator: prevents disguise, roleplay, dream, flashback, or
   quoted historical context from overwriting durable present-time facts.
7. Conflict validator: quarantines impossible hierarchy flips, sudden register
   jumps, or relationship contradictions until confirmed.

## Prompt Projection

- Build a single relationship projection service used by translation,
  reflection, repair, subtitle, EPUB/XHTML, DOCX/plain, and refine-only paths.
- Project compact facts:
  - active characters and aliases;
  - active relationship edges;
  - DB-directed addressing rules;
  - temporary scene role constraints;
  - rejected or forbidden aliases that are relevant to the current chunk.
- Log each projected fact with a selection reason.
- Keep markdown context as fallback display and compatibility input, not the
  only reasoning source.

## LLM Integration

- Keep deterministic validators authoritative.
- Ask the LLM only for structured candidate facts:

```json
{
  "relationships": [
    {
      "source": "Character A",
      "target": "Character B",
      "relationship_type": "mentor",
      "direction": "directed",
      "scope": "durable",
      "evidence_quote": "...",
      "confidence": 0.82
    }
  ]
}
```

- Never accept malformed JSON as no-op without logging.
- Retry once with a strict repair prompt, then quarantine parse failures.
- Feed Senior Editor reflection the same active relationship projection used
  by translation and repair.

## Storage and Migration

- Reuse the existing DB-addressing tables for `AddressingEdge`.
- Add relationship tables behind a feature flag:
  - `context_relationship_nodes`
  - `context_relationship_edges`
  - `context_relationship_evidence`
  - `context_relationship_conflicts`
- Import existing markdown relationships into the graph at job start, resume,
  and context resync.
- Export accepted graph facts back into markdown snapshots for compatibility.
- Keep REST routes backward-compatible; add graph inspection, quarantine,
  delete, and lock routes as internal beta APIs first.

## Observability

Every user-important relationship action should log a concise terminal event:

- candidate extracted;
- candidate rejected;
- candidate quarantined;
- edge accepted;
- edge superseded;
- lock respected;
- projection selected;
- projection skipped;
- repair/reflection conflict found;
- adapter validation blocked persistence.

Logs must include chunk index, source file id when available, pair, validator,
decision, confidence, and short evidence excerpts. Logs must not dump full book
text, secrets, or complete prompts.

## Test Matrix

- Character identity: `Tom` vs `Tomio`, `Alex` vs `Alles`, CJK labels,
  Hangul labels, RTL names, Thai/no-space names, accented Latin names.
- Object safety: `Gram` as a sword, weapon-to-weapon terminology, skills,
  spells, artifacts, titles, UI/system terms.
- Relationship direction: parent/child, superior/subordinate, mentor/student,
  master/servant, speaker/addressee, reverse-pair compatibility.
- Language profiles: English, French, Spanish, German, Vietnamese, Chinese,
  Japanese, Korean, Arabic, Russian, Hindi, Thai, and unknown/custom.
- Lifecycle: failed chunks, retries, resume, context resync, markdown import,
  DB export, manual locks, delete/quarantine.
- Adapters: TXT, EPUB/XHTML, DOCX/plain, subtitles, generic translation, and
  refine-only alignment.

## Runtime Modes

1. `project` is the beta default. Accepted graph facts are projected into live
   prompts, while markdown remains the compatibility fallback when graph state
   is empty.
2. `shadow` imports, validates, logs, and audits candidates without changing
   prompt output.
3. `off` skips relationship graph extraction and projection.

The remaining rollout work is a reactive UI inspector for graph conflicts and
manual correction. The internal beta REST routes already support graph reads,
pair audits, lock/unlock, quarantine, and delete operations.

## Release Criteria

- Full targeted and regression test suites pass.
- No prompt path depends on markdown-only relationship parsing when graph data
  is available.
- Bad chunk output cannot persist relationship facts.
- Every accepted relationship edge has provenance and evidence or a trusted
  manual source.
- Every rejected or quarantined edge has an audit reason.
- Unknown/custom languages remain neutral and do not inherit Vietnamese rules.
