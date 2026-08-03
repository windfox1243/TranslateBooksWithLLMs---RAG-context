# EPUB Script-Transition Normalization

When a CJK-authored EPUB (Chinese/Japanese/Korean) is translated to a Latin (or
other non-CJK) target, the *text* is translated but the *presentation and
packaging* stay CJK by default: the stylesheet still selects CJK font stacks
and CJK paragraph/leading conventions, and the OPF/NCX metadata still carries
the source-language title. This page documents the two passes that fix that,
implemented in `src/core/epub/cjk_typography.py` and
`src/core/epub/metadata_translator.py`.

---

## Typography normalization

`apply_script_normalization_to_epub_directory` rewrites offending CSS
declarations **in place** (never by injecting `!important` overrides — see the
module docstring for why) across every stylesheet, every `<style>` element and
every inline `style` attribute in the extracted EPUB.

### Gate

The pass runs only when **both** are true:

- the target language is not CJK (`zh`/`ja`/`ko`), **and**
- either the declared source language is CJK, **or** at least one stylesheet
  carries unambiguous CJK evidence.

"Unambiguous evidence" is a closed, narrower list than the properties actually
rewritten: a CJK/romanized-CJK font family, vertical writing mode,
`text-orientation`, `text-combine[-upright]`, `word-break: break-all|keep-all`,
`line-break`, or an ideographic `text-justify`. `line-height` and `text-indent`
are deliberately **not** evidence — a tight leading or a 2em indent is ordinary
Latin book typography, and treating them as evidence used to fire the whole
pass on a plain French→English book. Once the gate says yes, the transform
still runs in full: leading, indent and `@font-face`-suspect font stacks are
all normalized inside a book already established as CJK-authored, even though
none of those three counted as evidence to get there.

### What gets rewritten (closed list)

| Property (`-epub-`/`-webkit-` prefixed forms included) | Rewritten to |
|---|---|
| `font-family` (CJK/romanized-CJK token, or a family declared by an embedded `@font-face`) | the generic family alone (`serif`/`sans-serif`/`monospace`) |
| `text-indent` (`em`/`rem`/`ch` > 1.5, or `%` > 150) | `1.5em` / `150%` |
| `line-height` (ratio < 1.4) | `1.5` |
| `writing-mode` (contains `vertical`) | `horizontal-tb` |
| `text-orientation` | `mixed` |
| `text-combine-upright`, `text-combine` | `none` |
| `word-break` (`break-all`/`keep-all`) | `normal` |
| `line-break` | `auto` |
| `text-justify` (`inter-ideograph`/`inter-character`/`distribute`) | `auto` |

`ruby-position`/`ruby-align` are left untouched (harmless without `<ruby>`
content), and `punctuation-trim`/`hanging-punctuation`/`text-spacing`/
`text-autospace` are deliberately out of scope — no reader applies them
harmfully to Latin text. The `font` shorthand is not rewritten either: only the
properties above are in scope, by design.

The pass also fixes two structural artefacts no stylesheet edit can reach:
OPF `<meta>` font overrides (`duokan-body-font`, `duokan-title-font`,
`duokan-font-family` — a closed set) are removed, and a `page-progression-direction="rtl"`
left over from a vertical CJK layout is reset to `ltr` when the target is not
itself an RTL language. Every NCX beside the OPF gets its `xml:lang` updated to
the target language.

---

## Packaging metadata localization

`translate_opf_metadata` translates the OPF `dc:title` and `dc:description`
and propagates the accepted title to every `*.ncx` `docTitle/text`.

- **One LLM call for both fields**, using the same client/model the pipeline
  already built — no new provider, no retry, no fallback.
- **`dc:creator` is never touched.** Author names are neither translated nor
  transliterated.
- A description longer than 4000 characters is **not sent at all** (most CJK
  web-novel descriptions duplicate the already-translated intro page); the
  title is still translated in that case.
- The pipeline's attribution signature is stripped from the description before
  the request and re-appended verbatim afterward, exactly once.
- A candidate translation is **rejected** — keeping the original — when it is
  empty/whitespace-only, still contains CJK characters while the target isn't
  CJK, is more than ~4x the source length (runaway output), or — for the title
  only — spans more than one line. A rejected field never reaches the NCX: a
  title is never invented.

---

## Configuration

| Flag | Default | Effect when `false` |
|---|---|---|
| `EPUB_SCRIPT_NORMALIZATION_ENABLED` | `true` | No CSS/OPF/NCX typography rewriting; output byte-identical to the pre-normalization pipeline. |
| `EPUB_TRANSLATE_METADATA_ENABLED` | `true` | No extra LLM call; `dc:title`/`dc:description`/NCX `docTitle` stay in the source language. |

Both passes are individually try/except-guarded and can never fail an
otherwise-successful translation job; a failure is logged and the job
continues.

---

## Limitations

- **The cover raster** (e.g. `cover.jpg`) keeps its source-language title baked
  into the pixels — not fixable by this pipeline.
- **CJK filenames** (e.g. `封面.html`) are kept as-is; renaming them would
  require rewriting every manifest href and NCX `content/@src` for no
  reader-visible benefit.
- **Embedded CJK fonts are kept, not deleted.** Only the *references* from
  text selectors are neutralized, so the manifest stays valid; the font bytes
  become dead weight. The total size is logged, nothing is removed.
- **`dc:creator` is never translated or transliterated**, by design.
- **Japanese `<ruby>`/furigana stripping is out of scope.** Real content
  concern, but distinct from script-transition packaging fidelity.
