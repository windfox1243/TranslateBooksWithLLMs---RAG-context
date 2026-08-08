"""
Regression tests for issue #207 (EPUB content-loss cluster).

A. OPF dc:language must be a real ISO 639-1 code, not the first two letters
   of the language *name* ("Chinese" -> "zh", not "ch").
B. Percent-encoded manifest hrefs ("Chapter%201.xhtml") must be unquoted
   before resolving to a filesystem path, otherwise the chapter is skipped
   and ships untranslated.
C. Plain Text Mode must not silently delete table cell text or
   figure/picture-wrapped images.
"""
import os
import tempfile

import pytest
from lxml import etree

from src.core.epub.plain_extractor import (
    extract_plain_paragraphs,
    replace_body_with_paragraphs,
)
from src.core.epub.translator import _precount_chunks, _update_epub_metadata

# === A. dc:language metadata ===

OPF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:test-207</dc:identifier>
    <dc:title>Test Book</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest/>
  <spine/>
</package>
"""


def _run_update_metadata(target_language: str) -> str:
    """Run _update_epub_metadata on a minimal OPF and return dc:language."""
    with tempfile.TemporaryDirectory() as tmp:
        opf_path = os.path.join(tmp, "content.opf")
        with open(opf_path, "w", encoding="utf-8") as f:
            f.write(OPF_TEMPLATE)

        opf_tree = etree.parse(opf_path)
        _update_epub_metadata(opf_tree, opf_path, target_language)

        reparsed = etree.parse(opf_path)
        lang_el = reparsed.getroot().find(
            ".//{http://purl.org/dc/elements/1.1/}language"
        )
        assert lang_el is not None
        return lang_el.text


class TestOpfLanguageCode:
    @pytest.mark.parametrize(
        "language,expected",
        [
            ("Chinese", "zh"),
            ("German", "de"),
            ("Dutch", "nl"),
            ("Greek", "el"),
            ("Japanese", "ja"),
            ("French", "fr"),
        ],
    )
    def test_language_name_resolves_to_iso_code(self, language, expected):
        assert _run_update_metadata(language) == expected

    def test_locale_string_resolves_to_base_code(self):
        assert _run_update_metadata("en-US") == "en"

    def test_unresolvable_language_leaves_metadata_unchanged(self):
        # Same policy as the XHTML lang pass (lang_support): when the target
        # cannot be resolved, do not write a bogus code.
        assert _run_update_metadata("Klingon") == "en"


# === B. Percent-encoded manifest hrefs ===

XHTML_DOC = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>Chapter</title></head>
  <body>
    <p>This paragraph must be counted and translated like any other chapter
    content, even though the manifest href is percent-encoded.</p>
  </body>
</html>
"""


class TestPercentEncodedHrefs:
    @pytest.mark.asyncio
    async def test_precount_finds_percent_encoded_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(
                os.path.join(tmp, "Chapter 1.xhtml"), "w", encoding="utf-8"
            ) as f:
                f.write(XHTML_DOC)

            total_chunks, chunks_per_file = await _precount_chunks(
                content_files=["Chapter%201.xhtml"],
                opf_dir=tmp,
                max_tokens_per_chunk=500,
            )

        assert chunks_per_file[0] > 0, (
            "percent-encoded href was not resolved: chapter counted as 0 "
            "chunks, meaning it would ship untranslated"
        )
        assert total_chunks > 0

    @pytest.mark.asyncio
    async def test_precount_plain_href_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(
                os.path.join(tmp, "chapter1.xhtml"), "w", encoding="utf-8"
            ) as f:
                f.write(XHTML_DOC)

            total_chunks, chunks_per_file = await _precount_chunks(
                content_files=["chapter1.xhtml"],
                opf_dir=tmp,
                max_tokens_per_chunk=500,
            )

        assert chunks_per_file[0] > 0
        assert total_chunks > 0


# === C. Plain Text Mode: tables and figures ===

XHTML_NS = "http://www.w3.org/1999/xhtml"


def _parse_body(body_inner: str) -> etree._Element:
    doc = f"""<html xmlns="{XHTML_NS}"><body>{body_inner}</body></html>"""
    root = etree.fromstring(doc.encode("utf-8"))
    return root.find(f"{{{XHTML_NS}}}body")


def _all_imgs(element: etree._Element) -> list:
    return [el for el in element.iter() if el.tag.split("}")[-1] == "img"]


class TestPlainModeTables:
    def test_table_cell_text_is_extracted(self):
        body = _parse_body(
            "<table>"
            "<caption>Population by city</caption>"
            "<tr><th>City</th><th>Population</th></tr>"
            "<tr><td>Paris</td><td>2.1 million</td></tr>"
            "</table>"
        )
        paragraphs, _, _ = extract_plain_paragraphs(body)
        joined = " ".join(paragraphs)

        assert "Paris" in joined, "table cell text was deleted"
        assert "2.1 million" in joined
        assert "Population by city" in joined

    def test_table_text_survives_rebuild(self):
        body = _parse_body(
            "<p>Intro.</p>"
            "<table><tr><td>Cell text</td></tr></table>"
        )
        paragraphs, tags, images = extract_plain_paragraphs(body)
        replace_body_with_paragraphs(body, paragraphs, tags, images)

        rebuilt_text = " ".join("".join(body.itertext()).split())
        assert "Cell text" in rebuilt_text


class TestPlainModeFigures:
    def test_figure_wrapped_image_is_anchored(self):
        body = _parse_body(
            "<p>Before the figure.</p>"
            '<figure><img src="images/map.png" alt="Map"/>'
            "<figcaption>A map of the region</figcaption></figure>"
        )
        paragraphs, tags, images_by_paragraph = extract_plain_paragraphs(body)

        anchored = [img for imgs in images_by_paragraph.values() for img in imgs]
        assert any(
            img.get("src") == "images/map.png" for img in anchored
        ), "figure-wrapped <img> was deleted instead of anchored"

    def test_figcaption_text_is_extracted(self):
        body = _parse_body(
            '<figure><img src="x.png"/><figcaption>The caption</figcaption></figure>'
        )
        paragraphs, _, _ = extract_plain_paragraphs(body)
        assert "The caption" in " ".join(paragraphs)

    def test_picture_wrapped_image_is_anchored(self):
        body = _parse_body(
            "<p>Some text.</p>"
            '<picture><source srcset="big.webp"/><img src="fallback.jpg"/></picture>'
        )
        _, _, images_by_paragraph = extract_plain_paragraphs(body)

        anchored = [img for imgs in images_by_paragraph.values() for img in imgs]
        assert any(img.get("src") == "fallback.jpg" for img in anchored)

    def test_inline_figure_image_inside_paragraph_is_anchored(self):
        body = _parse_body(
            '<p>Text around <figure><img src="inline.png"/></figure> an inline figure.</p>'
        )
        _, _, images_by_paragraph = extract_plain_paragraphs(body)

        anchored = [img for imgs in images_by_paragraph.values() for img in imgs]
        assert any(img.get("src") == "inline.png" for img in anchored)

    def test_figure_image_survives_rebuild(self):
        body = _parse_body(
            "<p>Para.</p>"
            '<figure><img src="kept.png"/><figcaption>Cap</figcaption></figure>'
        )
        paragraphs, tags, images = extract_plain_paragraphs(body)
        replace_body_with_paragraphs(body, paragraphs, tags, images)

        srcs = [img.get("src") for img in _all_imgs(body)]
        assert "kept.png" in srcs

    def test_standalone_image_still_anchored(self):
        # Pre-existing behavior that must not regress.
        body = _parse_body('<p>Hello.</p><img src="standalone.png"/>')
        _, _, images_by_paragraph = extract_plain_paragraphs(body)

        anchored = [img for imgs in images_by_paragraph.values() for img in imgs]
        assert any(img.get("src") == "standalone.png" for img in anchored)

    def test_svg_still_dropped(self):
        # SVG cannot be re-anchored as <img>; it stays out of Plain Text Mode.
        body = _parse_body(
            "<p>Text.</p><svg xmlns='http://www.w3.org/2000/svg'><text>chart</text></svg>"
        )
        paragraphs, _, images_by_paragraph = extract_plain_paragraphs(body)
        assert "chart" not in " ".join(paragraphs)
        assert not images_by_paragraph


class TestWeakLlmSafety:
    """
    Plain Text Mode's contract: the LLM only ever sees plain text, never a
    tag, so even a small model (gemma3:4b style) cannot break the formatting.
    These tests assert the contract still holds after tables/figures stopped
    being dropped, and that image reattachment is purely mechanical (it does
    not depend on the quality of the LLM output).
    """

    BODY_INNER = (
        "<h1>Chapter One</h1>"
        "<p>Intro paragraph.</p>"
        "<table><caption>Stats</caption>"
        "<tr><th>City</th><th>Population</th></tr>"
        "<tr><td>Paris</td><td>2.1 million</td></tr></table>"
        '<figure><img src="images/map.png"/><figcaption>A map</figcaption></figure>'
        "<p>Closing paragraph.</p>"
    )

    @pytest.mark.asyncio
    async def test_llm_never_receives_tags(self, monkeypatch):
        import src.core.common.plain_text_pipeline as plain_pipeline

        sent_to_llm = []

        async def capture_llm(*, main_content, **kwargs):
            sent_to_llm.append(main_content)
            return main_content  # identity "translation"

        monkeypatch.setattr(plain_pipeline, "generate_translation_request", capture_llm)
        monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

        body = _parse_body(self.BODY_INNER)
        paragraphs, _, _ = extract_plain_paragraphs(body)
        await plain_pipeline.translate_paragraphs_plain(
            paragraphs=paragraphs,
            source_language="English",
            target_language="French",
            model_name="m",
            llm_client=object(),
            max_tokens_per_chunk=1000,
        )

        assert sent_to_llm, "nothing was sent to the LLM"
        blob = "\n".join(sent_to_llm)
        assert "<" not in blob and ">" not in blob, (
            "Plain Text Mode leaked markup to the LLM"
        )
        assert "img" not in blob and "src=" not in blob
        # The new content is there, as plain text only.
        assert "Paris" in blob
        assert "A map" in blob

    @pytest.mark.asyncio
    async def test_images_survive_a_sloppy_llm(self, monkeypatch):
        # A weak model that merges every paragraph into one blob and rewrites
        # text freely: the worst realistic output. Images must still be
        # reattached because anchoring never goes through the LLM.
        import src.core.common.plain_text_pipeline as plain_pipeline

        async def sloppy_llm(*, main_content, **kwargs):
            merged = " ".join(main_content.split())
            return "Traduction approximative: " + merged

        monkeypatch.setattr(plain_pipeline, "generate_translation_request", sloppy_llm)
        monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

        body = _parse_body(self.BODY_INNER)
        paragraphs, tags, images = extract_plain_paragraphs(body)
        translated, _, interrupted = await plain_pipeline.translate_paragraphs_plain(
            paragraphs=paragraphs,
            source_language="English",
            target_language="French",
            model_name="m",
            llm_client=object(),
            max_tokens_per_chunk=1000,
        )
        assert not interrupted

        replace_body_with_paragraphs(body, translated, tags, images)

        srcs = [img.get("src") for img in _all_imgs(body)]
        assert srcs == ["images/map.png"], (
            "image reattachment must not depend on LLM output quality"
        )
        # Output is valid flat XHTML: only known block tags at body level.
        allowed = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"}
        for child in body:
            assert child.tag.split("}")[-1] in allowed

    @pytest.mark.asyncio
    async def test_hallucinated_tags_are_stripped_from_llm_output(self, monkeypatch):
        # Small models sometimes invent markup (e.g. <sup>1</sup> around a
        # footnote number). The source never contains tags in Plain Text Mode,
        # so they must be stripped before the text is written back.
        import src.core.common.plain_text_pipeline as plain_pipeline

        async def hallucinating_llm(*, main_content, **kwargs):
            import re
            parts = re.split(r"\n{2,}", main_content)
            return "\n\n".join(
                p.replace("note 1", "note <sup>1</sup>") if p.strip() else p
                for p in parts
            )

        monkeypatch.setattr(plain_pipeline, "generate_translation_request", hallucinating_llm)
        monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

        translated, _, _ = await plain_pipeline.translate_paragraphs_plain(
            paragraphs=["Some paragraph with note 1 in it.", "Another paragraph."],
            source_language="English",
            target_language="French",
            model_name="m",
            llm_client=object(),
            max_tokens_per_chunk=1000,
        )

        joined = " ".join(translated)
        assert "<sup>" not in joined and "</sup>" not in joined
        assert "note 1" in joined, "inner text of the hallucinated tag must be kept"

    def test_markup_stripping_spares_legitimate_source_angle_brackets(self):
        from src.core.common.plain_text_pipeline import strip_hallucinated_markup

        # Source chunk contains real '<' (a code sample from a <pre> block):
        # the output is left untouched, even if it looks tag-like.
        source = "Run <make install> then check a < b in the loop."
        translated = "Lancez <make install> puis verifiez a < b dans la boucle."
        assert strip_hallucinated_markup(translated, source) == translated

        # Tag-free source: invented tags go away, their content stays.
        assert strip_hallucinated_markup(
            "Le 1<sup>er</sup> chapitre<br/>", "The 1st chapter"
        ) == "Le 1er chapitre"
