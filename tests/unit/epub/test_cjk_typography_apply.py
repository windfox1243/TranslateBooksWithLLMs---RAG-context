"""
Unit tests for the apply section of src/core/epub/cjk_typography.py.

Covers stylesheet I/O (encoding detection and round-tripping), the container
walk, the OPF/NCX structural fixes, and the four properties the pass must hold
whatever the input: the gate touches nothing when it says no, untouched files
keep their exact bytes, XHTML survives re-serialization intact, and a broken
file never fails the pass.

The container fixtures reproduce the shape of the reported book (a Chinese EPUB
produced by "Ag2S EpubLib" with duokan reader metadata); their markup fragments
are verbatim from it.
"""
from pathlib import Path

import pytest

from src.core.epub.cjk_typography import (
    OPF_FONT_OVERRIDE_METAS,
    apply_script_normalization_to_epub_directory,
    read_css_text,
    write_css_text,
)
from tests.unit.epub.conftest import (
    CONTENT_OPF,
    COVER_XHTML,
    REAL_CSS,
    TOC_NCX,
    _build_cjk_epub_dir,
    _write,
)

# `cjk_epub_dir` itself is a fixture defined in conftest.py -- pytest injects it
# automatically, no import needed.


def _snapshot(root: Path) -> dict:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


# ---------------------------------------------------------------------------
# Stylesheet I/O
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "gb18030"])
def test_read_css_text_detects_self_describing_or_first_guess_encodings(tmp_path, encoding):
    # UTF-8 is self-validating, the BOM is explicit, and gb18030 is the first
    # CJK codec tried, so all three are recovered exactly.
    css = 'p { font-family: "本"; }\n'
    path = _write(tmp_path / "main.css", css, encoding)
    text, detected = read_css_text(str(path))
    assert text == css
    assert detected == encoding


@pytest.mark.parametrize("encoding", ["gb18030", "big5", "shift_jis", "euc-kr"])
def test_read_css_text_honours_charset_declaration(tmp_path, encoding):
    css = f'@charset "{encoding}";\np {{ font-family: "本"; }}\n'
    path = _write(tmp_path / "main.css", css, encoding)
    text, detected = read_css_text(str(path))
    assert detected == encoding
    assert text == css


def test_undeclared_big5_is_decoded_by_the_first_codec_that_accepts_it(tmp_path):
    """Documented limitation: guessing between CJK codecs is ambiguous.

    Big5/Shift_JIS/EUC-KR bytes are almost always also valid gb18030, which is
    tried first, so an undeclared non-Simplified stylesheet can be mojibaked.
    The decode order is fixed by the plan, and this is harmless for this pass:
    every substitution it makes is ASCII, and the surrounding bytes are written
    back through the same codec they were read with, so the file round-trips.
    """
    css = 'p { font-family: "本"; }\n'
    path = _write(tmp_path / "main.css", css, "big5")
    text, encoding = read_css_text(str(path))
    assert encoding == "gb18030"
    assert text != css
    # Round-trip safety is what actually matters.
    write_css_text(str(path), text, encoding)
    assert path.read_bytes() == css.encode("big5")


def test_read_css_text_bom_wins_over_charset_declaration(tmp_path):
    css = '@charset "gb18030";\np { color: red }\n'
    path = _write(tmp_path / "main.css", css, "utf-8-sig")
    text, encoding = read_css_text(str(path))
    assert encoding == "utf-8-sig"
    assert text == css


def test_read_css_text_never_raises_on_undecodable_bytes(tmp_path):
    path = tmp_path / "main.css"
    # Invalid in UTF-8 and in every CJK codec tried, so the lossy read wins.
    path.write_bytes(b"p { color: red } \xff\xfe\x81\x40\xff")
    text, encoding = read_css_text(str(path))
    assert "p { color: red }" in text
    assert encoding


def test_write_css_text_round_trips_in_the_source_encoding(tmp_path):
    path = _write(tmp_path / "main.css", 'p { font-family: "宋体"; }\n', "gb18030")
    text, encoding = read_css_text(str(path))
    write_css_text(str(path), text.replace("宋体", "楷体"), encoding)
    assert "楷体" in path.read_bytes().decode("gb18030")
    assert read_css_text(str(path))[1] == "gb18030"


def test_write_css_text_falls_back_to_utf8_and_rewrites_charset(tmp_path):
    path = tmp_path / "main.css"
    # 'ascii' cannot encode the value, so the fallback path must trigger.
    write_css_text(str(path), '@charset "ascii";\np { content: "宋" }\n', "ascii")
    raw = path.read_bytes()
    assert raw.decode("utf-8").startswith('@charset "utf-8";')
    assert "宋" in raw.decode("utf-8")


# ---------------------------------------------------------------------------
# Criterion 1 — end to end on the real container shape
# ---------------------------------------------------------------------------

def test_end_to_end_on_the_real_container(cjk_epub_dir):
    result = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "French")

    assert result == {
        "applied": True,
        "css_files_rewritten": 1,
        "style_elements_rewritten": 1,
        "style_attributes_rewritten": 1,
        # main.css: 3 font-family + 1 text-indent + 3 line-height;
        # intro.xhtml <style>: 1 font-family + 1 line-height;
        # the inline style: 1 font-family + 1 line-height.
        "changes_by_property": {"font-family": 5, "text-indent": 1, "line-height": 5},
        "opf_metas_removed": 1,
        "progression_direction_reset": False,
        "ncx_lang_updated": 1,
        "embedded_font_bytes": 4096,
        "encoding_fallbacks": 0,
        "errors": 0,
    }

    css = (cjk_epub_dir / "OEBPS" / "Styles" / "main.css").read_text(encoding="utf-8")
    assert css.startswith('@charset "utf-8";')
    assert "font-family: serif;" in css
    assert "text-indent: 1.5em;" in css
    assert "line-height: 1.5;" in css
    assert "宋体" not in css
    # The @font-face family NAME survives, so the manifest stays coherent.
    assert 'font-family: "AaJLKSCDZK (Non-Commercial Use)";' in css

    intro = (cjk_epub_dir / "OEBPS" / "Text" / "intro.xhtml").read_text(encoding="utf-8")
    assert "p.quote { font-family: serif; line-height: 1.5; }" in intro
    # The rewriter substitutes the value only: the declaration's own spacing
    # (here, no space after the colon) is preserved exactly as authored.
    assert 'style="font-family:serif;line-height:1.5"' in intro
    # A style attribute with nothing to neutralize is left exactly as authored.
    assert 'style="margin-bottom:2em;"' in intro

    opf = (cjk_epub_dir / "OEBPS" / "content.opf").read_text(encoding="utf-8")
    assert "duokan-body-font" not in opf
    # Only the closed set goes; every other meta and the metadata survive.
    assert 'name="cover"' in opf
    assert 'name="generator"' in opf
    assert 'name="calibre:title_sort"' in opf
    assert "<dc:language>zh</dc:language>" in opf
    assert 'href="Fonts/zdy2.ttf"' in opf

    ncx = (cjk_epub_dir / "OEBPS" / "toc.ncx").read_text(encoding="utf-8")
    assert 'xml:lang="fr"' in ncx
    # docTitle is Phase 6's job (it needs the translated title).
    assert "<text>被渣后和前夫破镜重圆了</text>" in ncx

    # Fonts are reported, never deleted.
    assert (cjk_epub_dir / "OEBPS" / "Fonts" / "zdy2.ttf").exists()


def test_end_to_end_logs_through_the_callback(cjk_epub_dir):
    events = []
    apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "French",
        log_callback=lambda event, message: events.append((event, message)))
    # Nothing went wrong, so nothing is logged: the caller (Phase 5) owns the
    # success reporting.
    assert events == []


# ---------------------------------------------------------------------------
# Criteria 2 and 3 — the gate
# ---------------------------------------------------------------------------

def test_cjk_target_leaves_every_file_byte_identical(cjk_epub_dir):
    before = _snapshot(cjk_epub_dir)
    result = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "Japanese")

    assert result["applied"] is False
    assert result == {
        "applied": False,
        "css_files_rewritten": 0,
        "style_elements_rewritten": 0,
        "style_attributes_rewritten": 0,
        "changes_by_property": {},
        "opf_metas_removed": 0,
        "progression_direction_reset": False,
        "ncx_lang_updated": 0,
        "embedded_font_bytes": 0,
        "encoding_fallbacks": 0,
        "errors": 0,
    }
    assert _snapshot(cjk_epub_dir) == before


def test_unknown_source_with_cjk_css_is_gated_in(cjk_epub_dir):
    result = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), None, "French")
    assert result["applied"] is True
    assert result["css_files_rewritten"] == 1


def test_unknown_source_with_latin_css_is_gated_out(tmp_path):
    latin = ("body { font-family: Georgia, serif; line-height: 1.3 }\n"
             "p { text-indent: 2em; }\n")
    root = _build_cjk_epub_dir(tmp_path / "epub", latin)
    # The CJK markup fixtures would gate this in through their <style> block, so
    # replace them with Latin equivalents to isolate the stylesheet signal.
    _write(root / "OEBPS" / "Text" / "intro.xhtml",
           '<?xml version="1.0" encoding="utf-8"?>\n'
           '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
           '<style type="text/css">p { line-height: 1.3 }</style></head>'
           '<body><p style="text-indent:2em">Bonjour</p></body></html>\n')
    before = _snapshot(root)

    result = apply_script_normalization_to_epub_directory(str(root), None, "English")

    assert result["applied"] is False
    assert _snapshot(root) == before


# ---------------------------------------------------------------------------
# Criterion 4 — encoding
# ---------------------------------------------------------------------------

def test_gb18030_stylesheet_is_rewritten_in_gb18030(tmp_path):
    root = _build_cjk_epub_dir(tmp_path / "epub", "placeholder {}")
    css_path = root / "OEBPS" / "Styles" / "main.css"
    # The Chinese comment survives the rewrite, which is what makes the
    # write-back encoding observable: a rewritten declaration is pure ASCII.
    original = '/*正文*/\np{font-family:"宋体"}\n'
    css_path.write_bytes(original.encode("gb18030"))

    result = apply_script_normalization_to_epub_directory(
        str(root), "Chinese", "French")

    assert result["css_files_rewritten"] == 1
    expected = '/*正文*/\np{font-family:serif}\n'
    # The bytes really are gb18030, not UTF-8.
    assert css_path.read_bytes() == expected.encode("gb18030")
    assert css_path.read_bytes() != expected.encode("utf-8")
    # And re-reading through the public reader yields the neutralized text.
    text, encoding = read_css_text(str(css_path))
    assert encoding == "gb18030"
    assert "font-family:serif" in text
    assert "宋体" not in text
    assert "/*正文*/" in text


def test_guessed_encoding_is_reported_as_a_fallback(tmp_path):
    root = _build_cjk_epub_dir(tmp_path / "epub", "placeholder {}")
    (root / "OEBPS" / "Styles" / "main.css").write_bytes(
        'p{font-family:"宋体"}\n'.encode("gb18030"))
    result = apply_script_normalization_to_epub_directory(
        str(root), "Chinese", "French")
    assert result["encoding_fallbacks"] == 1


def test_utf8_stylesheet_is_not_reported_as_a_fallback(cjk_epub_dir):
    result = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "French")
    assert result["encoding_fallbacks"] == 0


# ---------------------------------------------------------------------------
# Criterion 5 — XHTML integrity
# ---------------------------------------------------------------------------

def test_xhtml_round_trip_keeps_self_closing_tags_and_namespace_prefixes(cjk_epub_dir):
    apply_script_normalization_to_epub_directory(str(cjk_epub_dir), "Chinese", "French")

    intro = (cjk_epub_dir / "OEBPS" / "Text" / "intro.xhtml").read_text(encoding="utf-8")
    # This is the assertion that guards against repeating rtl_support's
    # method='html' mistake: HTML serialization writes <br></br> and <img ...>.
    assert "<br/>" in intro
    assert '<img src="x.jpg"/>' in intro
    assert "<br></br>" not in intro
    assert intro.startswith("<?xml")

    # The cover page has nothing to neutralize, so it is not even rewritten —
    # but it also carries the SVG/xlink shape, so assert it explicitly.
    cover_path = cjk_epub_dir / "OEBPS" / "Text" / "cover.xhtml"
    assert cover_path.read_text(encoding="utf-8") == COVER_XHTML


def test_xhtml_with_svg_is_reserialized_intact_when_it_does_change(tmp_path):
    root = _build_cjk_epub_dir(tmp_path / "epub", "placeholder {}")
    svg_page = root / "OEBPS" / "Text" / "cover.xhtml"
    svg_page.write_text(
        COVER_XHTML.replace(
            "body { text-align: center; padding:0pt; margin: 0pt; }",
            "body { font-family: 宋体; text-align: center; }"),
        encoding="utf-8")

    result = apply_script_normalization_to_epub_directory(
        str(root), "Chinese", "French")

    assert result["style_elements_rewritten"] >= 1
    out = svg_page.read_text(encoding="utf-8")
    assert "body { font-family: serif; text-align: center; }" in out
    assert 'xmlns:xlink="http://www.w3.org/1999/xlink"' in out
    assert '<image width="1200" height="1600" xlink:href="../Images/cover.jpg"/>' in out
    assert "<!DOCTYPE html>" in out


# ---------------------------------------------------------------------------
# Criterion 6 — vertical Japanese
# ---------------------------------------------------------------------------

VERTICAL_CSS = (
    "html { -epub-writing-mode: vertical-rl; writing-mode: vertical-rl; }\n"
    "body { text-orientation: upright; line-break: strict; }\n"
)

RTL_SPINE_OPF = CONTENT_OPF.replace(
    '<spine toc="ncx">', '<spine toc="ncx" page-progression-direction="rtl">')


def _build_vertical_epub(tmp_path: Path) -> Path:
    root = _build_cjk_epub_dir(tmp_path / "epub", VERTICAL_CSS, RTL_SPINE_OPF)
    # Keep the vertical stylesheet the only style carrier under test.
    _write(root / "OEBPS" / "Text" / "intro.xhtml",
           '<?xml version="1.0" encoding="utf-8"?>\n'
           '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
           '<body><p>ほん</p></body></html>\n')
    return root


def test_vertical_japanese_to_latin_resets_progression_and_writing_mode(tmp_path):
    root = _build_vertical_epub(tmp_path)

    result = apply_script_normalization_to_epub_directory(
        str(root), "Japanese", "French")

    assert result["applied"] is True
    assert result["progression_direction_reset"] is True

    css = (root / "OEBPS" / "Styles" / "main.css").read_text(encoding="utf-8")
    assert "writing-mode: horizontal-tb;" in css
    assert "-epub-writing-mode: horizontal-tb;" in css
    assert "vertical-rl" not in css
    assert "text-orientation: mixed;" in css
    assert "line-break: auto;" in css

    opf = (root / "OEBPS" / "content.opf").read_text(encoding="utf-8")
    assert 'page-progression-direction="ltr"' in opf


def test_vertical_japanese_to_rtl_target_keeps_the_progression(tmp_path):
    root = _build_vertical_epub(tmp_path)

    result = apply_script_normalization_to_epub_directory(
        str(root), "Japanese", "Arabic")

    assert result["applied"] is True
    assert result["progression_direction_reset"] is False
    opf = (root / "OEBPS" / "content.opf").read_text(encoding="utf-8")
    assert 'page-progression-direction="rtl"' in opf
    # The typography pass still ran; only the page turn direction is left alone.
    css = (root / "OEBPS" / "Styles" / "main.css").read_text(encoding="utf-8")
    assert "horizontal-tb" in css


def test_ltr_spine_is_not_touched(cjk_epub_dir):
    result = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "French")
    assert result["progression_direction_reset"] is False
    opf = (cjk_epub_dir / "OEBPS" / "content.opf").read_text(encoding="utf-8")
    assert "page-progression-direction" not in opf


# ---------------------------------------------------------------------------
# Criterion 7 — untouched files keep their bytes
# ---------------------------------------------------------------------------

def test_stylesheet_without_cjk_features_is_not_rewritten(cjk_epub_dir):
    clean = "nav { display: none }\np.latin { font-family: Georgia, serif }\n"
    clean_path = _write(cjk_epub_dir / "OEBPS" / "Styles" / "extra.css", clean)
    before = clean_path.read_bytes()

    result = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "French")

    assert result["css_files_rewritten"] == 1     # main.css only
    assert clean_path.read_bytes() == before


def test_markup_without_style_carriers_is_not_rewritten(cjk_epub_dir):
    plain = ('<?xml version="1.0" encoding="utf-8"?>\n'
             '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
             '<body><p>texte</p></body></html>\n')
    plain_path = _write(cjk_epub_dir / "OEBPS" / "Text" / "plain.xhtml", plain)

    apply_script_normalization_to_epub_directory(str(cjk_epub_dir), "Chinese", "French")

    assert plain_path.read_bytes() == plain.encode("utf-8")


def test_opf_and_ncx_are_not_rewritten_when_nothing_changes(tmp_path):
    opf_without_meta = CONTENT_OPF.replace(
        '    <meta name="duokan-body-font" content="DK-SONGTI"/>\n', "")
    root = _build_cjk_epub_dir(tmp_path / "epub",
                               REAL_CSS.read_text(encoding="utf-8"),
                               opf_without_meta)
    ncx_path = root / "OEBPS" / "toc.ncx"
    _write(ncx_path, TOC_NCX.replace('xml:lang="zh"', 'xml:lang="fr"'))
    opf_before = (root / "OEBPS" / "content.opf").read_bytes()
    ncx_before = ncx_path.read_bytes()

    result = apply_script_normalization_to_epub_directory(str(root), "Chinese", "French")

    assert result["opf_metas_removed"] == 0
    assert result["ncx_lang_updated"] == 0
    assert (root / "OEBPS" / "content.opf").read_bytes() == opf_before
    assert ncx_path.read_bytes() == ncx_before


def test_unresolvable_target_language_leaves_the_ncx_alone(cjk_epub_dir):
    result = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "Klingon")
    assert result["applied"] is True
    assert result["ncx_lang_updated"] == 0
    ncx = (cjk_epub_dir / "OEBPS" / "toc.ncx").read_text(encoding="utf-8")
    assert 'xml:lang="zh"' in ncx


# ---------------------------------------------------------------------------
# Criterion 8 — idempotency
# ---------------------------------------------------------------------------

def test_running_twice_changes_nothing_the_second_time(cjk_epub_dir):
    first = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "French")
    assert first["applied"] is True
    after_first = _snapshot(cjk_epub_dir)

    second = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "French")

    assert second == {
        "applied": True,
        "css_files_rewritten": 0,
        "style_elements_rewritten": 0,
        "style_attributes_rewritten": 0,
        "changes_by_property": {},
        "opf_metas_removed": 0,
        "progression_direction_reset": False,
        "ncx_lang_updated": 0,
        "embedded_font_bytes": 4096,
        "encoding_fallbacks": 0,
        "errors": 0,
    }
    assert _snapshot(cjk_epub_dir) == after_first


def test_running_twice_on_the_vertical_book_is_idempotent(tmp_path):
    root = _build_vertical_epub(tmp_path)
    apply_script_normalization_to_epub_directory(str(root), "Japanese", "French")
    after_first = _snapshot(root)

    second = apply_script_normalization_to_epub_directory(str(root), "Japanese", "French")

    # The stylesheet no longer carries any CJK signal, so only the declared
    # source language still gates the pass in — and it finds nothing to do.
    assert second["applied"] is True
    assert second["progression_direction_reset"] is False
    assert second["changes_by_property"] == {}
    assert _snapshot(root) == after_first


# ---------------------------------------------------------------------------
# Criterion 9 — failure policy
# ---------------------------------------------------------------------------

def test_unreadable_stylesheet_is_counted_and_the_pass_continues(cjk_epub_dir, monkeypatch):
    from src.core.epub import cjk_typography

    broken = _write(cjk_epub_dir / "OEBPS" / "Styles" / "broken.css",
                    "p { color: red } /* unreadable */")
    real_decode = cjk_typography._decode_css_bytes

    def _decode(raw: bytes):
        if b"unreadable" in raw:
            raise OSError("simulated I/O failure")
        return real_decode(raw)

    monkeypatch.setattr(cjk_typography, "_decode_css_bytes", _decode)
    logged = []

    result = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "French",
        log_callback=lambda event, message: logged.append((event, message)))

    assert [event for event, _ in logged] == ["epub_script_norm_error"]

    assert result["errors"] == 1
    # Everything else still happened.
    assert result["applied"] is True
    assert result["css_files_rewritten"] == 1
    assert result["opf_metas_removed"] == 1
    assert result["ncx_lang_updated"] == 1
    assert broken.read_text(encoding="utf-8") == "p { color: red } /* unreadable */"


def test_malformed_markup_never_fails_the_pass(cjk_epub_dir):
    # Unclosed tags, a stray '<', no root element: XML parsing yields no root at
    # all, so the HTML fallback takes over and salvages the document instead of
    # aborting the pass.
    broken = _write(cjk_epub_dir / "OEBPS" / "Text" / "broken.xhtml",
                    "not markup at all < & > <p style='font-family:宋体'>")

    result = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "French")

    assert result["applied"] is True
    assert result["errors"] == 0
    assert result["css_files_rewritten"] == 1
    assert result["ncx_lang_updated"] == 1
    # The salvaged document keeps its text and loses the CJK font, and the
    # ideographs it does carry are not mojibaked by a Latin-1 guess.
    salvaged = broken.read_text(encoding="utf-8")
    assert 'style="font-family:serif"' in salvaged
    assert "not markup at all" in salvaged


def test_malformed_opf_is_counted_and_the_rest_still_runs(cjk_epub_dir, monkeypatch):
    from lxml import etree as real_etree

    from src.core.epub import cjk_typography

    real_parse = real_etree.parse

    def _parse(source, *args, **kwargs):
        if isinstance(source, str) and source.endswith(".opf"):
            raise real_etree.XMLSyntaxError("simulated", 0, 0, 0)
        return real_parse(source, *args, **kwargs)

    monkeypatch.setattr(cjk_typography.etree, "parse", _parse)

    result = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "French")

    assert result["errors"] == 1
    assert result["opf_metas_removed"] == 0
    assert result["css_files_rewritten"] == 1
    assert result["ncx_lang_updated"] == 1


def test_missing_directory_returns_a_zeroed_result(tmp_path):
    result = apply_script_normalization_to_epub_directory(
        str(tmp_path / "does-not-exist"), "Chinese", "French")
    # No stylesheet at all, but a CJK source still gates the pass in; it simply
    # finds nothing, and nothing raises.
    assert result["applied"] is True
    assert result["css_files_rewritten"] == 0
    assert result["errors"] == 0


# ---------------------------------------------------------------------------
# Container edge cases
# ---------------------------------------------------------------------------

def test_only_the_closed_meta_set_is_removed(tmp_path):
    opf = CONTENT_OPF.replace(
        '    <meta name="duokan-body-font" content="DK-SONGTI"/>\n',
        '    <meta name="duokan-body-font" content="DK-SONGTI"/>\n'
        '    <meta name="duokan-title-font" content="DK-KAITI"/>\n'
        '    <meta name="duokan-font-family" content="DK-SONGTI"/>\n'
        '    <meta name="duokan-page-template" content="tpl"/>\n'
        '    <meta name="duokan-gallery" content="g1"/>\n')
    root = _build_cjk_epub_dir(tmp_path / "epub",
                               REAL_CSS.read_text(encoding="utf-8"), opf)

    result = apply_script_normalization_to_epub_directory(str(root), "Chinese", "French")

    assert result["opf_metas_removed"] == 3
    text = (root / "OEBPS" / "content.opf").read_text(encoding="utf-8")
    for name in OPF_FONT_OVERRIDE_METAS:
        assert f'name="{name}"' not in text
    # Not pattern-matched: other duokan hints are harmless and must survive.
    assert 'name="duokan-page-template"' in text
    assert 'name="duokan-gallery"' in text


def test_container_without_opf_still_normalizes_styles(tmp_path):
    root = tmp_path / "epub"
    _write(root / "Styles" / "main.css", 'p { font-family: "宋体"; }\n')
    _write(root / "toc.ncx", TOC_NCX)

    result = apply_script_normalization_to_epub_directory(str(root), "Chinese", "French")

    assert result["css_files_rewritten"] == 1
    assert result["opf_metas_removed"] == 0
    # Without an OPF the NCX is looked for in the whole container.
    assert result["ncx_lang_updated"] == 1


def test_embedded_font_bytes_sums_every_font_format(tmp_path):
    root = _build_cjk_epub_dir(tmp_path / "epub", REAL_CSS.read_text(encoding="utf-8"))
    fonts = root / "OEBPS" / "Fonts"
    (fonts / "a.otf").write_bytes(b"o" * 100)
    (fonts / "b.woff").write_bytes(b"w" * 200)
    (fonts / "c.WOFF2").write_bytes(b"2" * 300)
    (fonts / "notafont.bin").write_bytes(b"x" * 999)

    result = apply_script_normalization_to_epub_directory(str(root), "Chinese", "French")

    assert result["embedded_font_bytes"] == 4096 + 100 + 200 + 300
    assert (fonts / "a.otf").exists()


def test_style_attribute_on_an_svg_element_is_normalized(tmp_path):
    root = _build_cjk_epub_dir(tmp_path / "epub", "placeholder {}")
    # Isolate the cover page as the only style carrier.
    _write(root / "OEBPS" / "Text" / "intro.xhtml",
           '<?xml version="1.0" encoding="utf-8"?>\n'
           '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
           '<body><p>texte</p></body></html>\n')
    page = root / "OEBPS" / "Text" / "cover.xhtml"
    page.write_text(
        COVER_XHTML.replace(
            '<div>', '<div style="font-family:宋体;writing-mode:vertical-rl">'),
        encoding="utf-8")

    result = apply_script_normalization_to_epub_directory(str(root), "Chinese", "French")

    assert result["style_attributes_rewritten"] == 1
    out = page.read_text(encoding="utf-8")
    assert 'style="font-family:serif;writing-mode:horizontal-tb"' in out
    assert 'xlink:href="../Images/cover.jpg"' in out


def test_nested_stylesheets_are_all_visited(tmp_path):
    root = _build_cjk_epub_dir(tmp_path / "epub", REAL_CSS.read_text(encoding="utf-8"))
    _write(root / "OEBPS" / "Styles" / "deep" / "extra.css", "p { text-indent: 2em }\n")

    result = apply_script_normalization_to_epub_directory(str(root), "Chinese", "French")

    assert result["css_files_rewritten"] == 2
    assert "text-indent: 1.5em" in (
        root / "OEBPS" / "Styles" / "deep" / "extra.css").read_text(encoding="utf-8")


def test_result_keys_are_exactly_the_documented_contract(cjk_epub_dir):
    result = apply_script_normalization_to_epub_directory(
        str(cjk_epub_dir), "Chinese", "French")
    assert set(result) == {
        "applied", "css_files_rewritten", "style_elements_rewritten",
        "style_attributes_rewritten", "changes_by_property", "opf_metas_removed",
        "progression_direction_reset", "ncx_lang_updated", "embedded_font_bytes",
        "encoding_fallbacks", "errors",
    }
    assert isinstance(result["applied"], bool)
    assert isinstance(result["progression_direction_reset"], bool)
    assert isinstance(result["changes_by_property"], dict)


def test_the_pure_section_stays_pure():
    """No filesystem or lxml import above the apply-section marker."""
    source = Path(__file__).resolve().parents[3] / "src" / "core" / "epub" / "cjk_typography.py"
    text = source.read_text(encoding="utf-8")
    marker = "# Apply section (filesystem / lxml)"
    pure, _, applied = text.partition(marker)
    assert applied, "apply-section marker not found"
    for forbidden in ("import os", "from lxml", "open(", "os.path"):
        assert forbidden not in pure, forbidden
