"""
Unit tests for the pure section of src/core/epub/cjk_typography.py.

Covers the language gate, CJK font-stack detection/mapping, the declaration
rewriter (including its masking rules), and the two structural invariants the
regex-based approach relies on: brace structure is never altered, and the
rewrite is idempotent.
"""
from pathlib import Path

import pytest

from src.core.epub.cjk_typography import (
    CJK_SCRIPT_CODES,
    NEUTRALIZATIONS,
    collect_font_face_families,
    contains_cjk_font_token,
    has_cjk_typography_features,
    is_cjk_language,
    map_cjk_font_to_generic,
    neutralize_css_text,
    neutralize_style_attribute,
    normalize_script_language,
    should_normalize_script,
)

FIXTURE_CSS = Path(__file__).resolve().parents[2] / "fixtures" / "cjk_epub" / "main.css"


@pytest.fixture(scope="module")
def real_stylesheet() -> str:
    return FIXTURE_CSS.read_text(encoding="utf-8")


# Every CSS sample below is run through the brace-structure and idempotency
# invariants. Keep new samples added here too.
CSS_SAMPLES = {
    "empty": "",
    "clean_latin": (
        "body { font-family: Georgia, serif; line-height: 1.6; text-indent: 1em; }\n"
        "p { text-align: justify; }\n"
    ),
    "quoted_cjk_family": 'p { font-family: "宋体"; }\n',
    "vertical_japanese": (
        "html { -epub-writing-mode: vertical-rl; writing-mode: vertical-rl; }\n"
        "body { text-orientation: upright; line-break: strict; word-break: break-all; }\n"
        "p { text-combine-upright: all; text-justify: inter-ideograph; }\n"
    ),
    "commented_out_declaration": (
        "/* p { font-family: 宋体; line-height: 120% } */\n"
        "p { color: red; }\n"
    ),
    "content_string": 'p { content: "font-family: 宋体"; line-height: 130%; }\n',
    "data_url": (
        "p { background: url(data:image/png;base64,iVBORw0KGgo=); line-height: 120%; }\n"
    ),
    "media_query": '@media screen { p { font-family: "宋体"; text-indent: 2em; } }\n',
    "font_face_only": (
        "@font-face { font-family: \"MyKai\"; src: url(../Fonts/zdy2.ttf); }\n"
        "h1 { font-family: \"MyKai\", sans-serif; }\n"
    ),
    "important_declarations": "p { line-height: 130% !important; font-family: SimSun !important; }\n",
}


# ---------------------------------------------------------------------------
# Language gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("language,expected", [
    ("Chinese", "zh"),
    ("chinese (traditional)", "zh"),
    ("Chinese (Simplified)", "zh"),
    ("zh", "zh"),
    ("zh-CN", "zh"),
    ("zh-TW", "zh"),
    ("zh-Hans", "zh"),
    ("zh-Hant", "zh"),
    ("cmn", "zh"),
    ("yue", "zh"),
    ("Japanese", "ja"),
    ("ja", "ja"),
    ("jpn", "ja"),
    ("ja-JP", "ja"),
    ("Korean", "ko"),
    ("ko", "ko"),
    ("kor", "ko"),
    ("French", "fr"),
    ("fr", "fr"),
    ("", None),
    (None, None),
    ("  ", None),
    ("Klingon", None),
])
def test_normalize_script_language(language, expected):
    assert normalize_script_language(language) == expected


@pytest.mark.parametrize("language", ["Javanese", "javanese", "jav", "Kongo", "kok", "Korowai"])
def test_long_language_names_do_not_misfire_on_two_letter_prefixes(language):
    """'Javanese' starts with 'ja' and 'Kongo' with 'ko', but neither is CJK.

    Prefix matching only applies to code-shaped input, and then only to exact
    primary subtags, so ISO codes like 'jav' (Javanese) and 'kok' (Konkani) are
    not captured either.
    """
    assert normalize_script_language(language) not in CJK_SCRIPT_CODES


@pytest.mark.parametrize("language,expected", [
    ("Chinese", True),
    ("zh-Hant", True),
    ("Japanese", True),
    ("Korean", True),
    ("French", False),
    ("Javanese", False),
    ("", False),
    (None, False),
])
def test_is_cjk_language(language, expected):
    assert is_cjk_language(language) is expected


# ---------------------------------------------------------------------------
# Font detection and mapping
# ---------------------------------------------------------------------------

def test_collect_font_face_families(real_stylesheet):
    assert collect_font_face_families(real_stylesheet) == frozenset(
        {"aajlkscdzk (non-commercial use)"}
    )


def test_collect_font_face_families_ignores_commented_out_rules():
    css = '/* @font-face { font-family: "Ghost"; } */ @font-face { font-family: "Real"; }'
    assert collect_font_face_families(css) == frozenset({"real"})


def test_collect_font_face_families_on_stylesheet_without_font_face():
    assert collect_font_face_families("p { color: red }") == frozenset()


@pytest.mark.parametrize("value,expected", [
    ('"宋体"', True),
    ("DK-SONGTI, st, \"宋体\", zw, sans-serif", True),
    ("SimSun, serif", True),
    ("Microsoft YaHei", True),
    ("MS Gothic", True),
    ("Malgun Gothic", True),
    ("Hiragino Kaku Gothic Pro", True),
    ("ゴシック", True),
    ("바탕", True),
    ("Georgia, serif", False),
    ("Century Gothic, sans-serif", False),
    ("serif", False),
    ("sans-serif", False),
    ("monospace", False),
    ("", False),
])
def test_contains_cjk_font_token(value, expected):
    assert contains_cjk_font_token(value) is expected


def test_contains_cjk_font_token_uses_font_face_suspect_list():
    families = frozenset({"aajlkscdzk (non-commercial use)"})
    value = '"AaJLKSCDZK (Non-Commercial Use)", sans-serif'
    assert contains_cjk_font_token(value) is False
    assert contains_cjk_font_token(value, families) is True


def test_font_face_suspect_list_is_an_accepted_false_positive():
    """Documented risk (plan R1), scoped to an already-gated CJK book.

    A font embedded by a CJK-authored book is in practice a CJK face or subset,
    so membership in the @font-face list is treated as a CJK signal by the
    *transform* even when the family name looks Latin.

    It is deliberately NOT evidence of a CJK source: embedding a font is
    ordinary in a Latin-script book, so `has_cjk_typography_features` ignores
    rule (c). The false positive is therefore confined to books the language
    gate (or another, unambiguous signal) already established as CJK-authored.
    """
    css = '@font-face { font-family: "Playfair Display"; src: url(pf.ttf); }\n' \
          'h1 { font-family: "Playfair Display", serif; }\n'
    out, counts = neutralize_css_text(css)
    assert counts == {"font-family": 1}
    assert "h1 { font-family: serif; }" in out

    # ... but on its own it does not turn the pass on.
    assert has_cjk_typography_features(css) is False
    assert should_normalize_script("English", "French", [css]) is False
    # A CJK source language does gate it in, and then the embedded face goes.
    assert should_normalize_script("Chinese", "French", [css]) is True


@pytest.mark.parametrize("value,expected", [
    ("DK-SONGTI, st, \"宋体\", zw, sans-serif", "serif"),
    ('"AaJLKSCDZK (Non-Commercial Use)", "楷体", sans-serif', "serif"),
    ("SimSun", "serif"),
    ("MS Mincho", "serif"),
    ("Batang", "serif"),
    ("FangSong", "serif"),
    ("STSong", "serif"),
    ("Noto Serif CJK SC", "serif"),
    ("明朝", "serif"),
    ("SimHei", "sans-serif"),
    ("Heiti SC", "sans-serif"),
    ("Microsoft YaHei", "sans-serif"),
    ("Microsoft JhengHei", "sans-serif"),
    ("Malgun Gothic", "sans-serif"),
    ("MS Gothic", "sans-serif"),
    ("Dotum", "sans-serif"),
    ("Gulim", "sans-serif"),
    ("PingFang SC", "sans-serif"),
    ("Meiryo", "sans-serif"),
    ("DengXian", "sans-serif"),
    ("Noto Sans CJK JP", "sans-serif"),
    ("黑体", "sans-serif"),
    ("ゴシック", "sans-serif"),
    ("고딕", "sans-serif"),
    ("SimSun-Mono", "monospace"),
    ("Courier New", "monospace"),
    ("UnknownCjkFace", "serif"),
    ("", "serif"),
])
def test_map_cjk_font_to_generic(value, expected):
    assert map_cjk_font_to_generic(value) == expected


def test_latin_font_stack_declaration_is_untouched():
    css = "h1 { font-family: Century Gothic, sans-serif; }\n"
    out, counts = neutralize_css_text(css)
    assert out == css
    assert counts == {}


# ---------------------------------------------------------------------------
# Golden test on the real stylesheet
# ---------------------------------------------------------------------------

def test_golden_real_stylesheet(real_stylesheet):
    out, counts = neutralize_css_text(real_stylesheet)

    # The @charset rule must survive as the very first line.
    assert out.splitlines()[0] == '@charset "utf-8";'

    # The @font-face block declares a family NAME: never rewritten.
    assert 'font-family: "AaJLKSCDZK (Non-Commercial Use)";\n  src: url(../Fonts/zdy2.ttf);' in out

    # p rule: CJK stack -> serif, 2em indent -> 1.5em, 135% leading -> 1.5.
    assert "font-family: serif;" in out
    assert "text-indent: 1.5em;" in out
    assert "line-height: 1.5;" in out
    assert "宋体" not in out
    assert "DK-SONGTI" not in out

    # h1.head and h3 reference the @font-face family AND a quoted CJK family
    # ("楷体"), so both stacks are replaced by the serif generic (the 楷 marker
    # classifies the stack, the trailing sans-serif fallback does not).
    assert out.count("font-family: serif;") == 3
    assert 'font-family: "AaJLKSCDZK (Non-Commercial Use)", "楷体", sans-serif;' not in out
    # 楷体 survives only in the Chinese comment /**楷体引文**/, never in a
    # font-family declaration (it appears three times in the source).
    assert real_stylesheet.count("楷体") == 3
    assert out.count("楷体") == 1
    assert "/**楷体引文**/" in out

    # h3's 150% leading is already comfortable and stays; the 0 indents have no
    # relative unit and stay too.
    assert "line-height: 150%;" in out
    assert out.count("text-indent: 0;") == 2

    # Untouched properties keep their exact original text.
    assert "margin-top: 6%;" in out
    assert "border-left: 0.2em /*竖线粗细*/ solid #780700 /*竖线颜色*/;" in out
    assert "color: #BA2213;" in out

    # Chinese comments are preserved verbatim.
    assert "/*行间距*/" in out
    assert "/*————————————————————通用————————————————————*/" in out

    assert counts == {"font-family": 3, "text-indent": 1, "line-height": 3}


def test_golden_real_stylesheet_declares_cjk_features(real_stylesheet):
    assert has_cjk_typography_features(real_stylesheet) is True


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def test_declaration_inside_comment_is_not_rewritten():
    css = CSS_SAMPLES["commented_out_declaration"]
    out, counts = neutralize_css_text(css)
    assert out == css
    assert counts == {}


def test_content_string_is_not_rewritten_but_neighbour_declaration_is():
    out, counts = neutralize_css_text(CSS_SAMPLES["content_string"])
    assert 'content: "font-family: 宋体";' in out
    assert "line-height: 1.5;" in out
    assert counts == {"line-height": 1}


def test_quoted_cjk_family_is_still_detected_through_the_string_mask():
    """A masked `"宋体"` must not hide the CJK family from the detector."""
    out, counts = neutralize_css_text(CSS_SAMPLES["quoted_cjk_family"])
    assert out == "p { font-family: serif; }\n"
    assert counts == {"font-family": 1}


def test_url_with_semicolon_does_not_break_value_boundaries():
    out, counts = neutralize_css_text(CSS_SAMPLES["data_url"])
    assert "url(data:image/png;base64,iVBORw0KGgo=)" in out
    assert "line-height: 1.5;" in out
    assert counts == {"line-height": 1}


def test_declarations_inside_media_query_are_rewritten():
    out, counts = neutralize_css_text(CSS_SAMPLES["media_query"])
    assert "font-family: serif;" in out
    assert "text-indent: 1.5em;" in out
    assert counts == {"font-family": 1, "text-indent": 1}


def test_font_face_family_name_is_never_rewritten_but_references_are():
    out, counts = neutralize_css_text(CSS_SAMPLES["font_face_only"])
    assert '@font-face { font-family: "MyKai"; src: url(../Fonts/zdy2.ttf); }' in out
    assert "h1 { font-family: serif; }" in out
    assert counts == {"font-family": 1}


def test_comment_between_declarations_does_not_hide_the_next_one():
    css = "p { text-align: justify;\n  /* indent */\n  text-indent: 2em; }\n"
    out, counts = neutralize_css_text(css)
    assert "/* indent */" in out
    assert "text-indent: 1.5em;" in out
    assert counts == {"text-indent": 1}


def test_important_flag_is_preserved():
    out, counts = neutralize_css_text(CSS_SAMPLES["important_declarations"])
    assert "line-height: 1.5 !important;" in out
    assert "font-family: serif !important;" in out
    assert counts == {"font-family": 1, "line-height": 1}


# ---------------------------------------------------------------------------
# The neutralization table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("declaration,expected", [
    # text-indent: relative units above the threshold only.
    ("text-indent: 2em", "text-indent: 1.5em"),
    ("text-indent: 2rem", "text-indent: 1.5em"),
    ("text-indent: 2ch", "text-indent: 1.5em"),
    ("text-indent: 200%", "text-indent: 150%"),
    ("text-indent: 1.5em", "text-indent: 1.5em"),
    ("text-indent: 1em", "text-indent: 1em"),
    ("text-indent: 0", "text-indent: 0"),
    ("text-indent: 24px", "text-indent: 24px"),
    ("text-indent: 12pt", "text-indent: 12pt"),
    ("text-indent: 5mm", "text-indent: 5mm"),
    ("text-indent: 150%", "text-indent: 150%"),
    # line-height: normalized ratio below 1.4 only.
    ("line-height: 130%", "line-height: 1.5"),
    ("line-height: 1.3", "line-height: 1.5"),
    ("line-height: 1.2em", "line-height: 1.5"),
    ("line-height: 1.35rem", "line-height: 1.5"),
    ("line-height: 1.4", "line-height: 1.4"),
    ("line-height: 160%", "line-height: 160%"),
    ("line-height: normal", "line-height: normal"),
    ("line-height: 18px", "line-height: 18px"),
    ("line-height: 14pt", "line-height: 14pt"),
    # writing-mode / text-orientation.
    ("writing-mode: vertical-rl", "writing-mode: horizontal-tb"),
    ("writing-mode: vertical-lr", "writing-mode: horizontal-tb"),
    ("writing-mode: tb-rl", "writing-mode: tb-rl"),
    ("writing-mode: horizontal-tb", "writing-mode: horizontal-tb"),
    ("text-orientation: upright", "text-orientation: mixed"),
    ("text-orientation: mixed", "text-orientation: mixed"),
    # text-combine.
    ("text-combine-upright: all", "text-combine-upright: none"),
    ("text-combine: horizontal", "text-combine: none"),
    ("text-combine-upright: none", "text-combine-upright: none"),
    # word-break / line-break / text-justify.
    ("word-break: break-all", "word-break: normal"),
    ("word-break: keep-all", "word-break: normal"),
    ("word-break: break-word", "word-break: break-word"),
    ("line-break: strict", "line-break: auto"),
    ("line-break: auto", "line-break: auto"),
    ("text-justify: inter-ideograph", "text-justify: auto"),
    ("text-justify: inter-character", "text-justify: auto"),
    ("text-justify: distribute", "text-justify: auto"),
    ("text-justify: inter-word", "text-justify: inter-word"),
    ("text-justify: auto", "text-justify: auto"),
    # Out-of-scope properties are never touched.
    ("ruby-position: over", "ruby-position: over"),
    ("ruby-align: center", "ruby-align: center"),
    ("hanging-punctuation: allow-end", "hanging-punctuation: allow-end"),
    ("punctuation-trim: start", "punctuation-trim: start"),
    ("text-spacing: ideograph-alpha", "text-spacing: ideograph-alpha"),
    ("text-autospace: ideograph-numeric", "text-autospace: ideograph-numeric"),
])
def test_single_declaration_rewriting(declaration, expected):
    css = "p { %s; }" % declaration
    out, _counts = neutralize_css_text(css)
    assert out == "p { %s; }" % expected


@pytest.mark.parametrize("prefix", ["", "-epub-", "-webkit-"])
def test_vendor_prefixed_forms_are_rewritten_and_counted_under_the_base_name(prefix):
    css = "html { %swriting-mode: vertical-rl; }" % prefix
    out, counts = neutralize_css_text(css)
    assert out == "html { %swriting-mode: horizontal-tb; }" % prefix
    assert counts == {"writing-mode": 1}


def test_unknown_vendor_prefix_is_not_matched():
    css = "html { -moz-writing-mode: vertical-rl; }"
    out, counts = neutralize_css_text(css)
    assert out == css
    assert counts == {}


def test_neutralization_table_is_the_closed_list_from_the_plan():
    assert [prop for prop, _ in NEUTRALIZATIONS] == [
        "font-family",
        "text-indent",
        "line-height",
        "writing-mode",
        "text-orientation",
        "text-combine-upright",
        "text-combine",
        "word-break",
        "line-break",
        "text-justify",
    ]


def test_vertical_japanese_stylesheet_is_fully_neutralized():
    out, counts = neutralize_css_text(CSS_SAMPLES["vertical_japanese"])
    assert "vertical" not in out
    assert out.count("horizontal-tb") == 2
    assert "text-orientation: mixed" in out
    assert "line-break: auto" in out
    assert "word-break: normal" in out
    assert "text-combine-upright: none" in out
    assert "text-justify: auto" in out
    assert counts == {
        "writing-mode": 2,
        "text-orientation": 1,
        "line-break": 1,
        "word-break": 1,
        "text-combine-upright": 1,
        "text-justify": 1,
    }


# ---------------------------------------------------------------------------
# Inline style attributes
# ---------------------------------------------------------------------------

def test_neutralize_style_attribute():
    out, counts = neutralize_style_attribute("font-family:宋体;line-height:120%")
    assert out == "font-family:serif;line-height:1.5"
    assert counts == {"font-family": 1, "line-height": 1}


def test_neutralize_style_attribute_leaves_latin_declarations_alone():
    value = "font-family:Georgia, serif;line-height:1.6;color:#333"
    out, counts = neutralize_style_attribute(value)
    assert out == value
    assert counts == {}


def test_neutralize_style_attribute_on_empty_value():
    assert neutralize_style_attribute("") == ("", {})


def test_empty_declaration_value_is_left_alone():
    css = "p { line-break: ; }"
    out, counts = neutralize_css_text(css)
    assert out == css
    assert counts == {}


# ---------------------------------------------------------------------------
# Structural invariants, over every sample
# ---------------------------------------------------------------------------

def _all_samples(real_css: str):
    samples = dict(CSS_SAMPLES)
    samples["real_stylesheet"] = real_css
    return samples


def test_brace_structure_is_never_altered(real_stylesheet):
    for name, css in _all_samples(real_stylesheet).items():
        out, _counts = neutralize_css_text(css)
        assert out.count("{") == css.count("{"), name
        assert out.count("}") == css.count("}"), name
        assert out.count(";") == css.count(";"), name


def test_neutralize_css_text_is_idempotent(real_stylesheet):
    for name, css in _all_samples(real_stylesheet).items():
        once, first_counts = neutralize_css_text(css)
        twice, second_counts = neutralize_css_text(once)
        assert twice == once, name
        assert second_counts == {}, name
        if not first_counts:
            assert once == css, name


def test_has_cjk_typography_features(real_stylesheet):
    assert has_cjk_typography_features(real_stylesheet) is True
    assert has_cjk_typography_features(CSS_SAMPLES["vertical_japanese"]) is True
    assert has_cjk_typography_features(CSS_SAMPLES["quoted_cjk_family"]) is True
    assert has_cjk_typography_features(CSS_SAMPLES["clean_latin"]) is False
    assert has_cjk_typography_features(CSS_SAMPLES["commented_out_declaration"]) is False
    assert has_cjk_typography_features("") is False


# Every entry of CJK_EVIDENCE, each in isolation.
@pytest.mark.parametrize("css", [
    'p { font-family: "宋体"; }',
    "p { font-family: SimSun, serif; }",
    "p { font-family: Microsoft YaHei; }",
    "html { writing-mode: vertical-rl; }",
    "html { -epub-writing-mode: vertical-lr; }",
    "p { text-orientation: upright; }",
    "p { text-combine-upright: all; }",
    "p { text-combine: horizontal; }",
    "p { word-break: break-all; }",
    "p { word-break: keep-all; }",
    "p { line-break: strict; }",
    "p { text-justify: inter-ideograph; }",
    "p { text-justify: distribute; }",
])
def test_unambiguous_cjk_signals_are_evidence(css):
    assert has_cjk_typography_features(css) is True


# Latin-plausible declarations that the transform still normalizes but that must
# never, on their own, decide that a book is CJK-authored.
@pytest.mark.parametrize("css", [
    "body { line-height: 1.3; }",
    "body { line-height: 135%; }",
    "p { text-indent: 2em; }",
    "p { text-indent: 200%; }",
    "p { word-break: normal; }",
    "p { text-justify: auto; }",
    "p { font-family: Century Gothic, sans-serif; }",
])
def test_latin_plausible_declarations_are_not_evidence(css):
    """They stay in NEUTRALIZATIONS; they are just not admissible as evidence."""
    assert has_cjk_typography_features(css) is False


def test_tight_leading_and_wide_indent_do_not_trigger_a_latin_to_latin_job():
    """Regression: French -> English on an ordinary Latin stylesheet.

    `line-height: 1.3` and `text-indent: 2em` are normal Latin book typography.
    When they counted as evidence, this job silently rewrote the book's leading
    and its indents.
    """
    css = ("body { font-family: Georgia, serif; line-height: 1.3 }"
           "p { text-indent: 2em }")
    assert has_cjk_typography_features(css) is False
    assert should_normalize_script("French", "English", [css]) is False


def test_embedded_latin_display_face_does_not_trigger_a_latin_to_latin_job():
    """Regression: English -> French on a book embedding its own display face.

    When rule (c) counted as evidence, this job dropped the embedded face and
    replaced it with a generic.
    """
    css = ('@font-face { font-family: "MyLatinDisplay"; src: url(a.otf) }'
           'h1 { font-family: "MyLatinDisplay", serif }')
    assert has_cjk_typography_features(css) is False
    assert should_normalize_script("English", "French", [css]) is False


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

CJK_CSS = ['p { font-family: "宋体"; }']
CLEAN_CSS = ["p { font-family: Georgia, serif; }"]


@pytest.mark.parametrize("source,target,css_texts,expected", [
    ("Chinese", "French", CLEAN_CSS, True),
    ("Chinese", "Japanese", CJK_CSS, False),
    ("French", "Chinese", CJK_CSS, False),
    (None, "French", CJK_CSS, True),
    (None, "French", CLEAN_CSS, False),
    ("French", "English", CLEAN_CSS, False),
    # A target that cannot be resolved does not disable the pass.
    ("Chinese", "Klingon", CLEAN_CSS, True),
    (None, None, CJK_CSS, True),
    # Mislabelled source, CJK stylesheet.
    ("English", "French", CJK_CSS, True),
    # CJK source with no stylesheets at all.
    ("zh-Hant", "French", [], True),
    ("Japanese", "Korean", CJK_CSS, False),
])
def test_should_normalize_script(source, target, css_texts, expected):
    assert should_normalize_script(source, target, css_texts) is expected


def test_should_normalize_script_accepts_any_iterable():
    assert should_normalize_script(None, "French", iter(CJK_CSS)) is True
