"""The refinement pass must match the glossary on its target forms.

Refinement filtered the glossary (source-language keys) against the
target-language draft, so for e.g. ZH->VI nothing ever matched and the refine
pass got no glossary at all. (Ported from upstream df0de763.)
"""

from src.core.translator import _build_chunk_glossary_block

TERMS = {"李明": "Lý Minh", "青云山": "núi Thanh Vân", "王": "Vương"}


def test_zh_to_vi_draft_matches_target_forms():
    draft = "Lý Minh leo lên núi Thanh Vân trước bình minh."
    block = _build_chunk_glossary_block(
        draft, {"glossary_terms": TERMS}, match_target_forms=True,
    )
    assert "李明 -> Lý Minh" in block
    assert "青云山 -> núi Thanh Vân" in block
    assert "Vương" not in block
    assert "KEEP THESE EXACT FORMS" in block


def test_source_key_matching_is_unchanged_for_translation():
    block = _build_chunk_glossary_block("李明走进了房间。", {"glossary_terms": TERMS})
    assert "李明 -> Lý Minh" in block
    assert "REQUIRED TRANSLATIONS" in block


def test_target_form_matching_finds_nothing_in_a_source_text():
    block = _build_chunk_glossary_block(
        "李明走进了房间。", {"glossary_terms": TERMS}, match_target_forms=True,
    )
    assert block == ""
