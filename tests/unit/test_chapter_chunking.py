"""Tests for deterministic, chapter-aware translation boundaries."""

from src.core.chunking.chapter_detector import (
    find_chapter_ranges,
    is_chapter_heading,
)
from src.core.common.plain_text_pipeline import build_plain_segments
from src.core.text_processor import split_text_into_chunks


def test_known_multilingual_chapter_labels_are_detected():
    for heading in (
        "Chapter 12",
        "Chapitre IV",
        "Capítulo 3",
        "Kapitel 7",
        "第十二章 新しい朝",
        "제 4 장 귀환",
    ):
        assert is_chapter_heading(heading)


def test_structural_heading_detection_is_language_independent():
    assert is_chapter_heading("An arbitrary title in any language", "h1")
    assert is_chapter_heading("Another arbitrary title", "heading2")
    assert not is_chapter_heading("Minor subheading", "h4")


def test_repeated_unknown_language_label_is_detected_conservatively():
    paragraphs = [
        "Bölüm 1",
        "First chapter body.",
        "Bölüm 2",
        "Second chapter body.",
    ]
    ranges = find_chapter_ranges(paragraphs)

    assert [(item.start, item.end) for item in ranges] == [(0, 2), (2, 4)]


def test_single_generic_numbered_line_does_not_create_a_boundary():
    paragraphs = [
        "A normal opening paragraph.",
        "Day 1",
        "The story continues without another matching heading.",
    ]
    ranges = find_chapter_ranges(paragraphs)

    assert [(item.start, item.end) for item in ranges] == [(0, 3)]


def test_txt_chapter_mode_never_leaks_neighbor_context():
    text = (
        "Chapter 1\n\nAlpha one. Alpha two.\n\n"
        "Chapter 2\n\nBeta one. Beta two."
    )
    chunks = split_text_into_chunks(
        text,
        max_tokens_per_chunk=8,
        chapter_mode=True,
    )

    for chunk in chunks:
        if "Chapter 1" in chunk["main_content"] or "Alpha" in chunk["main_content"]:
            assert "Beta" not in chunk["context_after"]
        if "Chapter 2" in chunk["main_content"] or "Beta" in chunk["main_content"]:
            assert "Alpha" not in chunk["context_before"]


def test_plain_segments_use_epub_or_docx_heading_kinds():
    paragraphs = ["Opening", "Body A", "Unknown-language title", "Body B"]
    kinds = ["h1", "p", "h1", "p"]
    segments = build_plain_segments(
        paragraphs,
        max_tokens_per_chunk=1000,
        paragraph_kinds=kinds,
        chapter_mode=True,
    )

    assert len(segments) == 2
    assert segments[0]["indices"] == [0, 1]
    assert segments[1]["indices"] == [2, 3]
