"""Tests for deterministic, chapter-aware translation boundaries."""

from src.core.chunking.chapter_detector import (
    find_chapter_ranges,
    is_chapter_heading,
)
from src.core.chunking.decorative_separator import is_decorative_separator
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


def test_repeated_unknown_language_label_with_title_needs_heading_separator():
    paragraphs = [
        "Bölüm 1: First chapter",
        "First chapter body.",
        "Bölüm 2: Second chapter",
        "Second chapter body.",
    ]
    ranges = find_chapter_ranges(paragraphs)

    assert [(item.start, item.end) for item in ranges] == [(0, 2), (2, 4)]


def test_repeated_unknown_language_roman_label_requires_heading_punctuation():
    paragraphs = [
        "Bölüm I: The beginning",
        "First chapter body.",
        "Bölüm II: The sequel",
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


def test_decorative_separators_are_not_chapter_boundaries():
    paragraphs = [
        "Chapter 1",
        "Opening body.",
        "===",
        "More body.",
        "----",
        "Final body.",
    ]

    assert is_decorative_separator("===")
    assert is_decorative_separator("----")
    assert not is_chapter_heading("===")
    assert not is_chapter_heading("----")

    ranges = find_chapter_ranges(paragraphs)
    assert [(item.start, item.end) for item in ranges] == [(0, 6)]


def test_repeated_english_prose_with_i_is_not_a_roman_heading():
    paragraphs = [
        "Episode 1: Opening",
        "That's why I gritted my teeth.",
        "Normal story text.",
        "But I can endure it.",
        "If I don't give up someday, a path will open up.",
        "That's why I smiled too, but...",
        "More normal story text.",
    ]
    ranges = find_chapter_ranges(paragraphs)

    assert [(item.start, item.end) for item in ranges] == [(0, 7)]


def test_repeated_prose_numbers_are_not_generic_headings():
    paragraphs = [
        "Episode 1: Opening",
        "If I.",
        "If I.",
        "If I....",
        "I...",
        "I.....",
        "It's been 1 year.",
        "It's been 10 years.",
        "It's been 100 years.",
        "More normal story text.",
    ]
    ranges = find_chapter_ranges(paragraphs)

    assert [(item.start, item.end) for item in ranges] == [(0, 10)]


def test_repeated_numbered_titles_still_work():
    paragraphs = [
        "1. First chapter",
        "First chapter body.",
        "2. Second chapter",
        "Second chapter body.",
    ]
    ranges = find_chapter_ranges(paragraphs)

    assert [(item.start, item.end) for item in ranges] == [(0, 2), (2, 4)]


def test_plain_segments_do_not_split_on_repeated_roman_pronoun_prose():
    paragraphs = [
        "Episode 1: Opening",
        "That's why I gritted my teeth.",
        "Normal story text.",
        "But I can endure it.",
        "If I don't give up someday, a path will open up.",
        "That's why I smiled too, but...",
        "More normal story text.",
    ]
    segments = build_plain_segments(
        paragraphs,
        max_tokens_per_chunk=1000,
        chapter_mode=True,
    )

    assert len(segments) == 1


def test_txt_chapter_mode_splits_oversized_chapter_within_same_chapter():
    body = "\n\n".join(
        f"Paragraph {i} has enough ordinary story text to force token chunking inside one chapter."
        for i in range(1, 16)
    )
    chunks = split_text_into_chunks(
        f"Chapter 1\n\n{body}",
        max_tokens_per_chunk=45,
        chapter_mode=True,
    )

    assert len(chunks) > 1
    assert {chunk["chapter_index"] for chunk in chunks} == {0}
    assert {chunk["chapter_title"] for chunk in chunks} == {"Chapter 1"}
    assert [chunk["chunk_in_chapter"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk["chunks_in_chapter"] == len(chunks) for chunk in chunks)


def test_txt_chapter_mode_does_not_create_separator_only_chunks():
    text = (
        "Chapter 1\n\n"
        "Alpha one. Alpha two.\n\n"
        "===\n\n"
        "Beta one. Beta two.\n\n"
        "----\n\n"
        "Gamma one. Gamma two."
    )

    chunks = split_text_into_chunks(
        text,
        max_tokens_per_chunk=8,
        chapter_mode=True,
    )

    assert len(chunks) > 1
    assert {chunk["chapter_index"] for chunk in chunks} == {0}
    assert not any(
        is_decorative_separator(chunk["main_content"])
        for chunk in chunks
    )
    assert any("===" in chunk["main_content"] for chunk in chunks)
    assert any("----" in chunk["main_content"] for chunk in chunks)


def test_plain_segments_split_oversized_structural_chapter_without_fake_ranges():
    paragraphs = ["Episode 1: Opening"] + [
        f"Paragraph {i} has enough ordinary story text to force token chunking inside one structural chapter."
        for i in range(1, 16)
    ]
    kinds = ["h1"] + ["p"] * 15
    segments = build_plain_segments(
        paragraphs,
        max_tokens_per_chunk=45,
        paragraph_kinds=kinds,
        chapter_mode=True,
    )

    assert len(segments) > 1
    assert {segment["chapter_index"] for segment in segments} == {0}
    assert {segment["chapter_title"] for segment in segments} == {"Episode 1: Opening"}
    assert all(not segment["partial"] for segment in segments)


def test_plain_segments_preserve_separators_without_standalone_units():
    paragraphs = [
        "Chapter 1",
        "Alpha one. Alpha two.",
        "===",
        "Beta one. Beta two.",
        "----",
        "Gamma one. Gamma two.",
    ]

    segments = build_plain_segments(
        paragraphs,
        max_tokens_per_chunk=8,
        chapter_mode=True,
    )

    assert len(segments) > 1
    assert {segment["chapter_index"] for segment in segments} == {0}
    assert not any(
        is_decorative_separator(segment["text"])
        for segment in segments
    )
    assert any("===" in segment["text"] for segment in segments)
    assert any("----" in segment["text"] for segment in segments)


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


def test_chapter_mode_may_create_more_units_to_preserve_short_chapter_boundaries():
    text = (
        "Chapter 1\n\nAlpha opening.\n\n"
        "Chapter 2\n\nBeta opening."
    )

    normal_chunks = split_text_into_chunks(
        text,
        max_tokens_per_chunk=1000,
        chapter_mode=False,
    )
    chapter_chunks = split_text_into_chunks(
        text,
        max_tokens_per_chunk=1000,
        chapter_mode=True,
    )

    assert len(normal_chunks) == 1
    assert len(chapter_chunks) == 2
    assert "Chapter 1" in chapter_chunks[0]["main_content"]
    assert "Chapter 2" not in chapter_chunks[0]["main_content"]
    assert "Chapter 2" in chapter_chunks[1]["main_content"]
    assert "Chapter 1" not in chapter_chunks[1]["main_content"]


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


def test_llm_chapter_boundary_verification():
    from src.core.chunking.chapter_detector import is_chapter_heading, find_chapter_ranges

    class DummyLLMClient:
        def make_request(self, prompt: str):
            if "Side Story 4" in prompt:
                return "YES, this is a chapter heading."
            return "NO"

    client = DummyLLMClient()
    assert is_chapter_heading("Side Story 4: The Beginning", llm_client=client) is True
    assert is_chapter_heading("This is normal prose text.", llm_client=client) is False

    ranges = find_chapter_ranges(
        ["Opening text", "Side Story 4: The Beginning", "Chapter text"],
        llm_client=client,
    )
    assert len(ranges) == 2
    assert ranges[1].title == "Side Story 4: The Beginning"
