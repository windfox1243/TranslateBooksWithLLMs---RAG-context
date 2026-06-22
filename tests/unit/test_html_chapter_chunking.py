"""Chapter-boundary tests for placeholder-based EPUB/DOCX chunking."""

from src.core.epub.html_chunker import HtmlChunker


def test_html_chapter_mode_never_merges_across_heading_boundaries():
    text = (
        "[id0]Chapter One[id1]"
        "[id2]Alpha body text.[id3]"
        "[id4]Chapter Two[id5]"
        "[id6]Beta body text.[id7]"
    )
    tag_map = {
        "[id0]": "<section><h1>",
        "[id1]": "</h1>",
        "[id2]": "<p>",
        "[id3]": "</p>",
        "[id4]": "<h1>",
        "[id5]": "</h1>",
        "[id6]": "<p>",
        "[id7]": "</p></section>",
    }

    chunks = HtmlChunker(
        max_tokens=1000,
        chapter_mode=True,
    ).chunk_html_with_placeholders(text, tag_map)

    assert len(chunks) == 2
    assert chunks[0]["chapter_index"] == 0
    assert chunks[1]["chapter_index"] == 1
    assert "Alpha body text." in chunks[0]["text"]
    assert "Beta body text." not in chunks[0]["text"]
    assert "Beta body text." in chunks[1]["text"]
