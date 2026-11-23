"""A file whose translation fails is written back exactly as it was read.

The translation pipeline edits the parsed tree in place before and while
chunks are sent. Writing that tree back for a failed file silently dropped
content the source still had. (Ported from upstream 0db1a0d7.)
"""
import zipfile
from pathlib import Path

import pytest

import src.core.epub.translator as translator_module
from src.core.common.translation_orchestrator import GenericTranslationOrchestrator
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.translator import translate_epub_file
from tests.unit.epub.conftest import _disable_attribution, _echo_llm_client

XHTML_NS = "http://www.w3.org/1999/xhtml"

CONTAINER = """<?xml version="1.0" encoding="utf-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>
"""

OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0" unique-identifier="id">
  <metadata><dc:title>Book</dc:title><dc:language>ja</dc:language><dc:identifier id="id">x</dc:identifier></metadata>
  <manifest><item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="ch1"/></spine>
</package>
"""

CHAPTER = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Ch</title></head>
<body>
<p>彼は<ruby>宇宙<rt>そら</rt></ruby>を見上げた。</p>
<div class="sharedaddy"><a href="https://example.invalid/share">Share on X</a></div>
</body></html>
"""


def _book(tmp_path: Path) -> Path:
    dest = tmp_path / "in.epub"
    with zipfile.ZipFile(dest, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", CONTAINER)
        archive.writestr("OEBPS/content.opf", OPF)
        archive.writestr("OEBPS/ch1.xhtml", CHAPTER)
    return dest


@pytest.mark.asyncio
async def test_failed_file_keeps_the_content_it_was_read_with(tmp_path, monkeypatch):
    async def failing_translation(self, source, **kwargs):
        # What the pipeline can do to the tree before it fails: edit it.
        body = source.find(f"{{{XHTML_NS}}}body")
        for child in list(body):
            body.remove(child)
        return False, TranslationMetrics()

    monkeypatch.setattr(translator_module, "_create_llm_client",
                        lambda **kwargs: _echo_llm_client())
    monkeypatch.setattr(GenericTranslationOrchestrator, "translate", failing_translation)
    _disable_attribution(monkeypatch)

    out = tmp_path / "out.epub"
    await translate_epub_file(
        input_filepath=str(_book(tmp_path)),
        output_filepath=str(out),
        source_language="Japanese",
        target_language="English",
    )

    with zipfile.ZipFile(out) as archive:
        written = archive.read("OEBPS/ch1.xhtml").decode("utf-8")
    assert "sharedaddy" in written
    assert "Share on X" in written
    assert "<rt>そら</rt>" in written
