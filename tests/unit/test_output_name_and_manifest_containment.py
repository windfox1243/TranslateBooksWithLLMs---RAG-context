"""Client-chosen names and EPUB manifest hrefs stay inside their directories.

Ported from upstream 2c698d80, 1fdb764b, 4161532d and fa14e26c.
"""

import os

import pytest

from src.api.services.file_service import FileService
from src.api.services.path_validator import PathValidator
from src.core.epub.translator import _parse_epub_manifest


@pytest.mark.parametrize(
    "requested, expected",
    [
        ("book (Vietnamese).epub", "book (Vietnamese).epub"),
        ("../../escape.txt", "escape.txt"),
        ("C:\\Windows\\run.bat", "run.bat.txt"),
        ("C:x.txt", "x.txt"),
        ("\\\\server\\share\\a.docx", "a.docx"),
        ("notes:stream.txt", "notes_stream.txt"),
        ("a<b>|c?.srt", "a_b__c_.srt"),
        ("CON", "_CON.txt"),
        ("nul.txt", "_nul.txt"),
        ("COM1.epub", "_COM1.epub"),
        ("report.exe", "report.exe.txt"),
        ("  ..  ", None),
        (None, None),
    ],
)
def test_safe_output_filename(requested, expected):
    assert PathValidator.safe_output_filename(requested) == expected


def test_resolve_output_file_stays_inside(tmp_path):
    resolved = PathValidator.resolve_output_file(tmp_path, "../../etc/passwd")
    assert resolved.parent == tmp_path
    assert resolved.name == "passwd.txt"
    with pytest.raises(ValueError):
        PathValidator.resolve_output_file(tmp_path, "..")


def test_open_action_refuses_executables(tmp_path):
    (tmp_path / "evil.bat").write_text("echo hi", encoding="utf-8")
    ok, message, path = FileService(str(tmp_path)).open_file("evil.bat")
    assert ok is False
    assert "not allowed" in message
    assert path is None


CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""

OPF = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="id">x</dc:identifier><dc:title>T</dc:title><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="up" href="../../victim.xhtml" media-type="application/xhtml+xml"/>
    <item id="abs" href="{absolute}" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/><itemref idref="up"/><itemref idref="abs"/></spine>
</package>"""


def test_manifest_items_outside_the_book_are_skipped(tmp_path):
    book = tmp_path / "book"
    (book / "META-INF").mkdir(parents=True)
    (book / "OEBPS").mkdir()
    (book / "META-INF" / "container.xml").write_text(CONTAINER, encoding="utf-8")
    victim = tmp_path / "victim.xhtml"
    victim.write_text("<html/>", encoding="utf-8")
    absolute = (tmp_path / "elsewhere.xhtml").as_posix()
    (book / "OEBPS" / "content.opf").write_text(OPF.format(absolute=absolute), encoding="utf-8")
    (book / "OEBPS" / "ch1.xhtml").write_text("<html/>", encoding="utf-8")

    logs = []
    manifest = _parse_epub_manifest(
        str(book), log_callback=lambda key, message="", *a, **k: logs.append(key),
    )

    assert manifest["content_files"] == ["ch1.xhtml"]
    assert logs.count("epub_href_outside_book") == 2
    assert os.path.exists(victim)
