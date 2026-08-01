"""Unit tests for item 2.1: archive path containment (zip slip).

Covers the shared helpers in `src.utils.security`, the EPUB upload validator,
and the two checkpoint-manager seams that join an attacker-controlled
`file_href` to a directory on disk.
"""
import zipfile
from pathlib import Path

import pytest

from src.persistence.checkpoint_manager import CheckpointManager
from src.utils.security import (
    SecureFileHandler,
    SecurityError,
    find_unsafe_archive_member,
    is_safe_archive_member,
    resolve_within,
    safe_extract_zip,
)

CONTAINER_XML = (
    '<?xml version="1.0"?>'
    '<container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/></rootfiles>'
    '</container>'
)

CLEAN_ENTRIES = {
    'mimetype': 'application/epub+zip',
    'META-INF/container.xml': CONTAINER_XML,
    'OEBPS/chapter1.xhtml': '<html><body><p>Hello</p></body></html>',
}


def _write_epub(path: Path, extra_entries=None) -> Path:
    """Build a minimal EPUB-shaped zip, optionally with extra entries."""
    entries = dict(CLEAN_ENTRIES)
    if extra_entries:
        entries.update(extra_entries)
    with zipfile.ZipFile(path, 'w') as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return path


# === 1. is_safe_archive_member truth table ===================================

UNSAFE_NAMES = [
    '../../evil.txt',
    '..\\..\\evil.txt',
    '/etc/passwd',
    'C:/evil.txt',
    'C:\\evil.txt',
    '//server/share/x',
    'OEBPS/../../evil.txt',
    '',
    'a\x00b',
]

SAFE_NAMES = [
    'mimetype',
    'META-INF/container.xml',
    'OEBPS/chapter1.xhtml',
    'OEBPS/',
    './OEBPS/ch1.xhtml',
    'OEBPS/im..ages/a.png',
]


@pytest.mark.parametrize('name', UNSAFE_NAMES)
def test_is_safe_archive_member_rejects(name):
    assert is_safe_archive_member(name) is False


@pytest.mark.parametrize('name', SAFE_NAMES)
def test_is_safe_archive_member_accepts(name):
    assert is_safe_archive_member(name) is True


def test_find_unsafe_archive_member_returns_first_offender():
    names = ['mimetype', '../../evil.txt', 'C:/other.txt']
    assert find_unsafe_archive_member(names) == '../../evil.txt'


def test_find_unsafe_archive_member_returns_none_when_clean():
    assert find_unsafe_archive_member(SAFE_NAMES) is None


# === 2. The EPUB upload validator rejects a traversal entry ==================

def test_validate_epub_file_rejects_traversal_entry(tmp_path):
    epub_path = _write_epub(
        tmp_path / 'evil.epub',
        {'../../evil.txt': 'pwned'},
    )
    handler = SecureFileHandler(tmp_path / 'uploads')

    result = handler._validate_epub_file(epub_path)

    assert result.is_valid is False
    assert 'evil.txt' in result.error_message


def test_validate_epub_file_accepts_clean_archive(tmp_path):
    epub_path = _write_epub(tmp_path / 'clean.epub')
    handler = SecureFileHandler(tmp_path / 'uploads')

    assert handler._validate_epub_file(epub_path).is_valid is True


# === 3. safe_extract_zip writes nothing when it rejects ======================

def test_safe_extract_zip_rejects_and_writes_nothing(tmp_path):
    epub_path = _write_epub(
        tmp_path / 'evil.epub',
        {'../../evil.txt': 'pwned'},
    )
    dest = tmp_path / 'extract_root' / 'dest'
    dest.mkdir(parents=True)

    with zipfile.ZipFile(epub_path, 'r') as zf:
        with pytest.raises(SecurityError):
            safe_extract_zip(zf, dest)

    assert list(dest.iterdir()) == []
    assert not (dest.parent / 'evil.txt').exists()
    assert not any(p.name == 'evil.txt' for p in tmp_path.rglob('*') if p.is_file())


# === 4. A clean archive extracts normally ====================================

def test_safe_extract_zip_extracts_clean_archive(tmp_path):
    epub_path = _write_epub(tmp_path / 'clean.epub')
    dest = tmp_path / 'dest'
    dest.mkdir()

    with zipfile.ZipFile(epub_path, 'r') as zf:
        safe_extract_zip(zf, dest)

    assert (dest / 'mimetype').is_file()
    assert (dest / 'META-INF' / 'container.xml').is_file()
    assert (dest / 'OEBPS' / 'chapter1.xhtml').is_file()


def test_resolve_within_rejects_escape(tmp_path):
    with pytest.raises(SecurityError):
        resolve_within(tmp_path, '../evil.txt')


def test_resolve_within_accepts_nested_path(tmp_path):
    resolved = resolve_within(tmp_path, 'OEBPS/ch1.xhtml')
    assert resolved == (tmp_path / 'OEBPS' / 'ch1.xhtml').resolve()


# === 5 & 6. save_epub_file containment =======================================

@pytest.fixture
def checkpoint_manager(tmp_path, monkeypatch):
    """A CheckpointManager rooted entirely inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return CheckpointManager(db_path=str(tmp_path / 'jobs.db'))


def test_save_epub_file_refuses_traversal_href(checkpoint_manager, tmp_path):
    translation_id = 'job-zip-slip'

    assert checkpoint_manager.save_epub_file(
        translation_id, '../../evil.txt', b'x'
    ) is False

    uploads_root = checkpoint_manager.uploads_dir
    assert not any(
        p.name == 'evil.txt' for p in uploads_root.rglob('*')
    )
    assert not any(
        p.name == 'evil.txt' for p in uploads_root.parent.rglob('*')
    )
    assert not any(p.name == 'evil.txt' for p in tmp_path.rglob('*'))


def test_save_epub_file_writes_nested_path(checkpoint_manager):
    translation_id = 'job-nested'

    assert checkpoint_manager.save_epub_file(
        translation_id, 'OEBPS/ch1.xhtml', b'x'
    ) is True

    written = (
        checkpoint_manager.uploads_dir
        / translation_id
        / 'translated_files'
        / 'OEBPS'
        / 'ch1.xhtml'
    )
    assert written.is_file()
    assert written.read_bytes() == b'x'


def test_restore_epub_files_copies_nested_path(checkpoint_manager, tmp_path):
    translation_id = 'job-restore'
    assert checkpoint_manager.save_epub_file(
        translation_id, 'OEBPS/ch1.xhtml', b'x'
    ) is True

    work_dir = tmp_path / 'work'
    work_dir.mkdir()

    assert checkpoint_manager.restore_epub_files(translation_id, work_dir) is True
    assert (work_dir / 'OEBPS' / 'ch1.xhtml').read_bytes() == b'x'
