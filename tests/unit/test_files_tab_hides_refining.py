"""The refine-after working file is not listed in the Files tab."""

from src.api.services.file_service import FileService


def test_refining_working_file_is_hidden(tmp_path):
    (tmp_path / "Book (French).epub").write_bytes(b"x")
    (tmp_path / ".Book (French).refining.epub").write_bytes(b"x")
    service = FileService(str(tmp_path))
    assert service._is_visible_file(tmp_path / "Book (French).epub") is True
    assert service._is_visible_file(tmp_path / ".Book (French).refining.epub") is False
