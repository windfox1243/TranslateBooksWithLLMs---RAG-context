"""
Unit tests for PathValidator class.

Tests filename validation to prevent directory traversal attacks
while allowing legitimate filenames with ellipsis and special characters.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import PathValidator directly from file to avoid importing flask dependencies
spec = importlib.util.spec_from_file_location(
    "path_validator",
    project_root / "src" / "api" / "services" / "path_validator.py"
)
path_validator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(path_validator_module)
PathValidator = path_validator_module.PathValidator


class TestPathValidator:
    """Tests for PathValidator class."""

    def test_validate_normal_filename(self):
        """Test validation of normal filenames."""
        is_valid, error = PathValidator.validate_filename("normal_file.txt")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_spaces(self):
        """Test validation of filenames with spaces."""
        is_valid, error = PathValidator.validate_filename("file with spaces.docx")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_ellipsis(self):
        """Test validation of filenames with ellipsis (three dots)."""
        is_valid, error = PathValidator.validate_filename("file...with...dots.txt")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_multiple_ellipsis(self):
        """Test validation of filenames with multiple ellipsis."""
        is_valid, error = PathValidator.validate_filename("voici un plan de cours complet, peux tu extraires... (French).docx")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_two_dots(self):
        """Test validation of filenames with two consecutive dots (no separators)."""
        is_valid, error = PathValidator.validate_filename("file..txt")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_four_dots(self):
        """Test validation of filenames with four consecutive dots."""
        is_valid, error = PathValidator.validate_filename("file....doc")
        assert is_valid is True
        assert error == ""

    def test_reject_directory_traversal_unix(self):
        """Test rejection of directory traversal attempts (Unix style)."""
        is_valid, error = PathValidator.validate_filename("../../../etc/passwd")
        assert is_valid is False
        assert "directory traversal" in error

    def test_reject_directory_traversal_windows(self):
        """Test rejection of directory traversal attempts (Windows style)."""
        is_valid, error = PathValidator.validate_filename("..\\..\\..\\windows\\system32")
        assert is_valid is False
        assert "directory traversal" in error or "path separators" in error

    def test_reject_embedded_directory_traversal(self):
        """Test rejection of embedded directory traversal."""
        is_valid, error = PathValidator.validate_filename("foo/../bar.txt")
        assert is_valid is False
        assert "directory traversal" in error or "path separators" in error

    def test_reject_absolute_path_unix(self):
        """Test rejection of absolute paths (Unix style)."""
        is_valid, error = PathValidator.validate_filename("/etc/passwd")
        assert is_valid is False
        assert "absolute path" in error or "path separators" in error

    def test_reject_absolute_path_windows(self):
        """Test rejection of absolute paths (Windows style)."""
        is_valid, error = PathValidator.validate_filename("C:\\Windows\\System32")
        assert is_valid is False
        assert "path" in error.lower()

    def test_reject_empty_filename(self):
        """Test rejection of empty filename."""
        is_valid, error = PathValidator.validate_filename("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_reject_path_separator_forward_slash(self):
        """Test rejection of filenames with forward slash separator."""
        is_valid, error = PathValidator.validate_filename("foo/bar.txt")
        assert is_valid is False
        assert "path separator" in error.lower()

    def test_reject_path_separator_backslash(self):
        """Test rejection of filenames with backslash separator."""
        is_valid, error = PathValidator.validate_filename("foo\\bar.txt")
        assert is_valid is False
        assert "path separator" in error.lower()

    def test_reject_filename_too_long(self):
        """Test rejection of filenames exceeding maximum length."""
        long_filename = "a" * 300
        is_valid, error = PathValidator.validate_filename(long_filename)
        assert is_valid is False
        assert "too long" in error.lower()

    def test_validate_filename_at_max_length(self):
        """Test validation of filename at maximum allowed length."""
        max_length_filename = "a" * 250 + ".txt"  # 254 chars total
        is_valid, error = PathValidator.validate_filename(max_length_filename)
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_special_characters(self):
        """Test validation of filenames with allowed special characters."""
        test_cases = [
            "file-name.txt",
            "file_name.txt",
            "file.name.txt",
            "file (copy).txt",
            "file [1].txt",
            "file {backup}.txt",
        ]
        for filename in test_cases:
            is_valid, error = PathValidator.validate_filename(filename)
            assert is_valid is True, f"Failed for: {filename}, error: {error}"

    def test_validate_filenames_list(self):
        """Test validation of a list of filenames."""
        filenames = [
            "file1.txt",
            "file2.docx",
            "file...with...dots.txt"
        ]
        is_valid, error = PathValidator.validate_filenames(filenames)
        assert is_valid is True
        assert error == ""

    def test_reject_filenames_list_with_invalid(self):
        """Test rejection of a list containing invalid filenames."""
        filenames = [
            "file1.txt",
            "../../../etc/passwd",
            "file3.docx"
        ]
        is_valid, error = PathValidator.validate_filenames(filenames)
        assert is_valid is False
        assert "passwd" in error

    def test_reject_filenames_not_list(self):
        """Test rejection when filenames parameter is not a list."""
        is_valid, error = PathValidator.validate_filenames("not_a_list")
        assert is_valid is False
        assert "must be a list" in error.lower()

    def test_reject_empty_filenames_list(self):
        """Test rejection of empty filenames list."""
        is_valid, error = PathValidator.validate_filenames([])
        assert is_valid is False
        assert "no filenames" in error.lower()

    def test_validate_filename_with_unicode(self):
        """Test validation of filenames with Unicode characters."""
        test_cases = [
            "fichier_français.txt",
            "文件.txt",
            "archivo_español.docx",
        ]
        for filename in test_cases:
            is_valid, error = PathValidator.validate_filename(filename)
            # Should be valid as long as no path separators or traversal
            assert is_valid is True, f"Failed for: {filename}, error: {error}"

    def test_reject_directory_traversal_at_start(self):
        """Test rejection of filenames starting with directory traversal."""
        is_valid, error = PathValidator.validate_filename("../file.txt")
        assert is_valid is False
        assert "directory traversal" in error or "absolute path" in error or "path separator" in error

    def test_reject_directory_traversal_at_end(self):
        """Test rejection of filenames ending with directory traversal."""
        is_valid, error = PathValidator.validate_filename("file/..")
        assert is_valid is False
        assert "directory traversal" in error or "path separator" in error

    def test_validate_filename_with_dot_prefix(self):
        """Test validation of hidden files (dot prefix)."""
        is_valid, error = PathValidator.validate_filename(".hidden_file.txt")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_only_dots_in_basename(self):
        """Test validation of filenames with only dots in basename."""
        is_valid, error = PathValidator.validate_filename("....txt")
        assert is_valid is True
        assert error == ""


class TestPathValidatorEdgeCases:
    """Tests for edge cases and security-critical scenarios."""

    def test_null_byte_injection(self):
        """Test handling of null byte injection attempts."""
        # Note: Python strings can contain null bytes
        filename_with_null = "file\x00.txt"
        is_valid, error = PathValidator.validate_filename(filename_with_null)
        # Should be valid (OS will handle null bytes)
        # This is acceptable as the actual file operations will fail safely
        assert is_valid is True

    def test_very_long_extension(self):
        """Test handling of very long file extensions."""
        filename = "file." + "x" * 100
        is_valid, error = PathValidator.validate_filename(filename)
        # Should be valid if under max length
        assert is_valid is True

    def test_multiple_extensions(self):
        """Test handling of multiple file extensions."""
        is_valid, error = PathValidator.validate_filename("file.tar.gz")
        assert is_valid is True
        assert error == ""

    def test_no_extension(self):
        """Test handling of filenames without extensions."""
        is_valid, error = PathValidator.validate_filename("README")
        assert is_valid is True
        assert error == ""

    def test_directory_traversal_variations(self):
        """Test various directory traversal attack patterns."""
        attack_patterns = [
            "../..",
            "..\\..",
            "..\\../",
            "../.\\",
            "....//",
            "..\\\\..\\\\",
        ]
        for pattern in attack_patterns:
            is_valid, error = PathValidator.validate_filename(pattern)
            assert is_valid is False, f"Should reject: {pattern}"

    def test_whitespace_only_filename(self):
        """Test handling of whitespace-only filenames."""
        # Note: The current implementation treats whitespace as valid characters
        # This could be debated, but we test current behavior
        is_valid, error = PathValidator.validate_filename("   ")
        # After stripping happens in the actual usage, this would fail differently
        # We test the validator as-is
        assert is_valid is True  # Validator allows it, but usage should strip

    def test_real_world_problematic_filename(self):
        """Test the real-world case that triggered this fix."""
        filename = "voici un plan de cours complet, peux tu extraires... (French).docx"
        is_valid, error = PathValidator.validate_filename(filename)
        assert is_valid is True
        assert error == ""


class TestValidateUploadPath:
    """Tests for validate_upload_path / is_within_directory (issue #209)."""

    @pytest.fixture
    def uploads_dir(self, tmp_path):
        d = tmp_path / "uploads"
        d.mkdir()
        (d / "book.txt").write_text("content", encoding="utf-8")
        return d

    def test_accepts_absolute_path_inside_uploads(self, uploads_dir):
        path, error = PathValidator.validate_upload_path(
            str(uploads_dir / "book.txt"), uploads_dir
        )
        assert error is None
        assert path == (uploads_dir / "book.txt").resolve()

    def test_accepts_relative_path_inside_uploads(self, uploads_dir):
        path, error = PathValidator.validate_upload_path("book.txt", uploads_dir)
        assert error is None
        assert path == (uploads_dir / "book.txt").resolve()

    def test_accepts_path_in_subdirectory(self, uploads_dir):
        sub = uploads_dir / "job123"
        sub.mkdir()
        (sub / "src.txt").write_text("x", encoding="utf-8")
        path, error = PathValidator.validate_upload_path(str(sub / "src.txt"), uploads_dir)
        assert error is None
        assert path == (sub / "src.txt").resolve()

    def test_rejects_absolute_path_outside_uploads(self, uploads_dir, tmp_path):
        secret = tmp_path / "secret.env"
        secret.write_text("API_KEY=xxx", encoding="utf-8")
        path, error = PathValidator.validate_upload_path(str(secret), uploads_dir)
        assert path is None
        assert "outside the uploads directory" in error

    def test_rejects_directory_traversal(self, uploads_dir, tmp_path):
        secret = tmp_path / "secret.env"
        secret.write_text("API_KEY=xxx", encoding="utf-8")
        path, error = PathValidator.validate_upload_path("../secret.env", uploads_dir)
        assert path is None
        assert "outside the uploads directory" in error

    def test_rejects_sibling_prefix_directory(self, tmp_path):
        # '/.../uploads-evil' must NOT be considered inside '/.../uploads'
        # (the bug a string startswith check would have).
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        evil = tmp_path / "uploads-evil"
        evil.mkdir()
        target = evil / "f.txt"
        target.write_text("x", encoding="utf-8")
        path, error = PathValidator.validate_upload_path(str(target), uploads)
        assert path is None
        assert error is not None

    def test_missing_path_returns_error(self, uploads_dir):
        path, error = PathValidator.validate_upload_path("", uploads_dir)
        assert path is None
        assert "Missing" in error
        path, error = PathValidator.validate_upload_path(None, uploads_dir)
        assert path is None
        assert "Missing" in error

    def test_nonexistent_path_inside_uploads_reports_not_found(self, uploads_dir):
        path, error = PathValidator.validate_upload_path("nope.txt", uploads_dir)
        assert path is None
        assert error == "File not found"

    def test_is_within_directory_true_false(self, tmp_path):
        d = tmp_path / "uploads"
        d.mkdir()
        inside = d / "f.txt"
        inside.write_text("x", encoding="utf-8")
        assert PathValidator.is_within_directory(inside, d) is True
        assert PathValidator.is_within_directory(tmp_path / "other.txt", d) is False


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
