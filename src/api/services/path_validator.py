"""
Path validation utilities for secure file operations
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Union


class PathValidator:
    """Validates file paths and names for security"""

    MAX_FILENAME_LENGTH = 255

    @staticmethod
    def is_within_directory(path: Union[str, Path], directory: Union[str, Path]) -> bool:
        """
        Return True only if `path` resolves to a location inside `directory`.

        Both arguments are resolved first (following symlinks and normalizing
        '..'), then compared component-wise via Path.relative_to — never a
        string startswith, which would treat '/uploads-evil' as inside
        '/uploads'.
        """
        try:
            path_resolved = Path(path).resolve()
            dir_resolved = Path(directory).resolve()
        except OSError:
            return False
        try:
            path_resolved.relative_to(dir_resolved)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_upload_path(
        file_path: Optional[str],
        uploads_dir: Union[str, Path],
    ) -> Tuple[Optional[Path], Optional[str]]:
        """
        Validate a client-supplied file path before reading it.

        Resolves the path (relative paths are taken under `uploads_dir`) and
        requires the result to live inside `uploads_dir`. This blocks arbitrary
        file reads via absolute paths or directory traversal.

        Returns (resolved_path, None) on success, or (None, error_message) on
        failure. Containment is checked before existence so the error never
        reveals whether an out-of-bounds file exists.
        """
        if not file_path:
            return None, "Missing field: file_path"

        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = Path(uploads_dir) / candidate

        if not PathValidator.is_within_directory(candidate, uploads_dir):
            return None, "Access denied: file path is outside the uploads directory"

        resolved = candidate.resolve()
        if not resolved.exists():
            return None, "File not found"

        return resolved, None

    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, str]:
        """
        Validate filename for security issues

        Args:
            filename: The filename to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "Filename cannot be empty"

        # Prevent directory traversal - check for path separators with '..'
        # This allows '...' or '....' but blocks '../' or '..\' patterns
        if filename.startswith('/') or filename.startswith('\\'):
            return False, "Invalid filename: absolute path not allowed"

        # Check for directory traversal patterns
        # Block: ../ or ..\ (with separators)
        if '/../' in filename or '\\..\\' in filename or '/..' in filename or '\\..' in filename:
            return False, "Invalid filename: directory traversal not allowed"

        # Also check if the normalized filename contains path separators
        # This catches cases like "foo/../bar" or "foo/bar"
        if '/' in filename or '\\' in filename:
            return False, "Invalid filename: path separators not allowed"

        # Check filename length
        if len(filename) > PathValidator.MAX_FILENAME_LENGTH:
            return False, f"Filename too long (max {PathValidator.MAX_FILENAME_LENGTH} characters)"

        # Prevent absolute paths
        if ':' in filename and len(filename) > 2 and filename[1] == ':':  # Windows absolute path
            return False, "Absolute paths not allowed"

        return True, ""

    @staticmethod
    def validate_filenames(filenames: list) -> Tuple[bool, str]:
        """
        Validate a list of filenames

        Args:
            filenames: List of filenames to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(filenames, list):
            return False, "Filenames must be a list"

        if len(filenames) == 0:
            return False, "No filenames provided"

        for filename in filenames:
            is_valid, error = PathValidator.validate_filename(filename)
            if not is_valid:
                return False, f"Invalid filename '{filename}': {error}"

        return True, ""
