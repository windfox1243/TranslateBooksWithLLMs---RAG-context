"""
Path validation utilities for secure file operations
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Union

from src.utils.security import is_within_directory

# DOS device names reserved by Windows (mirrors ntpath._reserved_names as of
# Python 3.13). PureWindowsPath.is_reserved() was deprecated in 3.13 and
# removed in 3.15, and its replacement only exists on 3.13+, so the check is
# spelled out here.
_WINDOWS_DEVICE_NAMES = frozenset(
    {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'}
    | {f'COM{c}' for c in '123456789¹²³'}
    | {f'LPT{c}' for c in '123456789¹²³'}
)


def is_windows_device_name(name: str) -> bool:
    """True when `name`'s stem is a DOS device name ('CON', 'NUL.txt', ...)."""
    return name.partition('.')[0].rstrip(' ').upper() in _WINDOWS_DEVICE_NAMES


class PathValidator:
    """Validates file paths and names for security"""

    MAX_FILENAME_LENGTH = 255

    # Extensions a translation job may write: the document types uploads
    # accept (SecureFileHandler.ALLOWED_EXTENSIONS). Any other requested
    # extension gets '.txt' appended, so a job can never drop an executable
    # ('.bat', '.exe', '.hta', ...) into the output directory.
    OUTPUT_EXTENSIONS = frozenset({
        '.txt', '.epub', '.srt', '.docx',
        '.text', '.log', '.md', '.markdown', '.rst', '.asc',
        '.csv', '.json', '.xml', '.html', '.htm', '.yaml', '.yml',
    })

    # Files the open action may hand to the OS default application:
    # translated documents and TTS audio only. Types whose default handler
    # can run active content (HTML in a browser, CSV formulas in a
    # spreadsheet, ...) are left out on purpose.
    OPENABLE_EXTENSIONS = frozenset({
        '.txt', '.text', '.md', '.markdown', '.rst', '.log', '.asc',
        '.srt', '.epub', '.docx',
        '.opus', '.mp3', '.wav', '.ogg',
    })

    @staticmethod
    def safe_output_filename(requested: Optional[str]) -> Optional[str]:
        """Reduce a client-supplied output filename to a safe bare name.

        Directory components (either separator, a drive prefix) are dropped,
        so '../../x.txt' or an absolute path become 'x.txt'. Characters
        Windows refuses become '_' (':' would otherwise write into an NTFS
        alternate data stream), a device name gets a '_' prefix, and a name
        whose extension is not in OUTPUT_EXTENSIONS gets '.txt' appended.
        Returns None when nothing usable remains.
        """
        if not isinstance(requested, str):
            return None
        name = requested.replace('\\', '/').rsplit('/', 1)[-1]
        if len(name) >= 2 and name[1] == ':':
            name = name[2:]
        name = ''.join('_' if (c in '<>:"|?*' or ord(c) < 32) else c for c in name)
        name = name.strip().strip('.').strip()
        if not name:
            return None
        if is_windows_device_name(name):
            name = f"_{name}"
        if Path(name).suffix.lower() not in PathValidator.OUTPUT_EXTENSIONS:
            name = f"{name}.txt"
        if len(name) > PathValidator.MAX_FILENAME_LENGTH:
            stem, suffix = os.path.splitext(name)
            name = stem[:PathValidator.MAX_FILENAME_LENGTH - len(suffix)] + suffix
        return name

    @staticmethod
    def resolve_output_file(output_dir: Union[str, Path], requested: Optional[str]) -> Path:
        """Return the path inside `output_dir` a job may write `requested` to.

        Raises ValueError when the name is unusable or would escape the
        directory (defence in depth on top of safe_output_filename).
        """
        name = PathValidator.safe_output_filename(requested)
        if name is None:
            raise ValueError("Invalid output filename")
        candidate = Path(output_dir) / name
        if not PathValidator.is_within_directory(candidate, output_dir):
            raise ValueError("Output filename resolves outside the output directory")
        return candidate

    @staticmethod
    def is_openable(path: Union[str, Path]) -> bool:
        """True when `path` has a document/audio extension safe to open."""
        return Path(path).suffix.lower() in PathValidator.OPENABLE_EXTENSIONS

    @staticmethod
    def is_within_directory(path: Union[str, Path], directory: Union[str, Path]) -> bool:
        """
        Return True only if `path` resolves to a location inside `directory`.

        Both arguments are resolved first (following symlinks and normalizing
        '..'), then compared component-wise via Path.relative_to — never a
        string startswith, which would treat '/uploads-evil' as inside
        '/uploads'.
        """
        return is_within_directory(path, directory)

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
