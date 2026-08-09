"""Replace one file with another, allowing for Windows sharing violations.

`os.replace` is atomic on both platforms, but on Windows it is atomic only when
it succeeds, and it fails outright whenever anything else holds the destination
open: the user reading a context file in an editor, an antivirus scanner, the
search indexer, a preview pane. POSIX has no equivalent -- the rename lands and
the other reader keeps its old handle -- so the failure is invisible on the
platform the code was written on and routine on the platform it ships to.

These holds are short. A bounded retry turns a hard failure into a pause of a
few hundred milliseconds; only a destination held open for the whole window
still raises, and then the error says what to close rather than surfacing as a
bare WinError.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Union

# Long enough to outlast a scanner or an indexer, short enough that a file a
# user left open in an editor reports rather than hanging the job.
_ATTEMPTS = 6
_INITIAL_DELAY_SECONDS = 0.05

# WinError 5 (access denied) and WinError 32 (in use by another process) are the
# two the sharing violation surfaces as. Everything else -- a missing source, a
# bad path, a full disk -- is a real error and is raised on the first attempt.
_SHARING_VIOLATION_WINERRORS = frozenset({5, 32})


def _is_sharing_violation(error: OSError) -> bool:
    return getattr(error, "winerror", None) in _SHARING_VIOLATION_WINERRORS


def replace_atomically(
    source: Union[str, Path],
    destination: Union[str, Path],
    *,
    attempts: int = _ATTEMPTS,
) -> None:
    """Move ``source`` onto ``destination``, retrying past transient holds.

    Raises the underlying ``OSError`` if the destination is still held when the
    attempts run out, with the likely cause named in the message.
    """
    delay = _INITIAL_DELAY_SECONDS
    for remaining in range(attempts - 1, -1, -1):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if remaining == 0 or not _is_sharing_violation(error):
                if _is_sharing_violation(error):
                    raise OSError(
                        error.errno,
                        f"Could not replace '{destination}': it is open in "
                        f"another program. Close it and try again.",
                    ) from error
                raise
            time.sleep(delay)
            delay *= 2
