"""Read and write the persisted novel-context document format."""
from __future__ import annotations

import base64
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from .constants import DYNAMIC_STATE_END, DYNAMIC_STATE_START, logger
from .dynamic_state import _split_dynamic_sections


def extract_dynamic_state_from_text(context_content: str) -> Optional[str]:
    start_tag = DYNAMIC_STATE_START
    end_tag = DYNAMIC_STATE_END
    start_idx = context_content.find(start_tag)
    end_idx = context_content.find(end_tag)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        state_text = context_content[start_idx + len(start_tag):end_idx].strip()
        # Clean up `# DYNAMIC RELATIONSHIP STATE` headers
        lines = state_text.splitlines()
        cleaned_lines = []
        for line in lines:
            if line.strip().upper().replace(" ", "") == "#DYNAMICRELATIONSHIPSTATE":
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()
    return None
def _dynamic_state_has_entries(dynamic_state: str) -> bool:
    addressing, relationships, has_sections = _split_dynamic_sections(
        dynamic_state
    )
    if not has_sections:
        return bool(str(dynamic_state or "").strip())
    return bool(addressing.strip() or relationships.strip())
def compress_dynamic_state(dynamic_text: str) -> str:
    compressed = zlib.compress(dynamic_text.encode('utf-8'))
    return base64.b64encode(compressed).decode('ascii')
def extract_global_lore(context_content: str) -> str:
    """Extracts the text before the DYNAMIC_STATE_START tag."""
    start_tag = DYNAMIC_STATE_START
    start_idx = context_content.find(start_tag)
    
    if start_idx != -1:
        return context_content[:start_idx].strip()
    return context_content.strip()
def decompress_dynamic_state(b64_compressed_state: str) -> str:
    """Decompresses a base64 zlib string back to plain text."""
    try:
        compressed = base64.b64decode(b64_compressed_state)
        return zlib.decompress(compressed).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to decompress dynamic state: {e}")
        return ""
