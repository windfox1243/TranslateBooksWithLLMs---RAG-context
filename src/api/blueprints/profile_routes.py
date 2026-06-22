"""Translation profile persistence routes."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from flask import Blueprint, jsonify, request

from src.config import TRANSLATION_PROFILES_DIR
from src.utils.novel_context import normalize_novel_context_filename


bp = Blueprint("profiles", __name__)

PROFILE_VERSION = 1
MAX_PROFILE_NAME_LENGTH = 80
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_. -]+$")
SENSITIVE_KEY_RE = re.compile(
    r"(?:^|_)(?:api_?key|token|password|secret|credential|authorization)(?:$|_)",
    re.IGNORECASE,
)

STRING_FIELDS = {
    "source_language": 100,
    "target_language": 100,
    "llm_provider": 50,
    "model": 200,
    "llm_api_endpoint": 2048,
    "glossary": 100,
    "custom_instruction_file": 255,
    "novel_context_file": 255,
    "tts_voice": 255,
    "tts_rate": 20,
    "tts_format": 20,
    "tts_bitrate": 20,
    "output_filename_pattern": 500,
}
BOOLEAN_FIELDS = {
    "bilingual_output",
    "text_cleanup",
    "auto_update_context",
    "plain_text_mode",
    "chapter_mode",
    "auto_pause_on_rate_limit",
    "tts_enabled",
}
INTEGER_FIELDS = {
    "parallel_workers": (1, 16),
}


def is_safe_filename(filename: str) -> bool:
    """Return whether a profile name is safe and reasonably sized."""
    return bool(
        filename
        and filename == filename.strip()
        and filename not in {".", ".."}
        and len(filename) <= MAX_PROFILE_NAME_LENGTH
        and SAFE_FILENAME_RE.fullmatch(filename)
    )


def _contains_sensitive_key(value) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            compact_key = re.sub(r"[^a-z0-9]", "", key_text.lower())
            sensitive_name = bool(SENSITIVE_KEY_RE.search(key_text)) or any(
                marker in compact_key
                for marker in (
                    "apikey",
                    "accesstoken",
                    "password",
                    "secret",
                    "credential",
                    "authorization",
                )
            )
            if sensitive_name or _contains_sensitive_key(child):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_sensitive_key(child) for child in value)
    return False


def _validate_endpoint(value: str) -> None:
    """Reject credentials embedded in an endpoint URL."""
    if not value:
        return
    parsed = urlsplit(value)
    if parsed.username or parsed.password:
        raise ValueError("API endpoints must not contain embedded credentials")
    sensitive_query_names = {
        "api_key", "apikey", "key", "token", "access_token", "secret", "password"
    }
    if any(name.lower() in sensitive_query_names for name, _ in parse_qsl(parsed.query)):
        raise ValueError("API endpoints must not contain credentials in query parameters")


def _normalize_profile(data) -> dict:
    """Validate profile data and return the versioned, secret-free schema."""
    if not isinstance(data, dict) or not data:
        raise ValueError("Missing profile data")
    if _contains_sensitive_key(data):
        raise ValueError("Profiles must not contain API keys or other credentials")

    normalized = {"profile_version": PROFILE_VERSION}
    for field, max_length in STRING_FIELDS.items():
        if field not in data:
            continue
        value = data[field]
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string")
        if len(value) > max_length:
            raise ValueError(f"{field} is too long")
        if field == "llm_api_endpoint":
            _validate_endpoint(value)
        if field == "novel_context_file" and value:
            value = normalize_novel_context_filename(value)
        normalized[field] = value

    for field in BOOLEAN_FIELDS:
        if field not in data:
            continue
        value = data[field]
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be a boolean")
        normalized[field] = value

    for field, (minimum, maximum) in INTEGER_FIELDS.items():
        if field not in data:
            continue
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(f"{field} must be between {minimum} and {maximum}")
        normalized[field] = value

    return normalized


def _profile_path(name: str) -> Path:
    return TRANSLATION_PROFILES_DIR / f"{name}.json"


@bp.route("/api/profiles", methods=["GET"])
def list_profiles():
    try:
        TRANSLATION_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        profiles = sorted(
            (file_path.stem for file_path in TRANSLATION_PROFILES_DIR.glob("*.json")),
            key=str.casefold,
        )
        return jsonify(profiles), 200
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500


@bp.route("/api/profiles/<name>", methods=["GET"])
def get_profile(name):
    if not is_safe_filename(name):
        return jsonify({"error": "Invalid profile name"}), 400

    file_path = _profile_path(name)
    if not file_path.exists():
        return jsonify({"error": "Profile not found"}), 404

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return jsonify(_normalize_profile(data)), 200
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({"error": f"Invalid profile file: {exc}"}), 422
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500


@bp.route("/api/profiles/<name>", methods=["POST"])
def save_profile(name):
    if not is_safe_filename(name):
        return jsonify({"error": "Invalid profile name"}), 400

    try:
        data = _normalize_profile(request.get_json(silent=True))
        TRANSLATION_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        file_path = _profile_path(name)
        temporary_path = file_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(file_path)
        return jsonify({"message": "Profile saved successfully"}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500


@bp.route("/api/profiles/<name>", methods=["DELETE"])
def delete_profile(name):
    if not is_safe_filename(name):
        return jsonify({"error": "Invalid profile name"}), 400

    file_path = _profile_path(name)
    if not file_path.exists():
        return jsonify({"error": "Profile not found"}), 404

    try:
        file_path.unlink()
        return jsonify({"message": "Profile deleted successfully"}), 200
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500


def create_profile_blueprint():
    return bp
