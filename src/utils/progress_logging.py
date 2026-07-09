"""Safe progress logging shared by translation pipeline layers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


MAX_LOG_VALUE_CHARS = 500


def _sanitize_log_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return "<nested>"
    if isinstance(value, dict):
        return {
            str(k): _sanitize_log_value(v, depth=depth + 1)
            for k, v in value.items()
            if str(k).casefold() not in {"api_key", "token", "secret", "password", "authorization"}
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_log_value(v, depth=depth + 1) for v in value[:20]]
    if isinstance(value, str):
        if len(value) > MAX_LOG_VALUE_CHARS:
            return value[:MAX_LOG_VALUE_CHARS].rstrip() + "...<truncated>"
        return value
    return value


def _call_log_callback(
    log_callback: Optional[Callable[..., Any]],
    event: str,
    message: str,
    data: Optional[Dict[str, Any]],
) -> None:
    if not log_callback:
        return
    try:
        if data is not None:
            log_callback(event, message, data=data)
        else:
            log_callback(event, message)
    except TypeError:
        log_callback(event, message)


def emit_progress_log(
    log_callback: Optional[Callable[..., Any]],
    event: str,
    message: str,
    *,
    level: str = "info",
    layer: str = "",
    file_id: str = "",
    chunk_index: Optional[int] = None,
    language_profile: str = "",
    data: Optional[Dict[str, Any]] = None,
    terminal: bool = True,
) -> None:
    """Emit a user-visible event to terminal logger and optional web callback."""

    payload: Dict[str, Any] = {
        "type": event,
    }
    if layer:
        payload["layer"] = layer
    if file_id:
        payload["file_id"] = file_id
    if chunk_index is not None:
        payload["chunk_index"] = chunk_index
    if language_profile:
        payload["language_profile"] = language_profile
    if data:
        payload.update(data)
    safe_payload = _sanitize_log_value(payload)

    if terminal:
        try:
            from src.utils.unified_logger import LogType, get_logger

            logger = get_logger()
            log_method = getattr(logger, level, logger.info)
            log_method(message, log_type=LogType.GENERAL, data=safe_payload)
        except Exception:
            pass

    _call_log_callback(log_callback, event, message, safe_payload)
