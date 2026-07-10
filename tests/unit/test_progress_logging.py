"""Regression tests for shared progress logging fanout."""

from src.utils.progress_logging import emit_progress_log


class CapturingLogger:
    def __init__(self):
        self.calls = []

    def info(self, message, log_type=None, data=None):
        self.calls.append((message, log_type, data))


def test_callback_receives_untruncated_payload_without_direct_terminal_duplicate(monkeypatch):
    logger = CapturingLogger()
    monkeypatch.setattr("src.utils.unified_logger.get_logger", lambda: logger)
    events = []
    response = f"<TRANSLATION>{'translated text ' * 80}</TRANSLATION>"

    emit_progress_log(
        lambda event, message, data=None: events.append((event, message, data)),
        "llm_response",
        "LLM Response received",
        data={"response": response, "api_key": "sk-xxxxxxxx"},
    )

    assert logger.calls == []
    assert len(events) == 1
    assert events[0][2]["response"] == response
    assert "api_key" not in events[0][2]


def test_terminal_fallback_uses_truncated_sanitized_payload(monkeypatch):
    logger = CapturingLogger()
    monkeypatch.setattr("src.utils.unified_logger.get_logger", lambda: logger)
    response = f"<TRANSLATION>{'translated text ' * 80}</TRANSLATION>"

    emit_progress_log(
        None,
        "llm_response",
        "LLM Response received",
        data={"response": response, "token": "sk-xxxxxxxx"},
    )

    assert len(logger.calls) == 1
    payload = logger.calls[0][2]
    assert payload["response"].endswith("...<truncated>")
    assert len(payload["response"]) < len(response)
    assert "token" not in payload


def test_callback_can_accept_positional_data_after_keyword_retry():
    events = []

    def callback(event, message, payload):
        events.append((event, message, payload))

    emit_progress_log(
        callback,
        "reflection_parse",
        "Senior Editor reflection parsed with json.",
        data={"parse_status": "json"},
    )

    assert events == [
        (
            "reflection_parse",
            "Senior Editor reflection parsed with json.",
            {"type": "reflection_parse", "parse_status": "json"},
        )
    ]
