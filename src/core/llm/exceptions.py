"""
LLM-specific exceptions.

This module defines all custom exceptions used in the LLM provider system.
"""


class ContextOverflowError(Exception):
    """
    Raised when the input text exceeds the model's context window.

    This typically occurs when a chunk is too large for the model to process
    in a single request.
    """
    pass


class RepetitionLoopError(Exception):
    """
    Raised when the model enters a repetition loop.

    This can occur with "thinking" models that get stuck repeating the same
    phrase or pattern, indicating the model has likely exceeded its effective
    context window or encountered an issue.
    """
    pass


class StructuredOutputSchemaError(RuntimeError):
    """Raised when a provider deterministically rejects an output schema."""

    pass


class ProviderRequestError(RuntimeError):
    """Sanitized terminal provider failure for an editor-stage request."""

    def __init__(self, failure_class: str, status_code: int | None = None):
        super().__init__(failure_class)
        self.failure_class = failure_class
        self.status_code = status_code


class RateLimitError(Exception):
    """
    Raised when the API returns HTTP 429 (Too Many Requests) and all retry
    attempts with backoff have been exhausted.

    This signals the translation pipeline to auto-pause and save a checkpoint
    so the user can resume later.

    Attributes:
        retry_after: Suggested wait time in seconds (from Retry-After header),
                     or None if not provided by the API.
        provider: Name of the LLM provider that was rate-limited.
    """

    def __init__(self, message: str, retry_after: int = None, provider: str = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.provider = provider
