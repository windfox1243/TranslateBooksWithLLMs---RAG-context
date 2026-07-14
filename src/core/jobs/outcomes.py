"""Typed outcomes shared by format-specific translation pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


ExecutionStatus = Literal["completed", "failed"]
QualityStatus = Literal["not_checked", "passed", "review_required"]


@dataclass(frozen=True)
class UnitTranslationOutcome:
    """Keep execution success separate from optional quality review.

    A usable, structurally valid draft is a completed translation even when a
    quality gate recommends human review. Only failures that prevent usable
    output belong in ``execution_status=\"failed\"``.
    """

    text: Optional[str]
    execution_status: ExecutionStatus
    quality_status: QualityStatus = "not_checked"
    execution_failure_class: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def completed(
        cls,
        text: str,
        *,
        quality_status: QualityStatus = "passed",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> "UnitTranslationOutcome":
        return cls(
            text=text,
            execution_status="completed",
            quality_status=quality_status,
            diagnostics=dict(diagnostics or {}),
        )

    @classmethod
    def review_required(
        cls,
        text: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> "UnitTranslationOutcome":
        return cls.completed(
            text,
            quality_status="review_required",
            diagnostics=diagnostics,
        )

    @classmethod
    def failed(
        cls,
        *,
        failure_class: str,
        text: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> "UnitTranslationOutcome":
        return cls(
            text=text,
            execution_status="failed",
            execution_failure_class=failure_class,
            diagnostics=dict(diagnostics or {}),
        )

    @property
    def is_completed(self) -> bool:
        return self.execution_status == "completed" and self.text is not None
