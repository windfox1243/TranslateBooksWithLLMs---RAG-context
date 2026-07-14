"""Small typed values shared by context analysis and prompt projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ContextAnalysisResult:
    """One source-unit analysis result before persistence."""

    lore_delta: str = ""
    dialogue_attribution: Dict[str, Any] = field(default_factory=dict)
    relationship_candidates: List[Dict[str, Any]] = field(default_factory=list)
    addressing_observations: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptContextBundle:
    """Scene-relevant context rendered for a draft or editor request."""

    global_invariants: str = ""
    entities: List[str] = field(default_factory=list)
    glossary: List[str] = field(default_factory=list)
    addressing: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    narrator: str = ""
    nearby_source: str = ""

    def render(self) -> str:
        sections = []
        for title, value in (
            ("GLOBAL INVARIANTS", self.global_invariants),
            ("RELEVANT ENTITIES", "\n".join(self.entities)),
            ("RELEVANT GLOSSARY", "\n".join(self.glossary)),
            ("ACTIVE ADDRESSING", "\n".join(self.addressing)),
            ("ACCEPTED RELATIONSHIPS", "\n".join(self.relationships)),
            ("NARRATOR POLICY", self.narrator),
            ("NEARBY SOURCE", self.nearby_source),
        ):
            if str(value or "").strip():
                sections.append(f"# {title}\n{str(value).strip()}")
        return "\n\n".join(sections)


@dataclass(frozen=True)
class SocialHierarchyEvidence:
    """Source-grounded directed social hierarchy independent of target language."""

    source: str
    target: str
    hierarchy: str
    rank_relation: str = "unknown"
    relative_age: str = "unknown"
    relationship_type: str = "associated"
    scope: str = "durable"
    evidence_quote: str = ""
    source_form: str = ""
    dialogue_turn_id: str = ""
    confidence: float = 1.0
    basis: str = "explicit social honorific"
