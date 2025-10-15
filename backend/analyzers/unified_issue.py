"""
Unified Issue Structure for Multi-Rule Analysis
Combines grammar, word-list, and tone issues into comprehensive suggestions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class RuleCategory(Enum):
    GRAMMAR = "grammar"
    WORD_LIST = "word-list"
    TONE_ISSUE = "tone-issue"


class RuleSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    INFO = "info"


@dataclass
class DetailedChange:
    """Represents a single rule's change within a multi-rule issue."""
    rule: str
    category: str
    change: str
    severity: str
    position: Optional[tuple] = None  # (start, end) if applicable

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "rule": self.rule,
            "category": self.category,
            "change": self.change,
            "severity": self.severity
        }
        if self.position:
            d["position"] = {"start": self.position[0], "end": self.position[1]}
        return d


@dataclass
class LocationInfo:
    """Structured location information."""
    page_or_slide_index: int
    element_index: int
    element_type: str
    display: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_or_slide_index": self.page_or_slide_index,
            "element_index": self.element_index,
            "element_type": self.element_type,
            "display": self.display
        }


@dataclass
class UnifiedIssue:
    """
    Comprehensive issue combining multiple rule violations.
    """
    issue_id: str
    rule_names: List[str]
    severity: str  # Highest severity among all rules
    categories: List[str]
    descriptions: List[str]
    location: LocationInfo
    found_text: str
    suggestion: str
    changes_summary: str
    detailed_changes: List[DetailedChange]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "rule_names": self.rule_names,
            "severity": self.severity,
            "categories": self.categories,
            "descriptions": self.descriptions,
            "location": self.location.to_dict(),
            "found_text": self.found_text,
            "suggestion": self.suggestion,
            "changes_summary": self.changes_summary,
            "detailed_changes": [c.to_dict() for c in self.detailed_changes],
            "metadata": self.metadata
        }


def calculate_overall_confidence(confidences: List[float]) -> float:
    """Calculate weighted average confidence."""
    if not confidences:
        return 1.0
    return round(sum(confidences) / len(confidences), 2)


def determine_severity(severities: List[str]) -> str:
    """Return highest severity level."""
    order = ["error", "warning", "suggestion", "info"]
    for sev in severities:
        if sev in severities:
            return sev
    return "info"
