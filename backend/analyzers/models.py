"""
Shared data models for the analyzer modules.

This module contains the core data structures used across all analyzers:
- Severity: Enum for issue severity levels
- Category: Enum for issue categories (extensible for future categories)
- StyleIssue: Dataclass representing a style/grammar issue found in a document
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict


class Severity(Enum):
    """Severity levels for style issues."""
    ERROR = "error"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    INFO = "info"


class Category(Enum):
    """Categories for style issues. Add new categories here as needed."""
    GRAMMAR = "grammar"
    WORD_LIST = "word-list"
    TONE = "tone"
    FILENAME = "filename"
    # Add more categories as you extend the analyzer:
    # FORMATTING = "formatting"
    # ACCESSIBILITY = "accessibility"


@dataclass
class StyleIssue:
    """
    Represents a single style or grammar issue found during document analysis.

    Attributes:
        rule_name: Identifier for the rule that detected this issue
        severity: How critical this issue is (ERROR, WARNING, SUGGESTION, INFO)
        category: Which category of rule detected this (GRAMMAR, WORD_LIST, etc.)
        description: Human-readable explanation of the issue
        location: Human-readable location string (e.g., "slide 3 - element 2")
        found_text: The original text containing the issue
        suggestion: Suggested correction or improved text
        page_or_slide_index: Zero-based index of the page/slide
        element_index: Zero-based index of the element within the page/slide
        confidence: Confidence score (0.0 to 1.0) of the detection
        method: Detection method used (e.g., "rule-based", "llm", "hybrid")
    """
    rule_name: str
    severity: Severity
    category: Category
    description: str
    location: str
    found_text: str
    suggestion: str
    page_or_slide_index: int
    element_index: int
    confidence: float = 1.0
    method: str = "rule-based"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the StyleIssue to a dictionary with enum values serialized."""
        d = asdict(self)
        d["severity"] = self.severity.value
        d["category"] = self.category.value
        return d
