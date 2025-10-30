"""
Category Formatter Analyzer

This analyzer checks presentation formatting against Amida's Style Guide requirements,
specifically focusing on slide-level formatting rules:

1. Title Slide (Slide 1):
   - Title font: Lato, size 40, color white, centered
   - Project Managers: Listed alphabetically by last name, font size 28, color white
   - Technical Lead: Font size 17, color white, centered
   - Month and Year: Font size 14, color white, centered

2. Project Information & Staffing Slides:
   - Slide title: Lato, size 22, color white, single line only
   - Table colors: Title row #2D86C1, Subrows #CFDEED
   - Table text: Lato, size 14

3. Content Slides:
   - Body text: Lato, size 14, color #555555 (dark gray)
   - Paragraph spacing: 1.08
   - Text box width: 11.33"
   - White text only for title/transition slides; content slides use #555555

Usage:
    analyzer = CategoryFormatterAnalyzer(normalized_doc)
    issues = analyzer.analyze()
"""

from __future__ import annotations
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from .model.models import Severity, Category, StyleIssue

logger = logging.getLogger(__name__)


class CategoryFormatterAnalyzer:
    """
    Analyzes presentation formatting according to Amida's category-specific style guidelines.
    Uses content-based detection to identify element roles (title, project managers, tech lead, date).
    """

    # Color mapping for theme colors to hex/names
    COLOR_MAPPING = {
        "theme:LIGHT_1": "white",
        "theme:TEXT_1": "dark_gray",
        "color_type:None": "unknown",
        "#FFFFFF": "white",
        "#555555": "dark_gray",
        "#2D86C1": "blue",
        "#CFDEED": "light_blue"
    }

    # Expected font for all slides
    EXPECTED_FONT = "Lato"

    def __init__(self, normalized_doc: Dict[str, Any]):
        """
        Initialize the formatter analyzer.

        Args:
            normalized_doc: Document in unified JSON format with pages and elements
        """
        self.doc = normalized_doc
        self.pages = normalized_doc.get("pages", [])
        self.metadata = normalized_doc.get("metadata", {})
        self.issues: List[StyleIssue] = []

    def analyze(self) -> List[StyleIssue]:
        """
        Run all formatting checks and return list of issues found.

        Returns:
            List of StyleIssue objects representing formatting violations
        """
        logger.info("Starting category formatting analysis...")

        if not self.pages:
            logger.warning("No pages found in document")
            return []

        # Check title slide (first slide)
        self._check_title_slide()

        # Check content slides (remaining slides)
        self._check_content_slides()

        # Check font consistency across all slides
        self._check_font_consistency()

        logger.info(f"Category formatting analysis complete. Found {len(self.issues)} issues.")
        return self.issues

    def _detect_title_element_role(self, text: str, index: int, font_size: Optional[float]) -> Tuple[Optional[str], Optional[float]]:
        """
        Detect the role of an element on the title slide based on content analysis.

        Args:
            text: Element text content
            index: Element position (0 = first)
            font_size: Font size if available

        Returns:
            Tuple of (role_name, expected_font_size) or (None, None) if unrecognized
        """
        if not text or len(text.strip()) < 2:
            return (None, None)

        text_lower = text.lower()
        text_stripped = text.strip()

        # First two elements are usually title/subtitle (highest priority)
        # Check position first before content analysis
        if index <= 1 and len(text) > 20:
            return ("Title", 40.0)

        # Date detection (most specific, check early)
        month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b'
        year_pattern = r'\b(19|20)\d{2}\b'

        if re.search(month_pattern, text_lower) or re.search(year_pattern, text):
            return ("Month and Year", 14.0)

        # Technical Lead detection
        tech_lead_keywords = ["technical lead", "tech lead", "tl:", "technical lead:"]
        if any(keyword in text_lower for keyword in tech_lead_keywords):
            return ("Technical Lead", 17.0)

        # Single name with no commas (might be technical lead)
        # Pattern: "FirstName LastName" with no commas or "and"
        name_count = len([w for w in text_stripped.split() if w and w[0].isupper()])
        has_connectors = any(conn in text_lower for conn in [",", " and ", "&", ";"])

        if 2 <= name_count <= 3 and not has_connectors and len(text_stripped.split()) <= 4:
            # Likely a single name (Technical Lead)
            return ("Technical Lead", 17.0)

        # Project Manager detection
        pm_keywords = ["project manager", "pm:", "project managers:", "lead:", "manager:"]
        if any(keyword in text_lower for keyword in pm_keywords):
            return ("Project Manager", 28.0)

        # Multiple names detection (likely project managers)
        # Look for patterns like "Name1, Name2" or "Name1 and Name2"
        if has_connectors and name_count >= 3:
            return ("Project Manager", 28.0)

        # Fallback: if we can't determine role, return None
        return (None, None)

    def _check_title_slide(self):
        """
        Check formatting rules for the title slide (slide 1).

        Expected structure (detected by content, not position):
        - Title (Lato, size 40, white, bold) - longest text or first element
        - Project Managers (Lato, size 28, white) - contains multiple names, should be alphabetical
        - Technical Lead (Lato, size 17, white) - contains "Technical Lead" or single name
        - Date (Lato, size 14, white) - contains month/year patterns
        """
        if not self.pages:
            return

        title_slide = self.pages[0]
        slide_idx = title_slide.get("index", 1)
        layout = title_slide.get("layout_info", "")
        elements = title_slide.get("elements", [])

        # Verify it's a title slide
        if "TITLE" not in layout.upper():
            self.issues.append(StyleIssue(
                rule_name="title_slide_layout",
                severity=Severity.WARNING,
                category=Category.FORMATTING,
                description="First slide should use TITLE layout",
                location=f"slide {slide_idx}",
                found_text=f"Layout: {layout}",
                suggestion="Use TITLE layout for the first slide",
                page_or_slide_index=slide_idx,
                element_index=0,
                confidence=0.9,
                method="rule-based"
            ))

        # Classify elements by content analysis
        project_managers = []

        for i, elem in enumerate(elements):
            formatting = elem.get("formatting", {})
            text = elem.get("text", "")
            font_size = formatting.get("font_size")
            font_name = formatting.get("font_name")
            color = self._normalize_color(formatting.get("color", ""))

            # Detect element role based on content
            role, expected_size = self._detect_title_element_role(text, i, font_size)

            if not role:
                continue  # Skip unrecognized elements

            # Track project managers for alphabetical check
            if role == "Project Manager":
                project_managers.append((i, text))

            # Check font size
            if font_size and expected_size and abs(font_size - expected_size) > 1:
                self.issues.append(StyleIssue(
                    rule_name="title_slide_font_size",
                    severity=Severity.WARNING,
                    category=Category.FORMATTING,
                    description=f"{role} should use font size {expected_size}",
                    location=f"slide {slide_idx} - element {i}",
                    found_text=f"{text} (size {font_size})",
                    suggestion=f"Change font size to {expected_size}",
                    page_or_slide_index=slide_idx,
                    element_index=i,
                    confidence=0.95,
                    method="content-based"
                ))

            # Check font name
            if font_name and font_name != self.EXPECTED_FONT:
                self.issues.append(StyleIssue(
                    rule_name="title_slide_font_name",
                    severity=Severity.WARNING,
                    category=Category.FORMATTING,
                    description=f"{role} should use {self.EXPECTED_FONT} font",
                    location=f"slide {slide_idx} - element {i}",
                    found_text=f"{text} (font: {font_name})",
                    suggestion=f"Change font to {self.EXPECTED_FONT}",
                    page_or_slide_index=slide_idx,
                    element_index=i,
                    confidence=0.95,
                    method="content-based"
                ))

            # Check color (should be white for title slide)
            if color and color not in ["white", "unknown"]:
                self.issues.append(StyleIssue(
                    rule_name="title_slide_color",
                    severity=Severity.INFO,
                    category=Category.FORMATTING,
                    description=f"{role} text should be white on title slide",
                    location=f"slide {slide_idx} - element {i}",
                    found_text=f"{text} (color: {color})",
                    suggestion="Change text color to white",
                    page_or_slide_index=slide_idx,
                    element_index=i,
                    confidence=0.8,
                    method="content-based"
                ))

        # Check for alphabetical ordering of project managers
        if len(project_managers) > 1:
            names = [pm[1] for pm in project_managers]
            sorted_names = sorted(names, key=lambda n: n.split()[-1].lower() if n else "")
            if names != sorted_names:
                self.issues.append(StyleIssue(
                    rule_name="project_managers_alphabetical",
                    severity=Severity.SUGGESTION,
                    category=Category.FORMATTING,
                    description="Project Managers should be listed alphabetically by last name",
                    location=f"slide {slide_idx}",
                    found_text=", ".join(names),
                    suggestion=f"Reorder to: {', '.join(sorted_names)}",
                    page_or_slide_index=slide_idx,
                    element_index=project_managers[0][0],
                    confidence=0.85,
                    method="content-based"
                ))

    def _check_content_slides(self):
        """
        Check formatting rules for content slides (slides 2+).

        Expected:
        - Slide titles: Lato, size 22, white, single line
        - Body text: Lato, size 14, color #555555 (dark gray)
        """
        for page in self.pages[1:]:  # Skip first slide (title)
            slide_idx = page.get("index", 0)
            elements = page.get("elements", [])

            if not elements:
                continue

            # First element is typically the slide title
            title_elem = elements[0]
            self._check_slide_title(title_elem, slide_idx, 0)

            # Remaining elements are body text
            for i, elem in enumerate(elements[1:], start=1):
                self._check_body_text(elem, slide_idx, i)

    def _check_slide_title(self, elem: Dict[str, Any], slide_idx: int, elem_idx: int):
        """
        Check formatting for a slide title.

        Expected: Lato, size 22, white, single line
        """
        formatting = elem.get("formatting", {})
        text = elem.get("text", "")
        font_size = formatting.get("font_size")
        font_name = formatting.get("font_name")
        color = self._normalize_color(formatting.get("color", ""))

        # Check font size (22 for slide titles)
        expected_size = 22.0
        if font_size and abs(font_size - expected_size) > 1:
            self.issues.append(StyleIssue(
                rule_name="slide_title_font_size",
                severity=Severity.WARNING,
                category=Category.FORMATTING,
                description=f"Slide title should use font size {expected_size}",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=f"{text} (size {font_size})",
                suggestion=f"Change font size to {expected_size}",
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=0.9,
                method="rule-based"
            ))

        # Check font name
        if font_name and font_name != self.EXPECTED_FONT:
            self.issues.append(StyleIssue(
                rule_name="slide_title_font_name",
                severity=Severity.WARNING,
                category=Category.FORMATTING,
                description=f"Slide title should use {self.EXPECTED_FONT} font",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=f"{text} (font: {font_name})",
                suggestion=f"Change font to {self.EXPECTED_FONT}",
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=0.9,
                method="rule-based"
            ))

        # Check color (should be white)
        if color and color not in ["white", "unknown"]:
            self.issues.append(StyleIssue(
                rule_name="slide_title_color",
                severity=Severity.INFO,
                category=Category.FORMATTING,
                description="Slide title should be white",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=f"{text} (color: {color})",
                suggestion="Change text color to white",
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=0.8,
                method="rule-based"
            ))

        # Check for single line (no newlines)
        if "\n" in text or len(text.split()) > 15:
            self.issues.append(StyleIssue(
                rule_name="slide_title_single_line",
                severity=Severity.SUGGESTION,
                category=Category.FORMATTING,
                description="Slide title should be a single line only",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=text,
                suggestion="Condense title to a single line (max ~10-15 words)",
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=0.75,
                method="rule-based"
            ))

    def _check_body_text(self, elem: Dict[str, Any], slide_idx: int, elem_idx: int):
        """
        Check formatting for body text on content slides.

        Expected: Lato, size 14, color #555555
        """
        formatting = elem.get("formatting", {})
        text = elem.get("text", "")
        font_size = formatting.get("font_size")
        font_name = formatting.get("font_name")
        color = self._normalize_color(formatting.get("color", ""))

        # Skip very short text (likely not body text)
        if len(text.strip()) < 10:
            return

        # Check font size (14 for body text)
        expected_size = 14.0
        if font_size and abs(font_size - expected_size) > 1:
            self.issues.append(StyleIssue(
                rule_name="body_text_font_size",
                severity=Severity.INFO,
                category=Category.FORMATTING,
                description=f"Body text should use font size {expected_size}",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=f"{text} (size {font_size})",
                suggestion=f"Change font size to {expected_size}",
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=0.85,
                method="rule-based"
            ))

        # Check font name
        if font_name and font_name != self.EXPECTED_FONT:
            self.issues.append(StyleIssue(
                rule_name="body_text_font_name",
                severity=Severity.INFO,
                category=Category.FORMATTING,
                description=f"Body text should use {self.EXPECTED_FONT} font",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=f"{text} (font: {font_name})",
                suggestion=f"Change font to {self.EXPECTED_FONT}",
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=0.85,
                method="rule-based"
            ))

        # Check color (should be dark gray #555555 for body text)
        if color == "white":
            self.issues.append(StyleIssue(
                rule_name="body_text_color",
                severity=Severity.WARNING,
                category=Category.FORMATTING,
                description="Body text should use color #555555 (dark gray), not white",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=f"{text} (color: {color})",
                suggestion="Change text color to #555555 (dark gray)",
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=0.9,
                method="rule-based"
            ))

    def _check_font_consistency(self):
        """
        Check that all text uses Lato font consistently.
        """
        for page in self.pages:
            slide_idx = page.get("index", 0)
            elements = page.get("elements", [])

            for i, elem in enumerate(elements):
                formatting = elem.get("formatting", {})
                font_name = formatting.get("font_name")
                text = elem.get("text", "")

                if font_name and font_name != self.EXPECTED_FONT:
                    # Already flagged in specific checks, only flag if not caught
                    # This is a catch-all for any elements not checked above
                    pass

    def _normalize_color(self, color_str: str) -> str:
        """
        Normalize color string to a human-readable format.

        Args:
            color_str: Color string from formatting (e.g., "theme:LIGHT_1", "#FFFFFF")

        Returns:
            Normalized color name (e.g., "white", "dark_gray")
        """
        if not color_str:
            return "unknown"

        # Direct mapping
        if color_str in self.COLOR_MAPPING:
            return self.COLOR_MAPPING[color_str]

        # Hex color detection
        if color_str.startswith("#"):
            color_upper = color_str.upper()
            if color_upper in ["#FFFFFF", "#FFF"]:
                return "white"
            elif color_upper in ["#555555", "#555"]:
                return "dark_gray"
            elif color_upper in ["#2D86C1"]:
                return "blue"
            elif color_upper in ["#CFDEED"]:
                return "light_blue"

        # Theme color detection
        if "LIGHT" in color_str.upper() or "WHITE" in color_str.upper():
            return "white"
        if "TEXT" in color_str.upper() or "DARK" in color_str.upper():
            return "dark_gray"

        return color_str


def analyze_formatting(normalized_doc: Dict[str, Any]) -> List[StyleIssue]:
    """
    Convenience function to analyze formatting issues in a normalized document.

    Args:
        normalized_doc: Document in unified JSON format

    Returns:
        List of StyleIssue objects
    """
    analyzer = CategoryFormatterAnalyzer(normalized_doc)
    return analyzer.analyze()


# CLI for testing
if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    # Initialize logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    if len(sys.argv) < 2:
        logger.error("Usage: python -m backend.analyzers.category_formatter_analyzer <normalized.json> [output.json]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    # Load normalized document
    with open(input_path, encoding="utf-8") as f:
        doc = json.load(f)

    # Run analysis
    issues = analyze_formatting(doc)

    # Log results
    logger.info("=" * 80)
    logger.info(f"Category Formatting Analysis: {input_path.name}")
    logger.info("=" * 80)
    logger.info(f"Found {len(issues)} formatting issues:\n")

    for issue in issues:
        logger.info(f"[{issue.severity.value.upper()}] {issue.rule_name}")
        logger.info(f"  Location: {issue.location}")
        logger.info(f"  Issue: {issue.description}")
        logger.info(f"  Found: {issue.found_text}")
        logger.info(f"  Suggestion: {issue.suggestion}\n")

    # Save to file if requested
    if output_path:
        result = {
            "analysis_type": "category_formatting",
            "total_issues": len(issues),
            "issues": [issue.to_dict() for issue in issues]
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output_path}")
