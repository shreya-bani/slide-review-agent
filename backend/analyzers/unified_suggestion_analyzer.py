"""
Unified Suggestion Analyzer
Generates comprehensive, unified rewrite suggestions combining all style issues.

This analyzer orchestrates grammar, tone, and wordlist analysis, then produces
unified suggestions per text element that address ALL detected issues simultaneously.

Key Features:
- Reuses existing analyzers (simple_style_checker + advance_style_analyzer)
- Groups issues by text location (page/slide + element)
- Generates single unified suggestion per text element via LLM
- Provides detailed explanations of all fixes applied

Usage:
    python -m backend.analyzers.unified_suggestion_analyzer <normalized.json> [output.json]

Returns a JSON report with:
- Document metadata
- All detected issues (grammar + tone + wordlist)
- Unified suggestions grouped by text element
- Processing statistics and timestamps
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from .simple_style_checker import check_document as run_grammar_check
from .advance_style_analyzer import AdvancedStyleAnalyzer
from ..config.settings import settings
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class UnifiedSuggestion:
    """
    Represents a unified suggestion for a text element addressing all issues.
    """
    page_or_slide_index: int
    element_index: int
    original_text: str
    suggested_text: str
    issues_addressed: List[Dict[str, Any]]
    explanation: str
    total_issues: int
    issue_categories: List[str]
    confidence: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UnifiedSuggestionAnalyzer:
    """
    Orchestrates all analyzers and generates unified suggestions per text element.
    """

    def __init__(self, enable_llm_rewrite: bool = True):
        """
        Initialize the unified analyzer.

        Args:
            enable_llm_rewrite: Whether to use LLM for unified rewrite generation
        """
        self.tone_analyzer = AdvancedStyleAnalyzer()
        self.enable_llm_rewrite = enable_llm_rewrite
        self.llm_client = None

        # Initialize LLM client if rewrites are enabled
        if self.enable_llm_rewrite:
            try:
                settings.validate_llm_config()
                self.llm_client = LLMClient()
                logger.info("LLM client initialized for unified rewrites")
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}. Unified rewrites disabled.")
                self.enable_llm_rewrite = False

    def analyze(self, normalized_doc: Dict[str, Any],
                input_filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete unified analysis pipeline.

        Args:
            normalized_doc: Normalized document structure from processors
            input_filepath: Optional path to input file for metadata

        Returns:
            Comprehensive analysis report with unified suggestions
        """
        start_time = datetime.now()
        logger.info("Starting unified suggestion analysis pipeline...")

        # Extract document metadata
        doc_metadata = self._extract_document_metadata(normalized_doc, input_filepath)

        # Step 1: Run grammar analysis
        logger.info("Running grammar analysis...")
        grammar_start = datetime.now()
        grammar_issues = run_grammar_check(normalized_doc)
        grammar_duration = (datetime.now() - grammar_start).total_seconds()
        logger.info(f"Grammar analysis: {len(grammar_issues)} issues in {grammar_duration:.2f}s")

        # Step 2: Run tone analysis
        logger.info("Running tone analysis...")
        tone_start = datetime.now()
        tone_result = self.tone_analyzer.analyze(normalized_doc)
        tone_duration = (datetime.now() - tone_start).total_seconds()
        tone_issues = tone_result.get("issues", [])
        logger.info(f"Tone analysis: {len(tone_issues)} issues in {tone_duration:.2f}s")

        # Step 3: Combine all issues
        all_issues = grammar_issues + tone_issues

        # Step 4: Group issues by text element location
        logger.info("Grouping issues by text element...")
        grouped_issues = self._group_issues_by_element(all_issues)
        logger.info(f"Grouped {len(all_issues)} issues into {len(grouped_issues)} text elements")

        # Step 5: Generate unified suggestions
        logger.info("Generating unified suggestions...")
        unified_start = datetime.now()
        unified_suggestions = self._generate_unified_suggestions(
            grouped_issues,
            normalized_doc
        )
        unified_duration = (datetime.now() - unified_start).total_seconds()
        logger.info(f"Generated {len(unified_suggestions)} unified suggestions in {unified_duration:.2f}s")

        # Step 6: Generate statistics
        stats = self._generate_statistics(grammar_issues, tone_issues, unified_suggestions, normalized_doc)

        # Build comprehensive report
        total_duration = (datetime.now() - start_time).total_seconds()

        report = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_duration_seconds": round(total_duration, 2),
                "grammar_duration_seconds": round(grammar_duration, 2),
                "tone_duration_seconds": round(tone_duration, 2),
                "unified_generation_duration_seconds": round(unified_duration, 2),
                "analyzer_version": "1.0.0",
                "analyzer_type": "unified_suggestion",
            },
            "summary": {
                "total_issues": len(all_issues),
                "grammar_issues": len(grammar_issues),
                "tone_issues": len(tone_issues),
                "elements_with_issues": len(grouped_issues),
                "unified_suggestions_generated": len(unified_suggestions),
                "severity_breakdown": stats["severity_breakdown"],
                "category_breakdown": stats["category_breakdown"],
                "rule_breakdown": stats["rule_breakdown"],
            },
            "content_statistics": stats["content_stats"],
            "unified_suggestions": [s.to_dict() for s in unified_suggestions],
            "unified_suggestions_by_slide": self._group_suggestions_by_slide(unified_suggestions),
            "all_issues_detailed": all_issues,
            "issues_by_category": {
                "grammar": self._categorize_issues(grammar_issues),
                "tone": self._categorize_issues(tone_issues),
            },
            "document_metadata": doc_metadata,
        }

        logger.info(
            f"Unified analysis complete: {len(all_issues)} issues → "
            f"{len(unified_suggestions)} unified suggestions in {total_duration:.2f}s"
        )
        return report

    def _extract_document_metadata(self, doc: Dict[str, Any],
                                   filepath: Optional[str]) -> Dict[str, Any]:
        """Extract comprehensive document metadata."""
        metadata = doc.get("metadata", {}) or {}

        return {
            "file_path": filepath or metadata.get("file_path"),
            "document_type": doc.get("document_type"),
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "creator": metadata.get("creator"),
            "creation_date": metadata.get("creation_date"),
            "total_pages": metadata.get("total_pages") or doc.get("extraction_info", {}).get("total_pages"),
            "normalized_at": doc.get("normalized_at"),
        }

    def _group_issues_by_element(self, issues: List[Dict[str, Any]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
        """
        Group all issues by their text element location.

        Returns:
            Dict mapping (page_index, element_index) -> list of issues
        """
        grouped = defaultdict(list)

        for issue in issues:
            page_idx = issue.get("page_or_slide_index", -1)
            elem_idx = issue.get("element_index", -1)
            key = (page_idx, elem_idx)
            grouped[key].append(issue)

        return dict(grouped)

    def _generate_unified_suggestions(
        self,
        grouped_issues: Dict[Tuple[int, int], List[Dict[str, Any]]],
        normalized_doc: Dict[str, Any]
    ) -> List[UnifiedSuggestion]:
        """
        Generate unified suggestions for each text element with issues.

        Args:
            grouped_issues: Issues grouped by (page_idx, element_idx)
            normalized_doc: Original normalized document

        Returns:
            List of UnifiedSuggestion objects
        """
        suggestions = []

        # Build element lookup table for fast access
        element_map = self._build_element_map(normalized_doc)

        for (page_idx, elem_idx), issues in grouped_issues.items():
            # Get original text
            original_text = self._get_element_text(element_map, page_idx, elem_idx)
            if not original_text or not original_text.strip():
                logger.warning(f"No text found for element at page={page_idx}, elem={elem_idx}")
                continue

            # Generate unified suggestion
            suggestion = self._create_unified_suggestion(
                page_idx,
                elem_idx,
                original_text,
                issues
            )

            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _build_element_map(self, doc: Dict[str, Any]) -> Dict[Tuple[int, int], str]:
        """Build a map of (page_idx, elem_idx) -> text for fast lookup."""
        element_map = {}

        pages = doc.get("pages", doc.get("slides", []))
        for page_idx, page in enumerate(pages):
            elements = page.get("elements", [])
            for elem_idx, element in enumerate(elements):
                text = element.get("text", "")
                if isinstance(text, str) and text.strip():
                    element_map[(page_idx, elem_idx)] = text.strip()

        return element_map

    def _get_element_text(self, element_map: Dict[Tuple[int, int], str],
                         page_idx: int, elem_idx: int) -> Optional[str]:
        """Get text for a specific element."""
        return element_map.get((page_idx, elem_idx))

    def _create_unified_suggestion(
        self,
        page_idx: int,
        elem_idx: int,
        original_text: str,
        issues: List[Dict[str, Any]]
    ) -> Optional[UnifiedSuggestion]:
        """
        Create a unified suggestion addressing all issues for a text element.

        Args:
            page_idx: Page/slide index
            elem_idx: Element index within page
            original_text: Original text content
            issues: All issues detected for this element

        Returns:
            UnifiedSuggestion or None if generation fails
        """
        if not issues:
            return None

        # Collect all issue details
        issue_categories = list(set(issue.get("category", "unknown") for issue in issues))
        total_issues = len(issues)

        # Attempt LLM-based unified rewrite
        if self.enable_llm_rewrite and self.llm_client:
            suggested_text, explanation = self._llm_unified_rewrite(
                original_text,
                issues
            )
        else:
            # Fallback: use existing suggestions if available
            suggested_text, explanation = self._fallback_unified_rewrite(
                original_text,
                issues
            )

        # If no improvement, skip
        if not suggested_text or suggested_text.strip() == original_text.strip():
            logger.debug(f"No unified suggestion generated for element at page={page_idx}, elem={elem_idx}")
            return None

        return UnifiedSuggestion(
            page_or_slide_index=page_idx,
            element_index=elem_idx,
            original_text=original_text,
            suggested_text=suggested_text,
            issues_addressed=issues,
            explanation=explanation,
            total_issues=total_issues,
            issue_categories=issue_categories,
            confidence=0.9 if self.enable_llm_rewrite else 0.7
        )

    def _llm_unified_rewrite(
        self,
        original_text: str,
        issues: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Use LLM to generate unified rewrite addressing all issues.

        Args:
            original_text: Original text
            issues: All detected issues

        Returns:
            (suggested_text, explanation)
        """
        # Build comprehensive issue description
        issue_descriptions = []
        for idx, issue in enumerate(issues, 1):
            rule = issue.get("rule_name", "unknown")
            category = issue.get("category", "unknown")
            desc = issue.get("description", "")
            found = issue.get("found_text", "")

            issue_desc = f"{idx}. [{category}] {rule}: {desc}"
            if found and found != original_text:
                issue_desc += f" (found: '{found}')"
            issue_descriptions.append(issue_desc)

        issues_text = "\n".join(issue_descriptions)

        # Construct LLM prompt
        prompt = f"""You are an expert editor reviewing content according to Amida's Style Guide.

ORIGINAL TEXT:
"{original_text}"

DETECTED ISSUES:
{issues_text}

TASK:
Rewrite the text to address ALL issues listed above while:
1. Preserving the core meaning and intent
2. Maintaining factual accuracy (names, dates, numbers, technical terms)
3. Keeping the same tone and formality level
4. Following Amida Style Guide rules (active voice, positive language, proper grammar)

Provide your response in exactly this format:

REWRITTEN TEXT:
[Your rewritten version here]

EXPLANATION:
[Brief explanation of changes made to address each issue]

Begin your response now:"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat(messages)

            if not response:
                logger.warning("Empty LLM response for unified rewrite")
                return original_text, "LLM rewrite failed (empty response)"

            # Parse response
            suggested_text, explanation = self._parse_llm_rewrite_response(response)

            # Validate suggestion
            if not suggested_text or len(suggested_text.strip()) < 3:
                logger.warning(f"Invalid LLM suggestion: '{suggested_text}'")
                return original_text, "LLM rewrite failed (invalid output)"

            # Basic sanity check: ensure not too different
            if len(suggested_text) > len(original_text) * 2:
                logger.warning(f"LLM suggestion too long ({len(suggested_text)} vs {len(original_text)} chars)")
                return original_text, "LLM rewrite rejected (output too long)"

            return suggested_text, explanation or "Unified rewrite addressing all detected issues"

        except Exception as e:
            logger.error(f"LLM unified rewrite error: {e}")
            return original_text, f"LLM rewrite failed: {str(e)}"

    def _parse_llm_rewrite_response(self, response: str) -> Tuple[str, str]:
        """
        Parse LLM response to extract rewritten text and explanation.

        Returns:
            (rewritten_text, explanation)
        """
        # Look for structured sections
        rewritten_match = None
        explanation_match = None

        # Pattern 1: REWRITTEN TEXT: ... EXPLANATION: ...
        import re
        rewritten_pattern = re.compile(r'REWRITTEN TEXT:\s*(.+?)(?:EXPLANATION:|$)', re.DOTALL | re.IGNORECASE)
        explanation_pattern = re.compile(r'EXPLANATION:\s*(.+)', re.DOTALL | re.IGNORECASE)

        rewritten_match = rewritten_pattern.search(response)
        explanation_match = explanation_pattern.search(response)

        if rewritten_match:
            rewritten = rewritten_match.group(1).strip()
        else:
            # Fallback: use entire response as rewrite
            lines = response.strip().split('\n')
            rewritten = '\n'.join(lines[:3]).strip()  # Take first few lines

        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            explanation = "Applied multiple style guide corrections"

        return rewritten, explanation

    def _fallback_unified_rewrite(
        self,
        original_text: str,
        issues: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Fallback rewrite using existing individual suggestions.

        Args:
            original_text: Original text
            issues: All detected issues

        Returns:
            (suggested_text, explanation)
        """
        # Collect all existing suggestions
        suggestions = [
            issue.get("suggestion", "")
            for issue in issues
            if issue.get("suggestion")
        ]

        if not suggestions:
            return original_text, "No individual suggestions available"

        # Simple heuristic: use the longest suggestion (likely most comprehensive)
        # or the first non-empty one
        best_suggestion = max(suggestions, key=len, default=original_text)

        issue_count = len(issues)
        categories = ", ".join(set(issue.get("category", "unknown") for issue in issues))

        explanation = (
            f"Combined {issue_count} issue(s) across {categories}. "
            "Using best available individual suggestion."
        )

        return best_suggestion, explanation

    def _generate_statistics(
        self,
        grammar_issues: List[Dict],
        tone_issues: List[Dict],
        unified_suggestions: List[UnifiedSuggestion],
        doc: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive statistics."""
        all_issues = grammar_issues + tone_issues

        # Severity breakdown
        severity_counts = {}
        for issue in all_issues:
            sev = issue.get("severity", "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Category breakdown
        category_counts = {}
        for issue in all_issues:
            cat = issue.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Rule breakdown
        rule_counts = {}
        for issue in all_issues:
            rule = issue.get("rule_name", "unknown")
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

        # Content statistics
        extraction_info = doc.get("extraction_info", {}) or {}
        pages = doc.get("pages", []) or []

        total_text_length = 0
        total_elements = 0
        element_types = {}

        for page in pages:
            for elem in page.get("elements", []):
                total_elements += 1
                text = elem.get("text", "") or ""
                total_text_length += len(text)

                etype = (elem.get("locator", {}) or {}).get("element_type", "unknown")
                element_types[etype] = element_types.get(etype, 0) + 1

        # Unified suggestion stats
        avg_issues_per_suggestion = (
            sum(s.total_issues for s in unified_suggestions) / len(unified_suggestions)
            if unified_suggestions else 0
        )

        return {
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "rule_breakdown": rule_counts,
            "content_stats": {
                "total_pages": extraction_info.get("total_pages", len(pages)),
                "total_elements": total_elements,
                "total_text_length": total_text_length,
                "element_types": element_types,
                "avg_elements_per_page": round(total_elements / max(len(pages), 1), 2),
                "elements_with_suggestions": len(unified_suggestions),
                "avg_issues_per_suggestion": round(avg_issues_per_suggestion, 2),
            }
        }

    def _categorize_issues(self, issues: List[Dict]) -> Dict[str, List[Dict]]:
        """Group issues by rule name."""
        categorized = {}
        for issue in issues:
            rule = issue.get("rule_name", "unknown")
            if rule not in categorized:
                categorized[rule] = []
            categorized[rule].append(issue)
        return categorized

    def _group_suggestions_by_slide(
        self,
        suggestions: List[UnifiedSuggestion]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Group unified suggestions by slide/page index."""
        by_slide = defaultdict(list)

        for suggestion in suggestions:
            slide_idx = suggestion.page_or_slide_index
            by_slide[slide_idx].append(suggestion.to_dict())

        # Sort by slide index
        return {k: by_slide[k] for k in sorted(by_slide.keys())}


def main():
    """CLI entry point for unified suggestion analysis."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python unified_suggestion_analyzer.py <input_json> [output_json] [--no-rewrite]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    enable_rewrite = "--no-rewrite" not in sys.argv

    # Validate input
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if output_path:
        output_file = Path(output_path)
    else:
        base_name = input_file.stem
        output_file = input_file.parent / f"{base_name}_unified_analysis.json"

    try:
        # Load normalized document
        logger.info(f"Loading document: {input_file}")
        with input_file.open("r", encoding="utf-8") as f:
            normalized_doc = json.load(f)

        # Run unified analysis
        analyzer = UnifiedSuggestionAnalyzer(enable_llm_rewrite=enable_rewrite)
        report = analyzer.analyze(normalized_doc, str(input_file))

        # Save report
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Unified analysis report saved to: {output_file}")

        # Print summary
        summary = report["summary"]
        print("\n" + "="*70)
        print("UNIFIED SUGGESTION ANALYSIS SUMMARY")
        print("="*70)
        print(f"Total Issues Detected:     {summary['total_issues']}")
        print(f"  Grammar Issues:          {summary['grammar_issues']}")
        print(f"  Tone Issues:             {summary['tone_issues']}")
        print(f"Elements with Issues:      {summary['elements_with_issues']}")
        print(f"Unified Suggestions:       {summary['unified_suggestions_generated']}")
        print("\nBy Severity:")
        for sev, count in summary['severity_breakdown'].items():
            print(f"  {sev:15s}: {count}")
        print("\nBy Category:")
        for cat, count in summary['category_breakdown'].items():
            print(f"  {cat:15s}: {count}")
        print("\nTop Rules Triggered:")
        for rule, count in sorted(summary['rule_breakdown'].items(),
                                  key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {rule:30s}: {count}")
        print("="*70)
        print(f"\nFull report: {output_file}")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
