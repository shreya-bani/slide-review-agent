"""
Combined Analysis Pipeline
Orchestrates unified style analysis (grammar + word-list + tone) to produce a comprehensive report.

Usage:
    python -m backend.analyzers.combine_analysis <normalized.json> [output.json]

Returns a detailed JSON report with:
- Document metadata
- Grammar issues (numerals, spacing, quotes, headings, titles, bullets)
- Word-list issues (terminology compliance)
- Tone issues (positive language, active voice with LLM suggestions)
- File naming convention check
- Summary statistics
- Processing timestamps

Architecture:
- Uses unified style_orchestrator which runs all analyzers with shared protection layer
- Protection data is computed once and reused across all analyzers for efficiency
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .style_orchestrator import check_document as run_grammar_check
from .file_naming_check import FileNamingChecker
from .category_formatter_analyzer import analyze_formatting
from .models import Category

logger = logging.getLogger(__name__)


class CombinedAnalyzer:
    """
    Orchestrates unified style analysis (grammar + word-list + tone) to produce a comprehensive report.
    Uses the style_orchestrator which ensures all analyzers share the same protection data.
    """

    def __init__(self):
        # Initialize filename checker (doesn't need protection data)
        try:
            self.filename_checker = FileNamingChecker()
        except Exception as e:
            logger.warning(f"Failed to initialize FileNamingChecker: {e}. Filename checks will be skipped.")
            self.filename_checker = None

        # Note: All style analyzers (grammar, word-list, tone) are now unified in style_orchestrator
        # They automatically share protection data for efficiency

    def analyze(self, normalized_doc: Dict[str, Any],
                input_filepath: Optional[str] = None,
                original_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete analysis pipeline on normalized document.

        Args:
            normalized_doc: Normalized document structure
            input_filepath: Optional path to input file for metadata
            original_filename: Optional original filename (before normalization)

        Returns:
            Comprehensive analysis report
        """
        start_time = datetime.now()

        logger.info("Starting combined analysis pipeline...")

        # Extract document metadata
        doc_metadata = self._extract_document_metadata(normalized_doc, input_filepath, original_filename)

        # Run unified style check (grammar + word-list + tone with shared protection layer)
        logger.info("Running unified style analysis (grammar + word-list + tone)...")
        style_start = datetime.now()
        style_result = run_grammar_check(normalized_doc, include_tone=True, return_wrapper=True)
        all_style_issues = style_result.get("issues", [])
        protection_data = style_result.get("protection_data", {})
        style_duration = (datetime.now() - style_start).total_seconds()

        # Separate issues by category for reporting
        grammar_issues = [i for i in all_style_issues if i.get("category") in [Category.GRAMMAR.value, Category.WORD_LIST.value]]
        tone_issues = [i for i in all_style_issues if i.get("category") == Category.TONE.value]

        grammar_duration = style_duration * 0.4  # Approximate split for backward compatibility
        tone_duration = style_duration * 0.6

        logger.info(f"Style analysis complete: {len(grammar_issues)} grammar/word-list issues, "
                   f"{len(tone_issues)} tone issues in {style_duration:.2f}s")

        # Run category formatting check
        logger.info("Running category formatting check...")
        formatting_start = datetime.now()
        formatting_issues_raw = analyze_formatting(normalized_doc)
        formatting_issues = [issue.to_dict() for issue in formatting_issues_raw]
        formatting_duration = (datetime.now() - formatting_start).total_seconds()
        logger.info(f"Category formatting check complete: {len(formatting_issues)} issues in {formatting_duration:.2f}s")

        # Run file naming check
        logger.info("Running file naming check...")
        filename_start = datetime.now()
        filename_result = self._check_filename(doc_metadata)
        filename_duration = (datetime.now() - filename_start).total_seconds()
        logger.info(f"File naming check complete in {filename_duration:.2f}s")

        # Convert filename check to issues format
        filename_issues = self._convert_filename_to_issues(filename_result)

        # Combine all issues
        all_issues = grammar_issues + tone_issues + formatting_issues + filename_issues

        # Generate statistics
        stats = self._generate_statistics(grammar_issues, tone_issues, formatting_issues, filename_issues, normalized_doc)
        
        # Build comprehensive report
        total_duration = (datetime.now() - start_time).total_seconds()
        
        report = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_duration_seconds": round(total_duration, 2),
                "grammar_duration_seconds": round(grammar_duration, 2),
                "tone_duration_seconds": round(tone_duration, 2),
                "formatting_duration_seconds": round(formatting_duration, 2),
                "filename_duration_seconds": round(filename_duration, 2),
                "analyzer_version": "1.0.0",
            },
            "summary": {
                "total_issues": len(all_issues),
                "grammar_issues": len(grammar_issues),
                "tone_issues": len(tone_issues),
                "formatting_issues": len(formatting_issues),
                "filename_issues": len(filename_issues),
                "issues_with_suggestions": sum(1 for i in all_issues if i.get("suggestion")),
                "severity_breakdown": stats["severity_breakdown"],
                "category_breakdown": stats["category_breakdown"],
                "rule_breakdown": stats["rule_breakdown"],
            },
            "content_statistics": stats["content_stats"],
            "issues_by_category": {
                "grammar": self._categorize_issues(grammar_issues),
                "tone": self._categorize_issues(tone_issues),
                "formatting": self._categorize_issues(formatting_issues),
                "filename": self._categorize_issues(filename_issues),
            },
            "all_issues": all_issues,
            "issues_by_slide": self._group_by_slide(all_issues),
            "document_metadata": doc_metadata,
            "filename_check": filename_result,
        }
        
        logger.info(f"Analysis complete: {len(all_issues)} total issues in {total_duration:.2f}s")
        return report

    def _extract_document_metadata(self, doc: Dict[str, Any],
                                   filepath: Optional[str],
                                   original_filename: Optional[str] = None) -> Dict[str, Any]:
        """Extract comprehensive document metadata."""
        metadata = doc.get("metadata", {}) or {}

        return {
            "file_path": filepath or metadata.get("file_path"),
            "original_filename": original_filename,  # Store original filename separately
            "document_type": doc.get("document_type"),
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "creator": metadata.get("creator"),
            "creation_date": metadata.get("creation_date"),
            "total_pages": metadata.get("total_pages") or doc.get("extraction_info", {}).get("total_pages"),
            "normalized_at": doc.get("normalized_at"),
        }

    def _generate_statistics(self, grammar_issues: List[Dict],
                            tone_issues: List[Dict],
                            formatting_issues: List[Dict],
                            filename_issues: List[Dict],
                            doc: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistics."""
        all_issues = grammar_issues + tone_issues + formatting_issues + filename_issues
        
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

    def _group_by_slide(self, issues: List[Dict]) -> Dict[int, List[Dict]]:
        """Group all issues by slide/page index."""
        by_slide = {}
        for issue in issues:
            slide_idx = issue.get("page_or_slide_index", -1)
            if slide_idx not in by_slide:
                by_slide[slide_idx] = []
            by_slide[slide_idx].append(issue)

        # Sort by slide index
        return {k: by_slide[k] for k in sorted(by_slide.keys())}

    def _check_filename(self, doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the document filename against Amida naming conventions.

        Args:
            doc_metadata: Document metadata containing original_filename

        Returns:
            Dictionary with filename check results
        """
        # Check if filename checker is available
        if self.filename_checker is None:
            logger.warning("FileNamingChecker not available, skipping filename check")
            return {
                "checked": False,
                "reason": "FileNamingChecker not initialized",
                "original_filename": None,
                "suggestions": None
            }

        # Try to get original filename first, fallback to file_path
        original_filename = doc_metadata.get("original_filename")

        if not original_filename:
            # Fallback to extracting from file_path
            file_path = doc_metadata.get("file_path")
            if file_path:
                from pathlib import Path
                original_filename = Path(file_path).name
            else:
                logger.warning("No original_filename or file_path in document metadata, skipping filename check")
                return {
                    "checked": False,
                    "reason": "No filename available",
                    "original_filename": None,
                    "suggestions": None
                }

        logger.info(f"Checking filename: {original_filename}")

        try:
            # Use concise mode to get just the 4 options
            suggestions = self.filename_checker.check_filename(original_filename, concise=True)

            if suggestions:
                # Parse the suggestions into a list (they should be line-separated)
                suggestion_list = [s.strip() for s in suggestions.strip().split('\n') if s.strip()]

                return {
                    "checked": True,
                    "original_filename": original_filename,
                    "suggestions": suggestion_list,
                    "compliant": self._is_filename_compliant(original_filename, suggestion_list)
                }
            else:
                logger.warning("LLM returned no filename suggestions")
                return {
                    "checked": False,
                    "reason": "LLM returned no response",
                    "original_filename": original_filename,
                    "suggestions": None
                }

        except Exception as e:
            logger.error(f"Error checking filename: {e}")
            return {
                "checked": False,
                "reason": f"Error: {str(e)}",
                "original_filename": original_filename,
                "suggestions": None
            }

    def _is_filename_compliant(self, original: str, suggestions: List[str]) -> bool:
        """
        Determine if the original filename is already compliant.

        A filename is considered compliant if it matches one of the suggestions
        or if all suggestions are very similar to the original (indicating minor issues).

        Args:
            original: Original filename
            suggestions: List of suggested filenames

        Returns:
            True if filename is compliant, False otherwise
        """
        if not suggestions:
            return False

        # Normalize for comparison (lowercase, strip whitespace)
        original_normalized = original.lower().strip()

        # Check if original matches any suggestion exactly
        for suggestion in suggestions:
            if suggestion.lower().strip() == original_normalized:
                return True

        # If no exact match, the filename needs correction
        return False

    def _convert_filename_to_issues(self, filename_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert filename check result into issues format for findings table.

        Args:
            filename_result: Result from _check_filename()

        Returns:
            List of issue dictionaries in standard format
        """
        issues = []

        if not filename_result.get("checked", False):
            # If check failed, return empty list
            return issues

        # Only create an issue if filename is not compliant
        if not filename_result.get("compliant", True):
            original_filename = filename_result.get("original_filename", "Unknown")
            suggestions = filename_result.get("suggestions", [])

            # Create a suggestion string with all options (with newlines for frontend display)
            suggestion_text = "\n".join([f"Option {i+1}: {s}" for i, s in enumerate(suggestions)])

            issue = {
                "category": Category.FILENAME.value,
                "rule_name": "filename-convention",
                "severity": "warning",
                "page_or_slide_index": -1,  # Not tied to a specific slide
                "element_index": -1,
                "found_text": original_filename,  # Frontend expects 'found_text'
                "description": "The filename does not fully comply with Amida's naming policy (ADMIN-POL-1-1). Common issues include missing or misplaced hyphens, incorrect date format, extra words (e.g., 'for [name]'), or missing owner initials. The assistant analyzed these patterns and suggested corrected versions based on standard rules.",  # Frontend expects 'description'
                "suggestion": suggestion_text if suggestions else None,
                "location": "Document filename",
                "element_type": "filename"
            }

            issues.append(issue)

        return issues

def main():
    """Simple main function to run analysis from command line."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python combine_analysis.py <input_json> [output_json]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
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
        output_file = input_file.parent / f"{base_name}_combined_analysis.json"
    
    try:
        # Load normalized document
        logger.info(f"Loading document: {input_file}")
        with input_file.open("r", encoding="utf-8") as f:
            normalized_doc = json.load(f)
        
        # Run combined analysis
        analyzer = CombinedAnalyzer()
        report = analyzer.analyze(normalized_doc, str(input_file))
        
        # Save report
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Analysis report saved to: {output_file}")
        
        # Print summary
        summary = report["summary"]
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Issues:          {summary['total_issues']}")
        print(f"  Grammar Issues:      {summary['grammar_issues']}")
        print(f"  Tone Issues:         {summary['tone_issues']}")
        print(f"  Formatting Issues:   {summary['formatting_issues']}")
        print(f"  Filename Issues:     {summary['filename_issues']}")
        print(f"With Suggestions:      {summary['issues_with_suggestions']}")
        print("\nBy Severity:")
        for sev, count in summary['severity_breakdown'].items():
            print(f"  {sev:15s}: {count}")
        print("\nBy Rule:")
        for rule, count in sorted(summary['rule_breakdown'].items(), 
                                  key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {rule:20s}: {count}")
        print("="*60)
        print(f"\nFull report saved: {output_file}")
        
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