"""
Combined Analysis Pipeline
Orchestrates grammar and tone analysis to produce a comprehensive report.

Usage:
    python -m backend.analyzers.combine_analysis <normalized.json> [output.json]
    
Returns a detailed JSON report with:
- Document metadata
- Grammar issues (numerals, spacing, quotes, word list, headings, titles, bullets)
- Tone issues (positive language, active voice with LLM suggestions)
- Summary statistics
- Processing timestamps
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .style_orchestrator import check_document as run_grammar_check
from .tone_analyzer import AdvancedStyleAnalyzer

logger = logging.getLogger(__name__)


class CombinedAnalyzer:
    """
    Orchestrates grammar and tone analysis to produce a unified report.
    """

    def __init__(self):
        self.tone_analyzer = AdvancedStyleAnalyzer()

    def analyze(self, normalized_doc: Dict[str, Any], 
                input_filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete analysis pipeline on normalized document.
        
        Args:
            normalized_doc: Normalized document structure
            input_filepath: Optional path to input file for metadata
            
        Returns:
            Comprehensive analysis report
        """
        start_time = datetime.now()
        
        logger.info("Starting combined analysis pipeline...")
        
        # Extract document metadata
        doc_metadata = self._extract_document_metadata(normalized_doc, input_filepath)
        
        # Run grammar analysis first
        logger.info("Running grammar analysis...")
        grammar_start = datetime.now()
        grammar_issues = run_grammar_check(normalized_doc)
        grammar_duration = (datetime.now() - grammar_start).total_seconds()
        logger.info(f"Grammar analysis complete: {len(grammar_issues)} issues found in {grammar_duration:.2f}s")
        
        # Run tone analysis second
        logger.info("Running tone analysis...")
        tone_start = datetime.now()
        tone_result = self.tone_analyzer.analyze(normalized_doc)
        tone_duration = (datetime.now() - tone_start).total_seconds()
        tone_issues = tone_result.get("issues", [])
        logger.info(f"Tone analysis complete: {len(tone_issues)} issues found in {tone_duration:.2f}s")
        
        # Combine all issues
        all_issues = grammar_issues + tone_issues
        
        # Generate statistics
        stats = self._generate_statistics(grammar_issues, tone_issues, normalized_doc)
        
        # Build comprehensive report
        total_duration = (datetime.now() - start_time).total_seconds()
        
        report = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_duration_seconds": round(total_duration, 2),
                "grammar_duration_seconds": round(grammar_duration, 2),
                "tone_duration_seconds": round(tone_duration, 2),
                "analyzer_version": "1.0.0",
            },
            "summary": {
                "total_issues": len(all_issues),
                "grammar_issues": len(grammar_issues),
                "tone_issues": len(tone_issues),
                "issues_with_suggestions": sum(1 for i in all_issues if i.get("suggestion")),
                "severity_breakdown": stats["severity_breakdown"],
                "category_breakdown": stats["category_breakdown"],
                "rule_breakdown": stats["rule_breakdown"],
            },
            "content_statistics": stats["content_stats"],
            "issues_by_category": {
                "grammar": self._categorize_issues(grammar_issues),
                "tone": self._categorize_issues(tone_issues),
            },
            "all_issues": all_issues,
            "issues_by_slide": self._group_by_slide(all_issues),
            "document_metadata": doc_metadata,
        }
        
        logger.info(f"Analysis complete: {len(all_issues)} total issues in {total_duration:.2f}s")
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

    def _generate_statistics(self, grammar_issues: List[Dict], 
                            tone_issues: List[Dict],
                            doc: Dict[str, Any]) -> Dict[str, Any]:
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
        print(f"Total Issues:        {summary['total_issues']}")
        print(f"  Grammar Issues:    {summary['grammar_issues']}")
        print(f"  Tone Issues:       {summary['tone_issues']}")
        print(f"With Suggestions:    {summary['issues_with_suggestions']}")
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