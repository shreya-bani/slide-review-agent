"""
Compliance Report Merger

Merges template compliance violations into a structured, LLM-friendly format.
Generates comprehensive context about violations with recommendations prompts.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ViolationContext:
    """Enriched violation with full context for LLM analysis."""
    slide_number: int
    violation_type: str
    severity: str
    field: str
    expected: str
    actual: str
    description: str
    context: Dict[str, Any]  # Additional context (surrounding text, slide info, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MergedReport:
    """LLM-ready merged compliance report."""
    document_name: str
    analysis_date: str
    executive_summary: str
    violation_summary: Dict[str, Any]
    violations_by_slide: Dict[int, List[ViolationContext]]
    violations_by_type: Dict[str, List[ViolationContext]]
    critical_issues: List[ViolationContext]
    recommendations_prompt: str
    raw_statistics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_name': self.document_name,
            'analysis_date': self.analysis_date,
            'executive_summary': self.executive_summary,
            'violation_summary': self.violation_summary,
            'violations_by_slide': {
                k: [v.to_dict() for v in vals]
                for k, vals in self.violations_by_slide.items()
            },
            'violations_by_type': {
                k: [v.to_dict() for v in vals]
                for k, vals in self.violations_by_type.items()
            },
            'critical_issues': [v.to_dict() for v in self.critical_issues],
            'recommendations_prompt': self.recommendations_prompt,
            'raw_statistics': self.raw_statistics
        }


class ComplianceReportMerger:
    """
    Merges template compliance report with extracted document data
    to create LLM-ready analysis context.
    """

    def __init__(self):
        self.input_document = None
        self.template_document = None
        self.compliance_report = None

    def merge_report(
        self,
        compliance_report: Dict[str, Any],
        input_document: Dict[str, Any],
        template_document: Optional[Dict[str, Any]] = None
    ) -> MergedReport:
        """
        Merge compliance report with document context.

        Args:
            compliance_report: Output from template_compliance_analyzer
            input_document: The original extracted PPTX JSON
            template_document: Optional template JSON for reference

        Returns:
            MergedReport ready for LLM analysis
        """
        self.compliance_report = compliance_report
        self.input_document = input_document
        self.template_document = template_document

        # Build enriched violations
        enriched_violations = self._enrich_violations()

        # Group violations
        violations_by_slide = self._group_by_slide(enriched_violations)
        violations_by_type = self._group_by_type(enriched_violations)

        # Extract critical issues
        critical_issues = [v for v in enriched_violations if v.severity == 'critical']

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            compliance_report,
            len(enriched_violations),
            len(critical_issues)
        )

        # Generate LLM prompt
        recommendations_prompt = self._generate_llm_prompt(
            enriched_violations,
            violations_by_slide,
            violations_by_type
        )

        # Build violation summary
        violation_summary = {
            'total_violations': compliance_report.get('total_violations', 0),
            'critical': compliance_report.get('violations_by_severity', {}).get('critical', 0),
            'warning': compliance_report.get('violations_by_severity', {}).get('warning', 0),
            'info': compliance_report.get('violations_by_severity', {}).get('info', 0),
            'slides_affected': len(violations_by_slide),
            'compliance_score': compliance_report.get('summary', {}).get('compliance_score', 0)
        }

        merged_report = MergedReport(
            document_name=input_document.get('metadata', {}).get('file_path', 'Unknown'),
            analysis_date=datetime.now().isoformat(),
            executive_summary=executive_summary,
            violation_summary=violation_summary,
            violations_by_slide=violations_by_slide,
            violations_by_type=violations_by_type,
            critical_issues=critical_issues,
            recommendations_prompt=recommendations_prompt,
            raw_statistics=compliance_report.get('summary', {})
        )

        return merged_report

    def _enrich_violations(self) -> List[ViolationContext]:
        """Enrich violations with document context."""
        enriched = []

        violations = self.compliance_report.get('violations', [])
        slides = self.input_document.get('slides', [])

        # Create slide lookup
        slide_map = {s['slide_index']: s for s in slides}

        for violation in violations:
            slide_idx = violation.get('slide_index', 0)
            slide = slide_map.get(slide_idx, {})

            # Build context
            context = {
                'slide_title': slide.get('title'),
                'slide_layout': slide.get('layout_name'),
                'element_index': violation.get('element_index')
            }

            # Add element context if available
            if violation.get('element_type') in ['content_text', 'content', 'formatting']:
                elem_idx = violation.get('element_index')
                if elem_idx is not None and elem_idx < len(slide.get('content_elements', [])):
                    element = slide['content_elements'][elem_idx]
                    context['element_text'] = element.get('text', '')[:100]  # First 100 chars
                    context['element_type'] = element.get('locator', {}).get('element_type')

            # Add table context if table violation
            if 'table' in violation.get('element_type', ''):
                table_idx = violation.get('element_index')
                if table_idx is not None and table_idx < len(slide.get('tables', [])):
                    table = slide['tables'][table_idx]
                    context['table_dimensions'] = f"{table.get('rows')}x{table.get('columns')}"

            enriched_violation = ViolationContext(
                slide_number=slide_idx,
                violation_type=violation.get('element_type', 'unknown'),
                severity=violation.get('severity', 'info'),
                field=violation.get('field_path', ''),
                expected=str(violation.get('expected_value', '')),
                actual=str(violation.get('actual_value', '')),
                description=violation.get('description', ''),
                context=context
            )

            enriched.append(enriched_violation)

        return enriched

    def _group_by_slide(self, violations: List[ViolationContext]) -> Dict[int, List[ViolationContext]]:
        """Group violations by slide number."""
        grouped = {}
        for v in violations:
            if v.slide_number not in grouped:
                grouped[v.slide_number] = []
            grouped[v.slide_number].append(v)
        return grouped

    def _group_by_type(self, violations: List[ViolationContext]) -> Dict[str, List[ViolationContext]]:
        """Group violations by type."""
        grouped = {}
        for v in violations:
            if v.violation_type not in grouped:
                grouped[v.violation_type] = []
            grouped[v.violation_type].append(v)
        return grouped

    def _generate_executive_summary(
        self,
        compliance_report: Dict[str, Any],
        total_violations: int,
        critical_count: int
    ) -> str:
        """Generate human-readable executive summary."""
        score = compliance_report.get('summary', {}).get('compliance_score', 0)
        slides_analyzed = compliance_report.get('summary', {}).get('total_slides_analyzed', 0)

        severity_dist = compliance_report.get('violations_by_severity', {})

        summary = f"""
TEMPLATE COMPLIANCE ANALYSIS SUMMARY

Document: {self.input_document.get('metadata', {}).get('file_path', 'Unknown')}
Analyzed: {slides_analyzed} slides
Compliance Score: {score}/100

VIOLATION OVERVIEW:
- Total Violations: {total_violations}
- Critical Issues: {severity_dist.get('critical', 0)}
- Warnings: {severity_dist.get('warning', 0)}
- Informational: {severity_dist.get('info', 0)}

ASSESSMENT:
"""
        if score >= 90:
            summary += "✓ Excellent compliance with template requirements.\n"
        elif score >= 70:
            summary += "⚠ Good compliance with some deviations from template.\n"
        elif score >= 50:
            summary += "⚠ Moderate compliance - several template violations detected.\n"
        else:
            summary += "✗ Poor compliance - significant template violations require attention.\n"

        if critical_count > 0:
            summary += f"⚠ URGENT: {critical_count} critical issues require immediate attention.\n"

        return summary.strip()

    def _generate_llm_prompt(
        self,
        all_violations: List[ViolationContext],
        by_slide: Dict[int, List[ViolationContext]],
        by_type: Dict[str, List[ViolationContext]]
    ) -> str:
        """
        Generate comprehensive LLM prompt for recommendations.

        This prompt will be sent to an LLM to generate actionable recommendations.
        """
        prompt = """You are an expert PowerPoint compliance analyst. You have been provided with a detailed analysis of a presentation that was compared against a template.

TASK: Analyze the violations below and provide actionable recommendations to bring this presentation into compliance with the template.

"""

        # Add document context
        doc_meta = self.input_document.get('metadata', {})
        prompt += f"""
DOCUMENT INFORMATION:
- File: {doc_meta.get('file_path', 'Unknown')}
- Total Slides: {doc_meta.get('total_slides', 0)}
- Author: {doc_meta.get('author', 'Unknown')}
- Last Modified: {doc_meta.get('modified', 'Unknown')}

"""

        # Add violation summary
        prompt += f"""
VIOLATION SUMMARY:
- Total Violations: {len(all_violations)}
- Critical: {len([v for v in all_violations if v.severity == 'critical'])}
- Warnings: {len([v for v in all_violations if v.severity == 'warning'])}
- Info: {len([v for v in all_violations if v.severity == 'info'])}

VIOLATIONS BY TYPE:
"""
        for vtype, violations in sorted(by_type.items(), key=lambda x: len(x[1]), reverse=True):
            prompt += f"- {vtype}: {len(violations)} violations\n"

        prompt += "\n"

        # Add detailed violations by slide (limit to top 20)
        prompt += "DETAILED VIOLATIONS (by slide):\n\n"

        slide_limit = 20
        violation_count = 0

        for slide_num in sorted(by_slide.keys()):
            if violation_count >= slide_limit:
                remaining = len(all_violations) - violation_count
                prompt += f"\n... and {remaining} more violations in remaining slides ...\n"
                break

            violations = by_slide[slide_num]
            prompt += f"SLIDE {slide_num}:\n"

            # Add slide context
            slide_data = next((s for s in self.input_document.get('slides', [])
                             if s['slide_index'] == slide_num), None)
            if slide_data:
                if slide_data.get('title'):
                    prompt += f"  Title: {slide_data['title']}\n"
                if slide_data.get('layout_name'):
                    prompt += f"  Layout: {slide_data['layout_name']}\n"

            # Add violations
            for v in violations[:10]:  # Limit violations per slide
                prompt += f"\n  [{v.severity.upper()}] {v.violation_type}\n"
                prompt += f"    Issue: {v.description}\n"
                if v.expected and len(v.expected) < 100:
                    prompt += f"    Expected: {v.expected}\n"
                if v.actual and len(v.actual) < 100:
                    prompt += f"    Actual: {v.actual}\n"
                if v.context.get('element_text'):
                    prompt += f"    Context: \"{v.context['element_text']}\"\n"

                violation_count += 1

            prompt += "\n"

        # Add instructions for LLM
        prompt += """
INSTRUCTIONS:
Based on the violations above, please provide:

1. PRIORITY ISSUES: List the top 5-10 most critical issues that need immediate attention.

2. RECOMMENDATIONS BY CATEGORY:
   - Content Issues: Text that doesn't match template expectations
   - Formatting Issues: Font, size, color deviations
   - Structure Issues: Layout, element count, table structure problems
   - Other Issues: Any other compliance concerns

3. ACTION ITEMS: Specific, actionable steps to fix each category of issues.

4. IMPACT ASSESSMENT: How these violations affect the overall presentation quality and compliance.

5. QUICK WINS: Easy fixes that can improve compliance score quickly.

Please format your response in clear sections with bullet points for easy readability.
"""

        return prompt

    def generate_text_report(self, merged_report: MergedReport) -> str:
        """Generate a formatted text report for display."""
        lines = []
        lines.append("=" * 80)
        lines.append("TEMPLATE COMPLIANCE REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(merged_report.executive_summary)
        lines.append("")
        lines.append("=" * 80)
        lines.append("VIOLATIONS BY SLIDE")
        lines.append("=" * 80)

        for slide_num in sorted(merged_report.violations_by_slide.keys())[:10]:
            violations = merged_report.violations_by_slide[slide_num]
            lines.append(f"\nSlide {slide_num}: {len(violations)} violations")
            for v in violations[:5]:  # Top 5 per slide
                lines.append(f"  [{v.severity.upper()}] {v.description}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("READY FOR LLM ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Prompt Length: {len(merged_report.recommendations_prompt)} characters")
        lines.append("Status: Ready to send to LLM for recommendations")

        return "\n".join(lines)


def merge_compliance_report(
    compliance_report_path: str,
    input_document_path: str,
    template_document_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> MergedReport:
    """
    Merge compliance report with document data.

    Args:
        compliance_report_path: Path to compliance report JSON
        input_document_path: Path to input document JSON
        template_document_path: Optional path to template JSON
        output_path: Optional path to save merged report

    Returns:
        MergedReport ready for LLM
    """
    # Load files
    with open(compliance_report_path, 'r', encoding='utf-8') as f:
        compliance_report = json.load(f)

    with open(input_document_path, 'r', encoding='utf-8') as f:
        input_document = json.load(f)

    template_document = None
    if template_document_path:
        with open(template_document_path, 'r', encoding='utf-8') as f:
            template_document = json.load(f)

    # Merge
    merger = ComplianceReportMerger()
    merged = merger.merge_report(compliance_report, input_document, template_document)

    # Save if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Merged report saved to {output_path}")

        # Also save the LLM prompt separately
        prompt_path = Path(output_path).parent / (Path(output_path).stem + "_llm_prompt.txt")
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(merged.recommendations_prompt)
        logger.info(f"LLM prompt saved to {prompt_path}")

    return merged


if __name__ == "__main__":
    import sys

    # Example usage
    compliance_report_file = r"C:\Users\Shreya B\Documents\GitHub\slide-review-agent\data\logs\template_compliance_report.json"
    input_document_file = r"C:\Users\Shreya B\Documents\GitHub\slide-review-agent\data\logs\enhanced_try.json"
    template_file = r"C:\Users\Shreya B\Documents\GitHub\slide-review-agent\backend\templates\amida ipr template.json"
    output_file = r"C:\Users\Shreya B\Documents\GitHub\slide-review-agent\data\logs\merged_compliance_report.json"

    try:
        print("Merging compliance report with document context...")
        merged = merge_compliance_report(
            compliance_report_file,
            input_document_file,
            template_file,
            output_file
        )

        # Generate text report
        merger = ComplianceReportMerger()
        text_report = merger.generate_text_report(merged)
        print(text_report)

        print(f"\n{'='*80}")
        print("FILES GENERATED:")
        print(f"  - Merged Report: {output_file}")
        print(f"  - LLM Prompt: {Path(output_file).parent / (Path(output_file).stem + '_llm_prompt.txt')}")
        print(f"\nNext Step: Send the LLM prompt to your LLM for recommendations!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
