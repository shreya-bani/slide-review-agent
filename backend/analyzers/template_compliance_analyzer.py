"""
Template Compliance Analyzer

Analyzes extracted PPTX JSON against a template to identify formatting inconsistencies.
Detects where actual values exist when template expects null/empty values.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Import template finder
from ..utils.template_finder import TemplateFinder

logger = logging.getLogger(__name__)


@dataclass
class TemplateViolation:
    """Represents a single template compliance violation."""
    slide_index: int
    element_type: str  # 'content', 'table', 'metadata'
    element_index: Optional[int] = None
    field_path: str = ""  # e.g., "formatting.font_name"
    expected_value: Any = None
    actual_value: Any = None
    severity: str = "warning"  # 'critical', 'warning', 'info'
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComplianceReport:
    """Complete template compliance report."""
    total_violations: int
    violations_by_severity: Dict[str, int]
    violations_by_field: Dict[str, int]
    violations: List[TemplateViolation]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_violations': self.total_violations,
            'violations_by_severity': self.violations_by_severity,
            'violations_by_field': self.violations_by_field,
            'violations': [v.to_dict() for v in self.violations],
            'summary': self.summary
        }


class TemplateComplianceAnalyzer:
    """
    Analyzes PPTX JSON against a template for formatting compliance.

    Rule-based analyzer that checks:
    1. Formatting fields that should be null but have values
    2. Missing required fields
    3. Type mismatches
    4. Unexpected additional fields
    """

    def __init__(self, template_path: Optional[str] = None, template_dir: Optional[str] = None):
        """
        Initialize analyzer with template.

        Args:
            template_path: Path to template JSON file. If None, auto-finds template.
            template_dir: Directory containing templates for auto-finding.
        """
        self.template = None
        self.template_path = template_path
        self.template_finder = TemplateFinder(template_dir)

        if template_path:
            self.load_template(template_path)

    def load_template(self, template_path: str) -> bool:
        """Load template JSON from file."""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                self.template = json.load(f)
            logger.info(f"Loaded template from {template_path}")
            self.template_path = template_path
            return True
        except Exception as e:
            logger.error(f"Failed to load template: {e}")
            return False

    def auto_find_and_load_template(self, filename: str) -> bool:
        """
        Automatically find and load the best matching template for a filename.

        Args:
            filename: Name of the uploaded file (e.g., "my_ipr_report.pptx")

        Returns:
            True if template found and loaded, False otherwise
        """
        template_path = self.template_finder.find_template(filename)

        if template_path:
            logger.info(f"Auto-matched template for '{filename}': {template_path.name}")
            return self.load_template(str(template_path))
        else:
            logger.error(f"No template found for '{filename}'")
            return False

    def analyze(self, input_json: Dict[str, Any], template_json: Optional[Dict[str, Any]] = None) -> ComplianceReport:
        """
        Analyze input JSON against template.

        Args:
            input_json: The extracted PPTX JSON to analyze
            template_json: Optional template to use (overrides loaded template)

        Returns:
            ComplianceReport with all violations found
        """
        template = template_json if template_json else self.template

        if not template:
            raise ValueError("No template available. Load template first or provide template_json.")

        violations = []

        # Analyze metadata
        violations.extend(self._analyze_metadata(input_json.get('metadata', {}), template.get('metadata', {})))

        # Analyze slides - match by slide index
        input_slides = input_json.get('slides', [])
        template_slides = template.get('slides', [])

        logger.debug(f"Input has {len(input_slides)} slides, Template has {len(template_slides)} slides")

        # Create slide index map for template
        template_slide_map = {s['slide_index']: s for s in template_slides}

        logger.debug(f"Template slide indices: {list(template_slide_map.keys())}")

        for input_slide in input_slides:
            slide_idx = input_slide.get('slide_index')

            # Check if this slide exists in template
            if slide_idx in template_slide_map:
                template_slide = template_slide_map[slide_idx]
                logger.debug(f"Comparing slide {slide_idx}")
                violations.extend(self._analyze_slide(input_slide, template_slide, slide_idx))
            else:
                # Slide not in template
                logger.debug(f"Slide {slide_idx} not in template (template has slides: {list(template_slide_map.keys())})")
                violation = TemplateViolation(
                    slide_index=slide_idx,
                    element_type='slide',
                    field_path='slide_index',
                    expected_value='(slide exists in template)',
                    actual_value=slide_idx,
                    severity='info',
                    description=f"Slide {slide_idx} exists in input but not in template"
                )
                violations.append(violation)

        # Build report
        report = self._build_report(violations)
        return report

    def _analyze_metadata(self, input_meta: Dict[str, Any], template_meta: Dict[str, Any]) -> List[TemplateViolation]:
        """Analyze metadata section for violations."""
        violations = []

        # Check theme fields (should be null/empty in template)
        if 'theme' in input_meta and 'theme' in template_meta:
            theme_violations = self._compare_objects(
                input_meta['theme'],
                template_meta['theme'],
                slide_index=0,
                element_type='metadata',
                field_prefix='theme'
            )
            violations.extend(theme_violations)

        return violations

    def _analyze_slide(self, input_slide: Dict[str, Any], template_slide: Dict[str, Any], slide_idx: int) -> List[TemplateViolation]:
        """Analyze a single slide for content and structure violations."""
        violations = []

        # Check slide title
        input_title = input_slide.get('title')
        template_title = template_slide.get('title')

        if template_title and input_title != template_title:
            violation = TemplateViolation(
                slide_index=slide_idx,
                element_type='slide_title',
                field_path='title',
                expected_value=template_title,
                actual_value=input_title,
                severity='warning',
                description=f"Slide title mismatch: expected '{template_title}', got '{input_title}'"
            )
            violations.append(violation)

        # Check layout name
        input_layout = input_slide.get('layout_name')
        template_layout = template_slide.get('layout_name')

        if template_layout and input_layout != template_layout:
            violation = TemplateViolation(
                slide_index=slide_idx,
                element_type='slide_layout',
                field_path='layout_name',
                expected_value=template_layout,
                actual_value=input_layout,
                severity='warning',
                description=f"Layout mismatch: expected '{template_layout}', got '{input_layout}'"
            )
            violations.append(violation)

        # Analyze content elements - match by index
        input_elements = input_slide.get('content_elements', [])
        template_elements = template_slide.get('content_elements', [])

        # Check element count
        if len(input_elements) != len(template_elements):
            violation = TemplateViolation(
                slide_index=slide_idx,
                element_type='content_count',
                field_path='content_elements',
                expected_value=len(template_elements),
                actual_value=len(input_elements),
                severity='info',
                description=f"Element count mismatch: expected {len(template_elements)}, got {len(input_elements)}"
            )
            violations.append(violation)

        # Compare each element
        for idx in range(min(len(input_elements), len(template_elements))):
            input_elem = input_elements[idx]
            template_elem = template_elements[idx]
            element_violations = self._analyze_element(input_elem, template_elem, slide_idx, idx)
            violations.extend(element_violations)

        # Analyze tables
        input_tables = input_slide.get('tables', [])
        template_tables = template_slide.get('tables', [])

        # Check table count
        if len(input_tables) != len(template_tables):
            violation = TemplateViolation(
                slide_index=slide_idx,
                element_type='table_count',
                field_path='tables',
                expected_value=len(template_tables),
                actual_value=len(input_tables),
                severity='warning',
                description=f"Table count mismatch: expected {len(template_tables)}, got {len(input_tables)}"
            )
            violations.append(violation)

        # Compare each table
        for idx in range(min(len(input_tables), len(template_tables))):
            input_table = input_tables[idx]
            template_table = template_tables[idx]
            table_violations = self._analyze_table(input_table, template_table, slide_idx, idx)
            violations.extend(table_violations)

        return violations

    def _analyze_element(self, input_elem: Dict[str, Any], template_elem: Dict[str, Any],
                        slide_index: int, elem_index: int) -> List[TemplateViolation]:
        """Analyze a content element for text and formatting violations."""
        violations = []

        # Compare text content
        input_text = input_elem.get('text', '')
        template_text = template_elem.get('text', '')

        if template_text and input_text != template_text:
            violation = TemplateViolation(
                slide_index=slide_index,
                element_type='content_text',
                element_index=elem_index,
                field_path='text',
                expected_value=template_text,
                actual_value=input_text,
                severity='critical',
                description=f"Text content mismatch at element {elem_index}: expected '{template_text}', got '{input_text}'"
            )
            violations.append(violation)

        # Compare locator (element structure)
        input_locator = input_elem.get('locator', {})
        template_locator = template_elem.get('locator', {})

        # Check bullet level
        if 'bullet_level' in template_locator and template_locator['bullet_level'] is not None:
            if input_locator.get('bullet_level') != template_locator.get('bullet_level'):
                violation = TemplateViolation(
                    slide_index=slide_index,
                    element_type='bullet_level',
                    element_index=elem_index,
                    field_path='locator.bullet_level',
                    expected_value=template_locator['bullet_level'],
                    actual_value=input_locator.get('bullet_level'),
                    severity='warning',
                    description=f"Bullet level mismatch at element {elem_index}"
                )
                violations.append(violation)

        # Analyze formatting - focus on non-null template values
        input_formatting = input_elem.get('formatting', {})
        template_formatting = template_elem.get('formatting', {})

        formatting_violations = self._compare_formatting(
            input_formatting,
            template_formatting,
            slide_index=slide_index,
            elem_index=elem_index
        )
        violations.extend(formatting_violations)

        return violations

    def _compare_formatting(self, input_fmt: Dict[str, Any], template_fmt: Dict[str, Any],
                           slide_index: int, elem_index: int) -> List[TemplateViolation]:
        """
        Compare formatting, focusing ONLY on non-null template values.

        This means: if template has a specific value (not null), the input must match it.
        """
        violations = []

        for key, template_value in template_fmt.items():
            # Skip null template values - we only care about actual formatting requirements
            if template_value is None:
                continue

            # Skip false boolean values in template (consider them as "don't care")
            if template_value is False:
                continue

            input_value = input_fmt.get(key)

            # Template has a requirement - check if input matches
            if input_value != template_value:
                # Determine severity based on field importance
                severity = 'critical' if key in ['font_name', 'font_size', 'font_color'] else 'warning'

                violation = TemplateViolation(
                    slide_index=slide_index,
                    element_type='formatting',
                    element_index=elem_index,
                    field_path=f'formatting.{key}',
                    expected_value=template_value,
                    actual_value=input_value,
                    severity=severity,
                    description=f"Formatting mismatch for '{key}': expected '{template_value}', got '{input_value}'"
                )
                violations.append(violation)

        return violations

    def _analyze_table(self, input_table: Dict[str, Any], template_table: Dict[str, Any],
                      slide_index: int, table_index: int) -> List[TemplateViolation]:
        """Analyze a table for structure and content violations."""
        violations = []

        # Check table dimensions
        input_rows = input_table.get('rows', 0)
        template_rows = template_table.get('rows', 0)
        input_cols = input_table.get('columns', 0)
        template_cols = template_table.get('columns', 0)

        if input_rows != template_rows:
            violation = TemplateViolation(
                slide_index=slide_index,
                element_type='table_structure',
                element_index=table_index,
                field_path='rows',
                expected_value=template_rows,
                actual_value=input_rows,
                severity='warning',
                description=f"Table row count mismatch: expected {template_rows}, got {input_rows}"
            )
            violations.append(violation)

        if input_cols != template_cols:
            violation = TemplateViolation(
                slide_index=slide_index,
                element_type='table_structure',
                element_index=table_index,
                field_path='columns',
                expected_value=template_cols,
                actual_value=input_cols,
                severity='warning',
                description=f"Table column count mismatch: expected {template_cols}, got {input_cols}"
            )
            violations.append(violation)

        # Analyze table cells - match by row/col index
        input_cells = input_table.get('cells', [])
        template_cells = template_table.get('cells', [])

        # Create cell map for easy lookup
        input_cell_map = {(c['row_index'], c['col_index']): c for c in input_cells}
        template_cell_map = {(c['row_index'], c['col_index']): c for c in template_cells}

        for (row, col), template_cell in template_cell_map.items():
            input_cell = input_cell_map.get((row, col))

            if not input_cell:
                violation = TemplateViolation(
                    slide_index=slide_index,
                    element_type='table_cell',
                    element_index=table_index,
                    field_path=f'cell[{row},{col}]',
                    expected_value='(cell exists)',
                    actual_value='(missing)',
                    severity='warning',
                    description=f"Missing table cell at row {row}, col {col}"
                )
                violations.append(violation)
                continue

            # Compare cell text
            input_text = input_cell.get('text', '')
            template_text = template_cell.get('text', '')

            if template_text and input_text != template_text:
                violation = TemplateViolation(
                    slide_index=slide_index,
                    element_type='table_cell_text',
                    element_index=table_index,
                    field_path=f'cell[{row},{col}].text',
                    expected_value=template_text,
                    actual_value=input_text,
                    severity='critical',
                    description=f"Table cell text mismatch at [{row},{col}]: expected '{template_text}', got '{input_text}'"
                )
                violations.append(violation)

            # Compare cell formatting (only non-null template values)
            cell_formatting_violations = self._compare_formatting(
                input_cell.get('formatting', {}),
                template_cell.get('formatting', {}),
                slide_index=slide_index,
                elem_index=table_index
            )
            violations.extend(cell_formatting_violations)

        return violations

    def _compare_objects(self, input_obj: Dict[str, Any], template_obj: Dict[str, Any],
                        slide_index: int, element_type: str, element_index: Optional[int] = None,
                        field_prefix: str = "") -> List[TemplateViolation]:
        """
        Compare two objects field by field.

        Key rule: If template has null/empty value but input has actual value, it's a violation.
        """
        violations = []

        for key, template_value in template_obj.items():
            input_value = input_obj.get(key)
            field_path = f"{field_prefix}.{key}" if field_prefix else key

            # Rule 1: Template is null, but input has a value
            if template_value is None and input_value is not None:
                # Skip if input value is also a "null-like" value
                if self._is_null_like(input_value):
                    continue

                violation = TemplateViolation(
                    slide_index=slide_index,
                    element_type=element_type,
                    element_index=element_index,
                    field_path=field_path,
                    expected_value=None,
                    actual_value=input_value,
                    severity=self._determine_severity(key, input_value),
                    description=f"Field '{key}' should be null but has value: {input_value}"
                )
                violations.append(violation)

            # Rule 2: Template is empty dict/list, but input has values
            elif isinstance(template_value, (dict, list)) and not template_value:
                if isinstance(input_value, dict) and input_value:
                    violation = TemplateViolation(
                        slide_index=slide_index,
                        element_type=element_type,
                        element_index=element_index,
                        field_path=field_path,
                        expected_value={},
                        actual_value=input_value,
                        severity='info',
                        description=f"Field '{key}' should be empty dict but has values"
                    )
                    violations.append(violation)
                elif isinstance(input_value, list) and input_value:
                    violation = TemplateViolation(
                        slide_index=slide_index,
                        element_type=element_type,
                        element_index=element_index,
                        field_path=field_path,
                        expected_value=[],
                        actual_value=input_value,
                        severity='info',
                        description=f"Field '{key}' should be empty list but has {len(input_value)} items"
                    )
                    violations.append(violation)

            # Rule 3: Type mismatch
            elif template_value is not None and input_value is not None:
                if type(template_value) != type(input_value):
                    violation = TemplateViolation(
                        slide_index=slide_index,
                        element_type=element_type,
                        element_index=element_index,
                        field_path=field_path,
                        expected_value=f"type:{type(template_value).__name__}",
                        actual_value=f"type:{type(input_value).__name__}",
                        severity='warning',
                        description=f"Type mismatch for '{key}': expected {type(template_value).__name__}, got {type(input_value).__name__}"
                    )
                    violations.append(violation)

        # Rule 4: Check for fields in input that don't exist in template
        for key in input_obj.keys():
            if key not in template_obj:
                field_path = f"{field_prefix}.{key}" if field_prefix else key
                violation = TemplateViolation(
                    slide_index=slide_index,
                    element_type=element_type,
                    element_index=element_index,
                    field_path=field_path,
                    expected_value="(field should not exist)",
                    actual_value=input_obj[key],
                    severity='info',
                    description=f"Unexpected field '{key}' not in template"
                )
                violations.append(violation)

        return violations

    def _is_null_like(self, value: Any) -> bool:
        """Check if value is null-like (None, empty string, empty list/dict, False, 0)."""
        if value is None or value is False:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        if isinstance(value, (list, dict)) and not value:
            return True
        if isinstance(value, (int, float)) and value == 0:
            return True
        return False

    def _determine_severity(self, field_name: str, value: Any) -> str:
        """Determine violation severity based on field and value."""
        # Critical: Font and color fields that should match template
        critical_fields = ['font_name', 'font_size', 'font_color', 'font_color_rgb', 'font_color_theme']
        if field_name in critical_fields:
            return 'critical'

        # Warning: Style fields that affect appearance
        warning_fields = ['is_bold', 'is_italic', 'is_underline', 'paragraph_alignment',
                         'background_color', 'background_color_rgb']
        if field_name in warning_fields:
            return 'warning'

        # Info: Other formatting fields
        return 'info'

    def _build_report(self, violations: List[TemplateViolation]) -> ComplianceReport:
        """Build comprehensive compliance report."""
        # Count by severity
        violations_by_severity = {'critical': 0, 'warning': 0, 'info': 0}
        for v in violations:
            violations_by_severity[v.severity] = violations_by_severity.get(v.severity, 0) + 1

        # Count by field
        violations_by_field = {}
        for v in violations:
            field = v.field_path.split('.')[-1]  # Get last part of path
            violations_by_field[field] = violations_by_field.get(field, 0) + 1

        # Build summary
        summary = {
            'total_slides_analyzed': len(set(v.slide_index for v in violations)),
            'most_common_violations': sorted(violations_by_field.items(), key=lambda x: x[1], reverse=True)[:10],
            'compliance_score': self._calculate_compliance_score(violations),
        }

        return ComplianceReport(
            total_violations=len(violations),
            violations_by_severity=violations_by_severity,
            violations_by_field=violations_by_field,
            violations=violations,
            summary=summary
        )

    def _calculate_compliance_score(self, violations: List[TemplateViolation]) -> float:
        """Calculate compliance score (0-100)."""
        if not violations:
            return 100.0

        # Weight violations by severity
        weights = {'critical': 3, 'warning': 2, 'info': 1}
        weighted_violations = sum(weights.get(v.severity, 1) for v in violations)

        # Simple scoring: assume perfect score at 0 violations, decrease with weighted violations
        # Max expected violations: arbitrary baseline (e.g., 100)
        max_expected = 100
        score = max(0, 100 - (weighted_violations / max_expected * 100))

        return round(score, 2)


def analyze_file(input_path: str, template_path: Optional[str] = None,
                 output_path: Optional[str] = None, auto_find_template: bool = True) -> ComplianceReport:
    """
    Analyze a PPTX JSON file against a template.

    Args:
        input_path: Path to input JSON file
        template_path: Path to template JSON file (optional if auto_find_template=True)
        output_path: Optional path to save report JSON
        auto_find_template: If True and template_path is None, auto-find best matching template

    Returns:
        ComplianceReport
    """
    # Load input
    with open(input_path, 'r', encoding='utf-8') as f:
        input_json = json.load(f)

    # Get original filename from input metadata
    original_filename = input_json.get('metadata', {}).get('file_path', '')
    if original_filename:
        original_filename = Path(original_filename).name

    # Load or find template
    template_json = None
    analyzer = TemplateComplianceAnalyzer()

    if template_path:
        # Use provided template
        with open(template_path, 'r', encoding='utf-8') as f:
            template_json = json.load(f)
        logger.info(f"Using provided template: {template_path}")
        
    elif auto_find_template and original_filename:
        # Auto-find template based on filename
        logger.info(f"Auto-finding template for: {original_filename}")
        if analyzer.auto_find_and_load_template(original_filename):
            template_json = analyzer.template
            template_name = Path(analyzer.template_path).name if analyzer.template_path else "Unknown"
            logger.info(f"Auto-matched template: {template_name}")
        else:
            raise ValueError(f"Could not find matching template for: {original_filename}")
    else:
        raise ValueError("Either template_path must be provided or auto_find_template must be True with valid filename")

    # Analyze
    report = analyzer.analyze(input_json, template_json)

    # Save report if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {output_path}")

    return report


# if __name__ == "__main__":
#     import sys

#     # Configure logging for main execution
#     logging.basicConfig(
#         level=logging.DEBUG,  # Changed to DEBUG to see more details
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )

#     # Example usage
#     input_file = r"C:\Users\Shreya B\Documents\GitHub\slide-review-agent\data\logs\001_Amida_Agentic_AI_solution_Strategic_Plan_Draft_Aug_normalized.json"
#     output_file = r"C:\Users\Shreya B\Documents\GitHub\slide-review-agent\data\logs\ai_report.json"

#     # You can either:
#     # 1. Provide explicit template path
#     # template_file = r"C:\Users\Shreya B\Documents\GitHub\slide-review-agent\backend\templates\amida ipr template.json"
#     # report = analyze_file(input_file, template_path=template_file, output_path=output_file)

#     # 2. Or use auto-finding (recommended!)
#     try:
#         logger.info("=" * 80)
#         logger.info("TEMPLATE COMPLIANCE ANALYSIS")
#         logger.info("=" * 80)
#         logger.info(f"Input file: {Path(input_file).name}")
#         logger.info("Auto-matching template...")

#         report = analyze_file(input_file, output_path=output_file, auto_find_template=True)

#         logger.info("=" * 80)
#         logger.info("TEMPLATE COMPLIANCE REPORT")
#         logger.info("=" * 80)
#         logger.info(f"Total Violations: {report.total_violations}")
#         logger.info(f"Compliance Score: {report.summary['compliance_score']}/100")

#         logger.info("Violations by Severity:")
#         for severity, count in report.violations_by_severity.items():
#             logger.info(f"  {severity.upper()}: {count}")

#         logger.info("Violations by Type:")
#         violation_types = {}
#         for v in report.violations:
#             vtype = v.element_type
#             violation_types[vtype] = violation_types.get(vtype, 0) + 1
#         for vtype, count in sorted(violation_types.items(), key=lambda x: x[1], reverse=True)[:10]:
#             logger.info(f"  {vtype}: {count}")

#         logger.info("Sample Violations (first 10):")
#         for i, v in enumerate(report.violations[:10], 1):
#             logger.info(f"{i}. [{v.severity.upper()}] Slide {v.slide_index}, {v.element_type}")
#             logger.info(f"   {v.description}")
#             if len(str(v.expected_value)) < 50 and len(str(v.actual_value)) < 50:
#                 logger.info(f"   Expected: {v.expected_value}")
#                 logger.info(f"   Actual: {v.actual_value}")

#         logger.info(f"Detailed report saved to: {output_file}")

#     except Exception as e:
#         logger.error(f"Error: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
