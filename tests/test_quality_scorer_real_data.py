"""
Test quality scorer with real data from the application.

This test uses actual category breakdown data to verify the quality scorer
calculates scores correctly based on only the categories that are used.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from backend.utils.quality_scorer import calculate_quality_score


# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

# Add handler if not already added
if not logger.handlers:
    logger.addHandler(console_handler)


def test_real_document_analysis():
    """
    Test with real document data:
    - Total issues: 43
    - Severity: 0 errors, 18 warnings, 12 info (note: some issues might have other severities)
    - Categories: grammar (16), word-list (2), tone (24), filename (1)
    - Pages: 33
    """

    analysis_summary = {
        "total_issues": 43,
        "category_breakdown": {
            "grammar": 16,
            "word-list": 2,
            "tone": 24,
            "filename": 1
        },
        "severity_breakdown": {
            "error": 0,
            "warning": 18,
            "info": 12
        }
    }

    content_statistics = {"total_pages": 33}

    result = calculate_quality_score(analysis_summary, content_statistics)

    logger.info("=" * 70)
    logger.info("QUALITY SCORE ANALYSIS - REAL DOCUMENT DATA")
    logger.info("=" * 70)
    logger.info("Document Statistics:")
    logger.info(f"  Total Issues: {analysis_summary['total_issues']}")
    logger.info(f"  Total Pages: {content_statistics['total_pages']}")
    logger.info(f"  Issues per Page: {result['issues_per_page']:.2f}")

    logger.info("-" * 70)
    logger.info(f"OVERALL SCORE: {result['overall_score']} / 100")
    logger.info(f"GRADE: {result['grade']}")
    logger.info(f"STATUS: {result['status']}")
    logger.info("-" * 70)

    logger.info("Category Weights (normalized):")
    total_weight_check = 0
    for cat, weight in result['methodology']['category_weights'].items():
        total_weight_check += weight
        logger.info(f"  {cat:15s}: {weight:5.1f}%")
    logger.info(f"  {'TOTAL':15s}: {total_weight_check:5.1f}%")

    logger.info("Category Analysis:")
    logger.info(f"  {'Category':<15} {'Issues':<8} {'Score':<8} {'Deduction':<12} {'Contribution':<15}")
    logger.info(f"  {'-' * 70}")

    for cat in ["grammar", "word-list", "tone", "filename"]:
        if cat in result['category_scores']:
            issues = analysis_summary['category_breakdown'].get(cat, 0)
            score = result['category_scores'][cat]
            deduction = result['deductions'][cat]
            weight = result['methodology']['category_weights'][cat]
            contribution = (score * weight / 100)
            logger.info(f"  {cat:<15} {issues:<8} {score:<8.1f} {deduction:<12.1f} {contribution:<15.2f}")

    logger.info("Methodology:")
    logger.info(f"  Base Deduction per Issue: {result['methodology']['base_deduction']}")
    logger.info(f"  Complexity Multiplier: {result['methodology']['complexity_multiplier']}")
    logger.info(
        f"  Effective Deduction: "
        f"{result['methodology']['base_deduction'] * result['methodology']['complexity_multiplier']:.2f} points per issue"
    )

    logger.info("Severity Breakdown (for reference only - not used in scoring):")
    for severity, count in analysis_summary['severity_breakdown'].items():
        percentage = (count / analysis_summary['total_issues'] * 100) if analysis_summary['total_issues'] > 0 else 0
        logger.info(f"  {severity:12s}: {count:3d} ({percentage:5.1f}%)")

    logger.info("=" * 70)
    logger.info("NOTE: Only categories present in the document are scored.")
    logger.info("NOTE: Weights are dynamically normalized to sum to 100%.")
    logger.info("NOTE: Severity levels do NOT affect scoring - all issues within")
    logger.info("      a category have equal weight.")
    logger.info("=" * 70)

    # Assertions to verify correctness
    assert result['overall_score'] > 0, "Score should be positive"
    assert result['overall_score'] <= 100, "Score should not exceed 100"
    assert result['grade'] in ['A', 'B', 'C', 'D', 'F'], "Grade should be valid"
    assert len(result['category_scores']) == 4, "Should have 4 categories scored"
    assert abs(sum(result['methodology']['category_weights'].values()) - 100.0) < 0.01, "Weights should sum to 100%"

    logger.info("✓ All assertions passed!")

    return result


if __name__ == "__main__":
    test_real_document_analysis()
