"""
Quality scoring algorithm for document analysis.

Calculates comprehensive quality scores based on:
- Category-specific weights
- Severity-based penalties
- Document size normalization
- Per-category scoring breakdown
"""

from typing import Dict, Any


def calculate_quality_score(
    analysis_summary: Dict[str, Any],
    content_statistics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate comprehensive quality score based on categories and severity.

    Args:
        analysis_summary: Summary containing total_issues and category_breakdown
        content_statistics: Statistics containing total_pages, total_elements

    Returns:
        Dictionary with overall_score, grade, status, category_scores, deductions,
        issues_per_page, and methodology

    Note:
        Severity is NOT considered in the calculation. All issues within a category
        have equal weight. Only the number of issues per category matters.

    Example:
        >>> calculate_quality_score(
        ...     {
        ...         "total_issues": 25,
        ...         "category_breakdown": {"grammar": 10, "tone": 8},
        ...         "severity_breakdown": {"error": 5, "warning": 12}
        ...     },
        ...     {"total_pages": 10}
        ... )
        {
            "overall_score": 82.3,
            "grade": "B",
            "status": "Good",
            "category_scores": {...},
            ...
        }
    """

    # Category weights (total = 100)
    # These weights determine how much each category contributes to overall score
    CATEGORY_WEIGHTS = {
        "grammar": 25,          # Grammar, punctuation, spelling
        "tone": 40,             # Tone, voice, sentiment
        "inclusivity": 10,      # Inclusive language, person-first
        "formatting": 15,       # Visual formatting, style
        "word-list": 10,        # Word preferences, terminology
    }

    # Base deduction per issue (will be scaled by document size)
    # Severity is NOT considered - all issues have equal weight within their category
    BASE_DEDUCTION_PER_ISSUE = 0.5  # 0.5 points per issue

    # Get data from inputs
    category_breakdown = analysis_summary.get("category_breakdown", {})
    total_issues = analysis_summary.get("total_issues", 0)
    total_pages = max(content_statistics.get("total_pages", 1), 1)

    # Handle perfect document (no issues)
    if total_issues == 0:
        return {
            "overall_score": 100.0,
            "grade": "A",
            "status": "Excellent",
            "category_scores": {cat: 100.0 for cat in CATEGORY_WEIGHTS.keys()},
            "deductions": {cat: 0.0 for cat in CATEGORY_WEIGHTS.keys()},
            "issues_per_page": 0.0,
            "methodology": {
                "category_weights": CATEGORY_WEIGHTS,
                "base_deduction": BASE_DEDUCTION_PER_ISSUE
            }
        }

    # Calculate issues per page (for scaling)
    issues_per_page = total_issues / total_pages

    # Document complexity factor: more issues per page = slightly higher penalty per issue
    # This ensures that densely problematic documents score lower
    complexity_multiplier = 1.0 + (issues_per_page * 0.1)

    # Initialize category scores (all start at 100)
    category_scores = {}
    deductions = {}

    # Calculate score for each category
    for category, weight in CATEGORY_WEIGHTS.items():
        category_issues = category_breakdown.get(category, 0)

        if category_issues == 0:
            # Perfect score for this category
            category_scores[category] = 100.0
            deductions[category] = 0.0
            continue

        # Calculate deduction based on number of issues
        # More issues = more deduction, scaled by complexity
        deduction_per_issue = BASE_DEDUCTION_PER_ISSUE * complexity_multiplier
        total_deduction = category_issues * deduction_per_issue

        # Cap deduction at 100 (category score can't go below 0)
        total_deduction = min(total_deduction, 100.0)

        category_scores[category] = max(0.0, 100.0 - total_deduction)
        deductions[category] = -total_deduction

    # Calculate weighted overall score
    # Each category contributes proportionally to its weight
    overall_score = sum(
        category_scores.get(cat, 100.0) * (weight / 100.0)
        for cat, weight in CATEGORY_WEIGHTS.items()
    )

    # Determine letter grade and status
    if overall_score >= 90:
        grade = "A"
        status = "Excellent"
    elif overall_score >= 80:
        grade = "B"
        status = "Good"
    elif overall_score >= 70:
        grade = "C"
        status = "Satisfactory"
    elif overall_score >= 60:
        grade = "D"
        status = "Needs Improvement"
    else:
        grade = "F"
        status = "Poor"

    return {
        "overall_score": round(overall_score, 1),
        "grade": grade,
        "status": status,
        "category_scores": {k: round(v, 1) for k, v in category_scores.items()},
        "deductions": {k: round(v, 1) for k, v in deductions.items()},
        "issues_per_page": round(issues_per_page, 2),
        "methodology": {
            "category_weights": CATEGORY_WEIGHTS,
            "base_deduction": BASE_DEDUCTION_PER_ISSUE,
            "complexity_multiplier": round(complexity_multiplier, 2)
        }
    }


if __name__ == "__main__":
    # Example usage
    example_summary = {
        "total_issues": 25,
        "category_breakdown": {
            "grammar": 10,
            "tone": 8,
            "inclusivity": 2,
            "formatting": 5,
            "word-list": 0
        },
        "severity_breakdown": {
            "error": 5,
            "warning": 12,
            "suggestion": 8
        }
    }

    example_stats = {
        "total_pages": 10
    }

    result = calculate_quality_score(example_summary, example_stats)

    print("Quality Score Calculation Example")
    print("=" * 50)
    print(f"Overall Score: {result['overall_score']} ({result['grade']})")
    print(f"Status: {result['status']}")
    print(f"\nCategory Scores:")
    for cat, score in result['category_scores'].items():
        deduction = result['deductions'][cat]
        print(f"  {cat:15s}: {score:5.1f} (deduction: {deduction:6.1f})")
    print(f"\nIssues per page: {result['issues_per_page']}")
    print(f"\nNote: Severity is NOT considered - all issues within a category have equal weight")
