"""
Test script to validate issue field mapping for database insertion.
"""
import json
from pathlib import Path

# Sample issue from actual analyzer output
sample_issue = {
    "rule_name": "positive_language",
    "severity": "warning",
    "category": "tone-issue",
    "description": "Use positive language to make communication clearer, more constructive, and solution-oriented.",
    "location": "slide 2, element 1",
    "found_text": "Amida's MCP team currently performs this mapping manually, which is time-consuming and not scalable.",
    "suggestion": "Amida's MCP team currently performs this mapping manually, which is labor-intensive and lacks scalability.",
    "page_or_slide_index": 1,
    "element_index": 0,
    "element_type": "bullet",
    "score": None,
    "method": "llm"
}

# Test field mappings
print("Testing field mappings for database insertion:")
print("-" * 60)

field_mappings = {
    "rule_name": ("rule_name", sample_issue.get("rule_name")),
    "severity": ("severity", sample_issue.get("severity")),
    "category": ("category", sample_issue.get("category")),
    "slide_number": ("page_or_slide_index", sample_issue.get("page_or_slide_index")),
    "element_type": ("element_type", sample_issue.get("element_type")),
    "element_index": ("element_index", sample_issue.get("element_index")),
    "original_text": ("found_text", sample_issue.get("found_text", "")),
    "suggested_text": ("suggestion", sample_issue.get("suggestion")),
    "explanation": ("description", sample_issue.get("description"))
}

all_passed = True
for db_field, (json_field, value) in field_mappings.items():
    status = "OK" if value is not None else "X"
    if value is None and db_field not in ["suggested_text", "element_type"]:  # Optional fields
        all_passed = False
        status = "X FAIL"
    print(f"{status} {db_field:20s} <- {json_field:25s} = {str(value)[:50]}")

print("-" * 60)
print(f"Result: {'PASS' if all_passed else 'FAIL'}")

# Test with actual result file
print("\n" + "=" * 60)
print("Testing with actual result file:")
print("=" * 60)

result_file = Path("data/outputs/042_amida_vacm_project_presentation_-may_2_2025_-final_result.json")
if result_file.exists():
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    issues = data.get("findings", [])
    print(f"Found {len(issues)} issues in result file")

    # Check categories
    categories = set(issue.get("category") for issue in issues)
    print(f"Categories: {sorted(categories)}")

    # Check if all required fields are present
    required_fields = ["rule_name", "severity", "category", "found_text", "description"]
    optional_fields = ["suggestion", "page_or_slide_index", "element_index", "element_type"]

    print("\nChecking field presence:")
    for i, issue in enumerate(issues[:3]):  # Check first 3
        print(f"\nIssue {i+1}:")
        for field in required_fields + optional_fields:
            has_field = field in issue
            value = issue.get(field)
            status = "OK" if has_field else "X"
            print(f"  {status} {field:25s}: {str(value)[:40] if value else 'None'}")
else:
    print("Result file not found!")
