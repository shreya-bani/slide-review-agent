"""Quick test of inclusivity analyzer"""
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.analyzers.inclusivity_analyzer import InclusivityAnalyzer, analyze_document


analyzer = InclusivityAnalyzer()

print("Testing document loading and processing")

# Test on actual document
doc_path = Path(__file__).parent.parent / "data" / "outputs" / "015_amida_vacm_project_presentation_-may_2_2025_-final_normalized.json"

if doc_path.exists():
    print(f"\nLoading document: {doc_path.name}")

    with doc_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    # Check document structure
    pages = doc.get("pages", [])
    print(f"Total pages in document: {len(pages)}")

    # Count and categorize elements
    total_elements = 0
    elements_with_text = 0
    elements_long_enough = 0
    sample_texts = []

    for page_idx, page in enumerate(pages):
        elements = page.get("elements", [])
        for elem_idx, elem in enumerate(elements):
            total_elements += 1
            text = elem.get("text", "")

            if text and isinstance(text, str):
                elements_with_text += 1
                text_len = len(text.strip())

                if text_len >= 10:
                    elements_long_enough += 1

                    # Collect samples with potential issues
                    text_lower = text.lower()
                    keywords = ['he ', 'his ', 'him ', 'she ', 'her ', 'guys', 'waitress',
                               'chairman', 'disabled', 'he or she']

                    if any(kw in text_lower for kw in keywords):
                        sample_texts.append({
                            'page': page_idx,
                            'elem': elem_idx,
                            'text': text,
                            'length': text_len
                        })

    print(f"Total elements: {total_elements}")
    print(f"Elements with text: {elements_with_text}")
    print(f"Elements >= 10 chars: {elements_long_enough}")
    print(f"Elements with potential inclusivity keywords: {len(sample_texts)}")

    if sample_texts:
        print("\nSample texts with keywords (first 5):")
        for item in sample_texts[:5]:
            print(f"\n  Page {item['page']}, Element {item['elem']} (len={item['length']}):")
            # Clean text for printing (replace problematic unicode chars)
            clean_text = item['text'][:150].encode('ascii', 'ignore').decode('ascii')
            print(f"  Text: {clean_text}")

    # Now run the analyzer
    print("Running analyze_document()...")

    issues = analyze_document(doc)
    print(f"\nTotal issues found: {len(issues)}")

    if issues:
        print("\nIssues detected:")
        for issue in issues[:10]:  # Show first 10
            print(f"\n  Rule: {issue.get('rule_name')}")
            print(f"  Location: {issue.get('location')}")
            print(f"  Found: {issue.get('found_text', '')[:80]}...")
            print(f"  Suggestion: {issue.get('suggestion', '')[:80]}...")
    else:
        print("\nNo issues found. Debugging why...")

        # Let's manually check a few texts
        if sample_texts:
            print("\nManually checking texts with keywords:")
            for item in sample_texts[:3]:
                clean_text = item['text'][:80].encode('ascii', 'ignore').decode('ascii')
                print(f"\n  Checking: {clean_text}")
                manual_issues = analyzer.check(item['text'], {}, item['page'], item['elem'])
                print(f"  Issues found: {len(manual_issues)}")
                if manual_issues:
                    for mi in manual_issues:
                        print(f"    - {mi.rule_name}: {mi.suggestion[:60]}...")
else:
    print(f"\nDocument not found: {doc_path}")
