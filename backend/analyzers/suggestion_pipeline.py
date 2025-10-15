"""
Unified Multi-Rule Consolidation Pipeline - FIXED VERSION
Applies patches sequentially: protection → grammar → word-list → tone
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from .unified_issue import (
    UnifiedIssue,
    DetailedChange,
    LocationInfo,
    calculate_overall_confidence,
    determine_severity
)

logger = logging.getLogger(__name__)


def consolidate_and_optimize(
    issues: List[Dict[str, Any]],
    protection_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Main entry point: consolidate all issues by location into unified output.
    """
    by_location = _group_by_location(issues)
    
    unified_issues = []
    for location_key, location_issues in by_location.items():
        unified = _merge_issues_at_location(location_issues, protection_data)
        unified_issues.append(unified)
    
    logger.info(
        f"Consolidated {len(issues)} raw issues into {len(unified_issues)} unified issues"
    )
    
    return [issue.to_dict() for issue in unified_issues]


def _group_by_location(issues: List[Dict[str, Any]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Group issues by (page_index, element_index)."""
    by_location = {}
    for issue in issues:
        key = (
            issue.get("page_or_slide_index", -1),
            issue.get("element_index", -1)
        )
        if key not in by_location:
            by_location[key] = []
        by_location[key].append(issue)
    return by_location


def _is_swot_keyword(text: str) -> bool:
    """Check if text starts with SWOT keywords that should be protected."""
    text_lower = text.strip().lower()
    swot_keywords = ['strengths:', 'weaknesses:', 'opportunities:', 'threats:', 'threats/risks:']
    return any(text_lower.startswith(kw) for kw in swot_keywords)


def _merge_issues_at_location(
    issues: List[Dict[str, Any]],
    protection_data: Dict[str, Any]
) -> UnifiedIssue:
    """
    Merge all issues at same location into one UnifiedIssue.
    Pipeline: grammar → word-list → tone (applied to corrected text)
    """
    if len(issues) == 1:
        return _single_issue_to_unified(issues[0])
    
    # Sort by priority
    sorted_issues = _sort_by_priority(issues)
    
    # Extract location info
    first = sorted_issues[0]
    location = LocationInfo(
        page_or_slide_index=first.get("page_or_slide_index", 0),
        element_index=first.get("element_index", 0),
        element_type=first.get("element_type", "content"),
        display=first.get("location", f"slide {first.get('page_or_slide_index', 0)}, element {first.get('element_index', 0)}")
    )
    
    issue_id = f"slide_{location.page_or_slide_index}_element_{location.element_index}"
    
    # Start with original text
    original_text = first.get("found_text", "")
    current_text = original_text
    detailed_changes = []
    confidences = []
    
    # Separate by category
    grammar_issues = [i for i in sorted_issues if i.get("category") == "grammar"]
    wordlist_issues = [i for i in sorted_issues if i.get("category") == "word-list"]
    tone_issues = [i for i in sorted_issues if i.get("category") == "tone-issue"]
    
    # STEP 1: Apply grammar fixes (modifies current_text)
    for issue in grammar_issues:
        current_text, changes = _apply_grammar_issue(current_text, issue, original_text)
        detailed_changes.extend(changes)
        confidences.append(issue.get("confidence", 1.0))
    
    # STEP 2: Apply word-list fixes (modifies already-corrected text)
    for issue in wordlist_issues:
        current_text, changes = _apply_wordlist_issue_to_corrected(current_text, issue)
        detailed_changes.extend(changes)
        confidences.append(issue.get("confidence", 1.0))
    
    # STEP 3: Apply tone fixes ONLY if not SWOT protected
    is_swot_protected = _is_swot_keyword(original_text)
    
    if not is_swot_protected:
        for issue in tone_issues:
            if issue.get("suggestion"):
                # Analyze what changed from grammar+wordlist corrected text
                changes = _extract_tone_changes(current_text, issue.get("suggestion"), issue)
                detailed_changes.extend(changes)
                current_text = issue.get("suggestion")
                confidences.append(issue.get("confidence", 0.85))
    else:
        logger.debug(f"Skipping tone rewrite for SWOT-protected text: {original_text[:50]}...")
    
    # Generate summary
    changes_summary = _generate_changes_summary(detailed_changes)
    context = _infer_context(original_text)
    
    # Build unified issue
    return UnifiedIssue(
        issue_id=issue_id,
        rule_names=[i.get("rule_name") for i in sorted_issues],
        severity=determine_severity([i.get("severity") for i in sorted_issues]),
        categories=list(set(i.get("category") for i in sorted_issues)),
        descriptions=[i.get("description") for i in sorted_issues],
        location=location,
        found_text=original_text,
        suggestion=current_text,
        changes_summary=changes_summary,
        detailed_changes=detailed_changes,
        metadata={
            "method": "multi-rule-pipeline",
            "rules_applied_count": len(detailed_changes),
            "total_rules_detected": len(sorted_issues),
            "overall_confidence": calculate_overall_confidence(confidences),
            "context_preserved": context,
            "swot_protected": is_swot_protected,
            "note": _generate_context_note(context, is_swot_protected),
            "pipeline_version": "2.1.0"
        }
    )


def _sort_by_priority(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort issues by application priority."""
    priority_order = {"grammar": 1, "word-list": 2, "tone-issue": 3}
    return sorted(issues, key=lambda i: priority_order.get(i.get("category", "tone-issue"), 99))


def _apply_grammar_issue(
    text: str,
    issue: Dict[str, Any],
    original: str
) -> Tuple[str, List[DetailedChange]]:
    """Apply grammar patches and extract changes."""
    patches = issue.get("fix_data", {}).get("patches", [])
    if not patches:
        return text, []
    
    # Apply patches in reverse order to maintain positions
    modified = _apply_patches_safe(text, patches)
    
    # Extract what changed
    changes = []
    for start, end, replacement in patches:
        if start < len(original) and end <= len(original):
            old_text = original[start:end]
            changes.append(DetailedChange(
                rule=issue.get("rule_name", "unknown"),
                category="grammar",
                change=f"{old_text} → {replacement}",
                severity=issue.get("severity", "suggestion"),
                position=(start, end)
            ))
    
    return modified, changes


def _apply_wordlist_issue_to_corrected(
    corrected_text: str,
    issue: Dict[str, Any]
) -> Tuple[str, List[DetailedChange]]:
    """
    Apply word-list fixes to ALREADY CORRECTED text.
    Need to re-find patterns since positions have changed.
    """
    changes = []
    current_text = corrected_text
    
    # Re-build word list (same as WordListChecker)
    word_list = {
        "e.g. ": "e.g., ",
        "i.e. ": "i.e., ",
        "built in": "built-in",
        "cyber attacks": "cyberattacks",
        "cyber-attacks": "cyberattacks",
        "cyber security": "cybersecurity",
        "cyber-security": "cybersecurity",
        "codesets": "code sets",
        "code-sets": "code sets",
        "health care": "healthcare",
        "public-sector": "public sector",
        "user friendly": "user-friendly",
        "web based": "web-based",
        "machine learning": "Machine Learning",
        "artificial intelligence": "Artificial Intelligence",
        "natural language processing": "Natural Language Processing",
    }
    
    # Apply each replacement to current text
    for wrong, correct in word_list.items():
        if "." in wrong:
            pattern = re.escape(wrong)
        else:
            pattern = r"\b" + re.escape(wrong) + r"\b"
        
        matches = list(re.finditer(pattern, current_text, re.IGNORECASE))
        
        # Apply in reverse order to maintain positions
        for match in reversed(matches):
            matched_text = match.group(0)
            if matched_text != correct:
                start, end = match.span()
                current_text = current_text[:start] + correct + current_text[end:]
                changes.append(DetailedChange(
                    rule="word_list",
                    category="word-list",
                    change=f"{matched_text} → {correct}",
                    severity="suggestion"
                ))
    
    return current_text, changes


def _apply_patches_safe(text: str, patches: List[tuple]) -> str:
    """
    Apply patches in reverse order to maintain positions.
    Validates positions before applying.
    """
    if not patches:
        return text
    
    # Sort by start position (descending) to apply from end to start
    sorted_patches = sorted(patches, key=lambda p: p[0], reverse=True)
    
    result = text
    for start, end, replacement in sorted_patches:
        # Validate positions
        if 0 <= start < end <= len(result):
            result = result[:start] + replacement + result[end:]
        else:
            logger.warning(f"Invalid patch position: {start}-{end} in text of length {len(result)}")
    
    return result


def _extract_tone_changes(
    before: str,
    after: str,
    issue: Dict[str, Any]
) -> List[DetailedChange]:
    """Extract tone-level changes by comparing before/after."""
    changes = []
    rule = issue.get("rule_name", "tone")
    
    if rule == "positive_language":
        negations_before = re.findall(
            r'\b(can\'t|cannot|don\'t|do not|isn\'t|is not|aren\'t|are not|no\s+\w+|not\s+\w+|inability|inadequate)\b',
            before,
            re.IGNORECASE
        )
        negations_after = re.findall(
            r'\b(can\'t|cannot|don\'t|do not|isn\'t|is not|aren\'t|are not|no\s+\w+|not\s+\w+|inability|inadequate)\b',
            after,
            re.IGNORECASE
        )
        
        if len(negations_before) > len(negations_after):
            for neg in negations_before:
                if neg.lower() not in [n.lower() for n in negations_after]:
                    changes.append(DetailedChange(
                        rule="positive_language",
                        category="tone-issue",
                        change=f"{neg} → clearer phrasing",
                        severity="warning"
                    ))
    
    elif rule == "active_voice":
        passive_patterns = [
            r'\b(is|are|was|were|been|being)\s+\w+ed\b',
            r'\b(is|are|was|were|been|being)\s+\w+en\b',
        ]
        for pattern in passive_patterns:
            if re.search(pattern, before, re.IGNORECASE) and not re.search(pattern, after, re.IGNORECASE):
                changes.append(DetailedChange(
                    rule="active_voice",
                    category="tone-issue",
                    change="Converted passive voice to active",
                    severity="info"
                ))
                break
    
    # Generic fallback
    if not changes:
        changes.append(DetailedChange(
            rule=rule,
            category="tone-issue",
            change=f"{rule.replace('_', ' ').title()} improvement applied",
            severity=issue.get("severity", "warning")
        ))
    
    return changes


def _generate_changes_summary(detailed_changes: List[DetailedChange]) -> str:
    """Generate human-readable summary."""
    count = len(detailed_changes)
    if count == 0:
        return "No changes applied"
    elif count == 1:
        return f"Applied 1 improvement: {detailed_changes[0].change}"
    else:
        summary_parts = [c.change for c in detailed_changes[:3]]
        summary = ", ".join(summary_parts)
        if count > 3:
            summary += f" (+ {count - 3} more)"
        return f"Applied {count} improvements: {summary}"


def _infer_context(text: str) -> str:
    """Infer SWOT context from text."""
    lower = text.lower()
    if lower.startswith(("weakness", "threat", "risk")):
        return "weakness"
    elif lower.startswith(("strength", "opportunit")):
        return "strength"
    else:
        return "neutral"


def _generate_context_note(context: str, swot_protected: bool) -> str:
    """Generate metadata note based on context."""
    if swot_protected:
        return "SWOT keyword protected - tone rewrite skipped to preserve exact meaning"
    elif context == "weakness":
        return "Maintained negative context appropriate for SWOT weaknesses section"
    elif context == "strength":
        return "Preserved positive framing for SWOT strengths section"
    else:
        return "Standard style improvements applied"


def _single_issue_to_unified(issue: Dict[str, Any]) -> UnifiedIssue:
    """Convert single issue to UnifiedIssue format."""
    location = LocationInfo(
        page_or_slide_index=issue.get("page_or_slide_index", 0),
        element_index=issue.get("element_index", 0),
        element_type=issue.get("element_type", "content"),
        display=issue.get("location", "")
    )
    
    detailed_changes = []
    if issue.get("fix_data", {}).get("patches"):
        for start, end, replacement in issue["fix_data"]["patches"]:
            old = issue.get("found_text", "")[start:end] if start < len(issue.get("found_text", "")) else "?"
            detailed_changes.append(DetailedChange(
                rule=issue.get("rule_name", "unknown"),
                category=issue.get("category", "grammar"),
                change=f"{old} → {replacement}",
                severity=issue.get("severity", "suggestion")
            ))
    else:
        detailed_changes.append(DetailedChange(
            rule=issue.get("rule_name", "unknown"),
            category=issue.get("category", "tone-issue"),
            change=f"{issue.get('rule_name', 'Rule').replace('_', ' ').title()} applied",
            severity=issue.get("severity", "warning")
        ))
    
    is_swot_protected = _is_swot_keyword(issue.get("found_text", ""))
    
    return UnifiedIssue(
        issue_id=f"slide_{location.page_or_slide_index}_element_{location.element_index}",
        rule_names=[issue.get("rule_name")],
        severity=issue.get("severity", "suggestion"),
        categories=[issue.get("category", "grammar")],
        descriptions=[issue.get("description", "")],
        location=location,
        found_text=issue.get("found_text", ""),
        suggestion=issue.get("suggestion", ""),
        changes_summary=_generate_changes_summary(detailed_changes),
        detailed_changes=detailed_changes,
        metadata={
            "method": issue.get("method", "rule-based"),
            "rules_applied_count": len(detailed_changes),
            "total_rules_detected": 1,
            "overall_confidence": issue.get("confidence", 1.0),
            "context_preserved": _infer_context(issue.get("found_text", "")),
            "swot_protected": is_swot_protected,
            "note": _generate_context_note(_infer_context(issue.get("found_text", "")), is_swot_protected),
            "pipeline_version": "2.1.0"
        }
    )