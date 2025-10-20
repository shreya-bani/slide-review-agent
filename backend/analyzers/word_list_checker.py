from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from .grammar_checker import StyleIssue, Severity, Category

class WordListChecker:
    """
    This is to focus only on word-list normalization.
    """
    def __init__(self, protection_data: Dict[str, Any]):
        self.protection_data = protection_data
        self.word_list = self._load_amida_word_list()
        self._abbrev_patterns = [
            (re.compile(r'(?i)\b(e\.g\.)(?!\s*,)'), "e.g.,"),
            (re.compile(r'(?i)\b(i\.e\.)(?!\s*,)'), "i.e.,"),
        ]

    def _load_amida_word_list(self) -> Dict[str, str]:
        return {
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

    def _is_protected(self, text: str, start: int, end: int) -> bool:
        substring = text[start:end]
        for category_items in self.protection_data.values():
            for item in category_items:
                if item in substring:
                    return True
        return False

    def _apply_patches(self, text: str, patches: List[tuple]) -> str:
        if not patches:
            return text
        patches.sort(key=lambda p: p[0], reverse=True)
        result = text
        for start, end, replacement in patches:
            result = result[:start] + replacement + result[end:]
        return result

    def check(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> List[StyleIssue]:
        issues: List[StyleIssue] = []
        patches: List[tuple] = []

        # e.g. / i.e. → ensure trailing comma
        for rx, replacement in self._abbrev_patterns:
            for m in rx.finditer(text):
                start, end = m.span(1)
                if not self._is_protected(text, start, end):
                    # Replace only the token, then insert comma immediately after it
                    patches.append((start, end, replacement[:-1]))
                    patches.append((end, end, ","))

        # Word-list preferred forms
        for wrong, correct in self.word_list.items():
            pat = r"\b" + re.escape(wrong) + r"\b"
            for m in re.finditer(pat, text, re.IGNORECASE):
                if not self._is_protected(text, m.start(), m.end()):
                    matched = m.group(0)
                    if matched != correct:
                        patches.append((m.start(), m.end(), correct))

        if patches:
            issues.append(StyleIssue(
                rule_name="word_list",
                severity=Severity.SUGGESTION,
                category=Category.WORD_LIST,
                description="Apply Amida Word List preferences (with safe e.g./i.e. comma handling)",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=text,
                suggestion=self._apply_patches(text, patches),
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=1.0,
            ))
        return issues

