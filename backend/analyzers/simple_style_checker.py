"""
Restructured Grammar & Style Checker
Uses ProtectionLayer from protection_layer.py for protected content detection.
"""

from __future__ import annotations
import json
import re
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Union

from ..config.settings import settings
from ..utils.llm_client import LLMClient
from .protection_layer import ProtectionLayer, LLMConfigError

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(settings, "log_level", logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

log_dir = settings.log_dir or (Path(settings.output_dir) / "logs")

# Data structures
class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    INFO = "info"


class Category(Enum):
    GRAMMAR = "grammar"
    WORD_LIST = "word-list"


@dataclass
class StyleIssue:
    rule_name: str
    severity: Severity
    category: Category
    description: str
    location: str
    found_text: str
    suggestion: str
    page_or_slide_index: int
    element_index: int
    confidence: float = 1.0
    method: str = "rule-based"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["category"] = self.category.value
        return d


# Grammar Checker
class GrammarChecker:
    def __init__(self, protection_data: Dict[str, Any]):
        self.protection_data = protection_data
        self.rules = self._initialize_rules()

    def _initialize_rules(self):
        return {
            "contractions": self._check_contractions,
            "and_or": self._check_and_or,
            "ampersand": self._check_ampersand,
            "numerals": self._check_numerals,
            "period_spacing": self._check_period_spacing,
            "quotation_marks": self._check_quotation_marks,
            "hyphens": self._check_hyphens,
        }

    _MONTHS = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    }

    def _is_probable_year(self, text: str, start: int, end: int) -> bool:
        token = text[start:end]
        try:
            n = int(token)
        except ValueError:
            return False
        if not (1900 <= n <= 4000):
            return False
        left = text[max(0, start-40):start].lower()
        right = text[end:min(len(text), end+40)].lower()
        if any(m in left or m in right for m in self._MONTHS):
            return True
        if re.search(r'\b(?:fy|cy|period|spring|summer|fall|winter|q[1-4])\b', left+right):
            return True
        if re.search(r'^\s*[–-]\s*\d{4}\b', right) or re.search(r'\b\d{4}\s*[–-]\s*$', left):
            return True
        return True

    def _in_code_token(self, text: str, start: int, end: int) -> bool:
        left = text[max(0, start-6):start]
        right = text[end:min(len(text), end+6)]
        return bool(
            re.search(r'[A-Za-z]-?[A-Za-z]{1,5}$', left) or
            re.search(r'^[A-Za-z-]{1,6}', right)
        )

    def _is_page_ref(self, text: str, start: int) -> bool:
        left = text[max(0, start-3):start]
        return bool(
            re.search(r'(?:^|\s)p\.?$', left, re.IGNORECASE) or
            re.search(r'(?:^|\s)pp\.?$', left, re.IGNORECASE)
        )

    def _in_numeric_range(self, text: str, start: int, end: int) -> bool:
        before = text[max(0, start-1):start]
        after = text[end:end+1]
        return (before in {'-', '–'} or after in {'-', '–'})

    def check(self, text: str, elem: dict, slide_idx: int, element_index: int) -> List[StyleIssue]:
        issues: List[StyleIssue] = []
        for rule_name, rule_func in self.rules.items():
            try:
                res = rule_func(text, elem, slide_idx, element_index)
                if res:
                    issues.extend(res if isinstance(res, list) else [res])
            except Exception as e:
                logger.exception("Rule %s failed: %s", rule_name, e)
        return issues

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

    #----- Rules-------
    def _check_contractions(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> Optional[StyleIssue]:
        patches: List[tuple] = []
        contractions = {
            "can't": "cannot", "won't": "will not", "don't": "do not",
            "doesn't": "does not", "didn't": "did not", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
            "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
            "it's": "it is", "that's": "that is", "there's": "there is",
            "we're": "we are", "they're": "they are", "you're": "you are",
        }
        for contraction, expansion in contractions.items():
            for m in re.finditer(rf"\b{re.escape(contraction)}\b", text, re.IGNORECASE):
                if not self._is_protected(text, m.start(), m.end()):
                    matched = m.group(0)
                    replacement = expansion.capitalize() if matched[0].isupper() else expansion
                    patches.append((m.start(), m.end(), replacement))
        if not patches:
            return None
        return StyleIssue(
            rule_name="contractions",
            severity=Severity.WARNING,
            category=Category.GRAMMAR,
            description="Do not use contractions in formal writing (Amida Style Guide p.6)",
            location=f"slide {slide_idx} - element {elem_idx}",
            found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx,
            element_index=elem_idx,
            confidence=1.0,
        )

    def _check_and_or(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> Optional[StyleIssue]:
        patches: List[tuple] = []
        for m in re.finditer(r"\band/or\b", text, re.IGNORECASE):
            if not self._is_protected(text, m.start(), m.end()):
                patches.append((m.start(), m.end(), "or"))
        if not patches:
            return None
        return StyleIssue(
            rule_name="and_or",
            severity=Severity.WARNING,
            category=Category.GRAMMAR,
            description="Do not use 'and/or' in formal writing (Amida Style Guide p.6)",
            location=f"slide {slide_idx} - element {elem_idx}",
            found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx,
            element_index=elem_idx,
            confidence=1.0,
        )

    def _check_ampersand(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> Optional[StyleIssue]:
        patches: List[tuple] = []
        for m in re.finditer(r"\s+&\s+", text):
            if not self._is_protected(text, m.start(), m.end()):
                patches.append((m.start(), m.end(), " and "))
        if not patches:
            return None
        return StyleIssue(
            rule_name="ampersand",
            severity=Severity.WARNING,
            category=Category.GRAMMAR,
            description="Avoid the use of '&' in formal writing (Amida Style Guide p.6)",
            location=f"slide {slide_idx} - element {elem_idx}",
            found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx,
            element_index=elem_idx,
            confidence=0.9,
        )
    
    def _adjacent_decimal(self, text: str, start: int, end: int) -> bool:
        # digit '.' <here>  or  <here> '.' digit
        return (
            (start >= 2 and text[start-1] == '.' and text[start-2].isdigit()) or
            (end+1 < len(text) and text[end] == '.' and text[end+1].isdigit())
        )

    def _check_numerals(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> List[StyleIssue]:
        issues: List[StyleIssue] = []
        spell_patches: List[tuple] = []

        # NEW (skip decimals/versions like 8.9, 2.3.4, and comma decimals 8,9):
        for m in re.finditer(r"(?<![\d.,])(\d{1,2})(?![\d.,])", text):
            start, end = m.span(1)
            if self._is_protected(text, start, end):
                continue
            if not self._is_sentence_start(text, start):
                continue
            if self._in_numeric_range(text, start, end):
                continue
            if self._in_code_token(text, start, end) or self._is_page_ref(text, start):
                continue
            if self._adjacent_decimal(text, start, end):
                continue
            nxt = self._next_word(text, end)
            if nxt in {"percent", "%", "million", "billion", "trillion"}:
                continue
            n = int(m.group(1))
            if 1 <= n <= 99:
                spell_patches.append((start, end, self._number_to_words(n)))

        if spell_patches:
            issues.append(StyleIssue(
                rule_name="numerals_spell_out",
                severity=Severity.SUGGESTION,
                category=Category.GRAMMAR,
                description="Spell out numbers below 100 in non-technical writing (Amida Style Guide p.5)",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=text,
                suggestion=self._apply_patches(text, spell_patches),
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=0.85,
            ))

        # Thousands separators for integers (skip decimals/years/etc.)
        comma_patches: List[tuple] = []
        for m in re.finditer(r"(?<![\d.])(\d{4,})(?!\d)", text):
            start, end = m.span(1)
            if self._is_protected(text, start, end):
                continue
            # Skip if this run of digits is adjacent to a decimal point, e.g., 0.8870
            if self._adjacent_decimal(text, start, end):
                continue
            if self._is_probable_year(text, start, end):
                continue
            if self._in_code_token(text, start, end) or self._is_page_ref(text, start):
                continue
            num = m.group(1)
            if "," not in num:
                comma_patches.append((start, end, f"{int(num):,}"))

        if comma_patches:
            issues.append(StyleIssue(
                rule_name="numerals_commas",
                severity=Severity.SUGGESTION,
                category=Category.GRAMMAR,
                description="Use commas for numbers of 4+ digits (Amida Style Guide p.5)",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=text,
                suggestion=self._apply_patches(text, comma_patches),
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=0.95,
            ))

        return issues

    def _number_to_words(self, n: int) -> str:
        ones = [
            "zero","one","two","three","four","five","six","seven","eight","nine",
            "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
            "seventeen","eighteen","nineteen"
        ]
        tens = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
        if n < 20:
            return ones[n]
        t, o = divmod(n, 10)
        return tens[t] if o == 0 else f"{tens[t]}-{ones[o]}"

    def _is_sentence_start(self, text: str, pos: int) -> bool:
        i = pos - 1
        while i >= 0 and text[i].isspace():
            i -= 1
        if i < 0:
            return True
        return text[i] in {'.', '!', '?', '\n', ':', ';'}

    def _next_word(self, text: str, pos: int) -> str:
        m = re.search(r"^\W*(\w+)", text[pos:])
        return m.group(1).lower() if m else ""

    def _check_period_spacing(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> Optional[StyleIssue]:
        patches: List[tuple] = []
        for m in re.finditer(r"(?:[.!?]) {2,}", text):
            start, end = m.span()
            if not self._is_protected(text, start, end):
                punct = text[start]
                patches.append((start, end, punct + " "))
        if not patches:
            return None
        return StyleIssue(
            rule_name="period_spacing",
            severity=Severity.SUGGESTION,
            category=Category.GRAMMAR,
            description="Only one space after period (Amida Style Guide p.5)",
            location=f"slide {slide_idx} - element {elem_idx}",
            found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx,
            element_index=elem_idx,
            confidence=1.0,
        )

    def _check_quotation_marks(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> Optional[StyleIssue]:
        patches: List[tuple] = []

        # Accept straight and smart closing quotes
        QUOTE = r"[\"'\u2019\u201D]"

        # 1) Move commas/periods INSIDE quotes:  'Title'.  ->  'Title.'
        for m in re.finditer(rf"({QUOTE})(\s*)([.,])", text):
            q, spaces, punct = m.group(1), m.group(2), m.group(3)
            start, end = m.span()
            if self._is_protected(text, start, end):
                continue
            # NOTE: order swapped so punctuation goes before the quote
            patches.append((start, end, f"{punct}{q}"))

        # 2) Keep semicolons/colons OUTSIDE quotes:  'Title';  (already correct)
        for m in re.finditer(rf"([;:])({QUOTE})", text):
            punct, q = m.group(1), m.group(2)
            start, end = m.span()
            if self._is_protected(text, start, end):
                continue
            patches.append((start, end, f"{q}{punct}"))

        if not patches:
            return None

        return StyleIssue(
            rule_name="quotation_marks",
            severity=Severity.SUGGESTION,
            category=Category.GRAMMAR,
            description="Commas/periods inside quotes; semicolons/colons outside (Amida Style Guide p.5)",
            location=f"slide {slide_idx} - element {elem_idx}",
            found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx,
            element_index=elem_idx,
            confidence=0.9,
        )


    def _check_hyphens(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> Optional[StyleIssue]:
        patches: List[tuple] = []
        for m in re.finditer(r"—", text):
            if not self._is_protected(text, m.start(), m.end()):
                patches.append((m.start(), m.end(), "–"))
        for m in re.finditer(r"--", text):
            if not self._is_protected(text, m.start(), m.end()):
                patches.append((m.start(), m.end(), "–"))
        if not patches:
            return None
        return StyleIssue(
            rule_name="hyphens",
            severity=Severity.SUGGESTION,
            category=Category.GRAMMAR,
            description="Use en dashes (–), not em dashes (—) or double hyphens (--) (Amida Style Guide p.5)",
            location=f"slide {slide_idx} - element {elem_idx}",
            found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx,
            element_index=elem_idx,
            confidence=1.0,
        )


# Word List Checker
class WordListChecker:
    def __init__(self, protection_data: Dict[str, Any]):
        self.protection_data = protection_data
        self.word_list = self._load_amida_word_list()

        # Handle e.g./i.e. separately with regex so we don't double-insert commas
        self._abbrev_patterns = [
            (re.compile(r'(?i)\b(e\.g\.)(?!\s*,)'), "e.g.,"),  # if not followed by comma
            (re.compile(r'(?i)\b(i\.e\.)(?!\s*,)'), "i.e.,"),  # if not followed by comma
        ]

    def _load_amida_word_list(self) -> Dict[str, str]:
        """Load Amida Style Guide word list (p.5-6)."""
        return {
            # Hyphenation rules
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
            # NOTE: remove "e.g." and "i.e." from here so we can control commas via regex
            # Capitalization preferences
            "machine learning": "Machine Learning",
            "artificial intelligence": "Artificial Intelligence",
            "natural language processing": "Natural Language Processing",
        }

    def check(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> List[StyleIssue]:
        issues: List[StyleIssue] = []
        patches: List[tuple] = []

        # 1) Abbreviation fix: add comma only if missing
        for rx, replacement in self._abbrev_patterns:
            for m in rx.finditer(text):
                start, end = m.span(1)  # only replace the matched token, not the following char
                if not self._is_protected(text, start, end):
                    # If already exactly "e.g.," / "i.e.,", skip (the negative lookahead should already avoid these)
                    patches.append((start, end, replacement[:-1]))  # replace 'e.g.' with 'e.g.'; comma added next line
                    # Insert the comma after the match
                    # NOTE: place the comma at 'end' position (after optional spaces is trickier; keep it simple)
                    patches.append((end, end, ","))

        # 2) Word list replacements (unchanged logic, but without e.g./i.e.)
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



    def _is_protected(self, text: str, start: int, end: int) -> bool:
        """Check if span is in protected items."""
        substring = text[start:end]
        for category_items in self.protection_data.values():
            for item in category_items:
                if item in substring:
                    return True
        return False

    def _apply_patches(self, text: str, patches: List[tuple]) -> str:
        """Apply replacements safely in reverse order."""
        if not patches:
            return text
        patches.sort(key=lambda p: p[0], reverse=True)
        result = text
        for start, end, replacement in patches:
            result = result[:start] + replacement + result[end:]
        return result


# Orchestrator
class AgenticStyleChecker:
    def __init__(self, use_llm: bool = True):
        self.llm = LLMClient() if use_llm else None
        self.protection_layer = ProtectionLayer(llm_client=self.llm)
        self.protection_data: Dict[str, Any] = {}
        self.stats = {
            "total_elements": 0,
            "protected_items": {},
            "by_severity": {s.value: 0 for s in Severity},
            "by_category": {c.value: 0 for c in Category},
        }

    def analyze_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("STYLE CHECKER - Starting document analysis")

        # STEP 1: Protected content via ProtectionLayer
        try:
            self.protection_data = self.protection_layer.detect_all_protected_content(document)
        except LLMConfigError as e:
            logger.error("Protection detection skipped: %s", e)
            self.protection_data = {}

        # Log stats
        total_protected = 0
        for category, items in self.protection_data.items():
            count = len(items)
            self.stats["protected_items"][category] = count
            total_protected += count
            logger.info(f"> {category}: {count} items")
        logger.info(f"Total protected items: {total_protected}")

        # STEP 2: Apply grammar & word list
        grammar_checker = GrammarChecker(self.protection_data)
        word_list_checker = WordListChecker(self.protection_data)

        all_issues: List[StyleIssue] = []
        for page in document.get("pages", []):
            slide_idx = page.get("index", 0)
            for elem in page.get("elements", []):
                text = (elem.get("text") or "").strip()
                if not text or len(text) < 3:
                    continue
                elem_idx = (elem.get("locator") or {}).get("element_index", 0)
                self.stats["total_elements"] += 1
                all_issues.extend(grammar_checker.check(text, elem, slide_idx, elem_idx))
                all_issues.extend(word_list_checker.check(text, elem, slide_idx, elem_idx))

        for issue in all_issues:
            self.stats["by_severity"][issue.severity.value] += 1
            self.stats["by_category"][issue.category.value] += 1

        return {
            "issues": [i.to_dict() for i in all_issues],
            "statistics": self.stats,
            "protection_data": self.protection_data,
            "total_issues": len(all_issues),
            "document_metadata": document.get("metadata", {}),
        }


def check_document(
    document: Dict[str, Any],
    *,
    use_llm: bool = True,
    return_wrapper: bool = False,
    precomputed_protection: Optional[Dict[str, Any]] = None,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    checker = AgenticStyleChecker(use_llm=use_llm)

    if precomputed_protection is not None:
        # Inject precomputed protection and bypass LLM calls
        checker.protection_data = precomputed_protection

        def _noop_detect(_doc: Dict[str, Any]) -> Dict[str, Any]:
            return precomputed_protection

        checker.protection_layer.detect_all_protected_content = _noop_detect  # type: ignore

    result = checker.analyze_document(document)
    return result if return_wrapper else result.get("issues", [])

# CLI entry
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m backend.analyzers.simple_style_checker <normalized_document.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    # Load document
    with input_path.open("r", encoding="utf-8") as f:
        document = json.load(f)

    # Always use LLM protection
    checker = AgenticStyleChecker(use_llm=True)
    result = checker.analyze_document(document)

    # Save to _analyzed.json automatically
    output_path = input_path.with_name(input_path.stem + "_analyzed.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Logging
    logger.info("Results saved to: %s", output_path)
    logger.info("Total issues found: %d", result["total_issues"])
    logger.info("By severity: %s", result["statistics"]["by_severity"])
    logger.info("By category: %s", result["statistics"]["by_category"])
    logger.info("Protected items: %s", result["protection_data"])

    # Preview first few issues
    for i, issue in enumerate(result["issues"][:5], 1):
        logger.info(
            "Issue %d: [%s] %s → Suggestion: %s",
            i, issue["severity"], issue["description"], issue["suggestion"]
        )


if __name__ == "__main__":
    main()