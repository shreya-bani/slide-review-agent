from __future__ import annotations
import re
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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


class GrammarChecker:
    """
    This is to focus only on grammar rules.
    """
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
        "january","february","march","april","may","june",
        "july","august","september","october","november","december"
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
        return bool(re.search(r'[A-Za-z]-?[A-Za-z]{1,5}$', left) or re.search(r'^[A-Za-z-]{1,6}', right))

    def _is_page_ref(self, text: str, start: int) -> bool:
        left = text[max(0, start-3):start]
        return bool(re.search(r'(?:^|\s)p\.?$', left, re.IGNORECASE) or re.search(r'(?:^|\s)pp\.?$', left, re.IGNORECASE))

    def _in_numeric_range(self, text: str, start: int, end: int) -> bool:
        before = text[max(0, start-1):start]
        after = text[end:end+1]
        return (before in {'-', '–'} or after in {'-', '–'})

    def _capitalize_number_words(self, s: str) -> str:
        return s if not s else s[0].upper() + s[1:]

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

    # ---------- grammar rules (unchanged behavior) ----------
    def _check_contractions(self, text, elem, slide_idx, elem_idx) -> Optional[StyleIssue]:
        patches: List[tuple] = []
        contractions = {
            "can't":"cannot","won't":"will not","don't":"do not","doesn't":"does not","didn't":"did not",
            "isn't":"is not","aren't":"are not","wasn't":"was not","weren't":"were not","hasn't":"has not",
            "haven't":"have not","hadn't":"had not","couldn't":"could not","shouldn't":"should not","wouldn't":"would not",
            "it's":"it is","that's":"that is","there's":"there is","we're":"we are","they're":"they are","you're":"you are",
        }
        for c, expansion in contractions.items():
            for m in re.finditer(rf"\b{re.escape(c)}\b", text, re.IGNORECASE):
                if not self._is_protected(text, m.start(), m.end()):
                    matched = m.group(0)
                    replacement = expansion.capitalize() if matched[0].isupper() else expansion
                    patches.append((m.start(), m.end(), replacement))
        if not patches:
            return None
        return StyleIssue(
            rule_name="contractions", severity=Severity.WARNING, category=Category.GRAMMAR,
            description="Do not use contractions in formal writing (Amida Style Guide p.6)",
            location=f"slide {slide_idx} - element {elem_idx}", found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx, element_index=elem_idx, confidence=1.0
        )

    def _check_and_or(self, text, elem, slide_idx, elem_idx):
        patches: List[tuple] = []
        for m in re.finditer(r"\band/or\b", text, re.IGNORECASE):
            if not self._is_protected(text, m.start(), m.end()):
                patches.append((m.start(), m.end(), "or"))
        if not patches:
            return None
        return StyleIssue(
            rule_name="and_or", severity=Severity.WARNING, category=Category.GRAMMAR,
            description="Do not use 'and/or' in formal writing (Amida Style Guide p.6)",
            location=f"slide {slide_idx} - element {elem_idx}", found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx, element_index=elem_idx, confidence=1.0
        )

    def _check_ampersand(self, text, elem, slide_idx, elem_idx):
        patches: List[tuple] = []
        for m in re.finditer(r"\s+&\s+", text):
            if not self._is_protected(text, m.start(), m.end()):
                patches.append((m.start(), m.end(), " and "))
        if not patches:
            return None
        return StyleIssue(
            rule_name="ampersand", severity=Severity.WARNING, category=Category.GRAMMAR,
            description="Avoid the use of '&' in formal writing (Amida Style Guide p.6)",
            location=f"slide {slide_idx} - element {elem_idx}", found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx, element_index=elem_idx, confidence=0.9
        )

    def _adjacent_decimal(self, text: str, start: int, end: int) -> bool:
        return ((start >= 2 and text[start-1] == '.' and text[start-2].isdigit())
                or (end+1 < len(text) and text[end] == '.' and text[end+1].isdigit()))

    def _check_numerals(self, text, elem, slide_idx, elem_idx):
        issues: List[StyleIssue] = []
        spell_patches: List[tuple] = []
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
            if 1 <= n <= 99 and self._is_sentence_start(text, start):
                word = self._number_to_words(n)
                word = self._capitalize_number_words(word)
                spell_patches.append((start, end, word))
        if spell_patches:
            issues.append(StyleIssue(
                rule_name="numerals_spell_out", severity=Severity.SUGGESTION, category=Category.GRAMMAR,
                description="Spell out numbers below 100 in non-technical writing (Amida Style Guide p.5)",
                location=f"slide {slide_idx} - element {elem_idx}", found_text=text,
                suggestion=self._apply_patches(text, spell_patches),
                page_or_slide_index=slide_idx, element_index=elem_idx, confidence=0.85
            ))
        comma_patches: List[tuple] = []
        for m in re.finditer(r"(?<![\d.])(\d{4,})(?!\d)", text):
            start, end = m.span(1)
            if self._is_protected(text, start, end):
                continue
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
                rule_name="numerals_commas", severity=Severity.SUGGESTION, category=Category.GRAMMAR,
                description="Use commas for numbers of 4+ digits (Amida Style Guide p.5)",
                location=f"slide {slide_idx} - element {elem_idx}", found_text=text,
                suggestion=self._apply_patches(text, comma_patches),
                page_or_slide_index=slide_idx, element_index=elem_idx, confidence=0.95
            ))
        return issues

    def _number_to_words(self, n: int) -> str:
        ones = ["zero","one","two","three","four","five","six","seven","eight","nine",
                "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
                "seventeen","eighteen","nineteen"]
        tens = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
        if n < 20: return ones[n]
        t, o = divmod(n, 10)
        return tens[t] if o == 0 else f"{tens[t]}-{ones[o]}"

    def _is_sentence_start(self, text: str, pos: int) -> bool:
        i = pos - 1
        while i >= 0 and text[i].isspace():
            i -= 1
        OPENERS = {'"', "'", '“', '‘', '(', '[', '{'}
        while i >= 0 and text[i] in OPENERS:
            i -= 1
            while i >= 0 and text[i].isspace():
                i -= 1
        if i < 0:
            return True
        return text[i] in {'.','!','?','\n',':',';'}

    def _next_word(self, text: str, pos: int) -> str:
        m = re.search(r"^\W*(\w+)", text[pos:])
        return m.group(1).lower() if m else ""

    def _check_period_spacing(self, text, elem, slide_idx, elem_idx):
        patches: List[tuple] = []
        for m in re.finditer(r"(?:[.!?]) {2,}", text):
            start, end = m.span()
            if not self._is_protected(text, start, end):
                punct = text[start]
                patches.append((start, end, punct + " "))
        if not patches:
            return None
        return StyleIssue(
            rule_name="period_spacing", severity=Severity.SUGGESTION, category=Category.GRAMMAR,
            description="Only one space after period (Amida Style Guide p.5)",
            location=f"slide {slide_idx} - element {elem_idx}", found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx, element_index=elem_idx, confidence=1.0
        )

    def _check_quotation_marks(self, text, elem, slide_idx, elem_idx):
        patches: List[tuple] = []
        QUOTE = r"[\"'\u2019\u201D]"
        for m in re.finditer(rf"({QUOTE})(\s*)([.,])", text):
            q, spaces, punct = m.group(1), m.group(2), m.group(3)
            start, end = m.span()
            if self._is_protected(text, start, end):
                continue
            patches.append((start, end, f"{punct}{q}"))
        for m in re.finditer(rf"([;:])({QUOTE})", text):
            punct, q = m.group(1), m.group(2)
            start, end = m.span()
            if self._is_protected(text, start, end):
                continue
            patches.append((start, end, f"{q}{punct}"))
        if not patches:
            return None
        return StyleIssue(
            rule_name="quotation_marks", severity=Severity.SUGGESTION, category=Category.GRAMMAR,
            description="Commas/periods inside quotes; semicolons/colons outside (Amida Style Guide p.5)",
            location=f"slide {slide_idx} - element {elem_idx}", found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx, element_index=elem_idx, confidence=0.9
        )

    def _check_hyphens(self, text, elem, slide_idx, elem_idx):
        patches: List[tuple] = []
        for m in re.finditer(r"—", text):
            if not self._is_protected(text, m.start(), m.end()):
                patches.append((m.start(), m.end(), "–"))
        for m in re.finditer(r"-{2,}", text):
            if not self._is_protected(text, m.start(), m.end()):
                patches.append((m.start(), m.end(), "–"))
        if not patches:
            return None
        return StyleIssue(
            rule_name="hyphens", severity=Severity.SUGGESTION, category=Category.GRAMMAR,
            description="Use en dashes (–) instead of em dashes (—) or multiple hyphens (--, ---, …) (Amida Style Guide p.5).",
            location=f"slide {slide_idx} - element {elem_idx}", found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx, element_index=elem_idx, confidence=1.0
        )
