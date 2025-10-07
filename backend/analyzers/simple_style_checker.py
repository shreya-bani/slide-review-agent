"""
Restructured Grammar & Style Checker
Single upfront LLM call for comprehensive protection categorization
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from typing import Union

from ..config.settings import settings
from ..utils.llm_client import LLMClient

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(settings, "log_level", logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

log_dir = settings.log_dir  or (Path(settings.output_dir) / "logs")

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


# Comprehensive Protection Detector
class ComprehensiveProtectionDetector:
    """Single LLM call to categorize all protected content types (no rule-based fallback)"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        self.protection_data: Dict[str, Any] = {}



    def detect_all_protected_content(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Single LLM call to categorize all protected content in document."""
        all_texts = self._collect_document_texts(document)
        if not all_texts:
            return self._get_empty_protection_data()

        if not self.llm:
            logger.warning("LLM client not provided — skipping protection detection")
            return self._get_empty_protection_data()
        
        full_text = "\n---PAGE BREAK---\n".join(all_texts)
        combined_text = full_text[:12000]  # cap by characters
        llm_result = self._llm_comprehensive_detection(combined_text)
        if llm_result:
            self.protection_data = llm_result
            logger.info(
                "LLM protection detection: names=%d technical=%d dates=%d numbers=%d ids=%d abbr=%d",
                len(llm_result.get("protected_names", [])),
                len(llm_result.get("technical_terms", [])),
                len(llm_result.get("dates", [])),
                len(llm_result.get("numbers", [])),
                len(llm_result.get("ids", [])),
                len(llm_result.get("abbreviations", [])),
            )
            return llm_result

        logger.warning("LLM protection detection failed — using empty protection data")
        return self._get_empty_protection_data()

    def _collect_document_texts(self, document: Dict[str, Any]) -> List[str]:
        texts = []
        for page in document.get("pages", []):
            for elem in page.get("elements", []):
                text = (elem.get("text") or "").strip()
                if text and len(text) >= 3:
                    texts.append(text)
        return texts

    def _llm_comprehensive_detection(self, combined_text: str) -> Optional[Dict[str, Any]]:
        """Single LLM call to detect all protected content using shared LLMClient.chat()."""
        system_prompt = """
You are a content analyzer that identifies items that should NOT be modified by grammar/word-list rules.

TASK
Analyze the provided text and extract protected items into SIX lists. Return ONLY raw JSON (no prose, no markdown), using the exact schema and key order shown below.

CATEGORIES (precise definitions + examples)
1) "protected_names": Proper names of people, organizations, products/brands, places, government entities, programs, and two-letter state abbreviations when used as names (e.g., "Amida", "NIST", "Medicaid", "Texas", "SC", "NC", "Kaplan", "Zach Woodard").
   • Include multiword names/titles as they appear.
   • Include product/initiative names (e.g., "Agentic AI" if used as a named concept).

2) "technical_terms": Alphanumeric or symbolic identifiers and formal standards that look like codes, versions, SKUs, model names, or contract IDs (e.g., "DIR-CPO-5140", "VASRD-6846", "ICD-10", "ISO 9000", "RFC 7231", "v2.1.3", "A100", "GPT-4o", "DRE").
   • Include hyphenated or slashed codes and those mixing letters/digits.
   • Include agency/contract numbers, catalog numbers, and version strings.

3) "dates": Any date/time-like expressions, including:
   • Calendar forms: "May 2025", "April 30 2025", "August 28, 2025", "2025-04-30", "7/4/2025".
   • Standalone years in the range 1900–2099 when used as dates: "2013", "2027", "2028".
   • Ranges with hyphen/en dash/em dash: "2013-2049", "2023–3000".
   • Fiscal/quarter formats: "FY2025", "Q3 2025".
   • Decades: "1990s".
   Treat all of the above as DATES (not numbers).

4) "numbers": Numeric expressions that are not classified as dates, including:
   • Plain numbers and decimals: "3.5", "2517".
   • Numbers with separators or currency: "2,517", "$1,200".
   • Percentages/words: "50%", "25 percent".
   • Ranges: "1-20", "2–3".
   • Ordinals and unit-bearing values: "1st", "90-day", "300M", "1M", "3 GB".
   EXCLUDE anything already tagged as a date.

5) "abbreviations": Abbreviations/initialisms that commonly constrain punctuation or casing in editing.
   • Include dotted forms: "U.S.", "U.K.", "Dr.", "Mr.", "Prof."
   • Also include common undotted initialisms when tightly bound to grammar choices: "IT", "ROI", "CTOs", "CDOs", "CIOs", "NLT".
   • EXCLUDE Latin e.g./i.e. here (see Rules below).

6) "ids": Unique instance identifiers that refer to a specific entity within a catalog, registry, or system.
   • Examples: "DIR-CPO-5140", "DIR-CPO-5498", "EMP-0073", "INV-2025-1432", "JIRA-1234", "ID: 5f3b8c12".
   • Distinction:
     – "technical_terms" = standards or category codes (e.g., "ISO 9000").
     – "ids" = specific assigned identifiers (e.g., "ISO-9000-2025-001").

RULES & EDGE CASES
• Do NOT infer or invent items. Extract only exact substrings present in the text.
• Preserve the original text exactly (casing, punctuation, hyphens, spaces).
• Years: Treat 1900–2099 as DATES unless clearly part of a code (e.g., "ISO 9000" → technical_terms).
• Codes: Prefer "technical_terms" over "numbers" for mixed letter–digit tokens (e.g., DIR-CPO-5140).
• Page refs: Do NOT classify numerals immediately preceded by "p." or "pp." as numbers/dates (they remain unclassified).
• Ranges: Keep the entire range token (e.g., "2013–2026", "1-20", "2–3") as a single item in the appropriate category.
• URLs, emails, file paths, and obvious hashes are NOT protected items—ignore them.
• e.g./i.e.: Do NOT add them to "abbreviations". (They are handled by a separate rule in the checker.)
• Duplicates: List each unique item only once, preserving first-seen order.
• If a list has no items, return an empty array.
• If anything is ambiguous, choose the most conservative category that prevents harmful edits (e.g., err toward "technical_terms" for letter–digit hybrids, and "dates" for 1900–2099 years).

RESPONSE FORMAT
Return ONLY a JSON object with these exact keys:
{
  "protected_names": [...],
  "technical_terms": [...],
  "dates": [...],
  "numbers": [...],
  "abbreviations": [...],
  "ids": [...]
}
"""
        prompt = f"Text to analyze:\n\n{combined_text[:10000]}"  # Limit text length

        # Use the shared LLM client (OpenAI-style chat completions)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        content = self.llm.chat(messages)  # returns str or None
        if not content:
            return None

        try:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group(0))
                required_keys = ["protected_names", "technical_terms", "dates", "numbers", "abbreviations", "ids"]
                if all(k in data for k in required_keys):
                    return data
                logger.warning("LLM response missing required keys. Found: %s", list(data.keys()))
        except Exception as e:
            logger.warning("Failed to parse LLM protection response: %s", e)

        return None

    def _get_empty_protection_data(self) -> Dict[str, Any]:
        return {
            "protected_names": [],
            "technical_terms": [],
            "dates": [],
            "numbers": [],
            "abbreviations": [],
            "ids": [],
        }

    def _is_protected(self, text: str, start: int, end: int) -> bool:
        """Check if span overlaps any protected content (bidirectional, case-insensitive)."""
        substring = text[start:end].strip()
        if not substring:
            return False
        low_sub = substring.lower()
        for category_items in self.protection_data.values():
            for item in category_items:
                li = str(item).strip().lower()
                if not li:
                    continue
                if low_sub in li or li in low_sub:
                    return True
        return False


# Grammar Checker (rules only)
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

    _MONTHS = {"january","february","march","april","may","june","july","august","september","october","november","december"}

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

    # rules
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

    def _check_numerals(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> List[StyleIssue]:
        issues: List[StyleIssue] = []
        spell_patches: List[tuple] = []
        for m in re.finditer(r"\b(\d{1,2})\b", text):
            start, end = m.span(1)
            if self._is_protected(text, start, end):
                continue
            if not self._is_sentence_start(text, start):
                continue
            if self._in_numeric_range(text, start, end):
                continue
            if self._in_code_token(text, start, end) or self._is_page_ref(text, start):
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

        comma_patches: List[tuple] = []
        for m in re.finditer(r"\b(\d{4,})\b", text):
            start, end = m.span(1)
            if self._is_protected(text, start, end):
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
        for m in re.finditer(r'(["\'])(\s*)([.,])', text):
            q, spaces, punct = m.group(1), m.group(2), m.group(3)
            start, end = m.span()
            if self._is_protected(text, start, end):
                continue
            patches.append((start, end, f'{q}{punct}'))
        for m in re.finditer(r'([;:])(["\'])', text):
            punct, q = m.group(1), m.group(2)
            start, end = m.span()
            if self._is_protected(text, start, end):
                continue
            patches.append((start, end, f'{q}{punct}'))
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

# Word List Checker (Amida Style Guide)
class WordListChecker:
    def __init__(self, protection_data: Dict[str, Any]):
        self.protection_data = protection_data
        self.word_list = self._load_amida_word_list()
        self.abbreviations = {"e.g.": "e.g.,", "i.e.": "i.e.,"}

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
            # Abbreviations requiring commas (p.6)
            "e.g.": "e.g.,",
            "i.e.": "i.e.,",
            # Capitalization preferences
            "machine learning": "Machine Learning",
            "artificial intelligence": "Artificial Intelligence",
            "natural language processing": "Natural Language Processing",
        }

    def check(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> List[StyleIssue]:
        """Check text against Amida Style Guide word list rules."""
        issues: List[StyleIssue] = []
        patches: List[tuple] = []

        for wrong, correct in self.word_list.items():
            if "." in wrong:
                pat = re.escape(wrong)  # exact match for abbreviations
            else:
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
                description="Apply Amida Word List preferences (Amida Style Guide p.5-6)",
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
        self.protection_detector = ComprehensiveProtectionDetector(self.llm)
        self.protection_data: Dict[str, Any] = {}
        self.stats = {
            "total_elements": 0,
            "protected_items": {},
            "by_severity": {s.value: 0 for s in Severity},
            "by_category": {c.value: 0 for c in Category},
        }

    def analyze_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("AMIDA STYLE CHECKER - Starting document analysis")

        # STEP 1: Single upfront LLM call for all protected content
        logger.info("\nSTEP 1: Detecting protected content (LLM-only, no fallback)...")
        self.protection_data = self.protection_detector.detect_all_protected_content(document)

        total_protected = 0
        for category, items in self.protection_data.items():
            count = len(items)
            self.stats["protected_items"][category] = count
            total_protected += count
            logger.info(f"> {category}: {count} items")
        logger.info(f"\nTotal protected items: {total_protected}")

        # STEP 2: Apply grammar and word list rules
        logger.info("\nSTEP 2: Applying Amida Style Guide rules...")
        grammar_checker = GrammarChecker(self.protection_data)
        word_list_checker = WordListChecker(self.protection_data)

        all_issues: List[StyleIssue] = []
        for page in document.get("pages", []):
            slide_idx = page.get("index", 0)
            for elem in page.get("elements", []):
                text = (elem.get("text") or "").strip()
                if not text or len(text) < 3:
                    continue

                loc = elem.get("locator", {}) or {}
                elem_idx = loc.get("element_index", 0)
                self.stats["total_elements"] += 1

                all_issues.extend(grammar_checker.check(text, elem, slide_idx, elem_idx))
                all_issues.extend(word_list_checker.check(text, elem, slide_idx, elem_idx))

                if self.stats["total_elements"] % 10 == 0:
                    logger.info(
                        "  Processed %d elements, found %d issues",
                        self.stats["total_elements"],
                        len(all_issues),
                    )

        for issue in all_issues:
            self.stats["by_severity"][issue.severity.value] += 1
            self.stats["by_category"][issue.category.value] += 1

        logger.info("ANALYSIS COMPLETED!")
        logger.info(f"Total elements analyzed: {self.stats['total_elements']}")
        logger.info(f"Total issues found: {len(all_issues)}")
        logger.info(f"Protected items: {total_protected}")

        logger.info("\n> Issues by severity:")
        for severity, count in sorted(self.stats["by_severity"].items()):
            if count > 0:
                logger.info(f"  {severity}: {count}")

        logger.info("\n> Issues by category:")
        for category, count in sorted(self.stats["by_category"].items()):
            if count > 0:
                logger.info(f"  {category}: {count}")

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
        checker.protection_detector.protection_data = precomputed_protection
        checker.protection_data = precomputed_protection

        def _noop_detect(_doc: Dict[str, Any]) -> Dict[str, Any]:
            return precomputed_protection
        checker.protection_detector.detect_all_protected_content = _noop_detect  # type: ignore

    result = checker.analyze_document(document)
    return result if return_wrapper else result.get("issues", [])


# CLI entry-
def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM-Only Style Checker")
    parser.add_argument("input", help="Normalized document JSON")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (will result in no protection)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        document = json.load(f)

    checker = AgenticStyleChecker(use_llm=not args.no_llm)
    result = checker.analyze_document(document)

    output_path = args.output or args.input.replace(".json", "_analyzed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", output_path)
    logger.info("Protected items: %s", result["protection_data"])


if __name__ == "__main__":
    main()
