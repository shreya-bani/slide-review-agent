"""
Grammar & Style Analyzer (rules-first, model-ready)
Implements Amida Style Guide IV (A–H) using lightweight heuristics and
structured issue reporting, with optional LLM post-fixes via groq_grammar_improver.

Covers:
A. Acronyms
B. Bullets
C. Headings (validated against style rules metadata you already extract)
D. Hyphens / Dashes
E. Numerals
F. Periods / sentence spacing
G. Quotation marks
H. Titles (Title Case)

Usage:
    from analyzers.grammar_style_analyzer import GrammarStyleAnalyzer
    analyzer = GrammarStyleAnalyzer(rules_path="rules/amida_style_rules.json")
    issues = analyzer.analyze_element(text, meta={"element_type": "title", "style": {...}, "location": {...}})
    # issues -> list[GrammarIssue]
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Iterable, Tuple
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# Optional deps
try:
    from num2words import num2words  # for E. Numerals (fallback when present)
except Exception:  # pragma: no cover
    num2words = None



# Data classes
@dataclass
class GrammarIssue:
    """Represents a single rule violation or suggestion."""
    rule: str                 # e.g., "Acronyms", "Bullets", "Headings"
    severity: str             # "info" | "warn" | "error"
    message: str              # human-readable explanation
    span: Optional[Tuple[int, int]] = None  # (start, end) char indices in text if known
    suggestions: Optional[List[str]] = None # zero or more concrete fix suggestions
    meta: Optional[Dict[str, Any]] = None   # any extra data


@dataclass
class ElementAnalysis:
    element_id: str
    element_type: str
    issues: List[GrammarIssue]



# Helpers: text parsing
_SENT_END_RE = re.compile(r"([.!?])(\s*)")
_QUOTE_RE = re.compile(r"(['\"])")
_ACRO_RE = re.compile(r"\b([A-Z]{2,}(?:[/-][A-Z]{2,})*)\b")  # VA, DRE, AI/ML, NLP, etc.
_WORD_RE = re.compile(r"[A-Za-z]+")
_BULLET_LINE_RE = re.compile(r'^\s*(?:[-*•]|[0-9]+[.)])\s+')
_EM_DASH = "—"
_EN_DASH = "–"

def _sentences(text: str) -> List[str]:
    # simple sentence split consistent with spacing fixes (avoid NLTK dependency here)
    parts, start = [], 0
    for m in _SENT_END_RE.finditer(text):
        end = m.end(1)
        parts.append(text[start:end])
        start = m.end()
    if start < len(text):
        parts.append(text[start:])
    # Strip but keep empty if only whitespace
    return [s.strip() for s in parts if s.strip()]

def _title_case_amida(s: str, rules: Dict[str, Any]) -> str:
    """
    Title case per H. Titles:
    - Capitalize first & last
    - Capitalize main words (nouns/pronouns/verbs/adverbs/adjectives)
    - Capitalize any word > 3 letters
    - Lowercase short articles/conjunctions/prepositions unless first/last
    """
    if not s:
        return s
    short_words = {"a","an","the","and","but","or","nor","for","so","yet",
                   "at","by","in","of","on","per","to","via","as"}
    tokens = re.split(r"(\s+|-|/)", s.strip())
    words = [w for w in tokens if _WORD_RE.search(w)]
    out = []
    for i, tok in enumerate(tokens):
        if not _WORD_RE.search(tok):
            out.append(tok)
            continue
        is_first = (len(out) == 0 or not _WORD_RE.search(out[-1]))
        # find next word token from the right to detect last
        rest_words = [t for t in tokens[tokens.index(tok)+1:] if _WORD_RE.search(t)]
        is_last = (len(rest_words) == 0)
        low = tok.lower()
        if is_first or is_last or len(low) > 3 or low not in short_words:
            out.append(low.capitalize())
        else:
            out.append(low)
    return "".join(out)

def _has_multiple_sentences(bullet_text: str) -> bool:
    # naive: more than one sentence end punctuation in a single bullet
    return len(re.findall(r"[.!?]", bullet_text)) > 1



# Analyzer
class GrammarStyleAnalyzer:
    def __init__(self, rules_path: str = "rules/amida_style_rules.json"):
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"Style rules not found at {rules_path}")
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

        self.acronym_examples = set((self.rules.get("acronym_rules", {})
                                              .get("examples", {})).keys())
        self.temporal_avoid = set(self.rules.get("temporal_words_to_avoid", []))

    # Public API
    def analyze_element(self, text: str, meta: Optional[Dict[str, Any]] = None) -> List[GrammarIssue]:
        """
        meta:
            - element_type: "title" | "bullet" | "body" | "note" | etc.
            - style: dict (font/style/color/size/spacing)
            - raw_lines: Optional[List[str]]; when bullets already split as lines
        """
        meta = meta or {}
        element_type = (meta.get("element_type") or "").lower()
        style = meta.get("style", {})
        raw_lines: Optional[List[str]] = meta.get("raw_lines")

        issues: List[GrammarIssue] = []
        issues += self._check_acronyms(text)
        issues += self._check_dashes(text)
        issues += self._check_period_spacing(text)
        issues += self._check_quotations(text)

        # Numerals only for non-technical writing (heuristic: skip if code-like)
        if element_type in {"title", "body", "note", "caption"}:
            issues += self._check_numerals(text)

        if element_type == "title":
            issues += self._check_title_case(text)

        # Bullets: validate per-line when available; else do a best-effort parse.
        if element_type == "bullet":
            bullet_lines = raw_lines or [l for l in text.splitlines() if l.strip()]
            issues += self._check_bullets(bullet_lines)

        # Headings: validate only when style info is present
        if element_type in {"title", "heading"} and style:
            issues += self._check_headings(style)

        return issues

    # A. Acronyms
    def _check_acronyms(self, text: str) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        # Find all all-caps tokens (>=2 chars), excluding obvious words like USA already handled as acronyms.
        found = set(m.group(1) for m in _ACRO_RE.finditer(text))
        if not found:
            return issues

        for acro in sorted(found):
            # If the first occurrence lacks a preceding full form in parentheses on first use
            # Heuristic: look for a "(ACRO)" immediately after a capitalized phrase.
            # This analyzer flags; LLM can propose concrete rewrite with the full form.
            # Accept whitelist from examples.
            pattern = rf"\b\([ ]*{re.escape(acro)}[ ]*\)"
            if re.search(pattern, text):
                continue  # already parenthesized somewhere
            # If acronym itself appears in examples, still require first-use rule unless exact match with example full form appears
            ex_map = self.rules.get("acronym_rules", {}).get("examples", {})
            full_example = ex_map.get(acro)
            has_full_example = bool(full_example and full_example in text)
            if has_full_example:
                continue

            issues.append(GrammarIssue(
                rule="Acronyms",
                severity="warn",
                message=f"First use of acronym “{acro}” should include the full phrase before it, followed by ({acro}).",
                suggestions=[f"… {acro} → Full Phrase ({acro})"],
                meta={"acronym": acro}
            ))
        return issues

    # B. Bullets
    def _check_bullets(self, lines: Iterable[str]) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not _BULLET_LINE_RE.search(stripped):
                # Not a formal bullet marker; still check capitalization if it looks like a bullet row
                pass

            # No terminal period (unless truly a complete multi-sentence case)
            if stripped.endswith("."):
                issues.append(GrammarIssue(
                    rule="Bullets",
                    severity="warn",
                    message=f"Bullet {i} ends with a period; per style, omit periods at the end of bullet points.",
                    suggestions=[stripped.rstrip(".")]
                ))

            # Capitalize first letter (unless case-sensitive)
            text_only = re.sub(_BULLET_LINE_RE, "", stripped).lstrip()
            if text_only and text_only[0].isalpha() and not text_only[0].isupper():
                issues.append(GrammarIssue(
                    rule="Bullets",
                    severity="info",
                    message=f"Bullet {i} should start with a capital letter.",
                    suggestions=[stripped.replace(text_only[:1], text_only[:1].upper(), 1)]
                ))

            # Multiple sentences → suggest semicolons or sub-bullets
            if _has_multiple_sentences(text_only):
                issues.append(GrammarIssue(
                    rule="Bullets",
                    severity="info",
                    message=f"Bullet {i} contains multiple sentences; split into sub-bullets or use semicolons.",
                ))
        return issues

    # C. Headings (style validation)
    def _check_headings(self, style: Dict[str, Any]) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        # Your document extractor already captures font/style/color/size/spacing
        # We validate presence and mismatches against rules.heading_styles
        target_font = self.rules.get("heading_styles", {}).get("font")
        if target_font and (style.get("font") and style.get("font") != target_font):
            issues.append(GrammarIssue(
                rule="Headings",
                severity="info",
                message=f"Heading font is “{style.get('font')}”; expected “{target_font}”."
            ))
        # If you also pass level (1..4), compare size/color/style
        if "level" in style:
            key = f"heading_{style['level']}"
            want = self.rules.get("heading_styles", {}).get(key)
            if want:
                for attr, expected in want.items():
                    actual = style.get(attr.lower()) or style.get(attr)
                    if actual is not None and str(actual) != str(expected):
                        issues.append(GrammarIssue(
                            rule="Headings",
                            severity="info",
                            message=f"Heading {style['level']} {attr} is “{actual}”; expected “{expected}”.",
                            meta={"attribute": attr, "expected": expected, "actual": actual}
                        ))
        return issues

    # D. Hyphens / Dashes
    def _check_dashes(self, text: str) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        if _EM_DASH in text:
            issues.append(GrammarIssue(
                rule="Hyphens",
                severity="info",
                message="Replace em dash (—) with en dash (–).",
                suggestions=[text.replace(_EM_DASH, _EN_DASH)]
            ))
        if "--" in text:
            issues.append(GrammarIssue(
                rule="Hyphens",
                severity="info",
                message="Replace double hyphen (--) with en dash (–).",
                suggestions=[text.replace("--", _EN_DASH)]
            ))
        return issues

    # E. Numerals
    def _check_numerals(self, text: str) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []

        # (1) Avoid starting a sentence with a digit
        for s in _sentences(text):
            s_stripped = s.lstrip()
            if s_stripped and s_stripped[0].isdigit():
                issues.append(GrammarIssue(
                    rule="Numerals",
                    severity="info",
                    message="Avoid starting a sentence with a number; spell it out.",
                    suggestions=[re.sub(r"^\s*([0-9]+)", lambda m: self._safe_num2words(int(m.group(1))).capitalize(), s, count=1)]
                ))

        # (2) Comma for 4+ digits (outside dates)
        for m in re.finditer(r"\b(\d{4,})\b", text):
            n = m.group(1)
            if re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", text):  # crude date guard
                continue
            if "," not in n:
                pretty = f"{int(n):,}"
                issues.append(GrammarIssue(
                    rule="Numerals",
                    severity="info",
                    message=f"Use a comma for numbers of four digits or more: “{n}” → “{pretty}”.",
                    span=(m.start(1), m.end(1)),
                    suggestions=[text[:m.start(1)] + pretty + text[m.end(1):]]
                ))

        # (3) Spell out <100 in non-technical contexts (heuristic)
        for m in re.finditer(r"\b([1-9]\d?)\b", text):
            val = int(m.group(1))
            # skip when followed by % or million/billion (those must be numerals)
            tail = text[m.end(): m.end()+10]
            if re.match(r"\s*(%|percent|million|billion)\b", tail, re.I):
                continue
            word = self._safe_num2words(val)
            if word:
                issues.append(GrammarIssue(
                    rule="Numerals",
                    severity="info",
                    message=f"Spell out numbers below 100 in non-technical writing: “{val}”.",
                    span=(m.start(1), m.end(1)),
                    suggestions=[text[:m.start(1)] + word + text[m.end(1):]]
                ))
        return issues

    def _safe_num2words(self, n: int) -> Optional[str]:
        if num2words:
            try:
                return num2words(n)
            except Exception:
                return None
        return None

    # F. Periods (spacing)
    def _check_period_spacing(self, text: str) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        # One space after period; catch double spaces after [.?!]
        if re.search(r"([.!?]) {2,}", text):
            fixed = re.sub(r"([.!?]) {2,}", r"\1 ", text)
            issues.append(GrammarIssue(
                rule="Periods",
                severity="info",
                message="Use a single space after sentence-ending punctuation.",
                suggestions=[fixed]
            ))
        # Remove ".." occurrence (accidental double periods)
        if ".." in text:
            issues.append(GrammarIssue(
                rule="Periods",
                severity="info",
                message="Remove duplicated periods.",
                suggestions=[text.replace("..", ".")]
            ))
        return issues

    # G. Quotation marks
    def _check_quotations(self, text: str) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        # Heuristics: comma/period generally inside quotes in US style
        # Find patterns like "word".,  -> period outside -> should go inside
        if re.search(r'"\s*[.,](?!\s)"', text):
            pass  # rare edge; not common

        # comma/period outside then quote -> move inside
        pattern = re.compile(r'(".*?")([.,])')
        for m in pattern.finditer(text):
            inside, punct = m.group(1), m.group(2)
            if not inside.endswith(punct):
                fixed = text[:m.start()] + inside[:-1] + punct + '"' + text[m.end():]
                issues.append(GrammarIssue(
                    rule="Quotation Marks",
                    severity="info",
                    message="Place commas/periods inside quotation marks.",
                    span=(m.start(), m.end()),
                    suggestions=[fixed]
                ))
        return issues

    # H. Titles
    def _check_title_case(self, text: str) -> List[GrammarIssue]:
        want = _title_case_amida(text, self.rules.get("title_case_rules", {}))
        if want != text:
            return [GrammarIssue(
                rule="Titles",
                severity="info",
                message="Use Title Case for titles.",
                suggestions=[want]
            )]
        return []
