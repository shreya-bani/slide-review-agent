"""
Advanced Grammar & Word-List Style Checker
Agentic AI approach with LLM enhancement and intelligent caching
"""

import json
import os
import re
import sys
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import logging

# Add this near the top of the file, after other imports
logger = logging.getLogger(__name__)
# ============================================================================
# CONFIGURATION
# ============================================================================

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
    
    def to_dict(self):
        d = asdict(self)
        d['severity'] = self.severity.value
        d['category'] = self.category.value
        return d

# ============================================================================
# LLM CLIENT WITH SMART CACHING
# ============================================================================

class LLMClient:
    def __init__(self, api_key: str = None, cache_file: str = ".llm_cache.json"):
        self.api_key = api_key or os.getenv("HF_API_KEY", "")
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.endpoint = "https://router.huggingface.co/v1/chat/completions"
        self.model = "google/gemma-2-2b-it:nebius"
        
    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Cache save failed: {e}", file=sys.stderr)
    
    def _cache_key(self, prompt: str, temperature: float) -> str:
        return hashlib.md5(f"{self.model}:{temperature}:{prompt}".encode()).hexdigest()
    
    def call(self, prompt: str, system: str = "", temperature: float = 0.1, max_tokens: int = 512) -> Optional[str]:
        if not self.api_key:
            return None
        
        cache_key = self._cache_key(prompt, temperature)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            response.raise_for_status()
            
            content = response.json()["choices"][0]["message"]["content"].strip()
            self.cache[cache_key] = content
            self._save_cache()
            return content
            
        except Exception as e:
            print(f"LLM call failed: {e}", file=sys.stderr)
            return None

# ============================================================================
# GRAMMAR RULES ENGINE
# ============================================================================

class GrammarChecker:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self):
        return {
            'numerals': self._check_numerals,
            'period_spacing': self._check_period_spacing,
            'quotation_marks': self._check_quotation_marks,
            'formal_writing': self._check_formal_writing,
            'bullets': self._check_bullets,
            'titles': self._check_titles,
        }
    
    def check(self, text: str, elem: dict, slide_idx: int, element_index: int) -> List[StyleIssue]:
        """Run all grammar checks on text"""
        issues = []
        for rule_name, rule_func in self.rules.items():
            try:
                result = rule_func(text, elem, slide_idx, element_index)
                if result:
                    issues.extend(result if isinstance(result, list) else [result])
            except Exception as e:
                print(f"Rule {rule_name} failed: {e}", file=sys.stderr)
        return issues
    
    # ========================================================================
    # NUMERAL RULES
    # ========================================================================
    
    def _check_numerals(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> List[StyleIssue]:
        """Check numeral spelling and comma formatting"""
        issues = []
        
        # Spell out numbers 1-99
        spell_patches = self._collect_numeral_spelling(text)
        if spell_patches:
            issues.append(StyleIssue(
                rule_name="numerals",
                severity=Severity.SUGGESTION,
                category=Category.GRAMMAR,
                description="Spell out numbers <100 (and at sentence start). Keep numerals for percent/million/billion unless sentence start.",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=text,
                suggestion=self._apply_patches(text, spell_patches),
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=0.95
            ))
        
        # Add commas to 4+ digit numbers
        comma_patches = self._collect_comma_formatting(text)
        if comma_patches:
            issues.append(StyleIssue(
                rule_name="numerals",
                severity=Severity.SUGGESTION,
                category=Category.GRAMMAR,
                description="Use commas for 4+ digits (except dates/IDs/URLs/years).",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=text,
                suggestion=self._apply_patches(text, comma_patches),
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=0.9
            ))
        
        return issues
    
    def _collect_numeral_spelling(self, text: str) -> List[Tuple[int, int, str]]:
        """Collect patches for spelling out numbers 1-99"""
        patches = []
        num_pattern = re.compile(r'\b(\d+)\b')
        
        for m in num_pattern.finditer(text):
            num_str = m.group(1)
            start, end = m.span(1)
            
            # Skip if in excluded context
            if self._is_excluded_context(text, start, end):
                continue
            
            try:
                n = int(num_str)
                if 1 <= n <= 99:
                    # Check if sentence start or if followed by unit word
                    at_start = self._is_sentence_start(text, start)
                    next_word = self._next_word(text, end)
                    
                    # Spell out if at sentence start or not followed by unit
                    if at_start or next_word not in {'percent', '%', 'million', 'billion'}:
                        patches.append((start, end, self._number_to_words(n)))
            except ValueError:
                continue
        
        return patches
    
    def _collect_comma_formatting(self, text: str) -> List[Tuple[int, int, str]]:
        """Add commas to numbers with 4+ digits"""
        patches = []
        
        for m in re.finditer(r'\b(\d{4,})\b', text):
            start, end = m.span(1)
            num_str = m.group(1)
            
            if self._is_excluded_context(text, start, end):
                continue
            
            if ',' not in num_str:
                formatted = f"{int(num_str):,}"
                patches.append((start, end, formatted))
        
        return patches
    
    def _is_excluded_context(self, text: str, start: int, end: int) -> bool:
        """Check if number is in excluded context (date, URL, ID, etc)"""
        window = text[max(0, start-30):min(len(text), end+30)]
        
        # Check for dates
        if self._looks_like_date(text, start, end):
            return True
        
        # Check for URLs/paths
        if re.search(r'(https?://|www\.|\.com|\.org|/|\\)', window, re.IGNORECASE):
            return True
        
        # Check for IDs (e.g., "ID-12345")
        if re.search(r'\b[A-Z]{2,}-[A-Z\d-]+', window):
            return True
        
        # Check for page numbers
        prev_word = self._prev_word(text, start)
        if prev_word in {'page', 'p', 'sec', 'section', 'schedule'}:
            return True
        
        return False
    
    def _looks_like_date(self, text: str, start: int, end: int) -> bool:
        """Check if number looks like a year or date"""
        num_str = text[start:end]
        
        # 4-digit year
        if len(num_str) == 4 and num_str.isdigit():
            year = int(num_str)
            if 1900 <= year <= 2100:
                return True
        
        # Month name nearby
        months = {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                  'january', 'february', 'march', 'april', 'june', 'july', 'august', 
                  'september', 'october', 'november', 'december'}
        
        prev = self._prev_word(text, start)
        if prev in months:
            return True
        
        return False
    
    def _is_sentence_start(self, text: str, pos: int) -> bool:
        """Check if position is at sentence start"""
        i = pos - 1
        while i >= 0 and text[i].isspace():
            i -= 1
        if i < 0:
            return True
        return text[i] in {'.', '!', '?', '\n', ':', ';'}
    
    def _prev_word(self, text: str, pos: int) -> str:
        """Get word immediately before position"""
        m = re.search(r'(\w+)\W*$', text[:pos])
        return m.group(1).lower() if m else ""
    
    def _next_word(self, text: str, pos: int) -> str:
        """Get word immediately after position"""
        m = re.search(r'^\W*(\w+)', text[pos:])
        return m.group(1).lower() if m else ""
    
    def _number_to_words(self, n: int) -> str:
        """Convert number to words (1-99)"""
        ones = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
                'seventeen', 'eighteen', 'nineteen']
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        
        if n < 20:
            return ones[n]
        
        tens_digit, ones_digit = divmod(n, 10)
        if ones_digit == 0:
            return tens[tens_digit]
        return f"{tens[tens_digit]}-{ones[ones_digit]}"
    
    # ========================================================================
    # SPACING RULES
    # ========================================================================
    
    def _check_period_spacing(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> Optional[StyleIssue]:
        """Check for single space after periods"""
        pattern = r'(?:[.!?…]|\.{3}) {2,}'
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            return None
        
        patches = []
        for m in matches:
            start, end = m.span()
            punct = text[start:start+1] if text[start] in '.!?' else '…'
            patches.append((start, end, punct + ' '))
        
        return StyleIssue(
            rule_name="period_spacing",
            severity=Severity.SUGGESTION,
            category=Category.GRAMMAR,
            description="Use exactly one space after sentence-ending punctuation.",
            location=f"slide {slide_idx} - element {elem_idx}",
            found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx,
            element_index=elem_idx,
            confidence=1.0
        )
    
    # ========================================================================
    # QUOTATION MARK RULES
    # ========================================================================
    
    def _check_quotation_marks(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> Optional[StyleIssue]:
        """Fix quotation mark punctuation placement"""
        patches = []
        
        # Pattern 1: Comma/period OUTSIDE closing quote → move INSIDE
        pattern1 = r'([\"”’])\s*([.,])(?=\s|$)'
        for m in re.finditer(pattern1, text):
            quote_part = m.group(1).strip()
            punct = m.group(2)
            
            # Verify there's an opening quote before this
            before = text[:m.start()]
            if self._has_matching_open_quote(before, quote_part):
                patches.append((m.start(), m.end(), punct + quote_part))
        
        # Pattern 2: Semicolon/colon INSIDE closing quote → move OUTSIDE  
        pattern2 = r'([;:])([\"”’])(?=\s|$)'
        for m in re.finditer(pattern2, text):
            punct = m.group(1)
            quote = m.group(2)
            
            # Verify this is actually closing a quote
            before = text[:m.start()]
            if self._has_matching_open_quote(before, quote):
                patches.append((m.start(), m.end(), quote + punct))
        
        if not patches:
            return None
        
        return StyleIssue(
            rule_name="quotation_marks",
            severity=Severity.SUGGESTION,
            category=Category.GRAMMAR,
            description="Place commas/periods inside closing quotes; semicolons/colons outside.",
            location=f"slide {slide_idx} - element {elem_idx}",
            found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx,
            element_index=elem_idx,
            confidence=0.85
        )
    
    def _has_matching_open_quote(self, before_text: str, close_quote: str) -> bool:
        """Check if there's a matching opening quote"""
# Check if there's a matching opening quote
        quote_map = {
            '"': '"',     # straight double
            '“': '“',     # opening curly double
            '”': '“',     # closing curly double → maps to opening curly
            "'": "'",     # straight single
            "‘": "‘",     # opening curly single
            "’": "‘",     # closing curly single → maps to opening curly
        }
        open_quote = quote_map.get(close_quote, close_quote)
        
        # Simple check: is there an unmatched opening quote?
        opens = before_text.count(open_quote)
        closes = before_text.count(close_quote)
        return opens > closes
    
    # ========================================================================
    # FORMAL WRITING RULES
    # ========================================================================
    
    def _check_formal_writing(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> Optional[StyleIssue]:
        """Check for contractions, and/or, etc., &"""
        patches = []
        
        # Contractions
        contractions = {
            "can't": "cannot", "won't": "will not", "don't": "do not",
            "doesn't": "does not", "didn't": "did not", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
            "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
            "it's": "it is", "that's": "that is", "there's": "there is",
        }
        
        for contraction, expansion in contractions.items():
            for m in re.finditer(rf'\b{re.escape(contraction)}\b', text, re.IGNORECASE):
                match_text = m.group()
                # Preserve capitalization
                if match_text[0].isupper():
                    repl = expansion[0].upper() + expansion[1:]
                else:
                    repl = expansion
                patches.append((m.start(), m.end(), repl))
        
        # and/or
        for m in re.finditer(r'\band/or\b', text):
            patches.append((m.start(), m.end(), 'or'))
        
        # & in formal text
        for m in re.finditer(r'\s+&\s+', text):
            patches.append((m.start(), m.end(), ' and '))
        
        if not patches:
            return None
        
        return StyleIssue(
            rule_name="formal_writing",
            severity=Severity.WARNING,
            category=Category.GRAMMAR,
            description="Avoid contractions, 'and/or', 'etc.', and '&' in formal writing.",
            location=f"slide {slide_idx} - element {elem_idx}",
            found_text=text,
            suggestion=self._apply_patches(text, patches),
            page_or_slide_index=slide_idx,
            element_index=elem_idx,
            confidence=1.0
        )
    
    # ========================================================================
    # BULLET RULES
    # ========================================================================
    
    def _check_bullets(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> List[StyleIssue]:
        """Check bullet formatting"""
        # Only process if this looks like bullet content
        if not self._is_bullet_element(text, elem):
            return []
        
        issues = []
        segments = self._extract_bullet_lines(text)
        
        for start, end, line in segments:
            # Check 1: Multiple sentences (no auto-fix)
            if self._has_multiple_sentences(line):
                issues.append(StyleIssue(
                    rule_name="bullets",
                    severity=Severity.SUGGESTION,
                    category=Category.GRAMMAR,
                    description="This bullet contains multiple sentences. Consider splitting into separate bullets or sub-bullets.",
                    location=f"slide {slide_idx} - element {elem_idx}",
                    found_text=text,
                    suggestion=text,  # No auto-fix
                    page_or_slide_index=slide_idx,
                    element_index=elem_idx,
                    confidence=0.9,
                    method="rule-based"
                ))
                continue
            
            # Check 2: Capitalization and period removal
            fixed = line
            if fixed and fixed[0].isalpha() and fixed[0].islower():
                fixed = fixed[0].upper() + fixed[1:]
            
            if fixed.rstrip().endswith('.'):
                fixed = fixed.rstrip('.').rstrip()
            
            if fixed != line:
                suggestion = text[:start] + fixed + text[end:]
                issues.append(StyleIssue(
                    rule_name="bullets",
                    severity=Severity.SUGGESTION,
                    category=Category.GRAMMAR,
                    description="Bullets: capitalize the first word; avoid periods at the end of bullet points.",
                    location=f"slide {slide_idx} - element {elem_idx}",
                    found_text=text,
                    suggestion=suggestion,
                    page_or_slide_index=slide_idx,
                    element_index=elem_idx,
                    confidence=0.95
                ))
        
        return issues
    
    def _is_bullet_element(self, text: str, elem: dict) -> bool:
        """Determine if element contains bullet content"""
        loc = elem.get("locator", {})
        if (loc.get("element_type") or "").lower() == "bullet":
            return True
        
        # Check for bullet markers
        bullet_pattern = r'^\s*[•◦▪◾○\-–—*]\s+'
        if re.match(bullet_pattern, text):
            return True
        
        # Check for numbered list
        if re.match(r'^\s*(?:\(?\d+\)?[.)]|\d+\))\s+', text):
            return True
        
        return False
    
    def _extract_bullet_lines(self, text: str) -> List[Tuple[int, int, str]]:
        """Extract individual bullet items from text"""
        lines = []
        offset = 0
        
        for raw_line in text.splitlines(keepends=True):
            line = raw_line.rstrip('\n\r')
            
            # Match bullet markers
            m_bullet = re.match(r'^\s*[•◦▪◾○\-–—*]\s+(.*)$', line)
            m_numbered = re.match(r'^\s*(?:\(?\d+\)?[.)]|\d+\))\s+(.*)$', line)
            
            if m_bullet or m_numbered:
                inner = (m_bullet or m_numbered).group(1)
                start = offset + (m_bullet.start(1) if m_bullet else m_numbered.start(1))
                end = offset + (m_bullet.end(1) if m_bullet else m_numbered.end(1))
                lines.append((start, end, inner))
            
            offset += len(raw_line)
        
        return lines
    
    def _has_multiple_sentences(self, text: str) -> bool:
        """Check if text contains multiple sentences"""
        # Pattern: period/!/?  followed by space and capital letter
        return bool(re.search(r'[.!?]\s+[A-Z]', text))
    
    # ========================================================================
    # TITLE CASE RULES
    # ========================================================================
    
    def _check_titles(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> Optional[StyleIssue]:
        """Check title case for title elements"""
        if not self._is_title_element(elem):
            return None
        
        title_cased = self._to_title_case(text)
        if title_cased == text:
            return None
        
        return StyleIssue(
            rule_name="titles",
            severity=Severity.SUGGESTION,
            category=Category.GRAMMAR,
            description="Use Title Case for titles.",
            location=f"slide {slide_idx} - element {elem_idx}",
            found_text=text,
            suggestion=title_cased,
            page_or_slide_index=slide_idx,
            element_index=elem_idx,
            confidence=0.9
        )
    
    def _is_title_element(self, elem: dict) -> bool:
        """Check if element is a title"""
        role = (elem.get("role") or "").lower()
        if role == "title":
            return True
        
        if elem.get("heading_level") == 1:
            return True
        
        st = elem.get("style", {})
        size = st.get("font_size") or elem.get("font_size")
        weight = (st.get("font_weight") or elem.get("font_weight") or "").lower()
        
        return size and isinstance(size, (int, float)) and size >= 18 and "bold" in weight
    
    def _to_title_case(self, text: str) -> str:
        """Convert text to title case per Amida style"""
        small_words = {
            'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'so', 'yet',
            'as', 'at', 'by', 'for', 'in', 'of', 'off', 'on', 'per', 'to', 'up', 'via',
            'v.', 'vs.', 'v', 'vs'
        }
        
        tokens = re.findall(r"\w[\w'/-]*|\s+|[^\w\s]", text, flags=re.UNICODE)
        words = [t for t in tokens if re.match(r'\w', t)]
        
        if not words:
            return text
        
        result = []
        word_positions = [i for i, t in enumerate(tokens) if re.match(r'\w', t)]
        first_pos = word_positions[0] if word_positions else -1
        last_pos = word_positions[-1] if word_positions else -1
        
        for i, token in enumerate(tokens):
            if not re.match(r'\w', token):
                result.append(token)
                continue
            
            is_first = (i == first_pos)
            is_last = (i == last_pos)
            
            # Always capitalize first and last
            if is_first or is_last:
                result.append(self._capitalize_word(token))
                continue
            
            # Capitalize if >3 chars or not a small word
            if len(token) > 3 or token.lower() not in small_words:
                result.append(self._capitalize_word(token))
            else:
                result.append(token.lower())
        
        return ''.join(result)
    
    def _capitalize_word(self, word: str) -> str:
        """Capitalize a word, handling hyphens and apostrophes"""
        if word.isupper():
            return word
        
        if '-' in word:
            return '-'.join(self._capitalize_word(part) for part in word.split('-'))
        
        if "'" in word:
            parts = word.split("'", 1)
            return parts[0][:1].upper() + parts[0][1:].lower() + "'" + parts[1]
        
        return word[:1].upper() + word[1:].lower()
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _apply_patches(self, text: str, patches: List[Tuple[int, int, str]]) -> str:
        """Apply patches to text in reverse order"""
        if not patches:
            return text
        
        patches.sort(key=lambda p: p[0], reverse=True)
        result = text
        
        for start, end, replacement in patches:
            result = result[:start] + replacement + result[end:]
        
        return result

# ============================================================================
# WORD LIST CHECKER
# ============================================================================

class WordListChecker:
    def __init__(self):
        self.word_list = self._load_word_list()
        self.abbreviations = {
            'e.g.': 'e.g.,',
            'i.e.': 'i.e.,',
        }
    
    def _load_word_list(self) -> Dict[str, str]:
        """Load Amida word list preferences"""
        return {
            # Hyphenation
            'built in': 'built-in',
            'cyber attacks': 'cyberattacks',
            'cyber security': 'cybersecurity',
            'cyber-security': 'cybersecurity',
            'code sets': 'code sets',
            'codesets': 'code sets',
            'code-sets': 'code sets',
            'end user': 'end user',  # noun
            'end-user': 'end-user',  # adjective (context-dependent)
            'health care': 'healthcare',
            'open source': 'open source',  # noun
            'open-source': 'open-source',  # adjective
            'public-sector': 'public sector',
            'user friendly': 'user-friendly',
            'web based': 'web-based',
            
            # Capitalization
            'machine learning': 'Machine Learning',
            'artificial intelligence': 'Artificial Intelligence',
            'natural language processing': 'Natural Language Processing',
        }
    
    def check(self, text: str, elem: dict, slide_idx: int, elem_idx: int) -> List[StyleIssue]:
        """Check text against word list"""
        issues = []
        patches = []
        
        # Simple word replacements
        for wrong, correct in self.word_list.items():
            pattern = r'\b' + re.escape(wrong) + r'\b'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                matched = m.group()
                if matched != correct:
                    patches.append((m.start(), m.end(), correct))
        
        if patches:
            issues.append(StyleIssue(
                rule_name="word_list",
                severity=Severity.SUGGESTION,
                category=Category.WORD_LIST,
                description="Apply Amida Word List preferences (hyphenation, noun/adj distinctions, capitalization).",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=text,
                suggestion=self._apply_patches(text, patches),
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=1.0
            ))
        
        # Abbreviation formatting (e.g., i.e.)
        abbr_patches = []
        for abbr, correct in self.abbreviations.items():
            pattern = re.escape(abbr) + r'\s*(?!,)'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                abbr_patches.append((m.start(), m.end(), correct + ' '))
        
        if abbr_patches:
            issues.append(StyleIssue(
                rule_name="word_list",
                severity=Severity.SUGGESTION,
                category=Category.WORD_LIST,
                description="Abbreviations like 'e.g.' and 'i.e.' should be followed by a comma.",
                location=f"slide {slide_idx} - element {elem_idx}",
                found_text=text,
                suggestion=self._apply_patches(text, abbr_patches),
                page_or_slide_index=slide_idx,
                element_index=elem_idx,
                confidence=1.0
            ))
        
        return issues
    
    def _apply_patches(self, text: str, patches: List[Tuple[int, int, str]]) -> str:
        """Apply patches in reverse order"""
        if not patches:
            return text
        
        patches.sort(key=lambda p: p[0], reverse=True)
        result = text
        
        for start, end, replacement in patches:
            result = result[:start] + replacement + result[end:]
        
        return result

# ============================================================================
# AGENTIC DOCUMENT ANALYZER
# ============================================================================

class AgenticStyleChecker:
    """
    Main agentic system that orchestrates grammar and word-list checking
    with intelligent decision-making and LLM enhancement
    """
    
    def __init__(self, use_llm: bool = True):
        self.llm = LLMClient() if use_llm else None
        self.grammar_checker = GrammarChecker(self.llm)
        self.word_checker = WordListChecker()
        self.stats = {
            'total_elements': 0,
            'llm_calls': 0,
            'rule_based': 0,
            'by_severity': {s.value: 0 for s in Severity},
            'by_category': {c.value: 0 for c in Category},
        }
    
    def analyze_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main agentic workflow:
        1. Parse document structure
        2. Identify elements requiring checking
        3. Apply rule-based checks
        4. Use LLM for complex cases
        5. Deduplicate and rank issues
        6. Return prioritized results
        """
        print("Agentic Style Checker initialized")
        print(f"LLM enhancement: {'enabled' if self.llm else 'disabled'}")
        print()
        
        all_issues = []
        
        for page in document.get('pages', []):
            slide_idx = page.get('index', 0)
            
            for elem in page.get('elements', []):
                text = (elem.get('text') or '').strip()
                if not text or len(text) < 3:
                    continue
                
                self.stats['total_elements'] += 1
                loc = elem.get('locator', {})
                elem_idx = loc.get('element_index', 0)
                
                # Run grammar checks
                grammar_issues = self.grammar_checker.check(text, elem, slide_idx, elem_idx)
                all_issues.extend(grammar_issues)
                
                # Run word-list checks
                word_issues = self.word_checker.check(text, elem, slide_idx, elem_idx)
                all_issues.extend(word_issues)
                
                # Progress indicator
                if self.stats['total_elements'] % 10 == 0:
                    print(f"Processed {self.stats['total_elements']} elements, found {len(all_issues)} issues")
        
        # Deduplicate issues
        unique_issues = self._deduplicate_issues(all_issues)
        
        # Update statistics
        for issue in unique_issues:
            self.stats['by_severity'][issue.severity.value] += 1
            self.stats['by_category'][issue.category.value] += 1
            if issue.method == 'llm':
                self.stats['llm_calls'] += 1
            else:
                self.stats['rule_based'] += 1
        
        print("Analysis completed.")
        print(f"Total issues found: {len(unique_issues)}")
        print()
        
        return {
            'issues': [issue.to_dict() for issue in unique_issues],
            'statistics': self.stats,
            'total_issues': len(unique_issues),
            'document_metadata': document.get('document_metadata', {})
        }
    
    def _deduplicate_issues(self, issues: List[StyleIssue]) -> List[StyleIssue]:
        """Remove duplicate issues, keeping highest confidence"""
        seen = {}
        
        for issue in issues:
            key = (
                issue.page_or_slide_index,
                issue.element_index,
                issue.rule_name,
                issue.description[:50]  # Partial match on description
            )
            
            if key not in seen or issue.confidence > seen[key].confidence:
                seen[key] = issue
        
        return list(seen.values())

# DOCUMENT CHECKER INTERFACE
def check_document(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Backward-compatible entry point for grammar/word-list only.
    Returns: list of issue dicts (no wrapper dict).
    """
    grammar_checker = GrammarChecker()
    word_checker = WordListChecker()

    issues: List[Dict[str, Any]] = []

    for page in document.get("pages", []):
        slide_idx = page.get("index", 0)
        for elem in page.get("elements", []):
            text = (elem.get("text") or "").strip()
            if not text or len(text) < 3:
                continue
            loc = elem.get("locator", {}) or {}
            elem_idx = loc.get("element_index", 0)

            # grammar rules
            for issue in grammar_checker.check(text, elem, slide_idx, elem_idx):
                issues.append(issue.to_dict())

            # word list rules
            for issue in word_checker.check(text, elem, slide_idx, elem_idx):
                issues.append(issue.to_dict())

    return issues

# CLI INTERFACE
def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced Grammar & Word-List Style Checker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python checker.py input.json
  python checker.py input.json --output results.json
  python checker.py input.json --no-llm
  python checker.py input.json --stats-only
        """
    )
    
    parser.add_argument('input', help='Input JSON file (normalized document)')
    parser.add_argument('-o', '--output', help='Output JSON file (default: input_issues.json)')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM enhancement')
    parser.add_argument('--stats-only', action='store_true', help='Show statistics only')
    
    args = parser.parse_args()
    
    # Load input
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            document = json.load(f)
    except Exception as e:
        print(f"Error loading input: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run checker
    checker = AgenticStyleChecker(use_llm=not args.no_llm)
    result = checker.analyze_document(document)
    
    # Save output
    if not args.stats_only:
        output_path = args.output or args.input.replace('.json', '_issues.json')
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Error saving output: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Print statistics
    print()
    print("=" * 70)
    print("ANALYSIS STATISTICS")
    print("=" * 70)
    print(f"Total Elements Analyzed:  {result['statistics']['total_elements']}")
    print(f"Total Issues Found:       {result['total_issues']}")
    print()
    print("By Method:")
    print(f"  Rule-Based:             {result['statistics']['rule_based']}")
    print(f"  LLM-Enhanced:           {result['statistics']['llm_calls']}")
    print()
    print("By Severity:")
    for sev, count in sorted(result['statistics']['by_severity'].items()):
        if count > 0:
            print(f"  {sev.capitalize():20} {count}")
    print()
    print("By Category:")
    for cat, count in sorted(result['statistics']['by_category'].items()):
        if count > 0:
            print(f"  {cat.capitalize():20} {count}")
    print("=" * 70)
    
    # Print top issues
    if not args.stats_only and result['issues']:
        print()
        print("TOP 5 ISSUES:")
        print("-" * 70)
        for i, issue in enumerate(result['issues'][:5], 1):
            print(f"{i}. [{issue['severity'].upper()}] {issue['rule_name']}")
            print(f"   Location: {issue['location']}")
            print(f"   {issue['description']}")
            print()

if __name__ == '__main__':
    main()