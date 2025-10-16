"""
Advanced Style Analyzer - REFINED VERSION (2025-10-06)

Two rules:
  1) positive_language  – n-gram + VADER-based negativity with safety rails
  2) active_voice       – passive/hedging detection and rewrites

Key Improvements in this version:
- Robust heading detection (with/without colon), incl. SWOT markers → skipped
- "Sentence-like" gating (avoid noun-only bullets)
- VADER hard threshold & neutral-word whitelist to reduce false positives
- Skip positive rewrites for SWOT/weakness headings
- Stronger fallbacks that avoid meaning flips
- LLM batch prompting with masking, constraint extraction, and VALIDATION
- Tolerant LLM output parsing (numbered OR plain lines)
- Accurate counts: do not count "no_improvement" as a suggestion
- Detailed rejection-reason logging for LLM outputs

CLI:
    python -m backend.analyzers.advance_style_analyzer <normalized.json> [--no-rewrite] [-v]
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import spacy
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..config.settings import settings  
from ..utils.llm_client import LLMClient
from .protection_layer import ProtectionLayer, LLMConfigError

# LOGGER
logger = logging.getLogger(__name__)
logger.setLevel(getattr(settings, "log_level", logging.INFO))

# CONFIG
INCLUDE_NOTES: bool = bool(getattr(settings, "analyzer_include_notes", False))
ENABLE_REWRITE: bool = bool(getattr(settings, "enable_rewrite", True))
USE_PROGRESS_BAR: bool = bool(getattr(settings, "debug", False))
MAX_BATCH_ITEMS: int = int(getattr(settings, "rewrite_batch_limit", 5000))

VADER_NEG_THRESHOLD: float = float(getattr(settings, "vader_neg_threshold", -0.05))
# Harder sentiment threshold for *purely sentiment* flags (<= baseline).
VADER_HARD_NEG: float = -0.40 if VADER_NEG_THRESHOLD > -0.40 else VADER_NEG_THRESHOLD

# Regex helpers for masking non-editable facts
NUM_PATTERN = re.compile(r"\b(?:\d{1,3}(?:[,\d]{0,3})*(?:\.\d+)?|\d+%|\$\d[\d,\.]*)\b")
DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b|\b\d{4}\b"
)
EMAIL_URL_PATTERN = re.compile(r"(?:(?:mailto:)?[\w\.-]+@[\w\.-]+\.\w+|https?://\S+)", re.IGNORECASE)

# Negation patterns
NEGATOR_WORDS = r"(?:not|no|never|hardly|barely|scarcely|rarely|seldom|without|lack|lacks|lacking)"
NEGATOR_CONTRACTIONS = (
    r"(?:can't|cannot|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|"
    r"shouldn't|wouldn't|couldn't|mustn't|ain't)"
)
ANY_NEGATION = re.compile(rf"\b({NEGATOR_WORDS}|{NEGATOR_CONTRACTIONS})\b", flags=re.IGNORECASE)
NOT_PLUS_QUALITY = re.compile(r"\bnot\s+(?:\w+\s+)?\w+\b", flags=re.IGNORECASE)
AUX_NOT_VERB = re.compile(r"\b(?:do|does|did)\s+not\s+\w+\b", flags=re.IGNORECASE)
BE_NOT_QUAL = re.compile(r"\b(?:is|are|was|were|am|be|been|being)\s+not\s+\w+\b", flags=re.IGNORECASE)
NO_NOUN = re.compile(r"\bno\s+\w+\b", flags=re.IGNORECASE)

# Heading detection
_HEADING_ONLY_RE = re.compile(r'^\s*([A-Z][A-Za-z0-9 /&\-\(\)]+:)\s*$')
_HEADING_RE = re.compile(r'^\s*([A-Z][A-Za-z0-9 /&\-\(\)]+:)\s*(.+)$')
_SWOT_MARKERS = re.compile(
    r'^\s*(key\s+)?(strengths?|weakness(?:es)?|opportunit(?:y|ies)|threats?(?:/risks?)?)\b[:\-]?\s*$',
    re.IGNORECASE
)

# Passive voice quick patterns (used in additional heuristic)
PASSIVE_PATTERNS = [
    re.compile(r'\b(is|are|was|were|been|being)\s+\w+ed\b', re.IGNORECASE),
    re.compile(r'\b(is|are|was|were|been|being)\s+\w+en\b', re.IGNORECASE),
]

# Domain neutral words that VADER can misread as negative
_NEG_WHITELIST = re.compile(r'\b(exploit|governance|weakness(?:es)?|risk(?:s)?|threats?)\b', re.IGNORECASE)

TokenSeq = Union[spacy.tokens.Doc, spacy.tokens.Span, Iterable[spacy.tokens.Token]]


class AdvancedStyleAnalyzer:
    """
    Two-rule style analyzer with improved prompts, validation, and robust parsing.
    """

    def __init__(
            self,
            *,
            use_llm: bool = True,
            protection_data: dict | None = None,
            protection_cache_path: Path | None = None,
        ):        
        self.analyzer = SentimentIntensityAnalyzer()
        nlp = spacy.load("en_core_web_sm")
        keep = {"tok2vec", "tagger", "attribute_ruler", "parser", "lemmatizer"}
        nlp.disable_pipes(*[p for p in nlp.pipe_names if p not in keep])
        self.nlp = nlp

        # Single LLM entrypoint
        self.llm = LLMClient()

        self.protection_layer = ProtectionLayer(llm_client=getattr(self, "llm", None))
        if protection_data:
            self.protection_layer.set_protection_data(protection_data)
        elif protection_cache_path and protection_cache_path.exists():
            self.protection_layer.load(protection_cache_path)

        # Make a fast local alias to avoid attribute lookups in tight loops
        self._prot = self.protection_layer

    # LLM wrapper
    def _llm_chat(self, system_msg: str, user_msg: str) -> Optional[str]:
        """Route all LLM calls through LLMClient; return text or None."""
        try:
            return self.llm.chat(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
            )
        except Exception as e:
            logger.warning("LLM chat failed: %s", e)
            return None

    # HEADING HELPERS
    def _is_heading_only(self, s: str) -> bool:
        return bool(_HEADING_ONLY_RE.match((s or "").strip()))

    def _looks_like_heading_no_colon(self, s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        if _SWOT_MARKERS.match(s):
            return True
        words = s.split()
        if len(words) > 4:
            return False
        doc = self.nlp(s)
        has_verb = any(t.pos_ in {"VERB", "AUX"} for t in doc)
        mostly_nouns = all(t.pos_ in {"NOUN", "PROPN", "ADJ", "PUNCT", "CCONJ", "ADP"} for t in doc)
        is_titleish = (s == s.title()) or s.isupper()
        return (not has_verb) and mostly_nouns and is_titleish

    def _is_heading_like(self, s: str) -> bool:
        return self._is_heading_only(s) or self._looks_like_heading_no_colon(s)

    def _split_heading(self, s: str) -> Tuple[Optional[str], str]:
        s = (s or "").strip()
        if self._is_heading_only(s):
            return s, ""
        m = _HEADING_RE.match(s)
        if m:
            return m.group(1), m.group(2)
        return None, s

    def _reattach_heading(self, heading: Optional[str], body: str) -> str:
        body = (body or "").strip()
        if heading:
            if not body:
                return heading.strip()
            return f"{heading} {body}".strip()
        return body
    
    # PROTECTED SPANS
    def _is_protected_span(self, text: str, start: int, end: int) -> bool:
        return self._prot.is_protected(text[start:end])
    
    def _collect_global_protect(self, texts: List[str], max_per_category: int = 50, hard_cap: int = 150) -> List[str]:
        """
        Return a short, deduped list of protected tokens that actually occur
        in the batch's texts. Longest tokens first to reduce partial overlaps.
        """
        data = self._prot.data or {}
        found: set[str] = set()
        for _, items in data.items():
            for tok in items[:max_per_category]:
                if not tok:
                    continue
                for t in texts:
                    if tok in t:
                        found.add(tok)
                        break
        return sorted(found, key=len, reverse=True)[:hard_cap]

    # SENTENCE / QUOTE FILTERS
    def _is_sentence_like(self, text: str) -> bool:
        """Prefer content with a verb or closing punctuation; avoids noun-only bullets."""
        text = (text or "").strip()
        if not text:
            return False
        if text.endswith(('.', '!', '?')):
            return True
        doc = self.nlp(text)
        return any(t.pos_ in {"VERB", "AUX"} for t in doc)

    def _is_quoted_block(self, text: str) -> bool:
        """Detect direct quotes / epigraphs we should not rewrite."""
        t = (text or "").strip()
        if not t:
            return False
        if (t.startswith(("“", '"', "‘", "'")) and t.endswith(("”", '"', "’", "'"))):
            return True
        if t.startswith(("“", '"')) or t.endswith(("”", '"')):
            return True
        if "…”" in t or '"…"' in t:
            return True
        return False

    # NORMALIZATION
    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r'[^\w\s]', '', (text or '').lower()).strip()

    @staticmethod
    def _finalize_sentence(s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip().strip('"').strip("'")
        if not s or s.endswith(":"):
            return s
        if s[-1] not in ".!?":
            s += "."
        return s

    # NEGATION / PASSIVE / HEDGING
    def _has_negation(self, text: str) -> bool:
        return bool(ANY_NEGATION.search(text or ""))

    @staticmethod
    def _is_passive_text(text: str) -> bool:
        return any(p.search(text or "") for p in PASSIVE_PATTERNS)

    @staticmethod
    def _has_hedging(text: str) -> bool:
        hedges = [
            r'\bmight be\b', r'\bcould be\b', r'\bwould be\b', r'\bshould be\b',
            r'\bmay be\b', r'\bprobably\b', r'\bperhaps\b', r'\bmaybe\b',
            r'\bsort of\b', r'\bkind of\b', r'\blooking for\b', r'\blooking to\b'
        ]
        tl = (text or "").lower()
        return any(re.search(h, tl) for h in hedges)

    def _any_not_plus_adjverb(self, doc_like: TokenSeq) -> bool:
        tokens = list(doc_like)
        for i, tok in enumerate(tokens):
            if getattr(tok, "lower_", "").lower() == "not":
                lookahead = tokens[i + 1:i + 3]
                if any(getattr(t, "pos_", "") in {"ADJ", "VERB", "AUX"} for t in lookahead):
                    return True
        return False

    def _negative_phrase_reason(self, text: str, doc: Optional[TokenSeq] = None) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        if ANY_NEGATION.search(text):
            if NOT_PLUS_QUALITY.search(text) or AUX_NOT_VERB.search(text) or BE_NOT_QUAL.search(text) or NO_NOUN.search(text):
                return True
            doc = doc or self.nlp(text)
            return self._any_not_plus_adjverb(doc)
        doc = doc or self.nlp(text)
        return self._any_not_plus_adjverb(doc)

    def _is_passive(self, sent: spacy.tokens.Span) -> bool:
        toks = list(sent)
        for i, tok in enumerate(toks):
            if tok.lemma_ == "be" and tok.pos_ == "AUX":
                for j in range(i + 1, min(i + 4, len(toks))):
                    nxt = toks[j]
                    if nxt.pos_ == "ADV":
                        continue
                    if nxt.tag_ == "VBN" and nxt.dep_ in {"ROOT", "ccomp", "xcomp", "advcl"}:
                        return True
                    if nxt.pos_ in {"VERB", "AUX"}:
                        break
        return False

    def _is_mushy(self, sent: spacy.tokens.Span) -> bool:
        return self._has_hedging(sent.text)

    # MASK / UNMASK
    def _mask_preserve(self, text: str) -> Tuple[str, Dict[str, str]]:
        repl: Dict[str, str] = {}

        def mask(pattern: re.Pattern, tag: str, s: str) -> str:
            def repl_fn(m: re.Match) -> str:
                key = f"[{tag}_{len(repl)}]"
                repl[key] = m.group(0)
                return key
            return pattern.sub(repl_fn, s)

        masked = mask(NUM_PATTERN, "NUM", text)
        masked = mask(DATE_PATTERN, "DATE", masked)
        masked = mask(EMAIL_URL_PATTERN, "LINK", masked)
        return masked, repl

    @staticmethod
    def _unmask(text: str, repl: Dict[str, str]) -> str:
        for k, v in repl.items():
            text = text.replace(k, v)
        return text

    # CONSTRAINTS & PROMPTS
    def _extract_preserve_constraints_fast(self, texts: List[str]) -> List[Dict[str, Any]]:
        constraints = []
        for text in texts:
            preserve_facts: List[str] = []
            context = "neutral"
            key_constraint = ""

            # Names (very rough)
            names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            preserve_facts.extend(names[:3])

            # Numbers / percents
            numbers = re.findall(r'\b\d+%?\b', text)
            preserve_facts.extend(numbers[:2])

            tl = text.lower().strip()
            if tl.startswith(("weakness", "threat", "risk")):
                context = "weakness"
                key_constraint = "Preserve negative nature; do not flip to positive."
            elif tl.startswith(("strength", "opportunit")):
                context = "strength"
                key_constraint = "Preserve positive nature."
            elif any(neg in tl for neg in ["lack", "limited", "no ", "not ", "unable", "insufficient", "cannot"]):
                context = "weakness"
                key_constraint = "This describes a limitation; keep it clear."

            tech_terms = re.findall(r'\b(?:AI|ML|API|BPA|HITRUST|SWOT|IT|FWA|LLM|EPS)\b', text)
            preserve_facts.extend(tech_terms[:2])

            constraints.append({
                "preserve_facts": list(dict.fromkeys(preserve_facts))[:5],  # dedupe + cap
                "context": context,
                "key_constraint": key_constraint
            })
        return constraints

    @staticmethod
    def _create_positive_prompt(texts: List[str], constraints: List[Dict[str, Any]]) -> Tuple[str, str]:
        system_msg = """You are an expert editor for positive, constructive writing.

RULES
- Preserve all facts in each [PRESERVE] list.
- Do not change meaning, scope, or certainty unless the original already had it.
- Keep placeholders like [NUM_*], [DATE_*], [LINK_*] exactly.
- Maintain similar length and style; finish sentences with appropriate punctuation.
- For rule=positive_language: make language constructive/clear; do NOT flip weaknesses into strengths.
- For rule=active_voice: prefer active voice; do NOT strengthen modality (e.g., 'may' → 'is') or add claims.
- If no improvement is needed, still return a minimally improved line that differs from input (clarity/typo).
- Improve positivity without changing actual meaning (e.g., "not clear" → "unclear", "not sure" → "unsure").

GOOD REWRITES (minimal, preserve meaning)
- "We cannot achieve this without more resources" → "We can achieve this with additional resources"
- "The system is not user-friendly" → "The system needs improved usability"
- "No significant progress was made" → "Progress has been limited"

BAD REWRITES (too different, changed meaning)
- "exploit AI BPA for NC" → "Our focus on program impact will help clients" ❌
- "Our sales content is very technical" → "We will develop accessible solutions" ❌

Return ONLY the rewritten sentences, numbered 1..N, one per line."""
        user_msg = "Rewrite these lines:\n\n"
        for i, (text, c) in enumerate(zip(texts, constraints), 1):
            pres = ", ".join(c.get("preserve_facts", [])[:5])
            ctx = c.get("context", "neutral")
            key = c.get("key_constraint", "")
            user_msg += f"{i}. {text}\n"
            user_msg += f"   [PRESERVE: {pres}]\n"
            user_msg += f"   [CONTEXT: {ctx}]\n"
            if key:
                user_msg += f"   [KEY: {key}]\n"
            user_msg += "\n"
        user_msg += "Rewritten:"
        return system_msg, user_msg

    @staticmethod
    def _create_active_prompt(texts: List[str], constraints: List[Dict[str, Any]]) -> Tuple[str, str]:
        system_msg = """You are an expert editor for active, direct writing.

RULES:
1) Preserve facts in [PRESERVE].
2) Convert passive to active voice.
3) Remove hedging words (might, could, probably, perhaps, maybe, sort of, kind of, may be).
4) If context is "weakness" or "threat", keep negative nature but clearer.
5) Maintain similar length.
6) Keep placeholders: [NUM_*], [DATE_*], [LINK_*].

GOOD REWRITES (preserve meaning)
- "The report was written by the team" → "The team wrote the report"
- "Improvements might be seen in Q3" → "Improvements will occur in Q3"
- "We are looking for engineers" → "We seek engineers"
- "States contracts are using a pilot approach" → "States use contracts in a pilot approach"

BAD REWRITES (changed meaning)
- "We are looking for engineers" → "We will hire top talent in Q3" ❌
- "could be expanded" → "is expanded" ❌

Return ONLY the rewritten sentences, numbered 1..N, one per line."""
        user_msg = "Convert to active voice:\n\n"
        for i, (text, c) in enumerate(zip(texts, constraints), 1):
            pres = ", ".join(c.get("preserve_facts", [])[:5])
            ctx = c.get("context", "neutral")
            key = c.get("key_constraint", "")
            user_msg += f"{i}. {text}\n"
            user_msg += f"   [PRESERVE: {pres}]\n"
            user_msg += f"   [CONTEXT: {ctx}]\n"
            if key:
                user_msg += f"   [KEY: {key}]\n"
            user_msg += "\n"
        user_msg += "Rewritten:"
        return system_msg, user_msg

    # VALIDATION
    def _semantic_similarity_check(self, original: str, suggestion: str) -> bool:
        orig_words = set(re.findall(r'\b[a-z]{4,}\b', (original or '').lower()))
        sug_words = set(re.findall(r'\b[a-z]{4,}\b', (suggestion or '').lower()))
        if not orig_words or not sug_words:
            return True
        overlap = len(orig_words & sug_words) / max(1, len(orig_words))
        return overlap >= 0.2

    def _validate_suggestion(self, original: str, suggestion: str, rule_name: str) -> Tuple[bool, str]:
        if not suggestion or len(suggestion.strip()) < 3:
            return False, "empty_or_too_short"
        if self._normalize_text(original) == self._normalize_text(suggestion):
            return False, "identical"
        if len(suggestion) > len(original) * 2.5:
            return False, "too_long"
        if len(suggestion) < len(original) * 0.25:
            return False, "too_short"
        if not self._semantic_similarity_check(original, suggestion):
            return False, "unrelated_content"

        if rule_name == "positive_language":
            if self._has_negation(suggestion) and not self._has_negation(original):
                return False, "introduced_negation"
        elif rule_name == "active_voice":
            if self._is_passive_text(suggestion) and not self._is_passive_text(original):
                return False, "still_passive"
            if self._has_hedging(suggestion) and not self._has_hedging(original):
                return False, "still_mushy"
        return True, "valid"

    def _validate_suggestion_with_constraints(
        self, original: str, suggestion: str, rule_name: str, constraint: Dict[str, Any]
    ) -> Tuple[bool, str]:
        ok, reason = self._validate_suggestion(original, suggestion, rule_name)
        if not ok:
            return False, reason

        preserve_facts = constraint.get("preserve_facts", [])[:5]
        context = constraint.get("context", "neutral")
        s_low = (suggestion or "").lower()

        # Ensure key facts still present (rough check)
        missing = 0
        for fact in preserve_facts:
            kws = [w for w in fact.lower().split() if len(w) > 3]
            if kws and not any(w in s_low for w in kws):
                missing += 1
        if missing >= 2:
            return False, f"missing_facts_{missing}"

        if context in ["weakness", "threat"]:
            weakness_tokens = ["limited", "lack", "no ", "not ", "unable", "insufficient", "without", "constrained", "shortage", "gap"]
            strength_tokens = ["strong", "robust", "capable", "provide", "deliver", "achieve", "successful", "effective", "established", "has "]
            sug_weak = any(tok in s_low for tok in weakness_tokens)
            sug_str = sum(1 for tok in strength_tokens if tok in s_low)
            if sug_str >= 2 and not sug_weak:
                return False, "flipped_weakness_to_strength"

        return True, "valid"

    # FALLBACKS
    def _fallback_positive_improved(self, s: str, context: str = "neutral") -> str:
        """Conservative edits; avoid meaning flips, avoid introducing new negations."""
        t = s
        if context in ("weakness", "threat"):
            replacements = [
                (r"\bvey\b", "very"),  # typo example
                (r"\s{2,}", " "),
            ]
        else:
            replacements = [
                (r"\bcan't\b", "cannot"),
                (r"\bnot\s+able\s+to\b", "unable to"),
                (r"\bdo\s+not\s+have\b", "lack"),
                (r"\bdoes\s+not\s+have\b", "lacks"),
                (r"\bnot\s+enough\b", "insufficient"),
                (r"\bno\s+significant\b", "limited"),
                (r"\s{2,}", " "),
            ]

        for pattern, repl in replacements:
            t2 = re.sub(pattern, repl, t, flags=re.IGNORECASE)
            if t2 != t:
                t = t2

        return t if t.strip() and self._normalize_text(t) != self._normalize_text(s) else s

    def _fallback_active_improved(self, s: str) -> str:
        t = s

        # Remove hedging
        hedges = [
            (r"\bmight\s+be\b", "is"),
            (r"\bcould\s+be\b", "is"),
            (r"\bwould\s+be\b", "is"),
            (r"\bshould\s+be\b", "is"),
            (r"\bmay\s+be\b", "is"),
            (r"\bprobably\s+", ""),
            (r"\bperhaps\s+", ""),
            (r"\bmaybe\s+", ""),
            (r"\bsort\s+of\s+", ""),
            (r"\bkind\s+of\s+", ""),
            (r"\blooking\s+for\b", "seek"),
            (r"\blooking\s+to\b", "seeking to"),
        ]
        for pattern, repl in hedges:
            t = re.sub(pattern, repl, t, flags=re.IGNORECASE)

        # Straightforward passive fixes
        passive_fixes = [
            (r"\b(is|are|was|were)\s+created\b", "creates"),
            (r"\b(is|are|was|were)\s+developed\b", "develops"),
            (r"\b(is|are|was|were)\s+built\b", "builds"),
            (r"\b(is|are|was|were)\s+used\b", "uses"),
            (r"\b(is|are|was|were)\s+needed\b", "needs"),
            (r"\b(is|are|was|were)\s+applied\b", "applies"),
            (r"\bbeing\s+(\w+ed|made|done)\b", r"\1"),
        ]
        for pattern, repl in passive_fixes:
            t = re.sub(pattern, repl, t, flags=re.IGNORECASE)

        t = re.sub(r'\s+', ' ', t).strip()
        return t if t.strip() and self._normalize_text(t) != self._normalize_text(s) else s

    # CORE PIPELINE
    def _iter_elements(self, normalized: Dict[str, Any]) -> Iterable[Tuple[int, int, Dict[str, Any]]]:
        slides = normalized.get("slides", normalized.get("pages", []))
        for slide_idx, slide in enumerate(slides):
            elements = slide.get("elements", []) or []
            for element_idx, element in enumerate(elements):
                yield slide_idx, element_idx, element

            if INCLUDE_NOTES:
                notes_text = slide.get("notes")
                if isinstance(notes_text, str) and notes_text.strip():
                    yield slide_idx, len(elements), {
                        "text": notes_text,
                        "element_type": "notes",
                        "locator": {
                            "page_or_slide_index": slide.get("index", slide_idx + 1),
                            "element_index": (len(elements) + 1),
                            "element_type": "notes",
                        },
                    }

    def _collect_sentences(self, normalized: Dict[str, Any]) -> List[Tuple[int, int, Dict[str, Any], spacy.tokens.Span]]:
        triplets = list(self._iter_elements(normalized))
        texts, meta = [], []
        for slide_idx, element_idx, element in triplets:
            t = element.get("text", "")
            if isinstance(t, str) and t.strip():
                if not self._is_heading_like(t):
                    texts.append(t)
                    meta.append((slide_idx, element_idx, element))

        if not texts:
            logger.info("Collected 0 sentences from document")
            return []

        sents: List[Tuple[int, int, Dict[str, Any], spacy.tokens.Span]] = []
        iterator = self.nlp.pipe(texts, batch_size=64)
        for (slide_idx, element_idx, element), doc in zip(meta, iterator):
            for sent in doc.sents:
                if sent.text.strip():
                    sents.append((slide_idx, element_idx, element, sent))
        logger.info(f"Collected {len(sents)} sentences from document")
        return sents

    @staticmethod
    def _locator(element: Dict[str, Any], slide_idx: int, element_idx: int) -> Tuple[int, int, str, str]:
        loc = element.get("locator") or {}
        etype = loc.get("element_type") or element.get("element_type") or "content"
        page = loc.get("page_or_slide_index") or loc.get("page_index") or (slide_idx + 1)
        elem = loc.get("element_index") or (element_idx + 1)
        return int(page), int(elem), str(etype), f"slide {page}, element {elem}"

    def _detect_negative_language(self, sents) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        iterator = tqdm(sents, desc="Negative check", unit="sent") if USE_PROGRESS_BAR else sents

        for slide_idx, element_idx, element, sent in iterator:
            txt = sent.text.strip()
            if self._is_heading_like(txt):
                continue
            if not self._is_sentence_like(txt):
                continue
            if self._is_quoted_block(txt):
                continue

            p, e, etype, location = self._locator(element, slide_idx, element_idx)

            # Explicit negation patterns
            if self._negative_phrase_reason(txt, sent):
                # Ignore whitelisted neutral domain words if no negation was detected
                if _NEG_WHITELIST.search(txt) and not self._has_negation(txt):
                    continue

                issues.append({
                    "rule_name": "positive_language",
                    "severity": "warning",
                    "category": "tone-issue",
                    "description": "Use positive language to make communication clearer, more constructive, and solution-oriented.",
                    "location": location,
                    "found_text": txt,
                    "suggestion": None,
                    "page_or_slide_index": p - 1,
                    "element_index": e - 1,
                    "element_type": etype,
                    "score": None,
                })
                continue

            # VADER-only negativity (strict threshold & whitelist)
            vs = self.analyzer.polarity_scores(txt)
            comp = vs.get("compound", 0.0)
            if comp <= VADER_HARD_NEG:
                if _NEG_WHITELIST.search(txt) and not self._has_negation(txt):
                    continue
                issues.append({
                    "rule_name": "positive_language",
                    "severity": "warning",
                    "category": "tone-issue",
                    "description": "Use positive language to make communication clearer, more constructive, and solution-oriented.",
                    "location": location,
                    "found_text": txt,
                    "suggestion": None,
                    "page_or_slide_index": p - 1,
                    "element_index": e - 1,
                    "element_type": etype,
                    "score": comp,
                })

        logger.info(f"Positive language issues: {len(issues)}")
        return issues

    def _detect_active_voice(self, sents) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        iterator = tqdm(sents, desc="Voice check", unit="sent") if USE_PROGRESS_BAR else sents

        for slide_idx, element_idx, element, sent in iterator:
            txt = sent.text.strip()
            if self._is_heading_like(txt):
                continue
            if not self._is_sentence_like(txt):
                continue
            if self._is_quoted_block(txt):
                continue

            p, e, etype, location = self._locator(element, slide_idx, element_idx)
            if self._is_passive(sent) or self._is_mushy(sent):
                issues.append({
                    "rule_name": "active_voice",
                    "severity": "info",
                    "category": "tone-issue",
                    "description": "Use active voice to make writing direct, strong, and easy to understand.",
                    "location": location,
                    "found_text": txt,
                    "suggestion": None,
                    "page_or_slide_index": p - 1,
                    "element_index": e - 1,
                    "element_type": etype,
                    "score": None,
                })

        logger.info(f"Active voice issues: {len(issues)}")
        return issues

    # LLM REWRITE (BATCH)
    def _rewrite_batch_llm_improved(
        self,
        items: List[Tuple[int, str, str, Optional[str], Optional[float], Optional[str]]],
    ) -> List[Tuple[Optional[str], str]]:
        if not ENABLE_REWRITE:
            return [(None, "disabled")] * len(items)

        validator = getattr(settings, "validate_llm_config", None)
        if validator is not None and callable(validator) and not validator():
            logger.error("LLM config invalid — skipping rewrites.")
            return [(None, "no_config")] * len(items)

        if not items:
            return []

        mode = items[0][2]  # "positive" or "active"

        masked_texts: List[str] = []
        masks: List[Dict[str, str]] = []
        originals: List[str] = []
        headings: List[Optional[str]] = []
        contexts: List[str] = []

        for _, body, _, heading, _, context in items:
            masked, mask_map = self._mask_preserve(body)
            masked_texts.append(masked)
            masks.append(mask_map)
            originals.append(body)
            headings.append(heading)
            contexts.append(context or "neutral")

        # Constraints include context awareness
        constraints = self._extract_preserve_constraints_fast(masked_texts)
        for i, ctx in enumerate(contexts):
            if ctx and ctx != "neutral":
                constraints[i]["context"] = ctx

        # NEW: collect protection items that actually appear in this batch
        global_protect = self._collect_global_protect(masked_texts)  # uses self._prot.data

        # Build prompts
        if mode == "positive":
            system_msg, user_msg = self._create_positive_prompt(masked_texts, constraints)
            rule_name = "positive_language"
        else:
            system_msg, user_msg = self._create_active_prompt(masked_texts, constraints)
            rule_name = "active_voice"

        # NEW: minimally augment the prompt with protection items
        if global_protect:
            # Add a clear, single line directive the model will respect.
            guard = "DO NOT ALTER these protected items if they appear in a line: " + "; ".join(global_protect)
            # Strengthen the system rules a touch, without changing your function signatures:
            system_msg += "\n\nAdditional RULE:\n- " + guard
            # Also echo in the user message so it's close to the examples:
            user_msg = guard + "\n\n" + user_msg


        # Single call via LLMClient
        content = self._llm_chat(system_msg, user_msg)
        if not content:
            return [(None, "empty_response")] * len(items)

        # Tolerant parsing
        numbered = re.findall(r'^\s*(\d+)[\.\)]\s*(.+)$', content, flags=re.MULTILINE)
        candidates: List[Optional[str]] = [None] * len(items)
        if numbered:
            for num_str, textline in numbered:
                idx = int(num_str) - 1
                if 0 <= idx < len(candidates) and not candidates[idx]:
                    candidates[idx] = textline.strip().strip('"').strip("'")
        else:
            rough = [ln.strip().strip('"').strip("'") for ln in content.splitlines() if ln.strip()]
            candidates = (rough + [None] * len(items))[:len(items)]

        results: List[Tuple[Optional[str], str]] = []
        for i, cand in enumerate(candidates):
            if not cand:
                results.append((None, "llm_no_parse"))
                continue

            cand = self._unmask(cand, masks[i])
            cand = self._reattach_heading(headings[i], cand)
            cand = self._finalize_sentence(cand)

            full_original = self._reattach_heading(headings[i], originals[i])
            ok, reason = self._validate_suggestion_with_constraints(full_original, cand, rule_name, constraints[i])

            if ok:
                results.append((cand, "llm_validated"))
            else:
                results.append((None, f"llm_invalid_{reason}"))

        return results

    # SUGGESTION GENERATION
    def _generate_suggestions_for_all(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pos_queue: List[Tuple[int, str, str, Optional[str], Optional[float], Optional[str]]] = []
        act_queue: List[Tuple[int, str, str, Optional[str], Optional[float], Optional[str]]] = []

        # Track LLM rejection reasons for logging
        failure_reasons: Dict[str, int] = {}

        def bump(reason: str):
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        # Assemble queues
        for idx, issue in enumerate(issues):
            if issue.get("suggestion"):
                continue
            original = issue.get("found_text", "") or ""
            if not original.strip():
                continue
            heading, body = self._split_heading(original)
            if not body:
                continue

            # infer context from original
            tl = original.lower().strip()
            context = None
            if tl.startswith(("weakness", "threat", "risk")):
                context = "weakness"
            elif tl.startswith(("strength", "opportunit")):
                context = "strength"

            score = issue.get("score")
            if issue["rule_name"] == "positive_language":
                pos_queue.append((idx, body, "positive", heading, score, context))
            else:
                act_queue.append((idx, body, "active", heading, score, context))

        def process_queue(queue: List[Tuple[int, str, str, Optional[str], Optional[float], Optional[str]]]) -> Dict[str, int]:
            if not queue:
                return {"llm": 0, "fallback": 0, "failed": 0}

            stats = {"llm": 0, "fallback": 0, "failed": 0}
            k = 0
            while k < len(queue):
                batch = queue[k:k + MAX_BATCH_ITEMS]

                # 1) Try LLM
                llm_results = self._rewrite_batch_llm_improved(batch)

                for (global_idx, body_only, _, heading, _, context), (sug, method) in zip(batch, llm_results):
                    if sug and method == "llm_validated":
                        issues[global_idx]["suggestion"] = sug
                        issues[global_idx]["method"] = "llm"
                        stats["llm"] += 1
                        continue

                    if method and method.startswith("llm_invalid_"):
                        bump(method.replace("llm_invalid_", "reject_"))
                    elif method:
                        bump(method)

                    # 2) Fallback (conservative)
                    if batch[0][2] == "positive":
                        sug_body = self._fallback_positive_improved(body_only, context=context or "neutral")
                        rule_name = "positive_language"
                    else:
                        sug_body = self._fallback_active_improved(body_only)
                        rule_name = "active_voice"

                    sug_full = self._reattach_heading(heading, sug_body)
                    sug_full = self._finalize_sentence(sug_full)

                    original_full = self._reattach_heading(heading, body_only)
                    ok, reason = self._validate_suggestion(original_full, sug_full, rule_name)

                    if ok and self._normalize_text(sug_full) != self._normalize_text(original_full):
                        issues[global_idx]["suggestion"] = sug_full
                        issues[global_idx]["method"] = "fallback"
                        stats["fallback"] += 1
                    else:
                        issues[global_idx]["method"] = "no_improvement"
                        stats["failed"] += 1
                        bump(f"fallback_{reason}")

                k += MAX_BATCH_ITEMS

            if failure_reasons:
                logger.info("Rewrite rejection reasons: " + ", ".join(f"{k}={v}" for k, v in failure_reasons.items()))
            return stats

        _ = process_queue(pos_queue)
        _ = process_queue(act_queue)

        logger.info("Rewrite complete")
        return issues

    # PUBLIC API
    def analyze(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        all_sents = self._collect_sentences(input_json)
        if not all_sents:
            return {
                "ok": True,
                "counts": {"positive_language": 0, "active_voice": 0, "with_suggestions": 0, "total": 0},
                "issues": [],
            }

        neg_issues = self._detect_negative_language(all_sents)
        voice_issues = self._detect_active_voice(all_sents)
        issues = neg_issues + voice_issues

        if not issues:
            return {
                "ok": True,
                "counts": {"positive_language": 0, "active_voice": 0, "with_suggestions": 0, "total": 0},
                "issues": [],
            }

        logger.info(
            "Rewrite start: total=%d (positive=%d, active=%d), batch_size=%d",
            len(issues),
            sum(1 for i in issues if i["rule_name"] == "positive_language"),
            sum(1 for i in issues if i["rule_name"] == "active_voice"),
            MAX_BATCH_ITEMS,
        )

        issues = self._generate_suggestions_for_all(issues)

        # Keep only issues with a real, different suggestion
        def _is_real_suggestion(item: Dict[str, Any]) -> bool:
            sug = item.get("suggestion")
            if not sug:
                return False
            return self._normalize_text(sug) != self._normalize_text(item.get("found_text", ""))

        filtered = [i for i in issues if _is_real_suggestion(i)]

        return {
            "ok": True,
            "counts": {
                "positive_language": sum(1 for i in filtered if i["rule_name"] == "positive_language"),
                "active_voice": sum(1 for i in filtered if i["rule_name"] == "active_voice"),
                "with_suggestions": len(filtered),
                "total": len(filtered),
            },
            "issues": filtered,
        }


# CLI
if __name__ == "__main__":
    import sys, json, logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python -m backend.analyzers.advance_style_analyzer <normalized.json>\n")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    if not in_path.exists():
        sys.stderr.write(f"ERROR: File not found: {in_path}\n")
        sys.exit(1)

    try:
        logger.info(f"Loading input: {in_path}")
        with in_path.open("r", encoding="utf-8") as f:
            normalized = json.load(f)

        # Ensure doc_id and protection cache path
        meta = normalized.setdefault("metadata", {})
        if "doc_id" not in meta:
            stem = in_path.stem
            prefix = stem.split("_", 1)[0] if "_" in stem else None
            meta["doc_id"] = prefix if prefix and prefix.isdigit() else stem

        cache_path = Path(settings.output_dir) / f"{meta['doc_id']}_protection.json"

        analyzer = AdvancedStyleAnalyzer(protection_cache_path=cache_path)
        result = analyzer.analyze(normalized)

        print(json.dumps(result, ensure_ascii=False, indent=2))
        counts = result.get("counts", {})
        logger.info(f"Issues: {counts.get('total', 0)}, Suggestions: {counts.get('with_suggestions', 0)}")

    except Exception as e:
        sys.stderr.write(f"ERROR: {type(e).__name__}: {e}\n")
        sys.exit(1)
