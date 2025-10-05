"""
Advanced Style Analyzer (single-file, class-based) - IMPROVED VERSION

Rules (exactly two):
1) positive_language  -> detects explicit negation + negative sentiment; single description
2) active_voice       -> detects passive voice + mushy/hedging; single description

Key Improvements:
- Specialized LLM prompts per rule type
- Validation layer for LLM suggestions
- Better fallback strategies
- Confidence scoring
- Retry logic for failed suggestions
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from time import perf_counter 
import requests
import spacy
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..config.settings import settings

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(getattr(settings, "log_level", logging.INFO))
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# Constants
VADER_NEG_THRESHOLD: float = getattr(settings, "vader_neg_threshold", -0.05)
INCLUDE_NOTES: bool = getattr(settings, "analyzer_include_notes", False)
ENABLE_REWRITE: bool = getattr(settings, "enable_rewrite", True)
USE_PROGRESS_BAR: bool = bool(getattr(settings, "debug", False))
MAX_BATCH_ITEMS: int = int(getattr(settings, "rewrite_batch_limit", 50))

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = getattr(settings, "huggingface_model", "google/gemma-2-2b-it:nebius")

# Regex helpers (masking for rewrite fidelity)
NUM_PATTERN = re.compile(r"\b(?:\d{1,3}(?:[,\d]{0,3})*(?:\.\d+)?|\d+%|\$\d[\d,\.]*)\b")
DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b|\b\d{4}\b"
)
EMAIL_URL_PATTERN = re.compile(r"(?:(?:mailto:)?[\w\.-]+@[\w\.-]+\.\w+|https?://\S+)", re.IGNORECASE)

NEGATOR_WORDS = r"(?:not|no|never|hardly|barely|scarcely|rarely|seldom|without|lack|lacks|lacking)"
NEGATOR_CONTRACTIONS = (
    r"(?:can't|cannot|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|shouldn't|wouldn't|"
    r"couldn't|mustn't|ain't)"
)
ANY_NEGATION = re.compile(rf"\b({NEGATOR_WORDS}|{NEGATOR_CONTRACTIONS})\b", flags=re.IGNORECASE)
NOT_PLUS_QUALITY = re.compile(r"\bnot\s+(?:\w+\s+)?\w+\b", flags=re.IGNORECASE)
AUX_NOT_VERB = re.compile(r"\b(?:do|does|did)\s+not\s+\w+\b", flags=re.IGNORECASE)
BE_NOT_QUAL = re.compile(r"\b(?:is|are|was|were|am|be|been|being)\s+not\s+\w+\b", flags=re.IGNORECASE)
NO_NOUN = re.compile(r"\bno\s+\w+\b", flags=re.IGNORECASE)

_HEADING_RE = re.compile(r'^\s*([A-Z][A-Za-z0-9 /&\-\(\)]+:)\s*(.*)$')

# Passive voice patterns for validation
PASSIVE_PATTERNS = [
    re.compile(r'\b(is|are|was|were|been|being)\s+\w+ed\b', re.IGNORECASE),
    re.compile(r'\b(is|are|was|were|been|being)\s+\w+en\b', re.IGNORECASE),
]

TokenSeq = Union[spacy.tokens.Doc, spacy.tokens.Span, Iterable[spacy.tokens.Token]]


class AdvancedStyleAnalyzer:
    """
    Two-rule style analyzer with guaranteed suggestions, validation, and improved prompts.
    """

    def __init__(self) -> None:
        self.analyzer = SentimentIntensityAnalyzer()
        _nlp = spacy.load("en_core_web_sm")
        keep = {"tok2vec", "tagger", "attribute_ruler", "parser", "lemmatizer"}
        _nlp.disable_pipes(*[p for p in _nlp.pipe_names if p not in keep])
        self.nlp = _nlp

    def _split_heading(self, s: str) -> Tuple[Optional[str], str]:
        """Extract heading if present."""
        m = _HEADING_RE.match((s or "").strip())
        if m:
            return m.group(1), m.group(2)
        return None, s

    def _reattach_heading(self, heading: Optional[str], body: str) -> str:
        """Prepend heading back to body."""
        body = (body or "").strip()
        if heading:
            if not body:
                return heading.strip()
            return f"{heading} {body}".strip()
        return body

    # ============ VALIDATION METHODS ============
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    def _has_negation(self, text: str) -> bool:
        """Check if text contains negation patterns."""
        return bool(ANY_NEGATION.search(text))

    def _is_passive_text(self, text: str) -> bool:
        """Check if text contains passive voice patterns."""
        return any(pattern.search(text) for pattern in PASSIVE_PATTERNS)

    def _has_hedging(self, text: str) -> bool:
        """Check for hedging/mushy language."""
        hedges = [r'\bmight be\b', r'\bcould be\b', r'\bwould be\b', 
                  r'\bprobably\b', r'\bperhaps\b', r'\bmaybe\b', 
                  r'\bsort of\b', r'\bkind of\b']
        return any(re.search(h, text, re.IGNORECASE) for h in hedges)

    def _validate_suggestion(self, original: str, suggestion: str, rule_name: str) -> Tuple[bool, str]:
        """
        Validate that suggestion is actually better than original.
        Returns: (is_valid, reason)
        """
        if not suggestion or len(suggestion.strip()) < 3:
            return False, "empty_or_too_short"
        
        # Must actually differ (normalized comparison)
        if self._normalize_text(original) == self._normalize_text(suggestion):
            return False, "identical"
        
        # Length sanity check (shouldn't be way longer/shorter)
        if len(suggestion) > len(original) * 2.0:
            return False, "too_long"
        
        if len(suggestion) < len(original) * 0.3:
            return False, "too_short"
        
        # Rule-specific validation
        if rule_name == "positive_language":
            # Suggestion should have fewer/no negations
            if self._has_negation(suggestion) and not self._has_negation(original):
                return False, "introduced_negation"
            
        elif rule_name == "active_voice":
            # Suggestion should be less passive
            if self._is_passive_text(suggestion) and not self._is_passive_text(original):
                return False, "still_passive"
            
            # Should have less hedging
            if self._has_hedging(suggestion) and not self._has_hedging(original):
                return False, "still_mushy"
        
        return True, "valid"

    # IMPROVED LLM METHODS

    def _create_positive_prompt(self, texts: List[str]) -> Tuple[str, str]:
        """Create specialized prompt for positive language."""
        system_msg = """You are an expert editor specializing in positive, constructive communication.

TASK: Rewrite sentences to be positive and solution-oriented.

RULES:
1. Replace negative phrases with affirmative statements
2. Focus on what IS rather than what ISN'T
3. Turn problems into opportunities or solutions
4. Maintain technical accuracy and meaning
5. Keep similar length to original
6. Preserve placeholders like [NUM_*], [DATE_*], [LINK_*]
7. Preserve names like Amida, Pradeep, etc.

EXAMPLES:
- "We cannot achieve this without more resources" → "We can achieve this with additional resources"
- "The system is not user-friendly" → "The system needs usability improvements"
- "No significant progress was made" → "Progress is limited; we need additional support"

Return ONLY the rewritten sentences, numbered 1 through N, one per line."""

        user_msg = "Rewrite these to be positive:\n\n"
        for i, text in enumerate(texts, 1):
            user_msg += f"{i}. {text}\n"
        user_msg += "\nRewritten (numbered 1..N):"
        
        return system_msg, user_msg

    def _create_active_prompt(self, texts: List[str]) -> Tuple[str, str]:
        """Create specialized prompt for active voice."""
        system_msg = """You are an expert editor specializing in active, direct writing.

TASK: Convert sentences to active voice and remove hedging language.

RULES:
1. Identify the actor and place before the verb
2. Remove "to be" + past participle constructions
3. Remove hedging words: might, could, probably, perhaps, maybe, sort of, kind of
4. Make statements direct and confident
5. Keep similar length to original
6. Preserve placeholders like [NUM_*], [DATE_*], [LINK_*]
7. Preserve names like Amida, Pradeep, etc.

EXAMPLES:
- "The report was written by the team" → "The team wrote the report"
- "Improvements might be seen in Q3" → "Improvements will occur in Q3"
- "We are looking for engineers" → "We seek engineers" or "We hire engineers"

Return ONLY the rewritten sentences, numbered 1 through N, one per line."""

        user_msg = "Convert to active voice:\n\n"
        for i, text in enumerate(texts, 1):
            user_msg += f"{i}. {text}\n"
        user_msg += "\nRewritten (numbered 1..N):"
        
        return system_msg, user_msg

    def _mask_preserve(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Mask numbers, dates, emails, URLs for preservation."""
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

    def _unmask(self, text: str, repl: Dict[str, str]) -> str:
        """Restore masked values."""
        for k, v in repl.items():
            text = text.replace(k, v)
        return text

    def _hf_post(self, payload: Dict[str, Any], retries: int = 3, timeout: int = 60) -> Dict[str, Any]:
        """Post to HuggingFace with retries."""
        headers = {"Authorization": f"Bearer {settings.huggingface_api_key}", "Content-Type": "application/json"}
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(HF_ENDPOINT, headers=headers, json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                if e.response is not None and (e.response.status_code == 429 or 500 <= e.response.status_code < 600):
                    last_exc = e
                else:
                    raise
            except requests.RequestException as e:
                last_exc = e
            time.sleep(0.5 * (2 ** attempt))
        assert last_exc is not None
        logger.error(f"LLM call failed after {retries+1} attempts: {last_exc}")
        raise last_exc

    def _rewrite_batch_llm_improved(self, items: List[Tuple[int, str, str, Optional[str]]]) -> List[Tuple[Optional[str], str]]:
        """
        Improved LLM rewriting with specialized prompts and validation.
        
        items: [(global_issue_index, body_only, mode, heading_or_None), ...]
        Returns: [(suggestion_or_None, method_used), ...] in same order
        """
        if not ENABLE_REWRITE:
            return [(None, "disabled")] * len(items)
        
        if not getattr(settings, "validate_llm_config", None) or not settings.validate_llm_config():
            return [(None, "no_config")] * len(items)

        # Separate by mode
        mode = items[0][2] if items else "positive"
        
        # Mask and prepare inputs
        masked_texts = []
        masks = []
        originals = []
        headings = []
        
        for idx, body, _, heading in items:
            masked, mask_map = self._mask_preserve(body)
            masked_texts.append(masked)
            masks.append(mask_map)
            originals.append(body)
            headings.append(heading)

        # Create specialized prompt
        if mode == "positive":
            system_msg, user_msg = self._create_positive_prompt(masked_texts)
        else:
            system_msg, user_msg = self._create_active_prompt(masked_texts)

        payload = {
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.3,
            "max_tokens": min(3000, 150 * len(items)),
            "top_p": 0.9,
        }

        try:
            data = self._hf_post(payload)
            content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            
            if not content:
                return [(None, "empty_response")] * len(items)

            # Parse numbered responses
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            results = []
            
            for idx in range(len(items)):
                candidate = None
                
                # Find matching numbered line
                for ln in lines:
                    if ln.startswith(f"{idx+1}.") or ln.startswith(f"{idx+1})") or ln.startswith(f"{idx+1} "):
                        candidate = re.sub(rf"^{idx+1}[\.\)]\s*", "", ln).strip().strip('"').strip("'")
                        break
                
                if candidate:
                    # Unmask
                    candidate = self._unmask(candidate, masks[idx])
                    # Reattach heading
                    candidate = self._reattach_heading(headings[idx], candidate)
                    # Finalize
                    candidate = self._finalize_sentence(candidate)
                    
                    # Validate
                    original_full = self._reattach_heading(headings[idx], originals[idx])
                    is_valid, reason = self._validate_suggestion(original_full, candidate, 
                                                                  "positive_language" if mode == "positive" else "active_voice")
                    
                    if is_valid:
                        results.append((candidate, "llm_validated"))
                    else:
                        logger.debug(f"LLM suggestion {idx+1} rejected: {reason}")
                        results.append((None, f"llm_invalid_{reason}"))
                else:
                    results.append((None, "llm_no_parse"))
            
            return results

        except Exception as e:
            logger.warning(f"LLM batch rewrite failed: {type(e).__name__}: {e}")
            return [(None, "llm_exception")] * len(items)

    # ============ IMPROVED FALLBACK METHODS ============

    def _fallback_positive_improved(self, s: str) -> str:
        """Improved positive phrasing fallback."""
        t = s
        
        # Specific negation patterns → positive alternatives
        replacements = [
            (r"\bnot\s+able\s+to\b", "unable to"),
            (r"\bnot\s+good\b", "needs improvement"),
            (r"\bnot\s+bad\b", "acceptable"),
            (r"\bnot\s+clear\b", "unclear"),
            (r"\bnot\s+complete\b", "incomplete"),
            (r"\bnot\s+enough\b", "insufficient"),
            (r"\bnot\s+working\b", "non-functional"),
            (r"\bno\s+progress\b", "limited progress"),
            (r"\bno\s+significant\b", "limited"),
            (r"\bno\s+(\w+)\b", r"limited \1"),
            (r"\bcan't\b", "cannot"),
            (r"\bwon't\b", "will not"),
            (r"\bdon't\b", "do not"),
            (r"\bdoesn't\b", "does not"),
        ]
        
        for pattern, replacement in replacements:
            t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
        
        # If still has strong negation, try generic flip
        if self._has_negation(t):
            t = re.sub(r"\bnot\s+(\w+)\b", r"needs \1 improvement", t, flags=re.IGNORECASE, count=1)
        
        return t if t.strip() and t != s else s

    def _fallback_active_improved(self, s: str) -> str:
        """Improved active voice fallback."""
        t = s
        
        # Remove hedging first
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
            (r"\ba\s+few\b", "several"),
            (r"\bsome\s+", ""),
            (r"\blooking\s+for\b", "seek"),
        ]
        
        for pattern, replacement in hedges:
            t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
        
        # Simple passive → active (conservative)
        # Only fix clear patterns
        t = re.sub(r"\b(is|are|was|were)\s+designed\s+for\b", r"serves", t, flags=re.IGNORECASE)
        t = re.sub(r"\b(is|are|was|were)\s+created\s+by\b", r"created", t, flags=re.IGNORECASE)
        t = re.sub(r"\b(is|are|was|were)\s+developed\s+by\b", r"developed", t, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        t = re.sub(r'\s+', ' ', t).strip()
        
        return t if t.strip() and t != s else s

    # ============ DETECTION METHODS (unchanged) ============
    
    def analyze(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        """Run both rules and produce suggestions (LLM or fallback)."""
        all_sents = self._collect_sentences(input_json)
        if not all_sents:
            return {"ok": True, "counts": {"positive_language": 0, "active_voice": 0, "with_suggestions": 0, "total": 0}, "issues": []}

        neg_issues = self._detect_negative_language(all_sents)
        voice_issues = self._detect_active_voice(all_sents)
        issues = neg_issues + voice_issues

        if not issues:
            return {"ok": True, "counts": {"positive_language": 0, "active_voice": 0, "with_suggestions": 0, "total": 0}, "issues": []}

        pos_cnt = sum(1 for i in issues if i["rule_name"] == "positive_language")
        act_cnt = sum(1 for i in issues if i["rule_name"] == "active_voice")
        logger.info(f"Rewrite start: total={len(issues)} (positive={pos_cnt}, active={act_cnt}), batch_size={MAX_BATCH_ITEMS}")

        issues = self._generate_suggestions_for_all(issues)

        return {
            "ok": True,
            "counts": {
                "positive_language": sum(1 for i in issues if i["rule_name"] == "positive_language"),
                "active_voice": sum(1 for i in issues if i["rule_name"] == "active_voice"),
                "with_suggestions": sum(1 for i in issues if i.get("suggestion")),
                "total": len(issues),
            },
            "issues": issues,
        }

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
                texts.append(t)
                meta.append((slide_idx, element_idx, element))

        if not texts:
            return []

        sents: List[Tuple[int, int, Dict[str, Any], spacy.tokens.Span]] = []
        iterator = self.nlp.pipe(texts, batch_size=64)
        for (slide_idx, element_idx, element), doc in zip(meta, iterator):
            for sent in doc.sents:
                if sent.text.strip():
                    sents.append((slide_idx, element_idx, element, sent))
        logger.info(f"Collected {len(sents)} sentences from document")
        return sents

    def _locator(self, element: Dict[str, Any], slide_idx: int, element_idx: int) -> Tuple[int, int, str, str]:
        loc = element.get("locator") or {}
        etype = loc.get("element_type") or element.get("element_type") or "content"
        page = loc.get("page_or_slide_index") or loc.get("page_index") or (slide_idx + 1)
        elem = loc.get("element_index") or (element_idx + 1)
        return int(page), int(elem), str(etype), f"slide {page}, element {elem}"

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
        tl = sent.text.lower()
        patterns = [
            r'\ba few\b', r'\bsome\s+\w+', r'\bmight be\b', r'\bcould be\b', r'\bwould be\b',
            r'\bprobably\b', r'\bperhaps\b', r'\bmaybe\b', r'\bsort of\b', r'\bkind of\b', r'\blooking for\b'
        ]
        return any(re.search(p, tl) for p in patterns)

    def _detect_negative_language(self, sents) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        iterator = tqdm(sents, desc="Negative check", unit="sent") if USE_PROGRESS_BAR else sents
        for slide_idx, element_idx, element, sent in iterator:
            txt = sent.text.strip()
            p, e, etype, location = self._locator(element, slide_idx, element_idx)

            if self._negative_phrase_reason(txt, sent):
                issues.append({
                    "rule_name": "positive_language",
                    "severity": "warning",
                    "category": "tone-issue",
                    "description": "Use positive language to make communication clearer, more constructive, and solution‑oriented.",
                    "location": location,
                    "found_text": txt,
                    "suggestion": None,
                    "page_or_slide_index": p - 1,
                    "element_index": e - 1,
                    "element_type": etype,
                    "score": None,
                })
                continue

            vs = self.analyzer.polarity_scores(txt)
            if vs.get("compound", 0.0) <= VADER_NEG_THRESHOLD:
                issues.append({
                    "rule_name": "positive_language",
                    "severity": "warning",
                    "category": "tone-issue",
                    "description": "Use positive language to make communication clearer, more constructive, and solution‑oriented.",
                    "location": location,
                    "found_text": txt,
                    "suggestion": None,
                    "page_or_slide_index": p - 1,
                    "element_index": e - 1,
                    "element_type": etype,
                    "score": vs["compound"],
                })
        logger.info(f"Positive language issues: {len(issues)}")
        return issues

    def _detect_active_voice(self, sents) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        iterator = tqdm(sents, desc="Voice check", unit="sent") if USE_PROGRESS_BAR else sents
        for slide_idx, element_idx, element, sent in iterator:
            txt = sent.text.strip()
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
                })
        logger.info(f"Active voice issues: {len(issues)}")
        return issues

    # ============ IMPROVED SUGGESTION GENERATION ============

    def _generate_suggestions_for_all(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate suggestions with validation and improved fallbacks."""
        pos_queue: List[Tuple[int, str, str, Optional[str]]] = []
        act_queue: List[Tuple[int, str, str, Optional[str]]] = []

        for idx, issue in enumerate(issues):
            if issue.get("suggestion"):
                continue
            original = issue.get("found_text", "")
            if not original:
                continue
            heading, body = self._split_heading(original)
            if issue["rule_name"] == "positive_language":
                pos_queue.append((idx, body, "positive", heading))
            else:
                act_queue.append((idx, body, "active", heading))

        def process_queue_improved(queue: List[Tuple[int, str, str, Optional[str]]]) -> Dict[str, int]:
            """Process queue with improved LLM and fallback."""
            if not queue:
                return {"llm": 0, "fallback": 0, "failed": 0}
            
            stats = {"llm": 0, "fallback": 0, "failed": 0}
            k = 0
            
            while k < len(queue):
                batch = queue[k:k + MAX_BATCH_ITEMS]
                mode = batch[0][2]
                
                # Try LLM first
                llm_results = self._rewrite_batch_llm_improved(batch)
                
                for (global_idx, body_only, _, heading), (sug, method) in zip(batch, llm_results):
                    if sug and method == "llm_validated":
                        # LLM succeeded
                        issues[global_idx]["suggestion"] = sug
                        issues[global_idx]["method"] = "llm"
                        stats["llm"] += 1
                    else:
                        # LLM failed, use improved fallback
                        if mode == "positive":
                            sug_body = self._fallback_positive_improved(body_only)
                        else:
                            sug_body = self._fallback_active_improved(body_only)
                        
                        sug = self._reattach_heading(heading, sug_body)
                        sug = self._finalize_sentence(sug)
                        
                        # Validate fallback too
                        original_full = self._reattach_heading(heading, body_only)
                        is_valid, reason = self._validate_suggestion(
                            original_full, sug,
                            "positive_language" if mode == "positive" else "active_voice"
                        )
                        
                        if is_valid:
                            issues[global_idx]["suggestion"] = sug
                            issues[global_idx]["method"] = "fallback"
                            stats["fallback"] += 1
                        else:
                            # Even fallback failed validation, use as-is but mark it
                            issues[global_idx]["suggestion"] = sug
                            issues[global_idx]["method"] = f"fallback_unvalidated_{reason}"
                            stats["failed"] += 1
                            logger.debug(f"Fallback for issue {global_idx} failed validation: {reason}")
                
                k += MAX_BATCH_ITEMS
            
            return stats

        # Process both queues
        pos_stats = process_queue_improved(pos_queue)
        act_stats = process_queue_improved(act_queue)

        logger.info(f"Rewrite complete: POS(llm={pos_stats['llm']}, fallback={pos_stats['fallback']}, failed={pos_stats['failed']}) "
                   f"ACT(llm={act_stats['llm']}, fallback={act_stats['fallback']}, failed={act_stats['failed']})")
        
        return issues

    def _finalize_sentence(self, s: str) -> str:
        """Finalize sentence punctuation."""
        s = s.strip().strip('"').strip("'")
        if not s or s.endswith(":"):
            return s
        if s[-1] not in ".!?":
            s += "."
        return s


# CLI (unchanged)
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="advance_style_analyzer",
        description="Run two-rule style checks (positive_language & active_voice) on normalized JSON."
    )
    parser.add_argument("input_json", help="Path to normalized JSON file")
    parser.add_argument("--no-rewrite", action="store_true", help="Disable LLM rewriting (fallback still applies)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.no_rewrite:
        globals()["ENABLE_REWRITE"] = False

    in_path = Path(args.input_json)
    if not in_path.exists():
        sys.stderr.write(f"ERROR: File not found: {in_path}\n")
        sys.exit(1)

    try:
        logger.info(f"Loading input file: {in_path}")
        with in_path.open("r", encoding="utf-8") as f:
            normalized = json.load(f)

        result = AdvancedStyleAnalyzer().analyze(normalized)
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
        sys.stdout.write("\n")
        sys.stdout.flush()

        total = result["counts"]["total"]
        with_sug = result["counts"]["with_suggestions"]
        logger.info(f"Total issues: {total}; with suggestions: {with_sug}")
        sys.exit(0 if result["ok"] else 1)

    except json.JSONDecodeError as e:
        sys.stderr.write(f"ERROR: Invalid JSON: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"ERROR: {type(e).__name__}: {e}\n")
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)