"""
Advanced Style Analyzer — Model-Calibrated (2025-10-15)

Two rules:
  1) positive_language  – document-calibrated valence + negation/hedging gates
  2) active_voice       – passive/hedging detection with minimal rewrites

Upgrades vs prior version:
- Transformer sentiment + per-document calibration (robust z-scores)
- Tone vector (valence, hedging, voice) to reduce false positives
- LLM returns 3 options (1a/1b/1c); we rank by valence, anti-hedge, anti-passive, and closeness
- spaCy NER-based preserve constraints (names/dates/ORG/etc.)
- Keeps original CLI and output schema; excludes micro-edits from counts
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import spacy

# SETTINGS / CLIENT
from ..config.settings import settings
from ..utils.llm_client import LLMClient

# Sentiment scorer
from .sentiment_scorer import SentimentScorer, calibrate_doc

# LOGGER
logger = logging.getLogger(__name__)
logger.setLevel(getattr(settings, "log_level", logging.INFO))

# CONFIG
INCLUDE_NOTES: bool = bool(getattr(settings, "analyzer_include_notes", False))
ENABLE_REWRITE: bool = bool(getattr(settings, "enable_rewrite", True))
USE_PROGRESS_BAR: bool = bool(getattr(settings, "debug", False))
MAX_BATCH_ITEMS: int = int(getattr(settings, "rewrite_batch_limit", 5000))

NEG_CONF_MIN: float = float(getattr(settings, "neg_conf_min", 0.15))  # low-confidence => neutral
NEG_Z_K: float = float(getattr(settings, "neg_z_k", 1.5))            # threshold k for mu - k*sigma

# Regex helpers for masking non-editable facts
NUM_PATTERN = re.compile(r"\b(?:\d{1,3}(?:[,\d]{0,3})*(?:\.\d+)?|\d+%|\$\d[\d,\.]*)\b")
DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b|\b\d{4}\b"
)
EMAIL_URL_PATTERN = re.compile(r"(?:(?:mailto:)?[\w\.-]+@[\w\.-]+\.\w+|https?://\S+)", re.IGNORECASE)

# Negation / hedging
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

# Passive voice quick patterns
PASSIVE_PATTERNS = [
    re.compile(r'\b(is|are|was|were|been|being)\s+\w+ed\b', re.IGNORECASE),
    re.compile(r'\b(is|are|was|were|been|being)\s+\w+en\b', re.IGNORECASE),
]

# Domain neutral words sometimes misread as negative (kept for extra safety)
_NEG_WHITELIST = re.compile(r'\b(exploit|governance|weakness(?:es)?|risk(?:s)?|threats?)\b', re.IGNORECASE)

TokenSeq = Union[spacy.tokens.Doc, spacy.tokens.Span, Iterable[spacy.tokens.Token]]


class AdvancedStyleAnalyzer:
    def __init__(self) -> None:
        # spaCy
        nlp = spacy.load("en_core_web_sm")
        keep = {"tok2vec", "tagger", "attribute_ruler", "parser", "lemmatizer", "ner"}
        nlp.disable_pipes(*[p for p in nlp.pipe_names if p not in keep])
        self.nlp = nlp

        # LLM
        self.llm = LLMClient()

        # Sentiment scorer (transformer)
        self.sent_scorer = SentimentScorer(getattr(settings, "sent_model_name", None))

    # -------------- LLM wrapper --------------
    def _llm_chat(self, system_msg: str, user_msg: str) -> Optional[str]:
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

    # -------------- Headings & sentences --------------
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

    def _is_sentence_like(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        if text.endswith(('.', '!', '?')):
            return True
        doc = self.nlp(text)
        return any(t.pos_ in {"VERB", "AUX"} for t in doc)

    def _is_quoted_block(self, text: str) -> bool:
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

    # -------------- Basic text utils --------------
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

    # -------------- Negation / passive / hedging --------------
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

    # -------------- Mask/unmask --------------
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

    # -------------- Constraints & prompts --------------
    def _extract_preserve_constraints_fast(self, texts: List[str]) -> List[Dict[str, Any]]:
        cons = []
        for text in texts:
            doc = self.nlp(text)
            keep = []
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "PRODUCT", "GPE", "DATE", "PERCENT", "MONEY", "CARDINAL", "NORP"}:
                    keep.append(ent.text)
            tech = re.findall(r'\b(?:AI|ML|API|HITRUST|SWOT|LLM|ECFR|RAG|BERT)\b', text)
            facts = list(dict.fromkeys(keep + tech))[:5]
            tl = text.lower().strip()
            if tl.startswith(("weakness", "threat", "risk")):
                context = "weakness"
            elif tl.startswith(("strength", "opportunit")):
                context = "strength"
            else:
                context = "neutral"
            cons.append({"preserve_facts": facts, "context": context, "key_constraint": ""})
        return cons

    @staticmethod
    def _create_positive_prompt(texts: List[str], constraints: List[Dict[str, Any]]) -> Tuple[str, str]:
        system_msg = (
            "You are an expert editor.\n"
            "RULES\n"
            "- Preserve all facts in each [PRESERVE].\n"
            "- Keep placeholders like [NUM_*], [DATE_*], [LINK_*].\n"
            "- Maintain meaning, scope, and certainty; do not flip weaknesses into strengths.\n"
            "- Prefer constructive, neutral phrasing. Finish sentences with proper punctuation.\n"
            "- For each line, return THREE alternatives labeled 1a/1b/1c ... N a/b/c. Keep similar length."
        )
        user_msg = "Rewrite these lines:\n\n"
        for i, (text, c) in enumerate(zip(texts, constraints), 1):
            pres = ", ".join(c.get("preserve_facts", [])[:5])
            ctx = c.get("context", "neutral")
            user_msg += f"{i}. {text}\n"
            user_msg += f"   [PRESERVE: {pres}]\n"
            user_msg += f"   [CONTEXT: {ctx}]\n\n"
        user_msg += "Return only the numbered alternatives as 1a/1b/1c etc."
        return system_msg, user_msg

    @staticmethod
    def _create_active_prompt(texts: List[str], constraints: List[Dict[str, Any]]) -> Tuple[str, str]:
        system_msg = (
            "You are an expert editor for active, direct writing.\n"
            "RULES:\n"
            "1) Preserve [PRESERVE].\n"
            "2) Convert passive to active; reduce hedging (might/could/probably/perhaps/maybe/sort of/kind of/may be).\n"
            "3) If context is weakness/threat, keep negative nature but clearer.\n"
            "4) Keep similar length and placeholders. End with proper punctuation.\n"
            "For each line, return THREE alternatives labeled 1a/1b/1c ...\n"
        )
        user_msg = "Convert to active voice:\n\n"
        for i, (text, c) in enumerate(zip(texts, constraints), 1):
            pres = ", ".join(c.get("preserve_facts", [])[:5])
            ctx = c.get("context", "neutral")
            user_msg += f"{i}. {text}\n"
            user_msg += f"   [PRESERVE: {pres}]\n"
            user_msg += f"   [CONTEXT: {ctx}]\n\n"
        user_msg += "Return only the numbered alternatives as 1a/1b/1c etc."
        return system_msg, user_msg

    # -------------- Validation --------------
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

    # -------------- Collection helpers --------------
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
        for (slide_idx, element_idx, element), doc in zip(meta, self.nlp.pipe(texts, batch_size=1000)):
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

    # -------------- Tone scoring helpers --------------
    def _tone_vector(self, text: str, sent_val: float) -> Tuple[float, float, float]:
        # valence in [-1,1] (approx), assertiveness = 1 - hedging, voice = passive prob proxy (0 or 1)
        hedging = 1.0 if self._has_hedging(text) else 0.0
        assertiveness = 1.0 - hedging
        voice = 1.0 if self._is_passive_text(text) else 0.0
        return sent_val, assertiveness, voice

    # -------------- Detection --------------
    def _detect_negative_language(self, all_sents) -> List[Dict[str, Any]]:
        """
        Document-calibrated negativity detection with:
        - transformer valence + robust z-score
        - absolute floor (neg_abs_floor)
        - minor-negativity guard (requires hedging or explicit negation)
        - whitelist bypass for domain terms
        """
        # Build candidate list for one-shot sentiment scoring
        cand_idx_map: List[int] = []
        cand_texts: List[str] = []
        for i, (slide_idx, element_idx, element, sent) in enumerate(all_sents):
            txt = sent.text.strip()
            if self._is_heading_like(txt) or not self._is_sentence_like(txt) or self._is_quoted_block(txt):
                continue
            cand_idx_map.append(i)
            cand_texts.append(txt)

        # Early exit
        if not cand_texts:
            logger.info("Positive/constructive issues: 0 (no candidates)")
            return []

        # Score once (batched) and calibrate per document
        batch = self.sent_scorer.score(cand_texts)
        mu, scale = calibrate_doc(batch.score)

        # Thresholds
        z_thresh = mu - NEG_Z_K * max(scale, 1e-6)
        abs_floor = float(getattr(settings, "neg_abs_floor", -0.15))  # configurable floor
        issues: List[Dict[str, Any]] = []

        for j, i in enumerate(cand_idx_map):
            slide_idx, element_idx, element, sent = all_sents[i]
            txt = cand_texts[j]
            p, e, etype, location = self._locator(element, slide_idx, element_idx)

            # Doc-calibrated negativity with confidence gate
            is_below_z = batch.score[j] < z_thresh
            is_below_abs = batch.score[j] < abs_floor
            is_neg_model = (is_below_z or is_below_abs) and (batch.conf[j] >= NEG_CONF_MIN)

            # Linguistic gates
            explicit_neg = self._negative_phrase_reason(txt, sent)
            has_hedge = self._has_hedging(txt)

            # Whitelist: if only model-negative and whitelist term appears without explicit negation, skip
            if is_neg_model and _NEG_WHITELIST.search(txt) and not explicit_neg:
                continue

            # Minor negativity (z-only, not under absolute floor) must co-occur with negation or hedging
            minor_neg = (not is_below_abs) and is_below_z
            if minor_neg and not (explicit_neg or has_hedge):
                continue

            if explicit_neg or is_neg_model:
                issues.append({
                    "rule_name": "positive_language",
                    "severity": "warning",
                    "category": "tone-issue",
                    "description": "Use constructive, neutral phrasing without changing facts.",
                    "location": location,
                    "found_text": txt,
                    "suggestion": None,
                    "page_or_slide_index": p - 1,
                    "element_index": e - 1,
                    "element_type": etype,
                    "score": batch.score[j],
                })

        logger.info(f"Positive/constructive issues: {len(issues)} (mu={mu:.3f}, sigma={scale:.3f}, z_th={z_thresh:.3f}, floor={abs_floor:.3f})")
        return issues


    def _detect_active_voice(self, sents) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for slide_idx, element_idx, element, sent in sents:
            txt = sent.text.strip()
            if self._is_heading_like(txt) or not self._is_sentence_like(txt) or self._is_quoted_block(txt):
                continue
            p, e, etype, location = self._locator(element, slide_idx, element_idx)
            if self._is_passive(sent) or self._is_mushy(sent):
                issues.append({
                    "rule_name": "active_voice",
                    "severity": "info",
                    "category": "tone-issue",
                    "description": "Use active voice to make writing direct and clear.",
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

    # -------------- LLM rewrite (3 candidates + ranking) --------------
    def _rank_candidate(self, original: str, candidate: str) -> float:
        # features: sentiment, anti-hedge, anti-passive, closeness
        s_val = self.sent_scorer.score([candidate]).score[0]
        anti_hedge = 0.0 if self._has_hedging(candidate) else 1.0
        anti_passive = 0.0 if self._is_passive_text(candidate) else 1.0
        sim = difflib.SequenceMatcher(None, original.lower(), candidate.lower()).ratio()
        # weights tuned lightly
        return (1.2 * s_val) + (0.4 * sim) + (0.6 * anti_hedge) + (0.4 * anti_passive)

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

        # Constraints include context awareness via NER
        constraints = self._extract_preserve_constraints_fast(masked_texts)
        for i, ctx in enumerate(contexts):
            if ctx and ctx != "neutral":
                constraints[i]["context"] = ctx

        # Build prompts
        if mode == "positive":
            system_msg, user_msg = self._create_positive_prompt(masked_texts, constraints)
            rule_name = "positive_language"
        else:
            system_msg, user_msg = self._create_active_prompt(masked_texts, constraints)
            rule_name = "active_voice"

        content = self._llm_chat(system_msg, user_msg)
        if not content:
            return [(None, "empty_response")] * len(items)

        # Parse 1a/1b/1c format (tolerant)
        grouped: Dict[int, List[str]] = {i + 1: [] for i in range(len(items))}
        for num, letter, textline in re.findall(r'^\s*(\d+)([a-c])[\.\)]\s*(.+)$', content, flags=re.MULTILINE):
            idx = int(num)
            if 1 <= idx <= len(items):
                grouped[idx].append(textline.strip().strip('"').strip("'"))

        # If model ignored a/b/c, fall back to standard 1..N lines
        if all(len(v) == 0 for v in grouped.values()):
            numbered = re.findall(r'^\s*(\d+)[\.\)]\s*(.+)$', content, flags=re.MULTILINE)
            for num_str, textline in numbered:
                idx = int(num_str)
                if 1 <= idx <= len(items):
                    grouped[idx].append(textline.strip().strip('"').strip("'"))

        results: List[Tuple[Optional[str], str]] = []
        for i in range(len(items)):
            cands = grouped.get(i + 1, [])
            if not cands:
                results.append((None, "llm_no_parse"))
                continue

            # unmask & finalize all candidates, then rank
            unmasked: List[str] = []
            for cand in cands:
                c = self._unmask(cand, masks[i])
                c = self._finalize_sentence(c)
                unmasked.append(c)

            # attach heading and validate
            best_cand = None
            best_score = -1e9
            for c in unmasked:
                sug_full = self._reattach_heading(headings[i], c)
                orig_full = self._reattach_heading(headings[i], originals[i])
                # quick pre-filter: avoid micro-edits (identical norm)
                if self._normalize_text(sug_full) == self._normalize_text(orig_full):
                    continue
                s = self._rank_candidate(orig_full, sug_full)
                if s > best_score:
                    best_score, best_cand = s, sug_full

            if not best_cand:
                results.append((None, "llm_invalid_all"))
                continue

            # final validation
            orig_full = self._reattach_heading(headings[i], originals[i])
            ok, reason = self._validate_suggestion_with_constraints(orig_full, best_cand, rule_name, constraints[i])
            if ok:
                results.append((best_cand, "llm_validated"))
            else:
                results.append((None, f"llm_invalid_{reason}"))

        return results

    # -------------- Suggestion assembly --------------
    def _generate_suggestions_for_all(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pos_queue: List[Tuple[int, str, str, Optional[str], Optional[float], Optional[str]]] = []
        act_queue: List[Tuple[int, str, str, Optional[str], Optional[float], Optional[str]]] = []

        failure_reasons: Dict[str, int] = {}
        def bump(reason: str):
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

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

                    # Conservative fallbacks
                    if batch[0][2] == "positive":
                        # small neutralizing substitutions only
                        t = body_only
                        replacements = [
                            (r"\s{2,}", " "),
                            (r"\bdo\s+not\s+have\b", "lack"),
                            (r"\bdoes\s+not\s+have\b", "lacks"),
                            (r"\bnot\s+able\s+to\b", "unable to"),
                            (r"\bno\s+significant\b", "limited"),
                        ]
                        for pat, rep in replacements:
                            t2 = re.sub(pat, rep, t, flags=re.IGNORECASE)
                            t = t2
                        sug_body = t
                        rule_name = "positive_language"
                    else:
                        # reduce hedging; trivial passive fixes
                        t = body_only
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
                        for pat, rep in hedges:
                            t = re.sub(pat, rep, t, flags=re.IGNORECASE)
                        passive_fixes = [
                            (r"\b(is|are|was|were)\s+created\b", "creates"),
                            (r"\b(is|are|was|were)\s+developed\b", "develops"),
                            (r"\b(is|are|was|were)\s+built\b", "builds"),
                            (r"\b(is|are|was|were)\s+used\b", "uses"),
                            (r"\b(is|are|was|were)\s+needed\b", "needs"),
                            (r"\b(is|are|was|were)\s+applied\b", "applies"),
                            (r"\bbeing\s+(\w+ed|made|done)\b", r"\1"),
                        ]
                        for pat, rep in passive_fixes:
                            t = re.sub(pat, rep, t, flags=re.IGNORECASE)
                        sug_body = t
                        rule_name = "active_voice"

                    sug_full = self._reattach_heading(heading, self._finalize_sentence(sug_body))
                    original_full = self._reattach_heading(heading, body_only)
                    ok, reason = self._validate_suggestion(original_full, sug_full, rule_name)

                    # exclude micro-edits
                    micro = (self._normalize_text(sug_full) == self._normalize_text(original_full))
                    if ok and not micro:
                        issues[global_idx]["suggestion"] = sug_full
                        issues[global_idx]["method"] = "fallback"
                        stats["fallback"] += 1
                    else:
                        issues[global_idx]["method"] = "no_improvement"
                        stats["failed"] += 1
                        bump(f"fallback_{reason}{'_micro' if micro else ''}")

                k += MAX_BATCH_ITEMS

            if failure_reasons:
                logger.info("Rewrite rejection reasons: " + ", ".join(f"{k}={v}" for k, v in failure_reasons.items()))
            return stats

        _ = process_queue(pos_queue)
        _ = process_queue(act_queue)
        logger.info("Rewrite complete")
        return issues

    # -------------- Public API --------------
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

        # Keep only issues with a real, different suggestion (exclude micro-edits)
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


# -------------- CLI --------------
if __name__ == "__main__":
    logging.basicConfig(level=getattr(settings, "log_level", logging.INFO), format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run style checks on normalized JSON.")
    parser.add_argument("input_json", help="Path to normalized JSON file")
    parser.add_argument("--no-rewrite", action="store_true", help="Disable LLM rewriting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.no_rewrite:
        ENABLE_REWRITE = False  # noqa: N816

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
