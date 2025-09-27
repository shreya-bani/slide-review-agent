"""
Tone analyzer for checking active voice and positive language
according to Amida Style Guide requirements.

Improvements:
- Phrase-level “negative wording” via spaCy PhraseMatcher
- Negation scope + intensifier/downtoner weighting
- Context exceptions (solution-oriented sentences)
- Rule-driven map (extend in amida_style_rules.json under "negative_phrases")
- Keeps your existing VADER positivity and passive-voice logic
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import spacy
from spacy.matcher import PhraseMatcher
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ToneIssue:
    """Represents a tone-related issue"""
    issue_type: str  # 'passive_voice', 'negative_language', 'low_positivity'
    original_text: str
    suggested_fix: str
    confidence: float
    explanation: str
    element_id: str
    page_line_ref: str  # e.g., 'Slide 3' or 'Page 2'


class ToneAnalyzer:
    """Analyzes text for tone compliance following exact workflow specifications"""

    def __init__(self, rules_path: Optional[str] = None) -> None:

        # Scoring gates
        self.MIN_TOKENS_FOR_SENTIMENT = 5        # skip sentiment for super short strings
        self.MIN_CHARS_FOR_SENTIMENT = 25        # skip for very short titles/captions
        self.POS_THRESH_BODY = 0.65              # body threshold
        self.POS_THRESH_TITLE = 0.55             # title threshold (titles are short/neutral)
        self.ALLOWLIST_TITLES = {"agenda", "about us", "financials"}  # don’t “improve” these


        # spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            raise

        # Sentiment
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.positivity_threshold = 0.5

        # Passive voice regex helpers (kept from your version)
        self.passive_patterns = [
            r'\b(?:was|were|been|being|is|are|am)\s+\w*ed\b',
            r'\b(?:was|were|been|being|is|are|am)\s+\w*en\b',
            r'\bwas\s+given\b', r'\bwere\s+taken\b', r'\bbeen\s+made\b'
        ]

        # Negation & modifier lexicons
        self.negators = {"not", "no", "never", "cannot", "can't", "won't", "isn't", "aren't", "doesn't", "don't"}
        self.intensifiers = {"very": 1.2, "highly": 1.3, "extremely": 1.4, "significantly": 1.3}
        self.downtoners = {"slightly": 0.8, "somewhat": 0.85, "rather": 0.9, "fairly": 0.9}

        # Load rule-driven phrase map from JSON (optional) and merge with defaults
        self.negative_map = self._default_negative_map()
        rules_path = rules_path or os.getenv("AMIDA_RULES_PATH", "src/rules/amida_style_rules.json")
        self._merge_negative_map_from_rules(rules_path)

        # Build PhraseMatcher for multi-word negatives
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        multiword = [p for p in self.negative_map.keys() if " " in p]
        if multiword:
            self.matcher.add("NEGATIVE_MULTI", [self.nlp.make_doc(p) for p in multiword])

    # ----------------------------- Public API -----------------------------

    def analyze_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze list of document elements for tone issues.
        Steps:
          A. Detect passive voice
          B. Score positivity (VADER)
          C. Detect negative language (rule-driven, negation-aware)
          D. Mark low positivity for LLM rewriting when < threshold
        """
        all_issues: List[ToneIssue] = []
        element_analyses: Dict[str, Any] = {}
        overall_stats = {
            'total_elements': len(elements),
            'passive_voice_count': 0,
            'negative_language_count': 0,
            'low_positivity_count': 0,
            'avg_positivity_score': 0.0,
            'elements_needing_llm_rewrite': 0
        }

        total_positivity = 0.0

        for element in elements:
            element_text = element.get('text', '') or ''
            element_id = element.get('element_id', 'unknown')

            if not element_text.strip():
                continue

            page_line_ref = self._extract_page_line_ref(element_id)

            # A. Passive Voice
            passive_issues = self._detect_passive_voice(element_text, element_id, page_line_ref)
            all_issues.extend(passive_issues)
            overall_stats['passive_voice_count'] += len(passive_issues)

            # B. Positivity (VADER)
            positivity_analysis = self._analyze_positivity(element_text)
            positivity_score = positivity_analysis['normalized_score']
            total_positivity += positivity_score

            # C. Negative Language (improved)
            negative_issues = self._detect_negative_language_v2(element_text, element_id, page_line_ref)
            all_issues.extend(negative_issues)
            overall_stats['negative_language_count'] += len(negative_issues)

            # D. Low positivity
            if positivity_score < self.positivity_threshold:
                overall_stats['low_positivity_count'] += 1
                overall_stats['elements_needing_llm_rewrite'] += 1
                all_issues.append(ToneIssue(
                    issue_type="low_positivity",
                    original_text=element_text,
                    suggested_fix="Needs positive language enhancement",
                    confidence=0.9,
                    explanation=f"Positivity score ({positivity_score:.2f}) below threshold ({self.positivity_threshold})",
                    element_id=element_id,
                    page_line_ref=page_line_ref
                ))

            element_analyses[element_id] = {
                'positivity_score': positivity_score,
                'needs_llm_rewrite': positivity_score < self.positivity_threshold,
                'passive_voice_detected': len(passive_issues) > 0,
                'negative_language_detected': len(negative_issues) > 0,
                'issues_count': len(passive_issues) + len(negative_issues)
            }

        overall_stats['avg_positivity_score'] = (total_positivity / len(elements) if elements else 0.0)

        return {
            'issues': [asdict(issue) for issue in all_issues],
            'element_analyses': element_analyses,
            'overall_stats': overall_stats,
            'recommendations': []  # keep empty per your spec
        }

    # ----------------------------- Helpers -----------------------------

    def _extract_page_line_ref(self, element_id: str) -> str:
        parts = element_id.split('_')
        if len(parts) >= 2:
            if 'page' in parts[0]:
                return f"Page {parts[1]}"
            if 'slide' in parts[0]:
                return f"Slide {parts[1]}"
        return element_id

    # Passive voice using spaCy
    def _detect_passive_voice(self, text: str, element_id: str, page_line_ref: str) -> List[ToneIssue]:
        issues: List[ToneIssue] = []
        doc = self.nlp(text)
        for sent in doc.sents:
            if self._is_passive_sentence(sent):
                # We intentionally do NOT do a regex rewrite here.
                # Let the LLM generate the active-voice version downstream.
                issues.append(ToneIssue(
                    issue_type="passive_voice",
                    original_text=sent.text.strip(),
                    suggested_fix="Needs LLM rewrite",
                    confidence=0.8,
                    explanation="Convert passive voice to active voice for more direct communication",
                    element_id=element_id,
                    page_line_ref=page_line_ref
                ))
        return issues


    def _is_passive_sentence(self, sentence) -> bool:
        has_be_aux = False
        has_vbn = False
        for token in sentence:
            if token.lemma_ == "be" and token.pos_ in {"AUX", "VERB"}:
                has_be_aux = True
            if token.tag_ == "VBN" and has_be_aux:
                has_vbn = True
                break
        # Optional “by”-phrase check – strong indicator but not required
        return has_be_aux and has_vbn

    def _suggest_active_voice_conversion(self, sentence: str) -> str:
        patterns = [
            (r'(.+)\s+was\s+(\w+ed)\s+by\s+(.+)', r'\3 \2 \1'),
            (r'(.+)\s+were\s+(\w+ed)\s+by\s+(.+)', r'\3 \2 \1'),
            (r'(.+)\s+is\s+(\w+ed)\s+by\s+(.+)', r'\3 \2 \1'),
            (r'(.+)\s+are\s+(\w+ed)\s+by\s+(.+)', r'\3 \2 \1')
        ]
        for pat, rep in patterns:
            if re.search(pat, sentence, re.IGNORECASE):
                converted = re.sub(pat, rep, sentence, flags=re.IGNORECASE)
                return f"Consider active voice: {converted}"
        return f"Consider rewriting in active voice: {sentence}"

    # Sentiment
    def _analyze_positivity(self, text: str) -> Dict[str, float]:
        s = self.sentiment_analyzer.polarity_scores(text)
        return {
            'raw_compound': s['compound'],
            'normalized_score': (s['compound'] + 1) / 2,  # [-1,1] -> [0,1]
            'positive': s['pos'],
            'negative': s['neg'],
            'neutral': s['neu']
        }

    # --- Negative language (improved) ---

    def _detect_negative_language_v2(self, text: str, element_id: str, page_line_ref: str) -> List[ToneIssue]:
        issues: List[ToneIssue] = []
        if not text.strip():
            return issues

        doc = self.nlp(text)

        # Simple context exception: if sentence already signals solution
        # e.g., “problem we solved by… / addressed through…”
        solution_verbs = {"solve", "mitigate", "address", "resolve", "fix"}
        solution_cues = {"by", "through", "with", "using", "so that", "therefore"}
        context_is_solution = any(tok.lemma_.lower() in solution_verbs for tok in doc)

        candidates = self._scan_negative_tokens(doc) + self._scan_negative_phrases(doc)

        for c in candidates:
            # Skip “not only … but also …”
            window_text = self._window_text(text, c["start"], c["end"], left=15, right=20).lower()
            if "not only" in window_text and "but also" in window_text:
                continue

            # Skip when immediately followed by explicit solution cue
            if context_is_solution:
                right = doc.char_span(c["end"], min(len(text), c["end"] + 30))
                if right and any(w.lower_ in solution_cues for w in right):
                    continue

            positive_alt = self._positive_alternative(c)
            if not positive_alt:
                continue

            confidence = 0.85 * c["weight"]
            if c["has_negator"]:
                confidence *= 0.8  # negated tokens already reduce negativity

            issues.append(ToneIssue(
                issue_type="negative_language",
                original_text=c["span"],
                suggested_fix=positive_alt,
                confidence=min(confidence, 0.99),
                explanation="Use positive phrasing to maintain solution-oriented tone",
                element_id=element_id,
                page_line_ref=page_line_ref
            ))
        return issues

    def _scan_negative_tokens(self, doc):
        findings = []
        for i, tok in enumerate(doc):
            lemma = tok.lemma_.lower()
            if lemma in self.negative_map:
                # small window around token
                l = max(0, i - 3)
                r = min(len(doc), i + 4)
                window = doc[l:r]
                has_negator = any(w.lower_ in self.negators for w in window)

                # weight for intensifiers/downtoners
                weight = 1.0
                for w in window:
                    lw = w.lower_
                    if lw in self.intensifiers:
                        weight *= self.intensifiers[lw]
                    if lw in self.downtoners:
                        weight *= self.downtoners[lw]

                findings.append({
                    "span": tok.text,
                    "start": tok.idx,
                    "end": tok.idx + len(tok),
                    "lemma": lemma,
                    "has_negator": has_negator,
                    "weight": weight
                })
        return findings

    def _scan_negative_phrases(self, doc):
        findings = []
        for _match_id, start, end in self.matcher(doc):
            span = doc[start:end]
            l = max(0, start - 3)
            r = min(len(doc), end + 4)
            has_negator = any(t.lower_ in self.negators for t in doc[l:r])
            findings.append({
                "span": span.text,
                "start": span.start_char,
                "end": span.end_char,
                "lemma": span.text.lower(),
                "has_negator": has_negator,
                "weight": 1.0
            })
        return findings

    def _positive_alternative(self, candidate) -> Optional[str]:
        # exact key (phrase) first
        alt = self.negative_map.get(candidate["lemma"])
        if alt:
            return alt
        # fallback by surface span
        return self.negative_map.get(candidate["span"].lower())

    def _window_text(self, text: str, start: int, end: int, left: int = 20, right: int = 25) -> str:
        l = max(0, start - left)
        r = min(len(text), end + right)
        return text[l:r]

    # --- Rules & defaults ---

    def _default_negative_map(self) -> Dict[str, str]:
        # Defaults (merged with JSON file entries)
        return {
            "problem": "opportunity",
            "issue": "consideration",
            "weakness": "area for improvement",
            "limitation": "constraint",
            "deficit": "gap",
            "shortcoming": "enhancement opportunity",
            "disadvantage": "tradeoff",
            "obstacle": "challenge",
            "setback": "adjustment",
            "barrier": "consideration",
            "hindrance": "factor to address",
            "impossible": "challenging",
            "fail": "requires adjustment",
            "failed": "requires adjustment",
            "failure": "learning opportunity",
            "worse": "less optimal",
            "can't": "unable to",
            "cannot": "unable to",
            "won't work": "requires an alternative approach",
            "difficult": "requires clarification",  # in context often “difficult to understand”
        }

    def _merge_negative_map_from_rules(self, rules_path: str) -> None:
        try:
            if not os.path.exists(rules_path):
                return
            with open(rules_path, "r", encoding="utf-8") as f:
                rules = json.load(f)
            # Add an optional section in JSON: "negative_phrases": {"problem": "opportunity", ...}
            external = rules.get("negative_phrases", {})
            if isinstance(external, dict):
                self.negative_map.update({k.lower(): v for k, v in external.items()})
        except Exception as e:
            logger.warning(f"Could not load negative_phrases from {rules_path}: {e}")


# --------- convenience function ---------

def analyze_tone(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    analyzer = ToneAnalyzer()
    return analyzer.analyze_elements(elements)
