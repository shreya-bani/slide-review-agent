"""
Tone analyzer for checking active voice and positive language
according to Amida Style Guide requirements and workflow specifications.

Uses spaCy (en_core_web_sm) for active voice detection.
Uses NLTK VADER for positive language scoring (-1 to 1, normalized 0-1).
"""

import spacy
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToneIssue:
    """Represents a tone-related issue"""
    issue_type: str  # 'passive_voice', 'negative_language'
    original_text: str
    suggested_fix: str
    confidence: float
    explanation: str
    element_id: str


class ToneAnalyzer:
    """Analyzes text for tone compliance following exact workflow specifications"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        # VADER sentiment analyzer for positivity scoring
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Positivity threshold as specified (0.7)
        self.positivity_threshold = 0.7
        
        # Passive voice patterns for detection
        self.passive_patterns = [
            r'\b(?:was|were|been|being|is|are|am)\s+\w*ed\b',
            r'\b(?:was|were|been|being|is|are|am)\s+\w*en\b',
            r'\bwas\s+given\b', r'\bwere\s+taken\b', r'\bbeen\s+made\b'
        ]
        
        # Common negative language patterns
        self.negative_patterns = [
            (r'\bnot\s+difficult\b', 'easy'),
            (r'\bnot\s+hard\b', 'simple'),
            (r'\bnot\s+impossible\b', 'achievable'),
            (r'\bnot\s+bad\b', 'good'),
            (r'\bisn\'?t\s+easy\b', 'is challenging'),
            (r'\bcan\'?t\s+', 'unable to '),
            (r'\bwon\'?t\s+work\b', 'requires an alternative approach'),
            (r'\bdifficult\s+to\s+understand\b', 'requires clarification')
        ]
    
    def analyze_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze list of document elements for tone issues
        Following workflow specification: detect + convert passive to active voice,
        score positivity, suggest LLM rewrites if < 0.7
        """
        all_issues = []
        element_analyses = {}
        overall_stats = {
            'total_elements': len(elements),
            'passive_voice_count': 0,
            'negative_language_count': 0,
            'avg_positivity_score': 0.0,
            'elements_needing_llm_rewrite': 0
        }
        
        total_positivity = 0.0
        
        for element in elements:
            element_text = element.get('text', '')
            element_id = element.get('element_id', 'unknown')
            
            if not element_text.strip():
                continue
            
            # A. Active Voice Detection (spaCy)
            passive_issues = self._detect_passive_voice(element_text, element_id)
            all_issues.extend(passive_issues)
            overall_stats['passive_voice_count'] += len(passive_issues)
            
            # B. Positive Language Scoring (VADER)
            positivity_analysis = self._analyze_positivity(element_text, element_id)
            positivity_score = positivity_analysis['normalized_score']
            total_positivity += positivity_score
            
            # Flag for LLM rewrite if positivity < 0.7
            needs_llm_rewrite = positivity_score < self.positivity_threshold
            if needs_llm_rewrite:
                overall_stats['elements_needing_llm_rewrite'] += 1
            
            # Detect negative language patterns
            negative_issues = self._detect_negative_language(element_text, element_id)
            all_issues.extend(negative_issues)
            overall_stats['negative_language_count'] += len(negative_issues)
            
            # Store per-element analysis
            element_analyses[element_id] = {
                'positivity_score': positivity_score,
                'needs_llm_rewrite': needs_llm_rewrite,
                'passive_voice_detected': len(passive_issues) > 0,
                'negative_language_detected': len(negative_issues) > 0,
                'issues_count': len(passive_issues) + len(negative_issues)
            }
        
        # Calculate overall statistics
        overall_stats['avg_positivity_score'] = (
            total_positivity / len(elements) if elements else 0.0
        )
        
        return {
            'issues': [asdict(issue) for issue in all_issues],
            'element_analyses': element_analyses,
            'overall_stats': overall_stats,
            'recommendations': self._generate_recommendations(overall_stats)
        }
    
    def _detect_passive_voice(self, text: str, element_id: str) -> List[ToneIssue]:
        """Detect passive voice using spaCy as specified"""
        issues = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            if self._is_passive_sentence(sent):
                active_suggestion = self._suggest_active_voice_conversion(sent.text)
                
                issue = ToneIssue(
                    issue_type="passive_voice",
                    original_text=sent.text.strip(),
                    suggested_fix=active_suggestion,
                    confidence=0.8,
                    explanation="Convert passive voice to active voice for more direct communication",
                    element_id=element_id
                )
                issues.append(issue)
        
        return issues
    
    def _analyze_positivity(self, text: str, element_id: str) -> Dict[str, float]:
        """Analyze positivity using VADER sentiment analysis"""
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Normalize compound score from [-1, 1] to [0, 1] as specified
        normalized_score = (scores['compound'] + 1) / 2
        
        return {
            'raw_compound': scores['compound'],
            'normalized_score': normalized_score,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def _detect_negative_language(self, text: str, element_id: str) -> List[ToneIssue]:
        """Detect negative language patterns"""
        issues = []
        
        for pattern, positive_alternative in self.negative_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                original_phrase = match.group()
                
                # Create suggested fix by replacing the negative phrase
                suggested_text = re.sub(
                    pattern, positive_alternative, text, 
                    count=1, flags=re.IGNORECASE
                )
                
                issue = ToneIssue(
                    issue_type="negative_language",
                    original_text=original_phrase,
                    suggested_fix=positive_alternative,
                    confidence=0.9,
                    explanation="Use positive phrasing to create clarity and maintain upbeat tone",
                    element_id=element_id
                )
                issues.append(issue)
        
        return issues
    
    def _is_passive_sentence(self, sentence) -> bool:
        """Check if sentence uses passive voice using spaCy"""
        # Look for auxiliary "be" + past participle pattern
        has_be_auxiliary = False
        has_past_participle = False
        
        for token in sentence:
            # Check for "be" auxiliary verbs
            if token.lemma_ == "be" and token.pos_ in ["AUX", "VERB"]:
                has_be_auxiliary = True
            
            # Check for past participle (VBN tag)
            if token.tag_ == "VBN" and has_be_auxiliary:
                has_past_participle = True
                break
        
        # Additional check for "by" agent phrase (strong passive indicator)
        has_by_phrase = any(
            token.text.lower() == "by" and token.i < len(sentence) - 1
            for token in sentence
        )
        
        return has_be_auxiliary and has_past_participle
    
    def _suggest_active_voice_conversion(self, sentence: str) -> str:
        """Suggest active voice conversion (basic pattern matching)"""
        # Simple pattern-based conversions
        active_patterns = [
            (r'(.+)\s+was\s+(\w+ed)\s+by\s+(.+)', r'\3 \2 \1'),
            (r'(.+)\s+were\s+(\w+ed)\s+by\s+(.+)', r'\3 \2 \1'),
            (r'(.+)\s+is\s+(\w+ed)\s+by\s+(.+)', r'\3 \2 \1'),
            (r'(.+)\s+are\s+(\w+ed)\s+by\s+(.+)', r'\3 \2 \1')
        ]
        
        for pattern, replacement in active_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                converted = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
                return f"Consider active voice: {converted}"
        
        return f"Consider rewriting in active voice: {sentence}"
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis statistics"""
        recommendations = []
        
        if stats['avg_positivity_score'] < self.positivity_threshold:
            recommendations.append(
                f"Overall positivity score ({stats['avg_positivity_score']:.2f}) is below threshold ({self.positivity_threshold}). "
                "Consider using more positive language throughout the document."
            )
        
        if stats['passive_voice_count'] > 0:
            recommendations.append(
                f"Found {stats['passive_voice_count']} instances of passive voice. "
                "Convert to active voice for more direct communication."
            )
        
        if stats['negative_language_count'] > 0:
            recommendations.append(
                f"Found {stats['negative_language_count']} instances of negative language patterns. "
                "Use positive phrasing to maintain an upbeat tone."
            )
        
        if stats['elements_needing_llm_rewrite'] > 0:
            recommendations.append(
                f"{stats['elements_needing_llm_rewrite']} elements need LLM rewriting due to low positivity scores."
            )
        
        if not recommendations:
            recommendations.append("Excellent tone analysis! Document demonstrates good active voice usage and positive language.")
        
        return recommendations


# Helper function for easy import
def analyze_tone(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience function to analyze tone"""
    analyzer = ToneAnalyzer()
    return analyzer.analyze_elements(elements)