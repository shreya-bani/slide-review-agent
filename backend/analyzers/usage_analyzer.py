"""
usage_analyzer.py
Enhanced Specificity (III.A) + Inclusivity (III.B) checks with improved detection and slide references.

Key Improvements:
- Better slide reference tracking with element context
- Enhanced 4W+1H detection using multiple methods
- More nuanced gendered language detection
- Improved vague term context analysis
- Better temporal word detection with context
- Enhanced inclusivity checks for person-first language
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Set
import json
import logging
import re

# Optional NLP deps
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

try:
    from sentence_transformers import SentenceTransformer
    from numpy.linalg import norm
    import numpy as np
    _ST = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _ST = None

# Optional LLM improver
try:
    from .groq_usage_improver import GroqUsageImprover
except Exception:
    GroqUsageImprover = None

logger = logging.getLogger(__name__)


# -------------------------
# Data structures
# -------------------------

@dataclass
class UsageIssue:
    issue_type: str             # "specificity" | "inclusivity"
    subtype: str                # "vague_term"|"temporal_word"|"missing_4w1h"|"gendered_term"|"noninclusive_pronoun"|"person_first"
    element_id: str
    original_text: str
    suggested_fix: str
    explanation: str
    confidence: float
    page_line_ref: Optional[str] = None
    detected_terms: Optional[List[str]] = None  # Specific terms that triggered the issue
    context_info: Optional[Dict[str, Any]] = None  # Additional context


# -------------------------
# Rules loader
# -------------------------

def _load_rules(rules_path: str) -> Dict[str, Any]:
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Enhanced reference tracking
# -------------------------

def _build_page_reference(element: Dict[str, Any]) -> str:
    """Build a comprehensive page/slide reference with context - matches tone analysis format."""
    element_id = element.get("element_id", "")
    
    # Use the element_id directly like tone analysis does
    # This will show references like "slide_2_title", "slide_3_shape_1", etc.
    if element_id:
        return element_id
    
    # Fallback if no element_id
    slide_num = element.get("slide_number", "Unknown")
    element_type = element.get("element_type", "").lower()
    return f"slide_{slide_num}_{element_type}"


# -------------------------
# Enhanced specificity helpers
# -------------------------

def _find_terms_with_context(text: str, terms: List[str]) -> List[Tuple[str, Tuple[int, int], str]]:
    """Find terms with surrounding context for better analysis."""
    hits: List[Tuple[str, Tuple[int, int], str]] = []
    for term in terms:
        pattern = rf"\b{re.escape(term)}\b"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = match.span()
            # Get context (15 chars before and after)
            context_start = max(0, start - 15)
            context_end = min(len(text), end + 15)
            context = text[context_start:context_end].strip()
            hits.append((term, (start, end), context))
    return hits

# Enhanced 4W+1H detection
_ENHANCED_4W1H_PATTERNS = {
    "who": {
        "entities": {"PERSON", "ORG", "NORP"},
        "patterns": [
            r"\b(team|staff|engineer|developer|user|client|customer|manager|director)\b",
            r"\b(we|our|they|their|stakeholder|participant)\b",
            r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",  # Proper names
        ]
    },
    "when": {
        "entities": {"DATE", "TIME", "EVENT"},
        "patterns": [
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            r"\b(today|tomorrow|yesterday|next week|last month|this year)\b",
            r"\b(by \w+|during \w+|after \w+|before \w+)\b",
            r"\b(Q[1-4]|quarter|fiscal year|FY)\b",
        ]
    },
    "what": {
        "entities": {"PRODUCT", "EVENT", "WORK_OF_ART"},
        "patterns": [
            r"\b(project|system|application|platform|tool|service|product)\b",
            r"\b(data|analysis|report|dashboard|interface|database)\b",
            r"\b(meeting|presentation|training|workshop|conference)\b",
        ]
    },
    "why": {
        "patterns": [
            r"\b(because|due to|so that|to |in order to|for the purpose of)\b",
            r"\b(reason|purpose|goal|objective|benefit|advantage)\b",
            r"\b(improve|enhance|reduce|increase|achieve|ensure)\b",
        ]
    },
    "how": {
        "patterns": [
            r"\b(by |via |using |through |with |implement|execute|develop)\b",
            r"\b(method|approach|process|procedure|technique|strategy)\b",
            r"\b(steps|phases|stages|workflow|methodology)\b",
        ]
    }
}

def _enhanced_4w1h_check(text: str) -> Dict[str, Any]:
    """Enhanced 4W+1H detection with pattern matching and NLP."""
    found = {"who": False, "when": False, "what": False, "why": False, "how": False}
    details = {"who": [], "when": [], "what": [], "why": [], "how": []}
    
    text_lower = text.lower()
    
    # Pattern-based detection
    for category, config in _ENHANCED_4W1H_PATTERNS.items():
        patterns = config.get("patterns", [])
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                found[category] = True
                details[category].extend(matches)
    
    # NLP-based entity detection
    if _NLP:
        doc = _NLP(text)
        entity_labels = {ent.label_ for ent in doc.ents}
        entity_texts = [ent.text for ent in doc.ents]
        
        for category, config in _ENHANCED_4W1H_PATTERNS.items():
            target_entities = config.get("entities", set())
            if entity_labels & target_entities:
                found[category] = True
                relevant_entities = [ent.text for ent in doc.ents if ent.label_ in target_entities]
                details[category].extend(relevant_entities)
        
        # Enhanced "what" detection using noun phrases
        if not found["what"]:
            noun_chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) >= 2]
            if noun_chunks:
                found["what"] = True
                details["what"].extend(noun_chunks[:3])  # Limit to first 3
    
    return {"found": found, "details": details}

# Enhanced temporal word detection
_ENHANCED_TEMPORAL_WORDS = [
    "currently", "at this time", "right now", "these days", "nowadays",
    "recently", "lately", "soon", "in the near future", "presently",
    "at present", "for now", "as of now", "at the moment", "today",
    "this week", "this month", "this year", "going forward"
]

def _is_temporal_context_acceptable(text: str, term: str) -> bool:
    """Check if temporal word usage is acceptable in context."""
    # Look for specific dates or timeframes near the temporal word
    term_pos = text.lower().find(term.lower())
    if term_pos == -1:
        return False
    
    # Check 50 characters before and after
    context_start = max(0, term_pos - 50)
    context_end = min(len(text), term_pos + len(term) + 50)
    context = text[context_start:context_end]
    
    # Acceptable if there's a specific date or timeframe nearby
    date_patterns = [
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\bQ[1-4]\s+\d{4}\b",
        r"\b(as of|effective|starting|beginning|ending)\s+\w+\b",
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, context, re.IGNORECASE):
            return True
    
    return False


# -------------------------
# Enhanced inclusivity helpers
# -------------------------

_ENHANCED_GENDERED_PRONOUNS = re.compile(r"\b(he|him|his|she|her|hers)\b", re.IGNORECASE)

_PERSON_FIRST_VIOLATIONS = [
    (r"\b(disabled|handicapped)\s+(person|people|individual|user)\b", "person with a disability"),
    (r"\b(autistic)\s+(person|people|child|adult)\b", "person with autism"),
    (r"\b(diabetic)\s+(person|people|patient)\b", "person with diabetes"),
    (r"\b(elderly|senior)\s+(person|people|user)\b", "older adult"),
    (r"\b(mentally ill)\s+(person|people|patient)\b", "person with mental illness"),
]

def _check_person_first_language(text: str) -> List[Tuple[str, str, str]]:
    """Check for person-first language violations."""
    violations = []
    for pattern, suggestion in _PERSON_FIRST_VIOLATIONS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            violations.append((match.group(), suggestion, match.group(1)))
    return violations

def _is_specific_person_context(text: str) -> bool:
    """Enhanced check for specific person context."""
    if not _NLP:
        # Fallback to simple pattern matching
        person_indicators = [
            r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",  # Proper names
            r"\b(Dr\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+\b",  # Titles with names
            r"\b(CEO|CTO|Director|Manager)\s+[A-Z][a-z]+\b",  # Roles with names
        ]
        for pattern in person_indicators:
            if re.search(pattern, text):
                return True
        return False
    
    doc = _NLP(text)
    # Check for PERSON entities or specific roles
    person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    if person_entities:
        return True
    
    # Check for specific job titles or roles that might refer to specific people
    specific_roles = ["ceo", "cto", "director", "manager", "president", "founder"]
    text_lower = text.lower()
    return any(role in text_lower for role in specific_roles)

# Enhanced gendered terms detection
_ENHANCED_GENDERED_TERMS = {
    # Job titles
    "chairman": "chair",
    "chairwoman": "chair", 
    "spokesman": "spokesperson",
    "spokeswoman": "spokesperson",
    "businessman": "business person",
    "businesswoman": "business person",
    "salesman": "salesperson",
    "saleswoman": "salesperson",
    "policeman": "police officer",
    "policewoman": "police officer",
    "fireman": "firefighter",
    "firewoman": "firefighter",
    "mailman": "mail carrier",
    "weatherman": "meteorologist",
    
    # Service roles
    "waitress": "server",
    "waiter": "server",
    "stewardess": "flight attendant",
    "hostess": "host",
    
    # Collective terms  
    "guys": "people",
    "mankind": "humanity",
    "manpower": "workforce",
    "man-hours": "person-hours",
    
    # Other terms
    "freshman": "first-year student",
    "upperclassman": "upper-class student",
}


# -------------------------
# Public API
# -------------------------

def analyze_usage(
    elements: List[Dict[str, Any]],
    rules_path: str = "src/rules/amida_style_rules.json"
) -> Dict[str, Any]:
    """Enhanced usage analysis with better detection and slide references."""
    rules = _load_rules(rules_path)
    vague_terms = rules.get("vague_terms", [])
    temporal_words = rules.get("temporal_words_to_avoid", _ENHANCED_TEMPORAL_WORDS)
    gendered_map = {**rules.get("gendered_terms", {}), **_ENHANCED_GENDERED_TERMS}

    # Optional LLM improver
    improver = GroqUsageImprover() if GroqUsageImprover else None

    issues: List[UsageIssue] = []
    counts = {
        "specificity_vague": 0,
        "specificity_temporal": 0,
        "specificity_missing_4w1h": 0,
        "inclusivity_gendered": 0,
        "inclusivity_pronoun": 0,
        "inclusivity_person_first": 0,
    }

    for el in elements:
        text = (el.get("text") or "").strip()
        if not text:
            continue

        elem_id = el.get("element_id", "")
        page_ref = _build_page_reference(el)

        # ---- Specificity: vague terms with context
        vague_hits = _find_terms_with_context(text, vague_terms)
        if vague_hits:
            detected_terms = [hit[0] for hit in vague_hits]
            llm = improver.rewrite_for_specificity(text) if improver else None
            suggestion = llm or f"Replace vague terms ({', '.join(detected_terms)}) with specific roles/entities using 4W+1H framework."
            
            issues.append(UsageIssue(
                issue_type="specificity",
                subtype="vague_term",
                element_id=elem_id,
                original_text=text,
                suggested_fix=suggestion,
                explanation=f"Replace vague words ({', '.join(detected_terms)}) with specific roles/entities (4W+1H).",
                confidence=0.85,
                page_line_ref=page_ref,
                detected_terms=detected_terms,
                context_info={"contexts": [hit[2] for hit in vague_hits]}
            ))
            counts["specificity_vague"] += 1

        # ---- Specificity: temporal wording with context analysis
        temporal_hits = _find_terms_with_context(text, temporal_words)
        problematic_temporal = [(term, span, ctx) for term, span, ctx in temporal_hits 
                              if not _is_temporal_context_acceptable(text, term)]
        
        if problematic_temporal:
            detected_terms = [hit[0] for hit in problematic_temporal]
            llm = improver.rewrite_for_specificity(text) if improver else None
            if not llm:
                # Enhanced placeholder replacement
                tmp = text
                for term, _, _ in problematic_temporal:
                    tmp = re.sub(rf"\b{re.escape(term)}\b", "<WHEN>", tmp, flags=re.IGNORECASE)
                llm = tmp
            
            issues.append(UsageIssue(
                issue_type="specificity",
                subtype="temporal_word",
                element_id=elem_id,
                original_text=text,
                suggested_fix=llm,
                explanation=f"Make temporal context explicit (e.g., 'as of January 2025') for: {', '.join(detected_terms)}",
                confidence=0.8,
                page_line_ref=page_ref,
                detected_terms=detected_terms
            ))
            counts["specificity_temporal"] += 1

        # ---- Specificity: enhanced 4W+1H completeness
        wh_analysis = _enhanced_4w1h_check(text)
        missing = [k for k, v in wh_analysis["found"].items() if not v]
        
        # Only flag if missing multiple elements and text is substantial
        if len(missing) >= 2 and len(text.split()) >= 8:
            llm = improver.rewrite_for_specificity(text) if improver else None
            suggestion = llm or f"Add concrete details for: {', '.join(missing)}. Consider: WHO (specific roles), WHEN (timeframes), WHAT (specific items), WHY (purpose/benefits), HOW (methods)."
            
            issues.append(UsageIssue(
                issue_type="specificity",
                subtype="missing_4w1h",
                element_id=elem_id,
                original_text=text,
                suggested_fix=suggestion,
                explanation=f"Add concrete details for: {', '.join(missing)}.",
                confidence=0.7,
                page_line_ref=page_ref,
                detected_terms=missing,
                context_info={"found_details": wh_analysis["details"]}
            ))
            counts["specificity_missing_4w1h"] += 1

        # ---- Inclusivity: enhanced gendered terms
        flagged_gendered = None
        for gterm, neutral in gendered_map.items():
            if re.search(rf"\b{re.escape(gterm)}\b", text, flags=re.IGNORECASE):
                flagged_gendered = (gterm, neutral)
                break
        
        if flagged_gendered:
            gterm, neutral = flagged_gendered
            llm = improver.rewrite_for_inclusivity(text) if improver else None
            suggestion = llm or re.sub(rf"\b{re.escape(gterm)}\b", neutral, text, flags=re.IGNORECASE)
            
            issues.append(UsageIssue(
                issue_type="inclusivity",
                subtype="gendered_term",
                element_id=elem_id,
                original_text=text,
                suggested_fix=suggestion,
                explanation=f"Use inclusive alternative: '{gterm}' → '{neutral}'.",
                confidence=0.9,
                page_line_ref=page_ref,
                detected_terms=[gterm]
            ))
            counts["inclusivity_gendered"] += 1

        # ---- Inclusivity: pronouns without specific person context
        if _ENHANCED_GENDERED_PRONOUNS.search(text) and not _is_specific_person_context(text):
            pronouns_found = _ENHANCED_GENDERED_PRONOUNS.findall(text)
            llm = improver.rewrite_for_inclusivity(text) if improver else None
            suggestion = llm or "Use singular 'they' or rephrase to avoid gendered pronouns when not referring to a specific person."
            
            issues.append(UsageIssue(
                issue_type="inclusivity",
                subtype="noninclusive_pronoun",
                element_id=elem_id,
                original_text=text,
                suggested_fix=suggestion,
                explanation="Use singular 'they' or rephrase to avoid gendered pronouns when not referring to a specific person.",
                confidence=0.75,
                page_line_ref=page_ref,
                detected_terms=pronouns_found
            ))
            counts["inclusivity_pronoun"] += 1

        # ---- Inclusivity: person-first language
        person_first_violations = _check_person_first_language(text)
        if person_first_violations:
            violation_text, suggestion_text, problematic_term = person_first_violations[0]
            llm = improver.rewrite_for_inclusivity(text) if improver else None
            suggestion = llm or text.replace(violation_text, suggestion_text)
            
            issues.append(UsageIssue(
                issue_type="inclusivity",
                subtype="person_first",
                element_id=elem_id,
                original_text=text,
                suggested_fix=suggestion,
                explanation=f"Use person-first language: '{violation_text}' → '{suggestion_text}'",
                confidence=0.85,
                page_line_ref=page_ref,
                detected_terms=[problematic_term]
            ))
            counts["inclusivity_person_first"] += 1

    # Slide-level coherence (enhanced)
    tangent = 1.0
    try:
        if _ST and len(elements) > 2:
            slide_groups = {}
            for el in elements:
                slide_num = el.get("slide_number", 1)
                if slide_num not in slide_groups:
                    slide_groups[slide_num] = []
                if el.get("text", "").strip():
                    slide_groups[slide_num].append(el["text"])
            
            # Calculate coherence within and across slides
            coherence_scores = []
            for slide_num, texts in slide_groups.items():
                if len(texts) > 1:
                    embeddings = _ST.encode(texts)
                    if embeddings.ndim > 1:
                        similarities = []
                        for i in range(len(embeddings) - 1):
                            v1, v2 = embeddings[i], embeddings[i + 1]
                            denom = (norm(v1) * norm(v2)) or 1e-9
                            sim = float((v1 @ v2) / denom)
                            similarities.append(sim)
                        if similarities:
                            coherence_scores.append(sum(similarities) / len(similarities))
            
            if coherence_scores:
                tangent = max(0.0, min(1.0, sum(coherence_scores) / len(coherence_scores)))
    except Exception as e:
        logger.warning(f"Error calculating coherence: {e}")
        tangent = 1.0

    return {
        "issues": [asdict(i) for i in issues],
        "overall_stats": {
            **counts,
            "avg_tangent_coherence": round(float(tangent), 3),
            "total_elements_analyzed": len([el for el in elements if el.get("text", "").strip()]),
        },
    }