"""
usage_analyzer.py
Specificity (III.A) + Inclusivity (III.B) checks with graceful LLM rewrites.

Inputs:
    elements: List[Dict]  # each has at least: element_id, text, slide_number (optional)

Config:
    - rules_path: JSON with keys you can extend anytime:
        * "vague_terms": ["people", "things", ...]
        * "temporal_words_to_avoid": ["currently", "at this time", ...]
        * "gendered_terms": {"waitress": "server", "chairman": "chair", ...}

Outputs:
    {
      "issues": [ {issue fields...}, ... ],
      "overall_stats": {...}
    }
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
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
except Exception:  # keeps analyzer usable without Groq file
    GroqUsageImprover = None  # type: ignore

logger = logging.getLogger(__name__)


# -------------------------
# Data structures
# -------------------------

@dataclass
class UsageIssue:
    issue_type: str             # "specificity" | "inclusivity"
    subtype: str                # "vague_term"|"temporal_word"|"missing_4w1h"|"gendered_term"|"noninclusive_pronoun"
    element_id: str
    original_text: str
    suggested_fix: str
    explanation: str
    confidence: float
    page_line_ref: Optional[str] = None


# -------------------------
# Rules loader
# -------------------------

def _load_rules(rules_path: str) -> Dict[str, Any]:
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Specificity helpers
# -------------------------

def _find_terms(text: str, terms: List[str]) -> List[Tuple[str, Tuple[int, int]]]:
    hits: List[Tuple[str, Tuple[int, int]]] = []
    for t in terms:
        for m in re.finditer(rf"\b{re.escape(t)}\b", text, flags=re.IGNORECASE):
            hits.append((t, m.span()))
    return hits

_QUESTION_TOKENS = {
    "who": {"PERSON", "ORG", "NORP"},
    "when": {"DATE", "TIME"},
    "what": set(),
    "why_keywords": {"because", "due to", "so that", "to ", "in order to"},
    "how_keywords": {"by ", "via ", "using ", "through "},
}

def _check_4w1h(text: str) -> Dict[str, bool]:
    found = {"who": False, "when": False, "what": False, "why": False, "how": False}
    if _NLP:
        doc = _NLP(text)
        ents = {e.label_ for e in doc.ents}
        if ents & _QUESTION_TOKENS["who"]:
            found["who"] = True
        if ents & _QUESTION_TOKENS["when"]:
            found["when"] = True
        found["what"] = any(ch.root.pos_ in {"NOUN", "PROPN"} for ch in doc.noun_chunks)
    low = text.lower()
    if any(k in low for k in _QUESTION_TOKENS["why_keywords"]):
        found["why"] = True
    if any(k in low for k in _QUESTION_TOKENS["how_keywords"]):
        found["how"] = True
    return found


def _tangent_score(sent_texts: List[str]) -> float:
    if not _ST or len(sent_texts) < 3:
        return 1.0
    emb = _ST.encode(sent_texts)
    if getattr(emb, "ndim", 1) == 1:
        return 1.0
    vals = []
    for i in range(len(emb) - 1):
        v1, v2 = emb[i], emb[i + 1]
        denom = (norm(v1) * norm(v2)) or 1e-9
        vals.append(float((v1 @ v2) / denom))
    return max(0.0, min(1.0, float(sum(vals) / len(vals))))


# -------------------------
# Inclusivity helpers
# -------------------------

_GENDERED_PRONOUNS = re.compile(r"\b(he|him|his|she|her|hers)\b", re.IGNORECASE)

def _is_specific_person_context(text: str) -> bool:
    if not _NLP:
        return False
    doc = _NLP(text)
    return any(ent.label_ == "PERSON" for ent in doc.ents)


# -------------------------
# Fallback rewrites (no LLM)
# -------------------------

_SPECIFICITY_FALLBACK = (
    "Rewrite to add concrete 4W+1H details. Replace vague terms with specific roles/entities. "
    "Replace temporal words with explicit dates/timeframes. "
    "Use placeholders <WHO>, <WHEN>, <WHAT>, <WHY>, <HOW> where unknown.\n\nOriginal: "
)

_INCLUSIVITY_FALLBACK = (
    "Rewrite with inclusive, gender-neutral language and unbiased job titles. "
    "Use singular 'they' where appropriate.\n\nOriginal: "
)


# -------------------------
# Public API
# -------------------------

def analyze_usage(
    elements: List[Dict[str, Any]],
    rules_path: str = "rules/amida_style_rules.json"
) -> Dict[str, Any]:
    rules = _load_rules(rules_path)
    vague_terms = rules.get("vague_terms", [])
    temporal_words = rules.get("temporal_words_to_avoid", [])
    gendered_map = rules.get("gendered_terms", {})

    # Optional LLM improver
    improver = GroqUsageImprover() if GroqUsageImprover else None

    issues: List[UsageIssue] = []
    counts = {
        "specificity_vague": 0,
        "specificity_temporal": 0,
        "specificity_missing_4w1h": 0,
        "inclusivity_gendered": 0,
        "inclusivity_pronoun": 0,
    }

    for el in elements:
        text = (el.get("text") or "").strip()
        if not text:
            continue

        elem_id = el.get("element_id", "")
        page_ref = el.get("page_line_ref") or str(el.get("slide_number", ""))

        # ---- Specificity: vague terms
        if _find_terms(text, vague_terms):
            llm = improver.rewrite_for_specificity(text) if improver else None
            suggestion = llm or (_SPECIFICITY_FALLBACK + text)
            issues.append(UsageIssue(
                issue_type="specificity",
                subtype="vague_term",
                element_id=elem_id,
                original_text=text,
                suggested_fix=suggestion,
                explanation="Replace vague words with specific roles/entities (4W+1H).",
                confidence=0.85,
                page_line_ref=page_ref,
            ))
            counts["specificity_vague"] += 1

        # ---- Specificity: temporal wording
        if _find_terms(text, temporal_words):
            llm = improver.rewrite_for_specificity(text) if improver else None
            if not llm:
                # simple placeholder replacement if no LLM
                tmp = text
                for t in temporal_words:
                    tmp = re.sub(rf"\b{re.escape(t)}\b", "<WHEN>", tmp, flags=re.IGNORECASE)
                llm = tmp
            issues.append(UsageIssue(
                issue_type="specificity",
                subtype="temporal_word",
                element_id=elem_id,
                original_text=text,
                suggested_fix=llm,
                explanation="Make temporal context explicit (e.g., 'as of January 2025') or use <WHEN>.",
                confidence=0.8,
                page_line_ref=page_ref,
            ))
            counts["specificity_temporal"] += 1

        # ---- Specificity: 4W+1H completeness
        wh = _check_4w1h(text)
        missing = [k for k, v in wh.items() if not v]
        if len(missing) >= 2 and len(text.split()) >= 8:
            llm = improver.rewrite_for_specificity(text) if improver else None
            suggestion = llm or (_SPECIFICITY_FALLBACK + text)
            issues.append(UsageIssue(
                issue_type="specificity",
                subtype="missing_4w1h",
                element_id=elem_id,
                original_text=text,
                suggested_fix=suggestion,
                explanation=f"Add concrete details for: {', '.join(missing)}.",
                confidence=0.7,
                page_line_ref=page_ref,
            ))
            counts["specificity_missing_4w1h"] += 1

        # ---- Inclusivity: gendered terms
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
                explanation=f"Use inclusive alternative for '{gterm}' → '{neutral}'.",
                confidence=0.9,
                page_line_ref=page_ref,
            ))
            counts["inclusivity_gendered"] += 1

        # ---- Inclusivity: pronouns without a specific person
        if _GENDERED_PRONOUNS.search(text) and not _is_specific_person_context(text):
            llm = improver.rewrite_for_inclusivity(text) if improver else None
            suggestion = llm or (_INCLUSIVITY_FALLBACK + text)
            issues.append(UsageIssue(
                issue_type="inclusivity",
                subtype="noninclusive_pronoun",
                element_id=elem_id,
                original_text=text,
                suggested_fix=suggestion,
                explanation="Use singular 'they' or rephrase to avoid gendered pronouns.",
                confidence=0.75,
                page_line_ref=page_ref,
            ))
            counts["inclusivity_pronoun"] += 1

    # Slide-level coherence (optional)
    tangent = 1.0
    try:
        if _ST:
            slide_texts = [e["text"] for e in elements if e.get("text")]
            tangent = _tangent_score(slide_texts) if slide_texts else 1.0
    except Exception:
        tangent = 1.0

    return {
        "issues": [asdict(i) for i in issues],
        "overall_stats": {
            **counts,
            "avg_tangent_coherence": round(float(tangent), 3),
        },
    }
