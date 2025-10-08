# backend/utils/helpers.py
from __future__ import annotations
import json
import re
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Generic text & JSON helpers
def normalize_whitespace(s: str) -> str:
    """Collapse consecutive whitespace and strip ends."""
    return re.sub(r"\s+", " ", s or "").strip()


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """De-duplicate strings case-insensitively while preserving order."""
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        k = normalize_whitespace(x).lower()
        if k and k not in seen:
            seen.add(k)
            out.append(normalize_whitespace(x))
    return out


def chunk_text(s: str, max_len: int) -> List[str]:
    """Chunk text without splitting mid-word when possible."""
    s = s or ""
    if len(s) <= max_len:
        return [s]
    chunks, start = [], 0
    while start < len(s):
        end = min(start + max_len, len(s))
        if end < len(s):
            # backtrack to last whitespace for a cleaner cut
            ws = s.rfind(" ", start, end)
            if ws > start + int(max_len * 0.5):
                end = ws
        chunks.append(s[start:end])
        start = end
    return chunks


def extract_json_from_llm_response(content: str) -> Dict[str, Any]:
    """
    Find and parse the first JSON object in a model response.
    Tolerant to pre/post text.
    """
    if not content:
        return {}
    try:
        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            return {}
        return json.loads(m.group(0))
    except Exception as e:
        logger.warning("extract_json_from_llm_response: parse failed: %s", e)
        return {}



# Protection data helpers (structure)
PROTECTION_KEYS = ["protected_names", "technical_terms", "dates", "numbers", "abbreviations", "ids"]

def get_empty_protection_data() -> Dict[str, List[str]]:
    """Standard, typed shape used across the app."""
    return {k: [] for k in PROTECTION_KEYS}


def coerce_protection_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Coerce arbitrary dict into our standardized protection-data shape.
    - fills missing keys
    - flattens non-list values to lists
    - dedupes & trims
    """
    out = get_empty_protection_data()
    if not isinstance(data, dict):
        return out
    for k in PROTECTION_KEYS:
        raw = data.get(k, [])
        if raw is None:
            raw = []
        if not isinstance(raw, (list, tuple, set)):
            raw = [str(raw)]
        out[k] = dedupe_preserve_order([str(x) for x in raw if str(x).strip()])
    return out


# Document text collection (normalized + raw fallbacks)
def _collect_texts_from_normalized(document: Dict[str, Any]) -> List[str]:
    """
    Normalized schema from DocumentNormalizer:
      doc['pages'][i]['elements'][j]['text']
    """
    texts: List[str] = []
    for page in document.get("pages", []):
        for elem in page.get("elements", []):
            t = normalize_whitespace(elem.get("text", ""))
            if len(t) >= 3:
                texts.append(t)
    return texts


def _collect_texts_from_raw_pptx(document: Dict[str, Any]) -> List[str]:
    """
    Raw PPTXReader output:
      doc['slides'][i]['content_elements'][j]['text']
    """
    texts: List[str] = []
    for slide in document.get("slides", []):
        for elem in slide.get("content_elements", []):
            t = normalize_whitespace(elem.get("text", ""))
            if len(t) >= 3:
                texts.append(t)
        # Optionally include slide title and notes if present
        for key in ("title", "notes"):
            val = normalize_whitespace(slide.get(key) or "")
            if len(val) >= 3:
                texts.append(val)
    return texts


def _collect_texts_from_raw_pdf(document: Dict[str, Any]) -> List[str]:
    """
    Raw PDFReader output:
      doc['pages'][i]['elements'][j]['text']
    """
    texts: List[str] = []
    for page in document.get("pages", []):
        for elem in page.get("elements", []):
            t = normalize_whitespace(elem.get("text", ""))
            if len(t) >= 3:
                texts.append(t)
    return texts


def collect_document_texts(document: Dict[str, Any]) -> List[str]:
    """
    Extract all textual content robustly.
    Priority:
      1) Normalized schema (DocumentNormalizer)
      2) Raw PPTX schema
      3) Raw PDF schema
    """
    if not isinstance(document, dict):
        return []

    # Prefer normalized shape
    texts = _collect_texts_from_normalized(document)
    if texts:
        return texts

    # Try raw PPTX shape
    pptx_texts = _collect_texts_from_raw_pptx(document)
    if pptx_texts:
        return pptx_texts

    # Try raw PDF shape
    pdf_texts = _collect_texts_from_raw_pdf(document)
    if pdf_texts:
        return pdf_texts

    return []


# Small convenience utilities
def combine_texts_with_breaks(texts: List[str], page_sep: str = "\n\n---PAGE BREAK---\n\n") -> str:
    """Join texts using a page-like separator (useful for prompting)."""
    return page_sep.join(texts or [])


def ensure_required_keys(d: Dict[str, Any], required: Iterable[str]) -> bool:
    """Return True iff all keys in required are present in dict d."""
    return all(k in d for k in required)
