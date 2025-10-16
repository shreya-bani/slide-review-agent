from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from backend.config.settings import settings
from backend.utils.helpers import (
    collect_document_texts,
    get_empty_protection_data,
    coerce_protection_data,
    extract_json_from_llm_response,
    combine_texts_with_breaks,
    PROTECTION_KEYS,
)
from backend.utils.llm_client import LLMClient  

logger = logging.getLogger(__name__)


class LLMConfigError(RuntimeError):
    """Raised when the LLM configuration is invalid (missing API key/model/endpoint)."""
    pass


class ProtectionLayer:
    """
    Encapsulates detection and storage of protected items.

    Public API:
      - detect_all_protected_content(document) -> Dict[str, List[str]]
      - set_protection_data(data) -> None
      - is_protected(text_span) -> bool
      - data property (current protection data)
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, cap_chars: Optional[int] = None):
        self.llm = llm_client
        base = cap_chars if cap_chars is not None else getattr(settings, "llm_chunk_size", 30000)
        self.cap_chars = max(20000, int(base))
        self._data: Dict[str, List[str]] = get_empty_protection_data()
        logger.debug("ProtectionLayer initialized with cap_chars=%d", self.cap_chars)

    # Properties
    @property
    def data(self) -> Dict[str, List[str]]:
        return self._data

    # Public methods
    def set_protection_data(self, data: Dict[str, Any]) -> None:
        """Manual override or precomputed injection."""
        self._data = coerce_protection_data(data)

    def detect_all_protected_content(self, document: Dict[str, Any]) -> Dict[str, List[str]]:
        texts = collect_document_texts(document)
        if not texts:
            logger.info("ProtectionLayer: no texts found in document.")
            self._data = get_empty_protection_data()
            return self._data

        if not settings.validate_llm_config():
            logger.error(
                "ProtectionLayer: Invalid LLM configuration; skipping detection. Ensure LLM_API_KEY, LLM_MODEL, and LLM_API_ENDPOINT are set."
            )
            raise LLMConfigError(
                "Invalid LLM configuration: set LLM_API_KEY, LLM_MODEL, and LLM_API_ENDPOINT."
            )

        if not self.llm:
            logger.warning("ProtectionLayer: no LLM client provided; returning empty protection data.")
            self._data = get_empty_protection_data()
            return self._data

        combined = combine_texts_with_breaks(texts)
        chunk_size = self.cap_chars
        chunks = [combined[i:i+chunk_size] for i in range(0, len(combined), chunk_size)]
        logger.info("ProtectionLayer: splitting into %d chunks (size=%d)", len(chunks), chunk_size)


        merged: Dict[str, List[str]] = {k: [] for k in PROTECTION_KEYS}
        seen: Dict[str, set] = {k: set() for k in PROTECTION_KEYS}

        api_calls = 0  # initialize counter

        for idx, chunk in enumerate(chunks, 1):
            logger.info("ProtectionLayer: processing chunk %d/%d", idx, len(chunks))

            system_prompt = (
                "You are the Protection Layer Extractor. Identify tokens that must NOT be altered by grammar or style rewrites.\n\n"
                "TASK\n"
                "Return ONLY raw JSON (no prose/markdown/code fences) with exactly these keys (in this order):\n"
                f"{', '.join(PROTECTION_KEYS)}\n\n"
                "DEFINITIONS\n"
                "- protected_names: proper names of people/organizations/products/brands/places/government entities/programs; also two-letter US states when used as names.\n"
                "- technical_terms: standards/protocols/model names/version strings/category or contract codes (NOT unique instance IDs).\n"
                "- dates: any date/time expression, incl. standalone years 1900–2099 used as dates and ranges.\n"
                "- numbers: Numeric expressions that are NOT dates (integers/decimals/currency/percentages/ranges/ordinals/units), including compounds like 90-day, 2nd-order, 3 GB/s, 300M, $2.3B.\n"
                "- abbreviations: items that constrain punctuation/casing (U.S., U.K., Dr., Mr., Prof., IT, ROI, CTOs, CIOs, NLT). EXCLUDE e.g. and i.e.\n"
                "- ids: unique instance identifiers (tickets/invoices/asset IDs/concrete contract numbers).\n\n"
                "RULES\n"
                "- Extract EXACT substrings; preserve casing/punctuation/hyphens/spaces.\n"
                "- Prefer technical_terms or ids for letter–digit hybrids; treat 1900–2099 as dates unless clearly part of a standard/code (e.g., ISO 9000).\n"
                "- Keep ranges intact as one token (1–20, 2013–2026). Ignore URLs/emails/paths/hashes.\n"
                "- Ignore plain numerals below 100 unless attached to a unit, symbol, or suffix (%, $, M, B, °, GB, etc.)."
                "- Numerals preceded by 'p.' or 'pp.' are NOT numbers/dates.\n"
                "- De-duplicate while preserving first-seen order.\n"
            )

            user_prompt = (
                "Analyze the following text and populate the six lists accordingly. Do not include any extra keys or commentary.\n\n"
                "RESPONSE FORMAT\n"
                "{\n"
                '  "protected_names": [],\n'
                '  "technical_terms": [],\n'
                '  "dates": [],\n'
                '  "numbers": [],\n'
                '  "abbreviations": [],\n'
                '  "ids": []\n'
                "}\n\n"
                f"TEXT START\n{chunk}\nTEXT END"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                content = self.llm.chat(messages)
                if not content:
                    logger.warning("ProtectionLayer: empty LLM response for chunk %d", idx)
                    parsed = {}
                else:
                    parsed = extract_json_from_llm_response(content) or {}
            except Exception as e:
                logger.warning("ProtectionLayer: LLM call failed for chunk %d: %s", idx, e)
                parsed = {}
            finally:
                api_calls += 1  #  count attempts regardless of outcome

            for key in PROTECTION_KEYS:
                parsed.setdefault(key, [])
            parsed = coerce_protection_data(parsed)

            for key in PROTECTION_KEYS:
                for item in parsed.get(key, []):
                    if item not in seen[key]:
                        merged[key].append(item)
                        seen[key].add(item)

        logger.info("ProtectionLayer: total API calls made = %d", api_calls)

        self._data = merged
        if not any(self._data.values()):
            logger.info("ProtectionLayer: no protected items detected in document.")
         # Log all protected items by category
        for key, items in self._data.items():
            if items:
                logger.info("ProtectionLayer: %s → %s", key, items)
            else:
                logger.info("ProtectionLayer: %s → [empty]", key)

        return self._data


    def is_protected(self, text_span: str) -> bool:
        """Case-insensitive containment either way (span in item OR item in span)."""
        if not text_span:
            return False
        s = text_span.strip().lower()
        for bucket in self._data.values():
            for item in bucket:
                ii = (item or "").strip().lower()
                if not ii:
                    continue
                if s in ii or ii in s:
                    return True
        return False
    
    def save(self, path: str | Path) -> None:
        """Save current protection data to JSON file."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            logger.info(f"ProtectionLayer: saved cache -> {p}")
        except Exception as e:
            logger.warning(f"ProtectionLayer: failed to save cache {path}: {e}")

    def load(self, path: str | Path) -> bool:
        """Load protection data from JSON cache if exists."""
        p = Path(path)
        if not p.exists():
            logger.info(f"ProtectionLayer: cache not found -> {p}")
            return False
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.set_protection_data(data)
            logger.info(f"ProtectionLayer: loaded cache -> {p}")
            return True
        except Exception as e:
            logger.warning(f"ProtectionLayer: failed to load cache {path}: {e}")
            return False

if __name__ == "__main__":
    import sys, json
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python -m backend.analyzers.protection_layer <normalized_document.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with path.open("r", encoding="utf-8") as f:
        document = json.load(f)

    llm = LLMClient()
    layer = ProtectionLayer(llm_client=llm)

    try:
        protection_data = layer.detect_all_protected_content(document)
        print(json.dumps(protection_data, indent=2, ensure_ascii=False))
    except LLMConfigError as e:
        print(f"ERROR: {e}")
        sys.exit(2)