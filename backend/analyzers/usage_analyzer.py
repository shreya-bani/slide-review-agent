"""
Usage Analyzer - Specificity and Inclusivity Rules

Two rules:
  1) specificity  – detect vague temporal references and generic terms like "people"
  2) inclusivity  – detect gendered language, promote singular "they", person-first language

Key Features:
- LLM-based detection with batch processing
- Protection layer integration (SWOT terms protected)
- Rule-based pre-filtering for efficiency
- Batch rewriting with validation
- Fallback mechanisms when LLM unavailable

CLI:
    python -m backend.analyzers.usage_analyzer <normalized.json> [--no-rewrite] [-v]
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import spacy
from tqdm import tqdm

from ..config.settings import settings
from ..services.llm_client import LLMClient
from .protection_layer import ProtectionLayer
from ..utils.prompt_loader import load_prompt
from .model.models import Category, Severity

# LOGGER
logger = logging.getLogger(__name__)
logger.setLevel(settings.get_log_level())

# CONFIG
INCLUDE_NOTES: bool = bool(getattr(settings, "analyzer_include_notes", False))
ENABLE_REWRITE: bool = bool(getattr(settings, "enable_rewrite", True))
USE_PROGRESS_BAR: bool = bool(getattr(settings, "debug", False))
MAX_BATCH_ITEMS: int = int(getattr(settings, "llm_chunk_size", 50))

# Regex patterns for detection
TEMPORAL_VAGUE_PATTERN = re.compile(
    r'\b(currently|at this time|presently|right now|at present)\b',
    re.IGNORECASE
)
GENERIC_PEOPLE_PATTERN = re.compile(
    r'\bpeople\b(?!\s+(instead|rather|first|centered|centric))',
    re.IGNORECASE
)

# Inclusivity patterns
GENDERED_TERMS_PATTERN = re.compile(
    r'\b(guys|waitress|waiter|chairman|chairwoman|mankind|manpower|policeman|fireman|mailman|stewardess)\b',
    re.IGNORECASE
)
PRONOUN_PATTERN = re.compile(
    r'\b(he|she|his|her|him|himself|herself)\b(?!\s+(or she|or he))',
    re.IGNORECASE
)
DISABILITY_PATTERN = re.compile(
    r'\b(is disabled|are disabled|disabled person|disabled people|handicapped|wheelchair-bound|suffering from)\b',
    re.IGNORECASE
)

# Regex helpers for masking
NUM_PATTERN = re.compile(r"\b(?:\d{1,3}(?:[,\d]{0,3})*(?:\.\d+)?|\d+%|\$\d[\d,\.]*)\b")
DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b|\b\d{4}\b"
)
EMAIL_URL_PATTERN = re.compile(r"(?:(?:mailto:)?[\w\.-]+@[\w\.-]+\.\w+|https?://\S+)", re.IGNORECASE)

# Heading detection (copied from tone_analyzer for consistency)
_HEADING_ONLY_RE = re.compile(r'^\s*([A-Z][A-Za-z0-9 /&\-\(\)]+:)\s*$')
_HEADING_RE = re.compile(r'^\s*([A-Z][A-Za-z0-9 /&\-\(\)]+:)\s*(.+)$')


class UsageAnalyzer:
    """
    Two-rule usage analyzer: specificity and inclusivity.
    Uses LLM for detection and rewriting with batch processing.
    """

    def __init__(
        self,
        *,
        use_llm: bool = True,
        protection_data: dict | None = None,
        protection_cache_path: Path | None = None,
    ):
        # spaCy for linguistic analysis
        nlp = spacy.load("en_core_web_sm")
        keep = {"tok2vec", "tagger", "attribute_ruler", "parser", "lemmatizer"}
        nlp.disable_pipes(*[p for p in nlp.pipe_names if p not in keep])
        self.nlp = nlp

        # LLM client
        self.llm = LLMClient() if use_llm else None

        # Protection layer
        self.protection_layer = ProtectionLayer(llm_client=self.llm)
        if protection_data:
            self.protection_layer.set_protection_data(protection_data)
        elif protection_cache_path and protection_cache_path.exists():
            self.protection_layer.load(protection_cache_path)

        # Fast local alias
        self._prot = self.protection_layer

    # LLM WRAPPER
    def _llm_chat(self, system_msg: str, user_msg: str) -> Optional[str]:
        """Route all LLM calls through LLMClient; return text or None."""
        if not self.llm:
            return None
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

    def _is_heading_like(self, s: str) -> bool:
        """Check if text looks like a heading (to skip analysis)."""
        s = (s or "").strip()
        if not s:
            return False
        if self._is_heading_only(s):
            return True
        words = s.split()
        if len(words) > 4:
            return False
        doc = self.nlp(s)
        has_verb = any(t.pos_ in {"VERB", "AUX"} for t in doc)
        mostly_nouns = all(t.pos_ in {"NOUN", "PROPN", "ADJ", "PUNCT", "CCONJ", "ADP"} for t in doc)
        is_titleish = (s == s.title()) or s.isupper()
        return (not has_verb) and mostly_nouns and is_titleish

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
    def _is_protected_span(self, text: str) -> bool:
        """Check if text contains protected content."""
        return self._prot.is_protected(text)

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

    # SENTENCE FILTER
    def _is_sentence_like(self, text: str) -> bool:
        """Prefer content with a verb or closing punctuation."""
        text = (text or "").strip()
        if not text:
            return False
        if text.endswith(('.', '!', '?')):
            return True
        doc = self.nlp(text)
        return any(t.pos_ in {"VERB", "AUX"} for t in doc)

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

            # Technical terms
            tech_terms = re.findall(r'\b(?:AI|ML|API|BPA|HITRUST|SWOT|IT|FWA|LLM|EPS)\b', text)
            preserve_facts.extend(tech_terms[:2])

            constraints.append({
                "preserve_facts": list(dict.fromkeys(preserve_facts))[:5],  # dedupe + cap
                "context": context,
                "key_constraint": key_constraint
            })
        return constraints

    @staticmethod
    def _build_blocks(texts, constraints, protected_items):
        lines_block = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))

        def one(i, c):
            pres = ", ".join(c.get("preserve_facts", [])[:5])
            ctx = c.get("context", "neutral")
            key = c.get("key_constraint", "")
            parts = [f"[PRESERVE: {pres}]", f"[CONTEXT: {ctx}]"]
            if key:
                parts.append(f"[KEY: {key}]")
            return f"{i}. " + " ".join(parts)

        constraints_block = "\n".join(one(i, c) for i, c in enumerate(constraints, 1))
        prot = "; ".join(protected_items) if protected_items else "None"
        return lines_block, constraints_block, prot

    @staticmethod
    def _create_specificity_prompt(texts, constraints, protected_items):
        template = load_prompt("specificity_prompt.txt")
        lines_block, constraints_block, prot = UsageAnalyzer._build_blocks(texts, constraints, protected_items)
        system_msg = template.format(
            lines_block=lines_block,
            constraints_block=constraints_block,
            protected_items=prot,
        )
        user_msg = "Rewritten:"
        return system_msg, user_msg

    @staticmethod
    def _create_inclusivity_prompt(texts, constraints, protected_items, *, context="", key_constraint=""):
        template = load_prompt("inclusivity_prompt.txt")
        lines_block, constraints_block, prot = UsageAnalyzer._build_blocks(texts, constraints, protected_items)
        system_msg = template.format(
            lines_block=lines_block,
            constraints_block=constraints_block,
            protected_items=prot,
            context=context or "neutral",
            key_constraint=key_constraint or "",
        )
        user_msg = "Rewritten:"
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
        return True, "valid"

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

    def _collect_texts(self, normalized: Dict[str, Any]) -> List[Tuple[int, int, Dict[str, Any], str]]:
        """Collect all text elements for analysis."""
        triplets = list(self._iter_elements(normalized))
        results: List[Tuple[int, int, Dict[str, Any], str]] = []

        for slide_idx, element_idx, element in triplets:
            t = element.get("text", "")
            if isinstance(t, str) and t.strip():
                if not self._is_heading_like(t):
                    results.append((slide_idx, element_idx, element, t.strip()))

        logger.info(f"Collected {len(results)} text elements from document")
        return results

    @staticmethod
    def _locator(element: Dict[str, Any], slide_idx: int, element_idx: int) -> Tuple[int, int, str, str]:
        loc = element.get("locator") or {}
        etype = loc.get("element_type") or element.get("element_type") or "content"
        page = loc.get("page_or_slide_index") or loc.get("page_index") or (slide_idx + 1)
        elem = loc.get("element_index") or (element_idx + 1)
        return int(page), int(elem), str(etype), f"slide {page}, element {elem}"

    # LLM-BASED DETECTION
    def _detect_issues_batch_llm(
        self,
        texts_data: List[Tuple[int, int, Dict[str, Any], str]],
        rule_type: str
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to detect issues in batch.
        rule_type: "specificity" or "inclusivity"
        """
        if not self.llm:
            logger.warning(f"{rule_type} detection skipped: LLM not available")
            return []

        validator = getattr(settings, "validate_llm_config", None)
        if validator is not None and callable(validator) and not validator():
            logger.warning(f"{rule_type} detection skipped: LLM config invalid")
            return []

        issues: List[Dict[str, Any]] = []

        # Process in batches
        batch_size = MAX_BATCH_ITEMS
        for batch_start in range(0, len(texts_data), batch_size):
            batch = texts_data[batch_start:batch_start + batch_size]

            # Prepare texts for LLM
            texts = [text for _, _, _, text in batch]

            # Skip protected content
            filtered_batch = []
            filtered_indices = []
            for idx, (slide_idx, element_idx, element, text) in enumerate(batch):
                if not self._is_protected_span(text):
                    filtered_batch.append((slide_idx, element_idx, element, text))
                    filtered_indices.append(idx)

            if not filtered_batch:
                continue

            filtered_texts = [text for _, _, _, text in filtered_batch]

            # Collect protection items
            global_protect = self._collect_global_protect(filtered_texts)

            # Build prompt
            lines_block = "\n".join(f"{i}. {t}" for i, t in enumerate(filtered_texts, 1))

            if rule_type == "specificity":
                template = load_prompt("specificity_detection_prompt.txt")
            else:
                template = load_prompt("inclusivity_detection_prompt.txt")

            prot = "; ".join(global_protect) if global_protect else "None"
            system_msg = template.format(
                lines_block=lines_block,
                protected_items=prot,
            )
            user_msg = "Analyze:"

            # Call LLM
            content = self._llm_chat(system_msg, user_msg)
            if not content:
                logger.warning(f"{rule_type} detection: LLM returned empty response for batch")
                continue

            # Parse JSON response
            try:
                # Extract JSON from response (handle markdown code blocks)
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    content = json_match.group(0)

                result = json.loads(content)
                detected_issues = result.get("issues", [])

                for issue_item in detected_issues:
                    line_num = issue_item.get("line", 0)
                    reason = issue_item.get("reason", "")

                    # Map back to original batch
                    if 1 <= line_num <= len(filtered_batch):
                        slide_idx, element_idx, element, text = filtered_batch[line_num - 1]
                        p, e, etype, location = self._locator(element, slide_idx, element_idx)

                        if rule_type == "specificity":
                            severity = Severity.SUGGESTION.value
                            description = f"Use specific terms and concrete dates: {reason}"
                        else:
                            severity = Severity.WARNING.value
                            description = f"Use inclusive language: {reason}"

                        issues.append({
                            "rule_name": rule_type,
                            "severity": severity,
                            "category": Category.USAGE.value,
                            "description": description,
                            "location": location,
                            "found_text": text,
                            "suggestion": None,
                            "page_or_slide_index": p - 1,
                            "element_index": e - 1,
                            "element_type": etype,
                        })

            except json.JSONDecodeError as e:
                logger.warning(f"{rule_type} detection: Failed to parse LLM JSON response: {e}")
                logger.debug(f"Response content: {content[:500]}")
                continue

        logger.info(f"{rule_type} issues detected: {len(issues)}")
        return issues

    def _detect_specificity_issues(self, texts_data) -> List[Dict[str, Any]]:
        """Detect specificity issues using LLM."""
        return self._detect_issues_batch_llm(texts_data, "specificity")

    def _detect_inclusivity_issues(self, texts_data) -> List[Dict[str, Any]]:
        """Detect inclusivity issues using LLM."""
        return self._detect_issues_batch_llm(texts_data, "inclusivity")

    # LLM REWRITE (BATCH)
    def _rewrite_batch_llm(
        self,
        items: List[Tuple[int, str, str, Optional[str]]],
    ) -> List[Tuple[Optional[str], str]]:
        """Batch rewrite using LLM with validation."""
        if not ENABLE_REWRITE:
            return [(None, "disabled")] * len(items)

        validator = getattr(settings, "validate_llm_config", None)
        if validator is not None and callable(validator) and not validator():
            logger.error("LLM config invalid — skipping rewrites.")
            return [(None, "no_config")] * len(items)

        if not items:
            return []

        mode = items[0][2]  # "specificity" or "inclusivity"

        masked_texts: List[str] = []
        masks: List[Dict[str, str]] = []
        originals: List[str] = []
        headings: List[Optional[str]] = []

        for _, body, _, heading in items:
            masked, mask_map = self._mask_preserve(body)
            masked_texts.append(masked)
            masks.append(mask_map)
            originals.append(body)
            headings.append(heading)

        # Constraints
        constraints = self._extract_preserve_constraints_fast(masked_texts)

        # Collect protection items
        global_protect = self._collect_global_protect(masked_texts)

        # Build prompts
        if mode == "specificity":
            system_msg, user_msg = self._create_specificity_prompt(masked_texts, constraints, global_protect)
            rule_name = "specificity"
        else:
            system_msg, user_msg = self._create_inclusivity_prompt(masked_texts, constraints, global_protect)
            rule_name = "inclusivity"

        # Add protection guard
        if global_protect:
            guard = "DO NOT ALTER these protected items if they appear in a line: " + "; ".join(global_protect)
            system_msg += "\n\nAdditional RULE:\n- " + guard
            user_msg = guard + "\n\n" + user_msg

        # Single LLM call
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
                    # Skip "NO_CHANGE" markers
                    if textline.strip().upper() == "NO_CHANGE":
                        candidates[idx] = None
                    else:
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
            ok, reason = self._validate_suggestion(full_original, cand, rule_name)

            if ok:
                results.append((cand, "llm_validated"))
            else:
                results.append((None, f"llm_invalid_{reason}"))

        return results

    # FALLBACK REWRITES
    def _fallback_specificity(self, text: str) -> str:
        """Conservative specificity fixes."""
        t = text
        replacements = [
            (r"\bcurrently\b", "[as of Month Year]"),
            (r"\bat this time\b", "[as of specific date]"),
            (r"\bpresently\b", "[as of Month Year]"),
            (r"\bright now\b", "[as of Month Year]"),
            (r"\bpeople\b", "[specific group]"),
        ]
        for pattern, repl in replacements:
            t2 = re.sub(pattern, repl, t, flags=re.IGNORECASE)
            if t2 != t:
                t = t2
                break
        return t if t != text else text

    def _fallback_inclusivity(self, text: str) -> str:
        """Conservative inclusivity fixes."""
        t = text
        replacements = [
            (r"\bguys\b", "people"),
            (r"\bchairman\b", "chairperson"),
            (r"\bchairwoman\b", "chairperson"),
            (r"\bwaitress\b", "server"),
            (r"\bwaiter\b", "server"),
            (r"\bmankind\b", "humanity"),
            (r"\bmanpower\b", "workforce"),
            (r"\bpoliceman\b", "police officer"),
            (r"\bfireman\b", "firefighter"),
            (r"\bmailman\b", "mail carrier"),
            (r"\bstewardess\b", "flight attendant"),
            (r"\bis disabled\b", "has a disability"),
            (r"\bare disabled\b", "have disabilities"),
            (r"\bdisabled person\b", "person with a disability"),
            (r"\bhandicapped\b", "person with a disability"),
            (r"\bwheelchair-bound\b", "uses a wheelchair"),
        ]
        for pattern, repl in replacements:
            t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
        return t if self._normalize_text(t) != self._normalize_text(text) else text

    # SUGGESTION GENERATION
    def _generate_suggestions_for_all(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        spec_queue: List[Tuple[int, str, str, Optional[str]]] = []
        incl_queue: List[Tuple[int, str, str, Optional[str]]] = []

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

            if issue["rule_name"] == "specificity":
                spec_queue.append((idx, body, "specificity", heading))
            else:
                incl_queue.append((idx, body, "inclusivity", heading))

        def process_queue(queue: List[Tuple[int, str, str, Optional[str]]], fallback_fn) -> Dict[str, int]:
            if not queue:
                return {"llm": 0, "fallback": 0, "failed": 0}

            stats = {"llm": 0, "fallback": 0, "failed": 0}
            k = 0
            while k < len(queue):
                batch = queue[k:k + MAX_BATCH_ITEMS]

                # 1) Try LLM
                llm_results = self._rewrite_batch_llm(batch)

                for (global_idx, body_only, _, heading), (sug, method) in zip(batch, llm_results):
                    if sug and method == "llm_validated":
                        issues[global_idx]["suggestion"] = sug
                        issues[global_idx]["method"] = "llm"
                        stats["llm"] += 1
                        continue

                    if method and method.startswith("llm_invalid_"):
                        bump(method.replace("llm_invalid_", "reject_"))
                    elif method:
                        bump(method)

                    # 2) Fallback
                    sug_body = fallback_fn(body_only)
                    sug_full = self._reattach_heading(heading, sug_body)
                    sug_full = self._finalize_sentence(sug_full)

                    original_full = self._reattach_heading(heading, body_only)
                    ok, reason = self._validate_suggestion(original_full, sug_full, batch[0][2])

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

        _ = process_queue(spec_queue, self._fallback_specificity)
        _ = process_queue(incl_queue, self._fallback_inclusivity)

        logger.info("Rewrite complete")
        return issues

    # PUBLIC API
    def analyze(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis entry point."""
        texts_data = self._collect_texts(input_json)
        if not texts_data:
            return {
                "ok": True,
                "counts": {"specificity": 0, "inclusivity": 0, "with_suggestions": 0, "total": 0},
                "issues": [],
            }

        spec_issues = self._detect_specificity_issues(texts_data)
        incl_issues = self._detect_inclusivity_issues(texts_data)
        issues = spec_issues + incl_issues

        if not issues:
            return {
                "ok": True,
                "counts": {"specificity": 0, "inclusivity": 0, "with_suggestions": 0, "total": 0},
                "issues": [],
            }

        logger.info(
            "Rewrite start: total=%d (specificity=%d, inclusivity=%d), batch_size=%d",
            len(issues),
            len(spec_issues),
            len(incl_issues),
            MAX_BATCH_ITEMS,
        )

        issues = self._generate_suggestions_for_all(issues)

        # Keep only issues with real suggestions
        def _is_real_suggestion(item: Dict[str, Any]) -> bool:
            sug = item.get("suggestion")
            if not sug:
                return False
            return self._normalize_text(sug) != self._normalize_text(item.get("found_text", ""))

        filtered = [i for i in issues if _is_real_suggestion(i)]

        return {
            "ok": True,
            "counts": {
                "specificity": sum(1 for i in filtered if i["rule_name"] == "specificity"),
                "inclusivity": sum(1 for i in filtered if i["rule_name"] == "inclusivity"),
                "with_suggestions": len(filtered),
                "total": len(filtered),
            },
            "issues": filtered,
        }


# CLI
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python -m backend.analyzers.usage_analyzer <normalized.json>\n")
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

        analyzer = UsageAnalyzer(protection_cache_path=cache_path)
        result = analyzer.analyze(normalized)

        print(json.dumps(result, ensure_ascii=False, indent=2))
        counts = result.get("counts", {})
        logger.info(f"Issues: {counts.get('total', 0)}, Suggestions: {counts.get('with_suggestions', 0)}")

    except Exception as e:
        sys.stderr.write(f"ERROR: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
