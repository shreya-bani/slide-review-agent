"""
Advanced Style Analyzer (single-file, class-based)

Rules (exactly two):
1) positive_language  -> detects explicit negation + negative sentiment; single description
2) active_voice       -> detects passive voice + mushy/hedging; single description

Guarantees:
- All flagged sentences get a suggestion.
- Automatic batching (no user-provided "tries"). Uses settings.rewrite_batch_limit or 50 by default.
- LLM unavailable? Fallback rule-based rewriter produces suggestions.

CLI:
  python -m backend.analyzers.advance_style_analyzer <normalized.json> [--no-rewrite] [-v]

Returns JSON with counts and issues.
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


TokenSeq = Union[spacy.tokens.Doc, spacy.tokens.Span, Iterable[spacy.tokens.Token]]


class AdvancedStyleAnalyzer:
    """
    Two-rule style analyzer with guaranteed suggestions and automatic batching.
    """

    # Construction / NLP
    def __init__(self) -> None:
        self.analyzer = SentimentIntensityAnalyzer()

        # Load spaCy with components required for sentences + POS/dep to avoid W108
        _nlp = spacy.load("en_core_web_sm")
        keep = {"tok2vec", "tagger", "attribute_ruler", "parser", "lemmatizer"}
        _nlp.disable_pipes(*[p for p in _nlp.pipe_names if p not in keep])
        self.nlp = _nlp

    def _split_heading(self, s: str) -> Tuple[Optional[str], str]:
        """
        If the text starts with a heading like 'Strengths:' or 'Opportunities:',
        return (heading_with_colon, body). Otherwise (None, s).
        """
        m = _HEADING_RE.match((s or "").strip())
        if m:
            return m.group(1), m.group(2)
        return None, s

    def _reattach_heading(self, heading: Optional[str], body: str) -> str:
        """Prepend the heading back to the rewritten body."""
        body = (body or "").strip()
        if heading:
            if not body:
                # heading only, return clean heading (no extra space or dot)
                return heading.strip()
            return f"{heading} {body}".strip()
        return body


    # Public entrypoint
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

        # Fill suggestions for every issue (no manual tries; auto-batch until done)
        pos_cnt = sum(1 for i in issues if i["rule_name"] == "positive_language")
        act_cnt = sum(1 for i in issues if i["rule_name"] == "active_voice")
        logger.info(f"Rewrite start: total={len(issues)} (positive={pos_cnt}, active={act_cnt}), batch_size={MAX_BATCH_ITEMS}")

        # Fill suggestions for every issue (no manual tries; auto-batch until done)
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

    # Input parsing
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

    # Locators
    def _locator(self, element: Dict[str, Any], slide_idx: int, element_idx: int) -> Tuple[int, int, str, str]:
        loc = element.get("locator") or {}
        etype = loc.get("element_type") or element.get("element_type") or "content"
        page = loc.get("page_or_slide_index") or loc.get("page_index") or (slide_idx + 1)
        elem = loc.get("element_index") or (element_idx + 1)
        return int(page), int(elem), str(etype), f"slide {page}, element {elem}"

    # Rule helpers
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

    # Rule executions
    def _detect_negative_language(self, sents) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        iterator = tqdm(sents, desc="Negative check", unit="sent") if USE_PROGRESS_BAR else sents
        for slide_idx, element_idx, element, sent in iterator:
            txt = sent.text.strip()
            p, e, etype, location = self._locator(element, slide_idx, element_idx)

            # negation pattern
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

            # VADER sentiment
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

    # Rewriting (LLM with fallback)
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

    def _unmask(self, text: str, repl: Dict[str, str]) -> str:
        for k, v in repl.items():
            text = text.replace(k, v)
        return text

    def _hf_post(self, payload: Dict[str, Any], retries: int = 3, timeout: int = 60) -> Dict[str, Any]:
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

    def _rewrite_batch_llm(self, items: List[Tuple[int, str, str, Optional[str]]]) -> List[Optional[str]]:
        """
        items: [(global_issue_index, body_only, mode, heading_or_None), ...]
            mode in {'positive','active'}

        Returns: list of rewritten full sentences (with heading reattached) or None,
                in the same order as 'items'.
        """
        if not ENABLE_REWRITE:
            return [None] * len(items)
        if not getattr(settings, "validate_llm_config", None) or not settings.validate_llm_config():
            return [None] * len(items)

        # Prepare masked, numbered input; keep per-item mask maps for BODY ONLY
        tagged_in: List[str] = []
        masks: List[Dict[str, str]] = []
        for idx1, (_, body, mode, _) in enumerate(items, 1):
            masked, mp = self._mask_preserve(body)
            masks.append(mp)
            tag = "POS" if mode == "positive" else "ACT"
            tagged_in.append(f"{idx1}. [{tag}] {masked}")

        system_msg = (
            "You will receive multiple lines. Each line may begin with a label ending in a colon (e.g., 'Strengths:' or 'Opportunities:').\n"
            "RULES:\n"
            "1) Each line is tagged: [POS] = rewrite to positive phrasing; [ACT] = rewrite to active, direct phrasing.\n"
            "2) KEEP ANY LEADING LABEL EXACTLY AS-IS if present (the input you get already excludes it; you'll only rewrite the body).\n"
            "3) Preserve placeholders like [NUM_*], [DATE_*], [LINK_*].\n"
            "4) Preserve subjects names like Amida, Presserve people names like Pradeep"
            "5) Return exactly N lines, numbered 1..N, with only the rewritten text (no tags), one per line."
        )

        user_msg = "Input:\n" + "\n".join(tagged_in) + "\n\nOutput:"

        payload = {
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": min(2000, 120 * len(items)),
            "top_p": 0.9,
        }

        try:
            data = self._hf_post(payload)
            content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            if not content:
                return [None] * len(items)

            # Parse numbered lines 1..N
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            outs: List[Optional[str]] = []
            for idx1 in range(1, len(items) + 1):
                candidate = None
                for ln in lines:
                    if ln.startswith(f"{idx1}.") or ln.startswith(f"{idx1})") or ln.startswith(f"{idx1} "):
                        candidate = re.sub(rf"^{idx1}[\.\)]\s*", "", ln).strip().strip('"').strip("'")
                        break

                if candidate:
                    # Unmask the body
                    candidate = self._unmask(candidate, masks[idx1 - 1])
                    # Reattach heading for that item
                    _, _, _, heading = items[idx1 - 1]
                    candidate = self._reattach_heading(heading, candidate)
                    if candidate and candidate[-1] not in ".!?":
                        candidate += "."
                    outs.append(candidate)
                else:
                    outs.append(None)

            return outs

        except Exception as e:
            logger.warning(f"LLM batch rewrite failed: {type(e).__name__}: {e}")
            return [None] * len(items)



    # Fallback rewriters (deterministic; ensure suggestions always exist)
    def _fallback_positive(self, s: str) -> str:
        """Super-simple positive phrasing fallback."""
        t = s
        # common negations → neutral/positive flips
        t = re.sub(r"\bnot\s+good\b", "needs improvement", t, flags=re.IGNORECASE)
        t = re.sub(r"\bnot\s+bad\b", "acceptable", t, flags=re.IGNORECASE)
        t = re.sub(r"\bno\s+(\w+)\b", r"limited \1", t, flags=re.IGNORECASE)
        t = re.sub(r"\bdon't\b", "avoid", t, flags=re.IGNORECASE)
        t = re.sub(r"\bdoesn't\b", "fails to", t, flags=re.IGNORECASE)
        t = re.sub(r"\bdidn't\b", "did not", t, flags=re.IGNORECASE)
        t = re.sub(r"\bcan't|cannot\b", "is unable to", t, flags=re.IGNORECASE)
        # remove 'not' + adj/verb → assertive phrasing
        t = re.sub(r"\bnot\s+(\w+)\b", r"\1 is limited", t, flags=re.IGNORECASE)
        return t if t.strip() else s

    def _fallback_active(self, s: str) -> str:
        """Simple passive→active / vague→direct heuristics."""
        t = s
        # passive be + VBN → 'We' + verb (very naive)
        t = re.sub(r"\b(is|are|was|were|be|been|being)\s+(\w+ed)\b", r"We \2", t, flags=re.IGNORECASE)
        # hedges
        t = re.sub(r"\bmight\s+be\b", "is", t, flags=re.IGNORECASE)
        t = re.sub(r"\bcould\s+be\b", "is", t, flags=re.IGNORECASE)
        t = re.sub(r"\bwould\s+be\b", "is", t, flags=re.IGNORECASE)
        t = re.sub(r"\bprobably\b|\bperhaps\b|\bmaybe\b", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\bsort of\b|\bkind of\b", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\ba few\b", "several", t, flags=re.IGNORECASE)
        t = re.sub(r"\bsome\s+(\w+)\b", r"\1", t, flags=re.IGNORECASE)
        return t if t.strip() else s

    # Suggestion pipeline
    def _generate_suggestions_for_all(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Build two queues: positive vs active
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

        # inner helper to process a queue in batches
        def process_queue(queue: List[Tuple[int, str, str, Optional[str]]]) -> Tuple[int, int]:
            if not queue:
                return (0, 0)
            k = 0
            batches, fallbacks = 0, 0
            while k < len(queue):
                batch = queue[k:k + MAX_BATCH_ITEMS]
                outs = self._rewrite_batch_llm(batch)
                batches += 1

                for (global_idx, body_only, mode, heading), sug in zip(batch, outs):
                    if not sug:
                        # fallback path
                        sug_body = self._fallback_positive(body_only) if mode == "positive" else self._fallback_active(body_only)
                        sug = self._reattach_heading(heading, sug_body)
                        # (optional) capitalize the first body word after a heading if you added the helper
                        if hasattr(self, "_capitalize_after_heading"):
                            sug = self._capitalize_after_heading(sug)
                        sug = self._finalize_sentence(sug)
                    # ensure it's not effectively identical to found_text
                    orig_full = issues[global_idx].get("found_text", "")
                    if hasattr(self, "_ensure_changed"):
                        sug = self._ensure_changed(orig_full, sug)
                    issues[global_idx]["suggestion"] = sug

                k += MAX_BATCH_ITEMS
            return (batches, fallbacks)

        # run queues and emit a single combined summary
        b_pos, f_pos = process_queue(pos_queue)
        b_act, f_act = process_queue(act_queue)

        total_items = len(pos_queue) + len(act_queue)
        total_batches = b_pos + b_act
        total_fallbacks = f_pos + f_act

        logger.info(f"Rewrite done: items={total_items}, batches={total_batches}, fallbacks={total_fallbacks}")
        return issues


        # inner helper
        def process_queue(queue: List[Tuple[int, str, str, Optional[str]]]) -> Tuple[int, int]:
            if not queue:
                return (0, 0)
            k = 0
            batches, fallbacks = 0, 0
            while k < len(queue):
                batch = queue[k:k + MAX_BATCH_ITEMS]
                outs = self._rewrite_batch_llm(batch)
                batches += 1
                for (global_idx, body_only, mode, heading), sug in zip(batch, outs):
                    if not sug:
                        fallbacks += 1
                        sug_body = self._fallback_positive(body_only) if mode == "positive" else self._fallback_active(body_only)
                        sug = self._reattach_heading(heading, sug_body)
                        sug = self._finalize_sentence(sug)
                    issues[global_idx]["suggestion"] = sug
                k += MAX_BATCH_ITEMS
            return (batches, fallbacks)

        # run queues and emit a single combined summary
        b_pos, f_pos = process_queue(pos_queue)
        b_act, f_act = process_queue(act_queue)

        total_items = len(pos_queue) + len(act_queue)
        total_batches = b_pos + b_act
        total_fallbacks = f_pos + f_act

        logger.info(f"Rewrite done: items={total_items}, batches={total_batches}, fallbacks={total_fallbacks}")
        return issues

    def _finalize_sentence(self, s: str) -> str:
        s = s.strip().strip('"').strip("'")
        # Do NOT force punctuation if the text ends with a heading colon
        if not s or s.endswith(":"):
            return s
        # Only add punctuation if truly missing
        if s[-1] not in ".!?":
            s += "."
        return s
    
    def _ensure_changed(self, original: str, suggestion: str) -> str:
        """If suggestion is effectively identical to original, nudge it minimally but safely."""
        o = (original or "").strip()
        s = (suggestion or "").strip()
        # If only difference is trailing punctuation/whitespace, it's acceptable.
        if o.rstrip().rstrip(".") == s.rstrip().rstrip("."):
            # If it's a heading-only line like 'X:', just return it unchanged (no dot)
            if s.endswith(":"):
                return s
            # Otherwise ensure a final period to register a safe, minimal change
            return self._finalize_sentence(s)
        return s



# CLI
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

    # Allow turning off LLM at runtime (fallback still guarantees suggestions)
    if args.no_rewrite:
        globals()["ENABLE_REWRITE"] = False  # simple runtime switch

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
