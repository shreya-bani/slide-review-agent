"""
groq_usage_improver.py
LLM helper for Usage (Specificity & Inclusivity) rewrites.

- Tight prompts to forbid invented specifics (proper nouns, numbers, dates, locations, orgs).
- Bans "and/or" and contractions; enforces placeholders <WHO>/<WHEN>/<WHAT>/<WHY>/<HOW>/<DATE>/<TIME>.
- Preserves sentence count (one-in â†’ one-out) with a small revision loop.
- If GROQ_API_KEY is missing or Groq SDK is unavailable, methods return None so callers can fall back.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import os
import logging
import re

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Optional spaCy splitter 
try:
    import spacy  # type: ignore
    _NLP = None  # lazy init
except Exception:  # pragma: no cover
    spacy = None
    _NLP = None

# Optional Groq client 
try:
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover
    Groq = None  # sentinel so we can degrade gracefully


# Configuration
@dataclass
class LLMConfig:
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2
    max_tokens: int = 120
    timeout_s: int = 15


PLACEHOLDERS = {"<WHO>", "<WHEN>", "<WHAT>", "<WHY>", "<HOW>", "<DATE>", "<TIME>"}

# Banned patterns
BANNED_CONTRACTIONS = (
    r"\b(aren't|can't|couldn't|didn't|doesn't|don't|hadn't|hasn't|haven't|he'd|he'll|he's|"
    r"I'd|I'll|I'm|I've|isn't|it'd|it'll|it's|let's|mightn't|mustn't|shan't|she'd|she'll|she's|"
    r"shouldn't|that's|there's|they'd|they'll|they're|they've|we'd|we'll|we're|we've|weren't|"
    r"what's|where's|who's|won't|wouldn't|you'd|you'll|you're|you've)\b"
)
RX_AND_OR = re.compile(r"\band\/or\b", re.IGNORECASE)
RX_CONTRACTIONS = re.compile(BANNED_CONTRACTIONS, re.IGNORECASE)
RX_NUMBER = re.compile(r"\d")
RX_PROPER = re.compile(r"\b([A-Z][a-z]+|[A-Z]{2,})\b")  # Titlecase or ALLCAPS tokens
RX_SENT_END = re.compile(r"([.!?])")  # coarse guard


# Utility functions
def _tokens(text: str) -> set:
    return set(re.findall(r"[A-Za-z0-9\-']+", text))


def _has_and_or(s: str) -> bool:
    return bool(RX_AND_OR.search(s))


def _has_contractions(s: str) -> bool:
    return bool(RX_CONTRACTIONS.search(s))


def _has_new_numbers(orig: str, out: str) -> bool:
    """
    Flags if output introduces digits that did not exist in the input (ignoring placeholders).
    """
    out_wo_ph = out
    for ph in PLACEHOLDERS:
        out_wo_ph = out_wo_ph.replace(ph, "")
    return (not RX_NUMBER.search(orig)) and bool(RX_NUMBER.search(out_wo_ph))


def _has_new_proper_nouns(orig: str, out: str) -> bool:
    """
    Heuristic: if output adds Titlecase/ALLCAPS tokens not present in input (ignoring sentence start),
    and they are not placeholders, treat as invented specifics.
    """
    oset = _tokens(orig)
    for tok in RX_PROPER.findall(out):
        if tok in PLACEHOLDERS:
            continue
        if tok in {"I", "We", "Our", "The"}:
            continue
        if tok not in oset:
            return True
    return False


def _needs_revision(orig: str, out: str) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    if _has_and_or(out):
        reasons.append("contains 'and/or'")
    if _has_contractions(out):
        reasons.append("contains contractions")
    if _has_new_numbers(orig, out):
        reasons.append("adds numbers not present in input")
    if _has_new_proper_nouns(orig, out):
        reasons.append("adds proper nouns/ALLCAPS not present in input")
    return (len(reasons) > 0, reasons)


def _ensure_one_sentence(text: str, fallback_original: str) -> str:
    """
    Enforce a single, declarative sentence. If we detect multiple sentence enders,
    keep up to the first ender; if none, append a period. If result is too short,
    fall back to the original.
    """
    stripped = text.strip()
    if not stripped:
        return fallback_original.strip()
    # Trim after the first sentence ender if multiple appear
    m = RX_SENT_END.search(stripped)
    if m:
        idx = m.end()
        candidate = stripped[:idx].strip()
    else:
        candidate = (stripped + ".").strip()

    # Avoid micro-sentences (<=3 words) when original had more content
    if len(candidate.split()) <= 3 and len(fallback_original.split()) > 3:
        return fallback_original.strip()
    return candidate


def _split_sentences(text: str) -> List[str]:
    if spacy is not None:
        global _NLP
        if _NLP is None:
            try:
                _NLP = spacy.load("en_core_web_sm")
            except Exception:
                _NLP = None
        if _NLP is not None:
            doc = _NLP(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            if sents:
                return sents
    # Regex fallback (simple)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]

# Prompt assembly rules and examples

RULES_EXTRA = """
5) Do not invent specifics. Add no new proper nouns, numbers, dates, locations, organizations,
   titles, or claims unless they occur in the input. If information is missing, use placeholders only:
   <WHO>, <WHEN>, <WHAT>, <WHY>, <HOW>, <DATE>, <TIME>.
6) Placeholders discipline: Use at most one of each placeholder type per sentence.
   Prefer the fewest placeholders needed for clarity.
7) Ban "and/or". Rewrite so the meaning is clear without it.
8) Ban contractions (e.g., donâ€™t, canâ€™t, weâ€™re). Always expand them (do not, cannot, we are).
9) Inclusivity: Use person-first, neutral language. Avoid gendered pronouns unless explicitly given.
   Prefer plain wording over clinical phrasing.
10) Fluency & brevity: Produce a natural, concise sentence. Remove redundancy (do not repeat
    the same idea or placeholder). No lists, bullets, or headings.
11) Stability: Preserve the original meaning and any facts present; do not add new claims
    or intensifiers.
12) Shape: Output exactly one declarative sentence ending with a period.

SELF-CHECK before finalizing: Did I add any specifics not in the input? Did I use "and/or"?
Any contractions? Did I overuse placeholders (more than one per type) or repeat ideas?
If yes to any, revise to comply and output one sentence.
"""

EXAMPLES_BAD_GOOD = """Examples â€” Bad â†’ Good
Bad: Weâ€™ll finish and/or deliver soon. â†’ Good: We will deliver by <DATE>.
Bad: Weâ€™re meeting in London. â†’ Good: We will meet on <DATE> at <LOCATION>.
Bad: He is disabled but a good engineer. â†’ Good: The engineer has a disability and excels in their field.
"""

INSTR_ADDENDUM = (
    "Never add new specifics; use placeholders if details are missing. "
    "Do not use 'and/or' or contractions. Use at most one of each placeholder per sentence. "
    "Keep it natural, concise, and person-first."
)


def _build_prompt(base_instruction: str, sentence: str, idx: int, total: int) -> str:
    return (
        f"{base_instruction.strip()}\n\n"
        "RULES:\n"
        "1) Preserve the original meaning and factual content.\n"
        "2) Maintain the same sentence count: exactly one sentence out for one sentence in.\n"
        "3) Keep it concise and professional; no hype.\n"
        "4) If information is missing, prefer placeholders over invention.\n"
        f"{RULES_EXTRA}\n\n"
        f"{EXAMPLES_BAD_GOOD}\n"
        f"SENTENCE {idx} of {total}:\n{sentence.strip()}\n"
        "Output exactly ONE improved sentence:"
    )


# Main class
class GroqUsageImprover:
    """
    Usage:
        improver = GroqUsageImprover()  # safe if no API key; methods return None
        text = improver.rewrite_for_specificity("Original text ...")
        text = improver.rewrite_for_inclusivity("Original text ...")
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.api_key = os.getenv("GROQ_API_KEY")
        self._client = None
        if self.api_key and Groq is not None:
            try:
                self._client = Groq(api_key=self.api_key)
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to init Groq client: %s", e)
                self._client = None
        else:
            if not self.api_key:
                logger.info("GROQ_API_KEY not set; GroqUsageImprover will no-op and return None.")
            if Groq is None:
                logger.info("groq SDK not available; GroqUsageImprover will no-op and return None.")

    # Public API 
    def rewrite_for_specificity(self, text: str) -> Optional[str]:
        """
        Rewrites each sentence to increase specificity (4W+1H) when safe,
        while forbidding invented specifics, 'and/or', and contractions.
        Returns None if LLM is unavailable.
        """
        if not self._client:
            return None
        if not text or not text.strip():
            return ""

        sentences = _split_sentences(text)
        out: List[str] = []
        base_instruction = (
            "Rewrite the following sentence to increase specificity and clarity where needed, "
            "adding placeholders for missing details, while preserving meaning. "
            + INSTR_ADDENDUM
        )
        for i, s in enumerate(sentences, start=1):
            improved = self._rewrite_same_sentence_count(s, base_instruction, i, len(sentences))
            out.append(improved)
        return " ".join(out).strip()

    def rewrite_for_inclusivity(self, text: str) -> Optional[str]:
        """
        Rewrites each sentence to improve inclusivity (person-first language, neutral pronouns),
        while forbidding invented specifics, 'and/or', and contractions.
        Returns None if LLM is unavailable.
        """
        if not self._client:
            return None
        if not text or not text.strip():
            return ""

        sentences = _split_sentences(text)
        out: List[str] = []
        base_instruction = (
            "Rewrite the following sentence to be inclusive and person-first, avoiding gendered terms "
            "unless explicitly given, and rephrasing to center the person rather than a condition. "
            + INSTR_ADDENDUM
        )
        for i, s in enumerate(sentences, start=1):
            improved = self._rewrite_same_sentence_count(s, base_instruction, i, len(sentences))
            out.append(improved)
        return " ".join(out).strip()

    # Internals
    def _rewrite_same_sentence_count(self, original_sentence: str, base_instruction: str,
                                     idx: int, total: int) -> str:
        """
        Calls the LLM to produce exactly one improved sentence per input sentence,
        then validates and optionally revises the output.
        """
        prompt = _build_prompt(base_instruction, original_sentence, idx, total)
        try:
            draft = self._call_groq(prompt, temperature=min(self.config.temperature, 0.2),
                                    max_tokens=self.config.max_tokens)
        except Exception as e:  # pragma: no cover
            logger.warning("Groq call failed (%s); keeping original sentence.", e)
            return original_sentence.strip()

        if not draft:
            return original_sentence.strip()

        draft = _ensure_one_sentence(draft.strip(), original_sentence)
        final = self._revise_if_needed(original_sentence, draft, base_instruction)
        final = _ensure_one_sentence(final.strip(), original_sentence)
        return final

    def _call_groq(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Minimal wrapper around Groq's Chat Completions-style API.
        """
        if not self._client:
            raise RuntimeError("Groq client not initialized")

        # Groq 'chat.completions.create' vs 'client.messages.create' depends on SDK version.
        # Try messages.create first (newer style), then fall back.
        try:
            resp = self._client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            # Newer SDKs: text is under resp.choices[0].message.content or resp.output_text
            out = getattr(resp, "output_text", None)
            if out is None:
                # Attempt to parse choices
                choices = getattr(resp, "choices", None)
                if choices and len(choices) > 0:
                    msg = getattr(choices[0], "message", None)
                    if msg and "content" in msg:
                        out = msg["content"]  # type: ignore
            if not out:
                raise ValueError("Empty response text")
            return str(out).strip()
        except Exception:
            # Fallback to chat.completions style
            resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content  # type: ignore[index]
            return str(text).strip()

    def _revise_if_needed(self, original: str, draft: str, base_instruction: str) -> str:
        needs, reasons = _needs_revision(original, draft)
        if not needs:
            return draft

        fix_note = (
            "REVISION REQUIRED: Remove banned patterns and follow placeholder rules. "
            "Do NOT add specifics. No 'and/or'. No contractions. "
            "Use placeholders <WHO>/<WHEN>/<WHAT>/<WHY>/<HOW>/<DATE>/<TIME> when details are missing. "
            f"Violations detected: {', '.join(reasons)}. Return exactly ONE sentence."
        )
        prompt = f"{base_instruction}\n\n{RULES_EXTRA}\n\nInput sentence:\n{original}\n\nDraft:\n{draft}\n\n{fix_note}"
        try:
            revised = self._call_groq(prompt, temperature=0.1, max_tokens=min(self.config.max_tokens, 100))
            revised = revised.strip()
        except Exception as e:  # pragma: no cover
            logger.debug("Revision call failed (%s). Keeping draft.", e)
            revised = draft

        needs2, _ = _needs_revision(original, revised)
        if not needs2:
            return revised

        # Last-resort local sanitization
        safe = RX_AND_OR.sub("and", revised)
        # Expand a few common contractions; leave others as-is but flagged
        def _expand(m: re.Match) -> str:
            table = {
                "don't": "do not", "can't": "cannot", "won't": "will not",
                "we're": "we are", "I'm": "I am", "it's": "it is",
            }
            g = m.group(0)
            low = g.lower()
            out = table.get(low, None)
            if out is None:
                # generic fallback: split at apostrophe if possible
                parts = re.split(r"'", g)
                return " ".join(parts) if len(parts) > 1 else g
            # Preserve capitalization of the first letter
            if g[0].isupper():
                return out[0].upper() + out[1:]
            return out

        safe = RX_CONTRACTIONS.sub(_expand, safe)

        # If numbers/proper nouns newly appeared, replace them with generic placeholders
        if _has_new_numbers(original, safe):
            safe = re.sub(r"\d[\d,\.]*", "<DATE>", safe)
        if _has_new_proper_nouns(original, safe):
            # Replace unknown proper tokens with <WHO> as a neutral placeholder
            toks = safe.split()
            orig_set = _tokens(original)
            new_toks: List[str] = []
            for t in toks:
                if RX_PROPER.fullmatch(t) and t not in orig_set and t not in PLACEHOLDERS:
                    new_toks.append("<WHO>")
                else:
                    new_toks.append(t)
            safe = " ".join(new_toks)

        return safe