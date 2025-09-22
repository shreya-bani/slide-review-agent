"""
groq_usage_improver.py
Keep SAME sentence count as input via per-sentence rewrites.

- Splits the input into sentences (spaCy if available; regex fallback).
- Calls LLM once per sentence with strict instructions to return ONE sentence.
- Post-processes to ensure exactly one sentence per input sentence.
- If LLM fails, returns the original sentence (preserves count).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import os, re, logging

logger = logging.getLogger(__name__)

# Optional spaCy for robust sentence splitting
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None


@dataclass
class LLMConfig:
    model: str = "llama-3.1-8b-instant"   # choose your Groq model
    temperature: float = 0.2
    max_tokens: int = 280
    per_sentence_max_tokens: int = 120
    enforce_same_sentence_count: bool = True


class GroqUsageImprover:
    """
    Usage:
        improver = GroqUsageImprover()
        out1 = improver.rewrite_for_specificity(text)
        out2 = improver.rewrite_for_inclusivity(text)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.config = config or LLMConfig()
        self._client = None
        if self.api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Unable to initialize Groq client: {e}")
                self._client = None

    # ---------- Public API ----------

    def rewrite_for_specificity(self, text: str) -> Optional[str]:
        instr = (
            "Improve specificity. Add concrete 4W+1H details where missing. "
            "Replace vague nouns with specific roles/entities. "
            "Replace temporal words (e.g., 'currently') with explicit dates/timeframes. "
            "Do NOT invent facts—use <WHO>, <WHEN>, <WHAT>, <WHY>, <HOW>, <TIME>, <DATE> when details are unknown."
        )
        return self._rewrite_same_sentence_count(text, instr)

    def rewrite_for_inclusivity(self, text: str) -> Optional[str]:
        instr = (
            "Rewrite using inclusive, gender-neutral language and unbiased job titles. "
            "Use singular 'they' where appropriate. Keep meaning; do NOT invent facts."
        )
        return self._rewrite_same_sentence_count(text, instr)

    # ---------- Core logic (guarantees same sentence count) ----------

    def _rewrite_same_sentence_count(self, text: str, instruction: str) -> Optional[str]:
        if not self._client:
            return None

        in_sents = self._split_sentences(text)
        if not in_sents:
            return ""

        out_sents: List[str] = []
        total = len(in_sents)

        for i, sent in enumerate(in_sents, 1):
            prompt = self._build_prompt(sent, instruction, i, total)
            try:
                resp = self._client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=min(self.config.per_sentence_max_tokens, self.config.max_tokens),
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                logger.warning(f"Groq rewrite failed on sentence {i}: {e}")
                return None

            clean = self._force_one_sentence(raw)
            if not clean:
                clean = sent.strip()  # fallback preserves count
            out_sents.append(clean)

        # Hard guarantee (should already match)
        if self.config.enforce_same_sentence_count and len(out_sents) != len(in_sents):
            out_sents = (out_sents + in_sents[len(out_sents):])[:len(in_sents)]

        return self._join(out_sents)

    # ---------- Utilities ----------

    @staticmethod
    def _join(sents: List[str]) -> str:
        text = " ".join(s.strip() for s in sents if s and s.strip())
        return re.sub(r"\s+([,.!?;:])", r"\1", text).strip()

    @staticmethod
    def _force_one_sentence(text: str) -> str:
        """Collapse to exactly ONE sentence: cut at first terminal punctuation; strip bullets; ensure ending."""
        t = re.sub(r"^\s*[-*•]\s*", "", text.strip())
        m = re.search(r"([^.?!]*[.?!])", t)
        if m:
            return m.group(1).strip()
        # If no terminal punctuation, take first non-empty line and add period
        first = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
        return first if first.endswith((".", "!", "?")) else (first + "." if first else "")

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        t = (text or "").strip()
        if not t:
            return []
        if _NLP:
            return [s.text.strip() for s in _NLP(t).sents if s.text.strip()]
        # Regex fallback: split on ., !, ? followed by space or EOL
        return [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]

    @staticmethod
    def _build_prompt(sentence: str, instruction: str, idx: int, total: int) -> str:
        return (
            "You are a precise copy editor.\n"
            f"INSTRUCTION: {instruction}\n\n"
            "RULES:\n"
            "1) Return exactly ONE sentence that rewrites the given sentence.\n"
            "2) Preserve the original meaning; do NOT invent facts.\n"
            "3) If details are missing, use placeholders <WHO>, <WHEN>, <WHAT>, <WHY>, <HOW>.\n"
            "4) No lists or headings; one concise declarative sentence only.\n\n"
            f"SENTENCE {idx} of {total}:\n{sentence}"
        )
