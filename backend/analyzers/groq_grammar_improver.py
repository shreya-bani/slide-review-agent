"""
Groq-powered Grammar & Style Improver
- Uses Groq LLM to produce *minimal*, style-compliant rewrites for IV (A–H).
- Preserves sentence count by default (configurable).
- Falls back to None if GROQ_API_KEY is missing or API errors occur.

Usage:
    from analyzers.groq_grammar_improver import GroqGrammarImprover, LLMConfig
    improver = GroqGrammarImprover()
    fixed = improver.rewrite(text, intent="acronyms|bullets|headings|dashes|numerals|periods|quotes|titles",
                             preserve_sentence_count=True, context_meta={...})
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import json
import logging

logger = logging.getLogger(__name__)

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None


@dataclass
class LLMConfig:
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2
    max_tokens: int = 320


class GroqGrammarImprover:
    def __init__(self, rules_path: str = "rules/amida_style_rules.json", config: Optional[LLMConfig] = None):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if (Groq and self.api_key) else None
        self.config = config or LLMConfig()
        # Load rules text to steer the model
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
        except FileNotFoundError:
            self.rules = {}

    def available(self) -> bool:
        return self.client is not None

    def rewrite(
        self,
        text: str,
        intent: str,
        preserve_sentence_count: bool = True,
        context_meta: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Returns improved text or None.
        """
        if not self.available():
            return None

        sys = self._system_prompt(preserve_sentence_count=preserve_sentence_count)
        user = self._user_prompt(text=text, intent=intent, meta=context_meta or {})
        try:
            resp = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # fail open; let rule-based fixes handle it
            logger.warning("GroqGrammarImprover error: %s", e)
            return None

    # ------------------ prompts ------------------

    def _system_prompt(self, preserve_sentence_count: bool) -> str:
        # Keep this short but binding; it references your IV rules.
        # Note: we embed only the needed rule bullets to keep token use low.
        sentences = "The number of sentences in your output must equal the number of sentences in the input." \
                    if preserve_sentence_count else \
                    "Keep meaning intact; be concise."
        return (
            "You are a precise copy editor for presentation slides. "
            "Apply the Amida Style Guide, Section IV (Grammar): "
            "Acronyms (full phrase before first use + acronym in parentheses), "
            "Bullets (no terminal periods; parallel; capitalize first word; split multiple sentences), "
            "Headings (Lato; sizes/colors by level), "
            "Hyphens (use en dash –; never em dash — or --), "
            "Numerals (spell out <100 in non-technical text; comma in 4+ digits; no comma in dates; avoid starting a sentence with digits; numerals before percent/million/billion), "
            "Periods (single space after sentence-ending punctuation), "
            "Quotation Marks (comma/period inside; semicolon/colon outside), "
            "Titles (Title Case: capitalize first/last/main words; words >3 letters). "
            f"{sentences} "
            "Return only the corrected text without explanations."
        )

    def _user_prompt(self, text: str, intent: str, meta: Dict[str, Any]) -> str:
        extra = ""
        if intent == "headings" and meta:
            # You can pass expected level to encourage correct casing/format; font enforcement is handled upstream
            level = meta.get("level")
            if level:
                extra = f"\nHint: This is a Heading {level}."
        return (
            "Input text:\n"
            f"\"\"\"\n{text}\n\"\"\"\n\n"
            "Apply the relevant rule(s) for: " + intent + "."
        )
