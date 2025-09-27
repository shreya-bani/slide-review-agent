"""
Groq-powered tone improvement system
Provides intelligent, contextual suggestions for tone improvements
using Groq's fast LLM API
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from groq import Groq
from dotenv import load_dotenv
import logging
import re

# Load environment variables if desired
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ToneImprovement:
    """Represents an intelligent tone improvement suggestion"""
    element_id: str
    original_text: str
    improved_text: str
    improvement_type: str  # 'active_voice', 'positive_language', 'overall_tone'
    explanation: str
    confidence: str  # 'high', 'medium', 'low'


class GroqToneImprover:
    """Uses Groq API to provide intelligent tone improvements"""

    def __init__(self):
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

        # Amida style guide context (summarized)
        self.amida_context = """
        Amida's voice should be:
        - Technical but not condescending
        - Intelligent but not assuming
        - Compelling but not hyperbolic
        - Conversational but not informal
        - Inquisitive but not indecisive
        - Passionate but not tangential

        Core values: Impact, Excellence, and Joy
        Always use active voice and positive language.
        """

        # Passive heuristic: BE-aux + past participle (-ed/-en)
        self.PASSIVE_RE = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b", re.I)

        # Guardrail terms we don't want hallucinated
        self.BANNED_TERMS = {
            "hiring", "engineer", "engineers", "maintenance", "seamless",
            "innovation", "innovative", "pioneering", "careers", "apply",
            "performance", "optimize", "optimal", "join our team",
            "mission", "vision", "in the coming months", "high-quality solutions"
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def improve_tone_issues(
        self,
        issues: List[Dict],
        elements: List[Dict],
        max_improvements: int = 10
    ) -> List[ToneImprovement]:
        """
        Generate intelligent improvements for tone issues using Groq
        """
        improvements: List[ToneImprovement] = []

        # Group issues by element for batch processing
        element_issues: Dict[str, List[Dict]] = {}
        for issue in issues:
            element_id = issue['element_id']
            element_issues.setdefault(element_id, []).append(issue)

        # Map element_id -> full element metadata (so we know element_type)
        elem_lookup: Dict[str, Dict] = {elem['element_id']: elem for elem in elements}

        processed = 0
        for element_id, element_issues_list in element_issues.items():
            if processed >= max_improvements:
                break

            elem_meta = elem_lookup.get(element_id, {})
            original_text = elem_meta.get('text', '') if elem_meta else ''
            if not original_text:
                continue

            try:
                improvement = self._get_groq_improvement(
                    original_text, element_issues_list, element_id, elem_meta
                )
                if improvement:
                    improvements.append(improvement)
                    processed += 1
            except Exception as e:
                logger.error(f"Error getting Groq improvement for {element_id}: {e}")
                continue

        return improvements

    # ------------------------------------------------------------------
    # Core LLM call with guardrails
    # ------------------------------------------------------------------

    def _get_groq_improvement(
        self,
        original_text: str,
        issues: List[Dict],
        element_id: str,
        elem_meta: Optional[Dict] = None
    ) -> Optional[ToneImprovement]:
        """Get improvement suggestion from Groq for a specific text element"""

        issue_descriptions = [f"- {i['issue_type']}: {i['explanation']}" for i in issues]
        element_type = (elem_meta or {}).get("element_type", "").lower()  # 'title' | 'bullet' | 'body' | ...

        # Per-element-type style rules
        if element_type == "title":
            style_block = (
                "- This is a TITLE. Rewrite as a concise Title Case NOUN PHRASE (not a sentence).\n"
                "- Do NOT introduce a subject like 'We' or 'The team'.\n"
                "- Do NOT use modal verbs (e.g., 'will').\n"
                "- Do NOT end with punctuation.\n"
                "- Keep it brief (<= 6 words preferred, 12 max)."
            )
        elif element_type == "bullet":
            style_block = (
                "- This is a BULLET. Keep it concise (<= 20 words), preferably a phrase.\n"
                "- Avoid marketing fluff or future-tense promises.\n"
                "- Only use a full sentence if the original bullet already is one."
            )
        else:
            # body or unknown
            style_block = (
                "- This is BODY text. Produce exactly ONE sentence in ACTIVE voice.\n"
                "- Keep it concise and professional."
            )

        # Base prompt with strict constraints
        prompt = f"""You are rewriting business presentation copy to match Amida's style guide.

{self.amida_context}

Element type: {element_type or "body/unknown"}
Original:
"{original_text}"

Detected issues:
{chr(10).join(issue_descriptions)}

Rewrite rules (MANDATORY):
{style_block}
- Keep the original meaning and factual content ONLY. Do NOT add claims, hiring notes, maintenance notices, or marketing language.
- Convert to ACTIVE voice when a sentence is required. If the original omits the agent, use a neutral agent phrase like "Amida's founders" or "the Amida team", or choose an intransitive active verb (e.g., "launched", "began") that preserves meaning without inventing facts.
- Keep the date(s), names, and numbers exactly as given. Do NOT add numbers that weren't present.
- Respect capitalization appropriate to the element type.

Return strictly this JSON (no code fences, no prose before/after):
{{
  "improved_text": "Final rewrite following the rules",
  "explanation": "Briefly explain how the rewrite follows the rules",
  "confidence": "high|medium|low"
}}"""

        if any(i['issue_type'] == 'passive_voice' for i in issues):
            prompt += "\nReminder: Ensure ACTIVE voice where a sentence is required."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2
            )

            content = response.choices[0].message.content
            first = self._parse_groq_response(content, original_text, element_id, issues)
            if not first:
                return None

            violations = self._violations(original_text, first.improved_text, element_type)
            if violations:
                corrective = "; ".join(violations)
                retry_prompt = prompt + f"""

Your last answer violated these rules: {corrective}.
Rewrite again following the rules exactly. Return JSON only with improved_text, explanation, confidence."""
                response2 = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": retry_prompt}],
                    max_tokens=300,
                    temperature=0.1
                )
                content2 = response2.choices[0].message.content
                second = self._parse_groq_response(content2, original_text, element_id, issues)
                if second and not self._violations(original_text, second.improved_text, element_type):
                    return second
            return first

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return None

    # ------------------------------------------------------------------
    # Parsing + Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` fences if present."""
        if "```" in text:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                return m.group(0)
        return text

    def _parse_groq_response(
        self,
        response_text: str,
        original_text: str,
        element_id: str,
        issues: List[Dict]
    ) -> Optional[ToneImprovement]:
        """Parse Groq's JSON response into a ToneImprovement object"""
        try:
            data = None
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                stripped = self._strip_code_fences(response_text)
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    # Fallback: crude line parsing
                    lines = response_text.strip().splitlines()
                    improved_text, explanation, confidence = "", "", "medium"
                    for line in lines:
                        if '"improved_text"' in line:
                            improved_text = line.split('"improved_text"')[1].split('"')[1]
                        elif '"explanation"' in line:
                            explanation = line.split('"explanation"')[1].split('"')[1]
                        elif '"confidence"' in line:
                            confidence = line.split('"confidence"')[1].split('"')[1].lower()
                    data = {
                        "improved_text": improved_text,
                        "explanation": explanation,
                        "confidence": confidence
                    }

            improved_text = (data.get('improved_text') or "").strip()
            explanation = (data.get('explanation') or "").strip()
            confidence = (data.get('confidence') or "medium").lower()

            if not improved_text or improved_text == original_text:
                return None

            improvement_type = "overall_tone"
            if any(i['issue_type'] == 'passive_voice' for i in issues):
                improvement_type = "active_voice"
            elif any(i['issue_type'] == 'negative_language' for i in issues):
                improvement_type = "positive_language"

            return ToneImprovement(
                element_id=element_id,
                original_text=original_text,
                improved_text=improved_text,
                improvement_type=improvement_type,
                explanation=explanation or "Improved tone and clarity",
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error parsing Groq response: {e}")
            return None

    def _violations(self, original: str, improved: str, element_type: str) -> List[str]:
        """Return a list of rule violations for an improved output."""
        problems: List[str] = []

        # Generic checks
        if element_type != "title":
            # count .!? across the string
            if sum(1 for ch in improved if ch in ".!?") > 1:
                problems.append("More than one sentence")

        # Passive construction remains (for sentence types)
        if element_type not in {"title", "bullet"} and self.PASSIVE_RE.search(improved):
            problems.append("Passive construction remains")

        # Marketing / maintenance / hiring fluff added?
        lower = improved.lower()
        if any(term in lower for term in self.BANNED_TERMS):
            problems.append("Added marketing/maintenance/hiring claims")

        # Numbers: must preserve all from original; don't add if none existed
        num_re = re.compile(r"\b\d{1,4}\b")
        orig_nums = set(num_re.findall(original))
        new_nums = set(num_re.findall(improved))
        if orig_nums and not orig_nums.issubset(new_nums):
            problems.append("Lost/changed numeric facts")
        if not orig_nums and new_nums:
            problems.append("Added numeric facts not in original")

        # Title-specific checks
        words = [w for w in re.split(r"\s+", improved.strip()) if w]
        if element_type == "title":
            if any(ch in improved for ch in ".!?;:"):
                problems.append("Title should not end with punctuation")
            if len(words) > 12:
                problems.append("Title too long")
            # Titles should not start with a full-sentence subject cue
            if re.match(r"^(we|the|our)\b", lower):
                problems.append("Title turned into a sentence")
            # Avoid modal "will" in titles
            if re.search(r"\bwill\b", lower):
                problems.append("Modal verb in title")
        elif element_type == "bullet":
            # Keep bullets short; avoid future-tense promises
            if len(words) > 20:
                problems.append("Bullet too long")
            if re.search(r"\bwill\b", lower):
                problems.append("Avoid 'will' in bullet")

        return problems

    # ------------------------------------------------------------------
    # Overall recommendations
    # ------------------------------------------------------------------

    def generate_overall_recommendations(
        self,
        tone_stats: Dict,
        improvements: List[ToneImprovement]
    ) -> List[str]:
        """Generate intelligent overall recommendations using Groq"""

        prompt = f"""Based on this tone analysis of a business presentation, provide 3-5 specific, actionable recommendations:

Analysis Results:
- Total elements analyzed: {tone_stats['total_elements']}
- Average positivity score: {tone_stats['avg_positivity_score']:.2f}/1.0
- Passive voice issues: {tone_stats['passive_voice_count']}
- Negative language patterns: {tone_stats['negative_language_count']}
- Elements needing improvement: {tone_stats['elements_needing_llm_rewrite']}

{len(improvements)} specific improvements were generated.

Provide practical recommendations for the document author to improve the overall tone, following Amida's style guide (technical but not condescending, compelling but not hyperbolic, positive and active voice).

Respond with a simple list, one recommendation per line, no bullets or numbers."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.4
            )

            lines = [
                ln.strip().lstrip('•-123456789. ')
                for ln in response.choices[0].message.content.strip().split('\n')
                if ln.strip() and not ln.strip().startswith('Based on')
            ]
            return lines[:5]

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [
                "Consider using more active voice throughout the document",
                "Focus on positive language to maintain an upbeat tone",
                "Review technical content for clarity without being condescending"
            ]


# Helper function
def get_groq_improvements(
    issues: List[Dict],
    elements: List[Dict],
    tone_stats: Dict
) -> Dict[str, Any]:
    """Get intelligent tone improvements using Groq"""
    try:
        improver = GroqToneImprover()
        improvements = improver.improve_tone_issues(issues, elements, max_improvements=5)
        recommendations = improver.generate_overall_recommendations(tone_stats, improvements)

        return {
            "improvements": improvements,
            "recommendations": recommendations,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error getting Groq improvements: {e}")
        return {
            "improvements": [],
            "recommendations": [
                f"Groq API error: {str(e)}",
                "Please check your GROQ_API_KEY in .env file"
            ],
            "success": False
        }
