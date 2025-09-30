"""
Tone checker (II. Tone → A. Active Voice)

Category: tone-analysis
Rules implemented:
- active-voice: Flag likely passive-voice and gerund-heavy ("mushy") sentences
  and suggest rewriting in active voice.

This version also optionally calls Groq to produce an AI rewrite ("ai_suggestion"),
while the heuristic/rule-based rewrite remains under "suggestion".
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Local config
from ..config.style_rules import amida_rules  # type: ignore
from ..config.settings import settings  # type: ignore

Issue = Dict[str, Any]
Span = Tuple[int, int, str]  # (start, end, sentence)



# Optional Groq integration

def _has_groq() -> bool:
    try:
        from groq import Groq  # noqa: F401
    except Exception:
        return False
    return bool(settings.llm_provider.lower() == "groq" and settings.groq_api_key)

def _groq_client():
    from groq import Groq  # type: ignore
    return Groq(api_key=settings.groq_api_key)

# One-shot prompt to turn a passive/mushy sentence into a concise, active rewrite.
# We keep it conservative: preserve original meaning, entities, tense when sensible.
_GROQ_PROMPT = """You are an expert technical editor following the Amida Style Guide (II. Tone → A. Active Voice).
Rewrite the sentence in clear, concise active voice, preserving meaning and proper nouns.
Avoid adding new information. Return ONLY the rewritten sentence, no commentary.

Sentence:
\"\"\"{sentence}\"\"\""""

def ai_rewrite_with_groq(sentence: str) -> Optional[str]:
    """Call Groq (if configured) to get an active-voice rewrite for a single sentence."""
    if not _has_groq():
        return None
    try:
        client = _groq_client()
        model = settings.groq_model or "llama-3.1-8b-instant"
        prompt = _GROQ_PROMPT.format(sentence=sentence.strip())
        # Chat completion (Groq SDK)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise, concise technical copy editor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=256,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Guard: return None if Groq returns empty or unchanged
        if text and text != sentence.strip():
            return text
    except Exception:
        # Fail silently; we still provide rule-based suggestions
        return None
    return None



# Config-driven patterns

def _compile_patterns():
    """
    Load regex patterns & settings from amida_rules.active_voice if present,
    else fall back to safe defaults.
    """
    cfg = getattr(amida_rules, "active_voice", {}) or {}
    passive_patterns = cfg.get("passive_patterns") or [
        # be + past participle (regular verbs)
        r"\b(?:am|is|are|was|were|be|been|being|['’]s|['’]re)\s+\w+ed\b(?:\s+by\b[^.?!]+)?",
        # common irregular participles with optional by-phrase
        r"\b(?:am|is|are|was|were|be|been|being|['’]s|['’]re)\s+("
        r"built|made|done|known|shown|given|taken|seen|kept|left|put|set|read|led|paid|sold|sent|"
        r"told|thought|brought|bought|caught|taught|won|begun|become|found|held|written|driven|"
        r"grown|spent|lost|won|met|meant|cut|hit|hurt)\b(?:\s+by\b[^.?!]+)?"
    ]
    mushy_patterns = cfg.get("mushy_patterns") or [
        # be + V-ing (progressive); allow adverbs in between
        r"\b(?:am|is|are|was|were)\b(?:\s+\w+){0,2}\s+\w+ing\b"
    ]

    severity = cfg.get("severity", "warning")
    max_flags = int(cfg.get("max_flags_per_element", 3))
    min_chars = int(cfg.get("min_chars_to_check", 30))
    return (
        [re.compile(p, re.IGNORECASE) for p in passive_patterns],
        [re.compile(p, re.IGNORECASE) for p in mushy_patterns],
        severity,
        max_flags,
        min_chars,
    )


_PASSIVE_PATTERNS, _MUSHY_PATTERNS, _SEVERITY, _MAX_FLAGS, _MIN_CHARS = _compile_patterns()


# Sentence splitting

_ABBR = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "v.", "etc.", "i.e.", "e.g.",
    "u.s.", "u.k.", "dept.", "no.", "fig.", "al."
}

def split_sentences(text: str) -> List[Span]:
    """
    Lightweight sentence splitter that respects common abbreviations and quotes.
    Returns list of (start, end, sentence).
    """
    spans: List[Span] = []
    i = 0
    start = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ".!?":
            # Candidate end if next is space/newline or end of text
            j = i + 1
            # Handle ellipsis "..."
            if ch == "." and text[i:i+3] == "...":
                i += 3
                continue
            # Peek back a token to see if last token is an abbreviation
            prev = re.search(r"([A-Za-z]\.)+$|(\b\w+\.)$", text[max(0, i-15):i+1])
            if prev:
                token = prev.group(0).lower()
                if token in _ABBR:
                    i += 1
                    continue
            # Advance over closing quotes/brackets
            while j < n and text[j] in ['"', "”", "’", "'", ")", "]"]:
                j += 1
            # Sentence boundary if EoT or whitespace
            if j >= n or (text[j].isspace()):
                end = j
                sent = text[start:end]
                if sent.strip():
                    spans.append((start, end, sent))
                # skip whitespace
                while j < n and text[j].isspace():
                    j += 1
                start = j
                i = j
                continue
        i += 1

    # tail
    if start < n:
        tail = text[start:]
        if tail.strip():
            spans.append((start, n, tail))
    return spans



# Detection helpers

def looks_passive(sentence: str) -> bool:
    return any(p.search(sentence) for p in _PASSIVE_PATTERNS)

def looks_mushy(sentence: str) -> bool:
    return any(p.search(sentence) for p in _MUSHY_PATTERNS)



# Heuristic rewrites

_BE_PAST_BY_RE = re.compile(
    r"\b(?P<be>am|is|are|was|were|be|been|being|['’]s|['’]re)\s+"
    r"(?P<part>\w+ed|built|made|done|known|shown|given|taken|seen|kept|left|put|set|read|led|paid|sold|sent|"
    r"told|thought|brought|bought|caught|taught|won|begun|become|found|held|written|driven|grown|spent|lost|met|meant|cut|hit|hurt)"
    r"\s+by\s+(?P<agent>[^.?!,;]+)", re.IGNORECASE
)

def _capitalize_word(w: str) -> str:
    return w[:1].upper() + w[1:] if w else w

def _to_base_s_form(verb: str) -> str:
    """
    Very rough past participle -> past/active guess:
    For safety, we just drop 'was|were' and keep participle as a past-tense look.
    """
    return verb

def rewrite_passive_simple(sentence: str) -> Optional[str]:
    """
    Attempt a minimal active rewrite for 'X was V-ed by Y' → 'Y V-ed X'.
    Returns None if no safe rewrite found.
    """
    m = _BE_PAST_BY_RE.search(sentence)
    if not m:
        return None
    part = m.group("part")
    agent = m.group("agent").strip()
    # Subject (patient) is the left context up to the be-verb
    left = sentence[:m.start()].strip()
    left = re.sub(r"\s*,\s*$", "", left)

    end_punct = ""
    if sentence.rstrip() and sentence.rstrip()[-1] in ".!?":
        end_punct = sentence.rstrip()[-1]

    active_verb = _to_base_s_form(part)
    agent_clean = agent.strip()
    if not agent_clean:
        return None

    agent_clean = agent_clean[0].upper() + agent_clean[1:] if sentence.strip().startswith(left) else agent_clean

    active = f"{_capitalize_word(agent_clean)} {active_verb} {left}"
    active = active.strip()
    if end_punct:
        active = active.rstrip(".!?") + end_punct
    return active


_BE_ING_RE = re.compile(
    r"\b(?P<subj>[A-Z][^,.!?;:]{0,40}?)\s+(?:am|is|are|was|were)\b\s+(?:\w+\s+){0,2}(?P<verb>\w+ing)\b",
    re.IGNORECASE
)

def rewrite_mushy_simple(sentence: str) -> Optional[str]:
    m = _BE_ING_RE.search(sentence)
    if not m:
        return None
    subj = m.group("subj").strip()
    ger = m.group("verb").lower()
    if ger.endswith("ing"):
        base = ger[:-3]
        if base == "":
            return None
        if sentence.strip().lower().startswith(subj.lower()):
            tail_after_ger = sentence[m.end():]
            active = f"{_capitalize_word(subj)} {base}{tail_after_ger}"
            return active
    return None



# Issue building

def _apply_patches(text: str, patches: List[Tuple[int, int, str]]) -> str:
    if not patches:
        return text
    patches.sort(key=lambda p: p[0], reverse=True)
    out = text
    for s, e, r in patches:
        out = out[:s] + r + out[e:]
    return out

def _build_issue(slide_idx, element_index, found_text, suggestion_text, ai_suggestion_text, description) -> Issue:
    issue: Issue = {
        "rule_name": "active-voice",
        "severity": _SEVERITY,
        "category": "tone-analysis",
        "description": description,
        "location": f"slide {slide_idx} • element {element_index if element_index is not None else '?'}",
        "found_text": found_text,
        "suggestion": suggestion_text,      # rule-based
        "page_or_slide_index": slide_idx,
        "element_index": element_index,
    }
    if ai_suggestion_text and ai_suggestion_text.strip() and ai_suggestion_text.strip() != found_text.strip():
        issue["ai_suggestion"] = ai_suggestion_text.strip()  # Groq-based
    return issue


def collect_active_voice_issue_for_element(text: str, slide_idx, element_index) -> Optional[Issue]:
    """
    Scan an element's text, flag up to N problematic sentences,
    and provide a suggested rewrite (heuristic, conservative).
    If Groq is configured, also attach an AI rewrite as 'ai_suggestion'.
    """
    if not text or len(text) < _MIN_CHARS:
        return None

    spans = split_sentences(text)
    if not spans:
        spans = [(0, len(text), text)]

    patches: List[Tuple[int, int, str]] = []
    ai_candidates: List[str] = []
    hits = 0

    for start, end, sent in spans:
        if hits >= _MAX_FLAGS:
            break

        # Try passive first
        if looks_passive(sent):
            rewrite = rewrite_passive_simple(sent)
            if rewrite and rewrite != sent:
                patches.append((start, end, rewrite))
                ai_candidates.append(sent.strip())
                hits += 1
                continue

        # Try mushy progressive
        if looks_mushy(sent):
            rewrite = rewrite_mushy_simple(sent)
            if rewrite and rewrite != sent:
                patches.append((start, end, rewrite))
                ai_candidates.append(sent.strip())
                hits += 1
                continue

    if not patches:
        return None

    suggestion = _apply_patches(text, patches)
    if suggestion == text:
        return None

    # Optionally get one consolidated AI rewrite from Groq for the FIRST hit sentence only
    ai_rewrite: Optional[str] = None
    if ai_candidates:
        ai_rewrite = ai_rewrite_with_groq(ai_candidates[0])

    desc = (
        "Prefer active voice: avoid passive constructions (“to be” + past participle, often with a “by” phrase) "
        "and weak progressive forms (e.g., “is hiring”). Where possible, rewrite to a direct subject–verb sentence."
    )
    return _build_issue(slide_idx, element_index, text, suggestion, ai_rewrite, desc)



# Document traversal

def _iter_pages(doc: Any) -> List[Dict[str, Any]]:
    """
    Normalize document structure so we can iterate pages safely
    even if the root is a list or has a direct elements array.
    """
    if isinstance(doc, dict):
        if "pages" in doc and isinstance(doc["pages"], list):
            return doc["pages"]
        if "elements" in doc and isinstance(doc["elements"], list):
            return [{"index": 0, "elements": doc["elements"]}]
        return [{"index": 0, "elements": [doc]}]
    elif isinstance(doc, list):
        if doc and isinstance(doc[0], dict) and "elements" in doc[0]:
            return doc  # list of pages
        return [{"index": 0, "elements": doc}]
    return [{"index": 0, "elements": []}]


def check_document(document: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []

    pages = _iter_pages(document)
    for page in pages:
        slide_idx = page.get("index", 0)
        for elem in page.get("elements", []):
            text = elem.get("text", "") or ""
            loc = elem.get("locator", {}) or {}
            element_index = loc.get("element_index")

            issue = collect_active_voice_issue_for_element(text, slide_idx, element_index)
            if issue:
                issues.append(issue)

    return issues



# CLI

def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python -m backend.analyzers.advance_style_analyzer <normalized.json> [--out path.json]")
        print("   or: python advance_style_analyzer.py <normalized.json> [--out path.json]")
        sys.exit(2)

    in_path = sys.argv[1]
    out_path: Optional[str] = None
    if len(sys.argv) == 3 and sys.argv[2].startswith("--out"):
        # allow "--out=..." or "--out ..."  (basic handling)
        if "=" in sys.argv[2]:
            out_path = sys.argv[2].split("=", 1)[1].strip()
        else:
            print("If you pass --out, use --out=/full/path.json")
            sys.exit(2)

    with open(in_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    issues = check_document(doc)

    out_json = json.dumps(issues, ensure_ascii=False, indent=2)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_json)
    else:
        print(out_json)


if __name__ == "__main__":
    main()
