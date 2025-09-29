"""
Style checker with separate rule outputs:
- Numerals:
  * Spell out numbers < 100 in non-technical text
  * Spell out if a sentence starts with a digit
  * Keep numerals for percent/million/billion unless at sentence start
  * Insert commas for 4+ digits (except dates/IDs/URLs/years)
- Period spacing:
  * Enforce exactly one space after sentence-ending punctuation (. ? ! … or '...')
- Quotation punctuation:
  * Commas/periods go INSIDE closing quotes
  * Semicolons/colons go OUTSIDE closing quotes

Each rule produces its own StyleIssue (no merged rule_name).
"""

import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# import rules (works both as module or direct script)
from ..config.style_rules import amida_rules

# helpers

_NUM_WORDS_1_TO_19 = [
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
    "seventeen","eighteen","nineteen"
]
_TENS = ["", "", "twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

DATE_MONTHS = {
    "jan","january","feb","february","mar","march","apr","april","may","jun","june",
    "jul","july","aug","august","sep","sept","september","oct","october","nov","november","dec","december"
}
UNIT_WORDS = {"percent", "%", "million", "billion"}  # units that prefer numerals

def num_to_words_1_to_99(n: int) -> str:
    if n < 0 or n > 99:
        return str(n)
    if n < 20:
        return _NUM_WORDS_1_TO_19[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_NUM_WORDS_1_TO_19[ones]}"

def looks_like_date_or_year(s: str, start: int, end: int) -> bool:
    token = s[start:end]
    if len(token) == 4 and token.isdigit():  # year like 2025
        return True
    before = s[start-1:start] if start > 0 else ""
    after = s[end:end+1] if end < len(s) else ""
    if before in {"-", "–"} or after in {"-", "–"}:  # numeric range 2–3 / 2-3
        return True
    return False

def word_immediately_before(s: str, start: int) -> str:
    wb = re.search(r"(\b\w+)\W*$", s[:start])
    return (wb.group(1).lower() if wb else "")

def next_token_after(s: str, end: int) -> str:
    wa = re.search(r"^\W*(\w+)", s[end:])
    return (wa.group(1).lower() if wa else "")

def is_sentence_start(s: str, start: int) -> bool:
    """
    True if the token at [start] is at sentence start:
    - beginning of string OR preceded only by whitespace or one of .!?… ) ( — – : ; \n
    """
    i = start - 1
    while i >= 0 and s[i].isspace():
        i -= 1
    if i < 0:
        return True
    return s[i] in {'.', '!', '?', '…', ')', '(', '—', '–', ':', ';', '\n'}

def should_skip_numeric_context(s: str, start: int, end: int, *, allow_sentence_start_override: bool = True) -> bool:
    """
    Skip contexts where numerals should remain as digits:
    - Month before number (dates)
    - Page/section/schedule refs: 'p.10', 'page 10', 'section 5', 'Schedule 70'
    - URLs, domains, paths, filenames, IDs (DIR-CPO-5139), plus-annotated refs (DIR+1, gdit.com+15)
    - Compound modifiers right after digit: 90-day, 30-minute
    - Percent/million/billion units -> numerals (unless sentence start and override allowed)
    """
    before = s[:start]
    after = s[end:]
    window = s[max(0, start-24):min(len(s), end+24)]

    # month immediately before (date)
    if word_immediately_before(s, start) in DATE_MONTHS:
        return True

    # page/section/schedule cues before the number
    if re.search(r"(?:\b(p\.|page|sec(?:tion)?|schedule))\s*$", before, re.IGNORECASE):
        return True

    # ID/URL/path/file-like context
    if re.search(r"(https?://|\w+\.\w{2,}|/|\\)", window, re.IGNORECASE):
        return True
    if re.search(r"[A-Z]{2,}-[A-Z]{2,}-?\d+", window):  # e.g., DIR-CPO-5139
        return True
    if re.search(r"\+\d", window):  # e.g., DIR+1, gdit.com+15
        return True
    if re.search(r"\.\w{2,4}\b", window):  # .pdf, .docx, etc.
        return True

    # compound modifier right after digit: 90-day, 30-minute
    if after[:1] in {"-", "–"}:
        return True

    # %/million/billion prefer numerals unless sentence start
    nxt = next_token_after(s, end)
    if nxt in UNIT_WORDS:
        if allow_sentence_start_override and is_sentence_start(s, start):
            return False  # convert at sentence start
        return True

    return False

def format_with_commas_if_needed(num_str: str) -> Optional[str]:
    """Add commas for 4+ digit integers (US style) unless it's a date/year context."""
    if ',' in num_str or not num_str.isdigit():
        return None
    if len(num_str) < 4:
        return None
    return f"{int(num_str):,}"

Patch = Tuple[int, int, str]

def apply_patches(text: str, patches: List[Patch]) -> str:
    if not patches:
        return text
    patches.sort(key=lambda p: p[0], reverse=True)  # apply right-to-left
    out = text
    for start, end, repl in patches:
        out = out[:start] + repl + out[end:]
    return out

# HEADING SUPPORT

_HEADING_NAME_BY_LEVEL = {
    1: "Heading 1",
    2: "Heading 2",
    3: "Heading 3",
    4: "Heading 4",
}

def _normalize_rgb(rgb):
    """Accept 'RGB 38-111-160', '38-111-160', or a tuple/list (38,111,160). Return 'RGB 38-111-160'."""
    if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
        return f"RGB {rgb[0]}-{rgb[1]}-{rgb[2]}"
    s = str(rgb).strip()
    if s.lower().startswith("rgb"):
        return "RGB " + s.split(" ", 1)[-1].replace(",", "-").replace(" ", "").replace("RGB", "").strip()
    s = s.replace(",", "-")
    if s and not s.lower().startswith("rgb"):
        return f"RGB {s}"
    return s

def _style_from_element(elem):
    """
    Extract the style info the checker needs.
    Expected fields (if present):
      elem['style']['font'], ['font_weight'] or ['style_name'], ['font_size'], ['color_rgb'],
      ['spacing_before'], ['spacing_after'].
    Also accept top-level convenience fields if your normalizer exposes them.
    """
    st = elem.get("style", {}) or {}
    return {
        "font": st.get("font") or elem.get("font"),
        "font_name_joined": (st.get("font_name_joined")
                             or st.get("style_name")
                             or " ".join(x for x in [st.get("font") or elem.get("font"),
                                                     st.get("font_weight") or elem.get("font_weight"),
                                                     st.get("font_style") or elem.get("font_style")] if x)),
        "size": st.get("font_size") or elem.get("font_size"),
        "color": _normalize_rgb(st.get("color_rgb") or elem.get("color_rgb")),
        "spacing_before": st.get("spacing_before") or elem.get("spacing_before"),
        "spacing_after": st.get("spacing_after") or elem.get("spacing_after"),
    }

def _expected_heading_spec(level_or_name: int | str):
    """Return (name, spec) from amida_rules.heading_styles, or (None, None) if unknown."""
    if isinstance(level_or_name, int):
        name = _HEADING_NAME_BY_LEVEL.get(level_or_name)
    else:
        name = level_or_name
    if not name:
        return None, None
    spec = amida_rules.heading_styles.get(name)
    return name, spec

def enrich_document_with_headings(document: dict) -> dict:
    """
    Map 'formatting' into 'style' and infer heading_level/heading_name heuristically
    so the headings rule can run even if upstream doesn’t label headings explicitly.
    """
    AMIDA_BLUE = (38, 111, 160)
    BLACK = (0, 8, 14)

    def tuple_from_color(c):
        if not c:
            return None
        s = str(c).lower().replace(",", "-")
        if s.startswith("rgb"):
            s = s.split(" ", 1)[-1]
        parts = [p for p in s.split("-") if p]
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            return None
        return tuple(int(p) for p in parts)

    for page in document.get("pages", []):
        for elem in page.get("elements", []):
            fmt = elem.get("formatting") or {}

            # Mirror into style expected by checker
            elem["style"] = {
                "font": fmt.get("font_name") or elem.get("font") or "Lato",
                "font_size": fmt.get("font_size"),
                "font_weight": ("Bold" if fmt.get("is_bold") else "Regular"),
                "font_style": ("Italic" if fmt.get("is_italic") else None),
                "color_rgb": fmt.get("color"),
                "style_name": " ".join(x for x in [
                    fmt.get("font_name") or elem.get("font") or "Lato",
                    "Bold" if fmt.get("is_bold") else None,
                    "Italic" if fmt.get("is_italic") else None,
                ] if x).strip(),
            }

            # Heuristic heading inference
            size = fmt.get("font_size")
            is_bold = bool(fmt.get("is_bold"))
            is_italic = bool(fmt.get("is_italic"))
            col = tuple_from_color(fmt.get("color"))

            level = None
            if size is not None:
                if size >= 20 and is_bold and (col in (AMIDA_BLUE, None)):
                    level = 1
                elif size >= 16 and not is_bold and not is_italic and col == AMIDA_BLUE:
                    level = 2
                elif 10 <= size <= 12 and is_bold and not is_italic and (col in (BLACK, None)):
                    level = 3
                elif 10 <= size <= 12 and is_italic and not is_bold and (col in (BLACK, None)):
                    level = 4

            if level:
                elem["heading_level"] = level
                elem["heading_name"] = _HEADING_NAME_BY_LEVEL[level]

    return document

# COLLECTORS

def collect_numeral_spelling_patches(text: str) -> List[Patch]:
    """Spell out numbers 1..99 with guards; includes sentence-start override."""
    rule = next(r for r in amida_rules.get_pattern_rules() if r.name == "numeral_spelling")
    pattern = re.compile(rule.pattern)
    patches: List[Patch] = []

    for m in pattern.finditer(text):
        num_str = m.group(1)
        start, end = m.span(1)

        if looks_like_date_or_year(text, start, end):
            continue

        at_sentence_start = is_sentence_start(text, start)
        if not at_sentence_start and should_skip_numeric_context(text, start, end, allow_sentence_start_override=True):
            continue

        try:
            n = int(num_str)
        except ValueError:
            continue

        if 1 <= n <= 99:
            patches.append((start, end, num_to_words_1_to_99(n)))

    return patches

def collect_thousands_comma_patches(text: str) -> List[Patch]:
    """Insert commas in 4+ digit integers with guards for dates/IDs/URLs/etc."""
    patches: List[Patch] = []
    for m in re.finditer(r"\b(\d{4,})\b", text):
        start, end = m.span(1)
        token = m.group(1)

        if looks_like_date_or_year(text, start, end):
            continue
        if word_immediately_before(text, start) in DATE_MONTHS:
            continue
        window = text[max(0, start-24):min(len(text), end+24)]
        if re.search(r"(https?://|\w+\.\w{2,}|/|\\)", window, re.IGNORECASE):
            continue
        if re.search(r"[A-Z]{2,}-[A-Z]{2,}-?\d+", window):
            continue
        if re.search(r"\+\d", window):
            continue
        if re.search(r"\.\w{2,4}\b", window):
            continue

        repl = format_with_commas_if_needed(token)
        if repl and repl != token:
            patches.append((start, end, repl))

    return patches

def collect_single_space_after_period_patches(text: str) -> List[Patch]:
    """Exactly one space after sentence-ending punctuation (. ? ! … or '...')."""
    patches: List[Patch] = []
    for m in re.finditer(r"(?:…|\.{3}|[.!?]) {2,}", text):
        start, end = m.span()
        if text[start:start+3] == "...":
            punct_only = "..."
        elif text[start] in ".!?":
            punct_only = text[start]
        else:
            punct_only = "…"
        patches.append((start, end, punct_only + " "))
    return patches

CLOSE_QUOTES = ['"', "”", "’", "'"]
OPEN_QUOTES  = ['"', "“", "‘", "'"]

def collect_quotation_punctuation_patches(text: str) -> List[Patch]:
    """
    Mechanical fixes only:
    1) Move comma/period INSIDE a closing quote.
    2) Move semicolon/colon OUTSIDE a closing quote.
    """
    patches: List[Patch] = []

    pattern_outside = re.compile(rf"([{''.join(map(re.escape, CLOSE_QUOTES))}])\s*([.,])")
    for m in pattern_outside.finditer(text):
        q, punct = m.group(1), m.group(2)
        s, e = m.span()
        patches.append((s, e, f"{punct}{q}"))

    pattern_inside = re.compile(rf"([;:])\s*([{''.join(map(re.escape, CLOSE_QUOTES))}])")
    for m in pattern_inside.finditer(text):
        punct, q = m.group(1), m.group(2)
        s, e = m.span()
        patches.append((s, e, f"{q}{punct}"))

    return patches

def collect_word_list_patches(text: str) -> List[Patch]:
    """
    Apply Amida's Word List preferences:
      - Simple substitutions (built in -> built-in, healthcare, etc.)
      - end user / end-user (noun vs adjective)
      - open source / open-source (noun vs adjective)
      - set up / set-up (verb vs noun)
      - ensure comma after e.g. / i.e.
    """
    patches: List[Patch] = []
    wl = getattr(amida_rules, "word_list", {}) or {}

    # simple one-to-one replacements
    simple_map: Dict[str, str] = {k: v for k, v in wl.items() if isinstance(v, str)}
    if simple_map:
        keys_sorted = sorted(simple_map.keys(), key=len, reverse=True)
        pattern = r"\b(" + "|".join(re.escape(k) for k in keys_sorted) + r")\b"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = m.span(1)
            matched = m.group(1)
            key_lower = matched.lower()
            repl = simple_map.get(key_lower)
            if repl and repl != matched:
                patches.append((start, end, repl))

    # e.g. / i.e. must be followed by a comma
    for m in re.finditer(r"\b(e\.g\.|i\.e\.)\s*(?!,)", text, flags=re.IGNORECASE):
        start, end = m.span()
        abbr = m.group(1)
        patches.append((start, end, abbr + ", "))

    # end user / end-user
    for m in re.finditer(r"\bend\s+user\b(?=\s+[A-Za-z])", text, flags=re.IGNORECASE):
        s, e = m.span()
        patches.append((s, e, "end-user"))
    for m in re.finditer(r"\bend-?user\b(?!\s+[A-Za-z])", text, flags=re.IGNORECASE):
        s, e = m.span()
        patches.append((s, e, "end user"))

    # open source / open-source
    for m in re.finditer(r"\bopen\s+source\b(?=\s+[A-Za-z])", text, flags=re.IGNORECASE):
        s, e = m.span()
        patches.append((s, e, "open-source"))
    for m in re.finditer(r"\bopen-?source\b(?!\s+[A-Za-z])", text, flags=re.IGNORECASE):
        s, e = m.span()
        patches.append((s, e, "open source"))

    # set up / set-up
    for m in re.finditer(r"\bset-up\b(?=\s+\w)", text, flags=re.IGNORECASE):
        s, e = m.span()
        patches.append((s, e, "set up"))
    for m in re.finditer(r"\b(the|a|an)\s+set\s+up\b", text, flags=re.IGNORECASE):
        s, e = m.span()
        determiner = m.group(1)
        patches.append((s, e, f"{determiner} set-up"))

    return patches

# TITLES RULE HELPERS

_SMALL_WORDS = {
    "a","an","the","and","but","or","nor","so","yet",
    "as","at","by","for","in","of","off","on","per","to","up","via",
    "v.","vs.","v","vs"
}

def _cap_first_letter(token: str) -> str:
    if token.isupper():
        return token
    if "-" in token:
        return "-".join(_cap_first_letter(part) for part in token.split("-"))
    if "'" in token:
        left, right = token.split("'", 1)
        return left[:1].upper() + left[1:].lower() + "'" + (right[:1].upper() + right[1:].lower() if right else "")
    return token[:1].upper() + token[1:].lower()

def titlecase_transform(s: str) -> str:
    tokens = re.findall(r"\w[\w'/-]*|\s+|[^\w\s]", s, flags=re.UNICODE)
    words = [t for t in tokens if re.match(r"\w", t)]
    if not words:
        return s
    result = []
    word_positions = [i for i, t in enumerate(tokens) if re.match(r"\w", t)]
    first_pos, last_pos = word_positions[0], word_positions[-1]
    for i, t in enumerate(tokens):
        if not re.match(r"\w", t):
            result.append(t); continue
        is_first = i == first_pos
        is_last  = i == last_pos
        base = t
        if is_first or is_last:
            result.append(_cap_first_letter(base)); continue
        if len(base) > 3 or base.lower() not in _SMALL_WORDS:
            result.append(_cap_first_letter(base))
        else:
            result.append(base.lower())
    return "".join(result)

def is_likely_title_element(elem: dict, page_idx, element_index) -> bool:
    if (elem.get("role") or "").lower() == "title":
        return True
    if elem.get("heading_level") == 1 or (elem.get("heading_name") == "Heading 1"):
        return True
    st = elem.get("style") or {}
    size = st.get("font_size") or elem.get("font_size")
    weight = (st.get("font_weight") or elem.get("font_weight") or "").lower()
    # Treat big bold text on any slide as a likely title/section header
    if size and isinstance(size, (int, float)) and size >= 18 and ("bold" in weight):
        return True
    return False

def collect_titles_patches(text: str, elem: dict, page_idx, element_index):
    if not text or not is_likely_title_element(elem, page_idx, element_index):
        return []
    proposed = titlecase_transform(text)
    if proposed != text:
        return [(0, len(text), proposed)]
    return []

def collect_inline_document_title_issues(text: str, slide_idx, element_index):
    issues = []
    for m in re.finditer(r"[\"“]([A-Za-z0-9][^\"”]{4,}?[A-Za-z0-9])[\”\"]", text):
        phrase = m.group(1)
        if len(phrase.split()) >= 2:
            s, e = m.span()
            suggestion = text[:s] + phrase + text[e:]
            if suggestion != text:
                issues.append({
                    "rule_name": "titles",
                    "severity": "suggestion",
                    "category": "grammar",
                    "description": "Document titles should be italicized, not quoted. Remove quotation marks and apply italics via formatting.",
                    "location": f"slide {slide_idx} • element {element_index if element_index is not None else '?'}",
                    "found_text": text,
                    "suggestion": suggestion,
                    "page_or_slide_index": slide_idx,
                    "element_index": element_index,
                })
    return issues


def collect_heading_style_issues_for_element(text: str, elem: dict, slide_idx, element_index) -> list[dict]:
    """
    Emit a *separate* issue when a heading's style doesn't match the guide.
    Expects your normalized element to include either:
      - elem['heading_level'] in {1,2,3,4}, OR
      - elem['role'] in {'heading1','heading2','heading3','heading4'} OR
      - elem['heading_name'] exactly "Heading 1" .. "Heading 4"
    If none are present, we skip silently.
    """
    hl = elem.get("heading_level")
    role = (elem.get("role") or "").lower()
    hname = elem.get("heading_name")

    if not hl:
        if role.startswith("heading") and role[-1:].isdigit():
            hl = int(role[-1])
    if not hname and hl in _HEADING_NAME_BY_LEVEL:
        hname = _HEADING_NAME_BY_LEVEL[hl]

    heading_name, expected = _expected_heading_spec(hname or hl)
    if not expected:
        return []

    actual = _style_from_element(elem)

    deltas = []
    if expected.get("font") and expected["font"] != actual["font_name_joined"]:
        deltas.append(f"font '{actual['font_name_joined'] or 'unknown'}' → '{expected['font']}'")

    exp_color = _normalize_rgb(expected.get("color"))
    if exp_color and exp_color != actual["color"]:
        deltas.append(f"color {actual['color'] or 'unknown'} → {exp_color}")

    if expected.get("size") and expected["size"] != actual["size"]:
        deltas.append(f"size {actual['size'] or 'unknown'} → {expected['size']}")

    exp_b = expected.get("spacing", {}).get("before")
    exp_a = expected.get("spacing", {}).get("after")
    if exp_b is not None and exp_b != actual["spacing_before"]:
        deltas.append(f"spacing-before {actual['spacing_before'] or 'unknown'} → {exp_b}")
    if exp_a is not None and exp_a != actual["spacing_after"]:
        deltas.append(f"spacing-after {actual['spacing_after'] or 'unknown'} → {exp_a}")

    if not deltas:
        return []

    description = (
        f"{heading_name} style mismatch: " + "; ".join(deltas) +
        ". Expected per Amida Style Guide (Headings)."
    )

    return [{
        "rule_name": "headings",
        "severity": "warning",
        "category": "grammar",
        "description": description,
        "location": f"slide {slide_idx} • element {element_index if element_index is not None else '?'}",
        "found_text": text,
        "suggestion": text,
        "page_or_slide_index": slide_idx,
        "element_index": element_index,
    }]

# BULLETS RULE HELPERS & COLLECTORS

_BULLET_MARKERS = r"[•●▪︎◦\-–—*]"

# Words that imply ordering (used to nudge toward numbered lists)
_ORDINAL_WORDS = {
    "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
    "eleventh","twelfth","thirteenth","fourteenth","fifteenth","sixteenth","seventeenth",
    "eighteenth","nineteenth","twentieth","next","then","finally","last"
}

# Abbreviations after which a period should NOT be treated as a sentence break
_NO_SENTENCE_BREAK_ABBR = {"e.g.", "i.e.", "etc.", "vs.", "v.", "al.", "Mr.", "Mrs.", "Ms.", "Dr."}


def _extract_bullet_segments(text: str, elem: dict):
    """
    Return list of (start, end, line_text) to treat as bullet content.

    If the normalizer already marks this element as a bullet
    (locator.element_type == 'bullet'), treat the entire text as one bullet.
    Otherwise, fall back to glyph/number detection line-by-line.
    """
    loc = elem.get("locator") or {}
    if (loc.get("element_type") or "").lower() == "bullet":
        return [(0, len(text), text)]

    # Fallback: detect bullet/numbered lines inside multi-line text
    lines = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\n\r")
        m_bullet = re.match(rf"^\s*(?:{_BULLET_MARKERS})\s+(.*)$", line)
        m_numbered = re.match(r"^\s*(?:\(?\d+\)?[.)]|\d+\))\s+(.*)$", line)
        if m_bullet or m_numbered:
            inner = (m_bullet or m_numbered).group(1)
            start = offset + (m_bullet.start(1) if m_bullet else m_numbered.start(1))
            end   = offset + (m_bullet.end(1)   if m_bullet else m_numbered.end(1))
            lines.append((start, end, inner))
        offset += len(raw_line)
    return lines


def _capitalize_first_alpha(s: str) -> str:
    """Capitalize the first alphabetic character (leave leading symbols/spaces alone)."""
    for i, ch in enumerate(s):
        if ch.isalpha():
            if ch.islower():
                return s[:i] + ch.upper() + s[i+1:]
            break
    return s


def _looks_numbered_lead(s: str) -> bool:
    """Heuristic: bullet looks like part of a sequence (so numbered list may be better)."""
    t = s.lstrip()
    if re.match(r"^(?:\(?\d+\)?[.)]|\d+\))\s+", t):  # 1., (1), 1)
        return True
    first_word = re.match(r"^[A-Za-z'-]+", t)
    if first_word and first_word.group(0).lower() in _ORDINAL_WORDS:
        return True
    if re.match(r"^step\s+\d+\b", t, flags=re.IGNORECASE):
        return True
    return False


def _remove_terminal_period(s: str) -> str:
    """
    Remove a terminal period for single-sentence bullets.
    Preserve quotes/parentheses around the final word, e.g., …word). -> …word)
    """
    # Strip whitespace at end but remember it
    trail_ws = re.search(r"\s*$", s).group(0)
    core = s[:len(s)-len(trail_ws)]

    # Acceptable end wrappers
    wrappers = r"[\"'”’)\]]*"
    if re.search(rf"{wrappers}\.$", core):
        return re.sub(rf"({wrappers})\.$", r"\1", core) + trail_ws
    return s


def _replace_internal_sentence_periods_with_semicolons(s: str) -> str:
    """
    If a single bullet contains multiple sentences, replace internal breaks with semicolons.
    Avoid replacing periods that are part of known abbreviations or numbers.
    """
    # Quick bail: if there's no obvious additional sentence boundary, return as-is
    if not re.search(r"[.!?]\s+\S", s):
        return s

    def _should_keep_period(prev_token: str) -> bool:
        # Keep if previous token ends with a protected abbreviation
        for ab in _NO_SENTENCE_BREAK_ABBR:
            if prev_token.endswith(ab):
                return True
        # Keep if the previous token looks like a number or decimal
        if re.search(r"\d\.$", prev_token):
            return True
        return False

    tokens = re.findall(r"\S+|\s+", s)
    out = []
    for i, tok in enumerate(tokens):
        # Look for a word that ends with '.' followed by at least one space and then a word-char
        if tok.endswith(".") and i + 1 < len(tokens) and re.match(r"\s+$", tokens[i+1]):
            nxt = tokens[i+2] if i + 2 < len(tokens) else ""
            if nxt and re.match(r"\w", nxt):
                prev_token = tok
                if not _should_keep_period(prev_token):
                    # Replace terminal '.' with ';'
                    tok = tok[:-1] + ";"
        out.append(tok)
    return "".join(out)


def _apply_bullet_transforms(line_text: str) -> str:
    """
    Apply bullet mechanics:
      - Capitalize first letter
      - Replace internal sentence breaks with semicolons (heuristic)
      - Remove a terminal period
    """
    t = line_text.strip("\n\r")
    t = _capitalize_first_alpha(t)
    t = _replace_internal_sentence_periods_with_semicolons(t)
    t = _remove_terminal_period(t)
    return t


def collect_bullets_issue(text: str, elem: dict, slide_idx, element_index) -> Optional[Dict[str, Any]]:
    """
    Enforce bullet guidance (Grammar → Bullets):
      - No period at the end of a bullet
      - If a single bullet contains multiple sentences, separate with semicolons
      - Capitalize the first letter of the first word (unless case-sensitive; heuristic)
      - Suggest numbered list when the bullet implies sequence/ranking
    Returns one (or two) StyleIssue dicts, or None.
    """
    segments = _extract_bullet_segments(text, elem)
    if not segments:
        return None

    # If the element contains multiple detected bullet lines, transform each; else transform whole text.
    if len(segments) > 1:
        patches: List[Tuple[int, int, str]] = []
        for start, end, line in segments:
            repl = _apply_bullet_transforms(line)
            if repl != line:
                patches.append((start, end, repl))
        if not patches:
            issue1 = None
        else:
            suggestion_text = apply_patches(text, patches)
            issue1 = {
                "rule_name": "bullets",
                "severity": "suggestion",
                "category": "grammar",
                "description": (
                    "Bullets: capitalize the first word; replace internal sentence breaks with semicolons; "
                    "avoid periods at the end of bullet points."
                ),
                "location": f"slide {slide_idx} • element {element_index if element_index is not None else '?'}",
                "found_text": text,
                "suggestion": suggestion_text,
                "page_or_slide_index": slide_idx,
                "element_index": element_index,
            }
    else:
        # Single segment (typical normalized bullet)
        start, end, line = segments[0]
        repl = _apply_bullet_transforms(line)
        if repl == line:
            issue1 = None
        else:
            issue1 = {
                "rule_name": "bullets",
                "severity": "suggestion",
                "category": "grammar",
                "description": (
                    "Bullets: capitalize the first word; replace internal sentence breaks with semicolons; "
                    "avoid periods at the end of bullet points."
                ),
                "location": f"slide {slide_idx} • element {element_index if element_index is not None else '?'}",
                "found_text": text,
                "suggestion": text[:start] + repl + text[end:],
                "page_or_slide_index": slide_idx,
                "element_index": element_index,
            }

    # Secondary nudge: suggest numbered list if the bullet likely represents a sequence/ranking
    text_to_check = segments[0][2] if segments else text
    sequence_hint = _looks_numbered_lead(text_to_check)
    issue2 = None
    if sequence_hint:
        issue2 = {
            "rule_name": "bullets",
            "severity": "suggestion",
            "category": "grammar",
            "description": "Bullets that indicate a sequence or ranking should be formatted as a numbered list.",
            "location": f"slide {slide_idx} • element {element_index if element_index is not None else '?'}",
            "found_text": text,
            "suggestion": text,  # structural advice; text stays the same
            "page_or_slide_index": slide_idx,
            "element_index": element_index,
        }

    return (issue1, issue2)

# # ACRONYMS RULE (Grammar → "acronyms")

# # Matches “Full Phrase (ABC)” definitions
# _ACRO_DEF_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9 .&/\-]+?)\s*\(\s*([A-Z]{2,})\s*\)")
# # Matches standalone acronyms (ABC or dotted A.B.C.)
# _ACRO_TOKEN_RE = re.compile(r"\b(?:([A-Z]{2,})|([A-Z](?:\.[A-Z])+\.))\b")
# # Normalize dotted forms like U.S. → US
# def _normalize_acronym(token: str) -> str:
#     return token.replace(".", "")

# # VA article misuse: “the VA” → “VA”
# _VA_ARTICLE_RE = re.compile(r"\b([Tt])he\s+(VA)\b")


# def _collect_acronym_definitions_with_offsets(text: str) -> dict[str, int]:
#     """
#     Return a map {ACR: offset_of_definition_open_paren} for any “Full Phrase (ACR)”.
#     If multiple definitions exist in the same element, keep the earliest offset.
#     """
#     defs: dict[str, int] = {}
#     for m in _ACRO_DEF_RE.finditer(text):
#         acr = m.group(2).upper()
#         # offset of the '(' that starts the acronym
#         offset = m.start(2)
#         if acr not in defs or offset < defs[acr]:
#             defs[acr] = offset
#     return defs


# def _collect_acronym_usages_with_offsets(text: str) -> list[tuple[str, int]]:
#     """
#     Return list of (ACR, start_offset) for each standalone acronym usage.
#     Dotted acronyms are normalized to undotted (e.g., U.S. → US).
#     """
#     uses: list[tuple[str, int]] = []
#     for m in _ACRO_TOKEN_RE.finditer(text):
#         token = m.group(1) or m.group(2) or ""
#         acr = _normalize_acronym(token)
#         # Skip single-letter or false positives (already guarded by regex)
#         if len(acr) >= 2 and acr.isupper():
#             uses.append((acr, m.start()))
#     return uses


# def _collect_va_article_patches(text: str) -> List[Patch]:
#     """
#     Remove the article before 'VA' when used as an acronym for the
#     Department of Veterans Affairs: 'the VA' → 'VA'.
#     """
#     patches: List[Patch] = []
#     for m in _VA_ARTICLE_RE.finditer(text):
#         s, e = m.span()
#         # Replace the whole 'the VA' / 'The VA' with just 'VA'
#         patches.append((s, e, "VA"))
#     return patches


# def collect_acronym_issues(text: str,
#                            elem: dict,
#                            slide_idx,
#                            element_index,
#                            defined_globally: set[str]) -> tuple[list[dict], set[str], List[Patch]]:
#     """
#     Enforce Acronyms rule:
#       - Full phrase must precede the acronym on first use: 'Full Phrase (ACR)'
#       - If acronym appears before its definition (in the same element or earlier in doc), flag it
#       - Special case: remove 'the' before 'VA' when VA is used as the acronym
#     Returns (issues, newly_defined_acronyms_in_this_element, va_patches).
#     """
#     issues: list[dict] = []

#     # In-element definitions and uses
#     defs_here = _collect_acronym_definitions_with_offsets(text)     # {ACR: def_offset}
#     uses_here = _collect_acronym_usages_with_offsets(text)          # [(ACR, use_offset), ...]

#     # 1) Check for first-use violations within this element
#     flagged_this_element: set[str] = set()
#     for acr, use_off in uses_here:
#         # Skip 'VA' article handling here; that’s handled via patches below
#         # If acronym has a definition in the same element, it must occur BEFORE first use
#         if acr in defs_here and use_off < defs_here[acr]:
#             if acr not in flagged_this_element:
#                 issues.append({
#                     "rule_name": "acronyms",
#                     "severity": "suggestion",
#                     "category": "grammar",
#                     "description": f"Define '{acr}' on first use as 'Full Phrase ({acr})' before using the acronym.",
#                     "location": f"slide {slide_idx} • element {element_index if element_index is not None else '?'}",
#                     "found_text": text,
#                     "suggestion": text,  # structural guidance; not auto-editable without knowing the full phrase
#                     "page_or_slide_index": slide_idx,
#                     "element_index": element_index,
#                 })
#                 flagged_this_element.add(acr)
#         # If no definition here AND not defined previously in the document, flag it
#         elif acr not in defs_here and acr not in defined_globally and acr not in flagged_this_element:
#             issues.append({
#                 "rule_name": "acronyms",
#                 "severity": "suggestion",
#                 "category": "grammar",
#                 "description": f"Spell out '{acr}' on first use followed by the acronym in parentheses (e.g., 'Full Phrase ({acr})').",
#                 "location": f"slide {slide_idx} • element {element_index if element_index is not None else '?'}",
#                 "found_text": text,
#                 "suggestion": text,
#                 "page_or_slide_index": slide_idx,
#                 "element_index": element_index,
#             })
#             flagged_this_element.add(acr)

#     # 2) Special case: “the VA” → “VA”
#     va_patches = _collect_va_article_patches(text)

#     # 3) Newly defined acronyms in this element become available for subsequent elements
#     newly_defined = set(defs_here.keys())

#     return issues, newly_defined, va_patches


# ISSUE BUILDER & MAIN CHECK

def build_rule_issue(rule_name: str, description: str, severity: str, category: str,
                     slide_idx: Any, element_index: Any, found_text: str,
                     patches: List[Patch]) -> Optional[Dict[str, Any]]:
    if not patches:
        return None
    suggestion_text = apply_patches(found_text, patches)
    if suggestion_text == found_text:
        return None
    return {
        "rule_name": rule_name,
        "severity": severity,
        "category": category,
        "description": description,
        "location": f"slide {slide_idx} • element {element_index if element_index is not None else '?'}",
        "found_text": found_text,
        "suggestion": suggestion_text,
        "page_or_slide_index": slide_idx,
        "element_index": element_index,
    }

def check_document(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Normalize/annotate headings so the headings rule can run
    document = enrich_document_with_headings(document)

    numeral_rule = next(r for r in amida_rules.get_pattern_rules() if r.name == "numeral_spelling")
    severity = numeral_rule.severity

    issues: List[Dict[str, Any]] = []
    defined_acronyms_global: set[str] = set()

    for page in document.get("pages", []):
        slide_idx = page.get("index")
        for elem in page.get("elements", []):
            text = elem.get("text", "") or ""
            loc = elem.get("locator", {}) or {}
            element_index = loc.get("element_index")


        #     acro_issues, newly_defined, va_patches = collect_acronym_issues(
        #        text, elem, slide_idx, element_index, defined_acronyms_global
        #   )
            # if acro_issues:
            #     issues.extend(acro_issues)
            # if va_patches:
            #     va_issue = build_rule_issue(
            #         rule_name="acronyms",
            #         description="For 'Department of Veterans Affairs (VA)', do not use an article before 'VA' (e.g., 'report for VA').",
            #         severity="suggestion",
            #         category="grammar",
            #         slide_idx=slide_idx,
            #         element_index=element_index,
            #         found_text=text,
            #         patches=va_patches
            #     )
            #     if va_issue:
            #         issues.append(va_issue)
            # # Make new definitions available to the rest of the document
            # defined_acronyms_global |= newly_defined


            # Build patches per rule on ORIGINAL text
            numeral_patches: List[Patch] = []
            numeral_patches += collect_numeral_spelling_patches(text)
            numeral_patches += collect_thousands_comma_patches(text)

            spacing_patches = collect_single_space_after_period_patches(text)
            quote_patches   = collect_quotation_punctuation_patches(text)
            word_list_patches = collect_word_list_patches(text)

            # HEADINGS: emit separate issues; no text edits
            heading_issues = collect_heading_style_issues_for_element(text, elem, slide_idx, element_index)
            issues.extend(heading_issues)
            # Titles: Title Case for likely titles
            titles_patches = collect_titles_patches(text, elem, slide_idx, element_index)
            titles_issue = build_rule_issue(
                rule_name="titles",
                description="Use Title Case for titles: capitalize the first and last words and all main words; any word longer than three letters is capitalized.",
                severity="suggestion",
                category="grammar",
                slide_idx=slide_idx,
                element_index=element_index,
                found_text=text,
                patches=titles_patches
            )
            if titles_issue:
                issues.append(titles_issue)

            # Titles: inline references to document titles should not be in quotes
            issues.extend(collect_inline_document_title_issues(text, slide_idx, element_index))

            # Bullets: enforce bullet mechanics + optional numbered-list nudge
            bullets_pair = collect_bullets_issue(text, elem, slide_idx, element_index)
            if bullets_pair:
                issue_main, issue_seq = bullets_pair
                if issue_main:
                    issues.append(issue_main)
                if issue_seq:
                    issues.append(issue_seq)

            # Emit one issue per rule if any patches exist
            num_issue = build_rule_issue(
                rule_name="numerals",
                description="Spell out numbers <100 (and at sentence start). Keep numerals for percent/million/billion unless sentence start. Use commas for 4+ digits (except dates/IDs/URLs/years).",
                severity=severity,
                category="grammar",
                slide_idx=slide_idx,
                element_index=element_index,
                found_text=text,
                patches=numeral_patches
            )
            if num_issue:
                issues.append(num_issue)

            spacing_issue = build_rule_issue(
                rule_name="period_single_space",
                description="Use exactly one space after sentence-ending punctuation (. ? ! … or '...').",
                severity=severity,
                category="grammar",
                slide_idx=slide_idx,
                element_index=element_index,
                found_text=text,
                patches=spacing_patches
            )
            if spacing_issue:
                issues.append(spacing_issue)

            quote_issue = build_rule_issue(
                rule_name="quotes_punctuation",
                description="Place commas/periods inside closing quotes; place semicolons/colons outside.",
                severity=severity,
                category="grammar",
                slide_idx=slide_idx,
                element_index=element_index,
                found_text=text,
                patches=quote_patches
            )
            if quote_issue:
                issues.append(quote_issue)

            word_issue = build_rule_issue(
                rule_name="word_list",
                description=(
                    "Apply Amida Word List preferences: hyphenation & compounds (e.g., 'built-in', "
                    "'cybersecurity'), noun/adj distinctions ('end user' vs 'end-user', "
                    "'open source' vs 'open-source'), 'set up' (verb) vs 'set-up' (noun), "
                    "capitalize Machine Learning/Artificial Intelligence/Natural Language Processing, "
                    "and ensure comma after e.g./i.e."
                ),
                severity=severity,
                category="word-list",
                slide_idx=slide_idx,
                element_index=element_index,
                found_text=text,
                patches=word_list_patches
            )
            if word_issue:
                issues.append(word_issue)

    return issues

# entrypoint

def main():
    # Usage:
    #   python -m backend.analyzers.simple_style_checker <normalized.json> [out.json]
    #   or: python simple_style_checker.py <normalized.json> [out.json]
    if len(sys.argv) not in (2, 3):
        print("Usage: python -m backend.analyzers.simple_style_checker <normalized.json> [out.json]")
        print("   or: python simple_style_checker.py <normalized.json> [out.json]")
        sys.exit(2)

    in_path = sys.argv[1]
    if not os.path.isfile(in_path):
        print(f"Input not found: {in_path}")
        sys.exit(1)

    # Default output in the SAME folder as the input:
    if len(sys.argv) == 3:
        out_path = sys.argv[2]
    else:
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_dir = os.path.dirname(os.path.abspath(in_path))
        out_path = os.path.join(out_dir, f"{base}_style_issues.json")

    # Read, analyze, write
    with open(in_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    issues = check_document(doc)

    # Save to file (same folder by default)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    # Also echo to stdout for quick inspection
    print(json.dumps(issues, ensure_ascii=False, indent=2))
    print(f"\nSaved issues to: {out_path}")

if __name__ == "__main__":
    main()

