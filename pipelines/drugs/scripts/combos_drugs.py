
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import List

# Treat common pharmaceutical salts/hydrates as formulation modifiers, not separate actives (matches guidance in README under Combination vs. Salt Detection)
SALT_TOKENS = {
    "calcium","sodium","potassium","magnesium","zinc","ammonium","meglumine","aluminum",
    "hydrochloride","nitrate","nitrite","sulfate","sulphate","phosphate","dihydrogen phosphate",
    "hydroxide",
    "acetate","tartrate","fumarate","oxalate","maleate","mesylate","tosylate","besylate","besilate",
    "bitartrate","succinate","citrate","lactate","gluconate","bicarbonate","carbonate",
    "bromide","chloride","iodide","nitrate","selenite","thiosulfate",
    "dihydrate","trihydrate","monohydrate","hydrate","hemihydrate","anhydrous",
    "decanoate","palmitate","stearate","pamoate","benzoate","valerate","propionate",
    "hydrobromide","docusate","hemisuccinate",
}

MEASUREMENT_TOKENS = {
    "ml","l","cc","mg","mcg","ug","g","kg","iu","lsu","mu",
    "meq","meqs","mol","mmol","percent","pct","ratio","per",
}

def _is_measurement_token(token: str) -> bool:
    token = token.lower()
    if token in MEASUREMENT_TOKENS:
        return True
    if token.endswith("ml") or token.endswith("mg"):
        return True
    if token in {"elemental","equivalent","equiv"}:
        return True
    return False

def split_combo_segments(s: str) -> List[str]:
    """Split combo-looking strings using context-aware separators."""
    if not isinstance(s, str) or not s:
        return []

    segments: List[str] = []
    buffer: List[str] = []
    length = len(s)
    i = 0

    def _flush_buffer() -> None:
        if not buffer:
            return
        segment = "".join(buffer).strip()
        if segment:
            segments.append(re.sub(r"\s+", " ", segment))
        buffer.clear()

    def _last_token() -> str:
        if not buffer:
            return ""
        text = "".join(buffer).rstrip()
        if not text:
            return ""
        return re.split(r"[ \t\n\r+/]+", text)[-1]

    def _next_token(start: int) -> str:
        j = start
        while j < length and s[j].isspace():
            j += 1
        k = j
        while k < length and not s[k].isspace():
            if s[k] in "+/":
                break
            k += 1
        return s[j:k]

    def _has_letters(token: str) -> bool:
        return any(ch.isalpha() for ch in token)

    while i < length:
        ch = s[i]
        if ch == "+":
            _flush_buffer()
            i += 1
            continue
        if s.startswith(" with ", i):
            _flush_buffer()
            i += len(" with ")
            continue
        if ch == "/":
            left = _last_token()
            right = _next_token(i + 1)
            if (
                left
                and right
                and _has_letters(left)
                and _has_letters(right)
                and not (left.lower() == "and" and right.lower() == "or")
                and not _is_measurement_token(left)
                and not _is_measurement_token(right)
            ):
                _flush_buffer()
                i += 1
                while i < length and s[i].isspace():
                    i += 1
                continue
        buffer.append(ch)
        i += 1

    _flush_buffer()
    return segments

def _is_salt_tail(segment: str) -> bool:
    """Returns True if segment looks like '<base molecule> <salt>' (optionally hyphenated)."""
    toks = re.split(r"[ \-]+", segment.strip())
    if len(toks) < 2:
        return False
    return toks[-1] in SALT_TOKENS

def _looks_like_oxygen_flow(s_norm: str) -> bool:
    """Identify strings like 'oxygen/liter' or 'oxygen per liter' => not combinations."""
    s = s_norm.lower()
    return bool(re.search(r"\boxygen\s*(?:/|(?:\s+per\s+))\s*(?:l|liter|litre|minute|min|hr|hour|ml)\b", s))

def looks_like_combination(s_norm: str, pnf_hit_count: int, who_hit_count: int) -> bool:
    """Heuristically flag whether a normalized string represents a multi-ingredient product."""
    # Bypass cases
    if _looks_like_oxygen_flow(s_norm):
        return False

    if pnf_hit_count > 1 or who_hit_count > 1:
        return True

    # Mask strength ratios to avoid false '/'
    dosage_ratio_rx = re.compile(r"""
        \b
        \d+(?:[\.,]\d+)?\s*(?:mg|g|mcg|ug|iu|lsu|mu)
        \s*/\s*
        (?:\d+(?:[\.,]\d+)?\s*)?(?:ml|l)
    \b
    """, re.IGNORECASE | re.VERBOSE)
    s_masked = dosage_ratio_rx.sub(" <DOSE> ", s_norm)

    slash_rx = re.compile(r"([a-z]+)\s*/\s*([a-z]+)", re.I)
    for match in slash_rx.finditer(s_masked):
        left, right = match.group(1).lower(), match.group(2).lower()
        if (left == "and" and right == "or") or _is_measurement_token(left) or _is_measurement_token(right):
            continue
        return True

    if "+" in s_masked:
        return True

    if re.search(r"\bwith\b", s_masked):
        # Explicit "with" wording almost always denotes multiple actives.
        return True

    segs = split_combo_segments(s_masked)
    if len(segs) >= 2:
        if len(segs) == 2 and (_is_salt_tail(segs[0]) or _is_salt_tail(segs[1])):
            return False
        # Three or more segments or non-salt pairs indicate a combination product.
        return True

    return False
