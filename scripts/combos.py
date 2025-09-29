
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import List

# Treat common pharmaceutical salts/hydrates as formulation modifiers, not separate actives
SALT_TOKENS = {
    "calcium","sodium","potassium","magnesium","zinc","ammonium",
    "hydrochloride","nitrate","nitrite","sulfate","sulphate","phosphate","dihydrogen phosphate",
    "acetate","tartrate","fumarate","oxalate","maleate","mesylate","tosylate","besylate",
    "bitartrate","succinate","citrate","lactate","gluconate","bicarbonate","carbonate",
    "bromide","chloride","iodide","nitrate","selenite","thiosulfate",
    "dihydrate","trihydrate","monohydrate","hydrate","hemihydrate","anhydrous",
    "decanoate","palmitate","stearate","pamoate","benzoate","valerate","propionate",
    "hydrobromide","docusate","hemisuccinate",
}

COMBO_SEP_RX = re.compile(r"\s*(?:\+|/| with )\s*")

def split_combo_segments(s: str) -> List[str]:
    """Split combo-looking strings on separators while keeping tidy whitespace."""
    if not isinstance(s, str) or not s:
        return []
    parts = [p.strip() for p in COMBO_SEP_RX.split(s) if p.strip()]
    # Collapse internal whitespace to keep downstream token comparisons consistent.
    return [re.sub(r"\s+", " ", p) for p in parts]

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
        \d+(?:[\.,]\d+)?\s*(?:mg|g|mcg|ug|iu|lsu)
        \s*/\s*
        (?:\d+(?:[\.,]\d+)?\s*)?(?:ml|l)
        \b
    """, re.IGNORECASE | re.VERBOSE)
    s_masked = dosage_ratio_rx.sub(" <DOSE> ", s_norm)

    if re.search(r"[a-z]\s*/\s*[a-z]", s_masked):
        segs = split_combo_segments(s_masked)
        if len(segs) == 1 and _is_salt_tail(segs[0]):
            return False
        if len(segs) == 2 and (_is_salt_tail(segs[0]) or _is_salt_tail(segs[1])):
            return False
        # Multiple non-salt spans separated by slash strongly imply a combination.
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
