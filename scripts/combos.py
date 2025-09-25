# ===============================
# File: scripts/combos.py
# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import List

COMBO_SEP_RX = re.compile(r"\s*(?:\+|/| with )\s*")


def split_combo_segments(s: str) -> List[str]:
    if not isinstance(s, str) or not s:
        return []
    parts = [p.strip() for p in COMBO_SEP_RX.split(s) if p.strip()]
    return [re.sub(r"\s+", " ", p) for p in parts]


def looks_like_combination(s_norm: str, pnf_hit_count: int, who_hit_count: int) -> bool:
    if pnf_hit_count > 1:
        return True
    if who_hit_count > 1:
        return True
    dosage_ratio_rx = re.compile(r"""
        \b
        \d+(?:[\.,]\d+)?\s*(?:mg|g|mcg|ug|iu)
        \s*/\s*
        (?:\d+(?:[\.,]\d+)?\s*)?(?:ml|l)
        \b
    """, re.IGNORECASE | re.VERBOSE)
    s_masked = dosage_ratio_rx.sub(" <DOSE> ", s_norm)
    if re.search(r"\bwith\b", s_masked):
        return True
    if "+" in s_masked:
        return True
    if re.search(r"[a-z]\s*/\s*[a-z]", s_masked):
        return True
    return len(split_combo_segments(s_masked)) >= 2