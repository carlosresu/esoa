# ===============================
# File: scripts/aho.py
# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import ahocorasick  # type: ignore

from .text_utils import normalize_compact, normalize_text


def build_molecule_automata(pnf_df) -> Tuple[ahocorasick.Automaton, ahocorasick.Automaton]:
    A_norm = ahocorasick.Automaton()
    A_comp = ahocorasick.Automaton()
    seen_norm = set()
    seen_comp = set()
    for gid, gname in pnf_df[["generic_id", "generic_name"]].drop_duplicates().itertuples(index=False):
        key_norm = normalize_text(gname)
        key_comp = normalize_compact(gname)
        if key_norm and (gid, key_norm) not in seen_norm:
            A_norm.add_word(key_norm, (gid, gname)); seen_norm.add((gid, key_norm))
        if key_comp and (gid, key_comp) not in seen_comp:
            A_comp.add_word(key_comp, (gid, gname)); seen_comp.add((gid, key_comp))
    if "synonyms" in pnf_df.columns:
        for gid, syns in pnf_df[["generic_id", "synonyms"]].itertuples(index=False):
            if isinstance(syns, str) and syns.strip():
                for s in syns.split("|"):
                    key_norm = normalize_text(s)
                    key_comp = normalize_compact(s)
                    if key_norm and (gid, key_norm) not in seen_norm:
                        A_norm.add_word(key_norm, (gid, s)); seen_norm.add((gid, key_norm))
                    if key_comp and (gid, key_comp) not in seen_comp:
                        A_comp.add_word(key_comp, (gid, s)); seen_comp.add((gid, key_comp))
    A_norm.make_automaton()
    A_comp.make_automaton()
    return A_norm, A_comp


def scan_pnf_all(text_norm: str, text_comp: str,
                 A_norm: ahocorasick.Automaton,
                 A_comp: ahocorasick.Automaton) -> Tuple[List[str], List[str]]:
    candidates: Dict[str, str] = {}
    for _, (gid, token) in A_norm.iter(text_norm):
        if gid not in candidates or len(token) > len(candidates[gid]):
            candidates[gid] = token
    for _, (gid, token) in A_comp.iter(text_comp):
        if gid not in candidates or len(token) > len(candidates[gid]):
            candidates[gid] = token
    items = sorted(candidates.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    gids = [gid for gid, _ in items]
    tokens = [tok for _, tok in items]
    return gids, tokens