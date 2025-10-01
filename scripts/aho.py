# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import ahocorasick  # type: ignore

from .text_utils import normalize_compact, normalize_text
from .pnf_aliases import expand_generic_aliases, SPECIAL_GENERIC_ALIASES


def build_molecule_automata(pnf_df) -> Tuple[ahocorasick.Automaton, ahocorasick.Automaton]:
    """Create normalized and compact automatons for PNF generics plus their synonyms."""
    A_norm = ahocorasick.Automaton()
    A_comp = ahocorasick.Automaton()
    seen_norm = set()
    seen_comp = set()
    for gid, gname in pnf_df[["generic_id", "generic_name"]].drop_duplicates().itertuples(index=False):
        alias_set = expand_generic_aliases(gname)
        alias_set.update(SPECIAL_GENERIC_ALIASES.get(gid, set()))
        if "synonyms" in pnf_df.columns:
            syns = pnf_df.loc[pnf_df["generic_id"] == gid, "synonyms"].dropna().astype(str)
            for syn in syns:
                for alias in syn.split("|"):
                    alias_set.update(expand_generic_aliases(alias))
        for alias in alias_set:
            key_norm = normalize_text(alias)
            key_comp = normalize_compact(alias)
            if key_norm and (gid, key_norm) not in seen_norm:
                # Store the longest normalized token once per (gid, normalized name).
                A_norm.add_word(key_norm, (gid, alias)); seen_norm.add((gid, key_norm))
            if key_comp and (gid, key_comp) not in seen_comp:
                # Include the compact form to catch spacing and hyphen differences.
                A_comp.add_word(key_comp, (gid, alias)); seen_comp.add((gid, key_comp))
    A_norm.make_automaton()
    A_comp.make_automaton()
    return A_norm, A_comp


def scan_pnf_all(text_norm: str, text_comp: str,
                 A_norm: ahocorasick.Automaton,
                 A_comp: ahocorasick.Automaton) -> Tuple[List[str], List[str]]:
    """Return ordered (gid, token) hits using whichever automaton produced the longest span."""
    candidates: Dict[str, str] = {}
    for _, (gid, token) in A_norm.iter(text_norm):
        # Keep the longest token emitted per gid when scanning the normalized string.
        if gid not in candidates or len(token) > len(candidates[gid]):
            candidates[gid] = token
    for _, (gid, token) in A_comp.iter(text_comp):
        # Merge compact hits, preferring longer matches to reduce noise.
        if gid not in candidates or len(token) > len(candidates[gid]):
            candidates[gid] = token
    items = sorted(candidates.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    gids = [gid for gid, _ in items]
    tokens = [tok for _, tok in items]
    return gids, tokens
