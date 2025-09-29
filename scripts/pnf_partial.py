#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional

Token = str

def _tokenize(s: str) -> List[Token]:
    """Lowercase text and return alphanumeric tokens suitable for prefix matching."""
    return re.findall(r"[a-z0-9]+", s.lower())

class PnfTokenIndex:
    """Index PNF generics by their tokenized names to enable token-boundary partial matches.
    Example: "tranexamic acid" => tokens ["tranexamic","acid"] so that inputs like "tranexamic tablet"
    can still map to the same generic id by matching the contiguous prefix ["tranexamic"].
    """
    def __init__(self) -> None:
        # first_token -> list of (generic_id, name_tokens, name_norm)
        self.by_first: Dict[Token, List[Tuple[str, List[Token], str]]] = {}

    @staticmethod
    def _name_tokens(name_norm: str) -> List[Token]:
        """Tokenize an already-normalized PNF name for partial matching."""
        return _tokenize(name_norm)

    def add(self, generic_id: str, generic_name_norm: str) -> None:
        """Register a PNF generic under its first token for fast prefix lookups."""
        toks = self._name_tokens(generic_name_norm)
        if not toks:
            return
        first = toks[0]
        self.by_first.setdefault(first, []).append((generic_id, toks, generic_name_norm))

    def build_from_pnf(self, pnf_df) -> "PnfTokenIndex":
        """Populate the index from the prepared PNF dataframe."""
        # expects columns: generic_id, generic_name (normalized externally if desired)
        for gid, gname in pnf_df[["generic_id","generic_name"]].drop_duplicates().itertuples(index=False):
            if not isinstance(gname, str):
                continue
            name_norm = gname.strip().lower()
            self.add(str(gid), name_norm)
        return self

    def best_partial_in_text(self, text_norm: str) -> Optional[Tuple[str, str]]:
        """Return (generic_id, matched_span_tokens) if a token-boundary *partial* match is found.
        Partial = contiguous prefix of the PNF generic appears in the text, but the full generic does not.
        We favor the longest prefix length; ties break on shorter overall generic length (more specific).

        Example: text_norm="tranexamic 500 mg tablet"
                 PNF: ["tranexamic","acid"] -> match_len=1 -> returns gid and "tranexamic"
        """
        text_toks = _tokenize(text_norm)
        if not text_toks:
            return None

        best: Tuple[int, int, str, str] = (0, 10**9, "", "")  # (match_len, generic_len, gid, matched_span)
        for i, tok in enumerate(text_toks):
            # only consider dictionary keyed by first token of generics
            candidates = self.by_first.get(tok, [])
            if not candidates:
                continue
            for gid, name_toks, _name_norm in candidates:
                # compute maximum contiguous overlap starting at position i between text_toks and name_toks
                L = 0
                while i + L < len(text_toks) and L < len(name_toks) and text_toks[i + L] == name_toks[L]:
                    L += 1
                # we only care about *partial* (proper prefix) matches: L >= 1 and L < len(name_toks)
                if L >= 1 and L < len(name_toks):
                    # track longest L; on tie, prefer the generic with fewer total tokens (to reduce overreach)
                    generic_len = len(name_toks)
                    matched_span = " ".join(text_toks[i:i+L])
                    cand = (L, generic_len, str(gid), matched_span)
                    if cand > best:
                        best = cand

        if best[0] > 0:
            return best[2], best[3]
        return None
