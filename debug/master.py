# START OF REPO FILES
# <scripts/aho.py>
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

# <scripts/combos.py>

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
    if not isinstance(s, str) or not s:
        return []
    parts = [p.strip() for p in COMBO_SEP_RX.split(s) if p.strip()]
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
        return True

    if "+" in s_masked:
        return True

    if re.search(r"\bwith\b", s_masked):
        return True

    segs = split_combo_segments(s_masked)
    if len(segs) >= 2:
        if len(segs) == 2 and (_is_salt_tail(segs[0]) or _is_salt_tail(segs[1])):
            return False
        return True

    return False

# <scripts/dose.py>
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, Optional
from math import isclose

from .text_utils import safe_to_float

PACK_RX = re.compile(r"\b(\d+)\s*(?:x|×)\s*(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|ug|iu)\b", re.I)
RATIO_RX_EXTRA = re.compile(r"(?P<num>\d+(?:[.,]\d+)?)\s?(?P<num_unit>mg|g|mcg|ug)\s*/\s?(?P<den>\d+(?:[.,]\d+)?)\s?(?P<den_unit>ml|l)\b", re.I)
PER_UNIT_WORDS = r"(?:tab(?:let)?s?|cap(?:sule)?s?|sachet(?:s)?|drop(?:s)?|gtt|actuation(?:s)?|spray(?:s)?|puff(?:s)?)"

DOSAGE_PATTERNS = [
    # amount-only (e.g., 500 mg)
    r"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\b",
    # amount per mL or L (e.g., 5 mg/5 mL, 1 g/100 L)
    r"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>\d+(?:[.,]\d+)?)\s*(?P<per_unit>ml|l)\b",
    # amount per unit-dose nouns (tab/cap/sachet/drop/actuation/spray/puff)
    rf"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>1)?\s*(?P<per_unit>{PER_UNIT_WORDS})\b",
    # compact noun suffix (e.g., mg/tab, mg/cap)
    rf"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s*/\s?(?P<per_unit>{PER_UNIT_WORDS})\b",
    # percent (optionally w/v or w/w)
    r"(?P<pct>\d+(?:[.,]\d+)?)\s?%(?:\s?(?:w/v|w/w))?",
]
DOSAGE_REGEXES = [re.compile(p, flags=re.I) for p in DOSAGE_PATTERNS]


def _unmask_pack_strength(s_norm: str) -> str:
    """Convert '10 x 500 mg'/'10×500 mg' to just '500 mg' for dose parsing."""
    def repl(m: re.Match):
        amt = m.group(2)
        unit = m.group(3)
        return f"{amt}{unit}"
    return PACK_RX.sub(repl, s_norm)


def parse_dose_struct_from_text(s_norm: str) -> Dict[str, Any]:
    if not isinstance(s_norm, str) or not s_norm:
        return {}
    s_proc = _unmask_pack_strength(s_norm)
    matches = []
    for rx in DOSAGE_REGEXES:
        for m in rx.finditer(s_proc):
            d = {k: (v.replace(",", ".") if isinstance(v, str) else v) for k, v in m.groupdict().items()}
            matches.append(d)
    for d in matches:
        if d.get("strength") and d.get("per_unit") in ("ml", "l"):
            per_val = float(d["per_val"]) if d.get("per_val") else 1.0
            if d.get("per_unit", "").lower() == "l":
                per_val *= 1000.0
            return {"dose_kind": "ratio", "strength": float(d["strength"]), "unit": d["unit"].lower(),
                    "per_val": per_val, "per_unit": "ml"}
    for d in matches:
        if d.get("strength"):
            return {"dose_kind": "amount", "strength": float(d["strength"]), "unit": d["unit"].lower()}
    for d in matches:
        if d.get("pct"):
            return {"dose_kind": "percent", "pct": float(d["pct"])}
    m = RATIO_RX_EXTRA.search(s_proc)
    if m:
        num = float(m.group("num"))
        num_unit = m.group("num_unit").lower()
        den = float(m.group("den"))
        if m.group("den_unit").lower() == "l":
            den *= 1000.0
        return {"dose_kind": "ratio", "strength": num, "unit": num_unit, "per_val": den, "per_unit": "ml"}
    return {}


def to_mg(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None or not isinstance(unit, str):
        return None
    u = unit.lower()
    if u == "mg":
        return value
    if u == "g":
        return value * 1000.0
    if u in ("mcg", "ug"):
        return value / 1000.0
    return None


def to_mg_match(value: float, unit: str):
    u = unit.lower()
    if u == "mg":
        return value
    if u == "g":
        return value * 1000.0
    if u in ("mcg", "ug"):
        return value / 1000.0
    return None


def safe_ratio_mg_per_ml(strength, unit, per_val):
    mg = to_mg(strength, unit)
    pv = safe_to_float(per_val)
    if mg is None or pv in (None, 0):
        return None
    return mg / pv


def extract_dosage(s_norm: str):
    if not isinstance(s_norm, str) or not s_norm:
        return None
    s_proc = _unmask_pack_strength(s_norm)
    matches = []
    for rx in DOSAGE_REGEXES:
        for m in rx.finditer(s_proc):
            d = {k: (v.replace(",", ".") if isinstance(v, str) else v) for k, v in m.groupdict().items()}
            matches.append(d)
    for d in matches:
        if d.get("strength") and d.get("per_unit") in ("ml", "l"):
            per_val = float(d["per_val"]) if d.get("per_val") else 1.0
            if d.get("per_unit", "").lower() == "l":
                per_val *= 1000.0
            return {"kind": "ratio", "strength": float(d["strength"]), "unit": d["unit"].lower(),
                    "per_val": per_val, "per_unit": "ml"}
    for d in matches:
        if d.get("strength"):
            return {"kind": "amount", "strength": float(d["strength"]), "unit": d["unit"].lower()}
    for d in matches:
        if d.get("pct"):
            return {"kind": "percent", "pct": float(d["pct"])}
    m = RATIO_RX_EXTRA.search(s_proc)
    if m:
        num = float(m.group("num"))
        num_unit = m.group("num_unit").lower()
        den = float(m.group("den"))
        if m.group("den_unit").lower() == "l":
            den *= 1000.0
        return {"kind": "ratio", "strength": num, "unit": num_unit, "per_val": den, "per_unit": "ml"}
    return None


def _eq(a: float, b: float) -> bool:
    """Exact equality with robust float check. Accept only true equality up to tiny machine epsilon.
    This enforces zero tolerance logically (e.g., 1 g == 1000 mg), but rejects 450 vs 500 mg.
    """
    # Use a very tight tolerance to avoid binary float artifacts while still enforcing exactness.
    return isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def dose_similarity(esoa_dose: dict, pnf_row) -> float:
    """Return 1.0 only for exact equality (after unit conversion); else 0.0.
    - amount: mg equality must hold exactly after conversion
    - ratio: mg/mL equality must hold exactly after conversion
    - percent: equal percentage
    """
    if not esoa_dose:
        return 0.0
    kind = esoa_dose.get("kind")
    if kind == "amount":
        mg_esoa = to_mg_match(esoa_dose["strength"], esoa_dose["unit"])
        mg_pnf = pnf_row.get("strength_mg")
        if mg_esoa is None or mg_pnf is None:
            return 0.0
        return 1.0 if _eq(mg_esoa, mg_pnf) else 0.0
    if kind == "ratio":
        if pnf_row.get("dose_kind") != "ratio":
            return 0.0
        v_esoa = to_mg_match(esoa_dose["strength"], esoa_dose["unit"])
        if v_esoa is None:
            return 0.0
        ratio_esoa = v_esoa / float(esoa_dose.get("per_val", 1.0))
        ratio_pnf = pnf_row.get("ratio_mg_per_ml")
        if ratio_pnf in (None, 0):
            return 0.0
        return 1.0 if _eq(ratio_esoa, ratio_pnf) else 0.0
    if kind == "percent":
        if pnf_row.get("dose_kind") != "percent":
            return 0.0
        pct_esoa = float(esoa_dose["pct"])
        pct_pnf = pnf_row.get("pct")
        if pct_pnf is None:
            return 0.0
        return 1.0 if _eq(pct_esoa, float(pct_pnf)) else 0.0
    return 0.0

# <scripts/match_features.py>
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, time, glob, os, re
from typing import Tuple, Optional, List, Dict, Set, Callable
import numpy as np, pandas as pd
from .aho import build_molecule_automata, scan_pnf_all
from .combos import looks_like_combination, split_combo_segments
from .routes_forms import extract_route_and_form
from .text_utils import _base_name, _normalize_text_basic, normalize_text, extract_parenthetical_phrases, STOPWORD_TOKENS
from .who_molecules import detect_all_who_molecules, load_who_molecules
from .brand_map import load_latest_brandmap, build_brand_automata, fda_generics_set
from .pnf_partial import PnfTokenIndex

# Local lightweight spinner so this module is self-contained
def _run_with_spinner(label: str, func: Callable[[], None]) -> float:
    import threading
    done = threading.Event()
    err = []

    def worker():
        try:
            func()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"; i = 0
    while not done.is_set():
        elapsed = time.perf_counter() - t0
        sys.stdout.write(f"\r{frames[i % len(frames)]} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        time.sleep(0.1); i += 1
    th.join()
    elapsed = time.perf_counter() - t0
    sys.stdout.write(f"\r✓ {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return elapsed

DOSE_OR_UNIT_RX = re.compile(r"(?:(\b\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ug|iu|lsu|ml|l|%)(?:\b|/))|(\b\d+(?:[.,]\d+)?\b))", re.I)
GENERIC_TOKEN_RX = re.compile(r"[a-z]+", re.I)

def _friendly_dose(d: dict) -> str:
    if not d: return ""
    kind = d.get("kind") or d.get("dose_kind")
    if kind == "amount": return f"{d.get('strength')}{d.get('unit','')}"
    if kind == "ratio":
        pv = d.get("per_val", 1)
        try: pv = int(pv)
        except Exception: pass
        return f"{d.get('strength')}{d.get('unit','')}/{pv}{d.get('per_unit','')}"
    if kind == "percent": return f"{d.get('pct')}%"
    return ""

def _segment_norm(seg: str) -> str:
    s = _normalize_text_basic(_base_name(seg))
    s = DOSE_OR_UNIT_RX.sub(" ", s)
    s = re.sub(r"\b(?:per|with|and)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_latest_who_dir(root_dir: str) -> str | None:
    who_dir = os.path.join(root_dir, "dependencies", "atcd", "output")
    candidates = glob.glob(os.path.join(who_dir, "who_atc_*_molecules.csv"))
    return max(candidates, key=os.path.getmtime) if candidates else None

def _tokenize_unknowns(s_norm: str) -> List[str]:
    return [m.group(0) for m in GENERIC_TOKEN_RX.finditer(s_norm)]

def build_features(pnf_df: pd.DataFrame, esoa_df: pd.DataFrame) -> pd.DataFrame:
    # 1) Validate inputs
    def _validate():
        required_pnf = {
            "generic_id","generic_name","synonyms","atc_code","route_allowed","form_token",
            "dose_kind","strength","unit","per_val","per_unit","pct","strength_mg","ratio_mg_per_ml"
        }
        missing = required_pnf - set(pnf_df.columns)
        if missing:
            raise ValueError(f"pnf_prepared.csv missing columns: {missing}")
        if "raw_text" not in esoa_df.columns:
            raise ValueError("esoa_prepared.csv must contain a 'raw_text' column")
    _run_with_spinner("Validate inputs", _validate)

    # 2) Map normalized PNF names to gid + original name
    pnf_name_to_gid: Dict[str, Tuple[str, str]] = {}
    def _pnf_map():
        for gid, gname in pnf_df[["generic_id","generic_name"]].drop_duplicates().itertuples(index=False):
            key = _normalize_text_basic(_base_name(str(gname)))
            if key and key not in pnf_name_to_gid:
                pnf_name_to_gid[key] = (gid, gname)
    _run_with_spinner("Index PNF names", _pnf_map)
    pnf_name_set: Set[str] = set(pnf_name_to_gid.keys())

    # 3) WHO molecules (names + regex)
    codes_by_name, candidate_names = ({}, [])
    who_name_set: Set[str] = set()
    who_regex = [None]
    def _load_who():
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        who_file = load_latest_who_dir(root_dir)
        if who_file and os.path.exists(who_file):
            cbn, cand = load_who_molecules(who_file)
            codes_by_name.update(cbn)
            candidate_names.extend(cand)
            who_name_set.update(cbn.keys())
            who_regex[0] = re.compile(r"\b(" + "|".join(map(re.escape, candidate_names)) + r")\b") if candidate_names else None
    _run_with_spinner("Load WHO molecules", _load_who)
    who_regex = who_regex[0]

    # 4) Brand map & FDA generics
    brand_df = [None]
    def _load_brand():
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        inputs_dir = os.path.join(project_root, "inputs")
        brand_df[0] = load_latest_brandmap(inputs_dir)
    _run_with_spinner("Load FDA brand map", _load_brand)
    has_brandmap = brand_df[0] is not None and not brand_df[0].empty
    if has_brandmap:
        B_norm = [None]; B_comp = [None]; brand_lookup = [{}]; fda_gens = [set()]
        _run_with_spinner("Build brand automata", lambda: (
            (lambda r: (B_norm.__setitem__(0, r[0]), B_comp.__setitem__(0, r[1]), brand_lookup.__setitem__(0, r[2])))(build_brand_automata(brand_df[0]))
        ))
        _run_with_spinner("Index FDA generics", lambda: fda_gens.__setitem__(0, fda_generics_set(brand_df[0])))
        B_norm, B_comp, brand_lookup, fda_gens = B_norm[0], B_comp[0], brand_lookup[0], fda_gens[0]
    else:
        B_norm = B_comp = None
        brand_lookup = {}
        fda_gens = set()

    # 5) PNF automata + partial index
    A_norm = [None]; A_comp = [None]
    _run_with_spinner("Build PNF automata", lambda: (
        (lambda r: (A_norm.__setitem__(0, r[0]), A_comp.__setitem__(0, r[1])))(build_molecule_automata(pnf_df))
    ))
    pnf_partial_idx = [None]
    _run_with_spinner("Build PNF partial index", lambda: pnf_partial_idx.__setitem__(0, PnfTokenIndex().build_from_pnf(pnf_df)))
    A_norm, A_comp, pnf_partial_idx = A_norm[0], A_comp[0], pnf_partial_idx[0]

    # 6) Base ESOA frame and text normalization
    df = [None]
    def _mk_base():
        tmp = esoa_df[["raw_text"]].copy()
        tmp["parentheticals"] = tmp["raw_text"].map(extract_parenthetical_phrases)
        tmp["esoa_idx"] = tmp.index
        tmp["normalized"] = tmp["raw_text"].map(normalize_text)
        tmp["norm_compact"] = tmp["normalized"].map(lambda s: re.sub(r"[ \-]", "", s))
        df.append(tmp)
    _run_with_spinner("Normalize ESOA text", _mk_base)
    df = df[-1]

    # 7) Dose/route/form on original normalized text
    def _dose_route_form_raw():
        from .dose import extract_dosage as _extract_dosage
        df["dosage_parsed_raw"] = df["normalized"].map(_extract_dosage)
        df["dose_recognized"] = df["dosage_parsed_raw"].map(_friendly_dose)
        df["route_raw"], df["form_raw"], df["route_evidence_raw"] = zip(*df["normalized"].map(extract_route_and_form))
    _run_with_spinner("Parse dose/route/form (raw)", _dose_route_form_raw)

    # 8) Brand → Generic swap (if brand map available)
    def _brand_swap():
        if not has_brandmap:
            df["match_basis"] = df["normalized"]
            df["did_brand_swap"] = False
            df["fda_dose_corroborated"] = False
            return
        mb_list, swapped = [], []
        fda_hits = []
        for norm, comp, form, friendly, parens in zip(
            df["normalized"], df["norm_compact"], df["form_raw"], df["dose_recognized"], df["parentheticals"]
        ):
            # Inline selection/scoring identical to prior implementation
            found_keys: List[str] = []
            lengths: Dict[str, int] = {}
            for _, bn in B_norm.iter(norm):  # type: ignore
                found_keys.append(bn); lengths[bn] = max(lengths.get(bn, 0), len(bn))
            for _, bn in B_comp.iter(comp):  # type: ignore
                found_keys.append(bn); lengths[bn] = max(lengths.get(bn, 0), len(bn))
            if not found_keys:
                mb_list.append(norm); swapped.append(False); fda_hits.append(False); continue
            uniq_keys = list(dict.fromkeys(found_keys))
            uniq_keys.sort(key=lambda k: (-lengths.get(k, len(k)), k))

            out = norm; replaced_any = False
            for bn in uniq_keys:
                options = brand_lookup.get(bn, [])
                chosen_generic = None
                if options:
                    # score
                    def _score(m):
                        sc = 0
                        if friendly and getattr(m, "dosage_strength", None) and friendly.lower() in (m.dosage_strength or "").lower(): sc += 2
                        if form and getattr(m, "dosage_form", None) and form.lower() == (m.dosage_form or "").lower(): sc += 1
                        gen_base = _normalize_text_basic(_base_name(m.generic))
                        if gen_base in pnf_name_to_gid: sc += 3
                        return sc
                    options = sorted(options, key=_score, reverse=True)
                    chosen_generic = options[0].generic if options else None
                if not chosen_generic:
                    continue
                gd_norm = normalize_text(chosen_generic)
                new_out = re.sub(rf"\b{re.escape(bn)}\b", gd_norm, out)
                if new_out != out:
                    replaced_any = True
                    out = new_out
            out = re.sub(r"\s+", " ", out).strip()
            # FDA dose corroboration
            fda_hit = False
            if options and friendly:
                ds = getattr(options[0], "dosage_strength", "") or ""
                if ds and friendly.lower() in ds.lower():
                    fda_hit = True
            mb_list.append(out); swapped.append(replaced_any); fda_hits.append(fda_hit)
        df["match_basis"] = mb_list
        df["did_brand_swap"] = swapped
        df["fda_dose_corroborated"] = fda_hits
    _run_with_spinner("Apply brand→generic swaps", _brand_swap)

    # 9) Dose/route/form on match_basis
    def _dose_route_form_basis():
        from .dose import extract_dosage as _extract_dosage
        df["dosage_parsed"] = df["match_basis"].map(_extract_dosage)
        df["route"], df["form"], df["route_evidence"] = zip(*df["match_basis"].map(extract_route_and_form))
    _run_with_spinner("Parse dose/route/form (basis)", _dose_route_form_basis)

    # 10) PNF hits (Aho-Corasick) on match_basis
    def _pnf_hits():
        primary_gid, primary_token, pnf_hits_gids, pnf_hits_tokens, pnf_hits_count = [], [], [], [], []
        for s_norm, s_comp in zip(df["match_basis"], df["norm_compact"]):
            gids, tokens = scan_pnf_all(s_norm, s_comp, A_norm, A_comp)
            pnf_hits_gids.append(gids); pnf_hits_tokens.append(tokens); pnf_hits_count.append(len(gids))
            if gids: primary_gid.append(gids[0]); primary_token.append(tokens[0])
            else: primary_gid.append(None); primary_token.append(None)
        df["pnf_hits_gids"] = pnf_hits_gids; df["pnf_hits_tokens"] = pnf_hits_tokens; df["pnf_hits_count"] = pnf_hits_count
        df["generic_id"] = primary_gid; df["molecule_token"] = primary_token
    _run_with_spinner("Scan PNF hits", _pnf_hits)

    # 11) Partial PNF fallback
    def _pnf_partial():
        partial_gids: List[Optional[str]] = [None] * len(df)
        partial_tokens: List[Optional[str]] = [None] * len(df)
        for i, (gid, txt) in enumerate(zip(df["generic_id"].tolist(), df["match_basis"].tolist())):
            if gid is not None:
                continue
            res = pnf_partial_idx.best_partial_in_text(str(txt))
            if res:
                pgid, matched_span = res
                partial_gids[i] = pgid
                partial_tokens[i] = matched_span
        mask_partial = pd.Series([g is not None for g in partial_gids])
        if mask_partial.any():
            df.loc[mask_partial, "generic_id"] = [g for g in partial_gids if g is not None]
            df.loc[mask_partial, "molecule_token"] = [t for t in partial_tokens if t is not None]
            df.loc[mask_partial, "pnf_hits_count"] = df.loc[mask_partial, "pnf_hits_count"].fillna(0).astype(int) + 1
    _run_with_spinner("Partial PNF fallback", _pnf_partial)

    # 12) WHO molecule detection
    def _who_detect():
        if who_regex:
            who_names_all, who_atc_all = [], []
            for txt in df["match_basis"].tolist():
                names, codes = detect_all_who_molecules(txt, who_regex, codes_by_name)
                who_names_all.append(names); who_atc_all.append(sorted(codes))
            df["who_molecules_list"] = who_names_all; df["who_atc_codes_list"] = who_atc_all
            df["who_molecules"] = df["who_molecules_list"].map(lambda xs: "|".join(xs) if xs else "")
            df["who_atc_codes"] = df["who_atc_codes_list"].map(lambda xs: "|".join(xs) if xs else "")
        else:
            df["who_molecules_list"] = [[] for _ in range(len(df))]
            df["who_atc_codes_list"] = [[] for _ in range(len(df))]
            df["who_molecules"] = ""; df["who_atc_codes"] = ""
    _run_with_spinner("Detect WHO molecules", _who_detect)

    # 13) Combination detection helpers
    def _combo_feats():
        def _known_generic_tokens(text_norm: str) -> List[str]:
            s = _segment_norm(text_norm)
            toks = _tokenize_unknowns(s)
            out = []
            for t in toks:
                if t in pnf_name_set or t in who_name_set or t in fda_gens:
                    out.append(t)
            seen=set(); res=[]
            for t in out:
                if t not in seen:
                    seen.add(t); res.append(t)
            return res
        known_counts = df["match_basis"].map(lambda s: len(_known_generic_tokens(s)))
        df["combo_known_generics_count"] = known_counts
        df["looks_combo_final"] = df["combo_known_generics_count"].ge(2)
        df["combo_reason"] = np.where(df["looks_combo_final"], "combo/known-generics>=2", "single/heuristic")
    _run_with_spinner("Compute combo features", _combo_feats)

    # 14) Unknown tokens extraction
    def _unknowns():
        def _unknown_kind_and_list(text_norm: str) -> Tuple[str, List[str]]:
            s = _segment_norm(text_norm)
            all_toks = _tokenize_unknowns(s)
            unknowns = []
            for t in all_toks:
                if (
                    (t not in pnf_name_set)
                    and (t not in who_name_set)
                    and (t not in fda_gens)
                    and (t not in STOPWORD_TOKENS)
                ):
                    unknowns.append(t)
            seen=set(); unknowns_uniq=[]
            for t in unknowns:
                if t not in seen:
                    seen.add(t); unknowns_uniq.append(t)
            if not unknowns_uniq:
                return "None", []
            if len(unknowns_uniq) == len(all_toks):
                if len(unknowns_uniq) == 1:
                    return "Single - Unknown", unknowns_uniq
                return "Multiple - All Unknown", unknowns_uniq
            return ("Multiple - Some Unknown", unknowns_uniq)
        kinds, lists_ = zip(*df["match_basis"].map(_unknown_kind_and_list))
        df["unknown_kind"] = kinds
        df["unknown_words_list"] = lists_
        df["unknown_words"] = df["unknown_words_list"].map(lambda xs: "|".join(xs) if xs else "")
    _run_with_spinner("Extract unknown tokens", _unknowns)

    # 15) Presence flags
    def _presence_flags():
        df["present_in_pnf"] = df["pnf_hits_count"].astype(int).gt(0)
        df["present_in_who"] = df["who_atc_codes"].astype(str).str.len().gt(0)
        df["present_in_fda_generic"] = df["match_basis"].map(lambda s: any(tok in fda_gens for tok in _tokenize_unknowns(_segment_norm(s))))
    _run_with_spinner("Compute presence flags", _presence_flags)

    return df

# <scripts/match_scoring.py>
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import List
import numpy as np, pandas as pd
from .dose import dose_similarity
from .text_utils import _base_name, _normalize_text_basic

def _route_ok(row: pd.Series) -> bool:
    r = row["route"]; allowed = row.get("route_allowed")
    if pd.isna(r) or not r: return True
    if isinstance(allowed, str) and allowed: return r in allowed.split("|")
    return True

def _pick_best(group: pd.DataFrame) -> pd.Series:
    if group.empty:
        return pd.Series({
            "atc_code_final": None, "dose_sim": 0.0, "form_ok": False, "route_ok": False,
            "match_quality": "unspecified", "selected_form": None, "selected_variant": None
        })
    esoa_form = group.iloc[0]["form"]; esoa_route = group.iloc[0]["route"]; esoa_dose = group.iloc[0]["dosage_parsed"]
    scored = []
    for _, row in group.iterrows():
        score = 0.0
        form_ok = bool(esoa_form and row.get("form_token") and esoa_form == row["form_token"])
        route_ok = bool(esoa_route and row.get("route_allowed") and esoa_route == row["route_allowed"])
        if form_ok: score += 40
        if route_ok: score += 30
        sim = dose_similarity(esoa_dose, row); score += sim * 30
        scored.append((score, sim, form_ok, route_ok, row))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, best_sim, best_form_ok, best_route_ok, best_row = scored[0]

    note = "OK"
    if best_sim < 1.0:
        note = "dose mismatch"
    if not best_form_ok and (note == "OK"):
        note = "no/poor form match"
    if not best_route_ok and (note == "OK"):
        note = "no/poor route match"

    strength = best_row.get("strength"); unit = best_row.get("unit") or ""
    per_val = best_row.get("per_val"); per_unit = best_row.get("per_unit") or ""; pct = best_row.get("pct")
    variant = f"{best_row.get('dose_kind')}:{strength}{unit}" if pd.notna(strength) else str(best_row.get("dose_kind"))
    if pd.notna(per_val):
        try: pv_int = int(per_val)
        except Exception: pv_int = per_val
        variant += f"/{pv_int}{per_unit}"
    if pd.notna(pct): variant += f" {pct}%"

    return pd.Series({
        "atc_code_final": best_row["atc_code"] if isinstance(best_row["atc_code"], str) and best_row["atc_code"] else None,
        "dose_sim": float(best_sim),
        "form_ok": bool(best_form_ok),
        "route_ok": bool(best_route_ok),
        "match_quality": note,
        "selected_form": best_row.get("form_token"),
        "selected_variant": variant,
    })

def _score_row(r: pd.Series) -> int:
    score = 0
    if pd.notna(r.get("generic_id")): score += 60
    if r.get("dosage_parsed"): score += 15
    if r.get("route_evidence"): score += 10
    if pd.notna(r.get("atc_code_final")): score += 15
    sim = r.get("dose_sim")
    try: sim = float(sim)
    except Exception: sim = 0.0
    if math.isnan(sim): sim = 0.0
    score += int(max(0.0, min(1.0, sim)) * 10)
    try:
        if r.get("did_brand_swap") and r.get("form_ok") and r.get("route_ok") and float(r.get("dose_sim", 0)) >= 1.0:
            score += 10
    except Exception:
        pass
    return score

def _union_molecules(row: pd.Series) -> List[str]:
    names = []
    for t in (row.get("pnf_hits_tokens") or []):
        if not isinstance(t, str): continue
        names.append(_normalize_text_basic(_base_name(t)))
    for t in (row.get("who_molecules_list") or []):
        if not isinstance(t, str): continue
        names.append(_normalize_text_basic(_base_name(t)))
    seen = set(); uniq = []
    for n in names:
        if not n or n in seen: continue
        seen.add(n); uniq.append(n)
    return uniq

def _mk_reason(series: pd.Series, default_ok: str) -> pd.Series:
    s = series.astype("string")
    s = s.fillna(default_ok)
    s = s.replace({"": default_ok, "unspecified": default_ok})
    return s.astype("string")

def _missing_combo(row: pd.Series) -> str:
    missing = []
    if not bool(row.get("dosage_parsed")):
        missing.append("dose")
    if not bool(row.get("form")):
        missing.append("form")
    if not bool(row.get("route")):
        missing.append("route")
    if not missing:
        return ""
    if len(missing) == 1:
        return f"no {missing[0]} available"
    if len(missing) == 2:
        return f"no {missing[0]} and {missing[1]} available"
    return "no dose, form, and route available"

def score_and_classify(features_df: pd.DataFrame, pnf_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()

    # Candidate rows that have PNF generic_id
    df_cand = df.loc[df["generic_id"].notna(), ["esoa_idx","generic_id","route","form","dosage_parsed"]].merge(pnf_df, on="generic_id", how="left")
    if not df_cand.empty:
        df_cand = df_cand[df_cand.apply(_route_ok, axis=1)]
        df_cand["dose_sim"] = df_cand.apply(lambda r: dose_similarity(r["dosage_parsed"], r), axis=1)
        df_cand["dose_sim"] = pd.to_numeric(df_cand["dose_sim"], errors="coerce").fillna(0.0)
        best_by_idx = df_cand.groupby("esoa_idx", sort=False).apply(_pick_best, include_groups=False)
        out = df.merge(best_by_idx, left_on="esoa_idx", right_index=True, how="left")
    else:
        out = df.copy()
        out[["atc_code_final","dose_sim","form_ok","route_ok","match_quality","selected_form","selected_variant"]] = [None, 0.0, False, False, "unspecified", None, None]

    # Confidence and molecules recognized
    out["confidence"] = out.apply(_score_row, axis=1)
    out["molecules_recognized_list"] = out.apply(_union_molecules, axis=1)
    out["molecules_recognized"] = out["molecules_recognized_list"].map(lambda xs: "|".join(xs) if xs else "")
    out["molecules_recognized_count"] = out["molecules_recognized_list"].map(lambda xs: len(xs or []))
    out["who_atc_count"] = out["who_atc_codes_list"].map(lambda xs: len(xs or []))

    # Probable ATC if absent in PNF but present in WHO
    out["probable_atc"] = np.where(~out["present_in_pnf"] & out["present_in_who"], out["who_atc_codes"], "")

    # Initialize tags
    out["bucket_final"] = ""
    out["why_final"] = ""
    out["reason_final"] = ""
    out["match_quality"] = ""
    out["match_molecule(s)"] = ""  # precise naming

    # Derive *match_molecule(s)* tags
    present_in_pnf = out["present_in_pnf"].astype(bool)
    present_in_who = out["present_in_who"].astype(bool)
    present_in_fda = out["present_in_fda_generic"].astype(bool)
    has_atc_in_pnf = out["atc_code_final"].astype(str).str.len().gt(0)

    out.loc[present_in_pnf & has_atc_in_pnf, "match_molecule(s)"] = "ValidMoleculeWithATCinPNF"
    out.loc[(~present_in_pnf) & present_in_who, "match_molecule(s)"] = "ValidMoleculeWithATCinWHO/NotInPNF"
    out.loc[(~present_in_pnf) & (~present_in_who) & present_in_fda, "match_molecule(s)"] = "ValidMoleculeNoATCinFDA/NotInPNF"

    # Brand-swapped variants
    out.loc[out["did_brand_swap"].astype(bool) & present_in_who, "match_molecule(s)"] = "ValidBrandSwappedForMoleculeWithATCinWHO"
    out.loc[out["did_brand_swap"].astype(bool) & present_in_pnf & has_atc_in_pnf, "match_molecule(s)"] = "ValidBrandSwappedForGenericInPNF"

    # AUTO-ACCEPT
    is_candidate_like = out["generic_id"].notna()
    form_ok_col = out["form_ok"].astype(bool)
    route_ok_col = out["route_ok"].astype(bool)
    auto_mask = is_candidate_like & present_in_pnf & has_atc_in_pnf & form_ok_col & route_ok_col
    out.loc[auto_mask, "bucket_final"] = "Auto-Accept"

    # Needs review
    needs_rev_mask = (out["bucket_final"] == "") & (is_candidate_like | present_in_who | present_in_fda)
    out.loc[needs_rev_mask, "bucket_final"] = "Needs review"
    out.loc[needs_rev_mask, "why_final"] = "Needs review"

    # Match quality for Needs review
    missing_strings = out.apply(lambda r: _missing_combo(r), axis=1).astype("string")
    dose_mismatch_general = (out["dose_sim"].astype(float) < 1.0) & out["dosage_parsed"].astype(bool)
    out.loc[needs_rev_mask & dose_mismatch_general, "match_quality"] = "dose mismatch"
    out.loc[needs_rev_mask & (~dose_mismatch_general) & (missing_strings.str.len() > 0), "match_quality"] = missing_strings
    out.loc[needs_rev_mask & (out["match_quality"] == ""), "match_quality"] = "unspecified"

    # Keep reason_final populated
    out.loc[needs_rev_mask, "reason_final"] = out.loc[needs_rev_mask, "match_quality"]
    out["reason_final"] = _mk_reason(out["reason_final"], "unspecified")

    # Others: Unknowns
    unknown_single = out["unknown_kind"].eq("Single - Unknown")
    unknown_multi_all = out["unknown_kind"].eq("Multiple - All Unknown")
    unknown_multi_some = out["unknown_kind"].eq("Multiple - Some Unknown")
    none_found = out["unknown_kind"].eq("None")
    fallback_unknown = (~is_candidate_like) & (~present_in_who) & (~present_in_fda) & (~none_found)

    def _annotate_unknown(s: str) -> str:
        if s == "Single - Unknown": return "Single - Unknown (unknown to PNF, WHO, FDA)"
        if s == "Multiple - All Unknown": return "Multiple - All Unknown (unknown to PNF, WHO, FDA)"
        if s == "Multiple - Some Unknown": return "Multiple - Some Unknown (some unknown to PNF, WHO, FDA)"
        return s

    for cond, reason in [
        (unknown_single | (fallback_unknown & unknown_single), "Single - Unknown"),
        (unknown_multi_all | (fallback_unknown & unknown_multi_all), "Multiple - All Unknown"),
        (unknown_multi_some | (fallback_unknown & unknown_multi_some), "Multiple - Some Unknown"),
    ]:
        mask = cond & (out["bucket_final"] == "")
        out.loc[mask, "bucket_final"] = "Others"
        out.loc[mask, "why_final"] = "Unknown"
        out.loc[mask, "reason_final"] = _annotate_unknown(reason)

    # Final safety net
    remaining = out["bucket_final"].eq("")
    out.loc[remaining, "bucket_final"] = "Needs review"
    out.loc[remaining, "why_final"] = "Needs review"
    out.loc[remaining, "reason_final"] = _mk_reason(out.loc[remaining, "match_quality"], "unspecified")

    # Dose recognized: N/A unless exact
    if "dose_recognized" in out.columns:
        out["dose_recognized"] = np.where(out["dose_sim"].astype(float) == 1.0, out["dose_recognized"], "N/A")

    return out

# <scripts/match_outputs.py>
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, time
import os, pandas as pd
from typing import Callable

# Local lightweight spinner so this module is self-contained
def _run_with_spinner(label: str, func: Callable[[], None]) -> float:
    import threading
    done = threading.Event()
    err = []

    def worker():
        try:
            func()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"; i = 0
    while not done.is_set():
        elapsed = time.perf_counter() - t0
        sys.stdout.write(f"\r{frames[i % len(frames)]} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        time.sleep(0.1); i += 1
    th.join()
    elapsed = time.perf_counter() - t0
    sys.stdout.write(f"\r✓ {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return elapsed

OUTPUT_COLUMNS = [
    "raw_text","normalized","match_basis",
    "molecules_recognized","molecules_recognized_count","probable_brands",
    "dose_recognized","dose_kind_detected","route","form",
    "present_in_pnf","present_in_who","present_in_fda_generic","probable_atc",
    "generic_id","molecule_token","pnf_hits_count","pnf_hits_tokens",
    "who_molecules","who_atc_codes","who_atc_count",
    "route_evidence","dosage_parsed","selected_form","selected_variant","dose_sim",
    "did_brand_swap","looks_combo_final","combo_reason","combo_known_generics_count",
    "unknown_kind","unknown_words",
    "atc_code_final","confidence",
    "match_molecule(s)","match_quality",
    "bucket_final","why_final","reason_final",
]

def _write_summary_text(out_small: pd.DataFrame, out_csv: str) -> None:
    total = len(out_small)
    lines = []
    lines.append("Distribution Summary")

    # Auto-Accept
    aa_rows = out_small.loc[out_small["bucket_final"].eq("Auto-Accept")].copy()
    aa = int(len(aa_rows)) if total else 0
    aa_pct = round(aa / float(total) * 100, 2) if total else 0.0
    lines.append(f"Auto-Accept: {aa:,} ({aa_pct}%)")
    if aa:
        exact_mask = (aa_rows["dose_sim"].astype(float) == 1.0) & aa_rows["route"].astype(str).ne("") & aa_rows["form"].astype(str).ne("")
        swapped_exact = aa_rows.loc[exact_mask & aa_rows["did_brand_swap"].astype(bool)]
        clean_exact = aa_rows.loc[exact_mask & (~aa_rows["did_brand_swap"].astype(bool))]
        if len(swapped_exact):
            pct = round(len(swapped_exact) / float(total) * 100, 2)
            lines.append(f"  ValidBrandSwappedForGenericInPNF: exact dose/form/route match: {len(swapped_exact):,} ({pct}%)")
        if len(clean_exact):
            pct = round(len(clean_exact) / float(total) * 100, 2)
            lines.append(f"  ValidGenericInPNF, exact dose/route/form match: {len(clean_exact):,} ({pct}%)")

    # Needs review
    nr_rows = out_small.loc[out_small["bucket_final"].eq("Needs review")].copy()
    nr = int(len(nr_rows))
    nr_pct = round(nr / float(total) * 100, 2) if total else 0.0
    lines.append(f"Needs review: {nr:,} ({nr_pct}%)")
    if nr:
        nr_rows["match_molecule(s)"] = nr_rows["match_molecule(s)"].replace({"": "UnspecifiedSource"})
        nr_rows["match_quality"] = nr_rows["match_quality"].replace({"": "unspecified"})
        grp = (nr_rows.groupby(["match_molecule(s)","match_quality"], dropna=False).size().reset_index(name="n"))
        grp["pct"] = (grp["n"] / float(total) * 100).round(2) if total else 0.0
        grp = grp.sort_values(by=["match_molecule(s)","match_quality","n"], ascending=[True, True, False])
        for _, row in grp.iterrows():
            lines.append(f"  {row['match_molecule(s)']}: {row['match_quality']}: {row['n']:,} ({row['pct']}%)")

    # Others
    oth_rows = out_small.loc[out_small["bucket_final"].eq("Others")].copy()
    oth = int(len(oth_rows))
    oth_pct = round(oth / float(total) * 100, 2) if total else 0.0
    lines.append(f"Others: {oth:,} ({oth_pct}%)")
    if oth:
        grouped_oth = (
            oth_rows.groupby(["why_final","reason_final"], dropna=False).size().reset_index(name="n")
        )
        grouped_oth["pct"] = (grouped_oth["n"] / float(total) * 100).round(2) if total else 0.0
        grouped_oth = grouped_oth.sort_values(by=["why_final","reason_final","n"], ascending=[True, True, False])
        for _, row in grouped_oth.iterrows():
            lines.append(f"  {row['why_final']}: {row['reason_final']}: {row['n']:,} ({row['pct']}%)")

    summary_path = os.path.join(os.path.dirname(out_csv), "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def write_outputs(out_df: pd.DataFrame, out_csv: str) -> str:
    out_small = out_df.copy()
    if "match_basis" not in out_small.columns:
        out_small["match_basis"] = out_small.get("normalized", "")
    out_small = out_small[[c for c in OUTPUT_COLUMNS if c in out_small.columns]].copy()

    _run_with_spinner("Write matched CSV", lambda: out_small.to_csv(out_csv, index=False, encoding="utf-8"))

    xlsx_out = os.path.splitext(out_csv)[0] + ".xlsx"
    def _to_excel():
        try:
            with pd.ExcelWriter(xlsx_out, engine="xlsxwriter") as writer:
                out_small.to_excel(writer, index=False, sheet_name="matched")
                ws = writer.sheets["matched"]
                ws.freeze_panes(1, 0)
                nrows, ncols = out_small.shape
                ws.autofilter(0, 0, nrows, ncols - 1)
        except Exception:
            with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
                out_small.to_excel(writer, index=False, sheet_name="matched")
                ws = writer.sheets["matched"]
                try:
                    ws.freeze_panes = "A2"
                    ws.auto_filter.ref = ws.dimensions
                except Exception:
                    pass
    _run_with_spinner("Write Excel", _to_excel)

    def _write_unknowns():
        unknown = out_small.loc[
            out_small["unknown_kind"].ne("None") & out_small["unknown_words"].astype(str).ne(""),
            ["unknown_words"]
        ].copy()
        words = []
        for s in unknown["unknown_words"]:
            for w in str(s).split("|"):
                w = w.strip()
                if w: words.append(w)
        if words:
            unk_df = pd.DataFrame({"word": words})
            unk_df = unk_df.groupby("word").size().reset_index(name="count").sort_values("count", ascending=False)
            unk_path = os.path.join(os.path.dirname(out_csv), "unknown_words.csv")
            unk_df.to_csv(unk_path, index=False, encoding="utf-8")
    _run_with_spinner("Write unknown words CSV", _write_unknowns)

    _run_with_spinner("Write summary.txt", lambda: _write_summary_text(out_small, out_csv))

    return out_csv

# <scripts/match.py>
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, time
import os, pandas as pd
from typing import Callable

# Local lightweight spinner so this module is self-contained
def _run_with_spinner(label: str, func: Callable[[], None]) -> float:
    import threading
    done = threading.Event()
    err = []

    def worker():
        try:
            func()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"; i = 0
    while not done.is_set():
        elapsed = time.perf_counter() - t0
        sys.stdout.write(f"\r{frames[i % len(frames)]} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        time.sleep(0.1); i += 1
    th.join()
    elapsed = time.perf_counter() - t0
    sys.stdout.write(f"\r✓ {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return elapsed

from .match_features import build_features
from .match_scoring import score_and_classify
from .match_outputs import write_outputs

def match(pnf_prepared_csv: str, esoa_prepared_csv: str, out_csv: str = "esoa_matched.csv") -> str:
    # Load inputs
    pnf_df = [None]
    esoa_df = [None]
    _run_with_spinner("Load PNF prepared CSV", lambda: pnf_df.__setitem__(0, pd.read_csv(pnf_prepared_csv)))
    _run_with_spinner("Load eSOA prepared CSV", lambda: esoa_df.__setitem__(0, pd.read_csv(esoa_prepared_csv)))

    # Build features — inner function prints its own sub-spinners; do not show outer spinner
    t0 = time.perf_counter()
    features_df = build_features(pnf_df[0], esoa_df[0])
    print(f"✓ {(time.perf_counter() - t0):7.2f}s Build features")

    # Score & classify
    out_df = [None]
    _run_with_spinner("Score & classify", lambda: out_df.__setitem__(0, score_and_classify(features_df, pnf_df[0])))

    # Write outputs — inner module prints its own sub-spinners; do not show outer spinner
    out_path = os.path.abspath(out_csv)
    t1 = time.perf_counter()
    write_outputs(out_df[0], out_path)
    print(f"✓ {(time.perf_counter() - t1):7.2f}s Write outputs")

    return out_path

# <scripts/prepare.py>
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd

from .routes_forms import map_route_token, parse_form_from_text
from .dose import parse_dose_struct_from_text, to_mg, safe_ratio_mg_per_ml
from .text_utils import clean_atc, normalize_text, slug_id


def prepare(pnf_csv: str, esoa_csv: str, outdir: str = ".") -> tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)

    pnf = pd.read_csv(pnf_csv)
    for col in ["Molecule", "Route", "ATC Code"]:
        if col not in pnf.columns:
            raise ValueError(f"pnf.csv is missing required column: {col}")

    pnf["generic_name"] = pnf["Molecule"].fillna("").astype(str)
    pnf["generic_id"] = pnf["generic_name"].map(slug_id)
    pnf["synonyms"] = ""
    pnf["route_tokens"] = pnf["Route"].map(map_route_token)
    pnf["atc_code"] = pnf["ATC Code"].map(clean_atc)

    text_cols = [c for c in ["Technical Specifications", "Specs", "Specification"] if c in pnf.columns]
    pnf["_tech"] = pnf[text_cols[0]].fillna("") if text_cols else ""
    pnf["_parse_src"] = (pnf["generic_name"].astype(str) + " " + pnf["_tech"].astype(str)).str.strip().map(normalize_text)

    parsed = pnf["_parse_src"].map(parse_dose_struct_from_text)
    pnf["dose_kind"] = parsed.map(lambda d: d.get("dose_kind"))
    pnf["strength"] = parsed.map(lambda d: d.get("strength"))
    pnf["unit"] = parsed.map(lambda d: d.get("unit"))
    pnf["per_val"] = parsed.map(lambda d: d.get("per_val"))
    pnf["per_unit"] = parsed.map(lambda d: d.get("per_unit"))
    pnf["pct"] = parsed.map(lambda d: d.get("pct"))
    pnf["form_token"] = pnf["_parse_src"].map(parse_form_from_text)

    pnf["strength_mg"] = pnf.apply(
        lambda r: to_mg(r.get("strength"), r.get("unit"))
        if (pd.notna(r.get("strength")) and isinstance(r.get("unit"), str) and r.get("unit"))
        else None,
        axis=1,
    )
    pnf["ratio_mg_per_ml"] = pnf.apply(
        lambda r: safe_ratio_mg_per_ml(r.get("strength"), r.get("unit"), r.get("per_val"))
        if (r.get("dose_kind") == "ratio" and str(r.get("per_unit")).lower() == "ml")
        else None,
        axis=1,
    )

    exploded = pnf.explode("route_tokens", ignore_index=True)
    exploded.rename(columns={"route_tokens": "route_allowed"}, inplace=True)
    keep = exploded[exploded["generic_name"].astype(bool)].copy()

    pnf_prepared = keep[[
        "generic_id", "generic_name", "synonyms", "atc_code",
        "route_allowed", "form_token", "dose_kind",
        "strength", "unit", "per_val", "per_unit", "pct",
        "strength_mg", "ratio_mg_per_ml",
    ]].copy()

    pnf_out = os.path.join(outdir, "pnf_prepared.csv")
    pnf_prepared.to_csv(pnf_out, index=False, encoding="utf-8")

    esoa = pd.read_csv(esoa_csv)
    if "DESCRIPTION" not in esoa.columns:
        raise ValueError("esoa.csv is missing required column: DESCRIPTION")
    esoa_prepared = esoa.rename(columns={"DESCRIPTION": "raw_text"}).copy()
    esoa_out = os.path.join(outdir, "esoa_prepared.csv")
    esoa_prepared.to_csv(esoa_out, index=False, encoding="utf-8")

    return pnf_out, esoa_out

# <scripts/routes_forms.py>
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import List, Optional, Tuple

FORM_TO_ROUTE = {
    "tablet": "oral", "tab": "oral", "tabs": "oral",
    "capsule": "oral", "cap": "oral", "caps": "oral",
    "syrup": "oral", "suspension": "oral", "solution": "oral",
    "sachet": "oral",
    "drop": "ophthalmic", "eye drop": "ophthalmic", "ear drop": "otic",
    "cream": "topical", "ointment": "topical", "gel": "topical", "lotion": "topical",
    "patch": "transdermal", "inhaler": "inhalation", "nebule": "inhalation", "neb": "inhalation",
    "ampoule": "intravenous", "amp": "intravenous", "ampul": "intravenous", "ampule": "intravenous",
    "vial": "intravenous", "vl": "intravenous", "inj": "intravenous", "injection": "intravenous",
    "suppository": "rectal",
    "mdi": "inhalation",
    "dpi": "inhalation",
    "metered dose inhaler": "inhalation",
    "dry powder inhaler": "inhalation",
    "spray": "nasal",
    "nasal spray": "nasal",
    "susp": "oral",
    "soln": "oral",
    "syr": "oral",
    "td": "transdermal",
    "supp": "rectal"
}
FORM_WORDS = sorted(set(FORM_TO_ROUTE.keys()), key=len, reverse=True)

ROUTE_ALIASES = {
    "po": "oral", "per orem": "oral", "by mouth": "oral",
    "iv": "intravenous", "intravenous": "intravenous",
    "im": "intramuscular", "intramuscular": "intramuscular",
    "sc": "subcutaneous", "subcut": "subcutaneous", "subcutaneous": "subcutaneous",
    "sl": "sublingual", "sublingual": "sublingual", "bucc": "buccal", "buccal": "buccal",
    "topical": "topical", "cutaneous": "topical", "dermal": "transdermal",
    "oph": "ophthalmic", "eye": "ophthalmic", "ophthalmic": "ophthalmic",
    "otic": "otic", "ear": "otic",
    "inh": "inhalation", "neb": "inhalation", "inhalation": "inhalation",
    "rectal": "rectal", "vaginal": "vaginal",
    "intrathecal": "intrathecal", "nasal": "nasal",
    "per os": "oral",
    "td": "transdermal",
    "transdermal": "transdermal",
    "intradermal": "intradermal",
    "id": "intradermal",
    "subdermal": "subcutaneous",
    "per rectum": "rectal",
    "pr": "rectal",
    "per vaginam": "vaginal",
    "pv": "vaginal",
    "per nasal": "nasal",
    "intranasal": "nasal",
    "inhaler": "inhalation"
}

def map_route_token(r) -> List[str]:
    if not isinstance(r, str):
        return []
    r = r.strip()
    table = {
        "Oral:": ["oral"],
        "Oral/Tube feed:": ["oral"],
        "Inj.:": ["intravenous", "intramuscular", "subcutaneous"],
        "IV:": ["intravenous"],
        "IV/SC:": ["intravenous", "subcutaneous"],
        "SC:": ["subcutaneous"],
        "Subdermal:": ["subcutaneous"],
        "Inhalation:": ["inhalation"],
        "Topical:": ["topical"],
        "Patch:": ["transdermal"],
        "Ophthalmic:": ["ophthalmic"],
        "Intraocular:": ["ophthalmic"],
        "Otic:": ["otic"],
        "Nasal:": ["nasal"],
        "Rectal:": ["rectal"],
        "Vaginal:": ["vaginal"],
        "Sublingual:": ["sublingual"],
        "Oral antiseptic:": ["oral"],
        "Oral/Inj.:": ["oral", "intravenous", "intramuscular", "subcutaneous"],
    }
    return table.get(r, [])

def parse_form_from_text(s_norm: str) -> Optional[str]:
    for fw in FORM_WORDS:
        if re.search(rf"\b{re.escape(fw)}\b", s_norm):
            return fw
    return None

def extract_route_and_form(s_norm: str) -> Tuple[Optional[str], Optional[str], str]:
    route_found = None
    form_found = None
    evidence = []
    for fw in FORM_WORDS:
        if re.search(rf"\b{re.escape(fw)}\b", s_norm):
            form_found = fw
            evidence.append(f"form:{fw}")
            break
    for alias, route in ROUTE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", s_norm):
            route_found = route
            evidence.append(f"route:{alias}->{route}")
            break
    if not route_found and form_found in FORM_TO_ROUTE:
        route_found = FORM_TO_ROUTE[form_found]
        evidence.append(f"impute_route:{form_found}->{route_found}")
    return route_found, form_found, ";".join(evidence)

# <scripts/text_utils.py>

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import unicodedata
from typing import Optional, List

PAREN_CONTENT_RX = re.compile(r"\(([^)]+)\)")

def _normalize_text_basic(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _base_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.split(r",| incl\.| including ", name, maxsplit=1)[0]
    return re.sub(r"\s+", " ", name).strip()

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^\w%/+\.\- ]+", " ", s)
    s = s.replace("microgram", "mcg").replace("μg", "mcg").replace("µg", "mcg")
    s = s.replace("cc", "ml").replace("milli litre", "ml").replace("milliliter", "ml")
    s = s.replace("gm", "g").replace("gms", "g").replace("milligram", "mg")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_compact(s: str) -> str:
    return re.sub(r"[ \-]", "", normalize_text(s))

def slug_id(name: str) -> str:
    base = normalize_text(str(name))
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")

def clean_atc(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\u00a0", " ").strip()

def safe_to_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace(",", ".").strip()
        return float(x)
    except Exception:
        return None

def extract_parenthetical_phrases(raw_text: str) -> List[str]:
    """Extract probable brand/details from the ORIGINAL text."""
    if not isinstance(raw_text, str) or "(" not in raw_text:
        return []
    items = [m.group(1).strip() for m in PAREN_CONTENT_RX.finditer(raw_text) if m.group(1).strip()]
    cleaned = []
    for it in items:
        if len(it) > 60:
            continue
        if re.fullmatch(r"[-/+\s]+", it):
            continue
        cleaned.append(re.sub(r"\s+", " ", it))
    seen = set()
    uniq = []
    for c in cleaned:
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    return uniq

from .combos import SALT_TOKENS
from .routes_forms import FORM_TO_ROUTE, ROUTE_ALIASES

STOPWORD_TOKENS = (
    set(SALT_TOKENS)
    | set(FORM_TO_ROUTE.keys())
    | set(ROUTE_ALIASES.keys())
    | {
        "ml","l","mg","g","mcg","ug","iu","lsu",
        "dose","dosing","unit","units","strength",
        "solution","suspension","syrup"
    }
)

# <scripts/who_molecules.py>
# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from .text_utils import _base_name, _normalize_text_basic


def load_who_molecules(who_csv: str) -> Tuple[Dict[str, set], List[str]]:
    who = pd.read_csv(who_csv)
    who["name_base"] = who["atc_name"].fillna("").map(_base_name)
    who["name_norm"] = who["atc_name"].fillna("").map(_normalize_text_basic)
    who["name_base_norm"] = who["name_base"].map(_normalize_text_basic)

    codes_by_name = defaultdict(set)
    for _, r in who.iterrows():
        codes_by_name[r["name_base_norm"]].add(r["atc_code"])
        codes_by_name[r["name_norm"]].add(r["atc_code"])

    candidate_names = sorted(
        set(list(who["name_norm"]) + list(who["name_base_norm"])),
        key=len, reverse=True
    )
    candidate_names = [n for n in candidate_names if len(n) > 2]
    return codes_by_name, candidate_names


def detect_all_who_molecules(text: str, regex, codes_by_name) -> Tuple[List[str], List[str]]:
    if not isinstance(text, str):
        return [], []
    nt = _normalize_text_basic(text)
    names = []
    for m in regex.finditer(nt):
        detected = m.group(1)
        base = _base_name(detected)
        bn = _normalize_text_basic(base)
        names.append(bn)
    names = list(dict.fromkeys(names))
    codes = sorted(set().union(*[codes_by_name.get(n, set()) for n in names])) if names else []
    return names, codes

# <scripts/fda_ph_drug_scraper.py>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import requests

from scripts.text_utils import normalize_text
from scripts.routes_forms import FORM_TO_ROUTE, parse_form_from_text

BASE_URL = "https://verification.fda.gov.ph"
HUMAN_DRUGS_URL = f"{BASE_URL}/drug_productslist.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; eSOA-BrandMap/1.0; +https://github.com/)"
}


def fetch_csv_export() -> List[Dict[str, str]]:
    with requests.Session() as s:
        r0 = s.get(HUMAN_DRUGS_URL, headers=HEADERS, timeout=30)
        r0.raise_for_status()
        r = s.get(f"{HUMAN_DRUGS_URL}?export=csv", headers=HEADERS, timeout=120)
        r.raise_for_status()
        text = r.text

    rows: List[Dict[str, str]] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    if not rows:
        raise RuntimeError("FDA export returned no rows.")
    return rows


def normalize_columns(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    key_map = {
        "Registration Number": "registration_number",
        "Generic Name": "generic_name",
        "Brand Name": "brand_name",
        "Dosage Strength": "dosage_strength",
        "Dosage Form": "dosage_form",
        "Pharmacologic Category": "pharmacologic_category",
        "Manufacturer": "manufacturer",
        "Country of Origin": "country_of_origin",
        "Application Type": "application_type",
        "Issuance Date": "issuance_date",
        "Expiry Date": "expiry_date",
        "Product Information": "product_information",
    }
    out: List[Dict[str, str]] = []
    for r in rows:
        nr: Dict[str, str] = {}
        for k, v in r.items():
            kk = key_map.get(k, k.lower().replace(" ", "_"))
            nr[kk] = v
        out.append(nr)
    return out


def infer_form_and_route(dosage_form: Optional[str]) -> (Optional[str], Optional[str]):
    if not isinstance(dosage_form, str) or not dosage_form.strip():
        return None, None
    norm = normalize_text(dosage_form)
    form_token = parse_form_from_text(norm)
    route = FORM_TO_ROUTE.get(form_token) if form_token else None
    return form_token, route


def build_brand_map(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for r in rows:
        brand = (r.get("brand_name") or "").strip()
        generic = (r.get("generic_name") or "").strip()
        if not brand or not generic:
            continue

        dosage_form = (r.get("dosage_form") or "").strip()
        dosage_strength = (r.get("dosage_strength") or "").strip()
        regno = (r.get("registration_number") or "").strip()

        form_token, route = infer_form_and_route(dosage_form)

        key = (
            brand.lower(),
            generic.lower(),
            (form_token or "").lower(),
            (route or "").lower(),
            dosage_strength.lower(),
        )
        if key in seen:
            continue
        seen.add(key)

        out.append(
            {
                "brand_name": brand,
                "generic_name": generic,
                "dosage_form": form_token or dosage_form or "",
                "route": route or "",
                "dosage_strength": dosage_strength,
                "registration_number": regno,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FDA PH brand→generic map (CSV export only)")
    ap.add_argument("--outdir", default="inputs", help="Output directory")
    ap.add_argument("--outfile", default=None, help="Optional explicit output CSV filename")
    args = ap.parse_args()

    rows = fetch_csv_export()
    rows = normalize_columns(rows)
    brand_map = build_brand_map(rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.outfile) if args.outfile else outdir / f"fda_brand_map_{Path.cwd().name}.csv"
    # Use explicit filename if provided; otherwise the caller supplies one via subprocess args

    fieldnames = ["brand_name", "generic_name", "dosage_form", "route", "dosage_strength", "registration_number"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(brand_map)


if __name__ == "__main__":
    main()

# <scripts/brand_map.py>
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import ahocorasick  # type: ignore
import pandas as pd

from .text_utils import _base_name, _normalize_text_basic, normalize_compact


@dataclass
class BrandMatch:
    brand: str
    generic: str
    dosage_form: str
    route: str
    dosage_strength: str


def _latest_brandmap_path(inputs_dir: str) -> Optional[str]:
    # Prefer renamed pattern
    pattern_new = os.path.join(inputs_dir, "fda_brand_map_*.csv")
    candidates = glob.glob(pattern_new)
    if not candidates:
        # Backward-compatibility with old name
        pattern_old = os.path.join(inputs_dir, "brand_map_*.csv")
        candidates = glob.glob(pattern_old)
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]


def load_latest_brandmap(inputs_dir: str) -> Optional[pd.DataFrame]:
    path = _latest_brandmap_path(inputs_dir)
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        for c in ["brand_name", "generic_name", "dosage_form", "route", "dosage_strength"]:
            if c not in df.columns:
                df[c] = ""
        return df
    except Exception:
        return None


def build_brand_automata(brand_df: pd.DataFrame) -> Tuple[ahocorasick.Automaton, ahocorasick.Automaton, Dict[str, List[BrandMatch]]]:
    A_norm = ahocorasick.Automaton()
    A_comp = ahocorasick.Automaton()
    mapping: Dict[str, List[BrandMatch]] = {}
    seen_norm = set()
    seen_comp = set()
    for _, r in brand_df.iterrows():
        b = str(r.get("brand_name") or "").strip()
        g = str(r.get("generic_name") or "").strip()
        if not b or not g:
            continue
        dosage_form = str(r.get("dosage_form") or "").strip()
        route = str(r.get("route") or "").strip()
        dosage_strength = str(r.get("dosage_strength") or "").strip()
        bn = _normalize_text_basic(_base_name(b))
        bc = normalize_compact(b)
        if not bn:
            continue
        mapping.setdefault(bn, []).append(BrandMatch(
            brand=b, generic=g, dosage_form=dosage_form, route=route, dosage_strength=dosage_strength
        ))
        if bn not in seen_norm:
            A_norm.add_word(bn, bn); seen_norm.add(bn)
        if bc and bc not in seen_comp:
            A_comp.add_word(bc, bn); seen_comp.add(bc)
    A_norm.make_automaton()
    A_comp.make_automaton()
    return A_norm, A_comp, mapping


def scan_brands(text_norm: str, text_comp: str,
                A_norm: ahocorasick.Automaton,
                A_comp: ahocorasick.Automaton) -> List[str]:
    found: Dict[str, int] = {}
    for _, bn in A_norm.iter(text_norm):
        found[bn] = max(found.get(bn, 0), len(bn))
    for _, bn in A_comp.iter(text_comp):
        found[bn] = max(found.get(bn, 0), len(bn))
    return [k for k, _ in sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))]


def fda_generics_set(brand_df: pd.DataFrame) -> Set[str]:
    """Return a set of normalized base generic names present in FDA brand map."""
    gens: Set[str] = set()
    if not isinstance(brand_df, pd.DataFrame) or "generic_name" not in brand_df.columns:
        return gens
    for g in brand_df["generic_name"].fillna("").astype(str).tolist():
        gb = _normalize_text_basic(_base_name(g))
        if gb:
            gens.add(gb)
    return gens

# <main.py>
# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py — Dose/form-aware preparation + dose-aware matching.

Exports:
- prepare(pnf_csv, esoa_csv, outdir) -> (pnf_prepared_csv, esoa_prepared_csv)
- match(pnf_prepared_csv, esoa_prepared_csv, out_csv) -> out_csv
- run_all(pnf_csv, esoa_csv, outdir, out_csv) -> out_csv
"""

import argparse

from scripts.prepare import prepare
from scripts.match import match


def run_all(pnf_csv: str, esoa_csv: str, outdir: str = ".", out_csv: str = "esoa_matched.csv") -> str:
    pnf_prepared, esoa_prepared = prepare(pnf_csv, esoa_csv, outdir)
    return match(pnf_prepared, esoa_prepared, out_csv)


def _cli():
    ap = argparse.ArgumentParser(description="Dose-aware drug matching pipeline")
    ap.add_argument("--pnf", required=False, default="pnf.csv")
    ap.add_argument("--esoa", required=False, default="esoa.csv")
    ap.add_argument("--outdir", required=False, default=".")
    ap.add_argument("--out", required=False, default="esoa_matched.csv")
    args = ap.parse_args()
    run_all(args.pnf, args.esoa, args.outdir, args.out)


if __name__ == "__main__":
    _cli()

# <run.py>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — Full ESOA pipeline with on-demand spinner and timing.

Console behavior:
  • Only the spinner/timer lines and the final timing summary are printed.
  • The heavy “Match & write outputs” step uses a tqdm progress bar that starts immediately.

File outputs:
  • scripts/match_outputs.py writes ./outputs/summary.txt (overwritten each run).
  • Finally runs resolve_unknowns.py to analyze unmatched terms and write its outputs under ./outputs.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

THIS_DIR: Path = Path(__file__).resolve().parent
DEFAULT_INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
ATCD_SUBDIR = Path("dependencies") / "atcd"
ATCD_SCRIPTS: tuple[str, ...] = ("atcd.R", "export.R", "filter.R")

# Ensure local imports when called from other CWDs
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


# ----------------------------
# Utilities
# ----------------------------
def _resolve_input_path(p: str | os.PathLike[str], default_subdir: str = DEFAULT_INPUTS_DIR) -> Path:
    if not p:
        raise FileNotFoundError("No input path provided.")
    pth = Path(p)
    if pth.is_file():
        return pth
    candidate = THIS_DIR / default_subdir / pth.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Input file not found: {pth!s}. "
        f"Tried: {pth.resolve()!s} and {candidate!s}. "
        f"Place the file under ./{default_subdir}/ or pass --pnf/--esoa with a correct path."
    )


def _ensure_outputs_dir() -> Path:
    outdir = THIS_DIR / OUTPUTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _ensure_inputs_dir() -> Path:
    inp = THIS_DIR / DEFAULT_INPUTS_DIR
    inp.mkdir(parents=True, exist_ok=True)
    return inp


def _assert_all_exist(root: Path, files: Iterable[str | os.PathLike[str]]) -> None:
    for f in files:
        fp = root / f
        if not fp.is_file():
            raise FileNotFoundError(f"Required file not found: {fp}")


# ----------------------------
# Spinner + timing
# ----------------------------
def run_with_spinner(label: str, func: Callable[[], None], start_delay: float = 0.0) -> float:
    """Run func() in a worker thread with a live spinner immediately (no delay).
    Returns elapsed seconds.
    """
    done = threading.Event()
    err: list[BaseException] = []

    def worker():
        try:
            func()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()

    spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    idx = 0

    while not done.is_set():
        elapsed = time.perf_counter() - t0
        frame = spinner_frames[idx % len(spinner_frames)]
        sys.stdout.write(f"\r{frame} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)

    th.join()
    elapsed = time.perf_counter() - t0
    sys.stdout.write(f"\r✓ {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()

    if err:
        raise err[0]
    return elapsed


# ----------------------------
# Steps (silent)
# ----------------------------
def install_requirements(req_path: str | os.PathLike[str]) -> None:
    req = Path(req_path) if req_path else None
    if not req or not req.is_file():
        return
    with open(os.devnull, "w") as devnull:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)],
            cwd=str(THIS_DIR),
            stdout=devnull,
            stderr=devnull,
        )


def run_r_scripts() -> None:
    atcd_dir = THIS_DIR / ATCD_SUBDIR
    if not atcd_dir.is_dir():
        return
    out_dir = atcd_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    rscript = shutil.which("Rscript")
    if not rscript:
        return
    try:
        _assert_all_exist(atcd_dir, ATCD_SCRIPTS)
    except FileNotFoundError:
        return
    with open(os.devnull, "w") as devnull:
        for script in ATCD_SCRIPTS:
            subprocess.run([rscript, script], check=True, cwd=str(atcd_dir), stdout=devnull, stderr=devnull)


def create_master_file(root_dir: Path) -> None:
    """Silent; best-effort concatenation for debugging (no console output)."""
    debug_dir = root_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = debug_dir / "master.py"

    files_to_concatenate = [
        root_dir / "scripts" / "aho.py",
        root_dir / "scripts" / "combos.py",
        root_dir / "scripts" / "dose.py",
        root_dir / "scripts" / "match_features.py",
        root_dir / "scripts" / "match_scoring.py",
        root_dir / "scripts" / "match_outputs.py",
        root_dir / "scripts" / "match.py",
        root_dir / "scripts" / "prepare.py",
        root_dir / "scripts" / "routes_forms.py",
        root_dir / "scripts" / "text_utils.py",
        root_dir / "scripts" / "who_molecules.py",
        root_dir / "scripts" / "fda_ph_drug_scraper.py",
        root_dir / "scripts" / "brand_map.py",
        root_dir / "main.py",
        root_dir / "run.py",
    ]

    header_text = "# START OF REPO FILES"
    footer_text = "# END OF REPO FILES"

    try:
        with output_file_path.open("w", encoding="utf-8", newline="\n") as outfile:
            outfile.write(header_text + "\n")
            for file_path in files_to_concatenate:
                if not file_path.is_file():
                    continue
                relative_path = file_path.relative_to(root_dir).as_posix()
                outfile.write(f"# <{relative_path}>\n")
                outfile.write(file_path.read_text(encoding="utf-8", errors="ignore"))
                outfile.write("\n")
            outfile.write(footer_text + "\n")
    except Exception:
        pass


def build_brand_map(inputs_dir: Path, outfile: Path | None) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_csv = outfile or (inputs_dir / f"fda_brand_map_{date_str}.csv")
    with open(os.devnull, "w") as devnull:
        subprocess.run(
            [sys.executable, "-m", "scripts.fda_ph_drug_scraper", "--outdir", str(inputs_dir), "--outfile", str(out_csv)],
            check=True,
            cwd=str(THIS_DIR),
            stdout=devnull,
            stderr=devnull,
        )
    return out_csv


def run_resolve_unknowns() -> None:
    """Run resolve_unknowns.py if present (either at project root or under scripts/)."""
    # Prefer scripts/resolve_unknowns.py when available, otherwise fallback to root-level resolve_unknowns.py
    mod_name = None
    if (THIS_DIR / "scripts" / "resolve_unknowns.py").is_file():
        mod_name = "scripts.resolve_unknowns"
    elif (THIS_DIR / "resolve_unknowns.py").is_file():
        mod_name = "resolve_unknowns"
    else:
        # If the script doesn't exist, nothing to do
        return
    with open(os.devnull, "w") as devnull:
        subprocess.run(
            [sys.executable, "-m", mod_name],
            check=True,
            cwd=str(THIS_DIR),
            stdout=devnull,
            stderr=devnull,
        )


# ----------------------------
# Main entry
# ----------------------------
def main_entry() -> None:
    parser = argparse.ArgumentParser(
        description="Run full ESOA pipeline (ATC → brand map → prepare → match) with spinner+timing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pnf", default=f"{DEFAULT_INPUTS_DIR}/pnf.csv", help="Path to PNF CSV")
    parser.add_argument("--esoa", default=f"{DEFAULT_INPUTS_DIR}/esoa.csv", help="Path to eSOA CSV")
    parser.add_argument("--out", default="esoa_matched.csv", help="Output CSV filename (saved under ./outputs)")
    parser.add_argument("--requirements", default="requirements.txt", help="Requirements file to install")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install of requirements")
    parser.add_argument("--skip-r", action="store_true", help="Skip running ATC R preprocessing scripts")
    parser.add_argument("--skip-brandmap", action="store_true", help="Skip building FDA brand map CSV")
    args = parser.parse_args()

    # Silent helper
    create_master_file(THIS_DIR)

    outdir = _ensure_outputs_dir()
    inputs_dir = _ensure_inputs_dir()
    pnf_path = _resolve_input_path(args.pnf)
    esoa_path = _resolve_input_path(args.esoa)
    out_path = outdir / Path(args.out).name

    from scripts.prepare import prepare as _prepare
    from scripts.match import match as _match

    timings: list[tuple[str, float]] = []

    if not args.skip_install and args.requirements:
        t = run_with_spinner("Install requirements", lambda: install_requirements(args.requirements))
        timings.append(("Install requirements", t))

    if not args.skip_r:
        t = run_with_spinner("ATC R preprocessing", run_r_scripts)
        timings.append(("ATC R preprocessing", t))

    if not args.skip_brandmap:
        t = run_with_spinner("Build FDA brand map", lambda: build_brand_map(inputs_dir, outfile=None))
        timings.append(("Build FDA brand map", t))

    t = run_with_spinner("Prepare inputs", lambda: _prepare(str(pnf_path), str(esoa_path), str(inputs_dir)))
    timings.append(("Prepare inputs", t))

    # Let tqdm own the console for matching; no outer spinner here.
    t0 = time.perf_counter()
    _match(
        str(inputs_dir / "pnf_prepared.csv"),
        str(inputs_dir / "esoa_prepared.csv"),
        str(out_path),
    )
    t_match = time.perf_counter() - t0
    print(f"✓ {t_match:7.2f}s Match & write outputs")
    timings.append(("Match & write outputs", t_match))

    t = run_with_spinner("Resolve unknowns", run_resolve_unknowns)
    timings.append(("Resolve unknowns", t))

    # Final timing summary (console only)
    print("\n=== Timing Summary ===")
    total = 0.0
    for name, secs in timings:
        print(f"• {name:<24} {secs:9.2f}s")
        total += secs
    print(f"{'-'*38}\n• Total{'':<21} {total:9.2f}s")


if __name__ == "__main__":
    try:
        main_entry()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise

# END OF REPO FILES
