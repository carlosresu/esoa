#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalize Annex F drug catalogue entries into a structured CSV that mirrors the
prepared PNF schema (dose, route, form) while surfacing the Annex F Drug Code as
the primary identifier.

The heuristics favour safe fallbacks: we only infer routes when the dosage form
or packaging strongly implies a modality (e.g., ampules/vials → intravenous,
tablets/capsules → oral).  Remaining ambiguities are left blank so downstream
reviewers can resolve them explicitly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ..constants import PIPELINE_INPUTS_DIR, PROJECT_ROOT
from .dose_drugs import parse_dose_struct_from_text, safe_ratio_mg_per_ml, to_mg
from .routes_forms_drugs import FORM_TO_ROUTE, extract_route_and_form, parse_form_from_text
from .text_utils_drugs import normalize_text, slug_id
from .combos_drugs import SALT_TOKENS

# Containers observed in Annex F (canonical form -> token variants)
# Recognized containers observed in Annex F (canonical form -> token variants).
CONTAINER_ALIASES = {
    "ampule": {"ampule", "ampul", "ampoule", "amp", "ampu"},
    "vial": {"vial", "vialx"},
    "bottle": {"bottle", "bot", "bottl"},
    "bag": {"bag", "bagxx"},
    "sachet": {"sachet", "sachets"},
    "capsule": {"capsule", "capsules", "cap", "caps"},
    "tablet": {"tablet", "tablets", "tab", "tabs", "tabx"},
    "tube": {"tube", "tub", "tubes"},
    "drops": {"drop", "drops"},
    "patch": {"patch", "patches"},
    "syringe": {"syringe", "syringes"},
    "nebule": {"nebule", "nebules"},
}

# Map volume/weight tokens to canonical units.
UNIT_ALIASES = {
    "ml": {"ml", "milliliter", "milliliters"},
    "l": {"l", "liter", "litre", "liters", "litres"},
    "g": {"g", "gram", "grams"},
    "mg": {"mg"},
    "mcg": {"mcg", "ug"},
}

# Reverse lookups for quick normalization during token scans.
UNIT_NORMAL = {alias: base for base, aliases in UNIT_ALIASES.items() for alias in aliases}
CONTAINER_NORMAL = {alias: base for base, aliases in CONTAINER_ALIASES.items() for alias in aliases}

# Tokens that should never be stripped from molecule names when building the Annex F generic.
SALT_WHITELIST = set(SALT_TOKENS)

# Additional words that add no value to the canonical molecule string.
GENERIC_STOPWORDS = {
    "ml",
    "l",
    "mg",
    "mcg",
    "ug",
    "g",
    "iu",
    "lsu",
    "per",
    "as",
    "with",
    "and",
    "w",
    "v",
    "w/v",
    "w/w",
    "x",
    "per",
    "dose",
    "doses",
    "sol",
    "soln",
    "susp",
    "syr",
    "usp",
    "bp",
    "ep",
    "nf",
}

GENERIC_STOPWORDS |= {  # forms/vehicles that do not affect the molecule identity
    "solution",
    "suspension",
    "syrup",
    "powder",
    "cream",
    "ointment",
    "gel",
    "lotion",
    "spray",
    "drops",
    "drop",
    "nebule",
    "neb",
}

# Compile regexes once so the preparation pass stays fast enough for CLI usage.
# Token regex helpers reused during parsing.
DIGIT_ONLY_RX = re.compile(r"^\d+(?:\.\d+)?$")
UNIT_FRAGMENT_RX = re.compile(r"(?:mg|mcg|ug|g|iu|lsu|ml|l|%)", re.I)


@dataclass(frozen=True)
class _NameEntry:
    tokens: Tuple[str, ...]
    canonical: str
    priority: int
    source: str


class _ReferenceNameResolver:
    """Token-based matching between Annex F free text and reference catalogues."""

    def __init__(self) -> None:
        self.token_index: Dict[str, List[_NameEntry]] = {}
        self._seen: set[Tuple[Tuple[str, ...], str]] = set()

    def register(self, label: str, canonical: Optional[str], priority: int, source: str) -> None:
        canonical = (canonical or label).strip()
        if not canonical:
            return
        tokens = tuple(normalize_text(label).split())
        if not tokens:
            return
        key = (tokens, canonical.lower())
        if key in self._seen:
            return
        self._seen.add(key)
        entry = _NameEntry(tokens=tokens, canonical=canonical, priority=priority, source=source)
        bucket = self.token_index.setdefault(tokens[0], [])
        bucket.append(entry)

    def resolve(self, raw_text: str) -> Optional[_NameEntry]:
        tokens = tuple(normalize_text(raw_text).split())
        if not tokens:
            return None
        best: Optional[_NameEntry] = None
        # Prefer longest span, then highest priority (lower number), then earliest position.
        best_score: Tuple[int, int, int] = (-1, 0, 0)
        for idx, tok in enumerate(tokens):
            bucket = self.token_index.get(tok)
            if not bucket:
                continue
            for entry in bucket:
                span = len(entry.tokens)
                if idx + span > len(tokens):
                    continue
                if entry.tokens != tokens[idx : idx + span]:
                    continue
                score = (span, -entry.priority, -idx)
                if score > best_score:
                    best = entry
                    best_score = score
        return best


def _read_csv(path: Path, usecols: Iterable[str]) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, usecols=list(usecols), dtype=str).fillna("")
    except Exception:
        return None
    return frame


def _register_column_values(
    resolver: _ReferenceNameResolver,
    path: Path,
    column: str,
    priority: int,
    source: str,
) -> bool:
    frame = _read_csv(path, [column])
    if frame is None or column not in frame.columns:
        return False
    for value in frame[column].dropna().astype(str):
        clean = value.strip()
        if not clean:
            continue
        resolver.register(clean, clean, priority, source)
    return True


def _register_alias_pairs(
    resolver: _ReferenceNameResolver,
    path: Path,
    alias_col: str,
    canonical_col: str,
    priority: int,
    source: str,
) -> None:
    frame = _read_csv(path, [alias_col, canonical_col])
    if frame is None:
        return
    for _, row in frame.iterrows():
        alias = str(row.get(alias_col, "") or "").strip()
        canonical = str(row.get(canonical_col, "") or "").strip()
        if not canonical:
            continue
        resolver.register(canonical, canonical, max(priority - 1, 0), source)
        if alias:
            resolver.register(alias, canonical, priority, source)


CUSTOM_ALIAS_ENTRIES: Tuple[Tuple[str, str, int], ...] = (
    ("Vitamins Intravenous Fat-Soluble", "Vitamins Intravenous, Fat-Soluble", 0),
    ("Vitamins Intravenous Trace Elements", "Vitamins Intravenous, Trace Elements", 0),
    ("Vitamins Intravenous Water-Soluble", "Vitamins Intravenous, Water-Soluble", 0),
    ("Vitamins Intravenous", "Vitamins Intravenous", 1),
    ("Dextrose in Lactated Ringer's Solution", "Dextrose in Lactated Ringer's Solution", 0),
)


@lru_cache(maxsize=1)
def _reference_name_resolver() -> _ReferenceNameResolver:
    resolver = _ReferenceNameResolver()

    # PNF prepared (preferred) then fallback to raw PNF.
    pnf_prepared = PIPELINE_INPUTS_DIR / "pnf_prepared.csv"
    if not _register_column_values(resolver, pnf_prepared, "generic_name", 0, "pnf_prepared"):
        _register_column_values(resolver, PIPELINE_INPUTS_DIR / "pnf.csv", "Molecule", 1, "pnf_raw")

    # WHO ATC molecule lists (allow multiple dated files).
    for path in sorted(PIPELINE_INPUTS_DIR.glob("who_atc*_molecules.csv")):
        _register_column_values(resolver, path, "atc_name", 5, f"who:{path.name}")

    # FDA brand map(s) contribute both generic and brand aliases.
    for path in sorted(PIPELINE_INPUTS_DIR.glob("fda_brand_map*.csv")):
        _register_alias_pairs(resolver, path, "brand_name", "generic_name", 3, f"fda:{path.name}")

    # DrugBank generics/brands (prefer freshly exported dependencies, then inputs copies).
    drugbank_candidates = [
        PROJECT_ROOT / "dependencies" / "drugbank_generics" / "output" / "drugbank_generics.csv",
        PIPELINE_INPUTS_DIR / "drugbank_generics.csv",
        PIPELINE_INPUTS_DIR / "generics.csv",
    ]
    for path in drugbank_candidates:
        _register_column_values(resolver, path, "generic", 4, f"drugbank:{path.name}")

    drugbank_brand_candidates = [
        PROJECT_ROOT / "dependencies" / "drugbank_generics" / "output" / "drugbank_brands.csv",
        PIPELINE_INPUTS_DIR / "drugbank_brands.csv",
    ]
    for path in drugbank_brand_candidates:
        _register_alias_pairs(resolver, path, "brand", "generic", 4, f"drugbank_brand:{path.name}")

    for alias, canonical, priority in CUSTOM_ALIAS_ENTRIES:
        resolver.register(alias, canonical, priority, "custom")

    return resolver


def _fallback_generic_name(raw_desc: str) -> str:
    """Strip dose/pack cues to surface the base Annex F molecule name when references miss."""
    if not isinstance(raw_desc, str):
        return ""
    norm = normalize_text(raw_desc)
    norm = norm.replace("+", " + ").replace("/", " / ")
    tokens: list[str] = []
    for tok in norm.split():
        if not tok or tok == "/":
            continue
        if DIGIT_ONLY_RX.fullmatch(tok):
            continue
        if tok in GENERIC_STOPWORDS and tok not in SALT_WHITELIST:
            continue
        if tok in CONTAINER_NORMAL:
            continue
        if tok in UNIT_NORMAL:
            continue
        if UNIT_FRAGMENT_RX.search(tok) and not tok.isalpha() and tok.lower() not in SALT_WHITELIST:
            continue
        if re.search(r"\d", tok):
            continue
        tokens.append(tok)

    cleaned: list[str] = []
    prev_plus = False
    for tok in tokens:
        if tok == "+":
            if not cleaned or prev_plus:
                continue
            prev_plus = True
            cleaned.append(tok)
            continue
        prev_plus = False
        cleaned.append(tok)

    while cleaned and cleaned[-1] == "+":
        cleaned.pop()
    while cleaned and cleaned[0] == "+":
        cleaned.pop(0)
    if not cleaned:
        # Fall back to the raw uppercase name when stripping produced nothing.
        return raw_desc.strip().upper()
    return re.sub(r"\s+\+\s+", " + ", " ".join(cleaned)).upper()


def _vitamin_descriptor(normalized_desc: str) -> Optional[str]:
    if "vitamins intravenous" not in normalized_desc:
        return None
    if "trace elements" in normalized_desc:
        return "Trace Elements"
    if "fat soluble" in normalized_desc:
        return "Fat-Soluble"
    if "water soluble" in normalized_desc:
        return "Water-Soluble"
    return None


def _derive_generic_name(
    raw_desc: str,
    normalized_desc: str,
    resolver: _ReferenceNameResolver,
) -> str:
    custom_descriptor = _vitamin_descriptor(normalized_desc)
    resolved = resolver.resolve(raw_desc) if isinstance(raw_desc, str) else None
    if resolved:
        name = resolved.canonical
    else:
        name = _fallback_generic_name(raw_desc)
    if custom_descriptor and custom_descriptor.lower() not in name.lower():
        name = f"{name} ({custom_descriptor})"
    return name.upper()


PACK_FREETEXT_SKIP = {
    "sterile",
    "sterilized",
    "solution",
    "suspension",
    "syrup",
    "powder",
    "cream",
    "ointment",
    "gel",
    "lotion",
    "topical",
    "oral",
    "for",
    "concentrate",
    "drops",
    "drop",
}


def _scan_packaging(tokens: Iterable[str]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Walk tokens from the tail to capture '[qty] [unit] [container]' patterns."""
    tokens = list(tokens)
    qty: Optional[float] = None
    unit: Optional[str] = None
    container: Optional[str] = None

    for i in range(len(tokens) - 1, -1, -1):
        tok = tokens[i]
        base_container = CONTAINER_NORMAL.get(tok)
        if base_container:
            container = base_container
            # Try to consume unit + quantity immediately preceding the container token.
            idx = i - 1
            # Skip filler words between the container and the numeric/unit pair (e.g., "solution").
            while idx >= 0 and tokens[idx] in PACK_FREETEXT_SKIP:
                idx -= 1
            if idx >= 0:
                unit_candidate = UNIT_NORMAL.get(tokens[idx])
                if unit_candidate:
                    unit = unit_candidate
                    if idx - 1 >= 0:
                        try:
                            qty = float(tokens[idx - 1])
                        except ValueError:
                            qty = None
                    continue
            # Alternate pattern: quantity directly before the container without explicit unit.
            if idx >= 0:
                try:
                    qty = float(tokens[idx])
                except ValueError:
                    qty = None
            break

    return qty, unit, container


def _normalize_pack_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    if unit == "l":
        return "ml"
    return unit


def _deduce_route_form(
    normalized: str,
    pack_container: Optional[str],
) -> Tuple[Optional[str], Optional[str], str]:
    """Combine text-derived clues with packaging hints to infer form + route."""
    form_primary = parse_form_from_text(normalized)
    route_primary, form_secondary, base_evidence = extract_route_and_form(normalized)
    evidence_parts = [part for part in base_evidence.split(";") if part]
    evidence_seen = set(evidence_parts)

    def _add_evidence(tag: str) -> None:
        if not tag or tag in evidence_seen:
            return
        evidence_seen.add(tag)
        evidence_parts.append(tag)

    form_token = form_primary or form_secondary
    route_token = route_primary
    norm_text = normalized.lower()

    def _register_form(form: str, route: Optional[str], reason: str) -> None:
        nonlocal form_token, route_token
        if form and not form_token:
            form_token = form
            _add_evidence(f"{reason}:form={form}")
        if route and (not route_token or (form and form.startswith("eye"))):
            route_token = route
            _add_evidence(f"{reason}:route={route}")

    keyword_forms: tuple[tuple[str, str, Optional[str]], ...] = (
        ("eye drops", "eye drops", "ophthalmic"),
        ("eye drop", "eye drops", "ophthalmic"),
        ("ear drops", "ear drops", "otic"),
        ("ear drop", "ear drops", "otic"),
        ("nasal drops", "nasal drops", "nasal"),
        ("nasal drop", "nasal drops", "nasal"),
        ("oral drops", "oral drops", "oral"),
        ("ovule", "ovule", "vaginal"),
        ("ovules", "ovule", "vaginal"),
        ("shampoo", "shampoo", "topical"),
        ("soap", "soap", "topical"),
        ("wash", "wash", "topical"),
        ("granules", "granule", "oral"),
        ("granule", "granule", "oral"),
        ("lozenge", "lozenge", "oral"),
        ("mouthwash", "mouthwash", "oral"),
    )

    if not form_token:
        for token, canonical, route in keyword_forms:
            if token in norm_text:
                _register_form(canonical, route or FORM_TO_ROUTE.get(canonical), f"keyword:{token}")
                break

    plural_map = (
        ("solutions", "solution"),
        ("suspensions", "suspension"),
        ("syrups", "syrup"),
        ("lotions", "lotion"),
        ("creams", "cream"),
        ("ointments", "ointment"),
    )
    if not form_token:
        for token, canonical in plural_map:
            if token in norm_text:
                _register_form(canonical, FORM_TO_ROUTE.get(canonical), f"keyword:{token}->{canonical}")
                break

    if not form_token and "solution" in norm_text:
        _register_form("solution", FORM_TO_ROUTE.get("solution"), "keyword:solution")
    if not form_token and "suspension" in norm_text:
        _register_form("suspension", FORM_TO_ROUTE.get("suspension"), "keyword:suspension")
    if not form_token and "syrup" in norm_text:
        _register_form("syrup", FORM_TO_ROUTE.get("syrup"), "keyword:syrup")

    # Packaging overrides: ampules/vials are injectable, nebules inhalation, etc.
    if pack_container in {"ampule", "vial"}:
        form_token = pack_container
        if route_token not in {"intravenous", "intramuscular", "subcutaneous"}:
            route_token = "intravenous"
            _add_evidence(f"packaging->{pack_container}:intravenous")
    elif pack_container == "nebule":
        form_token = "nebule"
        route_token = "inhalation"
        _add_evidence("packaging->nebule:inhalation")
    elif pack_container == "syringe":
        route_token = "intravenous"
        _add_evidence("packaging->syringe:intravenous")
    elif pack_container == "drops":
        if "eye" in norm_text:
            _register_form("eye drops", "ophthalmic", "packaging:drops_eye")
        elif "ear" in norm_text:
            _register_form("ear drops", "otic", "packaging:drops_ear")
        elif "nasal" in norm_text:
            _register_form("nasal drops", "nasal", "packaging:drops_nasal")
        else:
            _register_form("oral drops", "oral", "packaging:drops_oral")
    elif pack_container in {"bottle", "bag"}:
        if "intravenous" in norm_text or "iv" in norm_text.split():
            route_token = "intravenous"
            _add_evidence("text->intravenous")
        elif form_token in {"solution"}:
            # Large-volume solutions in bottles/bags typically indicate parenteral use.
            route_token = "intravenous"
            _add_evidence(f"packaging->{pack_container}:assume_intravenous")

    if not route_token and form_token in FORM_TO_ROUTE:
        route_token = FORM_TO_ROUTE[form_token]
        _add_evidence(f"form_impute->{form_token}:{route_token}")

    return route_token, form_token, ";".join(evidence_parts)


def prepare_annex_f(input_csv: str, output_csv: str) -> str:
    """Entry point used by CLI/automation to normalize Annex F into CSV form."""
    source_path = Path(input_csv)
    if not source_path.is_file():
        raise FileNotFoundError(f"Annex F CSV not found: {input_csv}")

    frame = pd.read_csv(source_path, dtype=str).fillna("")
    if "Drug Code" not in frame.columns or "Drug Description" not in frame.columns:
        raise ValueError("annex_f.csv must contain 'Drug Code' and 'Drug Description' columns")

    resolver = _reference_name_resolver()
    records = []
    for raw_code, raw_desc in frame[["Drug Code", "Drug Description"]].itertuples(index=False):
        desc = (raw_desc or "").strip()
        norm = normalize_text(desc)
        tokens = norm.split()
        pack_qty, pack_unit, pack_container = _scan_packaging(tokens)
        if pack_unit == "l" and pack_qty is not None and not math.isnan(pack_qty):
            pack_qty *= 1000.0
        pack_unit = _normalize_pack_unit(pack_unit)

        parsed_dose = parse_dose_struct_from_text(norm)
        dose_kind = parsed_dose.get("dose_kind")
        strength = parsed_dose.get("strength")
        unit = parsed_dose.get("unit")
        per_val = parsed_dose.get("per_val")
        per_unit = parsed_dose.get("per_unit")
        pct = parsed_dose.get("pct")

        strength_mg = to_mg(strength, unit) if strength is not None else None
        ratio_mg_per_ml = (
            safe_ratio_mg_per_ml(strength, unit, per_val)
            if dose_kind == "ratio" and strength is not None
            else None
        )

        route_allowed, form_token, route_evidence = _deduce_route_form(norm, pack_container)

        generic_name = _derive_generic_name(desc, norm, resolver)
        generic_id = slug_id(generic_name) if generic_name else ""

        records.append(
            {
                "drug_code": str(raw_code).strip(),
                "generic_id": generic_id,
                "generic_name": generic_name,
                "raw_description": desc,
                "normalized_description": norm,
                "dose_kind": dose_kind,
                "strength": strength,
                "unit": unit,
                "per_val": per_val,
                "per_unit": per_unit,
                "pct": pct,
                "strength_mg": strength_mg,
                "ratio_mg_per_ml": ratio_mg_per_ml,
                "route_allowed": route_allowed or "",
                "form_token": form_token or "",
                "route_evidence": route_evidence,
                "pack_quantity": pack_qty,
                "pack_unit": pack_unit,
                "pack_container": pack_container or "",
            }
        )

    out_frame = pd.DataFrame.from_records(records)
    out_frame.to_csv(output_csv, index=False, encoding="utf-8")
    return str(Path(output_csv).resolve())


def main() -> None:
    """CLI wrapper to allow `python -m pipelines.drugs.scripts.prepare_annex_f_drugs` execution."""
    inputs_dir = PIPELINE_INPUTS_DIR
    input_csv = inputs_dir / "annex_f.csv"
    output_csv = inputs_dir / "annex_f_prepared.csv"
    path = prepare_annex_f(str(input_csv), str(output_csv))
    print(f"Wrote Annex F prepared CSV: {path}")


if __name__ == "__main__":
    main()
