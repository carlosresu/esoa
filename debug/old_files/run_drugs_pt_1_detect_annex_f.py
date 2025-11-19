from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import duckdb
import pandas as pd

try:
    import ahocorasick
except ImportError:  # pragma: no cover - optional dependency
    ahocorasick = None


def _as_str(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return text


def normalize_text(value: str | float | None) -> str:
    text = _as_str(value).strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,;:-_")
    return text


_FORM_MAP = {
    "tab": "tablet",
    "tabs": "tablet",
    "tablet": "tablet",
    "tablets": "tablet",
    "cap": "capsule",
    "caps": "capsule",
    "capsule": "capsule",
    "capsules": "capsule",
    "sol": "solution",
    "solution": "solution",
    "susp": "suspension",
    "suspension": "suspension",
    "inj": "injection",
    "injection": "injection",
    "cream": "cream",
    "ointment": "ointment",
    "syrup": "syrup",
    "elixir": "elixir",
    "spray": "spray",
    "patch": "patch",
    "powder": "powder",
    "granule": "granule",
    "gel": "gel",
    "lotion": "lotion",
    "drops": "drops",
    "drop": "drops",
    "aerosol": "aerosol",
    "lozenge": "lozenge",
    "suppository": "suppository",
}


def canonical_form(raw: str | float | None) -> str:
    text = normalize_text(raw)
    if not text:
        return ""
    return _FORM_MAP.get(text, text)


_ROUTE_MAP = {
    "po": "oral",
    "per os": "oral",
    "oral": "oral",
    "iv": "intravenous",
    "intravenous": "intravenous",
    "im": "intramuscular",
    "intramuscular": "intramuscular",
    "sc": "subcutaneous",
    "subcutaneous": "subcutaneous",
    "sub cutaneous": "subcutaneous",
    "sub-cutaneous": "subcutaneous",
    "subq": "subcutaneous",
    "sub q": "subcutaneous",
    "topical": "topical",
    "ophthalmic": "ophthalmic",
    "otic": "otic",
    "sublingual": "sublingual",
    "transdermal": "transdermal",
    "nasal": "nasal",
    "buccal": "buccal",
    "rectal": "rectal",
    "vaginal": "vaginal",
    "intrathecal": "intrathecal",
    "inhalation": "inhalation",
    "respiratory": "respiratory",
}


def canonical_route(raw: str | float | None) -> str:
    text = normalize_text(raw).replace(".", "")
    if not text:
        return ""
    return _ROUTE_MAP.get(text, text)


_SALT_MAP = {
    "hcl": "hydrochloride",
    "hydrochlorid": "hydrochloride",
}


def canonical_salt_form(raw: str | float | None) -> str:
    text = normalize_text(raw)
    if not text:
        return ""
    return _SALT_MAP.get(text, text)


def split_multi(raw: str | float | None) -> List[str]:
    text = _as_str(raw)
    if not text:
        return []
    parts = re.split(r"[|;,]", text)
    return [p.strip() for p in parts if p and p.strip()]


class LexiconMatcher:
    def __init__(self, lexicon: Dict[str, object]):
        self.lexicon = lexicon
        self.automaton = None
        if ahocorasick and lexicon:
            automaton = ahocorasick.Automaton()
            for term, payload in lexicon.items():
                automaton.add_word(term, payload)
            automaton.make_automaton()
            self.automaton = automaton

    def iter(self, text: str) -> Iterable[Tuple[int, object]]:
        if self.automaton is not None:
            yield from self.automaton.iter(text)
            return
        for term, payload in self.lexicon.items():
            if term and term in text:
                yield (-1, payload)


def parse_dose(text: str) -> Dict[str, Optional[str]]:
    match = re.search(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|kg|%)", text, re.IGNORECASE)
    strength = match.group("val") if match else ""
    unit = match.group("unit") if match else ""
    per_val = ""
    per_unit = ""
    pct = ""
    if unit and unit.strip().lower() == "%":
        pct = strength
        strength = ""
        unit = ""
    ratio_match = re.search(r"(mg|mcg|g)\s*/\s*(\d+(?:\.\d+)?)\s*(ml|l)", text, re.IGNORECASE)
    if ratio_match:
        unit = ratio_match.group(1)
        per_val = ratio_match.group(2)
        per_unit = ratio_match.group(3)
        strength = "1"
    return {
        "parsed_strength": strength,
        "parsed_unit": unit,
        "parsed_per_val": per_val,
        "parsed_per_unit": per_unit,
        "parsed_pct": pct,
    }


def build_generic_lexicon(con: duckdb.DuckDBPyConnection) -> Dict[str, Set[int]]:
    lex = defaultdict(set)
    queries = [
        ("dim_molecule_set", "canonical_generic_name"),
        ("dim_pnf_generic", "generic_name"),
        ("dim_drugbank_generic", "generic"),
    ]
    for table, column in queries:
        df = con.execute(
            f"""
            SELECT molecule_set_id, {column}
            FROM {table}
            WHERE molecule_set_id IS NOT NULL AND {column} IS NOT NULL AND {column} <> ''
            """
        ).df()
        for _, row in df.iterrows():
            key = normalize_text(row[column])
            if key:
                lex[key].add(int(row["molecule_set_id"]))
    return dict(lex)


def build_component_map(con: duckdb.DuckDBPyConnection) -> Dict[int, List[str]]:
    df = con.execute(
        """
        SELECT b.molecule_set_id, m.molecule_name
        FROM bridge_molecule_set_member b
        JOIN dim_molecule m ON m.molecule_id = b.molecule_id
        WHERE m.molecule_name IS NOT NULL
        """
    ).df()
    comp: Dict[int, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        comp[int(row["molecule_set_id"])].append(row["molecule_name"])
    return comp


def build_simple_lookup(df: pd.DataFrame, key_col: str, value_col: str) -> Dict[int, str]:
    return {
        int(row[key_col]): row[value_col]
        for _, row in df.iterrows()
        if pd.notna(row[key_col]) and pd.notna(row[value_col])
    }


def detect_in_text(
    text: str,
    generic_matcher: LexiconMatcher,
) -> Dict[str, List[int]]:
    norm = normalize_text(text)
    molecule_set_ids: Set[int] = set()

    for _, payload in generic_matcher.iter(norm):
        for molecule_set_id in payload:
            molecule_set_ids.add(int(molecule_set_id))

    return {"molecule_set_ids": sorted(molecule_set_ids)}


def expand_pipe(values: Iterable[int]) -> str:
    return "|".join(str(v) for v in sorted(set(values)) if v is not None)


def join_names(values: Iterable[str]) -> str:
    return "|".join(sorted({v for v in values if v}))


def join_ints(values: Iterable[int]) -> str:
    unique = sorted({int(v) for v in values if v is not None})
    return "|".join(str(v) for v in unique)


def parse_form_hint(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", text)
    for token in tokens:
        canonical = canonical_form(token)
        if canonical:
            return canonical
    return ""


def parse_route_hint(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", text)
    for token in tokens:
        canonical = canonical_route(token)
        if canonical:
            return canonical
    return ""


def parse_salt_hint(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z-]+", text)
    for token in tokens:
        canonical = canonical_salt_form(token)
        if canonical:
            return canonical
    return ""


def main() -> None:
    root = Path(__file__).resolve().parent
    annex_path = root / "inputs" / "drugs" / "annex_f.csv"
    output_path = root / "outputs" / "drugs" / "annex_f_enriched.csv"
    db_path = root / "outputs" / "drugs" / "drugs_master.duckdb"

    if not annex_path.is_file():
        raise FileNotFoundError(f"Annex F CSV missing: {annex_path}")
    if not db_path.is_file():
        raise FileNotFoundError(f"DuckDB not found: {db_path}. Run run_drugs_pt_0b_load_master_to_duckdb.py first.")

    con = duckdb.connect(str(db_path))

    print("Building generic lexicon...")
    generic_lex = build_generic_lexicon(con)
    generic_matcher = LexiconMatcher(generic_lex)

    print("Loading supporting dimensions...")
    dim_molecule_set = build_simple_lookup(
        con.execute(
            "SELECT molecule_set_id, canonical_generic_name FROM dim_molecule_set"
        ).df(),
        "molecule_set_id",
        "canonical_generic_name",
    )
    component_map = build_component_map(con)


    chunk_iter = pd.read_csv(annex_path, dtype=str, chunksize=5000)
    enriched_chunks: List[pd.DataFrame] = []
    total_rows = 0
    rows_with_molecules = 0

    for chunk in chunk_iter:
        chunk = chunk.rename(columns={
            "Drug Code": "drug_code",
            "Drug Description": "raw_description",
        })

        records: List[Dict[str, object]] = []
        for _, row in chunk.iterrows():
            total_rows += 1
            description = row.get("raw_description", "") or ""
            detection = detect_in_text(description, generic_matcher)
            molecule_set_ids = detection["molecule_set_ids"]
            if molecule_set_ids:
                rows_with_molecules += 1

            canonical_names = [dim_molecule_set.get(ms_id, "") for ms_id in molecule_set_ids]
            components = []
            for ms_id in molecule_set_ids:
                components.extend(component_map.get(ms_id, []))

            dose_info = parse_dose(description)
            form_name = parse_form_hint(description)
            route_name = parse_route_hint(description)
            salt_name = parse_salt_hint(description)

            record = {
                "drug_code": row.get("drug_code", ""),
                "raw_description": description,
                "detected_molecule_set_ids": join_ints(molecule_set_ids),
                "canonical_generic_names": join_names(filter(None, canonical_names)),
                "component_molecules": join_names(components),
                **dose_info,
                "parsed_form_name": form_name,
                "parsed_route_name": route_name,
                "parsed_salt_form_name": salt_name,
            }
            records.append(record)

        enriched_chunks.append(pd.DataFrame(records))

    result = pd.concat(enriched_chunks, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Processed {total_rows} Annex F rows.")
    print(f"Rows with detected molecule sets: {rows_with_molecules}")
    print(f"Enriched output written to {output_path}")


if __name__ == "__main__":
    main()
