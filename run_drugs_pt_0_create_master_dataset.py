from __future__ import annotations

import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pipelines.drugs.pipeline import DrugsAndMedicinePipeline
from pipelines.drugs.scripts.prepare_drugs import prepare


def _run_subprocess(cmd: List[str], cwd: Path) -> None:
    subprocess.check_call(cmd, cwd=str(cwd))


def _natural_esoa_part_order(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    index = int(match.group(1)) if match else sys.maxsize
    return index, path.name


def _concatenate_csv(parts: List[Path], dest: Path) -> None:
    header: List[str] | None = None
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as out_handle:
        writer: csv.writer | None = None
        for part in parts:
            if not part.is_file():
                continue
            with part.open("r", newline="", encoding="utf-8-sig") as in_handle:
                reader = csv.reader(in_handle)
                try:
                    file_header = next(reader)
                except StopIteration:
                    continue
                if header is None:
                    header = file_header
                    writer = csv.writer(out_handle)
                    writer.writerow(header)
                elif file_header != header:
                    raise ValueError(
                        f"Header mismatch while concatenating {part.name}; expected {header} but found {file_header}."
                    )
                assert writer is not None
                for row in reader:
                    writer.writerow(row)


def _resolve_esoa_source(inputs_dir: Path) -> Path:
    part_files = sorted(inputs_dir.glob("esoa_pt_*.csv"), key=_natural_esoa_part_order)
    if part_files:
        combined = inputs_dir / "esoa_combined.csv"
        _concatenate_csv(part_files, combined)
        return combined
    for candidate in ("esoa_combined.csv", "esoa.csv", "esoa_prepared.csv"):
        path = inputs_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Unable to resolve an eSOA input. Provide esoa_combined.csv or esoa_pt_*.csv under inputs/drugs."
    )


def _ensure_path(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Expected {description} at {path} but it was not found.")
    return path


def _refresh_pnf_prepared(inputs_dir: Path) -> Path:
    pnf_raw = _ensure_path(inputs_dir / "pnf.csv", "PNF source CSV")
    esoa_source = _resolve_esoa_source(inputs_dir)
    prepare(str(pnf_raw), str(esoa_source), str(inputs_dir))
    return _ensure_path(inputs_dir / "pnf_prepared.csv", "pnf_prepared.csv output")


def _refresh_drugbank_exports(root: Path, inputs_dir: Path) -> Path:
    cmd = [sys.executable, "-m", "pipelines.drugs.scripts.run_drugbank_drugs"]
    _run_subprocess(cmd, root)
    generics = _ensure_path(inputs_dir / "drugbank_generics.csv", "drugbank_generics.csv")
    return generics


def _find_latest_timestamped_file(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' under {directory}")

    def _key(path: Path) -> tuple[datetime, str]:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
        if match:
            try:
                stamp = datetime.strptime(match.group(1), "%Y-%m-%d")
            except ValueError:
                stamp = datetime.min
        else:
            stamp = datetime.min
        return stamp, path.name

    candidates.sort(key=_key)
    return candidates[-1]


def _refresh_who_atc(root: Path, inputs_dir: Path) -> Path:
    DrugsAndMedicinePipeline._run_r_scripts(root, inputs_dir)
    return _find_latest_timestamped_file(inputs_dir, "who_atc_*_molecules.csv")


def _refresh_fda_brand_map(inputs_dir: Path) -> Path:
    DrugsAndMedicinePipeline._build_brand_map(inputs_dir)
    return _find_latest_timestamped_file(inputs_dir, "fda_brand_map_*.csv")


def _refresh_source_datasets(root: Path) -> Dict[str, Path]:
    inputs_dir = root / "inputs" / "drugs"
    datasets: Dict[str, Path] = {}

    print("  - Refreshing PNF prepared CSV via pipelines.drugs.scripts.prepare_drugs...")
    datasets["pnf"] = _refresh_pnf_prepared(inputs_dir)

    print("  - Refreshing WHO ATC extracts via ATCD R scripts...")
    datasets["who"] = _refresh_who_atc(root, inputs_dir)

    print("  - Building latest FDA brand map export...")
    datasets["fda_brand"] = _refresh_fda_brand_map(inputs_dir)

    print("  - Refreshing DrugBank generics export via R helper...")
    datasets["drugbank_generics"] = _refresh_drugbank_exports(root, inputs_dir)

    return datasets


def _as_str(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (float, int)) and pd.isna(value):
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def normalize_text(value: str | float | None) -> str:
    text = _as_str(value).strip()
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,;:-_")
    return text


_TRAILING_DESCRIPTOR_WORDS = {
    "tablet",
    "tablets",
    "tab",
    "tabs",
    "capsule",
    "capsules",
    "cap",
    "caps",
    "solution",
    "sol",
    "suspension",
    "susp",
    "injection",
    "inj",
    "vial",
    "ampule",
    "ampoule",
    "cream",
    "ointment",
    "syrup",
    "elixir",
    "drops",
    "drop",
    "spray",
    "patch",
    "powder",
    "granule",
    "gel",
    "lotion",
    "foam",
    "lozenge",
    "suppository",
}

_TRAILING_PHRASES = [
    ("for", "injection"),
    ("for", "oral", "solution"),
    ("for", "oral"),
    ("for", "intravenous"),
    ("for", "intramuscular"),
    ("for", "topical"),
    ("intravenous",),
    ("intramuscular",),
    ("oral",),
    ("topical",),
    ("ophthalmic",),
    ("otic",),
]

_DESCRIPTOR_VALUES = {
    "fat-soluble",
    "water-soluble",
    "trace element",
    "trace elements",
    "lipid complex",
}


def _strip_descriptor_parentheses(text: str) -> str:
    while True:
        match = re.search(r"\(([^()]+)\)\s*$", text)
        if not match:
            break
        inner = normalize_text(match.group(1))
        if inner and inner not in _DESCRIPTOR_VALUES:
            break
        text = text[: match.start()].strip(" ,;-")
    return text


def canonical_molecule_name(raw: str | float | None) -> str:
    text = _as_str(raw)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip(" ,;-")
    text = _strip_descriptor_parentheses(text)
    if not text:
        return ""
    words = [w.strip(" ,;-") for w in text.split(" ") if w.strip(" ,;-")]
    if not words:
        return ""
    while words:
        last = words[-1]
        lowered = last.lower()
        if any(char.isdigit() for char in last) or any(sym in last for sym in ["%", "/"]):
            words.pop()
            continue
        if lowered in _TRAILING_DESCRIPTOR_WORDS:
            words.pop()
            continue
        removed_phrase = False
        for phrase in _TRAILING_PHRASES:
            if len(words) >= len(phrase):
                tail = [w.lower() for w in words[-len(phrase):]]
                if tail == list(phrase):
                    del words[-len(phrase):]
                    removed_phrase = True
                    break
        if removed_phrase:
            continue
        break
    if not words:
        return ""
    cleaned = " ".join(words).strip(" ,;-")
    cleaned = _strip_descriptor_parentheses(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;-")
    return cleaned.title()


def molecule_combo_to_components(raw_generic: str | float | None) -> List[str]:
    canonical = canonical_molecule_name(raw_generic)
    if not canonical:
        return []
    parts = re.split(r"\s*(?:\+|/|&| and )\s*", canonical, flags=re.IGNORECASE)
    components: List[str] = []
    for part in parts:
        candidate = part.strip(" ,;-")
        if not candidate:
            continue
        if normalize_text(candidate) in _DESCRIPTOR_VALUES:
            continue
        components.append(candidate)
    return components


def molecule_key(name: str | float | None) -> str:
    normalized = normalize_text(name)
    normalized = normalized.replace(" ", "").replace("-", "")
    return normalized


def molecule_set_key(raw_generic: str | float | None) -> str:
    components = molecule_combo_to_components(raw_generic)
    keys = sorted(filter(None, (molecule_key(component) for component in components)))
    return "||".join(keys)


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
    "foam": "foam",
    "drops": "drops",
    "drop": "drops",
    "aerosol": "aerosol",
    "lozenge": "lozenge",
    "suppository": "suppository",
}


def canonical_form(raw_form: str | float | None) -> str:
    text = normalize_text(raw_form)
    if not text:
        return ""
    return _FORM_MAP.get(text, text)


_ROUTE_MAP = {
    "po": "oral",
    "per os": "oral",
    "oral": "oral",
    "p o": "oral",
    "iv": "intravenous",
    "intravenous": "intravenous",
    "im": "intramuscular",
    "intramuscular": "intramuscular",
    "sc": "subcutaneous",
    "subcutaneous": "subcutaneous",
    "sub cutaneous": "subcutaneous",
    "sub-cutaneous": "subcutaneous",
    "sub q": "subcutaneous",
    "subq": "subcutaneous",
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


def canonical_route(raw_route: str | float | None) -> str:
    text = normalize_text(raw_route)
    if not text:
        return ""
    cleaned = text.replace(".", "")
    return _ROUTE_MAP.get(cleaned, cleaned)


_SALT_MAP = {
    "hcl": "hydrochloride",
    "hydrochlorid": "hydrochloride",
}


def canonical_salt_form(raw_salt: str | float | None) -> str:
    text = normalize_text(raw_salt)
    if not text:
        return ""
    return _SALT_MAP.get(text, text)


def split_multi(raw: str | float | None) -> List[str]:
    text = _as_str(raw)
    if not text:
        return []
    parts = re.split(r"[|;,]", text)
    values = [part.strip() for part in parts if part and part.strip()]
    return values


def get_column(
    df: pd.DataFrame,
    logical_name: str,
    candidates: Sequence[str],
    *,
    required: bool = False,
) -> pd.Series:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        column = lower_map.get(candidate.lower())
        if column:
            return df[column]
    if required:
        raise RuntimeError(
            f"Missing required column '{logical_name}'. Tried alternatives: {', '.join(candidates)}"
        )
    return pd.Series([""] * len(df), index=df.index, dtype=object)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def load_pnf(path: Path) -> pd.DataFrame:
    return load_csv(path)


def load_who_atc(path: Path) -> pd.DataFrame:
    return load_csv(path)


def load_db_generics(path: Path) -> pd.DataFrame:
    return load_csv(path)


def load_fda_brands(path: Path) -> pd.DataFrame:
    return load_csv(path)


def build_dim_atc(df_who: pd.DataFrame) -> pd.DataFrame:
    dim = pd.DataFrame(
        {
            "atc_code": get_column(df_who, "atc_code", ["atc_code", "atc code"], required=True),
            "atc_name": get_column(df_who, "atc_name", ["atc_name", "atc name"]),
            "ddd": get_column(df_who, "ddd", ["ddd"]),
            "uom": get_column(df_who, "uom", ["uom"]),
            "adm_r": get_column(df_who, "adm_r", ["adm_r", "adm r", "adm-route"]),
            "note": get_column(df_who, "note", ["note", "notes"]),
        }
    )
    dim = dim[dim["atc_code"].astype(str).str.strip() != ""].copy()
    dim = dim.drop_duplicates(subset=["atc_code"], keep="first")
    return dim.reset_index(drop=True)


def _collect_molecule_sources(
    df_pnf: pd.DataFrame,
    df_db_generics: pd.DataFrame,
    df_fda_brands: pd.DataFrame,
    df_who: pd.DataFrame,
) -> pd.DataFrame:
    records: List[Dict[str, str]] = []

    generic_id_series = get_column(df_pnf, "generic_id", ["generic_id"], required=True)
    generic_name_series = get_column(
        df_pnf, "generic_name", ["generic_name", "generic"], required=True
    )
    for idx, raw in generic_name_series.items():
        value = _as_str(raw).strip()
        if not value:
            continue
        source_key = f"PNF:{_as_str(generic_id_series.loc[idx]).strip()}"
        records.append({"source_system": "PNF", "source_key": source_key, "raw_generic": value})

    db_generic_series = get_column(
        df_db_generics, "generic", ["generic", "generic_name"], required=True
    )
    for raw in db_generic_series:
        value = _as_str(raw).strip()
        if not value:
            continue
        records.append({"source_system": "DB_GENERIC", "source_key": f"DBG:{value}", "raw_generic": value})

    fda_generic_series = get_column(
        df_fda_brands, "generic_name", ["generic_name", "generic"], required=False
    )
    registration_series = get_column(
        df_fda_brands, "registration_number", ["registration_number", "registration"], required=False
    )
    for idx, raw in fda_generic_series.items():
        value = _as_str(raw).strip()
        if not value:
            continue
        reg = _as_str(registration_series.loc[idx]).strip()
        source_key = f"FDA_BRAND:{reg or idx}"
        records.append(
            {"source_system": "FDA_BRAND_MAP", "source_key": source_key, "raw_generic": value}
        )

    who_name_series = get_column(df_who, "atc_name", ["atc_name", "atc name"], required=False)
    who_code_series = get_column(df_who, "atc_code", ["atc_code", "atc code"], required=False)
    for idx, raw in who_name_series.items():
        value = _as_str(raw).strip()
        if not value:
            continue
        source_key = f"WHO:{_as_str(who_code_series.loc[idx]).strip() or idx}"
        records.append({"source_system": "WHO_ATC", "source_key": source_key, "raw_generic": value})

    return pd.DataFrame(records)


def build_molecule_dimensions(
    df_pnf: pd.DataFrame,
    df_db_generics: pd.DataFrame,
    df_fda_brands: pd.DataFrame,
    df_who: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    molecule_source_rows = _collect_molecule_sources(
        df_pnf, df_db_generics, df_fda_brands, df_who
    )

    molecule_map: Dict[str, str] = {}
    for raw in molecule_source_rows["raw_generic"]:
        components = molecule_combo_to_components(raw)
        for component in components:
            key = molecule_key(component)
            if not key:
                continue
            molecule_map.setdefault(key, component)

    molecule_items = sorted(molecule_map.items(), key=lambda item: item[0])
    dim_molecule = pd.DataFrame(
        {
            "molecule_id": np.arange(1, len(molecule_items) + 1),
            "molecule_key": [item[0] for item in molecule_items],
            "molecule_name": [item[1] for item in molecule_items],
            "primary_atc_code": ["" for _ in molecule_items],
        }
    )

    molecule_set_map: Dict[str, set[str]] = {}
    for raw in molecule_source_rows["raw_generic"]:
        components = molecule_combo_to_components(raw)
        key = molecule_set_key(raw)
        if not key:
            continue
        entry = molecule_set_map.setdefault(key, set())
        entry.update(components)

    set_items = sorted(
        ((key, sorted(values)) for key, values in molecule_set_map.items()), key=lambda item: item[0]
    )
    dim_molecule_set = pd.DataFrame(
        {
            "molecule_set_id": np.arange(1, len(set_items) + 1),
            "molecule_set_key": [item[0] for item in set_items],
            "canonical_generic_name": ["; ".join(item[1]) for item in set_items],
            "primary_atc_code": ["" for _ in set_items],
            "notes": ["" for _ in set_items],
        }
    )

    molecule_lookup = dict(zip(dim_molecule["molecule_key"], dim_molecule["molecule_id"]))
    bridge_rows: List[Dict[str, Any]] = []
    for _, row in dim_molecule_set.iterrows():
        set_key = row["molecule_set_key"]
        set_id = row["molecule_set_id"]
        component_keys = [key for key in set_key.split("||") if key]
        for comp_key in component_keys:
            molecule_id = molecule_lookup.get(comp_key)
            if molecule_id is None:
                continue
            bridge_rows.append(
                {
                    "molecule_set_id": set_id,
                    "molecule_id": molecule_id,
                    "role": "active",
                    "ratio_numerator": np.nan,
                    "ratio_denominator": np.nan,
                }
            )
    bridge_columns = [
        "molecule_set_id",
        "molecule_id",
        "role",
        "ratio_numerator",
        "ratio_denominator",
    ]
    bridge_df = pd.DataFrame(bridge_rows, columns=bridge_columns)

    molecule_set_lookup = dict(zip(dim_molecule_set["molecule_set_key"], dim_molecule_set["molecule_set_id"]))
    return dim_molecule, dim_molecule_set, bridge_df, molecule_set_lookup


def build_dim_pnf_generic(
    df_pnf: pd.DataFrame, molecule_set_lookup: Dict[str, int]
) -> pd.DataFrame:
    generic_id = get_column(df_pnf, "generic_id", ["generic_id"], required=True)
    generic_name = get_column(df_pnf, "generic_name", ["generic_name", "generic"], required=True)
    salt_form = get_column(df_pnf, "salt_form", ["salt_form", "salt"], required=False)
    route_allowed = get_column(df_pnf, "route_allowed", ["route_allowed", "route"], required=False)
    form_token = get_column(df_pnf, "form_token", ["form_token", "form"], required=False)
    dose_kind = get_column(df_pnf, "dose_kind", ["dose_kind"], required=False)
    strength = get_column(df_pnf, "strength", ["strength"], required=False)
    unit = get_column(df_pnf, "unit", ["unit"], required=False)
    per_val = get_column(df_pnf, "per_val", ["per_val"], required=False)
    per_unit = get_column(df_pnf, "per_unit", ["per_unit"], required=False)
    pct = get_column(df_pnf, "pct", ["pct"], required=False)
    strength_mg = get_column(df_pnf, "strength_mg", ["strength_mg"], required=False)
    ratio_mg_per_ml = get_column(df_pnf, "ratio_mg_per_ml", ["ratio_mg_per_ml"], required=False)
    atc_code = get_column(df_pnf, "atc_code", ["atc_code", "atc code"], required=False)

    keys = generic_name.apply(molecule_set_key)
    molecule_set_ids = pd.Series(
        pd.array([molecule_set_lookup.get(key) for key in keys], dtype="Int64"),
        index=df_pnf.index,
    )

    dim_pnf = pd.DataFrame(
        {
            "generic_id": generic_id,
            "generic_name": generic_name,
            "salt_form": salt_form,
            "synonyms": generic_name,
            "atc_code": atc_code,
            "route_allowed": route_allowed,
            "form_token": form_token,
            "dose_kind": dose_kind,
            "strength": strength,
            "unit": unit,
            "per_val": per_val,
            "per_unit": per_unit,
            "pct": pct,
            "strength_mg": strength_mg,
            "ratio_mg_per_ml": ratio_mg_per_ml,
            "molecule_set_id": molecule_set_ids,
        }
    )
    dim_pnf = dim_pnf.drop_duplicates(subset=["generic_id"], keep="first")
    return dim_pnf.reset_index(drop=True)


def build_dim_drugbank_generic(
    df_db_generics: pd.DataFrame, molecule_set_lookup: Dict[str, int]
) -> pd.DataFrame:
    generic = get_column(df_db_generics, "generic", ["generic", "generic_name"], required=True)
    dose = get_column(df_db_generics, "dose", ["dose", "doses"], required=False)
    form = get_column(df_db_generics, "form", ["form", "forms"], required=False)
    route = get_column(df_db_generics, "route", ["route", "routes"], required=False)
    atc = get_column(df_db_generics, "atc_code", ["atc_code", "atc"], required=False)
    drugbank_ids = get_column(
        df_db_generics, "drugbank_ids", ["drugbank_ids", "drugbank_id"], required=False
    )

    keys = generic.apply(molecule_set_key)
    molecule_set_ids = pd.Series(
        pd.array([molecule_set_lookup.get(key) for key in keys], dtype="Int64"),
        index=df_db_generics.index,
    )

    atc_primary = []
    for value in atc:
        tokens = split_multi(value)
        atc_primary.append(tokens[0] if tokens else _as_str(value))

    dim_db_generic = pd.DataFrame(
        {
            "drugbank_generic_id": np.arange(1, len(df_db_generics) + 1),
            "generic": generic,
            "dose_raw": dose,
            "form_raw": form,
            "route_raw": route,
            "atc_code": atc_primary,
            "drugbank_ids_raw": drugbank_ids,
            "molecule_set_id": molecule_set_ids,
        }
    )
    return dim_db_generic




def build_dim_route(
    dim_pnf_generic: pd.DataFrame,
    dim_drugbank_generic: pd.DataFrame,
    df_fda_brands: pd.DataFrame,
) -> pd.DataFrame:
    routes: set[str] = set()

    if "route_allowed" in dim_pnf_generic.columns:
        for value in dim_pnf_generic["route_allowed"]:
            for token in split_multi(value):
                canonical = canonical_route(token)
                if canonical:
                    routes.add(canonical)

    if "route_raw" in dim_drugbank_generic.columns:
        for value in dim_drugbank_generic["route_raw"]:
            for token in split_multi(value):
                canonical = canonical_route(token)
                if canonical:
                    routes.add(canonical)

    route_series = get_column(df_fda_brands, "route", ["route"], required=False)
    for value in route_series:
        canonical = canonical_route(value)
        if canonical:
            routes.add(canonical)

    route_list = sorted(routes)
    dim_route = pd.DataFrame(
        {
            "route_id": np.arange(1, len(route_list) + 1),
            "route_name": route_list,
            "route_key": route_list,
        }
    )
    return dim_route


def build_dim_form(
    dim_pnf_generic: pd.DataFrame,
    dim_drugbank_generic: pd.DataFrame,
    df_fda_brands: pd.DataFrame,
) -> pd.DataFrame:
    forms: set[str] = set()

    if "form_token" in dim_pnf_generic.columns:
        for value in dim_pnf_generic["form_token"]:
            for token in split_multi(value):
                canonical = canonical_form(token)
                if canonical:
                    forms.add(canonical)

    if "form_raw" in dim_drugbank_generic.columns:
        for value in dim_drugbank_generic["form_raw"]:
            for token in split_multi(value):
                canonical = canonical_form(token)
                if canonical:
                    forms.add(canonical)

    dosage_forms = get_column(df_fda_brands, "dosage_form", ["dosage_form", "form"], required=False)
    for value in dosage_forms:
        canonical = canonical_form(value)
        if canonical:
            forms.add(canonical)

    form_list = sorted(forms)
    dim_form = pd.DataFrame(
        {
            "form_id": np.arange(1, len(form_list) + 1),
            "form_name": form_list,
            "form_key": form_list,
        }
    )
    return dim_form


def build_dim_salt_form(dim_pnf_generic: pd.DataFrame) -> pd.DataFrame:
    salts: set[str] = set()
    if "salt_form" in dim_pnf_generic.columns:
        for value in dim_pnf_generic["salt_form"]:
            canonical = canonical_salt_form(value)
            if canonical:
                salts.add(canonical)
    salt_list = sorted(salts)
    dim_salt_form = pd.DataFrame(
        {
            "salt_form_id": np.arange(1, len(salt_list) + 1),
            "salt_form_name": salt_list,
            "salt_form_group": salt_list,
        }
    )
    return dim_salt_form


def _expand_or_default(values: List[str]) -> List[str]:
    return values if values else [""]


def build_staging_variants(
    dim_pnf_generic: pd.DataFrame,
    dim_drugbank_generic: pd.DataFrame,
    df_fda_brands: pd.DataFrame,
    molecule_set_lookup: Dict[str, int],
) -> pd.DataFrame:
    staging_rows: List[Dict[str, Any]] = []

    for row in dim_pnf_generic.itertuples(index=False):
        routes = _expand_or_default(split_multi(getattr(row, "route_allowed", "")))
        forms = _expand_or_default(split_multi(getattr(row, "form_token", "")))
        salt = canonical_salt_form(getattr(row, "salt_form", ""))
        for route_value in routes:
            for form_value in forms:
                staging_rows.append(
                    {
                        "source_system": "PNF",
                        "source_row_id": f"PNF::{row.generic_id}::{route_value}::{form_value}",
                        "molecule_set_id": getattr(row, "molecule_set_id"),
                        "salt_form_name": salt,
                        "form_name": canonical_form(form_value),
                        "route_name": canonical_route(route_value),
                        "dose_kind": _as_str(getattr(row, "dose_kind", "")),
                        "strength": _as_str(getattr(row, "strength", "")),
                        "unit": _as_str(getattr(row, "unit", "")),
                        "per_val": _as_str(getattr(row, "per_val", "")),
                        "per_unit": _as_str(getattr(row, "per_unit", "")),
                        "pct": _as_str(getattr(row, "pct", "")),
                        "strength_mg": _as_str(getattr(row, "strength_mg", "")),
                        "ratio_mg_per_ml": _as_str(getattr(row, "ratio_mg_per_ml", "")),
                        "pnf_generic_id": _as_str(getattr(row, "generic_id", "")),
                        "drugbank_generic_id": "",
                        "fda_registration_number": "",
                        "atc_code": _as_str(getattr(row, "atc_code", "")),
                    }
                )

    for row in dim_drugbank_generic.itertuples(index=False):
        doses = _expand_or_default(split_multi(getattr(row, "dose_raw", "")))
        forms = _expand_or_default(split_multi(getattr(row, "form_raw", "")))
        routes = _expand_or_default(split_multi(getattr(row, "route_raw", "")))
        for dose_value in doses:
            for form_value in forms:
                for route_value in routes:
                    staging_rows.append(
                        {
                            "source_system": "DB",
                            "source_row_id": f"DBG::{row.drugbank_generic_id}::{dose_value}::{form_value}::{route_value}",
                            "molecule_set_id": getattr(row, "molecule_set_id"),
                            "salt_form_name": "",
                            "form_name": canonical_form(form_value),
                            "route_name": canonical_route(route_value),
                            "dose_kind": "unknown",
                            "strength": _as_str(dose_value),
                            "unit": "",
                            "per_val": "",
                            "per_unit": "",
                            "pct": "",
                            "strength_mg": "",
                            "ratio_mg_per_ml": "",
                            "pnf_generic_id": "",
                            "drugbank_generic_id": str(row.drugbank_generic_id),
                            "fda_registration_number": "",
                            "atc_code": _as_str(getattr(row, "atc_code", "")),
                        }
                    )

    fda_brand_generic = get_column(
        df_fda_brands, "generic_name", ["generic_name", "generic"], required=False
    )
    registration = get_column(
        df_fda_brands, "registration_number", ["registration_number", "registration"], required=False
    )
    dosage_form = get_column(df_fda_brands, "dosage_form", ["dosage_form", "form"], required=False)
    route = get_column(df_fda_brands, "route", ["route"], required=False)
    dosage_strength = get_column(
        df_fda_brands, "dosage_strength", ["dosage_strength", "strength"], required=False
    )

    for idx in df_fda_brands.index:
        generic_name = fda_brand_generic.loc[idx]
        set_id = molecule_set_lookup.get(molecule_set_key(generic_name))
        registration_number = _as_str(registration.loc[idx]).strip()
        staging_rows.append(
            {
                "source_system": "FDA_BRAND_MAP",
                "source_row_id": f"FDA::{registration_number}:::{_as_str(dosage_form.loc[idx]).strip()}::"
                f"{_as_str(route.loc[idx]).strip()}::{_as_str(dosage_strength.loc[idx]).strip()}",
                "molecule_set_id": set_id,
                "salt_form_name": "",
                "form_name": canonical_form(dosage_form.loc[idx]),
                "route_name": canonical_route(route.loc[idx]),
                "dose_kind": "unknown",
                "strength": _as_str(dosage_strength.loc[idx]).strip(),
                "unit": "",
                "per_val": "",
                "per_unit": "",
                "pct": "",
                "strength_mg": "",
                "ratio_mg_per_ml": "",
                "pnf_generic_id": "",
                "drugbank_generic_id": "",
                "fda_registration_number": registration_number,
                "atc_code": "",
            }
        )

    staging = pd.DataFrame(staging_rows)
    if staging.empty:
        return staging

    def _all_central_empty(row: pd.Series) -> bool:
        return all(_as_str(row.get(col, "")).strip() == "" for col in ["form_name", "route_name", "strength"])

    mask_drop = staging["molecule_set_id"].isna() & staging.apply(_all_central_empty, axis=1)
    staging = staging[~mask_drop].reset_index(drop=True)
    return staging


def _map_dimension_id(series: pd.Series, dim: pd.DataFrame, name_col: str, id_col: str) -> pd.Series:
    if dim.empty:
        return pd.Series(pd.array([None] * len(series), dtype="Int64"), index=series.index)
    lookup = {row[name_col]: row[id_col] for _, row in dim.iterrows()}
    return pd.Series(pd.array([lookup.get(value) for value in series], dtype="Int64"), index=series.index)


def _aggregate_unique(values: Iterable[Any], sep: str = "|") -> str:
    collected = []
    seen = set()
    for value in values:
        text = _as_str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        collected.append(text)
    return sep.join(sorted(collected))


def build_master_variant(
    staging: pd.DataFrame,
    dim_form: pd.DataFrame,
    dim_route: pd.DataFrame,
    dim_salt_form: pd.DataFrame,
) -> pd.DataFrame:
    master_columns = [
        "variant_id",
        "molecule_set_id",
        "salt_form_id",
        "form_id",
        "route_id",
        "dose_kind",
        "strength",
        "unit",
        "per_val",
        "per_unit",
        "pct",
        "strength_mg",
        "ratio_mg_per_ml",
        "pnf_generic_ids",
        "drugbank_generic_ids",
        "fda_registration_numbers",
        "atc_codes",
        "source_flags",
    ]
    if staging.empty:
        return pd.DataFrame(columns=master_columns)

    staging = staging.copy()
    staging["form_id"] = _map_dimension_id(staging["form_name"], dim_form, "form_name", "form_id")
    staging["route_id"] = _map_dimension_id(staging["route_name"], dim_route, "route_name", "route_id")
    staging["salt_form_id"] = _map_dimension_id(
        staging["salt_form_name"], dim_salt_form, "salt_form_name", "salt_form_id"
    )

    group_cols = [
        "molecule_set_id",
        "salt_form_id",
        "form_id",
        "route_id",
        "dose_kind",
        "strength",
        "unit",
        "per_val",
        "per_unit",
        "pct",
        "strength_mg",
        "ratio_mg_per_ml",
    ]

    grouped = (
        staging.groupby(group_cols, dropna=False)
        .agg(
            {
                "pnf_generic_id": lambda s: _aggregate_unique(s),
                "drugbank_generic_id": lambda s: _aggregate_unique(s),
                "fda_registration_number": lambda s: _aggregate_unique(s),
                "atc_code": lambda s: _aggregate_unique(s),
                "source_system": lambda s: _aggregate_unique(s, sep="+"),
            }
        )
        .reset_index()
    )

    master = grouped.rename(
        columns={
            "pnf_generic_id": "pnf_generic_ids",
            "drugbank_generic_id": "drugbank_generic_ids",
            "fda_registration_number": "fda_registration_numbers",
            "atc_code": "atc_codes",
            "source_system": "source_flags",
        }
    )
    master.insert(0, "variant_id", np.arange(1, len(master) + 1))
    master = master[master_columns]
    return master


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs" / "drugs" / "master"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP 0] Refreshing source datasets...")
    dataset_paths = _refresh_source_datasets(root)
    pnf_path = dataset_paths["pnf"]
    who_path = dataset_paths["who"]
    fda_brands_path = dataset_paths["fda_brand"]
    db_generics_path = dataset_paths["drugbank_generics"]

    print("[STEP 1] Loading input CSVs...")

    df_pnf = load_pnf(pnf_path)
    df_who = load_who_atc(who_path)
    df_db_generics = load_db_generics(db_generics_path)
    df_fda_brands = load_fda_brands(fda_brands_path)

    print(f"Loaded PNF rows: {len(df_pnf)}")
    print(f"Loaded WHO ATC rows: {len(df_who)}")
    print(f"Loaded DrugBank generics rows: {len(df_db_generics)}")
    print(f"Loaded FDA brand map rows: {len(df_fda_brands)}")

    print("[STEP 2] Building dim_atc...")
    dim_atc = build_dim_atc(df_who)
    write_csv(dim_atc, output_dir / "dim_atc.csv")

    print("[STEP 3] Building dim_molecule and dim_molecule_set...")
    dim_molecule, dim_molecule_set, bridge_molecule_set_member, molecule_set_lookup = build_molecule_dimensions(
        df_pnf, df_db_generics, df_fda_brands, df_who
    )
    write_csv(dim_molecule, output_dir / "dim_molecule.csv")
    write_csv(dim_molecule_set, output_dir / "dim_molecule_set.csv")
    write_csv(bridge_molecule_set_member, output_dir / "bridge_molecule_set_member.csv")

    print("[STEP 4] Building dim_pnf_generic...")
    dim_pnf_generic = build_dim_pnf_generic(df_pnf, molecule_set_lookup)
    write_csv(dim_pnf_generic, output_dir / "dim_pnf_generic.csv")

    print("[STEP 5] Building dim_drugbank_generic...")
    dim_drugbank_generic = build_dim_drugbank_generic(df_db_generics, molecule_set_lookup)
    write_csv(dim_drugbank_generic, output_dir / "dim_drugbank_generic.csv")

    print("[STEP 6] Building dim_route, dim_form, and dim_salt_form...")
    dim_route = build_dim_route(dim_pnf_generic, dim_drugbank_generic, df_fda_brands)
    dim_form = build_dim_form(dim_pnf_generic, dim_drugbank_generic, df_fda_brands)
    dim_salt_form = build_dim_salt_form(dim_pnf_generic)
    write_csv(dim_route, output_dir / "dim_route.csv")
    write_csv(dim_form, output_dir / "dim_form.csv")
    write_csv(dim_salt_form, output_dir / "dim_salt_form.csv")

    print("[STEP 7] Building staging variants and master_variant...")
    staging_variants = build_staging_variants(dim_pnf_generic, dim_drugbank_generic, df_fda_brands, molecule_set_lookup)
    master_variant = build_master_variant(staging_variants, dim_form, dim_route, dim_salt_form)
    write_csv(master_variant, output_dir / "master_variant.csv")

    print("[DONE] Master dataset created in ./outputs/drugs/master/")


if __name__ == "__main__":
    main()
