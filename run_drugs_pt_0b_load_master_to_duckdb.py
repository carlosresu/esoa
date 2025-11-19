from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import duckdb


ColumnDef = Tuple[str, str]


def create_or_replace_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    csv_path: Path,
    columns: Sequence[ColumnDef],
) -> None:
    column_sql = ", ".join(f"{name} {dtype}" for name, dtype in columns)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} ({column_sql})")
    query = (
        f"INSERT INTO {table_name} SELECT * FROM read_csv_auto('{csv_path}', header=True, nullstr=['', 'NA', 'Na', 'na'], sample_size=-1, ignore_errors=True)"
    )
    con.execute(query)


def main() -> None:
    root = Path(__file__).resolve().parent
    master_dir = root / "outputs" / "drugs" / "master"
    db_path = root / "outputs" / "drugs" / "drugs_master.duckdb"

    expected_files = {
        "bridge_molecule_set_member.csv": [
            ("molecule_set_id", "BIGINT"),
            ("molecule_id", "BIGINT"),
            ("role", "TEXT"),
            ("ratio_numerator", "DOUBLE"),
            ("ratio_denominator", "DOUBLE"),
        ],
        "dim_atc.csv": [
            ("atc_code", "TEXT"),
            ("atc_name", "TEXT"),
            ("ddd", "DOUBLE"),
            ("uom", "TEXT"),
            ("adm_r", "TEXT"),
            ("note", "TEXT"),
        ],
        "dim_drugbank_generic.csv": [
            ("drugbank_generic_id", "BIGINT"),
            ("generic", "TEXT"),
            ("dose_raw", "TEXT"),
            ("form_raw", "TEXT"),
            ("route_raw", "TEXT"),
            ("atc_code", "TEXT"),
            ("drugbank_ids_raw", "TEXT"),
            ("molecule_set_id", "BIGINT"),
        ],
        "dim_form.csv": [
            ("form_id", "BIGINT"),
            ("form_name", "TEXT"),
            ("form_key", "TEXT"),
        ],
        "dim_molecule.csv": [
            ("molecule_id", "BIGINT"),
            ("molecule_key", "TEXT"),
            ("molecule_name", "TEXT"),
            ("primary_atc_code", "TEXT"),
        ],
        "dim_molecule_set.csv": [
            ("molecule_set_id", "BIGINT"),
            ("molecule_set_key", "TEXT"),
            ("canonical_generic_name", "TEXT"),
            ("primary_atc_code", "TEXT"),
            ("notes", "TEXT"),
        ],
        "dim_pnf_generic.csv": [
            ("generic_id", "TEXT"),
            ("generic_name", "TEXT"),
            ("salt_form", "TEXT"),
            ("synonyms", "TEXT"),
            ("atc_code", "TEXT"),
            ("route_allowed", "TEXT"),
            ("form_token", "TEXT"),
            ("dose_kind", "TEXT"),
            ("strength", "TEXT"),
            ("unit", "TEXT"),
            ("per_val", "DOUBLE"),
            ("per_unit", "TEXT"),
            ("pct", "DOUBLE"),
            ("strength_mg", "DOUBLE"),
            ("ratio_mg_per_ml", "DOUBLE"),
            ("molecule_set_id", "BIGINT"),
        ],
        "dim_route.csv": [
            ("route_id", "BIGINT"),
            ("route_name", "TEXT"),
            ("route_key", "TEXT"),
        ],
        "dim_salt_form.csv": [
            ("salt_form_id", "BIGINT"),
            ("salt_form_name", "TEXT"),
            ("salt_form_group", "TEXT"),
        ],
        "master_variant.csv": [
            ("variant_id", "BIGINT"),
            ("molecule_set_id", "BIGINT"),
            ("salt_form_id", "BIGINT"),
            ("form_id", "BIGINT"),
            ("route_id", "BIGINT"),
            ("dose_kind", "TEXT"),
            ("strength", "TEXT"),
            ("unit", "TEXT"),
            ("per_val", "DOUBLE"),
            ("per_unit", "TEXT"),
            ("pct", "DOUBLE"),
            ("strength_mg", "DOUBLE"),
            ("ratio_mg_per_ml", "DOUBLE"),
            ("pnf_generic_ids", "TEXT"),
            ("drugbank_generic_ids", "TEXT"),
            ("fda_registration_numbers", "TEXT"),
            ("atc_codes", "TEXT"),
            ("source_flags", "TEXT"),
        ],
    }

    if not master_dir.exists():
        raise RuntimeError(f"Master directory missing: {master_dir}")

    missing = [name for name in expected_files if not (master_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing master CSVs: {missing}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to DuckDB at {db_path}...")
    con = duckdb.connect(str(db_path))

    for filename, columns in expected_files.items():
        table = Path(filename).stem
        csv_path = master_dir / filename
        print(f"Loading {filename} into table {table}...")
        create_or_replace_table(con, table, csv_path, columns)

    print(f"[OK] Loaded {len(expected_files)} tables into DuckDB at {db_path}.")


if __name__ == "__main__":
    main()
