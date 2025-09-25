#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, pandas as pd

# Include match_basis in outputs for transparency/debugging.
OUTPUT_COLUMNS = [
    "raw_text","normalized","match_basis",
    "molecules_recognized","molecules_recognized_count","probable_brands",
    "dose_recognized","dose_kind_detected","route","form",
    "present_in_pnf","present_in_who","probable_atc",
    "generic_id","molecule_token","pnf_hits_count","pnf_hits_tokens",
    "who_molecules","who_atc_codes","who_atc_count",
    "route_evidence","dosage_parsed","selected_form","selected_variant","dose_sim",
    "looks_combo_raw","looks_combo_final","combo_reason","combo_segments_total","combo_id_molecules","combo_unknown_in_pnf","combo_unknown_in_who","combo_unknown_in_both",
    "atc_code_final","confidence","bucket_final","why_final",
]

def write_outputs(out_df: pd.DataFrame, out_csv: str) -> str:
    out_small = out_df.copy()
    # Some upstream versions may not have match_basis if users skip brand map; ensure column exists
    if "match_basis" not in out_small.columns:
        out_small["match_basis"] = out_small.get("normalized", "")
    out_small = out_small[[c for c in OUTPUT_COLUMNS if c in out_small.columns]].copy()

    out_small.to_csv(out_csv, index=False, encoding="utf-8")
    total = len(out_small)
    print(f"[match] wrote {out_csv} with {total:,} rows")
    xlsx_out = os.path.splitext(out_csv)[0] + ".xlsx"
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
    print(f"[match] wrote {xlsx_out} with filters and frozen header")
    grouped = (out_small.groupby(["bucket_final","why_final"], dropna=False).size().reset_index(name="n"))
    grouped["pct"] = (grouped["n"] / float(total) * 100).round(2)
    print("\n[summary] Distribution")
    aa = grouped[grouped["bucket_final"].eq("Auto-Accept")]["n"].sum()
    aa_pct = round(aa / float(total) * 100, 2) if total else 0.0
    print(f"  - Auto-Accept: {aa:,} ({aa_pct}%)")
    nr = grouped[grouped["bucket_final"].eq("Needs review")].copy()
    if not nr.empty:
        nr_total = int(nr["n"].sum())
        nr_total_pct = round(nr_total / float(total) * 100, 2) if total else 0.0
        def fam_order(s):
            if str(s).startswith("ValidMoleculeWithATC/NotInPNF:"): return (0, str(s))
            if str(s).startswith("Candidate:"): return (1, str(s))
            if str(s) == "BrandOnly/NoGenericInPNF/NoGenericInWHO": return (2, str(s))
            return (3, str(s))
        nr = nr.sort_values(by=["why_final"], key=lambda c: c.map(fam_order))
        print(f"  - Needs review: {nr_total:,} ({nr_total_pct}%)")
        for _, row in nr.iterrows():
            print(f"      • {row['why_final']}: {row['n']:,} ({row['pct']}%)")
    oth = grouped[grouped["bucket_final"].eq("Others")].copy()
    if not oth.empty:
        total_oth = int(oth["n"].sum())
        total_oth_pct = round(total_oth / float(total) * 100, 2) if total else 0.0
        print(f"  - Others: {total_oth:,} ({total_oth_pct}%)")
        for _, row in oth.sort_values("n", ascending=False).iterrows():
            label = row["why_final"] if row["why_final"] else "Combinations"
            print(f"      • {label}: {row['n']:,} ({row['pct']}%)")
    return out_csv
