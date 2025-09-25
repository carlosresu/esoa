#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, pandas as pd

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
    "atc_code_final","confidence","bucket_final","why_final","reason_final",
]

def write_outputs(out_df: pd.DataFrame, out_csv: str) -> str:
    out_small = out_df.copy()
    if "match_basis" not in out_small.columns:
        out_small["match_basis"] = out_small.get("normalized", "")
    out_small = out_small[[c for c in OUTPUT_COLUMNS if c in out_small.columns]].copy()

    out_small.to_csv(out_csv, index=False, encoding="utf-8")
    total = len(out_small)
    print(f"[match] wrote {out_csv} with {total:,} rows")

    # Excel
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

    # Unknown words export
    unknown = out_small.loc[out_small["unknown_kind"].ne("None") & out_small["unknown_words"].astype(str).ne(""), ["unknown_words"]].copy()
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
        print(f"[match] wrote {unk_path} with {len(unk_df):,} unique unknown words")
    else:
        print("[match] no unknown words to export")

    # Summary
    grouped = (out_small.groupby(["bucket_final","why_final","reason_final"], dropna=False).size().reset_index(name="n"))
    grouped["pct"] = (grouped["n"] / float(total) * 100).round(2)
    print("\n[summary] Distribution")

    # Auto-Accept total
    aa = int(grouped.loc[grouped["bucket_final"].eq("Auto-Accept"), "n"].sum())
    aa_pct = round(aa / float(total) * 100, 2) if total else 0.0
    print(f"  - Auto-Accept: {aa:,} ({aa_pct}%)")

    # Auto-Accept sub-breakdown (merge the nice idea from summarize_results.py)
    if aa:
        aa_rows = out_small.loc[out_small["bucket_final"].eq("Auto-Accept")].copy()
        # Two intuitive subtypes: brand→generic swap vs OK, no changes
        swaps = int(aa_rows.get("did_brand_swap", False).sum()) if "did_brand_swap" in aa_rows.columns else 0
        ok_no_change = max(0, aa - swaps)
        if swaps:
            pct = round(swaps / float(total) * 100, 2)
            print(f"      • brand→generic swap: {swaps:,} ({pct}%)")
        if ok_no_change:
            pct = round(ok_no_change / float(total) * 100, 2)
            print(f"      • OK, no changes: {ok_no_change:,} ({pct}%)")

    # Needs review (aka Candidate)
    nr = grouped[grouped["bucket_final"].eq("Candidate")].copy()
    if not nr.empty:
        nr_total = int(nr["n"].sum())
        nr_total_pct = round(nr_total / float(total) * 100, 2) if total else 0.0
        print(f"  - Needs review: {nr_total:,} ({nr_total_pct}%)")
        # Order by why then reason
        nr = nr.sort_values(by=["why_final","reason_final","n"], ascending=[True, True, False])
        for _, row in nr.iterrows():
            print(f"      • {row['why_final']}: {row['reason_final']}: {row['n']:,} ({row['pct']}%)")
    else:
        print(f"  - Needs review: 0 (0.00%)")

    # Others
    oth = grouped[grouped["bucket_final"].eq("Others")].copy()
    if not oth.empty:
        total_oth = int(oth["n"].sum()); total_oth_pct = round(total_oth / float(total) * 100, 2) if total else 0.0
        print(f"  - Others: {total_oth:,} ({total_oth_pct}%)")
        oth = oth.sort_values(by=["why_final","reason_final","n"], ascending=[True, True, False])
        for _, row in oth.iterrows():
            print(f"      • {row['why_final']}: {row['reason_final']}: {row['n']:,} ({row['pct']}%)")
    else:
        print(f"  - Others: 0 (0.00%)")

    return out_csv
