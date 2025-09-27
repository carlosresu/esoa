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
        sys.stdout.write(f"\r{frames[i % len(frames)]} {label} … {elapsed:0.1f}s")
        sys.stdout.flush()
        time.sleep(0.1); i += 1
    th.join()
    elapsed = time.perf_counter() - t0
    sys.stdout.write(f"\r✓ {label} — done in {elapsed:0.2f}s\n")
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
    "match_molecules","match_quality",
    "bucket_final","why_final","reason_final",
]

def _write_summary_text(out_small: pd.DataFrame, out_csv: str) -> None:
    total = len(out_small)
    lines = []
    lines.append("Distribution Summary")

    # Auto-Accept
    auto_rows = out_small.loc[out_small["bucket_final"].eq("Auto-Accept")].copy()
    aa = int(len(auto_rows)) if total else 0
    aa_pct = round(aa / float(total) * 100, 2) if total else 0.0
    lines.append(f"Auto-Accept: {aa:,} ({aa_pct}%)")
    if aa:
        swaps = int(auto_rows.get("did_brand_swap", False).sum()) if "did_brand_swap" in auto_rows.columns else 0
        ok_no_change = max(0, aa - swaps)
        if swaps:
            pct = round(swaps / float(total) * 100, 2)
            lines.append(f"  brand→generic swap: {swaps:,} ({pct}%)")
        if ok_no_change:
            pct = round(ok_no_change / float(total) * 100, 2)
            lines.append(f"  OK, no changes: {ok_no_change:,} ({pct}%)")

    # Needs review
    nr_rows = out_small.loc[out_small["bucket_final"].eq("Needs review")].copy()
    nr = int(len(nr_rows))
    nr_pct = round(nr / float(total) * 100, 2) if total else 0.0
    lines.append(f"Needs review: {nr:,} ({nr_pct}%)")
    if nr:
        grouped = (
            nr_rows.groupby(["why_final","reason_final"], dropna=False)
            .size().reset_index(name="n")
        )
        grouped["pct"] = (grouped["n"] / float(total) * 100).round(2) if total else 0.0
        grouped = grouped.sort_values(by=["why_final","reason_final","n"], ascending=[True, True, False])
        for _, row in grouped.iterrows():
            lines.append(f"  {row['why_final']}: {row['reason_final']}: {row['n']:,} ({row['pct']}%)")

    # Others (Unknown)
    oth_rows = out_small.loc[out_small["bucket_final"].eq("Others")].copy()
    oth = int(len(oth_rows))
    oth_pct = round(oth / float(total) * 100, 2) if total else 0.0
    lines.append(f"Others: {oth:,} ({oth_pct}%)")
    if oth:
        grouped_oth = (
            oth_rows.groupby(["why_final","reason_final"], dropna=False)
            .size().reset_index(name="n")
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
