#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, time
import os, pandas as pd
from typing import Callable, List

# Local lightweight spinner so this module is self-contained
def _run_with_spinner(label: str, func: Callable[[], None]) -> float:
    """Provide a local progress spinner so downstream modules stay dependency-free."""
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
    "esoa_idx","raw_text","parentheticals",
    "normalized","norm_compact","match_basis","match_basis_norm_basic",
    "probable_brands","did_brand_swap","fda_dose_corroborated",
    "molecules_recognized","molecules_recognized_list","molecules_recognized_count",
    "dose_recognized","dosage_parsed_raw","dosage_parsed",
    "route_raw","form_raw","route_evidence_raw",
    "route","route_source","route_text","form","form_source","form_text",
    "form_ok","route_ok","route_form_imputations","route_evidence",
    "present_in_pnf","present_in_who","present_in_fda_generic","probable_atc",
    "generic_id","molecule_token","pnf_hits_gids","pnf_hits_count","pnf_hits_tokens",
    "who_molecules_list","who_molecules","who_atc_codes_list","who_atc_codes","who_atc_count","who_atc_has_ddd","who_atc_adm_r",
    "selected_form","selected_route_allowed","selected_variant",
    "selected_dose_kind","selected_strength","selected_unit","selected_strength_mg",
    "selected_per_val","selected_per_unit","selected_ratio_mg_per_ml","selected_pct",
    "dose_sim","looks_combo_final","combo_reason","combo_known_generics_count",
    "unknown_kind","unknown_words_list","unknown_words",
    "atc_code_final","confidence",
    "match_molecule(s)","match_quality",
    "bucket_final","why_final","reason_final",
]

def _generate_summary_lines(out_small: pd.DataFrame, mode: str) -> List[str]:
    """Produce human-readable distribution summaries for review files."""
    total = len(out_small)
    lines: List[str] = ["Distribution Summary"]

    # Auto-Accept
    aa_rows = out_small.loc[out_small["bucket_final"].eq("Auto-Accept")].copy()
    aa = int(len(aa_rows)) if total else 0
    aa_pct = round(aa / float(total) * 100, 2) if total else 0.0
    lines.append(f"Auto-Accept: {aa:,} ({aa_pct}%)")
    if aa:
        exact_mask = (
            (aa_rows["dose_sim"].astype(float) == 1.0)
            & aa_rows["route"].astype(str).ne("")
            & aa_rows["form"].astype(str).ne("")
        )
        swapped_exact = aa_rows.loc[exact_mask & aa_rows["did_brand_swap"].astype(bool)]
        clean_exact = aa_rows.loc[exact_mask & (~aa_rows["did_brand_swap"].astype(bool))]

        aa_breakdown = []
        if len(swapped_exact):
            count = len(swapped_exact)
            pct = round(count / float(total) * 100, 2)
            aa_breakdown.append(
                (
                    count,
                    f"  ValidBrandSwappedForGenericInPNF: exact dose/form/route match: {count:,} ({pct}%)",
                )
            )
        if len(clean_exact):
            count = len(clean_exact)
            pct = round(count / float(total) * 100, 2)
            aa_breakdown.append(
                (
                    count,
                    f"  ValidGenericInPNF, exact dose/route/form match: {count:,} ({pct}%)",
                )
            )

        for _, line in sorted(aa_breakdown, key=lambda x: x[0], reverse=True):
            lines.append(line)

    # Needs review
    nr_rows = out_small.loc[out_small["bucket_final"].eq("Needs review")].copy()
    nr = int(len(nr_rows))
    nr_pct = round(nr / float(total) * 100, 2) if total else 0.0
    lines.append(f"Needs review: {nr:,} ({nr_pct}%)")
    if nr:
        nr_rows["match_molecule(s)"] = nr_rows["match_molecule(s)"].replace({"": "UnspecifiedSource"})
        nr_rows["match_quality"] = nr_rows["match_quality"].replace({"": "unspecified"})

        if mode == "default":
            grp = (
                nr_rows.groupby(["match_molecule(s)", "match_quality"], dropna=False)
                .size()
                .reset_index(name="n")
            )
            grp["pct"] = (grp["n"] / float(total) * 100).round(2) if total else 0.0
            grp = grp.sort_values(by=["n", "match_molecule(s)", "match_quality"], ascending=[False, True, True])
            for _, row in grp.iterrows():
                lines.append(f"  {row['match_molecule(s)']}: {row['match_quality']}: {row['n']:,} ({row['pct']}%)")
        elif mode == "molecule":
            grp = (
                nr_rows.groupby(["match_molecule(s)"], dropna=False)
                .size()
                .reset_index(name="n")
            )
            grp["pct"] = (grp["n"] / float(total) * 100).round(2) if total else 0.0
            grp = grp.sort_values(by=["n", "match_molecule(s)"], ascending=[False, True])
            for _, row in grp.iterrows():
                lines.append(f"  {row['match_molecule(s)']}: {row['n']:,} ({row['pct']}%)")
        elif mode == "match":
            grp = (
                nr_rows.groupby(["match_quality"], dropna=False)
                .size()
                .reset_index(name="n")
            )
            grp["pct"] = (grp["n"] / float(total) * 100).round(2) if total else 0.0
            grp = grp.sort_values(by=["n", "match_quality"], ascending=[False, True])
            for _, row in grp.iterrows():
                lines.append(f"  {row['match_quality']}: {row['n']:,} ({row['pct']}%)")

    # Others
    oth_rows = out_small.loc[out_small["bucket_final"].eq("Others")].copy()
    oth = int(len(oth_rows))
    oth_pct = round(oth / float(total) * 100, 2) if total else 0.0
    lines.append(f"Others: {oth:,} ({oth_pct}%)")
    if oth:
        grouped_oth = (
            oth_rows.groupby(["why_final", "reason_final"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        grouped_oth["pct"] = (grouped_oth["n"] / float(total) * 100).round(2) if total else 0.0
        grouped_oth = grouped_oth.sort_values(by=["n", "why_final", "reason_final"], ascending=[False, True, True])
        for _, row in grouped_oth.iterrows():
            lines.append(f"  {row['why_final']}: {row['reason_final']}: {row['n']:,} ({row['pct']}%)")

    return lines


def _write_summary_text(out_small: pd.DataFrame, out_csv: str) -> None:
    """Write multiple summary text files that slice the results by molecule or match quality."""
    base_dir = os.path.dirname(out_csv)
    summaries = [
        ("summary.txt", "default"),
        ("summary_molecule.txt", "molecule"),
        ("summary_match.txt", "match"),
    ]

    for filename, mode in summaries:
        summary_path = os.path.join(base_dir, filename)
        lines = _generate_summary_lines(out_small, mode)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

def write_outputs(
    out_df: pd.DataFrame,
    out_csv: str,
    *,
    timing_hook: Callable[[str, float], None] | None = None,
) -> str:
    """Persist the canonical CSV/XLSX outputs plus text summaries, reporting timings."""
    def _timed(label: str, func: Callable[[], None]) -> float:
        elapsed = _run_with_spinner(label, func)
        if timing_hook:
            timing_hook(label, elapsed)
        return elapsed

    out_small = out_df.copy()
    if "match_basis" not in out_small.columns:
        out_small["match_basis"] = out_small.get("normalized", "")
    out_small = out_small[[c for c in OUTPUT_COLUMNS if c in out_small.columns]].copy()

    _timed("Write matched CSV", lambda: out_small.to_csv(out_csv, index=False, encoding="utf-8"))

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
    _timed("Write Excel", _to_excel)

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
    _timed("Write unknown words CSV", _write_unknowns)

    _timed("Write summary.txt", lambda: _write_summary_text(out_small, out_csv))

    return out_csv
