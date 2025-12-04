"""
Pipeline runner functions for drug tagging.

These functions are called by the run_* scripts in the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .io_utils import reorder_columns_after, write_csv_and_parquet
from .spinner import run_with_spinner
from .tagger import UnifiedTagger
from .unified_constants import (
    GARBAGE_TOKENS,
    ALL_DRUG_SYNONYMS,
    DRUGBANK_COMPONENT_SYNONYMS,
    FORM_TO_ROUTES,
    FORM_EQUIVALENTS,
    get_valid_routes_for_form,
)


# Default paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_DIR / "raw" / "drugs"
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"

PIPELINE_RAW_DIR = Path(os.environ.get("PIPELINE_RAW_DIR", RAW_DIR))
PIPELINE_INPUTS_DIR = Path(os.environ.get("PIPELINE_INPUTS_DIR", INPUTS_DIR))
PIPELINE_OUTPUTS_DIR = Path(os.environ.get("PIPELINE_OUTPUTS_DIR", OUTPUTS_DIR))


def run_annex_f_tagging(
    annex_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Run Annex F tagging (Part 2).
    
    Returns dict with results summary.
    """
    if annex_path is None:
        annex_path = PIPELINE_RAW_DIR / "annex_f.csv"
    if output_path is None:
        output_path = PIPELINE_OUTPUTS_DIR / "annex_f_with_atc.csv"
    
    # Load Annex F
    if not annex_path.exists():
        raise FileNotFoundError(f"Annex F not found: {annex_path}")
    
    annex_df = pd.read_csv(annex_path)
    
    # Initialize and load tagger
    tagger = UnifiedTagger(
        outputs_dir=PIPELINE_OUTPUTS_DIR,
        inputs_dir=PIPELINE_INPUTS_DIR,
        verbose=False,
    )
    tagger.load()
    
    # Tag descriptions
    results_df = run_with_spinner(
        f"Tag {len(annex_df):,} Annex F entries",
        lambda: tagger.tag_descriptions(
            annex_df,
            text_column="Drug Description",
            id_column="Drug Code",
        )
    )
    
    # Merge results
    annex_df["row_idx"] = range(len(annex_df))
    # Include all columns from the tagger results
    # Core matching columns
    merge_cols = [
        "row_idx", "atc_code", "drugbank_id", "generic_name", "reference_text",
        "match_score", "match_reason", "sources",
        # Extracted form/route/dose
        "dose", "form", "route",
        # Extracted qualifiers
        "type_details", "release_details", "form_details",
        "salt_details", "brand_details", "indication_details", "alias_details",
        "diluent_details",
        # IV solution fields
        "iv_diluent_type", "iv_diluent_amount",
        # Structured dose information
        "dose_values", "dose_units", "dose_types", "total_volume_ml",
        # Computed amounts (w/v calculation for IV solutions)
        "drug_amount_mg", "diluent_amount_mg", "concentration_mg_per_ml",
    ]
    # Only include columns that exist in results_df
    merge_cols = [c for c in merge_cols if c in results_df.columns]
    merged = annex_df.merge(
        results_df[merge_cols],
        on="row_idx",
        how="left",
    ).drop(columns=["row_idx"])
    
    # Rename columns
    merged = merged.rename(columns={
        "generic_name": "matched_generic_name",
        "reference_text": "matched_reference_text",
        "sources": "matched_source",
    })
    
    # Reorder columns
    merged = reorder_columns_after(merged, "Drug Description", "matched_reference_text")
    
    # Write outputs
    PIPELINE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    run_with_spinner("Write outputs", lambda: write_csv_and_parquet(merged, output_path))
    
    tagger.close()
    
    # Summary
    total = len(merged)
    matched_atc = merged["atc_code"].notna().sum()
    matched_drugbank = merged["drugbank_id"].notna().sum()
    reason_counts = {str(reason): int(count) for reason, count in merged["match_reason"].value_counts().items() if pd.notna(reason)}

    results = {
        "total": total,
        "matched_atc": matched_atc,
        "matched_atc_pct": 100 * matched_atc / total if total else 0,
        "matched_drugbank": matched_drugbank,
        "matched_drugbank_pct": 100 * matched_drugbank / total if total else 0,
        "output_path": output_path,
        "reason_counts": reason_counts,
    }
    
    # Log metrics
    log_metrics("annex_f", {
        "total": total,
        "matched_atc": matched_atc,
        "matched_atc_pct": round(results["matched_atc_pct"], 2),
        "matched_drugbank": matched_drugbank,
        "matched_drugbank_pct": round(results["matched_drugbank_pct"], 2),
    })
    
    return results


def run_esoa_tagging(
    esoa_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> dict:
    """
    Run ESOA tagging (Part 3).
    
    Returns dict with results summary.
    """
    if esoa_path is None:
        esoa_path = PIPELINE_INPUTS_DIR / "esoa_combined.csv"
        if not esoa_path.exists():
            esoa_path = PIPELINE_INPUTS_DIR / "esoa_prepared.csv"
    if output_path is None:
        output_path = PIPELINE_OUTPUTS_DIR / "esoa_with_atc.csv"
    
    # Load ESOA
    if not esoa_path.exists():
        raise FileNotFoundError(f"ESOA not found: {esoa_path}")
    
    esoa_df = pd.read_csv(esoa_path)
    
    # Determine text column
    text_column = None
    for col in ["raw_text", "ITEM_DESCRIPTION", "DESCRIPTION", "Drug Description", "description"]:
        if col in esoa_df.columns:
            text_column = col
            break
    
    if not text_column:
        raise ValueError(f"No text column found. Columns: {list(esoa_df.columns)}")
    
    # Initialize and load tagger
    tagger = UnifiedTagger(
        outputs_dir=PIPELINE_OUTPUTS_DIR,
        inputs_dir=PIPELINE_INPUTS_DIR,
        verbose=False,
    )
    tagger.load()
    
    # Use tag_batch with deduplication for performance
    total = len(esoa_df)
    results_df = tagger.tag_batch(
        esoa_df,
        text_column=text_column,
        chunk_size=10000,
        show_progress=show_progress,
        deduplicate=True,
    )
    
    # Map results back to original rows by text
    # results_df has 'input_text' column with the original text
    results_df = results_df.rename(columns={"input_text": "_tag_text"})
    esoa_df["_tag_text"] = esoa_df[text_column].fillna("").astype(str)
    
    # Include all columns from the tagger results
    merge_cols = [
        "_tag_text", "atc_code", "drugbank_id", "generic_name", "reference_text",
        "match_score", "match_reason", "sources",
        # Extracted form/route/dose
        "dose", "form", "route",
        # Extracted qualifiers
        "type_details", "release_details", "form_details",
        "salt_details", "brand_details", "indication_details", "alias_details",
        "diluent_details",
        # IV solution fields
        "iv_diluent_type", "iv_diluent_amount",
        # Structured dose information
        "dose_values", "dose_units", "dose_types", "total_volume_ml",
        # Computed amounts (w/v calculation for IV solutions)
        "drug_amount_mg", "diluent_amount_mg", "concentration_mg_per_ml",
    ]
    # Only include columns that exist in results_df
    merge_cols = [c for c in merge_cols if c in results_df.columns]
    merged = esoa_df.merge(
        results_df[merge_cols],
        on="_tag_text",
        how="left",
    ).drop(columns=["_tag_text"])
    
    # Rename columns (standardized with annex_f_with_atc)
    merged = merged.rename(columns={
        "generic_name": "matched_generic_name",
        "reference_text": "matched_reference_text",
        "sources": "matched_source",
    })
    
    # Reorder columns
    merged = reorder_columns_after(merged, text_column, "matched_reference_text")
    
    # Write outputs
    PIPELINE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    run_with_spinner("Write outputs", lambda: write_csv_and_parquet(merged, output_path))
    
    tagger.close()
    
    # Summary
    matched_atc = merged["atc_code"].notna() & (merged["atc_code"] != "")
    matched_atc_count = matched_atc.sum()
    matched_drugbank = merged["drugbank_id"].notna() & (merged["drugbank_id"] != "")
    matched_drugbank_count = matched_drugbank.sum()
    
    results = {
        "total": total,
        "matched_atc": matched_atc_count,
        "matched_atc_pct": 100 * matched_atc_count / total if total else 0,
        "matched_drugbank": matched_drugbank_count,
        "matched_drugbank_pct": 100 * matched_drugbank_count / total if total else 0,
        "output_path": output_path,
    }
    
    reason_counts = {str(reason): int(count) for reason, count in merged["match_reason"].value_counts().items() if pd.notna(reason)}

    if verbose:
        print(f"\nESOA tagging complete: {output_path}")
        print(f"  Total: {total:,}")
        print(f"  Has ATC: {matched_atc_count:,} ({results['matched_atc_pct']:.1f}%)")
        print(f"  Has DrugBank ID: {matched_drugbank_count:,} ({results['matched_drugbank_pct']:.1f}%)")
        print("\nMatch reasons:")
        for reason, count in list(reason_counts.items())[:10]:
            pct = 100 * count / total if total else 0
            print(f"  {reason}: {count:,} ({pct:.1f}%)")
    
    # Log metrics
    log_metrics("esoa", {
        "total": total,
        "matched_atc": matched_atc_count,
        "matched_atc_pct": round(results["matched_atc_pct"], 2),
        "matched_drugbank": matched_drugbank_count,
        "matched_drugbank_pct": round(results["matched_drugbank_pct"], 2),
    })
    
    return results


def run_esoa_to_drug_code(
    esoa_path: Optional[Path] = None,
    annex_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Run ESOA to Drug Code matching (Part 4).
    
    Matches ESOA items to Annex F drug codes using EXACT matching:
    - Generic name must match exactly
    - ATC code must match (drug_code is unique per ATC)
    
    Returns dict with results summary.
    """
    if esoa_path is None:
        esoa_path = PIPELINE_OUTPUTS_DIR / "esoa_with_atc.csv"
        if not esoa_path.exists():
            esoa_path = PIPELINE_OUTPUTS_DIR / "esoa_with_atc.parquet"
    if annex_path is None:
        annex_path = PIPELINE_OUTPUTS_DIR / "annex_f_with_atc.csv"
        if not annex_path.exists():
            annex_path = PIPELINE_OUTPUTS_DIR / "annex_f_with_atc.parquet"
    if output_path is None:
        output_path = PIPELINE_OUTPUTS_DIR / "esoa_with_drug_code.csv"
    
    # Load data
    if not esoa_path.exists():
        raise FileNotFoundError(f"ESOA with ATC not found: {esoa_path}")
    if not annex_path.exists():
        raise FileNotFoundError(f"Annex F with ATC not found: {annex_path}")
    
    if str(esoa_path).endswith('.parquet'):
        esoa_df = run_with_spinner("Load ESOA", lambda: pd.read_parquet(esoa_path))
    else:
        esoa_df = run_with_spinner("Load ESOA", lambda: pd.read_csv(esoa_path))
    
    if str(annex_path).endswith('.parquet'):
        annex_df = run_with_spinner("Load Annex F", lambda: pd.read_parquet(annex_path))
    else:
        annex_df = run_with_spinner("Load Annex F", lambda: pd.read_csv(annex_path))
    
    if verbose:
        print(f"  ESOA rows: {len(esoa_df):,}")
        print(f"  Annex F rows: {len(annex_df):,}")
    
    # Build Annex F lookup index by generic name
    def normalize_for_match(s):
        if pd.isna(s):
            return ""
        return str(s).upper().strip()
    
    # Build synonym mappings from generics_master and merge with static constants
    generics_master_path = PIPELINE_OUTPUTS_DIR / "generics_master.csv"
    if not generics_master_path.exists():
        generics_master_path = PIPELINE_OUTPUTS_DIR / "generics_master.parquet"
    all_synonyms = dict(ALL_DRUG_SYNONYMS)  # Start with static synonyms from unified_constants
    
    if generics_master_path.exists():
        if str(generics_master_path).endswith('.parquet'):
            gm = pd.read_parquet(generics_master_path)
        else:
            gm = pd.read_csv(generics_master_path)
        for _, row in gm.iterrows():
            generic = str(row['generic_name']).upper().strip()
            synonyms_str = row.get('synonyms', '')
            if synonyms_str:
                for syn in str(synonyms_str).split('|'):
                    syn = syn.upper().strip()
                    if syn and syn != generic:
                        # Bidirectional mapping
                        all_synonyms[syn] = generic
                        all_synonyms[generic] = syn
        if verbose:
            print(f"  Loaded synonyms: {len(ALL_DRUG_SYNONYMS)} static + {len(all_synonyms) - len(ALL_DRUG_SYNONYMS)} from generics_master")
    
    def get_all_name_variants(name):
        """Get all possible name variants for matching."""
        variants = {name}
        if name in all_synonyms:
            variants.add(all_synonyms[name])
        # Also check if name appears as a value (reverse lookup)
        for syn, canonical in all_synonyms.items():
            if canonical == name:
                variants.add(syn)
        return variants
    
    # Use constants from unified_constants.py (imported at module level)
    # GARBAGE_TOKENS, ALL_DRUG_SYNONYMS, DRUGBANK_COMPONENT_SYNONYMS
    
    import re
    
    # IV diluent equivalence - diluents that are clinically interchangeable
    # NOTE: Water and Saline are NOT interchangeable (different osmolarity)
    # NOTE: Lactated Ringer's and Acetated Ringer's are NOT interchangeable (different buffer)
    DILUENT_EQUIVALENTS = {
        # Water variants
        "WATER": "WATER",
        "WATER FOR INJECTION": "WATER",
        "STERILE WATER": "WATER",
        "WFI": "WATER",
        # Normal saline variants (0.9% NaCl)
        "SODIUM CHLORIDE": "NORMAL_SALINE",
        "NORMAL SALINE": "NORMAL_SALINE",
        "NS": "NORMAL_SALINE",
        "0.9% SODIUM CHLORIDE": "NORMAL_SALINE",
        "0.9% NACL": "NORMAL_SALINE",
        # Half-normal saline (0.45% NaCl) - different from normal saline
        "0.45% SODIUM CHLORIDE": "HALF_SALINE",
        "0.45% NACL": "HALF_SALINE",
        "HALF NORMAL SALINE": "HALF_SALINE",
        # Lactated Ringer's - NOT equivalent to Acetated Ringer's
        "LACTATED RINGER'S": "LACTATED_RINGERS",
        "LACTATED RINGERS": "LACTATED_RINGERS",
        "LR": "LACTATED_RINGERS",
        "RL": "LACTATED_RINGERS",
        # Acetated Ringer's - NOT equivalent to Lactated Ringer's
        "ACETATED RINGER'S": "ACETATED_RINGERS",
        "ACETATED RINGERS": "ACETATED_RINGERS",
        "AR": "ACETATED_RINGERS",
    }
    
    def normalize_diluent(diluent: str) -> str:
        """Normalize diluent name to canonical form for comparison."""
        if not diluent:
            return None
        d = str(diluent).upper().strip()
        return DILUENT_EQUIVALENTS.get(d, d)  # Return canonical or original if not found
    
    # Unit conversion to mg (for weight-based units)
    UNIT_TO_MG = {
        "MG": 1.0,
        "G": 1000.0,
        "GM": 1000.0,
        "GRAM": 1000.0,
        "MCG": 0.001,
        "UG": 0.001,
        "MICROGRAM": 0.001,
        "KG": 1000000.0,
    }
    
    def parse_combo_dose(dose_str):
        """
        Parse combination doses like "500MG+125MG" or "500MG/125MG" or "500|MG|125".
        
        Returns: (component_doses_mg, total_mg, per_volume_ml) or (None, None, None) if not a combo
        
        Handles:
        - "500MG+125MG" → ([500, 125], 625, None) - tablet combo
        - "500MG/125MG" → ([500, 125], 625, None) - tablet combo
        - "250|MG|125" → ([250, 125], 375, None) - Annex F tablet
        - "400|MG|57|ML|35" → ([400, 57], 457, None) - Annex F suspension per 5mL (5mL implicit)
        - "457MG/5ML" → concentration, not combo (handled by parse_dose_to_mg)
        """
        if not dose_str or pd.isna(dose_str):
            return None, None, None
        
        dose_str = str(dose_str).upper().strip()
        
        # Skip if this is clearly a concentration (number/ML or number/L pattern)
        if re.search(r'\d+\s*(MG|G|MCG)?\s*/\s*\d*\s*M?L\b', dose_str):
            return None, None, None
        
        # Pattern 1: "500MG+125MG" (explicit combo with +)
        plus_match = re.findall(r'(\d+(?:\.\d+)?)\s*(MG|G|MCG)\s*\+\s*(\d+(?:\.\d+)?)\s*(MG|G|MCG)?', dose_str)
        if plus_match:
            components = []
            for match in plus_match:
                val1 = float(match[0])
                unit1 = match[1]
                val2 = float(match[2])
                unit2 = match[3] if match[3] else unit1
                
                mg1 = val1 * UNIT_TO_MG.get(unit1, 1.0)
                mg2 = val2 * UNIT_TO_MG.get(unit2, 1.0)
                components.extend([mg1, mg2])
            
            if components:
                return components, sum(components), None
        
        # Pattern 2: "500MG/125MG" (combo with / but BOTH have weight units)
        slash_match = re.match(r'^(\d+(?:\.\d+)?)\s*(MG|G|MCG)\s*/\s*(\d+(?:\.\d+)?)\s*(MG|G|MCG)$', dose_str)
        if slash_match:
            val1 = float(slash_match.group(1))
            unit1 = slash_match.group(2)
            val2 = float(slash_match.group(3))
            unit2 = slash_match.group(4)
            
            mg1 = val1 * UNIT_TO_MG.get(unit1, 1.0)
            mg2 = val2 * UNIT_TO_MG.get(unit2, 1.0)
            return [mg1, mg2], mg1 + mg2, None
        
        # Pattern 3: Annex F pipe format - "250|MG|125" or "400|MG|57|ML|35"
        # Parse all numeric values and identify doses vs volumes
        # For combo drugs like CO-AMOXICLAV: 400|MG|57|ML|35 = 400mg + 57mg per 5mL in 35mL
        # The 57 before ML is a dose component, not a volume!
        # BUT: "250|MG|1|G" means 250mg in a 1g vial - NOT a combo!
        parts = dose_str.replace(' ', '').split('|')
        doses = []
        bottle_vol = None
        last_was_dose = False
        last_unit = None
        
        i = 0
        while i < len(parts):
            part = parts[i]
            if re.match(r'^\d+(?:\.\d+)?$', part):
                num = float(part)
                # Check what comes after
                if i + 1 < len(parts):
                    next_part = parts[i + 1]
                    if next_part in ('MG', 'G', 'MCG'):
                        # Check if this is a vial size (e.g., "1|G" after "250|MG")
                        # Vial sizes are typically 1G, 2G, etc. - round numbers
                        # If we already have a dose in MG and this is in G, it's likely vial size
                        if last_unit == 'MG' and next_part == 'G' and num <= 10:
                            # This is likely a vial size, not a second dose
                            i += 2
                            continue
                        doses.append(num * UNIT_TO_MG.get(next_part, 1.0))
                        last_was_dose = True
                        last_unit = next_part
                        i += 2
                        continue
                    elif next_part == 'ML':
                        # If we just had a dose, this number is likely a second dose component
                        # e.g., 400|MG|57|ML|35 where 57 is the second dose
                        if last_was_dose and num < 1000:  # Reasonable dose range
                            doses.append(num)  # Assume MG
                            last_was_dose = True
                            i += 2  # Skip the ML
                            continue
                        else:
                            # This is a volume
                            bottle_vol = num
                            last_was_dose = False
                            i += 2
                            continue
                # Standalone number after MG - probably second dose component
                # But NOT if it's followed by G (vial size)
                if i > 0 and parts[i-1] in ('MG', 'G', 'MCG'):
                    # Check if next part is G (vial size indicator)
                    if i + 1 < len(parts) and parts[i + 1] == 'G':
                        i += 2  # Skip vial size
                        continue
                    doses.append(num)  # Assume same unit as previous
                    last_was_dose = True
                    i += 1
                    continue
            else:
                last_was_dose = False
                last_unit = None
            i += 1
        
        if len(doses) >= 2:
            return doses, sum(doses), bottle_vol
        
        return None, None, None
    
    def parse_dose_to_mg(dose_str):
        """
        Parse dose string to extract effective dose value and concentration.
        
        NORMALIZATION RULES:
        1. All weights converted to MG (G→1000mg, MCG→0.001mg)
        2. Bare numbers without units assumed to be MG (e.g., "275" → 275mg)
        3. Concentrations normalized to mg/mL
        4. Percentages converted to mg/mL (X% = X*10 mg/mL)
        5. Pipe-separated Annex F format normalized (e.g., "200|MG" → 200mg)
        
        Returns: (total_dose_mg, concentration_mg_per_ml, volume_ml, unit_type)
        """
        if not dose_str or pd.isna(dose_str):
            return None, None, None, None
        
        dose_str = str(dose_str).upper().strip()
        
        # First check for combination doses
        combo_components, combo_total, combo_vol = parse_combo_dose(dose_str)
        if combo_total is not None:
            return combo_total, None, combo_vol, "combo"
        
        # Normalize pipes to spaces for parsing
        dose_str = dose_str.replace("|", " ")
        
        # Clean up common formatting issues
        dose_str = re.sub(r'\s+', ' ', dose_str)  # Multiple spaces to single
        dose_str = re.sub(r'(\d)\s+(\d)', r'\1\2', dose_str)  # "200 000" → "200000"
        
        total_dose = None
        concentration = None
        volume_ml = None
        unit_type = None
        
        # Pattern 0: IU concentration like "1000IU/ML" or "1000 IU/ML" or "1000 I.U/ML"
        iu_conc_match = re.search(r'(\d+(?:\.\d+)?)\s*I\.?U\.?\s*/\s*(ML|L)', dose_str)
        if iu_conc_match:
            val = float(iu_conc_match.group(1))
            vol_unit = iu_conc_match.group(2)
            if vol_unit == "L":
                concentration = val / 1000.0
            else:
                concentration = val
            unit_type = "iu"
        
        # Pattern 0b: IU dose/volume like "1000IU/5ML" or "1000 I.U/5ML"
        iu_dose_vol_match = re.search(r'(\d+(?:\.\d+)?)\s*I\.?U\.?\s*/\s*(\d+(?:\.\d+)?)\s*(ML|L)', dose_str)
        if iu_dose_vol_match:
            dose_val = float(iu_dose_vol_match.group(1))
            vol_val = float(iu_dose_vol_match.group(2))
            vol_unit = iu_dose_vol_match.group(3)
            
            total_dose = dose_val
            if vol_unit == "L":
                volume_ml = vol_val * 1000.0
            else:
                volume_ml = vol_val
            
            if volume_ml and volume_ml > 0:
                concentration = total_dose / volume_ml
            unit_type = "iu"
        
        # Pattern 0c: Simple IU like "10IU" or "10 IU" or "10 I.U" or "200 000 IU"
        if unit_type is None:
            iu_simple_match = re.search(r'(\d+(?:\.\d+)?)\s*I\.?U\.?\b', dose_str)
            if iu_simple_match:
                total_dose = float(iu_simple_match.group(1))
                unit_type = "iu"
        
        # Pattern 1: concentration like "100MG/ML" or "100 MG/ML"
        if unit_type is None:
            conc_match = re.search(r'(\d+(?:\.\d+)?)\s*(MG|G|MCG|UG)/\s*(ML|L)', dose_str)
            if conc_match:
                val = float(conc_match.group(1))
                unit = conc_match.group(2)
                vol_unit = conc_match.group(3)
                
                # Convert to mg
                mg_val = val * UNIT_TO_MG.get(unit, 1.0)
                
                # Convert to per mL
                if vol_unit == "L":
                    concentration = mg_val / 1000.0
                else:
                    concentration = mg_val
                unit_type = "mg"
        
        # Pattern 2: dose/volume like "300MG/2ML" or "250MG/5ML 60ML" (suspension with bottle size)
        if unit_type is None or unit_type == "mg":
            dose_vol_match = re.search(r'(\d+(?:\.\d+)?)\s*(MG|G|MCG|UG)\s*/\s*(\d+(?:\.\d+)?)\s*(ML|L)', dose_str)
            if dose_vol_match:
                dose_val = float(dose_vol_match.group(1))
                dose_unit = dose_vol_match.group(2)
                vol_val = float(dose_vol_match.group(3))
                vol_unit = dose_vol_match.group(4)
                
                # Convert dose to mg
                total_dose = dose_val * UNIT_TO_MG.get(dose_unit, 1.0)
                
                # The denominator volume (e.g., 5ML in 250MG/5ML) for concentration
                denom_vol = vol_val * 1000.0 if vol_unit == "L" else vol_val
                
                # Calculate concentration
                if denom_vol and denom_vol > 0:
                    concentration = total_dose / denom_vol
                unit_type = "mg"
                
                # Look for a SEPARATE bottle volume after the concentration (e.g., "250MG/5ML 60ML")
                # This is the actual bottle size, distinct from the concentration denominator
                after_conc = dose_str[dose_vol_match.end():]
                bottle_match = re.search(r'(\d+(?:\.\d+)?)\s*(ML|L)\b', after_conc)
                if bottle_match:
                    bottle_val = float(bottle_match.group(1))
                    bottle_unit = bottle_match.group(2)
                    volume_ml = bottle_val * 1000.0 if bottle_unit == "L" else bottle_val
                else:
                    # No separate bottle size, use the denominator as volume
                    volume_ml = denom_vol
        
        # Pattern 3: simple dose like "40MG" or "40 MG" or "1GM" or "1 G"
        if total_dose is None and concentration is None and unit_type is None:
            simple_match = re.search(r'(\d+(?:\.\d+)?)\s*(MG|G|GM|GRAM|MCG|UG|MICROGRAM)\b', dose_str)
            if simple_match:
                val = float(simple_match.group(1))
                unit = simple_match.group(2)
                total_dose = val * UNIT_TO_MG.get(unit, 1.0)
                unit_type = "mg"
        
        # Pattern 3b: Annex F pipe format with unit - "200 MG" (from "200|MG")
        if total_dose is None and concentration is None and unit_type is None:
            annex_match = re.match(r'^(\d+(?:\.\d+)?)\s+(MG|G|MCG|UG)\s*$', dose_str)
            if annex_match:
                val = float(annex_match.group(1))
                unit = annex_match.group(2)
                total_dose = val * UNIT_TO_MG.get(unit, 1.0)
                unit_type = "mg"
        
        # Pattern 3c: bare numeric dose like "25" or "500" or "275" (assume MG)
        # This handles cases like "FLANAX 275" where 275 is naproxen sodium 275mg
        if total_dose is None and concentration is None and unit_type is None:
            # Match bare number, possibly with trailing non-unit text
            bare_match = re.match(r'^(\d+(?:\.\d+)?)\s*(?:$|[^A-Z0-9]|TAB|CAP|TABLET|CAPSULE)', dose_str)
            if bare_match:
                val = float(bare_match.group(1))
                # Treat as MG for reasonable tablet doses (0.1-10000 range)
                if 0.1 <= val <= 10000:
                    total_dose = val
                    unit_type = "mg"
        
        # Pattern 4: standalone volume like "15ML" or "500 ML" (only if not already extracted)
        if volume_ml is None:
            # Find ALL volume matches and take the LAST one (likely bottle size)
            vol_matches = list(re.finditer(r'(\d+(?:\.\d+)?)\s*(ML|L|CC)\b', dose_str))
            if vol_matches:
                last_match = vol_matches[-1]
                vol_val = float(last_match.group(1))
                vol_unit = last_match.group(2)
                if vol_unit == "L":
                    volume_ml = vol_val * 1000.0
                elif vol_unit == "CC":
                    volume_ml = vol_val  # CC = mL
                else:
                    volume_ml = vol_val
        
        # Pattern 5: percentage like "0.9%" or "5%" or ".9%"
        if total_dose is None and concentration is None and unit_type is None:
            pct_match = re.search(r'(\d*\.?\d+)\s*%', dose_str)
            if pct_match:
                pct_val = float(pct_match.group(1))
                # Fix common parsing errors: 9% is likely 0.9% for saline
                if pct_val == 9:
                    pct_val = 0.9  # Common error: .9% parsed as 9%
                # Convert percentage to mg/mL using w/v formula: X% = X g/100mL = X*10 mg/mL
                concentration = pct_val * 10.0
                unit_type = "pct"
        
        return total_dose, concentration, volume_ml, unit_type
    
    def get_dose_key(row):
        """
        Build a dose key for matching using structured dose columns,
        falling back to parsing the dose string if needed.
        
        Returns a tuple for precise matching:
        - IV solutions: ("iv", concentration_mg_per_ml, normalized_diluent, total_volume_ml)
        - Concentration drugs: ("conc", concentration_per_ml, total_volume_ml, unit_type)
        - Simple drugs: ("mg", total_mg) or ("iu", total_iu)
        """
        drug_mg = row.get("drug_amount_mg")
        conc = row.get("concentration_mg_per_ml")
        iv_type = row.get("iv_diluent_type")
        total_vol = row.get("total_volume_ml")
        dose_str = row.get("dose")
        
        # For IV solutions with diluent type, use concentration + diluent type + volume
        if pd.notna(iv_type) and iv_type:
            return ("iv", float(conc) if pd.notna(conc) else None, normalize_diluent(iv_type), float(total_vol) if pd.notna(total_vol) else None)
        
        # If structured columns available, use them
        if pd.notna(drug_mg) and drug_mg:
            if pd.notna(conc) and conc:
                return ("conc", float(conc), float(total_vol) if pd.notna(total_vol) else None, "mg")
            return ("mg", float(drug_mg))
        
        # Parse dose string to extract values
        parsed_dose, parsed_conc, parsed_vol, unit_type = parse_dose_to_mg(dose_str)
        
        # If we have a concentration, use concentration-based matching
        if parsed_conc is not None:
            return ("conc", parsed_conc, parsed_vol, unit_type)
        
        # If we have a simple dose value, use type-based matching
        if parsed_dose is not None:
            if unit_type == "iu":
                return ("iu", parsed_dose)
            return ("mg", parsed_dose)
        
        # Special handling: common IV solutions with only volume
        # Get description or generic name for context
        desc = str(row.get("DESCRIPTION") or row.get("Drug Description") or "").upper()
        generic = str(row.get("matched_generic_name") or "").upper()
        
        if parsed_vol is not None and parsed_vol > 0:
            # Check if this is plain NSS/PNSS (sodium chloride without specific percentage)
            is_nss = any(kw in desc for kw in ["PNSS", "NSS", "PLAIN NSS", "NORMAL SALINE", "N/S"]) or \
                     ("SODIUM CHLORIDE" in generic and "DEXTROSE" not in generic)
            if is_nss and "%" not in str(dose_str or ""):
                # Assume 0.9% for plain NSS: 0.9% = 9 mg/mL
                return ("conc", 9.0, parsed_vol, "pct")
            
            # Check if this is D5 (5% Dextrose) - "D5" prefix in description
            is_d5 = re.search(r'\bD5\b', desc) is not None or "5% DEXTROSE" in desc
            if is_d5 and "DEXTROSE" in generic and "%" not in str(dose_str or ""):
                # Assume 5% for D5: 5% = 50 mg/mL
                return ("conc", 50.0, parsed_vol, "pct")
            
            # Check if this is D10 (10% Dextrose)
            is_d10 = re.search(r'\bD10\b', desc) is not None or "10% DEXTROSE" in desc
            if is_d10 and "DEXTROSE" in generic and "%" not in str(dose_str or ""):
                # Assume 10% for D10: 10% = 100 mg/mL
                return ("conc", 100.0, parsed_vol, "pct")
        
        # No dose info available
        return None
    
    def doses_match(annex_key, esoa_key):
        """
        Compare dose keys for matching with ZERO TOLERANCE.
        
        For IV solutions: concentration + diluent type + volume must match EXACTLY
        For concentration drugs: concentration must match EXACTLY (unit type must be compatible)
        For simple drugs: total dose must match EXACTLY
        For IU drugs: IU must match other IU (not mg)
        
        Cross-type matching is allowed when equivalent:
        - "mg" 40mg can match "conc" 40mg/mL if volume context allows
        - "conc" with same concentration matches regardless of volume (volume optional)
        """
        if annex_key is None or esoa_key is None:
            return False
        
        annex_type, esoa_type = annex_key[0], esoa_key[0]
        
        # IV solutions only match other IV solutions
        if annex_type == "iv" or esoa_type == "iv":
            if annex_type != esoa_type:
                return False
            # IV solutions: concentration + diluent type + volume (ZERO TOLERANCE)
            a_conc, a_dil, a_vol = annex_key[1], annex_key[2], annex_key[3]
            e_conc, e_dil, e_vol = esoa_key[1], esoa_key[2], esoa_key[3]
            
            # Concentration must match EXACTLY
            if a_conc != e_conc:
                return False
            
            # Diluent type must match (using normalized equivalents)
            if a_dil != e_dil:
                return False
            
            # Volume must match EXACTLY if both present
            if a_vol is not None and e_vol is not None:
                if a_vol != e_vol:
                    return False
            
            return True
        
        # IU type matching - IU only matches IU, not mg
        if annex_type == "iu" or esoa_type == "iu":
            if annex_type != esoa_type:
                # Special case: IU with concentration can match IU simple
                # e.g., "1000IU/ML|5ML" vs "1000IU/ML|5ML"
                pass  # Fall through to conc matching below
            else:
                # Both are simple IU
                return annex_key[1] == esoa_key[1]
        
        # Both are "mg" type - comparison with small tolerance
        if annex_type == "mg" and esoa_type == "mg":
            a_mg, e_mg = annex_key[1], esoa_key[1]
            if a_mg is None or e_mg is None:
                return a_mg == e_mg
            diff = abs(a_mg - e_mg)
            rel_diff = diff / max(a_mg, e_mg, 1.0)
            # Allow 1% relative difference or 0.5 mg absolute difference
            return diff <= 0.5 or rel_diff <= 0.01
        
        # Combo type matching - combo total can match mg or other combo
        if annex_type == "combo" or esoa_type == "combo":
            # Get the dose values
            if annex_type == "combo":
                a_val = annex_key[1]
            elif annex_type == "mg":
                a_val = annex_key[1]
            else:
                a_val = None
                
            if esoa_type == "combo":
                e_val = esoa_key[1]
            elif esoa_type == "mg":
                e_val = esoa_key[1]
            else:
                e_val = None
            
            # Both must have values and match
            if a_val is not None and e_val is not None:
                if abs(a_val - e_val) < 0.01:
                    return True
            return False
        
        # Both are "conc" type - compare concentration only (volume is just packaging)
        if annex_type == "conc" and esoa_type == "conc":
            a_conc, a_vol = annex_key[1], annex_key[2]
            e_conc, e_vol = esoa_key[1], esoa_key[2]
            # Get unit types if available (4th element)
            a_unit = annex_key[3] if len(annex_key) > 3 else "mg"
            e_unit = esoa_key[3] if len(esoa_key) > 3 else "mg"
            
            # Unit types must be compatible (mg can match pct since both are mg/mL)
            if a_unit == "iu" and e_unit != "iu":
                return False
            if e_unit == "iu" and a_unit != "iu":
                return False
            
            # Concentration must match with small tolerance for floating point precision
            # Allow 1% relative difference or 0.1 mg/mL absolute difference
            if a_conc is None or e_conc is None:
                return a_conc == e_conc
            diff = abs(a_conc - e_conc)
            rel_diff = diff / max(a_conc, e_conc, 1.0)
            if diff > 0.1 and rel_diff > 0.01:  # More than 0.1 mg/mL AND more than 1%
                return False
            # Note: We do NOT require volume match - 5mL vial of 100mg/mL = 10mL vial of 100mg/mL
            # Both are the same drug at the same concentration, just different packaging
            return True
        
        # Cross-type matching: "mg" vs "conc"
        # This handles cases like Annex F "40|MG" vs ESOA "40MG/ML|1ML"
        if (annex_type == "mg" and esoa_type == "conc") or (annex_type == "conc" and esoa_type == "mg"):
            if annex_type == "mg":
                mg_val = annex_key[1]
                conc_val, vol = esoa_key[1], esoa_key[2]
                conc_unit = esoa_key[3] if len(esoa_key) > 3 else "mg"
            else:
                mg_val = esoa_key[1]
                conc_val, vol = annex_key[1], annex_key[2]
                conc_unit = annex_key[3] if len(annex_key) > 3 else "mg"
            
            # IU concentration can't match mg simple
            if conc_unit == "iu":
                return False
            
            # If concentration has volume, check if total dose matches
            if vol is not None and vol > 0:
                total_from_conc = conc_val * vol
                if abs(total_from_conc - mg_val) < 0.01:  # Small tolerance for floating point
                    return True
            
            # Also allow matching if concentration equals mg (1mL implied)
            if abs(conc_val - mg_val) < 0.01:
                return True
            
            return False
        
        # Cross-type matching: "iu" vs "conc" (IU concentration)
        if (annex_type == "iu" and esoa_type == "conc") or (annex_type == "conc" and esoa_type == "iu"):
            if annex_type == "iu":
                iu_val = annex_key[1]
                conc_val, vol = esoa_key[1], esoa_key[2]
                conc_unit = esoa_key[3] if len(esoa_key) > 3 else None
            else:
                iu_val = esoa_key[1]
                conc_val, vol = annex_key[1], annex_key[2]
                conc_unit = annex_key[3] if len(annex_key) > 3 else None
            
            # Only match if concentration is also IU type
            if conc_unit != "iu":
                return False
            
            # If concentration has volume, check if total IU matches
            if vol is not None and vol > 0:
                total_from_conc = conc_val * vol
                if abs(total_from_conc - iu_val) < 0.01:
                    return True
            
            # Also allow matching if concentration equals IU (1mL implied)
            if abs(conc_val - iu_val) < 0.01:
                return True
            
            return False
        
        return False
    
    def rank_candidate_for_drug_code(cand, esoa_row):
        """
        Rank candidates for Part 4 tie-breaking using all *_details columns.
        
        Lower score = better match.
        """
        score = 0
        cand_desc = str(cand.get("description", "")).upper()
        
        # Extract ESOA details for comparison
        esoa_release = str(esoa_row.get("release_details") or "").upper()
        esoa_type = str(esoa_row.get("type_details") or "").upper()
        esoa_form_det = str(esoa_row.get("form_details") or "").upper()
        esoa_indication = str(esoa_row.get("indication_details") or "").upper()
        esoa_salt = str(esoa_row.get("salt_details") or "").upper()
        esoa_alias = str(esoa_row.get("alias_details") or "").upper()
        esoa_iv_type = str(esoa_row.get("iv_diluent_type") or "").upper()
        esoa_iv_amount = str(esoa_row.get("iv_diluent_amount") or "").upper()
        
        # Release details match (e.g., MR, SR, XR, ER) - highest priority
        if esoa_release and esoa_release in cand_desc:
            score -= 10
        
        # Type details match (e.g., HUMAN, ANHYDROUS)
        if esoa_type and esoa_type in cand_desc:
            score -= 5
        
        # Form details match (e.g., FILM COATED, CHEWABLE)
        if esoa_form_det and esoa_form_det in cand_desc:
            score -= 5
        
        # Indication details match (e.g., FOR HEPATIC FAILURE)
        if esoa_indication and esoa_indication in cand_desc:
            score -= 5
        
        # Salt details match
        if esoa_salt and esoa_salt in cand_desc:
            score -= 3
        
        # Alias details match (e.g., VIT. D3 = CHOLECALCIFEROL)
        if esoa_alias and esoa_alias in cand_desc:
            score -= 2
        
        # IV diluent type match
        if esoa_iv_type and esoa_iv_type in cand_desc:
            score -= 5
        
        # IV diluent amount match (e.g., 0.9%, 0.45%)
        if esoa_iv_amount and esoa_iv_amount in cand_desc:
            score -= 3
        
        return score
    
    annex_lookup = {}  # generic_name -> list of candidates
    drugbank_lookup = {}  # drugbank_id -> list of candidates
    for _, row in annex_df.iterrows():
        drug_code = row.get("Drug Code")
        if pd.isna(drug_code):
            continue
        
        generic_raw = row.get("matched_generic_name") or row.get("generic_name") or ""
        drug_desc = row.get("Drug Description") or ""
        
        # Extract clean generics from pipe-separated string (Annex F also has garbage)
        annex_generics = []
        for part in str(generic_raw).split('|'):
            part = part.strip().upper()
            if not part or part in GARBAGE_TOKENS or len(part) <= 2:
                continue
            # Skip pure dose patterns (e.g., "500MG", "100ML")
            # But allow drug names with numbers (e.g., "GENTAMICIN C2", "VITAMIN B12")
            if re.match(r'^\d+(\.\d+)?\s*(MG|ML|MCG|G|IU|%|CC|L)$', part, re.IGNORECASE):
                continue
            if part.replace('.', '').isdigit():
                continue
            annex_generics.append(part)
        
        if not annex_generics:
            continue
        
        atc = normalize_for_match(row.get("atc_code"))
        drugbank_id = row.get("drugbank_id")
        if pd.notna(drugbank_id):
            drugbank_id = str(drugbank_id).strip()
        else:
            drugbank_id = None
        
        # Use structured dose columns for matching
        dose_key = get_dose_key(row)
        
        # Extract form and route from Annex F
        annex_form = normalize_for_match(row.get("form"))
        annex_route = normalize_for_match(row.get("route"))
        
        candidate = {
            "drug_code": drug_code,
            "atc_code": atc,
            "drugbank_id": drugbank_id,
            "generic_name": annex_generics[0],  # Primary generic
            "dose_key": dose_key,  # Structured dose key for matching
            "form": annex_form,
            "route": annex_route,
            "description": drug_desc,
        }
        
        # Index by each generic component and its synonyms
        for generic in annex_generics:
            if generic not in annex_lookup:
                annex_lookup[generic] = []
            annex_lookup[generic].append(candidate)
            
            # Also index by base name without parentheticals (e.g., "ASCORBIC ACID (VITAMIN C)" -> "ASCORBIC ACID")
            base_generic = re.sub(r'\s*\([^)]*\)', '', generic).strip()
            if base_generic and base_generic != generic:
                if base_generic not in annex_lookup:
                    annex_lookup[base_generic] = []
                annex_lookup[base_generic].append(candidate)
            
            # Also add synonym mappings (from unified_constants)
            if generic in ALL_DRUG_SYNONYMS:
                syn = ALL_DRUG_SYNONYMS[generic]
                if syn not in annex_lookup:
                    annex_lookup[syn] = []
                annex_lookup[syn].append(candidate)
            # Check base generic for synonyms too
            if base_generic and base_generic in ALL_DRUG_SYNONYMS:
                syn = ALL_DRUG_SYNONYMS[base_generic]
                if syn not in annex_lookup:
                    annex_lookup[syn] = []
                annex_lookup[syn].append(candidate)
        
        # Index by drugbank_id
        if drugbank_id:
            if drugbank_id not in drugbank_lookup:
                drugbank_lookup[drugbank_id] = []
            drugbank_lookup[drugbank_id].append(candidate)
    
    if verbose:
        print(f"  Annex F lookup: {len(annex_lookup):,} unique generics")
        print(f"  DrugBank lookup: {len(drugbank_lookup):,} unique drugbank_ids")
    
    def extract_clean_generics(generic_str):
        """Extract clean generic names from pipe-separated string."""
        if not generic_str:
            return []
        parts = [p.strip().upper() for p in str(generic_str).split('|')]
        # Filter out garbage and deduplicate while preserving order
        seen = set()
        clean = []
        for p in parts:
            if not p or p in GARBAGE_TOKENS or p in seen or len(p) <= 2:
                continue
            # Skip if looks like a pure dose (e.g., "500MG", "100ML", "10%")
            # But allow vitamin names like "B1", "B12", "B6"
            import re
            if re.match(r'^\d+(\.\d+)?\s*(MG|ML|MCG|G|IU|%|CC|L)$', p, re.IGNORECASE):
                continue
            # Skip pure numbers
            if p.replace('.', '').isdigit():
                continue
            seen.add(p)
            clean.append(p)
        return clean
    
    def extract_generics_from_description(desc):
        """Fallback: extract generic names from DESCRIPTION when generic_final is empty."""
        if not desc:
            return []
        desc = str(desc).upper()
        generics = []
        
        # Split on common separators
        # Handle "ALUMINUM+MAGNESIUM", "IBUPROFEN + PARACETAMOL", etc.
        parts = re.split(r'[+/]|\s+AND\s+|\s+\+\s+', desc)
        
        for part in parts:
            # Extract the first word(s) before dose info
            # e.g., "ALUMINUM 200MG" -> "ALUMINUM"
            match = re.match(r'^([A-Z][A-Z\s\-]+?)(?:\s*\d|\s*\(|$)', part.strip())
            if match:
                generic = match.group(1).strip()
                # Clean up
                generic = re.sub(r'\s+', ' ', generic)
                if generic and len(generic) > 2 and generic not in GARBAGE_TOKENS:
                    generics.append(generic)
        
        return generics
    
    # Match ESOA to Annex F - STRICT MATCHING
    # Only matches when: generic + dose + form + route all match (salt can vary)
    # Brand names are resolved to generics for matching (brand doesn't matter, only underlying generic)
    def match_to_drug_code(row):
        generic_raw = row.get("matched_generic_name") or row.get("generic_name") or ""
        
        # Fix known wrong synonyms (from unified_constants)
        for wrong, correct in DRUGBANK_COMPONENT_SYNONYMS.items():
            if wrong in str(generic_raw).upper():
                generic_raw = str(generic_raw).upper().replace(wrong, correct)
        
        generics = extract_clean_generics(generic_raw)
        
        # Fallback: if no generics from matched_generic_name, try extracting from DESCRIPTION
        esoa_desc = row.get("DESCRIPTION") or ""
        if not generics:
            generics = extract_generics_from_description(esoa_desc)
        
        if not generics:
            return None, "no_generic"
        
        # Use structured dose columns for matching (same as Annex F)
        esoa_dose_key = get_dose_key(row)
        esoa_form = normalize_for_match(row.get("form"))
        esoa_route = normalize_for_match(row.get("route"))
        
        # Try each generic component against the lookup
        candidates = []
        for generic in generics:
            # Try all name variants (original + synonyms)
            for variant in get_all_name_variants(generic):
                candidates.extend(annex_lookup.get(variant, []))
        
        if not candidates:
            return None, "generic_not_in_annex"
        
        # Deduplicate candidates by drug_code
        seen_codes = set()
        unique_candidates = []
        for c in candidates:
            if c["drug_code"] not in seen_codes:
                seen_codes.add(c["drug_code"])
                unique_candidates.append(c)
        candidates = unique_candidates
        
        # Use FORM_EQUIVALENTS from unified_constants.py for basic form equivalence
        # Use FORM_TO_ROUTES to check if forms can share the same route
        
        def forms_compatible(cand_form, esoa_form, cand_route=None, esoa_route=None):
            """
            Check if forms are compatible, considering:
            1. Direct form equivalence (from FORM_EQUIVALENTS)
            2. Forms that can share the same route (from FORM_TO_ROUTES)
            """
            if not esoa_form or not cand_form:
                return True  # Missing form = compatible
            
            cand_form_upper = cand_form.upper().strip()
            esoa_form_upper = esoa_form.upper().strip()
            
            if cand_form_upper == esoa_form_upper:
                return True
            
            # Check direct form equivalence from unified_constants
            if cand_form_upper in FORM_EQUIVALENTS:
                if esoa_form_upper in FORM_EQUIVALENTS.get(cand_form_upper, set()):
                    return True
            if esoa_form_upper in FORM_EQUIVALENTS:
                if cand_form_upper in FORM_EQUIVALENTS.get(esoa_form_upper, set()):
                    return True
            
            # Check if forms can share the same route using FORM_TO_ROUTES
            # Get all valid routes for each form
            cand_routes = set(FORM_TO_ROUTES.get(cand_form_upper, []))
            esoa_routes = set(FORM_TO_ROUTES.get(esoa_form_upper, []))
            
            # If either has no route mapping, try partial matching on form name
            if not cand_routes:
                # Try to find a matching key in FORM_TO_ROUTES
                for key in FORM_TO_ROUTES:
                    if key in cand_form_upper or cand_form_upper in key:
                        cand_routes.update(FORM_TO_ROUTES[key])
                        break
            if not esoa_routes:
                for key in FORM_TO_ROUTES:
                    if key in esoa_form_upper or esoa_form_upper in key:
                        esoa_routes.update(FORM_TO_ROUTES[key])
                        break
            
            # If we have routes from the data, use them to constrain
            if cand_route:
                cand_route_upper = cand_route.upper().strip()
                if cand_route_upper:
                    cand_routes = cand_routes & {cand_route_upper} if cand_routes else {cand_route_upper}
            if esoa_route:
                esoa_route_upper = esoa_route.upper().strip()
                if esoa_route_upper:
                    esoa_routes = esoa_routes & {esoa_route_upper} if esoa_routes else {esoa_route_upper}
            
            # Forms are compatible if they share at least one valid route
            if cand_routes and esoa_routes:
                # Expand route equivalences
                expanded_cand = set()
                expanded_esoa = set()
                
                route_synonyms = {
                    "ORAL": {"ORAL", "PO", "BY MOUTH"},
                    "PARENTERAL": {"PARENTERAL", "INTRAVENOUS", "IV", "INTRAMUSCULAR", "IM", "SUBCUTANEOUS", "SC"},
                    "INTRAVENOUS": {"INTRAVENOUS", "IV", "PARENTERAL"},
                    "INTRAMUSCULAR": {"INTRAMUSCULAR", "IM", "PARENTERAL"},
                    "SUBCUTANEOUS": {"SUBCUTANEOUS", "SC", "PARENTERAL"},
                    "INHALATION": {"INHALATION", "RESPIRATORY", "INHALED", "NEBULIZATION"},
                    "TOPICAL": {"TOPICAL", "EXTERNAL", "CUTANEOUS"},
                    "OPHTHALMIC": {"OPHTHALMIC", "EYE", "OCULAR"},
                    "RECTAL": {"RECTAL", "PR"},
                }
                
                for r in cand_routes:
                    expanded_cand.add(r)
                    if r in route_synonyms:
                        expanded_cand.update(route_synonyms[r])
                for r in esoa_routes:
                    expanded_esoa.add(r)
                    if r in route_synonyms:
                        expanded_esoa.update(route_synonyms[r])
                
                return bool(expanded_cand & expanded_esoa)
            
            # If no route info, fall back to permissive matching for certain form pairs
            # These are forms that are clearly compatible regardless of route
            compatible_pairs = [
                # Injectable containers
                {"AMPULE", "AMPOULE", "VIAL", "INJECTION", "BOTTLE"},
                # Oral liquids
                {"SYRUP", "SUSPENSION", "SOLUTION", "ELIXIR", "LIQUID", "DROPS"},
                # Oral solids
                {"TABLET", "CAPSULE", "CAPLET"},
                # Inhalation
                {"NEBULE", "NEBULIZER", "INHALER", "AEROSOL", "MDI", "DPI"},
                # Topical
                {"CREAM", "OINTMENT", "GEL", "LOTION"},
                # Reconstitutable
                {"GRANULE", "POWDER", "SACHET"},
            ]
            
            for group in compatible_pairs:
                if cand_form_upper in group and esoa_form_upper in group:
                    return True
            
            return False
        
        def route_matches(cand_route, esoa_route):
            if not esoa_route or not cand_route:
                return True  # Missing route = compatible
            
            cand_upper = cand_route.upper().strip()
            esoa_upper = esoa_route.upper().strip()
            
            if cand_upper == esoa_upper:
                return True
            
            # Route equivalence groups
            route_groups = {
                "ORAL": {"ORAL", "PO", "BY MOUTH"},
                "PARENTERAL": {"PARENTERAL", "INTRAVENOUS", "IV", "INTRAMUSCULAR", "IM", "SUBCUTANEOUS", "SC", "SQ"},
                "INTRAVENOUS": {"INTRAVENOUS", "IV", "PARENTERAL"},
                "INTRAMUSCULAR": {"INTRAMUSCULAR", "IM", "PARENTERAL"},
                "SUBCUTANEOUS": {"SUBCUTANEOUS", "SC", "SQ", "PARENTERAL"},
                "INHALATION": {"INHALATION", "RESPIRATORY", "INHALED", "NEBULIZATION"},
                "TOPICAL": {"TOPICAL", "EXTERNAL", "CUTANEOUS"},
                "OPHTHALMIC": {"OPHTHALMIC", "EYE", "OCULAR"},
                "OTIC": {"OTIC", "EAR", "AURAL"},
                "NASAL": {"NASAL", "INTRANASAL"},
                "RECTAL": {"RECTAL", "PR"},
                "VAGINAL": {"VAGINAL", "PV"},
            }
            
            # Find groups containing each route
            cand_groups = set()
            esoa_groups = set()
            for base, synonyms in route_groups.items():
                if cand_upper in synonyms or cand_upper == base:
                    cand_groups.update(synonyms)
                    cand_groups.add(base)
                if esoa_upper in synonyms or esoa_upper == base:
                    esoa_groups.update(synonyms)
                    esoa_groups.add(base)
            
            return bool(cand_groups & esoa_groups) if cand_groups and esoa_groups else False
        
        # STRICT MATCHING: Require generic + dose + form/route compatibility
        # Dose matching is REQUIRED - do not match different doses (400mg ≠ 600mg)
        # Only allow unit conversions (500mcg = 0.5mg)
        # Use route-aware form matching (forms compatible if they share valid routes)
        
        # No dose key = no match (we cannot verify dose equivalence)
        if not esoa_dose_key:
            return None, "no_perfect_match"
        
        perfect_matches = [
            c for c in candidates 
            if doses_match(c.get("dose_key"), esoa_dose_key)
            and forms_compatible(c.get("form"), esoa_form, c.get("route"), esoa_route)
            and route_matches(c.get("route"), esoa_route)
        ]
        
        if perfect_matches:
            # If multiple perfect matches, use tie-breaking with *_details columns
            if len(perfect_matches) > 1:
                perfect_matches.sort(key=lambda c: rank_candidate_for_drug_code(c, row))
            return perfect_matches[0]["drug_code"], "matched_perfect"
        
        # No perfect match found
        return None, "no_perfect_match"
    
    if verbose:
        print("\nMatching ESOA to Drug Codes...")
    
    results = esoa_df.apply(match_to_drug_code, axis=1, result_type="expand")
    esoa_df["drug_code"] = results[0]
    esoa_df["drug_code_match_reason"] = results[1]
    
    # Write outputs
    PIPELINE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    run_with_spinner("Write outputs", lambda: write_csv_and_parquet(esoa_df, output_path))
    
    # Summary
    total = len(esoa_df)
    matched = esoa_df["drug_code"].notna().sum()
    
    reason_counts = {str(reason): int(count) for reason, count in esoa_df["drug_code_match_reason"].value_counts().items() if pd.notna(reason)}
    result_summary = {
        "total": total,
        "matched": matched,
        "matched_pct": 100 * matched / total if total else 0,
        "output_path": output_path,
        "reason_counts": reason_counts,
    }
    
    if verbose:
        print(f"\nPart 4 complete: {output_path}")
        print(f"  Total: {total:,}")
        print(f"  Matched: {matched:,} ({result_summary['matched_pct']:.1f}%)")
        print("\nMatch reasons:")
        for reason, count in reason_counts.items():
            pct = 100 * count / total if total else 0
            print(f"  {reason}: {count:,} ({pct:.1f}%)")
    
    # Log metrics
    log_metrics("esoa_to_drug_code", {
        "total": total,
        "matched": matched,
        "matched_pct": round(result_summary["matched_pct"], 2),
    })
    
    return result_summary


def load_fda_food_lookup(inputs_dir: Path = None) -> dict:
    """
    Load FDA food data for fallback matching.
    
    Returns dict mapping normalized product names to registration info.
    """
    import glob
    
    if inputs_dir is None:
        inputs_dir = PIPELINE_INPUTS_DIR
    
    # Find latest FDA food file
    food_files = sorted(glob.glob(str(inputs_dir / "fda_food_*.csv")))
    if not food_files:
        food_files = sorted(glob.glob(str(inputs_dir / "fda_food_*.parquet")))
    
    if not food_files:
        return {}
    
    food_path = Path(food_files[-1])
    
    if str(food_path).endswith('.parquet'):
        food_df = pd.read_parquet(food_path)
    else:
        food_df = pd.read_csv(food_path)
    
    # Build lookup by brand_name and product_name
    lookup = {}
    for _, row in food_df.iterrows():
        brand = str(row.get("brand_name", "")).upper().strip()
        product = str(row.get("product_name", "")).upper().strip()
        reg_num = row.get("registration_number", "")
        
        if brand and brand != "-":
            lookup[brand] = {"type": "fda_food_brand", "registration": reg_num}
        if product and product != "-":
            lookup[product] = {"type": "fda_food_product", "registration": reg_num}
    
    return lookup


def check_fda_food_fallback(
    text: str,
    food_lookup: dict,
) -> tuple:
    """
    Check if text matches FDA food database.
    
    Returns (match_type, registration_number) or (None, None).
    """
    if not text or not food_lookup:
        return None, None
    
    text_upper = text.upper().strip()
    
    # Direct match
    if text_upper in food_lookup:
        info = food_lookup[text_upper]
        return info["type"], info.get("registration", "")
    
    # Token-based match (check if any token matches)
    tokens = text_upper.split()
    for token in tokens:
        if len(token) >= 4 and token in food_lookup:
            info = food_lookup[token]
            return f"{info['type']}_partial", info.get("registration", "")
    
    return None, None


def log_metrics(
    run_type: str,
    metrics: dict,
    metrics_path: Optional[Path] = None,
) -> None:
    """
    Log pipeline run metrics to history file.
    
    Args:
        run_type: Type of run (annex_f, esoa, esoa_to_drug_code)
        metrics: Dict with metric values
        metrics_path: Path to metrics history file
    """
    from datetime import datetime
    
    if metrics_path is None:
        metrics_path = PIPELINE_OUTPUTS_DIR / "metrics_history.csv"
    
    # Build row
    row = {
        "timestamp": datetime.now().isoformat(),
        "run_type": run_type,
        **metrics,
    }
    
    # Append to CSV
    file_exists = metrics_path.exists()
    
    metrics_df = pd.DataFrame([row])
    if file_exists:
        metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
    else:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)


def get_metrics_summary(metrics_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Get metrics history summary.
    
    Returns DataFrame with all historical metrics.
    """
    if metrics_path is None:
        metrics_path = PIPELINE_OUTPUTS_DIR / "metrics_history.csv"
    
    if not metrics_path.exists():
        return pd.DataFrame()
    
    return pd.read_csv(metrics_path)


def print_metrics_comparison(verbose: bool = True) -> None:
    """Print comparison of latest metrics vs previous runs."""
    df = get_metrics_summary()
    
    if df.empty:
        if verbose:
            print("No metrics history found.")
        return
    
    if verbose:
        print("\n" + "=" * 60)
        print("METRICS HISTORY")
        print("=" * 60)
        
        # Group by run_type and show latest
        for run_type in df["run_type"].unique():
            subset = df[df["run_type"] == run_type].tail(5)
            print(f"\n{run_type.upper()}:")
            print(subset.to_string(index=False))
