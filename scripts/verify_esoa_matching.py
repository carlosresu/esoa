#!/usr/bin/env python3
"""
Script to verify ESOA to Annex F drug code matching failures.
Analyzes systematic patterns in mismatched rows to identify gaps in the matching algorithm.
"""

import pandas as pd
import numpy as np
from collections import Counter
import sys
import os

# Add pipeline path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

def load_outputs():
    """Load the pipeline output files."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'drugs')
    
    esoa_drug_code = pd.read_csv(os.path.join(base_path, 'esoa_with_drug_code.csv'))
    esoa_atc = pd.read_csv(os.path.join(base_path, 'esoa_with_atc.csv'))
    annex_f = pd.read_csv(os.path.join(base_path, 'annex_f_with_atc.csv'))
    
    return esoa_drug_code, esoa_atc, annex_f

def analyze_generic_not_in_annex(df, sample_size=100):
    """Analyze the largest failure category: generic_not_in_annex."""
    failures = df[df['drug_code_match_reason'] == 'generic_not_in_annex'].copy()
    
    print(f"\n=== GENERIC_NOT_IN_ANNEX ANALYSIS ===")
    print(f"Total failures: {len(failures):,}")
    
    # Extract generic names and find most common
    generic_counts = Counter(failures['generic_final'].fillna(''))
    print(f"\nTop 20 most frequent unmatched generics:")
    for generic, count in generic_counts.most_common(20):
        if generic and generic.strip():
            print(f"  {generic}: {count}")
    
    # Sample random failures for manual inspection
    sample = failures.sample(n=min(sample_size, len(failures)), random_state=42)
    print(f"\n=== SAMPLE OF {len(sample)} UNMATCHED ROWS ===")
    for idx, row in sample.iterrows():
        print(f"Description: {row['DESCRIPTION']}")
        print(f"Extracted generic: {row['generic_final']}")
        print(f"Match reason: {row['drug_code_match_reason']}")
        print("-" * 80)
    
    return failures, generic_counts

def analyze_no_candidates(df, sample_size=50):
    """Analyze no_candidates failures from ATC matching."""
    failures = df[df['match_reason'] == 'no_candidates'].copy()
    
    print(f"\n=== NO_CANDIDATES ANALYSIS ===")
    print(f"Total failures: {len(failures):,}")
    
    # Extract generic names
    generic_counts = Counter(failures['generic_final'].fillna(''))
    print(f"\nTop 15 most frequent no_candidate generics:")
    for generic, count in generic_counts.most_common(15):
        if generic and generic.strip():
            print(f"  {generic}: {count}")
    
    # Sample for inspection
    sample = failures.sample(n=min(sample_size, len(failures)), random_state=42)
    print(f"\n=== SAMPLE OF {len(sample)} NO_CANDIDATE ROWS ===")
    for idx, row in sample.iterrows():
        print(f"Description: {row['DESCRIPTION']}")
        print(f"Extracted generic: {row['generic_final']}")
        print("-" * 80)
    
    return failures, generic_counts

def analyze_annex_f_failures(annex_df, sample_size=30):
    """Analyze Annex F matching failures."""
    failures = annex_df[annex_df['match_reason'] == 'no_candidates'].copy()
    
    print(f"\n=== ANNEX F NO_CANDIDATES ANALYSIS ===")
    print(f"Total failures: {len(failures):,}")
    
    # Extract generic names
    generic_counts = Counter(failures['matched_generic_name'].fillna(''))
    print(f"\nTop 15 most frequent Annex F unmatched generics:")
    for generic, count in generic_counts.most_common(15):
        if generic and generic.strip():
            print(f"  {generic}: {count}")
    
    # Sample for inspection
    sample = failures.sample(n=min(sample_size, len(failures)), random_state=42)
    print(f"\n=== SAMPLE OF {len(sample)} ANNEX F UNMATCHED ROWS ===")
    for idx, row in sample.iterrows():
        print(f"Drug Description: {row['Drug Description']}")
        print(f"Extracted generic: {row['matched_generic_name']}")
        print("-" * 80)
    
    return failures, generic_counts

def check_overlap_between_datasets(esoa_failures, annex_failures):
    """Check for overlapping drug names between ESOA and Annex F failures."""
    print(f"\n=== OVERLAP ANALYSIS ===")
    
    # Filter out empty strings
    esoa_generics = set([g for g in esoa_failures['generic_final'].fillna('') if g.strip()])
    annex_generics = set([g for g in annex_failures['matched_generic_name'].fillna('') if g.strip()])
    
    overlap = esoa_generics.intersection(annex_generics)
    print(f"ESOA unique generics failing: {len(esoa_generics):,}")
    print(f"Annex F unique generics failing: {len(annex_generics):,}")
    print(f"Overlapping generics: {len(overlap):,}")
    
    if overlap:
        print(f"\nTop 20 overlapping generics:")
        overlap_counter = Counter()
        for generic in overlap:
            esoa_count = esoa_failures[esoa_failures['generic_final'] == generic].shape[0]
            annex_count = annex_failures[annex_failures['matched_generic_name'] == generic].shape[0]
            overlap_counter[generic] = esoa_count + annex_count
        
        for generic, count in overlap_counter.most_common(20):
            print(f"  {generic}: {count}")
    
    return overlap

def analyze_success_patterns(df):
    """Analyze what makes matching successful."""
    successes = df[df['drug_code_match_reason'] != 'generic_not_in_annex'].copy()
    
    print(f"\n=== SUCCESS PATTERN ANALYSIS ===")
    print(f"Total successes: {len(successes):,}")
    
    # Analyze match reasons
    reason_counts = Counter(successes['drug_code_match_reason'])
    print(f"\nSuccess breakdown:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count} ({count/len(successes)*100:.1f}%)")
    
    # Sample some successful matches
    print(f"\n=== SAMPLE OF SUCCESSFUL MATCHES ===")
    sample = successes.sample(n=min(20, len(successes)), random_state=42)
    for idx, row in sample.iterrows():
        print(f"Description: {row['DESCRIPTION']}")
        print(f"Generic: {row['generic_final']}")
        print(f"Drug Code: {row['drug_code']}")
        print(f"Match reason: {row['drug_code_match_reason']}")
        print("-" * 80)

def main():
    """Main analysis function."""
    print("=== ESOA TO ANNEX F MATCHING VERIFICATION ===")
    
    # Load data
    esoa_drug_code, esoa_atc, annex_f = load_outputs()
    
    print(f"ESOA drug code file: {len(esoa_drug_code):,} rows")
    print(f"ESOA ATC file: {len(esoa_atc):,} rows")
    print(f"Annex F file: {len(annex_f):,} rows")
    
    # Analyze major failure categories
    esoa_generic_failures, esoa_generic_counts = analyze_generic_not_in_annex(esoa_drug_code)
    esoa_no_candidates, esoa_no_candidate_counts = analyze_no_candidates(esoa_atc)
    annex_failures, annex_counts = analyze_annex_f_failures(annex_f)
    
    # Check overlaps
    overlap = check_overlap_between_datasets(esoa_generic_failures, annex_failures)
    
    # Analyze successes
    analyze_success_patterns(esoa_drug_code)
    
    print(f"\n=== RECOMMENDATIONS ===")
    print("1. Focus on top 20 unmatched generics - they represent the bulk of failures")
    print("2. Check if these generics exist in reference datasets with different naming")
    print("3. Verify normalization logic for salt forms and combination drugs")
    print("4. Consider expanding reference datasets for commonly missing drugs")
    print("5. Review brand name handling - many failures appear to be brands")

if __name__ == "__main__":
    main()
