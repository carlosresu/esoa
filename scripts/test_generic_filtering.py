#!/usr/bin/env python3
"""
Quick test to verify generic token filtering fixes work correctly.
Tests problematic descriptions from the verification output.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

from drugs.scripts.tagger import UnifiedTagger

def test_problematic_descriptions():
    """Test descriptions that were producing pipe-delimited generics."""
    
    # Load tagger
    tagger = UnifiedTagger(verbose=True)
    tagger.load()
    
    # Test cases from verification output
    test_cases = [
        "KETOANALOGUE+AMINO ACIDS 600MG TAB (GENERIC)",
        "CLARITHROMYCIN 500 MG TABLET **", 
        "CELECOXIB 200MG TAB (OP)",
        "BUTAMIRATE CITRATE (SINECOD FORTE) 50MG TABLET",
        "MUPIROCIN OINTMENT 2%, 52%, 5 G, GRAM",
        "CLARITHROMYCIN, CLARITHROMED, 125MG/ML, FOR SUSPENSION",
        "EUROMED (POTASSIUM CHLORIDE)  2MEQ/ML 20ML/VIAL",
    ]
    
    print("=== TESTING GENERIC TOKEN FILTERING ===")
    
    for i, description in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        
        # Tag the description
        results = tagger.tag_descriptions([description], text_column="description")
        
        if results:
            result = results[0]
            generic = result.get('generic_name', 'None')
            match_reason = result.get('match_reason', 'None')
            
            print(f"  Extracted generic: {generic}")
            print(f"  Match reason: {match_reason}")
            
            # Check for problematic patterns
            if '|' in generic:
                print(f"  ❌ STILL HAS PIPE: {generic}")
            else:
                print(f"  ✅ Clean generic: {generic}")
        else:
            print("  ❌ No results returned")
    
    tagger.close()

if __name__ == "__main__":
    test_problematic_descriptions()
