"""
Unified drug tagging module.

This module provides functions for tagging drug descriptions with ATC codes
and DrugBank IDs using a unified algorithm for both Annex F and ESOA.

Submodules:
- constants: Token categories, scoring weights, normalization mappings
- tokenizer: Text tokenization and normalization
- lookup: Reference data lookup functions
- scoring: Candidate scoring and selection
- tagger: Main tagging interface

Usage:
    from pipelines.drugs.scripts.tagging import UnifiedTagger
    
    tagger = UnifiedTagger()
    tagger.load()
    results = tagger.tag_descriptions(df, text_column="Drug Description")
    tagger.close()
"""

from .tagger import UnifiedTagger, tag_descriptions, tag_single

__all__ = ["UnifiedTagger", "tag_descriptions", "tag_single"]
