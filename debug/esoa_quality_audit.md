# ESOA Tagging Quality Audit

## Status: FIXED (2025-01-06)

### Results After Fixes
- Testing improvements with sample combinations and salt patterns

---

## Fixes Implemented

### Fix 1: Trailing salt suffix stripping (tokenizer.py)
**Problem**: ESOA uses "DEXAMETHASONE SODIUM PHOSPHATE" instead of "( as SODIUM PHOSPHATE)"

**Solution**: Added trailing salt suffix stripping in `extract_drug_details`:
- Strips: SODIUM PHOSPHATE, SODIUM SUCCINATE, SODIUM SULFATE, etc.
- Captures stripped salt in `salt_details`
- Only strips if there's a meaningful base name remaining

**Also in `extract_generic_tokens` and `normalize_tokens`**: Added `is_trailing_salt_suffix()` check to exclude trailing salts from multiword generic matching.

**Verified**:
- DEXAMETHASONE SODIUM PHOSPHATE → DEXAMETHASONE (salt: SODIUM PHOSPHATE) ✓
- PREDNISOLONE SODIUM PHOSPHATE → PREDNISOLONE (salt: SODIUM PHOSPHATE) ✓
- HYDROCORTISONE SODIUM SUCCINATE → HYDROCORTISONE (salt: SODIUM SUCCINATE) ✓

---

### Fix 2: Plus sign normalization (tokenizer.py)
**Problem**: "IBUPROFEN+PARACETAMOL" wasn't splitting (no spaces around +)

**Solution**: Added normalization in `extract_drug_details`:
```python
if "+" in working and " + " not in working:
    working = re.sub(r"\+", " + ", working)
```

---

### Fix 3: Combination mixture lookup (tagger.py)
**Problem**: `_lookup_mixture` was generating uppercase keys but mixtures table has lowercase

**Solution**: 
- Fixed case: `normalized = [self._apply_synonyms(g.upper()).lower() for g in generics]`
- Added deduplication to remove substrings (e.g., "ascorbic" when "ascorbic acid" present)

**Verified**:
- IBUPROFEN+PARACETAMOL → ACETAMINOPHEN + IBUPROFEN ✓
- ASCORBIC ACID+ZINC → ASCORBIC ACID + ZINC ✓
- LIDOCAINE + EPINEPHRINE → EPINEPHRINE + LIDOCAINE ✓
- TRAMADOL+PARACETAMOL → TRAMADOL AND PARACETAMOL ✓

---

### Fix 4: Standalone element handling in combo extraction (tokenizer.py)
**Problem**: ZINC was filtered as salt token in "ASCORBIC ACID+ZINC"

**Solution**: Modified combo part extraction to keep salt tokens if they're standalone:
```python
if len(all_words) == 1:
    words.append(word)  # Keep ZINC if it's the only word
```

---

## Files Modified
- `pipelines/drugs/scripts/tokenizer.py` - Salt suffix stripping, plus normalization, combo extraction
- `pipelines/drugs/scripts/tagger.py` - Mixture lookup case fix, deduplication

---

## Remaining Issues
- Multi-component combinations (3+ drugs) may still not match if not in reference
- Brand names in combinations (e.g., "TDL PLUS") may cause extra tokens
