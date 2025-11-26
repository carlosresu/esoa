"""Match Annex F entries to ATC codes using PNF and DrugBank generics/mixtures."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, TypeVar

import pandas as pd

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


BASE_DIR = Path(__file__).resolve().parent
DRUGS_DIR = BASE_DIR / "inputs" / "drugs"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DRUGS_DIR = OUTPUTS_DIR / "drugs"
PARALLEL_CONFIG_PATH = BASE_DIR / "parallel_config.txt"
_REFERENCE_ROWS_SHARED: list[dict] | None = None
_REFERENCE_INDEX_SHARED: dict[str, list[int]] | None = None
_BRAND_PATTERNS_SHARED: list[tuple[re.Pattern, str]] | None = None
_GENERIC_PHRASES_SHARED: list[str] | None = None
_GENERIC_AUTOMATON_SHARED = None
_MIXTURE_LOOKUP_SHARED: dict[str, list[dict]] | None = None
_DRUGBANK_BY_ID_SHARED: dict[str, list[dict]] | None = None
_GENERIC_TO_DRUGBANK_SHARED: dict[str, str] | None = None


T = TypeVar("T")


def _run_with_spinner(label: str, func: Callable[[], T]) -> T:
    """Run func() in a worker thread while showing a simple CLI spinner."""
    import threading

    done = threading.Event()
    result: list[T] = []
    err: list[BaseException] = []

    def worker():
        try:
            result.append(func())
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)
        finally:
            done.set()

    start = time.perf_counter()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    frames = "|/-\\"
    idx = 0
    while not done.wait(0.1):
        elapsed = time.perf_counter() - start
        sys.stdout.write(f"\r{frames[idx % len(frames)]} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        idx += 1
    thread.join()
    elapsed = time.perf_counter() - start
    status = "done" if not err else "error"
    sys.stdout.write(f"\r[{status}] {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return result[0] if result else None  # type: ignore[return-value]


def _flag_passed(name: str) -> bool:
    flag = f"--{name}"
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def _load_parallel_config() -> dict[str, str]:
    if not PARALLEL_CONFIG_PATH.exists():
        return {}
    try:
        cfg: dict[str, str] = {}
        for line in PARALLEL_CONFIG_PATH.read_text().splitlines():
            if "=" in line:
                key, val = line.split("=", 1)
                cfg[key.strip()] = val.strip()
        return cfg
    except Exception:
        return {}


def _save_parallel_config(config: dict[str, str]) -> None:
    try:
        keys = ("backend", "workers")
        lines = [f"{k}={config[k]}" for k in keys if k in config]
        PARALLEL_CONFIG_PATH.write_text("\n".join(lines))
    except Exception:
        pass


def _effective_workers(requested: int | None) -> int:
    """Pick a sensible worker count: default to (cores - 1) but at least 1."""
    if requested is None or requested <= 0:
        cores = os.cpu_count() or 1
        return max(1, cores - 1)
    return max(1, requested)


def _strip_commas(token: str) -> str:
    return token.rstrip(",").strip()


def split_with_parentheses(text: str) -> List[str]:
    """Split on spaces while keeping parentheses content together; drop commas and parens."""
    if text is None:
        return []
    chars = str(text)
    tokens: List[str] = []
    current: List[str] = []
    depth = 0

    for ch in chars:
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            if depth:
                depth -= 1
            continue
        if ch.isspace() and depth == 0:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(ch)

    if current:
        tokens.append("".join(current))

    cleaned = [_strip_commas(tok) for tok in tokens]
    return [tok.upper() for tok in cleaned if tok]


def format_number_token(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text.upper() if text else None
    try:
        num = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text.upper() if text else None
    if math.isnan(num):
        return None
    if num.is_integer():
        return str(int(num))
    text = f"{num:.15g}"
    return text.rstrip("0").rstrip(".")


def tokens_from_field(value) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).replace("_", " ")
    return split_with_parentheses(text)


def _read_table(base_path: Path, required: bool = True) -> pd.DataFrame:
    parquet_path = base_path.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if base_path.exists():
        df = pd.read_csv(base_path)
        try:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)
        except Exception:
            pass
        return df
    if required:
        raise FileNotFoundError(base_path)
    return pd.DataFrame()


def _write_csv_and_parquet(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(csv_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def parse_pipe_tokens(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    return [tok for tok in str(value).split("|") if tok]


def _maybe_none(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _as_str_or_empty(value) -> str:
    val = _maybe_none(value)
    if val is None:
        return ""
    return str(val)


def _memory_snapshot() -> dict | None:
    if psutil is None:
        return None
    try:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        return {
            "available": int(vm.available),
            "total": int(vm.total),
            "swap_used": int(sm.used),
        }
    except Exception:
        return None


def _swap_thrash(before: dict | None, after: dict | None) -> bool:
    if not before or not after:
        return False
    avail_drop = before.get("available", 0) - after.get("available", 0)
    total = max(1, after.get("total", 1))
    swap_jump = after.get("swap_used", 0) - before.get("swap_used", 0)
    return (avail_drop > 0.2 * total) or (swap_jump > 256 * 1024 * 1024)


def _autotune_parallelism(
    annex_records: list[dict],
    reference_rows: list[dict],
    reference_index: dict[str, list[int]],
    brand_patterns: list[tuple[re.Pattern, str]],
    generic_automaton,
    generic_phrases: list[str],
    generic_atc_map: dict[str, list[str]],
    mixture_lookup: dict[str, list[dict]],
    drugbank_refs_by_id: dict[str, list[dict]],
) -> tuple[int, int, str]:
    total = len(annex_records)
    if not total:
        return (8, 300, "process")
    # Use the full Annex F payload to benchmark realistic performance.
    sample = annex_records
    cpu = os.cpu_count() or 4
    max_workers = max(4, min(total, cpu * 2))
    workers = 4
    best: tuple[float, int, int] | None = None
    baseline_mem = _memory_snapshot()
    while workers <= max_workers:
        chunk = max(1, math.ceil(total / workers))
        before = _memory_snapshot()
        start = time.perf_counter()
        try:
            match_annex_with_atc(
                sample,
                reference_rows,
                reference_index,
                brand_patterns,
                generic_automaton,
                generic_phrases,
                generic_atc_map,
                mixture_lookup,
                drugbank_refs_by_id,
                max_workers=workers,
                chunk_size=chunk,
                use_threads=False,
            )
            duration = time.perf_counter() - start
            after = _memory_snapshot()
            thrash = _swap_thrash(before, after) or _swap_thrash(baseline_mem, after)
            if thrash:
                break
            if best is None or duration < best[0]:
                best = (duration, workers, chunk)
            elif best and duration > best[0] * 1.15:
                # Overhead is outweighing gains; stop exploring higher worker counts.
                break
        except Exception:
            pass
        workers += 2
    if best is None:
        return (8, 300, "process")
    best_time, best_workers, best_chunk = best
    backend = "process"
    print(
        f"[autotune] chose workers={best_workers}, chunk_size={best_chunk}, "
        f"time={best_time:.2f}s, psutil={'on' if psutil else 'off'}"
    )
    return best_workers, best_chunk, backend


def _ordered_overlap(
    fuzzy_tokens: List[str], fuzzy_counts: Counter, candidate_tokens: Iterable[str]
) -> List[str]:
    candidate_counts = Counter(candidate_tokens)
    overlap_counts = fuzzy_counts & candidate_counts
    overlap: List[str] = []
    for tok in fuzzy_tokens:
        if overlap_counts.get(tok, 0) > 0:
            overlap.append(tok)
            overlap_counts[tok] -= 1
    return overlap


_NUMERIC_RX = re.compile(r"^-?\d+(?:\.\d+)?$")
_COMBINED_WEIGHT_RX = re.compile(r"^(-?[\d,.]+)\s*(MG|G|MCG|UG|KG)$", re.IGNORECASE)
_WEIGHT_UNIT_FACTORS = {
    "MG": 1.0,
    "G": 1000.0,
    "MCG": 0.001,
    "UG": 0.001,
    "KG": 1_000_000.0,
}
_UNIT_TOKENS = {"MG", "G", "MCG", "UG", "KG", "ML", "L"}

NATURAL_STOPWORDS = {
    "AS",
    "IN",
    "FOR",
    "TO",
    "WITH",
    "EQUIV",
    "EQUIV.",
    "AND",
    "OF",
    "OR",
    "NOT",
    "THAN",
    "HAS",
    "DURING",
    "THIS",
    "W/",
    "W",
    "PLUS",
    "APPROX",
    "APPROXIMATELY",
    "PRE",
    "FILLED",
    "PRE-FILLED",
}

GENERIC_JUNK_TOKENS = {
    "SOLUTION",
    "SOLUTIONS",
    "SOLN",
    "IRRIGATION",
    "IRRIGATING",
    "INJECTION",
    "INJECTIONS",
    "INJECTABLE",
    "INFUSION",
    "INFUSIONS",
    "DILUENT",
    "DILUTION",
    "POWDER",
    "POWDERS",
    "MICRONUTRIENT",
    "FORMULA",
    "FORMULATION",
    "WATER",
    "VEHICLE",
}

FORM_CANON = {
    "TAB": "TABLET",
    "TABS": "TABLET",
    "TABLET": "TABLET",
    "TABLETS": "TABLET",
    "CAP": "CAPSULE",
    "CAPS": "CAPSULE",
    "CAPSULE": "CAPSULE",
    "CAPSULES": "CAPSULE",
    "BOT": "BOTTLE",
    "BOTT": "BOTTLE",
    "BOTTLE": "BOTTLE",
    "BOTTLES": "BOTTLE",
    "VIAL": "VIAL",
    "VIALS": "VIAL",
    "INJ": "INJECTION",
    "INJECTABLE": "INJECTION",
    "SYR": "SYRUP",
    "SYRUP": "SYRUP",
}

ROUTE_CANON = {
    "PO": "ORAL",
    "OR": "ORAL",
    "ORAL": "ORAL",
    "IV": "INTRAVENOUS",
    "INTRAVENOUS": "INTRAVENOUS",
    "IM": "INTRAMUSCULAR",
    "INTRAMUSCULAR": "INTRAMUSCULAR",
    "SC": "SUBCUTANEOUS",
    "SUBCUTANEOUS": "SUBCUTANEOUS",
    "SUBCUT": "SUBCUTANEOUS",
    "NASAL": "NASAL",
    "TOPICAL": "TOPICAL",
    "RECTAL": "RECTAL",
    "OPHTHALMIC": "OPHTHALMIC",
    "BUCCAL": "BUCCAL",
}

SALT_TOKENS = {
    "CALCIUM",
    "SODIUM",
    "POTASSIUM",
    "MAGNESIUM",
    "ZINC",
    "AMMONIUM",
    "MEGLUMINE",
    "ALUMINUM",
    "HYDROCHLORIDE",
    "NITRATE",
    "NITRITE",
    "SULFATE",
    "SULPHATE",
    "PHOSPHATE",
    "DIHYDROGEN PHOSPHATE",
    "HYDROXIDE",
    "DIPROPIONATE",
    "ACETATE",
    "TARTRATE",
    "FUMARATE",
    "OXALATE",
    "MALEATE",
    "MESYLATE",
    "TOSYLATE",
    "BESYLATE",
    "BESILATE",
    "BITARTRATE",
    "SUCCINATE",
    "CITRATE",
    "LACTATE",
    "GLUCONATE",
    "BICARBONATE",
    "CARBONATE",
    "BROMIDE",
    "CHLORIDE",
    "IODIDE",
    "SELENITE",
    "THIOSULFATE",
    "DIHYDRATE",
    "TRIHYDRATE",
    "MONOHYDRATE",
    "HYDRATE",
    "HEMIHYDRATE",
    "ANHYDROUS",
    "DECANOATE",
    "PALMITATE",
    "STEARATE",
    "PAMOATE",
    "BENZOATE",
    "VALERATE",
    "PROPIONATE",
    "HYDROBROMIDE",
    "DOCUSATE",
    "HEMISUCCINATE",
}

CATEGORY_GENERIC = "generic"
CATEGORY_SALT = "salt"
CATEGORY_DOSE = "dose"
CATEGORY_FORM = "form"
CATEGORY_ROUTE = "route"
CATEGORY_OTHER = "other"

PRIMARY_WEIGHTS = {
    CATEGORY_GENERIC: 5,
    CATEGORY_SALT: 4,
    CATEGORY_DOSE: 4,
    CATEGORY_FORM: 3,
    CATEGORY_ROUTE: 3,
    CATEGORY_OTHER: 1,
}
SECONDARY_WEIGHTS = {
    CATEGORY_GENERIC: 3,
    CATEGORY_SALT: 3,
    CATEGORY_DOSE: 3,
    CATEGORY_FORM: 4,
    CATEGORY_ROUTE: 4,
    CATEGORY_OTHER: 1,
}

GENERIC_MISS_PENALTY_PRIMARY = 6
GENERIC_MISS_PENALTY_SECONDARY = 4
GENERIC_MATCH_REQUIRED = True
GENERIC_REF_MISMATCH_TOLERANCE_PRIMARY = 1
GENERIC_REF_MISMATCH_TOLERANCE_SECONDARY = 1
GENERIC_REF_EXTRA_PENALTY_PRIMARY = 4
GENERIC_REF_EXTRA_PENALTY_SECONDARY = 3

# Dose matching: penalize when reference has a dose that doesn't match Annex F
# This prevents matching AMLODIPINE 2.5mg to AMLODIPINE 5mg
# Set high enough to make mismatched doses score lower than no match
DOSE_MISMATCH_PENALTY = 20
# Require exact dose match when Annex F has a specific numeric dose
REQUIRE_DOSE_MATCH = True

# Common spelling/lexical variants we want to normalize across Annex F and references.
GENERIC_SYNONYMS = {
    "BECLOMETASONE": "BECLOMETHASONE",
    "AMIDOTRIZOATE": "DIATRIZOATE",
    "DIATRIZOIC": "DIATRIZOATE",
    "DIATRIZOIC ACID": "DIATRIZOATE",
    "DIATRIZOIC ACID DIHYDRATE": "DIATRIZOATE",
}

# Compound names that should be treated as generics even though they contain salt tokens.
# These are complete drug names, not salt modifiers.
COMPOUND_GENERICS = {
    "SODIUM CHLORIDE",
    "POTASSIUM CHLORIDE",
    "CALCIUM CHLORIDE",
    "MAGNESIUM SULFATE",
    "MAGNESIUM SULPHATE",
    "SODIUM BICARBONATE",
    "POTASSIUM BICARBONATE",
    "CALCIUM CARBONATE",
    "CALCIUM GLUCONATE",
    "POTASSIUM PHOSPHATE",
    "SODIUM PHOSPHATE",
    "ALUMINUM HYDROXIDE",
    "MAGNESIUM HYDROXIDE",
    "FERROUS SULFATE",
    "FERROUS SULPHATE",
    "ZINC SULFATE",
    "ZINC SULPHATE",
    "SODIUM LACTATE",
    "CALCIUM LACTATE",
    "SODIUM CITRATE",
    "POTASSIUM CITRATE",
    "SODIUM ACETATE",
    "POTASSIUM ACETATE",
}

# ATC codes that indicate combination products (contain multiple active ingredients).
# When resolving ties, prefer single-agent ATCs over these.
COMBINATION_ATC_PATTERNS = {
    "A10BD",  # Blood glucose lowering drugs, combinations
    "C09BA",  # ACE inhibitors and diuretics
    "C09BB",  # ACE inhibitors and calcium channel blockers
    "C09BX",  # ACE inhibitors, other combinations
    "C09DA",  # Angiotensin II receptor blockers and diuretics
    "C09DB",  # Angiotensin II receptor blockers and calcium channel blockers
    "C09DX",  # Angiotensin II receptor blockers, other combinations
    "C10BA",  # HMG CoA reductase inhibitors in combination with other lipid modifying agents
    "C10BX",  # HMG CoA reductase inhibitors, other combinations
    "M05BB",  # Bisphosphonates, combinations
    "R03AL",  # Adrenergics in combination with anticholinergics
    "R03AK",  # Adrenergics in combination with corticosteroids
    "R03DA20",  # Combinations of xanthines
    "R03DA55",  # Aminophylline, combinations
    "R03DB",  # Xanthines and adrenergics
}

# ATC codes ending in these patterns typically indicate combinations
# (codes ending in 20, 30, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 are often combinations)
COMBINATION_ATC_SUFFIXES = {"20", "30", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59"}


def _is_numeric_token(token: str) -> bool:
    if not isinstance(token, str):
        return False
    return bool(_NUMERIC_RX.match(token.strip()))


def _format_number_text(value) -> str:
    formatted = format_number_token(value)
    return formatted if formatted is not None else ""


def _convert_to_mg(number_text: str, unit: str) -> str | None:
    try:
        num = float(str(number_text).replace(",", ""))
    except Exception:
        return None
    factor = _WEIGHT_UNIT_FACTORS.get(unit.upper())
    if factor is None:
        return None
    mg_val = num * factor
    return _format_number_text(mg_val)


def _normalize_tokens(tokens: List[str], drop_stopwords: bool = False) -> List[str]:
    expanded: List[str] = []
    for tok in tokens:
        if tok is None:
            continue
        expanded.extend(str(tok).split())

    normalized: List[str] = []
    i = 0
    while i < len(expanded):
        raw_tok = expanded[i]
        tok_clean = raw_tok.replace(",", "").strip("()").strip()
        tok_upper = tok_clean.upper()

        if tok_upper.endswith("%") and _is_numeric_token(tok_upper.rstrip("%")):
            pct_val = _format_number_text(tok_upper.rstrip("%"))
            if pct_val:
                normalized.append(pct_val)
                normalized.append("PCT")
            i += 1
            continue

        combined_match = _COMBINED_WEIGHT_RX.match(tok_upper)
        if combined_match:
            orig_val = _format_number_text(combined_match.group(1))
            orig_unit = combined_match.group(2).upper()
            mg_val = _convert_to_mg(combined_match.group(1), orig_unit)
            if mg_val and orig_val:
                # Keep both original and normalized values for better matching
                normalized.append(orig_val)
                normalized.append(orig_unit)
                if mg_val != orig_val or orig_unit != "MG":
                    normalized.append(mg_val)
                    normalized.append("MG")
            i += 1
            continue

        if _is_numeric_token(tok_upper) and i + 1 < len(expanded):
            next_clean = expanded[i + 1].replace(",", "").strip("()").strip().upper()
            if next_clean in _WEIGHT_UNIT_FACTORS:
                orig_val = _format_number_text(tok_upper)
                mg_val = _convert_to_mg(tok_upper, next_clean)
                if mg_val and orig_val:
                    # Keep both original and normalized values for better matching
                    normalized.append(orig_val)
                    normalized.append(next_clean)
                    if mg_val != orig_val or next_clean != "MG":
                        normalized.append(mg_val)
                        normalized.append("MG")
                    i += 2
                    continue

        if _is_numeric_token(tok_upper):
            tok_upper = _format_number_text(tok_upper)

        tok_upper = FORM_CANON.get(tok_upper, tok_upper)
        tok_upper = ROUTE_CANON.get(tok_upper, tok_upper)
        tok_upper = GENERIC_SYNONYMS.get(tok_upper, tok_upper)

        # Always drop purely natural-language stopwords.
        if tok_upper == "PER":
            prev = normalized[-1] if normalized else None
            next_tok = (
                expanded[i + 1].replace(",", "").strip("()").strip().upper()
                if i + 1 < len(expanded)
                else None
            )
            if not (
                (prev and (_is_numeric_token(prev) or prev in _UNIT_TOKENS))
                or (next_tok and (_is_numeric_token(next_tok) or next_tok in _UNIT_TOKENS))
            ):
                i += 1
                continue
        if tok_upper in NATURAL_STOPWORDS or not tok_upper:
            i += 1
            continue

        normalized.append(tok_upper)
        i += 1
    return normalized


def _classify_token(token: str) -> str:
    tok = token.upper()
    if tok in FORM_CANON.values():
        return CATEGORY_FORM
    if tok in ROUTE_CANON.values():
        return CATEGORY_ROUTE
    if tok in SALT_TOKENS:
        return CATEGORY_SALT
    if tok in {"MG", "G", "MCG", "UG", "KG", "ML", "L", "PCT"} or _is_numeric_token(tok):
        return CATEGORY_DOSE
    return CATEGORY_GENERIC


def _is_combination_atc(atc_code: str | None) -> bool:
    """Check if an ATC code indicates a combination product."""
    if not atc_code:
        return False
    atc_upper = atc_code.upper().strip()
    # Check explicit combination patterns
    if any(atc_upper.startswith(pat) for pat in COMBINATION_ATC_PATTERNS):
        return True
    # Check suffix-based combinations (e.g., codes ending in 20, 30, 50-59)
    if len(atc_upper) >= 7:  # Full ATC code like A10BF01
        suffix = atc_upper[-2:]
        if suffix in COMBINATION_ATC_SUFFIXES:
            return True
    return False


def _detect_compound_generics(tokens: List[str]) -> set[str]:
    """Detect tokens that are part of compound generic names like SODIUM CHLORIDE."""
    compound_tokens: set[str] = set()
    token_str = " ".join(tokens)
    for compound in COMPOUND_GENERICS:
        if compound in token_str:
            for part in compound.split():
                compound_tokens.add(part)
    return compound_tokens


def _extract_numeric_doses(tokens: List[str]) -> set[str]:
    """Extract numeric dose values from tokens (values near dose units)."""
    doses: set[str] = set()
    dose_units = {"MG", "G", "MCG", "UG", "KG", "ML", "L", "PCT"}
    for i, tok in enumerate(tokens):
        if _is_numeric_token(tok):
            # Check if next token is a dose unit
            if i + 1 < len(tokens) and tokens[i + 1] in dose_units:
                doses.add(tok)
            # Check if previous token is a dose unit (for reversed order)
            elif i > 0 and tokens[i - 1] in dose_units:
                doses.add(tok)
            # Check if any nearby token (within 2) is a dose unit
            elif any(tokens[j] in dose_units for j in range(max(0, i-2), min(len(tokens), i+3)) if j != i):
                doses.add(tok)
    return doses


def _check_dose_overlap(annex_doses: set[str], ref_doses: set[str]) -> bool:
    """Check if there's any overlap in numeric dose values."""
    if not annex_doses or not ref_doses:
        return True  # No dose to compare, allow match
    return bool(annex_doses & ref_doses)


def _categorize_tokens(tokens: List[str]) -> dict[str, Counter]:
    cat_counts: dict[str, Counter] = {
        CATEGORY_GENERIC: Counter(),
        CATEGORY_SALT: Counter(),
        CATEGORY_DOSE: Counter(),
        CATEGORY_FORM: Counter(),
        CATEGORY_ROUTE: Counter(),
        CATEGORY_OTHER: Counter(),
    }
    # Detect compound generics first
    compound_parts = _detect_compound_generics(tokens)
    for tok in tokens:
        # If token is part of a compound generic, classify as generic not salt
        if tok in compound_parts:
            cat_counts[CATEGORY_GENERIC][tok] += 1
        else:
            cat = _classify_token(tok)
            if cat not in cat_counts:
                cat = CATEGORY_OTHER
            cat_counts[cat][tok] += 1
    return cat_counts


def _extract_generic_tokens(tokens: List[str]) -> set[str]:
    generics: set[str] = set()
    compound_parts = _detect_compound_generics(tokens)
    for tok in tokens:
        if tok in compound_parts or _classify_token(tok) == CATEGORY_GENERIC:
            generics.add(tok)
    return generics


def _extract_high_value_generics(tokens: List[str]) -> set[str]:
    return {tok for tok in _extract_generic_tokens(tokens) if tok not in GENERIC_JUNK_TOKENS}


def _tokens_from_generic_hits(generic_hits: Iterable[str]) -> set[str]:
    """Extract tokens from generic hits, filtering out junk tokens."""
    tokens: set[str] = set()
    for phrase in generic_hits or []:
        for piece in phrase.split():
            tok = GENERIC_SYNONYMS.get(piece.upper(), piece.upper())
            # Filter out junk tokens that shouldn't be used for matching
            if tok and tok not in GENERIC_JUNK_TOKENS:
                tokens.add(tok)
    return tokens


def _build_brand_map(fda_df: pd.DataFrame) -> list[tuple[re.Pattern, str]]:
    patterns: list[tuple[re.Pattern, str]] = []
    for row in fda_df.to_dict(orient="records"):
        brand = _as_str_or_empty(row.get("brand_name")).strip()
        generic = _as_str_or_empty(row.get("generic_name")).strip()
        if not brand or not generic:
            continue
        pat = re.compile(rf"(?i)\b{re.escape(brand)}\b")
        patterns.append((pat, generic))
    return patterns


def _apply_brand_swaps(text: str, brand_patterns: list[tuple[re.Pattern, str]]) -> str:
    if not isinstance(text, str):
        return ""
    updated = text
    for pat, repl in brand_patterns:
        updated = pat.sub(repl, updated)
    return updated


def _build_aho_automaton(phrases: list[str]):
    root = {"next": {}, "fail": None, "out": []}
    for phrase in phrases:
        node = root
        for ch in phrase:
            node = node["next"].setdefault(ch, {"next": {}, "fail": None, "out": []})
        node["out"].append(phrase)
    queue = []
    for child in root["next"].values():
        child["fail"] = root
        queue.append(child)
    while queue:
        current = queue.pop(0)
        for ch, nxt in current["next"].items():
            queue.append(nxt)
            f = current["fail"]
            while f and ch not in f["next"]:
                f = f["fail"]
            nxt["fail"] = f["next"][ch] if f and ch in f["next"] else root
            nxt["out"].extend(nxt["fail"]["out"] if nxt["fail"] else [])
    return root


def _aho_find(text: str, automaton) -> set[str]:
    node = automaton
    matches: set[str] = set()
    for ch in text:
        while node and ch not in node["next"]:
            node = node["fail"]
        if not node:
            node = automaton
            continue
        node = node["next"][ch]
        for out in node.get("out", []):
            matches.add(out)
    return matches


def _normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _build_mixture_lookup(mixture_df: pd.DataFrame) -> dict[str, list[dict]]:
    lookup: dict[str, list[dict]] = {}
    for row in mixture_df.to_dict(orient="records"):
        raw_key = _as_str_or_empty(row.get("ingredient_components_key"))
        if not raw_key:
            continue
        parts = [_normalize_phrase(part) for part in raw_key.split("||") if part]
        if not parts:
            continue
        key = "||".join(sorted(parts))
        lookup.setdefault(key, []).append(dict(row))
    return lookup


def _build_generic_phrases(drugbank_df: pd.DataFrame) -> list[str]:
    phrases: set[str] = set()
    for row in drugbank_df.to_dict(orient="records"):
        for col in ("canonical_generic_name", "lexeme"):
            phrase = _normalize_phrase(_as_str_or_empty(row.get(col)))
            if len(phrase) >= 3:
                phrases.add(phrase)
    for raw, canon in GENERIC_SYNONYMS.items():
        phrases.add(_normalize_phrase(raw))
        phrases.add(_normalize_phrase(canon))
    return sorted(phrases)


def _build_generic_to_atc_map(drugbank_df: pd.DataFrame) -> dict[str, list[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for row in drugbank_df.to_dict(orient="records"):
        atc = _as_str_or_empty(row.get("atc_code"))
        if not atc:
            continue
        for col in ("canonical_generic_name", "lexeme", "generic_components_key"):
            phrase = _normalize_phrase(_as_str_or_empty(row.get(col)))
            if len(phrase) >= 3:
                lookup[phrase].add(atc)
    for raw, canon in GENERIC_SYNONYMS.items():
        norm_raw = _normalize_phrase(raw)
        norm_canon = _normalize_phrase(canon)
        if norm_canon in lookup:
            lookup[norm_raw].update(lookup[norm_canon])
    return {phrase: sorted(codes) for phrase, codes in lookup.items() if codes}


def _normalize_annex_df(df: pd.DataFrame, brand_patterns: list[tuple[re.Pattern, str]]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    descriptions = df["Drug Description"].to_list()
    fuzzy_vals = []
    for desc in descriptions:
        description = _apply_brand_swaps(_as_str_or_empty(desc), brand_patterns)
        base_tokens = split_with_parentheses(description)
        normalized_tokens = _normalize_tokens(base_tokens, drop_stopwords=False)
        fuzzy_vals.append("|".join(normalized_tokens))
    df["fuzzy_basis"] = fuzzy_vals
    return df


def _precompute_annex_records(
    annex_df: pd.DataFrame,
    brand_patterns: list[tuple[re.Pattern, str]],
    generic_phrases: list[str],
    generic_atc_map: dict[str, list[str]] | None = None,
) -> list[dict]:
    records = []
    generic_automaton = _build_aho_automaton(generic_phrases) if generic_phrases else None
    generic_atc_map = generic_atc_map or {}
    for row in annex_df.to_dict(orient="records"):
        description = _apply_brand_swaps(_as_str_or_empty(row.get("Drug Description")), brand_patterns)
        base_tokens = split_with_parentheses(description)
        tokens_primary = _normalize_tokens(base_tokens, drop_stopwords=True)
        tokens_secondary = _normalize_tokens(base_tokens, drop_stopwords=False)
        row["fuzzy_basis"] = "|".join(tokens_secondary)
        generic_hits = _aho_find(_normalize_phrase(description), generic_automaton) if generic_automaton else set()
        generic_hit_atcs: set[str] = set()
        for hit in generic_hits:
            generic_hit_atcs.update(generic_atc_map.get(hit, ()))
        primary_generic_tokens = _extract_generic_tokens(tokens_primary)
        secondary_generic_tokens = _extract_generic_tokens(tokens_secondary)
        primary_high_value_generics = _extract_high_value_generics(tokens_primary)
        secondary_high_value_generics = _extract_high_value_generics(tokens_secondary)
        generic_hit_tokens = _tokens_from_generic_hits(generic_hits)
        records.append(
            row
            | {
                "fuzzy_tokens_primary": tokens_primary,
                "fuzzy_tokens_secondary": tokens_secondary,
                "fuzzy_counts_primary": Counter(tokens_primary),
                "fuzzy_counts_secondary": Counter(tokens_secondary),
                "annex_cat_primary": _categorize_tokens(tokens_primary),
                "annex_cat_secondary": _categorize_tokens(tokens_secondary),
                "norm_description": _normalize_phrase(description),
                "generic_hits": generic_hits,
                "generic_hit_atcs": sorted(generic_hit_atcs),
                "primary_generic_tokens": primary_generic_tokens,
                "secondary_generic_tokens": secondary_generic_tokens,
                "primary_high_value_generics": primary_high_value_generics,
                "secondary_high_value_generics": secondary_high_value_generics,
                "generic_hit_tokens": generic_hit_tokens,
            }
        )
    return records


def _build_reference_display(raw_ref: dict) -> str | None:
    if not raw_ref:
        return None
    parts: list[str] = []

    generic = None
    for key in ("canonical_generic_name", "generic_name", "atc_name", "lexeme", "generic_components"):
        text = _as_str_or_empty(raw_ref.get(key))
        if text:
            generic = text
            break
    if generic:
        parts.append(generic)

    # Handle dose display - standardize all doses to mg
    dose_piece = None
    unit_piece = None

    # First try strength_mg (already normalized to mg)
    strength_mg = _as_str_or_empty(raw_ref.get("strength_mg"))
    if strength_mg:
        dose_piece = strength_mg
        unit_piece = "MG"
    else:
        # Try dose_norm (which should be in mg format like "5mg")
        dose_norm = _as_str_or_empty(raw_ref.get("dose_norm"))
        if dose_norm:
            # dose_norm may include unit, extract just the number if needed
            if "MG" in dose_norm.upper():
                dose_piece = dose_norm
                unit_piece = None  # Already included
            else:
                dose_piece = dose_norm
                unit_piece = "MG"
        else:
            # Fall back to original strength + unit, but convert to mg
            strength = _as_str_or_empty(raw_ref.get("strength"))
            unit = _as_str_or_empty(raw_ref.get("unit")).upper()
            if strength and unit:
                # Convert to mg if possible
                mg_val = _convert_to_mg(strength, unit)
                if mg_val:
                    dose_piece = mg_val
                    unit_piece = "MG"
                else:
                    # Can't convert, use original
                    dose_piece = strength
                    unit_piece = unit
            else:
                # Try other dose fields
                for key in ("raw_dose", "pct", "per_val"):
                    text = _as_str_or_empty(raw_ref.get(key))
                    if text:
                        dose_piece = text
                        if key == "pct":
                            unit_piece = "%"
                        break

    if dose_piece:
        parts.append(dose_piece)
    if unit_piece:
        parts.append(unit_piece)

    salt = None
    for key in ("salt_names", "salt_form"):
        text = _as_str_or_empty(raw_ref.get(key))
        if text:
            salt = text
            break
    if salt:
        parts.append(salt)

    form = None
    for key in ("form_norm", "raw_form", "form_token"):
        text = _as_str_or_empty(raw_ref.get(key))
        if text:
            form = text
            break
    if form:
        parts.append(form)

    route = None
    for key in ("route_norm", "raw_route", "route_allowed", "adm_r"):
        text = _as_str_or_empty(raw_ref.get(key))
        if text:
            route = text
            break
    if route:
        parts.append(route)

    display = " ".join(part for part in parts if part).strip()
    if display:
        return display

    lex = _as_str_or_empty(raw_ref.get("lexicon"))
    return lex or None


def _reference_sort_key(rec: dict) -> tuple[int, str, str]:
    source_priority = {"pnf": 0, "drugbank": 1, "drugbank_mixture": 2}
    priority = source_priority.get(rec.get("source"), 99)
    name = _as_str_or_empty(rec.get("name")).upper()
    ident = _as_str_or_empty(rec.get("id")).upper()
    return (priority, name, ident)


def build_pnf_reference(df: pd.DataFrame) -> list[dict]:
    rows = []
    for row in df.to_dict(orient="records"):
        lexicon = _as_str_or_empty(row.get("lexicon"))
        lexicon_secondary = _as_str_or_empty(row.get("lexicon_secondary"))
        primary_tokens = _normalize_tokens(parse_pipe_tokens(lexicon), drop_stopwords=True)
        secondary_tokens = _normalize_tokens(parse_pipe_tokens(lexicon_secondary), drop_stopwords=False)
        rows.append(
            {
                "source": "pnf",
                "id": _maybe_none(row.get("generic_id")),
                "name": _maybe_none(row.get("generic_name")),
                "lexicon": lexicon,
                "lexicon_secondary": lexicon_secondary,
                "primary_tokens": primary_tokens,
                "secondary_tokens": secondary_tokens,
                "primary_cat_counts": _categorize_tokens(primary_tokens),
                "secondary_cat_counts": _categorize_tokens(secondary_tokens),
                "primary_generic_tokens": _extract_generic_tokens(primary_tokens),
                "secondary_generic_tokens": _extract_generic_tokens(secondary_tokens),
                "primary_high_value_generics": _extract_high_value_generics(primary_tokens),
                "secondary_high_value_generics": _extract_high_value_generics(secondary_tokens),
                "atc_code": _maybe_none(row.get("atc_code")),
                "raw_reference_row": dict(row),
            }
        )
    return rows


def build_drugbank_reference(df: pd.DataFrame) -> list[dict]:
    rows = []
    for row in df.to_dict(orient="records"):
        primary_tokens: List[str] = []
        for col in ("lexeme", "generic_components_key", "canonical_generic_name"):
            primary_tokens.extend(split_with_parentheses(row.get(col)))
        primary_tokens = _normalize_tokens(primary_tokens, drop_stopwords=True)

        secondary_tokens: List[str] = []
        for col in ("route_norm", "form_norm", "salt_names", "dose_norm"):
            secondary_tokens.extend(split_with_parentheses(row.get(col)))
        secondary_tokens = _normalize_tokens(secondary_tokens, drop_stopwords=False)

        name = _maybe_none(row.get("canonical_generic_name")) or _maybe_none(
            row.get("lexeme")
        )

        raw_reference_row = dict(row)

        rows.append(
            {
                "source": "drugbank",
                "id": _maybe_none(row.get("drugbank_id")),
                "name": name,
                "lexicon": "|".join(primary_tokens),
                "lexicon_secondary": "|".join(secondary_tokens),
                "primary_tokens": primary_tokens,
                "secondary_tokens": secondary_tokens,
                "primary_cat_counts": _categorize_tokens(primary_tokens),
                "secondary_cat_counts": _categorize_tokens(secondary_tokens),
                "primary_generic_tokens": _extract_generic_tokens(primary_tokens),
                "secondary_generic_tokens": _extract_generic_tokens(secondary_tokens),
                "primary_high_value_generics": _extract_high_value_generics(primary_tokens),
                "secondary_high_value_generics": _extract_high_value_generics(secondary_tokens),
                "atc_code": _maybe_none(row.get("atc_code")),
                "raw_reference_row": raw_reference_row,
            }
        )
    return rows


def build_reference_rows(
    pnf_df: pd.DataFrame, drugbank_df: pd.DataFrame
) -> list[dict]:
    refs: list[dict] = []
    refs.extend(build_pnf_reference(pnf_df))
    refs.extend(build_drugbank_reference(drugbank_df))
    return refs


def _build_reference_index(reference_rows: list[dict]) -> dict[str, list[int]]:
    reference_index: dict[str, list[int]] = defaultdict(list)
    for idx, ref in enumerate(reference_rows):
        for tok in ref.get("primary_tokens", ()):
            reference_index[tok].append(idx)
    return reference_index


def _group_drugbank_refs_by_id(reference_rows: list[dict]) -> dict[str, list[dict]]:
    lookup: dict[str, list[dict]] = defaultdict(list)
    for ref in reference_rows:
        if ref.get("source") == "drugbank" and ref.get("id"):
            lookup[str(ref["id"])].append(ref)
    return lookup


def _build_generic_to_drugbank_id(drugbank_df: pd.DataFrame) -> dict[str, str]:
    """Build a lookup from normalized generic name to DrugBank ID."""
    lookup: dict[str, str] = {}
    for row in drugbank_df.to_dict(orient="records"):
        db_id = _as_str_or_empty(row.get("drugbank_id"))
        if not db_id:
            continue
        for col in ("canonical_generic_name", "lexeme"):
            name = _as_str_or_empty(row.get(col))
            if name:
                norm_name = name.upper().strip()
                if norm_name and norm_name not in lookup:
                    lookup[norm_name] = db_id
    return lookup


def _empty_match_record(annex_row: dict) -> dict:
    return {
        "Drug Code": annex_row.get("Drug Code"),
        "Drug Description": annex_row.get("Drug Description"),
        "fuzzy_basis": _as_str_or_empty(annex_row.get("fuzzy_basis")),
        "matched_reference_raw": None,
        "matched_source": None,
        "matched_generic_name": None,
        "matched_lexicon": None,
        "match_count": None,
        "matched_secondary_lexicon": None,
        "secondary_match_count": None,
        "atc_code": None,
        "drugbank_id": None,
        "primary_matching_tokens": None,
        "secondary_matching_tokens": None,
    }


def _build_match_record(
    annex_row: dict,
    ref: dict,
    primary_score: int,
    secondary_score: int,
    generic_to_drugbank: dict[str, str] | None = None,
) -> dict:
    primary_overlap = ref.get("primary_overlap") or []
    secondary_overlap = ref.get("secondary_overlap") or []
    raw_ref = ref.get("raw_reference_row") or {}
    raw_display = _build_reference_display(raw_ref)
    if raw_display:
        raw_display = raw_display.upper()

    def _resolve_atc_code() -> str | None:
        generic_hit_atcs = annex_row.get("generic_hit_atcs") or []
        atc = _as_str_or_empty(ref.get("atc_code"))
        if ref.get("source") == "drugbank_mixture":
            if generic_hit_atcs:
                return "|".join(sorted(set(generic_hit_atcs)))
            return atc or None
        if atc:
            return atc
        if generic_hit_atcs:
            return "|".join(sorted(set(generic_hit_atcs)))
        return None

    def _resolve_drugbank_id() -> str | None:
        # Get drugbank_id from the reference if available
        db_id = _as_str_or_empty(ref.get("id"))
        if db_id and ref.get("source") in ("drugbank", "drugbank_mixture"):
            return db_id
        # Also check raw_reference_row for drugbank_id
        raw_db_id = _as_str_or_empty(raw_ref.get("drugbank_id"))
        if raw_db_id:
            return raw_db_id
        # For PNF matches, try to look up by generic name
        if generic_to_drugbank:
            generic_name = _as_str_or_empty(ref.get("name")).upper().strip()
            if generic_name and generic_name in generic_to_drugbank:
                return generic_to_drugbank[generic_name]
        return None

    return {
        "Drug Code": annex_row.get("Drug Code"),
        "Drug Description": annex_row.get("Drug Description"),
        "fuzzy_basis": _as_str_or_empty(annex_row.get("fuzzy_basis")),
        "matched_reference_raw": raw_display,
        "matched_source": ref.get("source"),
        "matched_generic_name": ref.get("name"),
        "matched_lexicon": ref.get("lexicon"),
        "match_count": primary_score,
        "matched_secondary_lexicon": ref.get("lexicon_secondary"),
        "secondary_match_count": secondary_score,
        "atc_code": _resolve_atc_code(),
        "drugbank_id": _resolve_drugbank_id(),
        "primary_matching_tokens": "|".join(primary_overlap) if primary_overlap else None,
        "secondary_matching_tokens": "|".join(secondary_overlap) if secondary_overlap else None,
    }


def _score_annex_row(
    annex_row: dict,
    reference_rows: list[dict],
    reference_index: dict[str, list[int]],
    brand_patterns: list[tuple[re.Pattern, str]],
    generic_automaton,
    mixture_lookup: dict[str, list[dict]],
    drugbank_refs_by_id: dict[str, list[dict]],
    generic_to_drugbank: dict[str, str] | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    matched_rows: list[dict] = []
    tie_rows: list[dict] = []
    unresolved_rows: list[dict] = []

    fuzzy_tokens_primary = annex_row.get("fuzzy_tokens_primary") or []
    fuzzy_tokens_secondary = annex_row.get("fuzzy_tokens_secondary") or []
    fuzzy_counts_primary = annex_row.get("fuzzy_counts_primary") or Counter(fuzzy_tokens_primary)
    fuzzy_counts_secondary = annex_row.get("fuzzy_counts_secondary") or Counter(fuzzy_tokens_secondary)
    annex_cat_primary = annex_row.get("annex_cat_primary") or _categorize_tokens(fuzzy_tokens_primary)
    annex_cat_secondary = annex_row.get("annex_cat_secondary") or _categorize_tokens(fuzzy_tokens_secondary)
    annex_generic_total = sum(annex_cat_primary.get(CATEGORY_GENERIC, Counter()).values())
    annex_generic_total_secondary = sum(annex_cat_secondary.get(CATEGORY_GENERIC, Counter()).values())
    annex_primary_generic_tokens = annex_row.get("primary_generic_tokens") or set()
    annex_primary_high_value_generics = annex_row.get("primary_high_value_generics") or set()
    annex_generic_hit_tokens = annex_row.get("generic_hit_tokens") or set()

    if not fuzzy_tokens_primary:
        matched_rows.append(_empty_match_record(annex_row))
        return matched_rows[0], tie_rows, unresolved_rows

    candidate_indices: set[int] = set()
    for tok in fuzzy_tokens_primary:
        for idx in reference_index.get(tok, ()):
            candidate_indices.add(idx)
    if not candidate_indices:
        candidate_refs = list(reference_rows)
    else:
        candidate_refs = [reference_rows[i] for i in sorted(candidate_indices)]

    # Mixture detection via Aho-Corasick on normalized phrase.
    norm_desc = annex_row.get("norm_description") or ""
    generic_hits = annex_row.get("generic_hits") or set()
    if len(generic_hits) >= 2:
        component_key = "||".join(sorted(generic_hits))
        mixture_rows = mixture_lookup.get(component_key, [])
        for mix in mixture_rows:
            mix_id = _as_str_or_empty(mix.get("mixture_drugbank_id")) or _as_str_or_empty(mix.get("mixture_id"))
            mix_name = _as_str_or_empty(mix.get("mixture_name"))
            attached = False
            if mix_id and mix_id in drugbank_refs_by_id:
                candidate_refs.extend(drugbank_refs_by_id[mix_id])
                attached = True
            if not attached:
                primary_tokens = _normalize_tokens(split_with_parentheses(mix_name), drop_stopwords=True)
                secondary_tokens = _normalize_tokens(split_with_parentheses(mix_name), drop_stopwords=False)
                candidate_refs.append(
                    {
                        "source": "drugbank_mixture",
                        "id": mix_id or mix.get("mixture_id"),
                        "name": mix_name,
                        "lexicon": "|".join(primary_tokens),
                        "lexicon_secondary": "|".join(secondary_tokens),
                        "primary_tokens": primary_tokens,
                        "secondary_tokens": secondary_tokens,
                        "primary_cat_counts": _categorize_tokens(primary_tokens),
                        "secondary_cat_counts": _categorize_tokens(secondary_tokens),
                        "primary_generic_tokens": _extract_generic_tokens(primary_tokens),
                        "secondary_generic_tokens": _extract_generic_tokens(secondary_tokens),
                        "primary_high_value_generics": _extract_high_value_generics(primary_tokens),
                        "secondary_high_value_generics": _extract_high_value_generics(secondary_tokens),
                        "atc_code": None,
                        "raw_reference_row": mix,
                    }
                )

    best_primary = -10**9
    best_primary_records: list[dict] = []
    for ref in candidate_refs:
        ref_primary_generic_tokens = ref.get("primary_generic_tokens") or set()
        ref_primary_high_value_generics = ref.get("primary_high_value_generics") or set()
        require_generic_overlap = GENERIC_MATCH_REQUIRED and annex_generic_total > 0
        if require_generic_overlap and not (annex_primary_generic_tokens & ref_primary_generic_tokens):
            continue
        if annex_primary_high_value_generics and not (
            annex_primary_high_value_generics & ref_primary_high_value_generics
        ):
            continue
        if annex_generic_hit_tokens and not (annex_generic_hit_tokens & ref_primary_generic_tokens):
            continue
        primary_overlap = _ordered_overlap(
            fuzzy_tokens_primary, fuzzy_counts_primary, ref.get("primary_tokens", ())
        )
        ref_primary_counts = ref.get("primary_cat_counts", {})
        generic_overlap = sum(
            (annex_cat_primary.get(CATEGORY_GENERIC, Counter()) & ref_primary_counts.get(CATEGORY_GENERIC, Counter())).values()
        )
        if GENERIC_MATCH_REQUIRED and annex_generic_total and generic_overlap == 0:
            continue
        cat_scores = []
        for cat, ref_counts in ref_primary_counts.items():
            match_count = sum((annex_cat_primary.get(cat, Counter()) & ref_counts).values())
            mismatch = max(0, sum(ref_counts.values()) - match_count)
            weight = PRIMARY_WEIGHTS.get(cat, 1)
            mismatch_penalty = mismatch
            if cat == CATEGORY_GENERIC:
                excess = max(0, mismatch - GENERIC_REF_MISMATCH_TOLERANCE_PRIMARY)
                mismatch_penalty += excess * GENERIC_REF_EXTRA_PENALTY_PRIMARY
            cat_scores.append(weight * match_count - mismatch_penalty)
        generic_missing = max(0, annex_generic_total - generic_overlap)
        primary_score = sum(cat_scores) - (GENERIC_MISS_PENALTY_PRIMARY * generic_missing if annex_generic_total else 0)

        if primary_score == 0:
            continue
        if primary_score < best_primary:
            continue
        if primary_score > best_primary:
            best_primary_records = []
            best_primary = primary_score
        best_primary_records.append(
            ref
            | {
                "primary_overlap": primary_overlap,
                "primary_score": primary_score,
            }
        )

    if not best_primary_records or best_primary == -10**9:
        matched_rows.append(_empty_match_record(annex_row))
        return matched_rows[0], tie_rows, unresolved_rows

    best_secondary = -10**9
    finalists: list[dict] = []
    # Extract numeric doses from Annex F for dose matching
    annex_doses = _extract_numeric_doses(fuzzy_tokens_secondary)

    for rec in best_primary_records:
        secondary_overlap = _ordered_overlap(
            fuzzy_tokens_secondary, fuzzy_counts_secondary, rec.get("secondary_tokens", ())
        )
        ref_secondary_tokens = rec.get("secondary_tokens", [])
        ref_secondary_counts = rec.get("secondary_cat_counts", {})
        generic_overlap_secondary = sum(
            (annex_cat_secondary.get(CATEGORY_GENERIC, Counter()) & ref_secondary_counts.get(CATEGORY_GENERIC, Counter())).values()
        )

        # Check dose matching - extract numeric doses from reference
        ref_doses = _extract_numeric_doses(ref_secondary_tokens)
        dose_matches = _check_dose_overlap(annex_doses, ref_doses)

        cat_scores = []
        for cat, ref_counts in ref_secondary_counts.items():
            match_count = sum((annex_cat_secondary.get(cat, Counter()) & ref_counts).values())
            mismatch = max(0, sum(ref_counts.values()) - match_count)
            weight = SECONDARY_WEIGHTS.get(cat, 1)
            bonus = 0
            if cat == CATEGORY_DOSE:
                numeric_match = sum(
                    (annex_cat_secondary.get(cat, Counter()) & ref_counts).values()
                )
                bonus = numeric_match
            mismatch_penalty = mismatch
            if cat == CATEGORY_GENERIC:
                excess = max(0, mismatch - GENERIC_REF_MISMATCH_TOLERANCE_SECONDARY)
                mismatch_penalty += excess * GENERIC_REF_EXTRA_PENALTY_SECONDARY
            cat_scores.append(weight * match_count + bonus - mismatch_penalty)
        generic_missing_secondary = max(0, annex_generic_total_secondary - generic_overlap_secondary)
        secondary_score = sum(cat_scores) - (
            GENERIC_MISS_PENALTY_SECONDARY * generic_missing_secondary if annex_generic_total_secondary else 0
        )

        # Apply dose mismatch penalty if Annex F has doses but they don't match reference
        if REQUIRE_DOSE_MATCH and annex_doses and ref_doses and not dose_matches:
            secondary_score -= DOSE_MISMATCH_PENALTY
        if secondary_score > best_secondary:
            finalists = []
            best_secondary = secondary_score
        if secondary_score == best_secondary:
            finalists.append(
                rec
                | {
                    "secondary_overlap": secondary_overlap,
                    "secondary_score": secondary_score,
                }
            )

    if not finalists:
        matched_rows.append(_empty_match_record(annex_row))
        return matched_rows[0], tie_rows, unresolved_rows

    if len(finalists) == 1:
        winner = finalists[0]
        matched_rows.append(
            _build_match_record(annex_row, winner, best_primary, best_secondary, generic_to_drugbank)
        )
        return matched_rows[0], tie_rows, unresolved_rows

    sorted_finalists = sorted(finalists, key=_reference_sort_key)
    atc_set = {_as_str_or_empty(rec.get("atc_code")) for rec in sorted_finalists}
    if len(atc_set) == 1:
        # Check if this is an IV solution (BOTTLE, BAG, SOLUTION in description)
        annex_desc_upper = _as_str_or_empty(annex_row.get("Drug Description")).upper()
        is_iv_solution = any(
            term in annex_desc_upper
            for term in ("BOTTLE", "BAG", "SOLUTION", "INFUSION", "AMPULE", "VIAL")
        )

        def _form_route_score(rec: dict) -> tuple[int, int, int]:
            sec_counts = rec.get("secondary_cat_counts", {})
            form_match = sum(
                (annex_cat_secondary.get(CATEGORY_FORM, Counter()) & sec_counts.get(CATEGORY_FORM, Counter())).values()
            )
            route_match = sum(
                (annex_cat_secondary.get(CATEGORY_ROUTE, Counter()) & sec_counts.get(CATEGORY_ROUTE, Counter())).values()
            )
            # For IV solutions, prefer INTRAVENOUS route
            iv_bonus = 0
            if is_iv_solution:
                ref_display = _as_str_or_empty(rec.get("lexicon_secondary")).upper()
                if "INTRAVENOUS" in ref_display:
                    iv_bonus = 1
            return (route_match, form_match, iv_bonus)

        sorted_finalists = sorted(
            sorted_finalists,
            key=lambda rec: (
                -_form_route_score(rec)[2],  # IV bonus first
                -_form_route_score(rec)[0],
                -_form_route_score(rec)[1],
                _reference_sort_key(rec),
            ),
        )
        winner = sorted_finalists[0]
        matched_rows.append(
            _build_match_record(annex_row, winner, best_primary, best_secondary, generic_to_drugbank)
        )
        for rec in sorted_finalists:
            tie_rows.append(
                _build_match_record(
                    annex_row,
                    rec,
                    rec.get("primary_score", best_primary),
                    rec.get("secondary_score", best_secondary),
                    generic_to_drugbank,
                )
            )
        return matched_rows[0], tie_rows, unresolved_rows

    # Multiple different ATCs - try to pick the best one
    # Strategy: prefer single-agent ATCs over combination ATCs
    # Also prefer ATCs that match more components for combination drugs
    annex_desc = _as_str_or_empty(annex_row.get("Drug Description"))
    is_combination_drug = "+" in annex_desc

    def _atc_preference_score(atc: str) -> int:
        """Score an ATC code - lower is better."""
        is_combo_atc = _is_combination_atc(atc)
        if is_combination_drug:
            return 0 if is_combo_atc else 1  # Prefer combo ATCs for combo drugs
        else:
            return 1 if is_combo_atc else 0  # Prefer single ATCs for single drugs

    # Group finalists by ATC and score each ATC
    atc_to_recs: dict[str, list[dict]] = {}
    for rec in sorted_finalists:
        atc = _as_str_or_empty(rec.get("atc_code"))
        atc_to_recs.setdefault(atc, []).append(rec)

    # Sort ATCs by preference
    sorted_atcs = sorted(atc_to_recs.keys(), key=_atc_preference_score)

    # Check if we have a clear winner (best ATC is strictly better than second-best)
    if len(sorted_atcs) >= 2:
        best_atc_score = _atc_preference_score(sorted_atcs[0])
        second_atc_score = _atc_preference_score(sorted_atcs[1])
        if best_atc_score < second_atc_score:
            # We have a clear winner - pick the best record from the winning ATC
            winning_recs = atc_to_recs[sorted_atcs[0]]
            # Sort by source (prefer PNF) then by secondary score
            winning_recs_sorted = sorted(
                winning_recs,
                key=lambda r: (
                    0 if r.get("source") == "pnf" else 1,
                    -(r.get("secondary_score") or 0),
                    _reference_sort_key(r),
                ),
            )
            winner = winning_recs_sorted[0]
            matched_rows.append(
                _build_match_record(annex_row, winner, best_primary, best_secondary, generic_to_drugbank)
            )
            for rec in sorted_finalists:
                unresolved_rows.append(
                    _build_match_record(
                        annex_row,
                        rec,
                        rec.get("primary_score", best_primary),
                        rec.get("secondary_score", best_secondary),
                        generic_to_drugbank,
                    )
                )
            return matched_rows[0], tie_rows, unresolved_rows

    # No clear winner - still unresolved
    matched_rows.append(_empty_match_record(annex_row))
    for rec in sorted_finalists:
        unresolved_rows.append(
            _build_match_record(
                annex_row,
                rec,
                rec.get("primary_score", best_primary),
                rec.get("secondary_score", best_secondary),
                generic_to_drugbank,
            )
        )
    return matched_rows[0], tie_rows, unresolved_rows


def _init_reference_rows(reference_rows: list[dict]) -> None:
    global _REFERENCE_ROWS_SHARED
    _REFERENCE_ROWS_SHARED = reference_rows


def _init_shared(
    reference_rows: list[dict],
    reference_index: dict[str, list[int]],
    brand_patterns: list[tuple[re.Pattern, str]],
    generic_phrases: list[str],
    mixture_lookup: dict[str, list[dict]],
    drugbank_refs_by_id: dict[str, list[dict]],
    generic_to_drugbank: dict[str, str] | None = None,
) -> None:
    global _REFERENCE_ROWS_SHARED
    global _REFERENCE_INDEX_SHARED
    global _BRAND_PATTERNS_SHARED
    global _GENERIC_PHRASES_SHARED
    global _GENERIC_AUTOMATON_SHARED
    global _MIXTURE_LOOKUP_SHARED
    global _DRUGBANK_BY_ID_SHARED
    global _GENERIC_TO_DRUGBANK_SHARED
    _REFERENCE_ROWS_SHARED = reference_rows
    _REFERENCE_INDEX_SHARED = reference_index
    _BRAND_PATTERNS_SHARED = brand_patterns
    _GENERIC_PHRASES_SHARED = generic_phrases
    # Annex rows already carry generic_hits; avoid building automaton per worker.
    _GENERIC_AUTOMATON_SHARED = None
    _MIXTURE_LOOKUP_SHARED = mixture_lookup
    _DRUGBANK_BY_ID_SHARED = drugbank_refs_by_id
    _GENERIC_TO_DRUGBANK_SHARED = generic_to_drugbank


def _process_annex_row_worker(annex_row: dict) -> tuple[dict, list[dict], list[dict]]:
    reference_rows = _REFERENCE_ROWS_SHARED or []
    reference_index = _REFERENCE_INDEX_SHARED or {}
    brand_patterns = _BRAND_PATTERNS_SHARED or []
    generic_automaton = _GENERIC_AUTOMATON_SHARED
    mixture_lookup = _MIXTURE_LOOKUP_SHARED or {}
    drugbank_refs_by_id = _DRUGBANK_BY_ID_SHARED or {}
    generic_to_drugbank = _GENERIC_TO_DRUGBANK_SHARED or {}
    return _score_annex_row(
        annex_row,
        reference_rows,
        reference_index,
        brand_patterns,
        generic_automaton,
        mixture_lookup,
        drugbank_refs_by_id,
        generic_to_drugbank,
    )


def match_annex_with_atc(
    annex_df: pd.DataFrame,
    reference_rows: list[dict],
    reference_index: dict[str, list[int]],
    brand_patterns: list[tuple[re.Pattern, str]],
    generic_automaton,
    generic_phrases: list[str],
    generic_atc_map: dict[str, list[str]] | None = None,
    mixture_lookup: dict[str, list[dict]] | None = None,
    drugbank_refs_by_id: dict[str, list[dict]] | None = None,
    generic_to_drugbank: dict[str, str] | None = None,
    max_workers: int | None = None,
    chunk_size: int = 172,
    use_threads: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mixture_lookup = mixture_lookup or {}
    drugbank_refs_by_id = drugbank_refs_by_id or {}
    generic_to_drugbank = generic_to_drugbank or {}
    annex_records = (
        annex_df
        if isinstance(annex_df, list)
        else _precompute_annex_records(annex_df, brand_patterns, generic_phrases, generic_atc_map)
    )
    matched_rows: list[dict] = []
    tie_rows: list[dict] = []
    unresolved_rows: list[dict] = []

    worker_count = _effective_workers(max_workers)
    use_threads = bool(use_threads)
    backend_label = "threads" if use_threads else "processes"
    print(f"[match_annex_with_atc] workers={worker_count} backend={backend_label}")
    if worker_count <= 1:
        for rec in annex_records:
            match_row, ties, unresolved = _score_annex_row(
                rec,
                reference_rows,
                reference_index,
                brand_patterns,
                generic_automaton,
                mixture_lookup,
                drugbank_refs_by_id,
                generic_to_drugbank,
            )
            matched_rows.append(match_row)
            tie_rows.extend(ties)
            unresolved_rows.extend(unresolved)
        return (
            pd.DataFrame(matched_rows),
            pd.DataFrame(tie_rows),
            pd.DataFrame(unresolved_rows),
        )

    if use_threads:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            for match_row, ties, unresolved in executor.map(
                lambda rec: _score_annex_row(
                    rec,
                    reference_rows,
                    reference_index,
                    brand_patterns,
                    generic_automaton,
                    mixture_lookup,
                    drugbank_refs_by_id,
                    generic_to_drugbank,
                ),
                annex_records,
            ):
                matched_rows.append(match_row)
                tie_rows.extend(ties)
                unresolved_rows.extend(unresolved)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_shared,
            initargs=(reference_rows, reference_index, brand_patterns, generic_phrases, mixture_lookup, drugbank_refs_by_id, generic_to_drugbank),
        ) as executor:
            for match_row, ties, unresolved in executor.map(
                _process_annex_row_worker, annex_records, chunksize=max(1, chunk_size)
            ):
                matched_rows.append(match_row)
                tie_rows.extend(ties)
                unresolved_rows.extend(unresolved)

    return (
        pd.DataFrame(matched_rows),
        pd.DataFrame(tie_rows),
        pd.DataFrame(unresolved_rows),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match Annex F entries to ATC codes.")
    parser.add_argument(
        "--workers",
        type=int,
        default=14,
        help="Number of workers to use (defaults to 8; raise for more CPU, lower if memory is tight).",
    )
    parser.add_argument(
        "--use-threads",
        action="store_true",
        help="Use threads instead of processes (handy if process forking is slow).",
    )
    parser.add_argument(
        "--backend",
        choices=["process", "thread", "auto"],
        default="process",
        help="Backend to use for scoring; process by default, auto benchmarks a sample to pick process or thread.",
    )
    parser.add_argument(
        "--benchmark-rows",
        type=int,
        default=200,
        help="Number of Annex F rows to benchmark when backend=auto.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Batch size per worker for process pool execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_parallel_config()
    user_workers = _flag_passed("workers")
    user_chunk = _flag_passed("chunk-size")
    user_backend = _flag_passed("backend") or _flag_passed("use-threads")
    backend = args.backend
    workers = args.workers
    chunk_size = args.chunk_size
    if args.use_threads:
        backend = "thread"
    if config:
        if not user_backend:
            backend = config.get("backend", backend)
        if not user_workers and config.get("workers"):
            try:
                workers = int(config["workers"])
            except Exception:
                pass
    backend = str(backend).lower()
    annex_lex_path = DRUGS_DIR / "annex_f_lexicon.csv"
    annex_raw_path = DRUGS_DIR / "annex_f.csv"
    drugbank_path = DRUGS_DIR / "drugbank_generics_master.csv"
    pnf_path = DRUGS_DIR / "pnf_lexicon.csv"
    fda_brand_path = DRUGS_DIR / "fda_drug_2025-11-12.csv"
    mixture_path = DRUGS_DIR / "drugbank_mixtures_master.csv"

    # Brand swapping is disabled for now; keep patterns empty.
    brand_patterns: list[tuple[re.Pattern, str]] = []

    def _load_annex_source() -> pd.DataFrame:
        df = _read_table(annex_raw_path, required=False)
        if not df.empty:
            return df
        return _read_table(annex_lex_path, required=True)

    annex_source_df = _run_with_spinner("Load Annex F source", _load_annex_source)

    def _normalize_annex() -> pd.DataFrame:
        annex_df = _normalize_annex_df(annex_source_df, brand_patterns)
        _write_csv_and_parquet(annex_df, annex_lex_path)
        return annex_df

    annex_df = _run_with_spinner("Normalize Annex F descriptions", _normalize_annex)

    mixture_df = _run_with_spinner("Load DrugBank mixtures", lambda: _read_table(mixture_path, required=False))
    pnf_df = _run_with_spinner("Load PNF lexicon", lambda: _read_table(pnf_path, required=True))
    drugbank_df = _run_with_spinner("Load DrugBank generics", lambda: _read_table(drugbank_path, required=True))

    generic_phrases = _run_with_spinner("Build generic phrase list", lambda: _build_generic_phrases(drugbank_df))
    generic_automaton = _run_with_spinner(
        "Build generic automaton", lambda: _build_aho_automaton(generic_phrases) if generic_phrases else None
    )
    generic_atc_map = _run_with_spinner(
        "Index generic phrases to ATC codes", lambda: _build_generic_to_atc_map(drugbank_df)
    )
    mixture_lookup = _run_with_spinner(
        "Index mixture components", lambda: _build_mixture_lookup(mixture_df) if not mixture_df.empty else {}
    )

    reference_rows = _run_with_spinner(
        "Assemble reference rows", lambda: build_reference_rows(pnf_df, drugbank_df)
    )
    reference_index = _run_with_spinner("Build reference token index", lambda: _build_reference_index(reference_rows))
    drugbank_refs_by_id = _run_with_spinner(
        "Index DrugBank references by id", lambda: _group_drugbank_refs_by_id(reference_rows)
    )
    generic_to_drugbank = _run_with_spinner(
        "Build generic to DrugBank ID lookup", lambda: _build_generic_to_drugbank_id(drugbank_df)
    )

    annex_records = _run_with_spinner(
        "Prepare Annex F records",
        lambda: _precompute_annex_records(annex_df, brand_patterns, generic_phrases, generic_atc_map),
    )

    run_autotune = False
    if run_autotune:
        workers, _chunk_autotune, backend = _run_with_spinner(
            "Autotune parallelism",
            lambda: _autotune_parallelism(
                annex_records,
                reference_rows,
                reference_index,
                brand_patterns,
                generic_automaton,
                generic_phrases,
                generic_atc_map,
                mixture_lookup,
                drugbank_refs_by_id,
            ),
        )
        _save_parallel_config({"backend": backend, "workers": str(workers)})

    if not user_chunk:
        chunk_size = max(1, math.ceil(len(annex_records) / max(1, workers)))

    use_threads = args.use_threads or backend == "thread"
    if backend == "auto":
        sample_size = min(len(annex_records), max(10, args.benchmark_rows))
        sample = annex_records[:sample_size]

        def _time_backend(thread_flag: bool) -> float:
            start = time.perf_counter()
            _ = match_annex_with_atc(
                sample,
                reference_rows,
                reference_index,
                brand_patterns,
                None if thread_flag else generic_automaton,
                generic_phrases,
                generic_atc_map,
                mixture_lookup,
                drugbank_refs_by_id,
                max_workers=workers,
                chunk_size=chunk_size,
                use_threads=thread_flag,
            )
            return time.perf_counter() - start

        def _benchmark_backends() -> tuple[float, float]:
            return _time_backend(True), _time_backend(False)

        t_thread, t_process = _run_with_spinner("Benchmark process vs thread", _benchmark_backends)
        use_threads = t_thread <= t_process
        backend = "thread" if use_threads else "process"
        print(f"[auto-backend] thread={t_thread:.2f}s, process={t_process:.2f}s -> using {'threads' if use_threads else 'processes'}")

    match_df, tie_df, unresolved_df = _run_with_spinner(
        "Match Annex F against references",
        lambda: match_annex_with_atc(
            annex_records,
            reference_rows,
            reference_index,
            brand_patterns,
            None if use_threads else generic_automaton,
            generic_phrases,
            generic_atc_map,
            mixture_lookup,
            drugbank_refs_by_id,
            generic_to_drugbank,
            max_workers=workers,
            chunk_size=chunk_size,
            use_threads=use_threads or backend == "thread",
        ),
    )

    OUTPUTS_DRUGS_DIR.mkdir(parents=True, exist_ok=True)
    match_path = OUTPUTS_DRUGS_DIR / "annex_f_with_atc.csv"
    ties_path = OUTPUTS_DRUGS_DIR / "annex_f_atc_ties.csv"
    unresolved_path = OUTPUTS_DRUGS_DIR / "annex_f_atc_unresolved.csv"

    def reorder(df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "Drug Code",
            "Drug Description",
            "fuzzy_basis",
            "matched_reference_raw",
            "matched_source",
            "matched_generic_name",
            "matched_lexicon",
            "match_count",
            "matched_secondary_lexicon",
            "secondary_match_count",
            "atc_code",
            "drugbank_id",
            "primary_matching_tokens",
            "secondary_matching_tokens",
        ]
        existing = [c for c in cols if c in df.columns]
        remaining = [c for c in df.columns if c not in existing]
        return df.loc[:, existing + remaining]

    def _write_outputs() -> None:
        _write_csv_and_parquet(reorder(match_df), match_path)
        _write_csv_and_parquet(reorder(tie_df), ties_path)
        _write_csv_and_parquet(reorder(unresolved_df), unresolved_path)

    _run_with_spinner("Write Annex F outputs", _write_outputs)

    print(f"Annex F ATC matches saved to {match_path}")
    print(f"Acceptable ties saved to {ties_path}")
    print(f"Unresolved ties saved to {unresolved_path}")


if __name__ == "__main__":
    main()
