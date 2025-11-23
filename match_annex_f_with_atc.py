"""Match Annex F entries to ATC codes using PNF, DrugBank, and WHO lexicons."""

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
            mg_val = _convert_to_mg(combined_match.group(1), combined_match.group(2))
            if mg_val:
                normalized.append(mg_val)
                normalized.append("MG")
            i += 1
            continue

        if _is_numeric_token(tok_upper) and i + 1 < len(expanded):
            next_clean = expanded[i + 1].replace(",", "").strip("()").strip().upper()
            if next_clean in _WEIGHT_UNIT_FACTORS:
                mg_val = _convert_to_mg(tok_upper, next_clean)
                if mg_val:
                    normalized.append(mg_val)
                    normalized.append("MG")
                    i += 2
                    continue

        if _is_numeric_token(tok_upper):
            tok_upper = _format_number_text(tok_upper)

        tok_upper = FORM_CANON.get(tok_upper, tok_upper)
        tok_upper = ROUTE_CANON.get(tok_upper, tok_upper)

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


def _categorize_tokens(tokens: List[str]) -> dict[str, Counter]:
    cat_counts: dict[str, Counter] = {
        CATEGORY_GENERIC: Counter(),
        CATEGORY_SALT: Counter(),
        CATEGORY_DOSE: Counter(),
        CATEGORY_FORM: Counter(),
        CATEGORY_ROUTE: Counter(),
        CATEGORY_OTHER: Counter(),
    }
    for tok in tokens:
        cat = _classify_token(tok)
        if cat not in cat_counts:
            cat = CATEGORY_OTHER
        cat_counts[cat][tok] += 1
    return cat_counts


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
    return sorted(phrases)


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
    annex_df: pd.DataFrame, brand_patterns: list[tuple[re.Pattern, str]], generic_phrases: list[str]
) -> list[dict]:
    records = []
    generic_automaton = _build_aho_automaton(generic_phrases) if generic_phrases else None
    for row in annex_df.to_dict(orient="records"):
        description = _apply_brand_swaps(_as_str_or_empty(row.get("Drug Description")), brand_patterns)
        base_tokens = split_with_parentheses(description)
        tokens_primary = _normalize_tokens(base_tokens, drop_stopwords=True)
        tokens_secondary = _normalize_tokens(base_tokens, drop_stopwords=False)
        row["fuzzy_basis"] = "|".join(tokens_secondary)
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
                "generic_hits": _aho_find(_normalize_phrase(description), generic_automaton)
                if generic_automaton
                else set(),
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

    dose_piece = None
    for key in ("dose_norm", "raw_dose", "strength_mg", "strength", "pct", "per_val"):
        text = _as_str_or_empty(raw_ref.get(key))
        if text:
            dose_piece = text
            break

    unit_piece = None
    for key in ("unit", "per_unit"):
        text = _as_str_or_empty(raw_ref.get(key))
        if text:
            unit_piece = text
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
    source_priority = {"pnf": 0, "drugbank": 1, "who": 2}
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
                "atc_code": _maybe_none(row.get("atc_code")),
                "raw_reference_row": raw_reference_row,
            }
        )
    return rows


def build_who_reference(df: pd.DataFrame) -> list[dict]:
    rows = []
    for row in df.to_dict(orient="records"):
        primary_tokens = _normalize_tokens(split_with_parentheses(row.get("atc_name")), drop_stopwords=True)

        secondary_tokens: List[str] = []
        for col in ("adm_r", "uom"):
            secondary_tokens.extend(split_with_parentheses(row.get(col)))
        secondary_tokens = _normalize_tokens(secondary_tokens, drop_stopwords=False)

        rows.append(
            {
                "source": "who",
                "id": _maybe_none(row.get("atc_code")),
                "name": _maybe_none(row.get("atc_name")),
                "lexicon": "|".join(primary_tokens),
                "lexicon_secondary": "|".join(secondary_tokens),
                "primary_tokens": primary_tokens,
                "secondary_tokens": secondary_tokens,
                "primary_cat_counts": _categorize_tokens(primary_tokens),
                "secondary_cat_counts": _categorize_tokens(secondary_tokens),
                "atc_code": _maybe_none(row.get("atc_code")),
                "raw_reference_row": dict(row),
            }
        )
    return rows


def build_reference_rows(
    pnf_df: pd.DataFrame, drugbank_df: pd.DataFrame, who_df: pd.DataFrame
) -> list[dict]:
    refs: list[dict] = []
    refs.extend(build_pnf_reference(pnf_df))
    refs.extend(build_drugbank_reference(drugbank_df))
    refs.extend(build_who_reference(who_df))
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
        "primary_matching_tokens": None,
        "secondary_matching_tokens": None,
    }


def _build_match_record(
    annex_row: dict, ref: dict, primary_score: int, secondary_score: int
) -> dict:
    primary_overlap = ref.get("primary_overlap") or []
    secondary_overlap = ref.get("secondary_overlap") or []
    raw_ref = ref.get("raw_reference_row") or {}
    raw_display = _build_reference_display(raw_ref)
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
        "atc_code": ref.get("atc_code"),
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
                        "atc_code": None,
                        "raw_reference_row": mix,
                    }
                )

    best_primary = -10**9
    best_primary_records: list[dict] = []
    for ref in candidate_refs:
        primary_overlap = _ordered_overlap(
            fuzzy_tokens_primary, fuzzy_counts_primary, ref.get("primary_tokens", ())
        )
        ref_primary_counts = ref.get("primary_cat_counts", {})
        cat_scores = []
        for cat, ref_counts in ref_primary_counts.items():
            match_count = sum((annex_cat_primary.get(cat, Counter()) & ref_counts).values())
            mismatch = max(0, sum(ref_counts.values()) - match_count)
            weight = PRIMARY_WEIGHTS.get(cat, 1)
            cat_scores.append(weight * match_count - mismatch)
        primary_score = sum(cat_scores)
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
    for rec in best_primary_records:
        secondary_overlap = _ordered_overlap(
            fuzzy_tokens_secondary, fuzzy_counts_secondary, rec.get("secondary_tokens", ())
        )
        ref_secondary_counts = rec.get("secondary_cat_counts", {})
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
            cat_scores.append(weight * match_count + bonus - mismatch)
        secondary_score = sum(cat_scores)
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
            _build_match_record(annex_row, winner, best_primary, best_secondary)
        )
        return matched_rows[0], tie_rows, unresolved_rows

    sorted_finalists = sorted(finalists, key=_reference_sort_key)
    atc_set = {_as_str_or_empty(rec.get("atc_code")) for rec in sorted_finalists}
    if len(atc_set) == 1:
        def _form_route_score(rec: dict) -> tuple[int, int]:
            sec_counts = rec.get("secondary_cat_counts", {})
            form_match = sum(
                (annex_cat_secondary.get(CATEGORY_FORM, Counter()) & sec_counts.get(CATEGORY_FORM, Counter())).values()
            )
            route_match = sum(
                (annex_cat_secondary.get(CATEGORY_ROUTE, Counter()) & sec_counts.get(CATEGORY_ROUTE, Counter())).values()
            )
            return (route_match, form_match)

        sorted_finalists = sorted(
            sorted_finalists,
            key=lambda rec: (
                -_form_route_score(rec)[0],
                -_form_route_score(rec)[1],
                _reference_sort_key(rec),
            ),
        )
        winner = sorted_finalists[0]
        matched_rows.append(
            _build_match_record(annex_row, winner, best_primary, best_secondary)
        )
        for rec in sorted_finalists:
            tie_rows.append(
                _build_match_record(
                    annex_row,
                    rec,
                    rec.get("primary_score", best_primary),
                    rec.get("secondary_score", best_secondary),
                )
            )
        return matched_rows[0], tie_rows, unresolved_rows

    matched_rows.append(_empty_match_record(annex_row))
    for rec in sorted_finalists:
        unresolved_rows.append(
            _build_match_record(
                annex_row,
                rec,
                rec.get("primary_score", best_primary),
                rec.get("secondary_score", best_secondary),
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
) -> None:
    global _REFERENCE_ROWS_SHARED
    global _REFERENCE_INDEX_SHARED
    global _BRAND_PATTERNS_SHARED
    global _GENERIC_PHRASES_SHARED
    global _GENERIC_AUTOMATON_SHARED
    global _MIXTURE_LOOKUP_SHARED
    global _DRUGBANK_BY_ID_SHARED
    _REFERENCE_ROWS_SHARED = reference_rows
    _REFERENCE_INDEX_SHARED = reference_index
    _BRAND_PATTERNS_SHARED = brand_patterns
    _GENERIC_PHRASES_SHARED = generic_phrases
    # Annex rows already carry generic_hits; avoid building automaton per worker.
    _GENERIC_AUTOMATON_SHARED = None
    _MIXTURE_LOOKUP_SHARED = mixture_lookup
    _DRUGBANK_BY_ID_SHARED = drugbank_refs_by_id


def _process_annex_row_worker(annex_row: dict) -> tuple[dict, list[dict], list[dict]]:
    reference_rows = _REFERENCE_ROWS_SHARED or []
    reference_index = _REFERENCE_INDEX_SHARED or {}
    brand_patterns = _BRAND_PATTERNS_SHARED or []
    generic_automaton = _GENERIC_AUTOMATON_SHARED
    mixture_lookup = _MIXTURE_LOOKUP_SHARED or {}
    drugbank_refs_by_id = _DRUGBANK_BY_ID_SHARED or {}
    return _score_annex_row(
        annex_row,
        reference_rows,
        reference_index,
        brand_patterns,
        generic_automaton,
        mixture_lookup,
        drugbank_refs_by_id,
    )


def match_annex_with_atc(
    annex_df: pd.DataFrame,
    reference_rows: list[dict],
    reference_index: dict[str, list[int]],
    brand_patterns: list[tuple[re.Pattern, str]],
    generic_automaton,
    generic_phrases: list[str],
    mixture_lookup: dict[str, list[dict]],
    drugbank_refs_by_id: dict[str, list[dict]],
    max_workers: int | None = None,
    chunk_size: int = 172,
    use_threads: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annex_records = (
        annex_df
        if isinstance(annex_df, list)
        else _precompute_annex_records(annex_df, brand_patterns, generic_phrases)
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
            initargs=(reference_rows, reference_index, brand_patterns, generic_phrases, mixture_lookup, drugbank_refs_by_id),
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
    who_path = DRUGS_DIR / "who_atc_2025-11-20.csv"
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
    who_df = _run_with_spinner("Load WHO ATC export", lambda: _read_table(who_path, required=True))
    drugbank_df = _run_with_spinner("Load DrugBank generics", lambda: _read_table(drugbank_path, required=True))

    generic_phrases = _run_with_spinner("Build generic phrase list", lambda: _build_generic_phrases(drugbank_df))
    generic_automaton = _run_with_spinner(
        "Build generic automaton", lambda: _build_aho_automaton(generic_phrases) if generic_phrases else None
    )
    mixture_lookup = _run_with_spinner(
        "Index mixture components", lambda: _build_mixture_lookup(mixture_df) if not mixture_df.empty else {}
    )

    reference_rows = _run_with_spinner(
        "Assemble reference rows", lambda: build_reference_rows(pnf_df, drugbank_df, who_df)
    )
    reference_index = _run_with_spinner("Build reference token index", lambda: _build_reference_index(reference_rows))
    drugbank_refs_by_id = _run_with_spinner(
        "Index DrugBank references by id", lambda: _group_drugbank_refs_by_id(reference_rows)
    )

    annex_records = _run_with_spinner(
        "Prepare Annex F records", lambda: _precompute_annex_records(annex_df, brand_patterns, generic_phrases)
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
            mixture_lookup,
            drugbank_refs_by_id,
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
