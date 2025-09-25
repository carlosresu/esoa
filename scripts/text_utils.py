# ===============================
# File: scripts/text_utils.py
# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import unicodedata
from typing import Optional


def _normalize_text_basic(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _base_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.split(r",| incl\.| including ", name, maxsplit=1)[0]
    return re.sub(r"\s+", " ", name).strip()


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^\w%/+\.\- ]+", " ", s)
    s = s.replace("microgram", "mcg").replace("μg", "mcg").replace("µg", "mcg")
    s = s.replace("cc", "ml").replace("milli litre", "ml").replace("milliliter", "ml")
    s = s.replace("gm", "g").replace("gms", "g").replace("milligram", "mg")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_compact(s: str) -> str:
    return re.sub(r"[ \-]", "", normalize_text(s))


def slug_id(name: str) -> str:
    base = normalize_text(str(name))
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")


def clean_atc(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\u00a0", " ").strip()


def safe_to_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace(",", ".").strip()
        return float(x)
    except Exception:
        return None