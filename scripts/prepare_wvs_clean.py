#!/usr/bin/env python3
"""
Prepare a cleaned WVS Wave 7 CSV for AutoKD (two-phase + RAG).

Spec: minimal high-impact cleaning for statistically valid, LLM-safe, RAG-ready,
cross-country comparable, stable automated experimentation.

Outputs:
  - data/processed/wvs/wvs7_clean.csv
  - data/processed/wvs/wvs7_variable_catalog.json (if codebook available)

Input (default):
  - data/wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv

DO NOT: one-hot encode, scale normalize, reverse code, impute, drop rows, convert all to categorical.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_CSV = PROJECT_ROOT / "data" / "wvs" / "WVS_Cross-National_Wave_7_csv_v6_0.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "wvs"
OUT_CSV = OUT_DIR / "wvs7_clean.csv"
OUT_CATALOG_JSON = OUT_DIR / "wvs7_variable_catalog.json"

# Columns to DROP (interviewer IDs, routing, admin, technical) — keep analytic only
DROP_COLUMNS: Set[str] = {
    "version", "doi", "A_WAVE", "A_STUDY", "C_COW_NUM", "C_COW_ALPHA",
    "D_INTERVIEW", "S007", "J_INTDATE", "K_TIME_START", "K_TIME_END", "K_DURATION",
    "Q_MODE", "O1_LONGITUDE", "O2_LATITUDE", "L_INTERVIEWER_NUMBER",
    "S_INTLANGUAGE", "LNGE_ISO", "E_RESPINT", "F_INTPRIVACY", "E1_LITERACY",
    "S018", "S025",
    "N_REGION_ISO", "N_REGION_WVS", "N_REGION_NUTS2", "N_REG_NUTS1", "N_TOWN",
}

# Keep: Q*, B_COUNTRY*, A_YEAR, FW_*, W_WEIGHT, PWGHT, X* (demographics), G_*, H_* (settlement/urb), survey_id, country, country_code, year, sampling_weight
KEEP_PATTERNS = (
    r"^Q\d",           # Q variables
    r"^B_COUNTRY", r"^A_YEAR", r"^FW_", r"^W_WEIGHT", r"^PWGHT",
    r"^X\d",           # demographics
    r"^G_", r"^H_",   # settlement / urb
)

MISSING_THRESHOLD = 0.95  # drop column if > 95% missing
MIN_UNIQUE = 2            # drop column if only 1 unique value


def _is_analytic_column(name: str) -> bool:
    if name in DROP_COLUMNS:
        return False
    for pat in KEEP_PATTERNS:
        if re.match(pat, name):
            return True
    return False


def _convert_negative_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace all values < 0 with NaN (CRITICAL for valid statistics)."""
    for col in df.columns:
        if df[col].dtype.kind in "iufb":  # numeric
            df[col] = df[col].mask(df[col] < 0, np.nan)
        else:
            # object/string: try coerce and replace negative numeric strings
            try:
                s = pd.to_numeric(df[col], errors="coerce")
                df[col] = s.mask(s < 0, np.nan)
            except Exception:
                pass
    return df


def _standardize_country_year(df: pd.DataFrame) -> pd.DataFrame:
    """Create country, country_code, year, survey_id."""
    # B_COUNTRY_ALPHA is typically 3-letter (ISO-like)
    if "B_COUNTRY_ALPHA" not in df.columns:
        return df
    df = df.copy()
    df["country_code"] = df["B_COUNTRY_ALPHA"].astype(str).str.strip()
    df["country"] = df["country_code"]  # keep code as name if no mapping; optional: pycountry lookup
    if "A_YEAR" in df.columns:
        df["year"] = pd.to_numeric(df["A_YEAR"], errors="coerce")
    elif "FW_START" in df.columns:
        df["year"] = pd.to_numeric(df["FW_START"], errors="coerce")
    else:
        df["year"] = np.nan
    df["survey_id"] = df["country_code"].astype(str) + "_" + df["year"].astype("Int64").astype(str)
    return df


def _rename_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve weight; rename primary weight column to sampling_weight (W_WEIGHT preferred, else PWGHT)."""
    if "W_WEIGHT" in df.columns:
        df = df.rename(columns={"W_WEIGHT": "sampling_weight"})
    elif "PWGHT" in df.columns:
        df = df.rename(columns={"PWGHT": "sampling_weight"})
    return df


def _drop_low_information_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with > 95% missing OR only 1 unique value."""
    n = len(df)
    to_drop: List[str] = []
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing / max(n, 1) > MISSING_THRESHOLD:
            to_drop.append(col)
            continue
        nuniq = df[col].nunique(dropna=True)
        if nuniq < MIN_UNIQUE:
            to_drop.append(col)
    return df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")


def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal/continuous stay numeric; do not one-hot or convert everything to categorical."""
    # Keep as-is; only fix obvious object columns that are numeric
    for col in df.columns:
        if df[col].dtype == object:
            try:
                s = pd.to_numeric(df[col], errors="coerce")
                if s.notna().sum() > len(df) * 0.5:
                    df[col] = s
            except Exception:
                pass
    return df


def clean_wvs_csv(
    raw_path: Path,
    out_path: Path,
    *,
    drop_analytic_only: bool = True,
) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns the cleaned DataFrame.
    Does NOT drop rows (no df.dropna() on rows).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)

    # 1) Convert all special missing codes (< 0) → NaN
    print("  Converting negative codes to NaN...")
    df = _convert_negative_to_nan(df)
    df.replace({"": pd.NA, " ": pd.NA}, inplace=True)

    # 2) Keep only analytic variables
    if drop_analytic_only:
        keep_cols = [c for c in df.columns if _is_analytic_column(c)]
        dropped = set(df.columns) - set(keep_cols)
        df = df[keep_cols].copy()
        print(f"  Kept {len(keep_cols)} analytic columns, dropped {len(dropped)} technical/admin.")

    # 3) Standardize country + year
    print("  Standardizing country, year, survey_id...")
    df = _standardize_country_year(df)

    # 4) Preserve weight → sampling_weight
    df = _rename_weight(df)

    # 5) Data types (ordinal/continuous stay numeric; no one-hot)
    df = _ensure_types(df)

    # 6) Drop low-information columns (>95% missing or 1 unique)
    print("  Dropping low-information columns...")
    before = len(df.columns)
    df = _drop_low_information_columns(df)
    print(f"  Dropped {before - len(df.columns)} low-info columns.")

    # 7) Do NOT drop rows
    print(f"  Rows unchanged: {len(df)}")

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


def build_variable_catalog_json(
    codebook_txt_path: Path,
    out_json_path: Path,
    cleaned_columns: List[str],
) -> None:
    """
    Build wvs7_variable_catalog.json from codebook (variable, label, type, min, max, construct, description).
    Uses existing codebook parser; maps to the required JSON schema.
    """
    try:
        from core.variable_catalog import parse_wvs_codebook_txt
    except ImportError:
        print("  [SKIP] Variable catalog JSON not built (core.variable_catalog not available).")
        return
    if not codebook_txt_path.exists():
        print(f"  [SKIP] Codebook not found: {codebook_txt_path}")
        return

    cards = parse_wvs_codebook_txt(codebook_txt_path)
    catalog: List[Dict[str, Any]] = []
    for c in cards:
        if c.var_name not in cleaned_columns:
            continue
        # Infer min/max from options if available
        opts = c.options or []
        codes = [int(o.get("code", 0)) for o in opts if isinstance(o.get("code"), (int, float))]
        vmin = min(codes) if codes else None
        vmax = max(codes) if codes else None
        catalog.append({
            "variable": c.var_name,
            "label": (c.label or "").strip(),
            "type": "ordinal" if codes else "nominal",
            "min": vmin,
            "max": vmax,
            "construct": (c.module or "").strip()[:80],
            "description": (c.question or c.label or "").strip()[:500],
        })
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_json_path} ({len(catalog)} variables)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean WVS Wave 7 CSV for AutoKD")
    ap.add_argument("--input", type=str, default=str(RAW_CSV), help="Raw WVS CSV path")
    ap.add_argument("--output", type=str, default=str(OUT_CSV), help="Output cleaned CSV path")
    ap.add_argument("--catalog-output", type=str, default=str(OUT_CATALOG_JSON), help="Output variable catalog JSON path")
    ap.add_argument("--codebook-txt", type=str, default="", help="Codebook txt (pdftotext output); if empty, catalog not built")
    ap.add_argument("--keep-all-cols", action="store_true", help="Do not drop non-analytic columns (only missing/low-info)")
    args = ap.parse_args()

    raw_path = Path(args.input)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_path}")

    out_path = Path(args.output)
    df = clean_wvs_csv(
        raw_path,
        out_path,
        drop_analytic_only=not args.keep_all_cols,
    )

    codebook_txt = Path(args.codebook_txt) if args.codebook_txt else (PROJECT_ROOT / "data" / "wvs" / "wvs7_codebook_v6.txt")
    build_variable_catalog_json(codebook_txt, Path(args.catalog_output), list(df.columns))


if __name__ == "__main__":
    main()
