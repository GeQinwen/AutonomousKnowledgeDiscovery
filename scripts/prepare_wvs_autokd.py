#!/usr/bin/env python3
"""
Prepare an AutoKD-friendly WVS dataset CSV.

Input (default):
  data/processed/wvs/wvs7_clean.csv

Output (default):
  data/wvs7_autokd.csv

Why this exists:
- WVS is extremely wide. AutoKD's schema prompt becomes huge and hypothesis
  generation degrades. We "thin" columns based on missingness and keep a
  manageable number of columns.
- We also coerce low-cardinality integer-coded columns to string so they behave
  like categorical variables (avoids treating codes as continuous magnitudes).

This script is safe to run from any working directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "wvs" / "wvs7_clean.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "wvs7_autokd.csv"


def _parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _compute_missingness(
    input_csv: Path,
    chunksize: int,
) -> Tuple[int, pd.Series]:
    """Return (total_rows, missing_counts_by_col)."""
    header = pd.read_csv(input_csv, nrows=0)
    cols = list(header.columns)
    missing_counts = pd.Series(0, index=cols, dtype="int64")
    total_rows = 0

    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0).astype("int64")

    return total_rows, missing_counts


def _is_integer_like(series: pd.Series) -> bool:
    """True if non-null values are all integer-like (within a tolerance)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return False
    return bool(np.all(np.isclose(s % 1, 0)))


def prepare_wvs_for_autokd(
    input_csv: Path,
    output_csv: Path,
    *,
    missing_threshold: float = 0.85,
    max_cols: int = 0,  # 0 = no cap (keep all candidates that pass missingness)
    sample_rows: int = 5000,
    chunksize: int = 20000,
    categorical_max_unique: int = 20,
    treat_low_card_int_as_category: bool = True,
    always_keep: List[str] | None = None,
) -> Dict:
    if always_keep is None:
        always_keep = []

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # 1) Compute missingness accurately (chunked, wide-friendly)
    total_rows, missing_counts = _compute_missingness(input_csv, chunksize=chunksize)
    missing_rate = (missing_counts / max(total_rows, 1)).astype(float)

    # 2) Get a small sample for cheap heuristics (nunique, integer-like)
    df_sample = pd.read_csv(input_csv, nrows=sample_rows, low_memory=False)
    sample_nunique = df_sample.nunique(dropna=True)

    # 3) Decide required columns
    # WVS-ish keys: keep if present
    default_keep = [
        "survey_id",
        "B_COUNTRY_ALPHA",
        "B_COUNTRY",
        "A_YEAR",
        "FW_START",
        "FW_END",
        "S017",  # sometimes a year field depending on release
    ]
    required = []
    for c in (default_keep + always_keep):
        if c in missing_rate.index and c not in required:
            required.append(c)

    # 4) Drop extreme-missing and constant columns (except required)
    cols_all = list(missing_rate.index)
    dropped_missing = [c for c in cols_all if c not in required and missing_rate.get(c, 1.0) > missing_threshold]
    dropped_constant = [c for c in cols_all if c not in required and sample_nunique.get(c, 0) <= 1]
    dropped = set(dropped_missing) | set(dropped_constant)

    candidates = [c for c in cols_all if c not in required and c not in dropped]

    # 5) Optionally cap total columns (required + best candidates). max_cols <= 0 means no cap.
    if max_cols > 0 and len(candidates) > max_cols:
        # Prefer low-missing columns first; break ties by higher (sample) cardinality
        def rank_key(c: str) -> Tuple[float, float, str]:
            return (float(missing_rate.get(c, 1.0)), -float(sample_nunique.get(c, 0)), c)

        candidates_sorted = sorted(candidates, key=rank_key)
        candidates = candidates_sorted[:max_cols]

    selected_cols = required + [c for c in candidates if c not in required]

    # 6) Load selected columns only
    df = pd.read_csv(input_csv, usecols=selected_cols, low_memory=False)

    # 7) Coerce low-cardinality integer-coded numeric columns to string (categorical)
    coerced_to_string: List[str] = []
    if treat_low_card_int_as_category:
        for col in selected_cols:
            if col in required:
                continue
            if col not in df.columns:
                continue

            nunique = int(sample_nunique.get(col, df[col].nunique(dropna=True)))
            if nunique <= categorical_max_unique and pd.api.types.is_numeric_dtype(df[col]):
                if _is_integer_like(df[col]):
                    # Keep missing as <NA>, preserve integer codes, but ensure categorical behavior downstream
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype("string")
                    coerced_to_string.append(col)

    # 8) Make sure key IDs are strings (prevents accidental numeric treatment)
    for key_col in ["survey_id", "B_COUNTRY_ALPHA", "B_COUNTRY"]:
        if key_col in df.columns:
            df[key_col] = df[key_col].astype("string")

    # 9) Write output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    report = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "total_rows": int(total_rows),
        "input_num_columns": int(len(cols_all)),
        "selected_num_columns": int(len(selected_cols)),
        "selected_columns": selected_cols,
        "required_columns_kept": required,
        "dropped_due_to_missing_threshold": {
            "threshold": float(missing_threshold),
            "count": int(len(dropped_missing)),
        },
        "dropped_due_to_constant_in_sample": {
            "sample_rows": int(min(sample_rows, total_rows)),
            "count": int(len(dropped_constant)),
        },
        "coerced_low_card_int_to_string": {
            "categorical_max_unique": int(categorical_max_unique),
            "count": int(len(coerced_to_string)),
            "columns": coerced_to_string,
        },
    }

    report_path = output_csv.with_suffix(output_csv.suffix + ".prep_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--missing-threshold", type=float, default=0.85)
    parser.add_argument("--max-cols", type=int, default=0, help="Max columns to keep (0 = no cap, keep all that pass missingness)")
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--categorical-max-unique", type=int, default=20)
    parser.add_argument("--no-cast-low-card-int", action="store_true")
    parser.add_argument(
        "--always-keep",
        type=str,
        default="",
        help="Comma-separated list of columns to always keep (if present).",
    )

    args = parser.parse_args()

    report = prepare_wvs_for_autokd(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        missing_threshold=float(args.missing_threshold),
        max_cols=int(args.max_cols),
        sample_rows=int(args.sample_rows),
        chunksize=int(args.chunksize),
        categorical_max_unique=int(args.categorical_max_unique),
        treat_low_card_int_as_category=not bool(args.no_cast_low_card_int),
        always_keep=_parse_csv_list(args.always_keep),
    )

    print("[OK] WVS AutoKD dataset prepared.")
    print(f"   Output: {report['output_csv']}")
    print(
        f"   Columns: {report['selected_num_columns']} selected "
        f"(from {report['input_num_columns']})"
    )
    print(
        f"   Dropped (missing>{report['dropped_due_to_missing_threshold']['threshold']:.2f}): "
        f"{report['dropped_due_to_missing_threshold']['count']}"
    )
    print(
        f"   Coerced low-card int -> string: {report['coerced_low_card_int_to_string']['count']}"
    )


if __name__ == "__main__":
    main()

