#!/usr/bin/env python3
"""
Build a searchable WVS variable catalog from the official codebook.

Inputs (defaults):
  - data/wvs/F00011055-WVS7_Codebook_Variables_report_V6.0.pdf
  - data/wvs/wvs7_codebook_v6.txt (generated via pdftotext if missing)

Outputs (default):
  - data/processed/wvs/wvs7_variable_catalog.jsonl

This script is safe to run from any working directory.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
import sys
from typing import Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Allow running from any working directory:
# make `core.*` importable by adding the repository root to sys.path.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.variable_catalog import (
    VariableCard,
    parse_wvs_codebook_txt_for_columns,
    read_csv_columns,
    save_variable_cards_jsonl,
)

DEFAULT_PDF = PROJECT_ROOT / "data" / "wvs" / "F00011055-WVS7_Codebook_Variables_report_V6.0.pdf"
DEFAULT_TXT = PROJECT_ROOT / "data" / "wvs" / "wvs7_codebook_v6.txt"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "wvs" / "wvs7_variable_catalog.jsonl"
DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "wvs" / "WVS_Cross-National_Wave_7_csv_v6_0.csv"
DEFAULT_COVERAGE_CSV = PROJECT_ROOT / "data" / "processed" / "wvs" / "wvs7_clean.csv"
_OPTION_LIKE_LABEL_RE = re.compile(r"^\s*-?\d+\s*(?:[\-–\.\):]\s*){1,4}\S+")


def _ensure_txt_from_pdf(pdf_path: Path, txt_path: Path) -> None:
    if txt_path.exists():
        return
    if not pdf_path.exists():
        raise FileNotFoundError(f"Codebook PDF not found: {pdf_path}")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["pdftotext", str(pdf_path), str(txt_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "pdftotext not found. Install poppler-utils (Ubuntu/Debian) or poppler (conda)."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"pdftotext failed (exit={e.returncode}). stderr: {e.stderr[:500]}"
        ) from e


def _lint_and_filter_cards(
    cards: Sequence[VariableCard],
    *,
    drop_fully_empty: bool,
    strict_lint: bool,
) -> Tuple[list[VariableCard], dict]:
    fully_empty_vars = []
    option_like_label_vars = []
    out: list[VariableCard] = []

    for c in cards:
        is_empty = (
            not (c.label or "").strip()
            and not (c.question or "").strip()
            and not (c.options or [])
            and not (c.missing_codes or {})
        )
        if is_empty:
            fully_empty_vars.append(c.var_name)
            if drop_fully_empty:
                continue

        if _OPTION_LIKE_LABEL_RE.match((c.label or "").strip()):
            option_like_label_vars.append(c.var_name)

        out.append(c)

    if strict_lint and (fully_empty_vars or option_like_label_vars):
        raise RuntimeError(
            "Catalog lint failed: "
            f"fully_empty={len(fully_empty_vars)}, option_like_label={len(option_like_label_vars)}"
        )

    stats = {
        "fully_empty_count": len(fully_empty_vars),
        "fully_empty_examples": fully_empty_vars[:12],
        "option_like_label_count": len(option_like_label_vars),
        "option_like_label_examples": option_like_label_vars[:12],
        "dropped_count": len(cards) - len(out),
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=str, default=str(DEFAULT_PDF))
    ap.add_argument("--txt", type=str, default=str(DEFAULT_TXT))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument(
        "--csv-header",
        type=str,
        default=str(DEFAULT_RAW_CSV),
        help="CSV path used for header intersection filtering (set empty string to disable).",
    )
    ap.add_argument(
        "--coverage-csv",
        type=str,
        default=str(DEFAULT_COVERAGE_CSV),
        help="Report columns in this CSV header missing from the final catalog (set empty string to disable).",
    )
    ap.add_argument(
        "--fail-on-missing-coverage",
        action="store_true",
        help="Fail build if coverage CSV has columns absent from final catalog.",
    )
    ap.add_argument(
        "--strict-lint",
        action="store_true",
        help="Fail if lint finds fully empty cards or option-like labels.",
    )
    ap.add_argument(
        "--keep-fully-empty",
        action="store_true",
        help="Keep fully empty cards instead of dropping them.",
    )
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    txt_path = Path(args.txt)
    out_path = Path(args.out)
    csv_header_path = Path(args.csv_header) if str(args.csv_header).strip() else None
    coverage_csv_path = Path(args.coverage_csv) if str(args.coverage_csv).strip() else None

    _ensure_txt_from_pdf(pdf_path, txt_path)
    if csv_header_path is not None:
        cards = parse_wvs_codebook_txt_for_columns(txt_path, csv_header_path)
    else:
        # Keep behavior explicit if caller disables CSV-based filtering.
        from core.variable_catalog import parse_wvs_codebook_txt
        cards = parse_wvs_codebook_txt(txt_path)
    cards, lint_stats = _lint_and_filter_cards(
        cards,
        drop_fully_empty=not args.keep_fully_empty,
        strict_lint=args.strict_lint,
    )
    missing_coverage: list[str] = []
    if coverage_csv_path is not None and coverage_csv_path.exists():
        coverage_cols = read_csv_columns(coverage_csv_path)
        card_vars = {c.var_name for c in cards}
        missing_coverage = [c for c in coverage_cols if c and c not in card_vars]
        if args.fail_on_missing_coverage and missing_coverage:
            raise RuntimeError(
                f"Coverage check failed: {len(missing_coverage)} columns missing from catalog. "
                f"Examples: {missing_coverage[:12]}"
            )
    save_variable_cards_jsonl(cards, out_path)

    modules = sorted({c.module for c in cards if c.module})
    print("[OK] WVS variable catalog built.")
    print(f"   txt: {txt_path}")
    print(f"   csv_header: {csv_header_path if csv_header_path is not None else 'disabled'}")
    print(f"   coverage_csv: {coverage_csv_path if coverage_csv_path is not None else 'disabled'}")
    print(f"   out: {out_path}")
    print(f"   variables: {len(cards)}")
    print(f"   modules: {len(modules)}")
    print(
        "   lint: "
        f"fully_empty={lint_stats['fully_empty_count']} "
        f"(dropped={lint_stats['dropped_count']}), "
        f"option_like_label={lint_stats['option_like_label_count']}"
    )
    if lint_stats["fully_empty_examples"]:
        print(f"   lint empty examples: {lint_stats['fully_empty_examples']}")
    if lint_stats["option_like_label_examples"]:
        print(f"   lint option-like label examples: {lint_stats['option_like_label_examples']}")
    print(f"   coverage_missing: {len(missing_coverage)}")
    if missing_coverage:
        print(f"   coverage_missing_examples: {missing_coverage[:12]}")
    if modules:
        print(f"   example module: {modules[0]}")


if __name__ == "__main__":
    main()

