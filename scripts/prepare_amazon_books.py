"""
Prepare Amazon Books dataset for AutoKD.

Builds a flat CSV from:
  1. Books_5.json.gz       — 5-core reviews
  2. meta_Books.json.gz    — product metadata (title, brand, category, price, ...)

Layers:
  - Review fields: rating, helpful_votes, verified_purchase, has_image, product_format, text, etc.
  - Product metadata: title, brand, price, category, sales_rank, also_buy/view counts (joined on asin)

Output: data/amazon_books_autokd.csv
"""

import argparse
import gzip
import json
import re
import time
from datetime import datetime
from pathlib import Path

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REVIEWS_PATH = DATA_DIR / "Books_5.json.gz"
META_PATH = DATA_DIR / "meta_Books.json.gz"
OUTPUT_PATH = DATA_DIR / "amazon_books_autokd.csv"
REPORT_PATH = DATA_DIR / "amazon_books_autokd.prep_report.json"

CHUNK_SIZE = 500_000


def parse_review(r: dict) -> dict:
    vote = r.get("vote", "0") or "0"
    helpful_votes = int(str(vote).replace(",", "")) if vote else 0

    unix_ts = r.get("unixReviewTime")
    review_year = None
    date_str = None
    if unix_ts:
        try:
            dt = datetime.fromtimestamp(int(unix_ts))
            review_year = dt.year
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, OSError, OverflowError):
            pass

    review_text = " ".join((r.get("reviewText") or "").split())
    summary_text = " ".join((r.get("summary") or "").split())

    style = r.get("style") or {}
    product_format = ""
    if isinstance(style, dict):
        product_format = (
            style.get("Format:", "") or style.get("format", "") or ""
        ).strip()

    return {
        "asin": r.get("asin", ""),
        "reviewerID": r.get("reviewerID", ""),
        "rating": float(r.get("overall", 0)),
        "helpful_votes": helpful_votes,
        "verified_purchase": 1 if r.get("verified", False) else 0,
        "has_image": 1 if r.get("image") else 0,
        "product_format": product_format,
        "year": review_year,
        "date": date_str,
        "timestamp": int(unix_ts) if unix_ts else None,
        "review_text": review_text,
        "summary": summary_text,
    }


def collect_aggregate_stats(reviews_path: Path, max_reviews: int | None = None):
    """First pass over reviews: collect per-reviewer and per-product rating stats.

    Returns two DataFrames (reviewer_agg, product_agg) indexed by ID with
    columns for count, avg_rating, and rating_std.
    """
    print("Pass 1: collecting per-reviewer and per-product rating statistics ...")
    t0 = time.time()
    rev_n: Counter = Counter()
    rev_sum: dict[str, float] = defaultdict(float)
    rev_sq: dict[str, float] = defaultdict(float)
    prod_n: Counter = Counter()
    prod_sum: dict[str, float] = defaultdict(float)
    prod_sq: dict[str, float] = defaultdict(float)

    with gzip.open(reviews_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_reviews is not None and i >= max_reviews:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("reviewerID", "")
            asin = r.get("asin", "")
            rating = float(r.get("overall", 0))
            if rid:
                rev_n[rid] += 1
                rev_sum[rid] += rating
                rev_sq[rid] += rating * rating
            if asin:
                prod_n[asin] += 1
                prod_sum[asin] += rating
                prod_sq[asin] += rating * rating
            if (i + 1) % 5_000_000 == 0:
                print(f"  ... {i + 1:,} reviews scanned ({time.time() - t0:.0f}s)")

    def _build_agg(counts, sums, sq_sums, prefix):
        ids = list(counts.keys())
        n_arr = np.array([counts[k] for k in ids], dtype=np.int64)
        mean_arr = np.array([sums[k] / counts[k] for k in ids])
        var_arr = np.array([sq_sums[k] / counts[k] - (sums[k] / counts[k]) ** 2
                            for k in ids])
        std_arr = np.sqrt(np.clip(var_arr, 0, None))
        return pd.DataFrame({
            f"{prefix}_review_count": n_arr,
            f"{prefix}_avg_rating": np.round(mean_arr, 2),
            f"{prefix}_rating_std": np.round(std_arr, 2),
        }, index=ids)

    reviewer_agg = _build_agg(rev_n, rev_sum, rev_sq, "reviewer")
    product_agg = _build_agg(prod_n, prod_sum, prod_sq, "product")
    elapsed = time.time() - t0
    print(f"  Stats for {len(reviewer_agg):,} reviewers, "
          f"{len(product_agg):,} products ({elapsed:.0f}s)")
    return reviewer_agg, product_agg


def load_metadata(path: Path) -> pd.DataFrame:
    print(f"Loading product metadata from {path} ...")
    rows = []
    seen = set()
    t0 = time.time()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = r.get("asin", "")
            if not asin or asin in seen:
                continue
            seen.add(asin)

            raw_price = r.get("price", "")
            price = None
            if raw_price:
                cleaned = re.sub(r"[^\d.]", "", str(raw_price))
                try:
                    price = float(cleaned) if cleaned else None
                except ValueError:
                    price = None

            cats = r.get("category") or []
            main_cat = r.get("main_cat", "")
            if isinstance(cats, list) and cats:
                flat = []
                for c in cats:
                    if isinstance(c, list):
                        flat.extend(c)
                    else:
                        flat.append(str(c))
                sub_cat = flat[-1] if flat else ""
            else:
                sub_cat = ""

            also_buy = r.get("also_buy") or []
            also_view = r.get("also_view") or []

            rank_dict = r.get("rank") or r.get("salesRank") or {}
            sales_rank = None
            if isinstance(rank_dict, dict) and rank_dict:
                for v in rank_dict.values():
                    try:
                        sales_rank = int(re.sub(r"[^\d]", "", str(v)))
                    except (ValueError, TypeError):
                        pass
                    break
            elif isinstance(rank_dict, str):
                m = re.search(r"[\d,]+", rank_dict.replace(",", ""))
                if m:
                    try:
                        sales_rank = int(m.group())
                    except ValueError:
                        pass

            desc = r.get("description") or []
            if isinstance(desc, list):
                desc = " ".join(str(d) for d in desc)
            desc = str(desc).strip()

            rows.append({
                "asin": asin,
                "product_title": " ".join((r.get("title") or "").split())[:200],
                "product_brand": (r.get("brand") or "").strip(),
                "product_price": price,
                "product_main_category": main_cat.strip(),
                "product_sub_category": sub_cat.strip(),
                "product_sales_rank": sales_rank,
                "product_also_buy_count": len(also_buy) if isinstance(also_buy, list) else 0,
                "product_also_view_count": len(also_view) if isinstance(also_view, list) else 0,
                "product_has_description": 1 if len(desc) > 10 else 0,
            })
            if (i + 1) % 500_000 == 0:
                print(f"  ... {i + 1:,} products scanned ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    # Keep only products whose main_cat is "Books" — the raw metadata
    # contains cross-contaminated categories (e.g. "Buy a Kindle",
    # "Audible Audiobooks", HTML tags) that break heterogeneity tests.
    before = len(df)
    df = df[df["product_main_category"] == "Books"].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} non-Books products ({dropped/before*100:.2f}%)")
    print(f"  Loaded metadata for {len(df):,} products in {time.time() - t0:.0f}s")
    return df


# ---------------------------------------------------------------------------
# Text feature precomputation
# ---------------------------------------------------------------------------

def compute_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute text-derived features from review_text, summary, and product_title.

    All features are scalar (numeric or binary) so the DSL executor can use them
    directly. Raw text columns are kept for reference but excluded from DSL.
    """
    rt = df["review_text"].fillna("").astype(str)
    sm = df["summary"].fillna("").astype(str)

    # --- Review structural features ---
    df["review_length"] = rt.str.len()
    df["review_word_count"] = rt.str.split().str.len().fillna(0).astype(int)
    # Sentence count: split on . ! ? followed by space or end
    df["review_sentence_count"] = rt.str.count(r"[.!?]+(?:\s|$)").clip(lower=1)

    # --- Lexical complexity ---
    words = rt.str.split()
    word_lengths = words.apply(lambda ws: np.mean([len(w) for w in ws]) if isinstance(ws, list) and ws else 0.0)
    df["review_avg_word_length"] = word_lengths.round(2)
    unique_counts = words.apply(lambda ws: len(set(w.lower() for w in ws)) if isinstance(ws, list) and ws else 0)
    total_counts = words.str.len().fillna(1).clip(lower=1)
    df["review_unique_word_ratio"] = (unique_counts / total_counts).round(4)

    # --- Punctuation / emotion signals ---
    df["review_exclamation_count"] = rt.str.count("!")
    df["review_question_count"] = rt.str.count(r"\?")
    alpha_chars = rt.str.count(r"[A-Za-z]").clip(lower=1)
    upper_chars = rt.str.count(r"[A-Z]")
    df["review_uppercase_ratio"] = (upper_chars / alpha_chars).round(4)

    # --- Summary features ---
    df["summary_length"] = sm.str.len()
    df["summary_word_count"] = sm.str.split().str.len().fillna(0).astype(int)

    # --- Binary content indicators ---
    df["review_has_exclamation"] = (df["review_exclamation_count"] > 0).astype(int)
    df["review_has_question"] = (df["review_question_count"] > 0).astype(int)

    return df


# Text-derived feature columns (appended after raw text columns)
TEXT_FEATURE_COLUMNS = [
    "review_length", "review_word_count", "review_sentence_count",
    "review_avg_word_length", "review_unique_word_ratio",
    "review_exclamation_count", "review_question_count", "review_uppercase_ratio",
    "summary_length", "summary_word_count",
    "review_has_exclamation", "review_has_question",
]

# Reviewer & product aggregates (cross-row, computed in first pass)
AGGREGATE_COLUMNS = [
    "reviewer_review_count", "reviewer_avg_rating", "reviewer_rating_std",
    "product_review_count", "product_avg_rating", "product_rating_std",
]

# Final column order
COLUMNS = [
    "asin", "reviewerID",
    "rating", "helpful_votes",
    "verified_purchase", "has_image", "product_format", "year", "date", "timestamp",
    "product_title", "product_brand", "product_price",
    "product_main_category", "product_sub_category",
    "product_sales_rank", "product_also_buy_count", "product_also_view_count",
    "product_has_description",
] + AGGREGATE_COLUMNS + [
    "review_text", "summary",
] + TEXT_FEATURE_COLUMNS


def process_and_write_chunk(raw_rows, meta_lookup, output, write_header,
                            reviewer_agg=None, product_agg=None):
    parsed = [parse_review(r) for r in raw_rows]
    df = pd.DataFrame(parsed)

    # Join product metadata
    if meta_lookup is not None:
        meta_cols = meta_lookup.columns.tolist()
        meta_sub = df[["asin"]].join(meta_lookup, on="asin", how="left")
        for c in meta_cols:
            df[c] = meta_sub[c].values

    # Join reviewer & product aggregate statistics
    if reviewer_agg is not None:
        for col in reviewer_agg.columns:
            df[col] = df["reviewerID"].map(reviewer_agg[col])
    if product_agg is not None:
        for col in product_agg.columns:
            df[col] = df["asin"].map(product_agg[col])

    # Ensure all columns exist
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS]

    # Clean text fields for CSV safety
    for tc in ["review_text", "summary", "product_title"]:
        if tc in df.columns:
            df[tc] = df[tc].fillna("").astype(str).str.replace(
                r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', regex=True
            )

    # Precompute text-derived features
    df = compute_text_features(df)

    # Ensure all columns exist
    for c in TEXT_FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = 0
    df = df[COLUMNS]

    df.to_csv(output, mode="a" if not write_header else "w",
              header=write_header, index=False, escapechar='\\')
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="Prepare Amazon Books dataset for AutoKD")
    parser.add_argument("--max-reviews", type=int, default=None,
                        help="Cap raw reviews (for testing)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sample", type=int, default=500_000,
                        help="Also write a stratified random sample of this size (0 to skip)")
    args = parser.parse_args()
    output = Path(args.output) if args.output else OUTPUT_PATH

    if not REVIEWS_PATH.exists():
        print(f"ERROR: {REVIEWS_PATH} not found")
        return

    # Load product metadata
    meta_lookup = None
    if META_PATH.exists():
        meta = load_metadata(META_PATH)
        meta_lookup = meta.set_index("asin")
        del meta
    else:
        print(f"WARNING: {META_PATH} not found, skipping product metadata")

    # First pass: collect per-reviewer and per-product aggregate stats
    reviewer_agg, product_agg = collect_aggregate_stats(REVIEWS_PATH, args.max_reviews)

    # Stream reviews in chunks, join metadata + aggregates, write CSV
    print(f"\nPass 2: streaming reviews, joining metadata + aggregates, writing CSV ...")
    t0 = time.time()
    rows_written = 0
    chunk_buf = []
    first_chunk = True

    with gzip.open(REVIEWS_PATH, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_reviews is not None and i >= args.max_reviews:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_buf.append(r)

            if len(chunk_buf) >= CHUNK_SIZE:
                rows_written += process_and_write_chunk(
                    chunk_buf, meta_lookup, output, first_chunk,
                    reviewer_agg, product_agg,
                )
                first_chunk = False
                chunk_buf = []
                elapsed = time.time() - t0
                print(f"  ... {rows_written:,} rows written ({elapsed:.0f}s, "
                      f"{rows_written/elapsed:.0f} rows/s)")

    if chunk_buf:
        rows_written += process_and_write_chunk(
            chunk_buf, meta_lookup, output, first_chunk,
            reviewer_agg, product_agg,
        )

    elapsed = time.time() - t0
    print(f"\n[OK] Saved to {output}")
    print(f"  Rows: {rows_written:,}")
    print(f"  Columns: {len(COLUMNS)}")
    print(f"  Time: {elapsed:.0f}s ({rows_written/elapsed:.0f} rows/s)")
    print(f"  File size: {output.stat().st_size / (1024**3):.2f} GB")

    # ---- Build stratified random sample across the FULL dataset ----
    if args.sample and args.sample > 0 and rows_written > args.sample:
        sample_path = output.parent / "amazon_books_sampled.csv"
        print(f"\nBuilding stratified random sample ({args.sample:,} rows) ...")
        t_s = time.time()
        # Pass 1: reservoir-style collection via chunked reading.
        # Each chunk samples proportionally so every row in the full
        # CSV has equal probability of selection.
        sample_frac = args.sample / rows_written * 1.5  # oversample, trim later
        rng = np.random.RandomState(42)
        chunks = []
        for chunk in pd.read_csv(output, chunksize=500_000):
            sampled = chunk.sample(frac=min(1.0, sample_frac), random_state=rng)
            chunks.append(sampled)
        combined = pd.concat(chunks, ignore_index=True)
        # Stratify by rating: ensure each rating level is represented
        if "rating" in combined.columns:
            frames = []
            for _, grp in combined.groupby("rating"):
                n = min(len(grp), max(500, int(args.sample * len(grp) / len(combined))))
                frames.append(grp.sample(n=min(n, len(grp)), random_state=42))
            combined = pd.concat(frames, ignore_index=True)
        if len(combined) > args.sample:
            combined = combined.sample(n=args.sample, random_state=42)
        combined.to_csv(sample_path, index=False)
        print(f"  Saved {len(combined):,} rows -> {sample_path}")
        print(f"  Ratings: {dict(combined['rating'].value_counts().sort_index())}")
        print(f"  Sub-categories: {combined['product_sub_category'].nunique()} unique")
        print(f"  Years: {int(combined['year'].min())}-{int(combined['year'].max())}")
        print(f"  Time: {time.time() - t_s:.0f}s")
        del combined, chunks

    # Quick column report
    print(f"\n{'='*60}")
    print("Column summary (first 10k rows):")
    print(f"{'='*60}")
    sample = pd.read_csv(output, nrows=10_000)
    for c in sample.columns:
        null_pct = sample[c].isna().mean() * 100
        print(f"  {c:40s} {str(sample[c].dtype):12s} null={null_pct:5.1f}%")

    # Prep report
    report = {
        "output_path": str(output),
        "rows": rows_written,
        "columns": COLUMNS,
        "file_size_gb": round(output.stat().st_size / (1024**3), 2),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nPrep report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
