#!/usr/bin/env python3
"""
Pre-scan seeding: inject significant cross-module associations into the
insight graph before the main discovery loop.

Runs fast Pearson correlations on all substantive variable pairs and injects
significant findings as seed insights. This eliminates the cold-start problem
and gives the orchestrator diverse anchors across the full variable space.

Usage:
  cd AutoKD
  python scripts/seed_prescan.py --data data/amazon_books_sampled.csv \
      --memory data/memory_amazon_books/insight_graph.json \
      --profile config/profiles/amazon_books.yaml

Prerequisites:
  - Prepared CSV (run the appropriate prepare_*.py script)
  - Dataset profile YAML (in config/profiles/)
  - Variable catalog JSONL (built by build_*_variable_catalog.py)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.types import Insight, RelationType
from memory.insight_graph import InsightGraph
from core.dataset_profile import (
    load_profile,
    id_columns,
    text_columns,
    control_only_columns,
    drop_columns,
    weight_columns,
    column_labels,
    derived_pairs,
)


# ── Shared helpers (also used by core/orchestration.py) ──────────────


def get_variable_modules_from_catalog(catalog_path: str) -> Dict[str, str]:
    """Load var_name -> module mapping from a JSONL variable catalog."""
    mapping: Dict[str, str] = {}
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            for line in f:
                card = json.loads(line)
                name = (card.get("var_name") or "").strip()
                module = (card.get("module") or "").strip()
                if name and module:
                    mapping[name] = module
    except Exception:
        pass
    return mapping


def get_substantive_columns(df: pd.DataFrame) -> List[str]:
    """Return columns suitable for pre-scan correlation testing."""
    skip: set = set()
    skip.update(id_columns() or [])
    skip.update(text_columns() or [])
    skip.update(control_only_columns() or [])
    skip.update(drop_columns() or [])
    skip.update(weight_columns() or [])
    # Common non-analytic columns regardless of profile
    skip.update({"review_text", "summary", "product_title", "product_brand",
                 "date", "timestamp", "survey_id"})

    cols: List[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if df[c].isna().mean() > 0.90:
            continue
        if df[c].nunique(dropna=True) < 2:
            continue
        # Must be numeric or coercible to numeric
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        else:
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().mean() >= 0.5:
                cols.append(c)
    return cols


def prescan_correlations(
    df: pd.DataFrame,
    cols: List[str],
    var_modules: Dict[str, str],
    taut_pairs: List[frozenset],
    min_p: float = 0.01,
    min_effect: float = 0.03,
    max_seeds: int = 150,
) -> List[dict]:
    """Run fast Pearson correlations on all column pairs.

    Returns a list of result dicts sorted by priority (cross-module first,
    then by descending |effect_size|).
    """
    results: List[dict] = []
    tested = 0

    for i, cx in enumerate(cols):
        mod_x = var_modules.get(cx, "")
        for cy in cols[i + 1:]:
            mod_y = var_modules.get(cy, "")

            # Skip tautological / derived pairs
            if taut_pairs and frozenset({cx, cy}) in taut_pairs:
                continue

            # Paired valid observations
            mask = df[cx].notna() & df[cy].notna()
            if mask.sum() < 30:
                continue

            x_num = pd.to_numeric(df.loc[mask, cx], errors="coerce")
            y_num = pd.to_numeric(df.loc[mask, cy], errors="coerce")
            valid = x_num.notna() & y_num.notna()
            x_clean = x_num[valid].values
            y_clean = y_num[valid].values
            n = len(x_clean)
            if n < 30:
                continue

            tested += 1
            try:
                r, p = sp_stats.pearsonr(x_clean, y_clean)
            except Exception:
                continue
            if np.isnan(r) or np.isnan(p):
                continue
            if p >= min_p or abs(r) < min_effect:
                continue

            is_cross = (
                bool(mod_x) and bool(mod_y)
                and mod_x != mod_y
                and mod_x not in ("Identifiers", "Unknown")
                and mod_y not in ("Identifiers", "Unknown")
            )
            results.append({
                "x": cx, "y": cy,
                "effect": float(r), "p": float(p),
                "n": n, "coverage": n / len(df),
                "mod_x": mod_x, "mod_y": mod_y,
                "cross": is_cross,
            })

    print(f"  Tested {tested} pairs, found {len(results)} significant (p<{min_p}, |r|>={min_effect})")

    # Prioritize: cross-module first, then by descending |effect|
    results.sort(key=lambda r: (not r["cross"], -abs(r["effect"])))

    if len(results) > max_seeds:
        cross = [r for r in results if r["cross"]]
        within = [r for r in results if not r["cross"]]
        if len(cross) > max_seeds:
            # Too many cross-module: take top by |effect|
            results = cross[:max_seeds]
        else:
            remaining = max_seeds - len(cross)
            results = cross + within[:remaining]

    return results


def compute_seed_validity(p: float, effect: float, n: int) -> float:
    """Simplified validity score matching evaluator logic."""
    p_score = 1.0 if p < 0.001 else (0.95 if p < 0.01 else 0.85)
    e_score = 1.0 / (1.0 + math.exp(-8 * (abs(effect) - 0.12)))
    p_wt = 0.3 + 0.5 / (1.0 + (n / 1000.0) ** 1.5)
    return min(1.0, p_score * p_wt + e_score * (1.0 - p_wt))


def build_module_pair_seeds(
    scan_results: List[dict],
    labels: Dict[str, str],
    top_pairs_per_seed: int = 5,
) -> List[Tuple[Insight, dict]]:
    """Aggregate prescan correlations into one seed per module pair.

    Instead of 150+ individual assoc findings, creates ~20-40 "research brief"
    seeds that describe the correlation landscape between two modules.  Each
    seed lists the top variable pairs worth investigating, giving the
    orchestrator a roadmap without pre-computing finished answers.
    """
    # Group results by (module_x, module_y) pair
    _skip_mods = {"Identifiers", "Unknown", ""}
    pair_buckets: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in scan_results:
        mx, my = r.get("mod_x", ""), r.get("mod_y", "")
        if not mx or not my or mx in _skip_mods or my in _skip_mods:
            continue
        key = tuple(sorted([mx, my]))
        pair_buckets[key].append(r)

    seeds: List[Tuple[Insight, dict]] = []
    for (mod_a, mod_b), bucket in sorted(pair_buckets.items(),
                                          key=lambda kv: -len(kv[1])):
        # Sort by |effect| descending within this module pair
        bucket.sort(key=lambda r: -abs(r["effect"]))
        top = bucket[:top_pairs_per_seed]
        is_cross = mod_a != mod_b

        # Build readable sentence
        pair_descs = []
        all_cols = set()
        card_modules: Dict[str, str] = {}
        for r in top:
            lx = labels.get(r["x"], r["x"])
            ly = labels.get(r["y"], r["y"])
            direction = "+" if r["effect"] > 0 else "-"
            pair_descs.append(f"{lx} ↔ {ly} (r={r['effect']:+.2f})")
            all_cols.update([r["x"], r["y"]])
            if r["mod_x"]:
                card_modules[r["x"]] = r["mod_x"]
            if r["mod_y"]:
                card_modules[r["y"]] = r["mod_y"]

        n_total = len(bucket)
        avg_effect = sum(abs(r["effect"]) for r in bucket) / len(bucket)
        strongest = top[0]

        if is_cross:
            sentence = (
                f"Cross-module link: {mod_a} × {mod_b} — "
                f"{n_total} significant correlations (avg |r|={avg_effect:.2f}). "
                f"Strongest pairs: {'; '.join(pair_descs)}. "
                f"These warrant controlled investigation and moderation analysis."
            )
        else:
            sentence = (
                f"Within-module structure: {mod_a} — "
                f"{n_total} internal correlations (avg |r|={avg_effect:.2f}). "
                f"Key relationships: {'; '.join(pair_descs)}."
            )

        insight = Insight(
            id=f"prescan_{uuid.uuid4().hex[:8]}",
            sentence=sentence,
            source="seed_prescan",
            created_at=datetime.now(),
            metadata={
                "knowledge_type": "cross_module_profile" if is_cross else "within_module_profile",
                "columns": sorted(all_cols),
                "card_modules": card_modules,
                "is_seed": True,
                "is_cross_module": is_cross,
                "module_pair": [mod_a, mod_b],
                "n_significant_pairs": n_total,
                "avg_effect": avg_effect,
                "strongest_effect": strongest["effect"],
                "strongest_pair": [strongest["x"], strongest["y"]],
                "top_pairs": [[r["x"], r["y"], r["effect"]] for r in top],
                "round_id": 0,
                # Small positive scores so seeds are competitive as anchors
                # in refinement mode (0.0 would be filtered out).  Scored low
                # enough that any real discovery outranks them.
                "overall_score": 0.35,
                "novelty_score": 0.3,
                "validity_score": 0.4,
                "surprise_class": "EXPECTED",
            },
        )
        # Pass module info for EXTENDS injection
        seeds.append((insight, {"mod_a": mod_a, "mod_b": mod_b, "cross": is_cross}))

    return seeds


def inject_prescan_seeds(
    insight_graph: InsightGraph,
    seeds: List[Tuple[Insight, dict]],
) -> int:
    """Inject module-pair summary seeds and add EXTENDS relations between cross-module pairs.

    Creates a sparse research-roadmap graph: one node per module pair,
    EXTENDS edges between seeds that share a module.
    """
    added = 0
    seed_ids_by_module: Dict[str, List[str]] = defaultdict(list)

    for insight, info in seeds:
        try:
            insight_graph.insights[insight.id] = insight
            insight_graph.graph.add_node(insight.id, insight=insight)
            insight_graph._get_insight_embedding(insight.id, insight.sentence)
            added += 1

            for mod in (info.get("mod_a", ""), info.get("mod_b", "")):
                if mod and mod not in ("Identifiers", "Unknown"):
                    seed_ids_by_module[mod].append(insight.id)
        except Exception as e:
            print(f"  [WARN] Failed to add seed: {e}")

    # EXTENDS edges: connect seeds that share a module (transitive bridge)
    extends = 0
    for mod, ids in seed_ids_by_module.items():
        for i in range(len(ids)):
            for j in range(i + 1, min(len(ids), i + 3)):  # max 2 edges per node per module
                try:
                    insight_graph.graph.add_edge(
                        ids[i], ids[j],
                        relation_type=RelationType.EXTENDS,
                        metadata={},
                    )
                    extends += 1
                except Exception:
                    pass

    insight_graph.save()
    print(f"  Injected {added} module-pair seeds, {extends} EXTENDS edges.")
    return added


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-scan seeding: inject cross-module associations into the insight graph"
    )
    ap.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    ap.add_argument("--memory", type=str, required=True, help="Path to insight_graph.json")
    ap.add_argument("--profile", type=str, default=None, help="Dataset profile YAML")
    ap.add_argument("--catalog", type=str, default=None, help="Variable catalog JSONL")
    ap.add_argument("--min-p", type=float, default=0.01, help="Max p-value threshold")
    ap.add_argument("--min-effect", type=float, default=0.03, help="Min |effect| threshold")
    ap.add_argument("--max-seeds", type=int, default=150, help="Max number of seeds to inject")
    args = ap.parse_args()

    # Load profile if provided
    if args.profile:
        load_profile(args.profile)
        print(f"[INFO] Loaded profile: {args.profile}")

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    print(f"[INFO] Loading data from {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    MAX_ROWS = 500_000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=0).reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Load insight graph
    memory_path = Path(args.memory)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    insight_graph = InsightGraph(storage_path=str(memory_path))

    # Check for existing prescan seeds
    existing = insight_graph.get_all_insights()
    existing_prescan = [i for i in existing if i.source == "seed_prescan"]
    if existing_prescan:
        print(f"[SKIP] {len(existing_prescan)} prescan seeds already exist. "
              "Reset memory first to re-seed.")
        return

    # Get variable modules
    var_modules: Dict[str, str] = {}
    catalog_path = args.catalog
    if not catalog_path:
        # Try profile
        from core.dataset_profile import profile_get
        catalog_path = profile_get("catalog_path")
    if catalog_path and Path(catalog_path).exists():
        var_modules = get_variable_modules_from_catalog(catalog_path)
        print(f"  Variable catalog: {len(var_modules)} variables with module assignments")

    # Get substantive columns and run correlations
    cols = get_substantive_columns(df)
    print(f"  Substantive columns: {len(cols)}")

    taut_pairs = derived_pairs() or []
    labels = column_labels() or {}

    results = prescan_correlations(
        df, cols, var_modules, taut_pairs,
        min_p=args.min_p, min_effect=args.min_effect, max_seeds=args.max_seeds,
    )

    if not results:
        print("[INFO] No significant associations found.")
        return

    # Build and inject module-pair summary seeds
    seeds = build_module_pair_seeds(results, labels)
    count = inject_prescan_seeds(insight_graph, seeds)

    # Summary
    n_cross = sum(1 for _, info in seeds if info.get("cross"))
    mods_covered: set = set()
    for _, info in seeds:
        for m in (info.get("mod_a", ""), info.get("mod_b", "")):
            if m and m not in ("Identifiers", "Unknown"):
                mods_covered.add(m)

    print(f"\n{'='*60}")
    print(f"Pre-scan seeding complete")
    print(f"{'='*60}")
    print(f"  Module-pair briefs: {count}")
    print(f"  Cross-module:       {n_cross}")
    print(f"  Within-module:      {count - n_cross}")
    print(f"  Modules covered:    {len(mods_covered)} ({', '.join(sorted(mods_covered))})")
    print(f"  Graph total:        {len(insight_graph.get_all_insights())} insights")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
