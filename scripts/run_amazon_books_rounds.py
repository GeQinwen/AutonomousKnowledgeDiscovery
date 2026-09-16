#!/usr/bin/env python3
"""
Run AutoKD on Amazon Books dataset (two-phase DSL flow).

Prerequisites:
  1. Prepared CSV at data/amazon_books_autokd.csv (or pass --data)
  2. Variable catalog JSONL at the profile's catalog_path
  3. Ollama running with configured models

Usage:
  cd AutoKD && python scripts/run_amazon_books_rounds.py [--rounds N] [--data PATH] [--goal TEXT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import config
from core.dataset_profile import load_profile
from core.orchestration import DiscoveryEngine

DEFAULT_PROFILE = PROJECT_ROOT / "config" / "profiles" / "amazon_books.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AutoKD on Amazon Books (two-phase DSL flow)")
    ap.add_argument("--rounds", type=int, default=5, help="Number of discovery rounds")
    ap.add_argument("--profile", type=str, default=str(DEFAULT_PROFILE), help="Dataset profile YAML")
    ap.add_argument("--data", type=str, default=None, help="Override data path")
    ap.add_argument("--memory", type=str, default=None, help="Override memory path")
    ap.add_argument("--goal", type=str, default=None, help="Override discovery goal")
    ap.add_argument("--placeholder", action="store_true", help="Use placeholder LLM (no Ollama)")
    ap.add_argument(
        "--sample-rows", type=int, default=500_000,
        help="Sample N rows for faster execution (0 = all rows). Default 500k.",
    )
    args = ap.parse_args()

    # Load dataset profile
    profile = load_profile(args.profile)
    print(f"[INFO] Loaded dataset profile: {profile.get('name', 'unknown')}")

    # Resolve paths (CLI overrides > profile)
    data_path = Path(args.data or profile.get("data_path", "data/amazon_books_autokd.csv"))
    memory_str = args.memory or profile.get("memory_path", "data/memory_amazon_books/insight_graph.json")
    goal = args.goal or profile.get("goals", {}).get("primary", "general discovery")

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    # Apply profile to config
    config.config.setdefault("goals", {})["primary"] = goal
    if profile.get("goals", {}).get("secondary"):
        config.config.setdefault("goals", {})["secondary"] = profile["goals"]["secondary"]

    config.config.setdefault("data", {})["data_path"] = str(data_path)
    if profile.get("weight_columns") is not None:
        config.config["data"]["weight_columns"] = profile["weight_columns"]
    if profile.get("control_only_columns") is not None:
        config.config["data"]["control_only_columns"] = profile["control_only_columns"]

    vc = config.config.setdefault("variable_catalog", {})
    vc["enabled"] = bool(profile.get("use_two_phase", False))
    catalog_path = profile.get("catalog_path")
    if catalog_path:
        vc["catalog_path"] = catalog_path
    if profile.get("concept_menu"):
        vc["concept_menu"] = profile["concept_menu"]
    if profile.get("always_keep"):
        vc["always_keep"] = profile["always_keep"]
    if "max_retrieved_variables" in profile:
        vc["max_retrieved_variables"] = profile["max_retrieved_variables"]

    if profile.get("use_two_phase") and catalog_path and not Path(catalog_path).exists():
        print(f"Error: Variable catalog not found: {catalog_path}")
        sys.exit(1)

    if profile.get("domain_description"):
        config.config.setdefault("agents", {}).setdefault("critic", {})[
            "domain_description"
        ] = profile["domain_description"]

    config.config.setdefault("memory", {})["storage_path"] = memory_str
    config.config["memory"].setdefault("preexisting_similarity_threshold", 0.88)

    # Effect floor (min_meaningful_effect_size) 0.03, matching config.yaml.
    agents_cfg = config.config.setdefault("agents", {})
    agents_cfg.setdefault("critic", {})["min_meaningful_effect_size"] = 0.03
    agents_cfg.setdefault("evaluator", {})["min_meaningful_effect_size"] = 0.03

    # Theme cooldown: a theme used once in the last 4 rounds is avoided.
    vc = config.config.setdefault("variable_catalog", {})
    vc["theme_cooldown_window"] = 4
    vc["theme_cooldown_threshold"] = 1

    if args.placeholder:
        config.config.setdefault("llm", {})["provider"] = "placeholder"

    # Ensure memory directory exists
    memory_path = Path(memory_str)
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Use pre-built sample if available ----
    actual_data_path = data_path
    if args.sample_rows > 0:
        sample_path = data_path.parent / "amazon_books_sampled.csv"
        if sample_path.exists():
            import pandas as pd
            n_rows = sum(1 for _ in open(sample_path)) - 1
            print(f"Using pre-built sample: {sample_path.name} ({n_rows:,} rows)")
            actual_data_path = sample_path
        else:
            print(f"WARNING: Pre-built sample not found at {sample_path}")
            print(f"  Falling back to full dataset (slow).")

    print()
    print(f"Amazon Books mode (two-phase DSL flow)")
    if args.placeholder:
        print("  LLM: placeholder (no Ollama)")
    print(f"  Data: {actual_data_path}")
    print(f"  Memory: {memory_str}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Goal: {goal[:80]}...")
    print()

    engine = DiscoveryEngine(data_path=str(actual_data_path), memory_path=memory_str)
    engine.run(num_rounds=args.rounds)


if __name__ == "__main__":
    main()
