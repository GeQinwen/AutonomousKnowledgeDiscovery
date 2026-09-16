#!/usr/bin/env python3
"""
Run AutoKD in WVS mode (two-phase: general idea → RAG → refiner) on the full WVS dataset.

Prerequisites:
  1. WVS CSV at the profile's data_path (or pass --data)
  2. Variable catalog JSONL at the profile's catalog_path (or pass --catalog)
  3. Ollama running with configured models

Usage:
  cd AutoKD && python scripts/run_wvs_rounds.py [--rounds N] [--data PATH] [--memory PATH] [--catalog PATH] [--goal TEXT]
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

DEFAULT_PROFILE = PROJECT_ROOT / "config" / "profiles" / "wvs.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AutoKD on WVS (two-phase flow)")
    ap.add_argument("--rounds", type=int, default=2, help="Number of discovery rounds")
    ap.add_argument("--profile", type=str, default=str(DEFAULT_PROFILE), help="Dataset profile YAML")
    ap.add_argument(
        "--data", type=str, default=None,
        help="Override data path (default: from profile)",
    )
    ap.add_argument(
        "--memory", type=str, default=None,
        help="Override memory path (default: from profile)",
    )
    ap.add_argument(
        "--goal", type=str, default=None,
        help="Override discovery goal (default: from profile)",
    )
    ap.add_argument(
        "--catalog", type=str, default="",
        help="Override variable catalog JSONL path",
    )
    ap.add_argument(
        "--placeholder", action="store_true",
        help="Use placeholder LLM (no Ollama).",
    )
    ap.add_argument(
        "--ablation-no-graph", action="store_true",
        help=(
            "Ablation A1: store insights as a FLAT collection with no relations. "
            "Retrieval falls back to flat cosine similarity over the same context "
            "budget, and anchor selection uses node metadata only."
        ),
    )
    ap.add_argument(
        "--ablation-global-only", action="store_true",
        help=(
            "Ablation A3: Orchestrator always selects global_exploration, so every "
            "round runs with no anchors. Isolates the value of mode scheduling; the "
            "insight graph is still built and still read by retrieval."
        ),
    )
    ap.add_argument(
        "--ablation-no-persistence", action="store_true",
        help=(
            "Ablation A0: the loop keeps NO memory across rounds. No round "
            "context, no anchors, and no cross-round dedup (CROSS_ROUND, "
            "PRE-EXISTING, SELF-DUPLICATE, STRUCTURAL_DUPLICATE), plus the "
            "Historian's own injection-time duplicate check is bypassed so "
            "rediscovery is COUNTED rather than silently collapsed. Keeps "
            "TAUTOLOGICAL_DERIVED_PAIR, DSL validation, the Evaluator and the "
            "Critic. Tests cumulative accumulation, the differentiator claimed "
            "in Section 1; distinct from A1, which keeps the node store."
        ),
    )
    ap.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Seed Python/NumPy RNGs so mode, anchor and theme scheduling are "
            "reproducible across arms (LLM sampling stays stochastic)."
        ),
    )
    args = ap.parse_args()

    if args.seed is not None:
        import random as _random
        _random.seed(args.seed)
        try:
            import numpy as _np
            _np.random.seed(args.seed)
        except Exception:
            pass

    # Load dataset profile
    profile = load_profile(args.profile)
    print(f"[INFO] Loaded dataset profile: {profile.get('name', 'unknown')}")

    # Resolve paths (CLI overrides > profile > legacy defaults)
    data_path = Path(args.data or profile.get("data_path", "data/processed/wvs/wvs7_clean.csv"))
    memory_str = args.memory or profile.get("memory_path", "data/memory_wvs/insight_graph.json")
    goal = args.goal or profile.get("goals", {}).get("primary", "general discovery")
    catalog_path = Path(args.catalog) if args.catalog else Path(
        profile.get("catalog_path") or "data/processed/wvs/wvs7_variable_catalog.jsonl"
    )

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    if profile.get("use_two_phase") and not catalog_path.exists():
        print(f"Error: Variable catalog not found: {catalog_path}")
        sys.exit(1)

    # Apply profile to config
    config.config.setdefault("goals", {})["primary"] = goal
    if profile.get("goals", {}).get("secondary"):
        config.config["goals"]["secondary"] = profile["goals"]["secondary"]
    config.config.setdefault("data", {})["data_path"] = str(data_path)
    if profile.get("weight_columns") is not None:
        config.config["data"]["weight_columns"] = profile["weight_columns"]
    if profile.get("control_only_columns") is not None:
        config.config["data"]["control_only_columns"] = profile["control_only_columns"]

    if args.placeholder:
        config.config.setdefault("llm", {})["provider"] = "placeholder"

    vc = config.config.setdefault("variable_catalog", {})
    vc["enabled"] = bool(profile.get("use_two_phase", False))
    if catalog_path:
        vc["catalog_path"] = str(catalog_path)
    if profile.get("concept_menu"):
        vc["concept_menu"] = profile["concept_menu"]
    if profile.get("always_keep"):
        vc["always_keep"] = profile["always_keep"]

    if profile.get("domain_description"):
        config.config.setdefault("agents", {}).setdefault("critic", {})[
            "domain_description"
        ] = profile["domain_description"]

    # Effect floor (min_meaningful_effect_size) 0.03, matching config.yaml.
    agents_cfg = config.config.setdefault("agents", {})
    agents_cfg.setdefault("evaluator", {})["min_meaningful_effect_size"] = 0.03
    agents_cfg.setdefault("critic", {})["min_meaningful_effect_size"] = 0.03

    # Let the CLI --rounds flag be the sole cap; remove the orchestrator's
    # internal max_rounds limit (which defaults to 2000 from config.yaml).
    orch_cfg = config.config.setdefault("orchestrator", {})
    orch_cfg["max_rounds"] = args.rounds

    config.config.setdefault("memory", {}).setdefault("preexisting_similarity_threshold", 0.88)

    # Ablation flag must be set before DiscoveryEngine builds the agents,
    # since Historian/Orchestrator read it in their constructors.
    config.config.setdefault("ablation", {})["no_graph_structure"] = bool(args.ablation_no_graph)
    config.config["ablation"]["global_mode_only"] = bool(args.ablation_global_only)
    config.config["ablation"]["no_persistence"] = bool(args.ablation_no_persistence)

    # Ensure memory directory exists
    memory_path = Path(memory_str)
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"WVS mode (two-phase: general idea -> RAG -> refiner)")
    if args.placeholder:
        print("  LLM: placeholder (no Ollama)")
    print(f"  Data: {data_path}")
    print(f"  Catalog: {catalog_path}")
    print(f"  Memory: {memory_str}")
    print(f"  Rounds: {args.rounds}")
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    _arms = []
    if args.ablation_no_graph:
        _arms.append("A1 (no graph structure — flat insight store)")
    if args.ablation_global_only:
        _arms.append("A3 (global exploration only — no mode scheduling)")
    if args.ablation_no_persistence:
        _arms.append("A0 (no persistence — no context, no anchors, no cross-round dedup)")
    print(f"  Arm: {'ABLATION ' + ' + '.join(_arms) if _arms else 'CONTROL (full system)'}")
    print(f"  Goal: {goal[:80]}...")
    print()

    engine = DiscoveryEngine(data_path=str(data_path), memory_path=memory_str)
    engine.run(num_rounds=args.rounds)


if __name__ == "__main__":
    main()
