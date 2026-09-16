#!/usr/bin/env python3
"""
Run AutoKD on SciSciNet (two-phase: general idea -> RAG -> refiner).

Prerequisites:
  1. SciSciNet CSV at data/sciscinet_cs_autokd.csv (or pass --data)
  2. Variable catalog JSONL at the profile's catalog_path (or pass --catalog)
  3. Ollama running with configured models

Usage:
  cd AutoKD && python scripts/run_sciscinet_rounds.py [--rounds N] [--data PATH] [--memory PATH] [--goal TEXT]
  cd AutoKD && python scripts/run_sciscinet_rounds.py --variant all --rounds 10
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

DEFAULT_PROFILE = PROJECT_ROOT / "config" / "profiles" / "sciscinet.yaml"

# Variant-specific defaults
VARIANT_DEFAULTS = {
    "cs": {
        "data_path": "data/sciscinet_cs_autokd.csv",
        "memory_path": "data/memory_sciscinet_cs/insight_graph.json",
        "goal": "Discover factors that drive scientific impact, disruption, and innovation in computer science research",
        "runs_dir": "data/runs_sciscinet_cs",
    },
    "all": {
        "data_path": "data/sciscinet_all_autokd.csv",
        "memory_path": "data/memory_sciscinet_all/insight_graph.json",
        "goal": "Discover factors that drive scientific impact, disruption, and innovation across all academic fields",
        "runs_dir": "data/runs_sciscinet_all",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AutoKD on SciSciNet (two-phase DSL flow)")
    ap.add_argument("--rounds", type=int, default=2, help="Number of discovery rounds")
    ap.add_argument("--variant", choices=["cs", "all"], default="cs",
                     help="SciSciNet variant: cs (CS papers only) or all (all fields, stratified)")
    ap.add_argument("--profile", type=str, default=str(DEFAULT_PROFILE), help="Dataset profile YAML")
    ap.add_argument("--data", type=str, default=None, help="Override data path")
    ap.add_argument("--memory", type=str, default=None, help="Override memory path")
    ap.add_argument("--goal", type=str, default=None, help="Override discovery goal")
    ap.add_argument("--catalog", type=str, default="", help="Override variable catalog JSONL path")
    ap.add_argument("--placeholder", action="store_true", help="Use placeholder LLM (no Ollama).")
    ap.add_argument(
        "--ablation-no-graph", action="store_true",
        help=(
            "Ablation A1: store insights as a FLAT collection with no relations. "
            "Retrieval falls back to flat cosine similarity over the same context "
            "budget, and anchor selection uses node metadata only."
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

    vd = VARIANT_DEFAULTS[args.variant]

    # Resolve paths (CLI overrides > profile > variant defaults)
    data_path = Path(args.data or profile.get("data_path", vd["data_path"]))
    memory_str = args.memory or profile.get("memory_path", vd["memory_path"])
    goal = args.goal or profile.get("goals", {}).get("primary", vd["goal"])
    catalog_path = Path(args.catalog) if args.catalog else Path(
        profile.get("catalog_path") or "data/processed/sciscinet/sciscinet_variable_catalog.jsonl"
    )

    # For 'all' variant, override paths from profile
    if args.variant == "all" and not args.data:
        data_path = Path(vd["data_path"])
    if args.variant == "all" and not args.memory:
        memory_str = vd["memory_path"]
    if args.variant == "all" and not args.goal:
        goal = vd["goal"]

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

    agents_cfg = config.config.setdefault("agents", {})
    critic_cfg = agents_cfg.setdefault("critic", {})
    # Effect floor (min_meaningful_effect_size) 0.03, matching config.yaml.
    critic_cfg["min_meaningful_effect_size"] = 0.03

    eval_cfg = agents_cfg.setdefault("evaluator", {})
    eval_cfg["min_meaningful_effect_size"] = 0.03

    orch_cfg = config.config.setdefault("orchestrator", {})
    orch_cfg["early_stop_stall_rounds"] = 30
    # The CLI --rounds flag is the sole cap; without this the orchestrator's own
    # max_rounds (2000 from config.yaml) raises StopIteration first and would
    # silently truncate an unbounded run.
    orch_cfg["max_rounds"] = args.rounds

    # Ablation flag must be set before DiscoveryEngine builds the agents,
    # since Historian/Orchestrator read it in their constructors.
    config.config.setdefault("ablation", {})["no_graph_structure"] = bool(args.ablation_no_graph)

    # Variable catalog retrieval limits (0 = uncapped, which floods the prompt)
    vc.setdefault("max_retrieved_variables", 15)
    vc.setdefault("max_prompt_variable_cards", 25)

    # Theme cooldown: a theme used once in the last 6 rounds is avoided.
    vc["theme_cooldown_window"] = 6
    vc["theme_cooldown_threshold"] = 1
    # Forced exploration applies to WVS only.
    vc["forced_explore_every"] = 0

    # Artifacts directory per variant
    config.config.setdefault("artifacts", {})["base_dir"] = vd["runs_dir"]

    config.config.setdefault("memory", {}).setdefault("preexisting_similarity_threshold", 0.88)

    # Ensure memory directory exists
    memory_path = Path(memory_str)
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"SciSciNet mode (two-phase: general idea -> RAG -> DSL refiner)")
    print(f"  Variant: {args.variant}")
    if args.placeholder:
        print("  LLM: placeholder (no Ollama)")
    print(f"  Data: {data_path}")
    print(f"  Catalog: {catalog_path}")
    print(f"  Memory: {memory_str}")
    print(f"  Rounds: {args.rounds} (ceiling; early-stop after "
          f"{orch_cfg['early_stop_stall_rounds']} consecutive empty rounds)")
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    print(f"  Arm: {'ABLATION A1 (no graph structure — flat insight store)' if args.ablation_no_graph else 'CONTROL (full insight graph)'}")
    print(f"  Goal: {goal[:80]}...")
    print()

    engine = DiscoveryEngine(data_path=str(data_path), memory_path=memory_str)
    engine.run(num_rounds=args.rounds)


if __name__ == "__main__":
    main()
