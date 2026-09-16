#!/usr/bin/env python3
"""Ablation metrics at a matched round budget (Section 6.4, Table 6).

Every arm is scored over the discovered insights it stored within its first
``--budget`` rounds, so all arms spend the same discovery budget:

  n                   stored insights
  estimands/insight   distinct estimands (test type, {X, Y}, moderator, group)
                      per stored insight
  exact rediscovery   share of stored insights whose DSL specification exactly
                      repeats an earlier stored one
  deep-test share     share of stored insights using interact or heterogeneity
  delta depth         relative change in deep-test share versus the first arm

Arms are produced with the runner flags (``--ablation-no-persistence``,
``--ablation-no-graph``, ``--ablation-global-only``), each run writing its own
``--memory`` graph.

Usage:
  python evaluation/ablation_metrics.py --budget 1000 \\
      --arm "AutoKD=data/memory_wvs/insight_graph.json" \\
      --arm "w/o persistence=data/memory_wvs_no_persistence/insight_graph.json" \\
      --arm "w/o graph structure=data/memory_wvs_no_graph/insight_graph.json" \\
      --arm "global-only=data/memory_wvs_global_only/insight_graph.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.relation_classifier import estimand_key, test_type_rank  # noqa: E402

# Nodes written by cold-start seeding (or loaded as references) are not findings.
_NON_FINDING_SOURCES = frozenset({"statistical_profiling", "seed_prescan", "paper"})


def load_insights(graph_path: str, budget: int) -> List[dict]:
    """Discovered insights stored within the first ``budget`` rounds of a run."""
    data = json.loads(Path(graph_path).read_text())
    kept = []
    for ins in data.get("insights", []):
        meta = ins.get("metadata") or {}
        if ins.get("source") in _NON_FINDING_SOURCES:
            continue
        if str(meta.get("is_seed", "")).lower() in ("true", "1"):
            continue
        if int(meta.get("round_id") or 0) > budget:
            continue
        kept.append(ins)
    return kept


def arm_metrics(insights: List[dict]) -> Dict[str, float]:
    """Accumulation and depth metrics for one arm."""
    n = len(insights)
    if n == 0:
        return {"n": 0, "estimands_per_insight": 0.0, "exact_rediscovery": 0.0, "deep_test_share": 0.0}
    metas = [ins.get("metadata") or {} for ins in insights]
    estimands = {estimand_key(SimpleNamespace(metadata=meta)) for meta in metas}
    estimands.discard(None)
    specs = [meta.get("dsl_sentence") or ins.get("sentence", "") for meta, ins in zip(metas, insights)]
    deep = sum(1 for meta in metas if test_type_rank(meta.get("test_type", "")) >= 2)
    return {
        "n": n,
        "estimands_per_insight": len(estimands) / n,
        "exact_rediscovery": (n - len(set(specs))) / n,
        "deep_test_share": deep / n,
    }


def _parse_arm(value: str) -> Tuple[str, str]:
    name, sep, path = value.partition("=")
    if not sep or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=GRAPH_PATH, got {value!r}")
    return name.strip(), path.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="AutoKD ablation metrics at a matched round budget")
    ap.add_argument("--budget", type=int, default=1000, help="Round budget every arm is truncated to")
    ap.add_argument(
        "--arm", type=_parse_arm, action="append", required=True,
        help="NAME=GRAPH_PATH, repeated per arm; the first arm is the reference for delta depth",
    )
    args = ap.parse_args()

    rows = [(name, arm_metrics(load_insights(path, args.budget))) for name, path in args.arm]
    reference = rows[0][1]["deep_test_share"]

    print(f"Ablation metrics at a matched {args.budget}-round budget\n")
    header = (f"{'Variant':<24}{'n':>7}{'Estimands/insight':>20}{'Exact rediscovery':>20}"
              f"{'Deep-test share':>18}{'Delta depth':>14}")
    print(header)
    print("-" * len(header))
    for i, (name, m) in enumerate(rows):
        delta = "-" if i == 0 or reference == 0 else f"{m['deep_test_share'] / reference - 1:+.1%}"
        print(f"{name:<24}{m['n']:>7}{m['estimands_per_insight']:>20.3f}"
              f"{m['exact_rediscovery']:>20.1%}{m['deep_test_share']:>18.1%}{delta:>14}")


if __name__ == "__main__":
    main()
