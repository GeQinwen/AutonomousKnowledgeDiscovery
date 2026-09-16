#!/usr/bin/env python3
"""Open-ended quality evaluation (Section 5.1, Tables 1-2).

Three steps, one subcommand each:

  select   Rank every discovered insight by novelty * min(1, validity / 0.6) and
           keep the top K with a diversity filter that skips any candidate whose
           cosine similarity to an already-selected insight exceeds a threshold.
  pair     Align the selected insights with the same number of gold claims by
           Hungarian matching on sentence-embedding cosine similarity.
  analyze  Summarize the blinded pairwise ratings: the share of non-tie
           responses that prefer the system finding (ties excluded), by dataset,
           dimension and rater group, and Fleiss' kappa within each rater group.

Usage:
  python evaluation/open_ended_quality.py select --graph data/memory_wvs/insight_graph.json \\
      --diversity-threshold 0.65 --out top10_wvs.jsonl
  python evaluation/open_ended_quality.py pair --system top10_wvs.jsonl --gold gold_claims_wvs.txt
  python evaluation/open_ended_quality.py analyze --ratings ratings.csv

``pair`` reads plain-text files (one finding per line) or JSONL with a
``sentence``/``text`` field.  ``analyze`` expects one CSV row per rater, pair and
dimension with columns ``rater, rater_group, dataset, pair, dimension, winner``,
where ``rater_group`` is ``human`` or ``llm`` and ``winner`` is ``system``,
``gold`` or ``tie``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VALIDITY_FLOOR = 0.6
DIMENSIONS = ["More interesting", "More surprising", "Most useful", "Overall better"]
WINNERS = ("system", "gold", "tie")

# Nodes written by cold-start seeding (or loaded as references) are not findings.
_NON_FINDING_SOURCES = frozenset({"statistical_profiling", "seed_prescan", "paper"})


# ── Top-K selection ──────────────────────────────────────────────────────

def load_discovered_insights(graph_path: str) -> List[dict]:
    """Insights accepted by the discovery loop (seed nodes excluded)."""
    data = json.loads(Path(graph_path).read_text())
    return [
        ins for ins in data.get("insights", [])
        if ins.get("source") not in _NON_FINDING_SOURCES
        and str((ins.get("metadata") or {}).get("is_seed", "")).lower() not in ("true", "1")
    ]


def ranking_score(insight: dict, validity_floor: float = VALIDITY_FLOOR) -> float:
    """novelty * min(1, validity / validity_floor): validity acts as a soft floor."""
    meta = insight.get("metadata") or {}
    novelty = float(meta.get("novelty_score") or 0.0)
    validity = float(meta.get("validity_score") or 0.0)
    return novelty * min(1.0, validity / validity_floor)


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)


def select_top_k(
    insights: Sequence[dict],
    embeddings: np.ndarray,
    k: int,
    diversity_threshold: float,
    validity_floor: float = VALIDITY_FLOOR,
) -> List[int]:
    """Indices of the top-k insights after greedy cosine-diversity filtering."""
    scores = np.array([ranking_score(ins, validity_floor) for ins in insights])
    embeddings = np.asarray(embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    chosen: List[int] = []
    # Descending score; tied scores (common) are visited from the highest index down.
    for idx in np.argsort(scores)[::-1]:
        if len(chosen) >= k:
            break
        if all(float(embeddings[idx] @ embeddings[j]) / (norms[idx] * norms[j] + 1e-9) <= diversity_threshold
               for j in chosen):
            chosen.append(int(idx))
    return chosen


# ── Pairing with gold claims ─────────────────────────────────────────────

def pair_with_gold(system_embeddings: np.ndarray, gold_embeddings: np.ndarray) -> List[Tuple[int, int, float]]:
    """Hungarian matching that maximizes total cosine similarity; returns (system, gold, similarity)."""
    from scipy.optimize import linear_sum_assignment

    sim = _unit(np.asarray(system_embeddings, dtype=float)) @ _unit(np.asarray(gold_embeddings, dtype=float)).T
    rows, cols = linear_sum_assignment(-sim)
    return [(int(r), int(c), float(sim[r, c])) for r, c in zip(rows, cols)]


# ── Rating analysis ──────────────────────────────────────────────────────

def load_ratings(path: str) -> List[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    required = {"rater", "rater_group", "dataset", "pair", "dimension", "winner"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit(f"ratings file is missing columns: {sorted(missing)}")
    for row in rows:
        row["rater_group"] = row["rater_group"].strip().lower()
        row["winner"] = row["winner"].strip().lower()
        if row["winner"] not in WINNERS:
            raise SystemExit(f"unexpected winner value {row['winner']!r}; expected one of {WINNERS}")
    return rows


def win_rate(rows: Sequence[dict]) -> float:
    """Percentage of non-tie responses preferring the system finding."""
    system = sum(1 for r in rows if r["winner"] == "system")
    gold = sum(1 for r in rows if r["winner"] == "gold")
    return 100.0 * system / (system + gold) if system + gold else float("nan")


def fleiss_kappa(rows: Sequence[dict]) -> float:
    """Fleiss' kappa with pairs as items and system / gold / tie as categories."""
    counts: Dict[Tuple[str, str], List[int]] = {}
    for r in rows:
        item = counts.setdefault((r["dataset"], r["pair"]), [0, 0, 0])
        item[WINNERS.index(r["winner"])] += 1
    if not counts:
        return float("nan")
    table = np.array(list(counts.values()), dtype=float)
    raters = table.sum(axis=1)
    if raters.min() != raters.max() or raters[0] < 2:
        return float("nan")  # every item needs the same number of raters
    n = raters[0]
    agreement = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    category_share = table.sum(axis=0) / table.sum()
    chance = float((category_share ** 2).sum())
    return float((agreement.mean() - chance) / (1 - chance)) if chance < 1 else float("nan")


def print_tables(rows: Sequence[dict]) -> None:
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    present = {r["dimension"] for r in rows}
    dims = [d for d in DIMENSIONS if d in present] + sorted(present - set(DIMENSIONS))

    def cell(value: float, fmt: str) -> str:
        return "-" if np.isnan(value) else format(value, fmt)

    head = f"{'Dataset':<14}{'Rater':<8}" + "".join(f"{d:>18}" for d in dims)
    print("Win rate (%, ties excluded)")
    print(head)
    for ds in datasets + ["All"]:
        for label, group in (("Human", "human"), ("LLM", "llm"), ("All", None)):
            sub = [r for r in rows if (ds == "All" or r["dataset"] == ds)
                   and (group is None or r["rater_group"] == group)]
            print(f"{ds:<14}{label:<8}" + "".join(
                f"{cell(win_rate([r for r in sub if r['dimension'] == d]), '.1f'):>18}" for d in dims))

    print("\nInter-rater agreement (Fleiss' kappa)")
    print(head)
    for label, group in (("Human", "human"), ("LLM", "llm")):
        for ds in datasets + ["All"]:
            sub = [r for r in rows if r["rater_group"] == group and (ds == "All" or r["dataset"] == ds)]
            print(f"{ds:<14}{label:<8}" + "".join(
                f"{cell(fleiss_kappa([r for r in sub if r['dimension'] == d]), '.3f'):>18}" for d in dims))


# ── Command line ─────────────────────────────────────────────────────────

def _read_texts(path: str) -> List[str]:
    text = Path(path).read_text()
    if path.endswith(".jsonl"):
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [str(r.get("sentence") or r.get("text") or "") for r in records]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _encode(texts: Sequence[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL).encode(list(texts), convert_to_numpy=True, show_progress_bar=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="AutoKD open-ended quality evaluation")
    sub = ap.add_subparsers(dest="command", required=True)

    sel = sub.add_parser("select", help="Top-K diverse insights from an insight graph")
    sel.add_argument("--graph", required=True)
    sel.add_argument("--diversity-threshold", type=float, required=True,
                     help="Skip candidates whose cosine similarity to a selected insight exceeds this")
    sel.add_argument("--k", type=int, default=10)
    sel.add_argument("--out", help="Write the selection as JSONL")

    par = sub.add_parser("pair", help="Hungarian pairing of system findings with gold claims")
    par.add_argument("--system", required=True)
    par.add_argument("--gold", required=True)

    ana = sub.add_parser("analyze", help="Win rates and Fleiss' kappa from pairwise ratings")
    ana.add_argument("--ratings", required=True)

    args = ap.parse_args()

    if args.command == "select":
        insights = load_discovered_insights(args.graph)
        chosen = select_top_k(insights, _encode([i["sentence"] for i in insights]), args.k, args.diversity_threshold)
        records = [{
            "id": insights[i]["id"],
            "sentence": insights[i]["sentence"],
            "score": round(ranking_score(insights[i]), 4),
        } for i in chosen]
        for rank, rec in enumerate(records, 1):
            print(f"{rank:>2}. [{rec['score']:.3f}] {rec['sentence']}")
        if args.out:
            Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in records))
    elif args.command == "pair":
        system, gold = _read_texts(args.system), _read_texts(args.gold)
        if len(system) != len(gold):
            raise SystemExit(f"need equal-length lists, got {len(system)} system and {len(gold)} gold")
        for s, g, sim in sorted(pair_with_gold(_encode(system), _encode(gold))):
            print(f"{s + 1:>2}. [{sim:.2f}] {system[s]}\n    gold: {gold[g]}")
    else:
        print_tables(load_ratings(args.ratings))


if __name__ == "__main__":
    main()
