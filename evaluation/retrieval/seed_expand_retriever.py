"""Seed-and-expand retrieval over the insight graph (knowledge extraction).

Given a natural-language query:
  1. Embed the query once and compute its cosine similarity to every
     discovered insight (cold-start seed nodes are not retrievable).
  2. Select the ``k_seeds`` most similar insights as seed nodes.
  3. Expand the seeds by one hop along graph edges (both directions, any
     relation type) to gather structurally related candidates.
  4. Score every candidate by a weighted combination of semantic similarity
     and stored validity, and return the top K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from memory.insight_graph import InsightGraph
from core.types import Insight

# Nodes written by cold-start seeding (or loaded as references) are not findings.
_NON_FINDING_SOURCES = frozenset({"statistical_profiling", "seed_prescan", "paper"})


def is_discovered(insight: Insight) -> bool:
    """True for insights accepted by the discovery loop, False for seeds and references."""
    if insight.source in _NON_FINDING_SOURCES:
        return False
    return not (insight.metadata or {}).get("is_seed")


@dataclass
class RetrievedItem:
    insight: Insight
    score: float        # similarity_weight * similarity + validity_weight * validity
    similarity: float   # cosine similarity to the query
    validity: float     # validity score stored on the insight
    is_seed: bool       # whether the insight was selected as a similarity seed
    rank: int = 0


@dataclass
class SeedExpandConfig:
    k_seeds: int = 5                 # number of similarity seeds
    similarity_weight: float = 0.6   # weight on semantic similarity
    validity_weight: float = 0.4     # weight on stored validity
    include_predecessors: bool = True
    include_successors: bool = True


class SeedExpandRetriever:
    """Seed by similarity, expand one hop along graph edges, rank by similarity and validity."""

    def __init__(self, insight_graph: InsightGraph, config: SeedExpandConfig | None = None):
        self.graph = insight_graph
        self.config = config or SeedExpandConfig()

    def retrieve(self, query_text: str, k: int = 5) -> List[RetrievedItem]:
        """Return the top-``k`` insights for ``query_text``."""
        hits = self.graph.find_similar_insights(
            sentence=query_text, threshold=-1.0, max_results=10**9,
        )
        hits = [(ins, sim) for ins, sim in hits if is_discovered(ins)]
        if not hits:
            return []
        similarity: Dict[str, float] = {ins.id: float(sim) for ins, sim in hits}

        seed_ids = [ins.id for ins, _ in hits[: self.config.k_seeds]]
        candidates: Set[str] = set(seed_ids)
        for sid in seed_ids:
            candidates.update(self._neighbors(sid))

        scored: List[RetrievedItem] = []
        for cid in candidates:
            ins = self.graph.get_insight(cid)
            if ins is None:
                continue
            sim = similarity.get(cid, 0.0)
            validity = float((ins.metadata or {}).get("validity_score") or 0.0)
            scored.append(RetrievedItem(
                insight=ins,
                score=self.config.similarity_weight * sim + self.config.validity_weight * validity,
                similarity=sim,
                validity=validity,
                is_seed=cid in seed_ids,
            ))
        scored.sort(key=lambda r: r.score, reverse=True)
        scored = scored[:k]
        for i, item in enumerate(scored, 1):
            item.rank = i
        return scored

    def _neighbors(self, node_id: str) -> Iterable[str]:
        """One-hop neighbors of ``node_id`` in both directions, discovered insights only."""
        g = self.graph.graph
        if node_id not in g:
            return []
        adjacent: Set[str] = set()
        if self.config.include_successors:
            adjacent.update(g.successors(node_id))
        if self.config.include_predecessors:
            adjacent.update(g.predecessors(node_id))
        out: Set[str] = set()
        for nid in adjacent:
            ins = self.graph.get_insight(nid)
            if ins is not None and is_discovered(ins):
                out.add(nid)
        return out
