"""Insight Graph - persistent graph storage for Discovery Memory."""

import json
import math
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from datetime import datetime
import networkx as nx
import numpy as np

from core.types import Insight, InsightRelation, RelationType

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None


class InsightGraph:
    """Graph-based storage for Discovery Memory."""
    
    def __init__(
        self, 
        storage_path: str = "./data/memory/insight_graph.json",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()
        self.insights: Dict[str, Insight] = {}
        
        # Embedding settings - required, no fallback
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required but not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.embedding_model_name = embedding_model
        self._embedding_model: Optional[SentenceTransformer] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}  # Cache embeddings by insight ID
        
        self._load()
    
    def _load(self):
        """Load graph from storage."""
        if self.storage_path.exists():
            if self.storage_path.suffix == '.json':
                self._load_json()
            elif self.storage_path.suffix == '.pkl':
                self._load_pickle()
        else:
            # Initialize with seed knowledge if this is first run
            self._initialize_seed_knowledge()
    
    def _load_json(self):
        """Load graph from JSON file."""
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
            
        # Load insights
        for insight_data in data.get('insights', []):
            insight = Insight(
                id=insight_data['id'],
                sentence=insight_data['sentence'],
                source=insight_data['source'],
                created_at=datetime.fromisoformat(insight_data['created_at']),
                metadata=insight_data.get('metadata', {})
            )
            self.insights[insight.id] = insight
            self.graph.add_node(insight.id, insight=insight)
            
            # Embeddings will be computed on-demand in find_similar_insights
        
        # Load relations (skip legacy 'similar_to' type and ignore 'strength' field)
        for rel_data in data.get('relations', []):
            relation_type_str = rel_data['relation_type']
            # Skip legacy 'similar_to' relation type
            if relation_type_str == 'similar_to':
                continue
            # Migrate legacy relation types
            if relation_type_str == 'supports':
                relation_type_str = 'extends'
            if relation_type_str == 'refines':
                relation_type_str = 'narrows'
            try:
                relation_type = RelationType(relation_type_str)
            except ValueError:
                # Skip unknown relation types
                continue
            
            self.graph.add_edge(
                rel_data['source_id'],
                rel_data['target_id'],
                relation_type=relation_type,
                metadata=rel_data.get('metadata', {})
            )
    
    def _load_pickle(self):
        """Load graph from pickle file."""
        with open(self.storage_path, 'rb') as f:
            data = pickle.load(f)
            self.graph = data.get('graph', nx.DiGraph())
            self.insights = data.get('insights', {})
    
    def _initialize_seed_knowledge(self):
        """Initialize with some seed knowledge from literature/web."""
        # This can be populated with initial insights
        pass
    
    def _get_embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model. Required - raises error if unavailable."""
        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"[OK] Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.embedding_model_name}': {e}. "
                    "Ensure sentence-transformers is installed and the model name is correct."
                ) from e
        
        return self._embedding_model
    
    def _get_insight_embedding(self, insight_id: str, sentence: str) -> np.ndarray:
        """Get embedding for an insight, using cache if available."""
        # Check cache first
        if insight_id in self._embedding_cache:
            return self._embedding_cache[insight_id]
        
        # Generate embedding
        model = self._get_embedding_model()
        embedding = model.encode(sentence, convert_to_numpy=True)
        self._embedding_cache[insight_id] = embedding
        return embedding
    
    def _compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings, normalized to [0, 1] range."""
        # Cosine similarity (range [-1, 1])
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        # Normalize to [0, 1] range (maps [-1, 1] → [0, 1])
        return (cosine_sim + 1) / 2
    
    def save(self):
        """Save graph to storage."""
        if self.storage_path.suffix == '.json':
            self._save_json()
        elif self.storage_path.suffix == '.pkl':
            self._save_pickle()
    
    def _save_json(self):
        """Save graph to JSON file."""
        data = {
            'insights': [
                {
                    'id': insight.id,
                    'sentence': insight.sentence,
                    'source': insight.source,
                    'created_at': insight.created_at.isoformat(),
                    'metadata': insight.metadata
                }
                for insight in self.insights.values()
            ],
            'relations': [
                {
                    'source_id': source,
                    'target_id': target,
                    'relation_type': self.graph[source][target]['relation_type'].value,
                    'metadata': self.graph[source][target].get('metadata', {})
                }
                for source, target in self.graph.edges()
            ]
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_pickle(self):
        """Save graph to pickle file."""
        data = {
            'graph': self.graph,
            'insights': self.insights
        }
        with open(self.storage_path, 'wb') as f:
            pickle.dump(data, f)
    
    def add_insight(self, insight: Insight) -> str:
        """Add a new insight to the graph."""
        self.insights[insight.id] = insight
        self.graph.add_node(insight.id, insight=insight)
        
        # Pre-compute embedding
        self._get_insight_embedding(insight.id, insight.sentence)
        
        self.save()
        return insight.id
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        metadata: Optional[Dict] = None,
        save: bool = True
    ):
        """Add a relation between two insights.

        Set ``save=False`` when writing a batch of edges and call ``save()``
        once afterwards — each save rewrites the whole graph file.
        """
        if source_id not in self.insights or target_id not in self.insights:
            raise ValueError("Both insights must exist in the graph")

        self.graph.add_edge(
            source_id,
            target_id,
            relation_type=relation_type,
            metadata=metadata or {}
        )
        if save:
            self.save()
    
    def get_insight(self, insight_id: str) -> Optional[Insight]:
        """Get an insight by ID."""
        return self.insights.get(insight_id)
    
    def get_neighbors(
        self,
        insight_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 1,
        direction: str = "both",
    ) -> List[Insight]:
        """Get neighboring insights (connected by relations).

        Parameters
        ----------
        direction : str
            ``"out"`` follows only successors (old behaviour),
            ``"in"`` follows only predecessors,
            ``"both"`` follows both directions (new default).
        """
        if insight_id not in self.graph:
            return []

        neighbors: List[Insight] = []
        visited = {insight_id}

        def _adjacent(node_id: str):
            """Yield (neighbor_id, edge_data) in the requested direction."""
            if direction in ("out", "both"):
                for nid in self.graph.successors(node_id):
                    yield nid, self.graph[node_id][nid]
            if direction in ("in", "both"):
                for nid in self.graph.predecessors(node_id):
                    yield nid, self.graph[nid][node_id]

        def collect_neighbors(node_id: str, depth: int):
            if depth > max_depth:
                return
            for neighbor_id, edge_data in _adjacent(node_id):
                if neighbor_id in visited:
                    continue
                rel_type = edge_data.get('relation_type')
                if relation_types is None or rel_type in relation_types:
                    if neighbor_id in self.insights:
                        neighbors.append(self.insights[neighbor_id])
                        visited.add(neighbor_id)
                        if depth < max_depth:
                            collect_neighbors(neighbor_id, depth + 1)

        collect_neighbors(insight_id, 0)
        return neighbors
    
    def get_context_around(
        self,
        anchor_ids: List[str],
        max_insights: int = 10,
        relation_types: Optional[List[RelationType]] = None
    ) -> List[Insight]:
        """Get context around anchor insights for a round."""
        context = []
        seen_ids: Set[str] = set()
        
        # Start with anchor insights
        for anchor_id in anchor_ids:
            if anchor_id in self.insights:
                context.append(self.insights[anchor_id])
                seen_ids.add(anchor_id)
        
        # Add neighbors
        for anchor_id in anchor_ids:
            neighbors = self.get_neighbors(anchor_id, relation_types, max_depth=2)
            for neighbor in neighbors:
                if neighbor.id not in seen_ids and len(context) < max_insights:
                    context.append(neighbor)
                    seen_ids.add(neighbor.id)
        
        return context[:max_insights]
    
    def find_similar_insights(
        self,
        sentence: str,
        threshold: float = 0.7,
        max_results: int = 5
    ) -> List[tuple[Insight, float]]:
        """Find insights similar to a given sentence using sentence embeddings."""
        if not self.insights:
            return []
        
        # Get embedding for query sentence
        model = self._get_embedding_model()
        query_embedding = model.encode(sentence, convert_to_numpy=True)
        
        similarities = []
        for insight in self.insights.values():
            # Get embedding for insight (from cache or compute)
            insight_embedding = self._get_insight_embedding(insight.id, insight.sentence)
            
            # Compute similarity
            similarity = self._compute_similarity(query_embedding, insight_embedding)
            
            if similarity >= threshold:
                similarities.append((insight, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def get_all_insights(self) -> List[Insight]:
        """Get all insights in the graph."""
        return list(self.insights.values())
    
    def get_statistics(self) -> Dict:
        """Get statistics about the graph."""
        num_insights = len(self.insights)

        # Cluster info (weakly-connected components, cheap O(V+E))
        if num_insights > 0:
            components = list(nx.weakly_connected_components(self.graph))
            num_clusters = len(components)
            largest_cluster_size = max(len(c) for c in components)
        else:
            num_clusters = 0
            largest_cluster_size = 0

        # DEEPENS chain count: edges with DEEPENS type
        chain_count = sum(
            1 for _, _, d in self.graph.edges(data=True)
            if d.get('relation_type') == RelationType.DEEPENS
        )

        # Max in-degree (anti-hub monitoring)
        max_in_degree = 0
        if num_insights > 0:
            in_degrees = dict(self.graph.in_degree())
            max_in_degree = max(in_degrees.values()) if in_degrees else 0

        return {
            'num_insights': num_insights,
            'num_relations': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / num_insights if num_insights else 0,
            'max_in_degree': max_in_degree,
            'num_clusters': num_clusters,
            'largest_cluster_size': largest_cluster_size,
            'chain_count': chain_count,
            'relation_types': {
                rel_type.value: sum(
                    1 for _, _, data in self.graph.edges(data=True)
                    if data.get('relation_type') == rel_type
                )
                for rel_type in RelationType
            }
        }

    # ── New graph-structural helpers ─────────────────────────────────

    def get_chain_ancestors(
        self,
        insight_id: str,
        max_depth: int = 5,
    ) -> List[Insight]:
        """Follow DEEPENS edges *forward* (successors) to find the chain
        of progressive deepening this insight participates in.

        Since edges point new → old, the **targets** of DEEPENS edges are
        the *simpler* ancestors (assoc that was deepened into interact).
        """
        chain: List[Insight] = []
        visited = {insight_id}
        current = insight_id
        for _ in range(max_depth):
            found_next = False
            for succ in self.graph.successors(current):
                edge_data = self.graph[current][succ]
                if edge_data.get('relation_type') == RelationType.DEEPENS:
                    if succ not in visited and succ in self.insights:
                        chain.append(self.insights[succ])
                        visited.add(succ)
                        current = succ
                        found_next = True
                        break
            if not found_next:
                break
        return chain

    def get_cluster_members(self, insight_id: str) -> List[Insight]:
        """Return all insights in the same weakly-connected component."""
        if insight_id not in self.graph:
            return []
        for component in nx.weakly_connected_components(self.graph):
            if insight_id in component:
                return [
                    self.insights[nid]
                    for nid in component
                    if nid in self.insights
                ]
        return []

    def find_related_insights_composite(
        self,
        sentence: str,
        columns: List[str],
        test_type: str,
        max_results: int = 3,
        embedding_threshold: float = 0.72,
        max_in_degree: int = 15,
    ) -> List[Tuple[Insight, float, RelationType]]:
        """Find related insights using composite multi-signal scoring.

        Two-tier approach:
        - Tier 1 (chain parent): Find the best insight with same core
          variables at a lower test-type depth (→ DEEPENS edge).
        - Tier 2 (thematic peers): Score remaining candidates with
          embedding similarity + column overlap + anti-hub penalty
          (→ EXTENDS or NARROWS edge).

        Returns list of ``(insight, composite_score, relation_type)``.
        """
        from core.relation_classifier import (
            classify_relation, core_columns as _core_columns,
            test_type_rank, insight_modules as _insight_modules,
        )

        if not self.insights:
            return []

        model = self._get_embedding_model()
        query_embedding = model.encode(sentence, convert_to_numpy=True)

        src_cols = set(columns or [])
        src_core = _core_columns(Insight(
            id="__query__", sentence=sentence, source="",
            created_at=datetime.now(),
            metadata={"columns": columns, "test_type": test_type},
        ))
        src_rank = test_type_rank(test_type)

        # Compute similarities for all candidates above threshold
        candidates: List[Tuple[Insight, float]] = []
        for insight in self.insights.values():
            emb = self._get_insight_embedding(insight.id, insight.sentence)
            sim = self._compute_similarity(query_embedding, emb)
            if sim >= embedding_threshold and sim < 0.92:
                candidates.append((insight, sim))

        if not candidates:
            return []

        # ── Tier 1: chain parent (max 1) ───────────────────────────
        chain_parent: Optional[Tuple[Insight, float]] = None
        if src_rank > 1:
            best_chain_sim = -1.0
            for ins, sim in candidates:
                tgt_core = _core_columns(ins)
                tgt_rank = test_type_rank(
                    (ins.metadata or {}).get("test_type", "")
                )
                # Target must be at lower depth and its core cols are a
                # subset of source's core cols (source deepens target)
                if 0 < tgt_rank < src_rank and tgt_core and tgt_core <= src_core:
                    if sim > best_chain_sim:
                        best_chain_sim = sim
                        chain_parent = (ins, sim)

        results: List[Tuple[Insight, float, RelationType]] = []
        chain_parent_id: Optional[str] = None

        if chain_parent is not None:
            ins, sim = chain_parent
            results.append((ins, sim, RelationType.DEEPENS))
            chain_parent_id = ins.id

        # ── Tier 2: thematic peers (up to max_results - len(results)) ──
        remaining = max_results - len(results)
        if remaining <= 0:
            return results

        scored_peers: List[Tuple[Insight, float, RelationType]] = []
        for ins, sim in candidates:
            if ins.id == chain_parent_id:
                continue
            # Anti-hub hard cap
            in_deg = self.graph.in_degree(ins.id) if ins.id in self.graph else 0
            if in_deg >= max_in_degree:
                continue

            # Composite score
            s_emb = (sim - embedding_threshold) / (0.92 - embedding_threshold)
            s_emb = max(0.0, min(1.0, s_emb))

            tgt_core = _core_columns(ins)
            if src_core and tgt_core:
                s_col = len(src_core & tgt_core) / len(src_core | tgt_core)
            else:
                s_col = 0.0

            p_hub = 1.0 / (1.0 + math.log1p(in_deg))

            composite = 0.40 * s_emb + 0.30 * s_col + 0.30 * p_hub

            # Classify relation
            src_insight = Insight(
                id="__query__", sentence=sentence, source="",
                created_at=datetime.now(),
                metadata={"columns": columns, "test_type": test_type},
            )
            rel_type = classify_relation(src_insight, ins)

            scored_peers.append((ins, composite, rel_type))

        scored_peers.sort(key=lambda x: x[1], reverse=True)
        results.extend(scored_peers[:remaining])
        return results


