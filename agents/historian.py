"""Historian Agent - manages Discovery Memory."""

from typing import List, Optional, Tuple
from memory.insight_graph import InsightGraph
from core.types import Insight, InsightRelation, RelationType, InsightScore
from agents.base_agent import BaseAgent
from core.config import config


class Historian(BaseAgent):
    """Historian agent with read/write access to Discovery Memory."""
    
    def __init__(self, insight_graph: InsightGraph):
        super().__init__("historian")
        self.insight_graph = insight_graph
        self.max_context = self.get_config("max_context_insights", 10)
        # duplicate_threshold is read from the global memory config, not the agent config
        self.duplicate_threshold = config.get("memory.duplicate_threshold", 0.92)
        self.quality_weight = self.get_config("quality_weight_in_context", 0.3)
        # Ablation A1: flat insight store with no relations (see config.yaml: ablation)
        self.no_graph_structure = bool(config.get("ablation.no_graph_structure", False))
        # Ablation A0: no memory across rounds (see config.yaml: ablation).
        self.no_persistence = bool(config.get("ablation.no_persistence", False))
        # Ablation A0: number of stores the bypassed duplicate check would have
        # collapsed into an existing node (rediscoveries).
        self.rediscovery_count: int = 0

    def get_context(
        self,
        anchor_insight_ids: List[str],
        max_insights: Optional[int] = None,
        relation_types: Optional[List[RelationType]] = None
    ) -> List[Insight]:
        """Retrieve relevant context around anchor insights.

        Uses tiered retrieval when *relation_types* is not specified:
        priority DEEPENS chain > NARROWS > EXTENDS > CONTRADICTS, filling the
        budget incrementally so chain context is always included first.
        """
        # Ablation A0: the loop carries nothing between rounds, so no prior
        # insight may enter this round's prompt.  Checked before the anchor
        # guard so it holds even if a caller passes anchors.
        if self.no_persistence:
            return []

        if not anchor_insight_ids:
            return []

        max_insights = max_insights or self.max_context

        # Ablation A1: no relations exist, so tiered traversal would return only
        # the anchors.  Fill the same budget by flat semantic similarity instead.
        if self.no_graph_structure:
            return self._get_context_flat(anchor_insight_ids, max_insights)

        # If caller specified explicit relation_types, honour that
        if relation_types is not None:
            context = self.insight_graph.get_context_around(
                anchor_insight_ids,
                max_insights=max_insights,
                relation_types=relation_types,
            )
        else:
            # Tiered retrieval: DEEPENS → NARROWS → EXTENDS → CONTRADICTS
            context: List[Insight] = []
            seen: set = set()

            # Always include anchors first
            for aid in anchor_insight_ids:
                ins = self.insight_graph.get_insight(aid)
                if ins and ins.id not in seen:
                    context.append(ins)
                    seen.add(ins.id)

            tiers = [
                ([RelationType.DEEPENS], 2),    # chain context, deeper traversal
                ([RelationType.NARROWS], 1),     # same-theme conditions
                ([RelationType.EXTENDS], 1),     # cross-theme for diversity
                ([RelationType.CONTRADICTS], 1), # opposing findings on the same estimand
            ]
            for rel_types, depth in tiers:
                if len(context) >= max_insights:
                    break
                for aid in anchor_insight_ids:
                    for nb in self.insight_graph.get_neighbors(
                        aid, relation_types=rel_types, max_depth=depth
                    ):
                        if nb.id not in seen and len(context) < max_insights:
                            context.append(nb)
                            seen.add(nb.id)

        # Optionally reorder by quality if quality_weight > 0
        if self.quality_weight > 0 and context:
            context = self._reorder_by_quality(context, anchor_insight_ids)

        return context
    
    def _get_context_flat(
        self,
        anchor_insight_ids: List[str],
        max_insights: int
    ) -> List[Insight]:
        """Ablation A1 retrieval: flat embedding neighbourhood, no edges.

        Deliberately mirrors get_context() in everything except how neighbours
        are chosen: anchors first, then the most cosine-similar stored insights
        until the SAME budget is filled, then the same quality reorder.  This
        keeps context quantity constant so the comparison isolates the value of
        typed relational structure over plain vector similarity.
        """
        context: List[Insight] = []
        seen: set = set()

        for aid in anchor_insight_ids:
            ins = self.insight_graph.get_insight(aid)
            if ins and ins.id not in seen:
                context.append(ins)
                seen.add(ins.id)

        for aid in anchor_insight_ids:
            if len(context) >= max_insights:
                break
            anchor = self.insight_graph.get_insight(aid)
            if anchor is None:
                continue
            # threshold=0.0 → rank all insights by similarity; anchors are
            # filtered out by `seen` (they would otherwise self-match at ~1.0).
            for cand, _sim in self.insight_graph.find_similar_insights(
                anchor.sentence,
                threshold=0.0,
                max_results=max_insights * 3,
            ):
                if cand.id in seen:
                    continue
                context.append(cand)
                seen.add(cand.id)
                if len(context) >= max_insights:
                    break

        if self.quality_weight > 0 and context:
            context = self._reorder_by_quality(context, anchor_insight_ids)

        return context

    def _reorder_by_quality(
        self,
        context: List[Insight],
        anchor_insight_ids: List[str]
    ) -> List[Insight]:
        """Reorder context insights to balance relevance and quality."""
        if not context or self.quality_weight == 0:
            return context
        
        anchor_set = set(anchor_insight_ids)
        scored = []
        
        for insight in context:
            # Relevance score: 1.0 if anchor, 0.5 if neighbor
            relevance = 1.0 if insight.id in anchor_set else 0.5
            
            # Quality score from metadata
            overall_score = insight.metadata.get("overall_score", 0.5)
            validity_score = insight.metadata.get("validity_score", 0.5)
            quality = (overall_score * 0.6) + (validity_score * 0.4)
            
            # Combined score: relevance (1 - quality_weight) + quality (quality_weight)
            combined = (relevance * (1 - self.quality_weight)) + (quality * self.quality_weight)
            scored.append((insight, combined))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [insight for insight, _ in scored]
    
    @staticmethod
    def _extract_core_vars(dsl: str, all_cols: set) -> frozenset:
        """Extract core DSL variables (X, Y, group) before 'controlling for'."""
        import re
        core = dsl.split(" controlling for")[0].strip()
        core = re.sub(r"\s*weights\s*=\s*\S+", "", core).strip()
        # Extract column names that appear in the core part
        found = set()
        for col in all_cols:
            if col in core:
                found.add(col)
        return frozenset(found) if found else frozenset(all_cols)

    def find_similar_insights(
        self,
        sentence: str,
        threshold: float = 0.7,
        max_results: int = 5
    ) -> List[Tuple[Insight, float]]:
        """Find insights similar to a given sentence.
        
        Args:
            sentence: Sentence to find similar insights for
            threshold: Minimum similarity threshold (0-1)
            max_results: Maximum number of results to return
            
        Returns:
            List of (insight, similarity_score) tuples, sorted by similarity
        """
        if not sentence or not sentence.strip():
            return []
        
        return self.insight_graph.find_similar_insights(sentence, threshold, max_results)
    
    def find_structurally_similar_insights(
        self,
        columns: List[str],
        threshold: float = 0.8,
        skip_sources: Optional[List[str]] = None,
    ) -> List[Insight]:
        """Find insights that use a similar set of columns (Jaccard similarity).

        Args:
            columns: List of column names to check against
            threshold: Minimum Jaccard similarity (0-1) to consider similar
            skip_sources: Insight sources to exclude (e.g. ["seed_prescan"])

        Returns:
            List of insights that use similar column sets
        """
        if not columns:
            return []

        _skip = set(skip_sources) if skip_sources else set()
        target_cols = set(columns)
        results = []

        all_insights = self.get_all_insights()
        for insight in all_insights:
            if _skip and insight.source in _skip:
                continue
            # Use CORE columns (X, Y, group, moderator) if parsed spec is
            # available; otherwise fall back to all columns.  This ensures
            # that two insights testing the same X→Y with different controls
            # are recognized as duplicates.
            parsed = (insight.metadata or {}).get("parsed")
            if parsed:
                insight_cols = set()
                for key in ("x", "y", "group", "moderator"):
                    val = parsed.get(key)
                    if val:
                        insight_cols.add(val)
            else:
                insight_cols = set(insight.metadata.get("columns", []))
            if not insight_cols:
                continue

            # Jaccard Similarity: Intersection / Union
            intersection = len(target_cols.intersection(insight_cols))
            union = len(target_cols.union(insight_cols))

            if union > 0:
                similarity = intersection / union
                if similarity >= threshold:
                    results.append(insight)

        return results
    
    def inject_insight(
        self,
        insight: Insight,
        related_insight_ids: List[str],
        relation_types: List[RelationType]
    ) -> str:
        """Inject a new insight into Discovery Memory with relations.
        
        Args:
            insight: The insight to inject
            related_insight_ids: List of related insight IDs
            relation_types: List of relation types (must match length of related_insight_ids)
            
        Returns:
            The ID of the injected insight (or existing ID if duplicate)
            
        Raises:
            ValueError: If relation_types length doesn't match related_insight_ids
        """
        if not insight or not insight.sentence or not insight.sentence.strip():
            raise ValueError("Insight must have a non-empty sentence")
        
        if len(related_insight_ids) != len(relation_types):
            raise ValueError(
                f"Number of relation_types ({len(relation_types)}) must match "
                f"number of related_insight_ids ({len(related_insight_ids)})"
            )
        
        # Check for duplicates before adding:
        # structural signature first, semantic similarity second (filtered by structure).
        # Compare on CORE variables only (X, Y, group) — ignore controls.
        new_cols = set((insight.metadata or {}).get("columns", []) or [])
        new_type = (insight.metadata or {}).get("test_type", "")
        _dsl = (insight.metadata or {}).get("dsl_sentence", "") or insight.sentence
        new_core = self._extract_core_vars(_dsl, new_cols)
        similar = self.find_similar_insights(
            insight.sentence,
            threshold=self.duplicate_threshold,
            max_results=5,
        )
        if similar:
            filtered = []
            for cand, score in similar:
                cmeta = cand.metadata or {}
                ccols = set(cmeta.get("columns", []) or [])
                ctype = cmeta.get("test_type", "")
                _cdsl = cmeta.get("dsl_sentence", "") or cand.sentence
                cand_core = self._extract_core_vars(_cdsl, ccols)
                if new_core and cand_core and new_type and ctype:
                    # Match on core variables (X, Y, group), ignoring controls
                    if ctype == new_type and new_core == cand_core:
                        filtered.append((cand, score))
                elif new_cols and ccols and new_type and ctype:
                    # Fallback to full-column Jaccard for non-DSL insights
                    inter = len(new_cols.intersection(ccols))
                    union = len(new_cols.union(ccols))
                    jacc = (inter / union) if union > 0 else 0.0
                    if ctype == new_type and jacc >= 0.67:
                        filtered.append((cand, score))
                elif score >= max(self.duplicate_threshold, 0.98):
                    # Fallback when structural metadata is missing on either side.
                    filtered.append((cand, score))
            if filtered:
                existing = sorted(filtered, key=lambda x: x[1], reverse=True)[0][0]
                # Ablation A0: store the node separately and count the rediscovery.  Other
                # arms return the existing node's id here, which would hide rediscoveries
                # and force the exact-duplicate rate to 0 by construction.
                if self.no_persistence:
                    self.rediscovery_count += 1
                    insight.metadata = dict(insight.metadata or {})
                    insight.metadata["rediscovery_of"] = existing.id
                    insight.metadata["rediscovery_round_gap"] = (
                        int((insight.metadata.get("round_id") or 0))
                        - int(((existing.metadata or {}).get("round_id") or 0))
                    )
                    print(f"    [REDISCOVERY] #{self.rediscovery_count}: "
                          f"{insight.sentence[:55]}... (first seen in "
                          f"{existing.id}, round "
                          f"{(existing.metadata or {}).get('round_id')})")
                else:
                    self._maybe_update_metadata(existing, insight)
                    return existing.id

        insight_id = self.insight_graph.add_insight(insight)

        # Ablation A1: flat store — the node is kept, relations are never written.
        if self.no_graph_structure:
            return insight_id

        # Add relations to related insights
        relations_added = 0
        for related_id, rel_type in zip(related_insight_ids, relation_types):
            if related_id in self.insight_graph.insights:
                try:
                    self.insight_graph.add_relation(
                        source_id=insight_id,
                        target_id=related_id,
                        relation_type=rel_type
                    )
                    relations_added += 1
                except ValueError as e:
                    # Relation might already exist or be invalid, skip
                    print(f"[WARN] Could not add relation {insight_id} -> {related_id}: {e}")
        
        if relations_added == 0 and related_insight_ids:
            print(f"[WARN] No relations added for insight {insight_id} (related insights may not exist)")

        # Structural pass: link relationships the round's context never showed us.
        self._link_deepening_parents(insight_id)

        return insight_id

    # Maximum deepening parents linked per insight.
    MAX_DEEPENING_PARENTS = 3

    def _link_deepening_parents(self, insight_id: str) -> int:
        """Link a new insight to the findings it deepens, anchors aside.

        Edges built from the round's context insights
        (`Critic._find_related_insights`) record which findings the generator
        was shown, not which findings the new one actually builds on.  A
        deepening parent's core variables are a strict subset of the new
        insight's, at a lower test-type rank.

        This pass runs *after* the insight is stored, so it compares complete
        metadata (`parsed`, `card_modules`, `effect_size`), which is strictly
        more than the Critic sees at judging time.

        Deliberately limited to deepening parents: linking thematic peers by
        shared variables would add many edges that `classify_relation` labels
        NARROWS almost without exception, lowering the meaning carried per edge.

        Returns the number of edges added.
        """
        source = self.insight_graph.get_insight(insight_id)
        if source is None:
            return 0

        from core.relation_classifier import (
            classify_relation, core_columns, test_type_rank,
        )

        src_core = core_columns(source)
        src_rank = test_type_rank((source.metadata or {}).get("test_type", ""))
        if not src_core or src_rank < 2:
            # Nothing shallower to deepen (assoc/diff are already the floor).
            return 0

        existing_targets = set(self.insight_graph.graph.successors(insight_id))

        parents = []
        for cand in self.insight_graph.get_all_insights():
            if cand.id == insight_id or cand.id in existing_targets:
                continue
            cand_rank = test_type_rank((cand.metadata or {}).get("test_type", ""))
            if not 0 < cand_rank < src_rank:
                continue
            cand_core = core_columns(cand)
            # Strict subset: the parent asks a simpler question about the same
            # variables.  Equality is not deepening, it is a duplicate.
            if not cand_core or not cand_core < src_core:
                continue
            parents.append((cand, cand_rank, len(cand_core)))

        # Nearest parents first: the deepest, largest-core ancestor is the
        # direct one, so a chain forms link by link rather than skipping levels.
        parents.sort(key=lambda p: (-p[1], -p[2]))

        added = 0
        for cand, _, _ in parents[:self.MAX_DEEPENING_PARENTS]:
            try:
                self.insight_graph.add_relation(
                    source_id=insight_id,
                    target_id=cand.id,
                    relation_type=classify_relation(source, cand),
                    save=False,
                )
                added += 1
            except ValueError as e:
                print(f"[WARN] Could not add structural relation "
                      f"{insight_id} -> {cand.id}: {e}")

        if added:
            self.insight_graph.save()
        return added

    def _maybe_update_metadata(self, existing: Insight, new: Insight):
        """Update existing insight metadata if new insight has better scores."""
        updated = False
        
        # Update overall_score if new is better
        existing_score = existing.metadata.get("overall_score", 0.0)
        new_score = new.metadata.get("overall_score", 0.0)
        if new_score > existing_score:
            existing.metadata["overall_score"] = new_score
            updated = True
        
        # Update validity_score if new is better
        existing_validity = existing.metadata.get("validity_score", 0.0)
        new_validity = new.metadata.get("validity_score", 0.0)
        if new_validity > existing_validity:
            existing.metadata["validity_score"] = new_validity
            updated = True
        
        # Update round_id to most recent
        existing_round = existing.metadata.get("round_id", 0)
        new_round = new.metadata.get("round_id", 0)
        if new_round > existing_round:
            existing.metadata["round_id"] = new_round
            updated = True
        
        if updated:
            self.insight_graph.save()
    
    def get_insight(self, insight_id: str) -> Optional[Insight]:
        """Get an insight by ID.
        
        Args:
            insight_id: The ID of the insight to retrieve
            
        Returns:
            The insight if found, None otherwise
        """
        if not insight_id:
            return None
        return self.insight_graph.get_insight(insight_id)
    
    def get_all_insights(self) -> List[Insight]:
        """Get all insights in Discovery Memory.
        
        Returns:
            List of all insights, ordered by creation time (oldest first)
        """
        insights = self.insight_graph.get_all_insights()
        insights.sort(key=lambda x: x.created_at)
        return insights
    
    def get_insights_by_quality(
        self,
        min_score: float = 0.0,
        max_results: Optional[int] = None
    ) -> List[Insight]:
        """Get insights filtered by quality score.
        
        Args:
            min_score: Minimum overall_score to include
            max_results: Maximum number of results to return
            
        Returns:
            List of insights with score >= min_score, sorted by score (descending)
        """
        all_insights = self.insight_graph.get_all_insights()
        
        scored = []
        for insight in all_insights:
            overall_score = insight.metadata.get("overall_score", 0.0)
            if overall_score >= min_score:
                scored.append((insight, overall_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        results = [insight for insight, _ in scored]
        if max_results:
            results = results[:max_results]
        
        return results
    
    def get_statistics(self) -> dict:
        """Get statistics about Discovery Memory.
        
        Returns:
            Dictionary with statistics including:
            - num_insights: Total number of insights
            - num_relations: Total number of relations
            - avg_degree: Average number of connections per insight
            - relation_types: Count of each relation type
        """
        return self.insight_graph.get_statistics()
    
    def validate_insight(self, insight: Insight) -> Tuple[bool, Optional[str]]:
        """Validate an insight before injection.
        
        Args:
            insight: The insight to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not insight:
            return False, "Insight is None"
        
        if not insight.sentence or not insight.sentence.strip():
            return False, "Insight sentence is empty"
        
        if len(insight.sentence.strip()) < 10:
            return False, "Insight sentence is too short (minimum 10 characters)"
        
        if len(insight.sentence) > 500:
            return False, "Insight sentence is too long (maximum 500 characters)"
        
        if not insight.id:
            return False, "Insight ID is missing"
        
        return True, None
    
    def process(self, action: str, **kwargs) -> any:
        """Process various historian actions.
        
        Supported actions:
            - "get_context": Get context around anchor insights
            - "inject_insight": Inject a new insight with relations
            - "find_similar": Find similar insights
            - "get_insight": Get insight by ID
            - "get_all_insights": Get all insights
            - "get_statistics": Get graph statistics
            - "validate_insight": Validate an insight
            
        Raises:
            ValueError: If action is unknown or required parameters are missing
        """
        if action == "get_context":
            return self.get_context(
                kwargs.get("anchor_insight_ids", []),
                kwargs.get("max_insights"),
                kwargs.get("relation_types")
            )
        elif action == "inject_insight":
            return self.inject_insight(
                kwargs["insight"],
                kwargs.get("related_insight_ids", []),
                kwargs.get("relation_types", [])
            )
        elif action == "find_similar":
            return self.find_similar_insights(
                kwargs["sentence"],
                kwargs.get("threshold", 0.7),
                kwargs.get("max_results", 5)
            )
        elif action == "get_insight":
            return self.get_insight(kwargs.get("insight_id"))
        elif action == "get_all_insights":
            return self.get_all_insights()
        elif action == "get_statistics":
            return self.get_statistics()
        elif action == "validate_insight":
            return self.validate_insight(kwargs.get("insight"))
        else:
            raise ValueError(f"Unknown action: {action}. Supported: get_context, inject_insight, find_similar, get_insight, get_all_insights, get_statistics, validate_insight")


