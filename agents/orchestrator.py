"""Orchestrator Agent - coordinates discovery rounds."""

import random
from typing import List, Optional, Dict, Tuple
from core.types import RoundContext, RoundResult, Insight
from agents.base_agent import BaseAgent
from agents.historian import Historian
from core.config import config


class Orchestrator(BaseAgent):
    """Orchestrator that coordinates rounds and decides focus areas."""
    
    def __init__(self, historian: Historian):
        super().__init__("orchestrator")
        self.historian = historian
        self.round_count = 0

        # Read from top-level orchestrator config (config.yaml: orchestrator: ...)
        self.exploration_weight = config.get("orchestrator.exploration_weight", 0.3)
        self.refinement_weight = config.get("orchestrator.refinement_weight", 0.4)
        self.conflict_resolution_weight = config.get("orchestrator.conflict_resolution_weight", 0.3)
        self.global_exploration_weight = config.get("orchestrator.global_exploration_weight", 0.15)
        self.global_exploration_min_insights = config.get("orchestrator.global_exploration_min_insights", 12)
        self.max_rounds = config.get("orchestrator.max_rounds", 100)
        
        # Conflict detection is estimand-matched (see _detect_conflicts), so the
        # orchestrator.conflict_*_threshold config keys are not read.

        # Ablations A1 (no graph structure) and A3 (global exploration only); see config.yaml: ablation
        self.no_graph_structure = bool(config.get("ablation.no_graph_structure", False))
        self.global_mode_only = bool(config.get("ablation.global_mode_only", False))
        # Ablation A0: no memory across rounds, so no anchors in any mode.
        self.no_persistence = bool(config.get("ablation.no_persistence", False))

    def start_round(
        self,
        goal: str,
        mode: Optional[str] = None
    ) -> RoundContext:
        """Start a new discovery round."""
        self.round_count += 1
        
        if self.round_count > self.max_rounds:
            raise StopIteration("Maximum rounds reached")
        
        if mode is None:
            mode = self._decide_mode()
        
        focus_area, anchor_insight_ids = self._choose_focus(mode)
        
        context_insights = self.historian.get_context(anchor_insight_ids)
        
        return RoundContext(
            round_id=self.round_count,
            focus_area=focus_area,
            anchor_insight_ids=anchor_insight_ids,
            retrieved_insights=context_insights,
            goal=goal,
            mode=mode
        )
    
    def _decide_mode(self) -> str:
        """Decide what mode the next round should be in based on graph state."""
        # Ablation A3: every round is global exploration with no anchors.  Checked
        # before any graph inspection so it also holds in round 1; the insight graph
        # is still built and read by retrieval.
        if self.global_mode_only:
            return "global_exploration"

        all_insights = self.historian.get_all_insights()

        if not all_insights:
            # First round - always exploration
            return "exploration"
        
        stats = self.historian.get_statistics()
        num_insights = stats['num_insights']
        avg_degree = stats['avg_degree']
        
        conflicts = self._detect_conflicts(all_insights)
        has_conflicts = len(conflicts) > 0
        
        # Calculate dynamic weights based on graph state
        exploration_weight = self.exploration_weight
        refinement_weight = self.refinement_weight
        conflict_weight = self.conflict_resolution_weight
        global_weight = self.global_exploration_weight

        # Gate global exploration until we have enough stored knowledge to "avoid the map"
        try:
            min_ins = int(self.global_exploration_min_insights or 0)
        except (TypeError, ValueError):
            min_ins = 0
        if min_ins > 0 and num_insights < min_ins:
            global_weight = 0.0
        
        # Adjust weights based on graph structure (cluster + chain aware)
        num_clusters = stats.get('num_clusters', 1)
        chain_count = stats.get('chain_count', 0)

        if self.no_graph_structure:
            # Ablation A1: cluster/chain statistics are degenerate without edges
            # (num_clusters == num_insights, chain_count == 0), so only the size-based
            # early-exploration rule applies.
            if num_insights < 10:
                exploration_weight *= 2.0
                global_weight *= 0.2
        elif num_clusters <= 3 and num_insights < 10:
            # Early stage: few clusters, sparse — explore broadly
            exploration_weight *= 2.0
            global_weight *= 0.2
        elif chain_count < num_insights * 0.15:
            # Many insights but few DEEPENS chains — boost refinement
            # to encourage assoc → interact → heterogeneity deepening
            refinement_weight *= 1.5
        elif num_clusters < 5 and chain_count > num_insights * 0.3:
            # Dense clusters with many chains — seek new territory
            global_weight *= 1.5
            exploration_weight *= 1.2
        
        # Boost conflict resolution if conflicts exist
        if has_conflicts:
            conflict_weight *= 1.5
            global_weight *= 0.5
        else:
            # Reduce conflict resolution weight if no conflicts
            conflict_weight *= 0.3
        
        total_weight = exploration_weight + refinement_weight + conflict_weight + global_weight
        if total_weight <= 0:
            return "exploration"
        exploration_weight /= total_weight
        refinement_weight /= total_weight
        conflict_weight /= total_weight
        global_weight /= total_weight
        
        modes = ["exploration", "refinement", "conflict_resolution", "global_exploration"]
        weights = [exploration_weight, refinement_weight, conflict_weight, global_weight]
        
        return random.choices(modes, weights=weights)[0]
    
    @staticmethod
    def _insight_modules(insight: Insight) -> set:
        """Return the set of module names an insight's columns belong to.

        First checks for ``card_modules`` in insight metadata (populated by the
        variable catalog for any dataset).  Falls back to hardcoded WVS Q-number
        ranges for backward compatibility.
        """
        _SKIP = {"identifiers", "demographics", "demographics and ses"}
        meta = insight.metadata or {}

        # --- Fast path: catalog-provided module mapping ---
        card_modules = meta.get("card_modules")
        if card_modules:
            mods = set()
            for mod in card_modules.values():
                if mod and mod.lower() not in _SKIP:
                    mods.add(mod)
            return mods

        # --- Fallback: WVS Q-number ranges ---
        import re
        _RANGES = [
            ((1, 45), "Social values"), ((46, 56), "Wellbeing"),
            ((57, 105), "Trust"), ((106, 111), "Economic"),
            ((112, 120), "Corruption"), ((121, 130), "Migration"),
            ((131, 151), "Security"), ((152, 157), "Postmaterialism"),
            ((158, 163), "Science"), ((164, 175), "Religion"),
            ((176, 198), "Ethics"), ((199, 234), "Political participation"),
            ((235, 259), "Political culture"), ((260, 290), "Demographics"),
        ]
        cols = meta.get("columns", [])
        mods = set()
        for c in cols:
            m = re.match(r"Q(\d+)", c)
            if m:
                qn = int(m.group(1))
                for (lo, hi), name in _RANGES:
                    if lo <= qn <= hi and name != "Demographics":
                        mods.add(name)
                        break
        return mods

    def _choose_focus(self, mode: str) -> tuple[str, List[str]]:
        """Choose focus area and anchor insights based on mode.

        For exploration mode, uses graph structure to find:
        1. Insights at module boundaries (connected to insights from different modules)
        2. Isolated insights (degree 0)
        3. Under-explored insights (low degree)
        """
        # Ablation A0: anchoring on a stored insight IS persistence — it carries
        # a prior round's finding into this round's prompt.  Return no anchors in
        # every mode.  Placed before get_all_insights() so the arm never reads
        # the store at all during focus selection.
        if self.no_persistence:
            return ("independent round (no memory, no anchors)", [])

        all_insights = self.historian.get_all_insights()

        if mode == "global_exploration":
            return ("global exploration (no anchors)", [])

        if not all_insights:
            return ("initial exploration", [])

        if mode == "exploration":
            from core.types import RelationType
            insight_graph = self.historian.insight_graph

            # ── Depth frontiers (30%): assoc insights or research briefs with no DEEPENS child ──
            # These are prime targets for interact / heterogeneity deepening.
            # Seed briefs (cross_module_profile) are also eligible — they describe
            # variable pairs that haven't been formally tested yet.
            _frontier_types = {"assoc", ""}  # "" catches seeds with no test_type
            _frontier_ktypes = {"cross_module_profile", "within_module_profile"}
            depth_frontiers = []
            for insight in all_insights:
                tt = insight.metadata.get("test_type", "")
                kt = insight.metadata.get("knowledge_type", "")
                if tt not in _frontier_types and kt not in _frontier_ktypes:
                    continue
                if self.no_graph_structure:
                    # Ablation A1: without edges we cannot tell which shallow
                    # insights have already been deepened, so every shallow
                    # insight stays a candidate.
                    depth_frontiers.append(insight)
                    continue
                has_deepens_child = any(
                    data.get("relation_type") == RelationType.DEEPENS
                    for _, _, data in insight_graph.graph.in_edges(insight.id, data=True)
                )
                if not has_deepens_child:
                    depth_frontiers.append(insight)

            # ── Module boundary insights (40%): insights with EXTENDS edges ──
            boundary_insights = []
            for insight in all_insights:
                modules = self._insight_modules(insight)
                if self.no_graph_structure:
                    # Ablation A1: no neighbours exist to compare modules
                    # against, so use the node's own cross-module span — the
                    # closest structure-free analogue of a boundary insight.
                    if len(modules) >= 2:
                        boundary_insights.append(insight)
                    continue
                neighbors = insight_graph.get_neighbors(insight.id, max_depth=1)
                for nb in neighbors:
                    nb_mods = self._insight_modules(nb)
                    if nb_mods and modules and not nb_mods.issubset(modules):
                        boundary_insights.append(insight)
                        break

            # ── Module saturation: deprioritize over-represented modules ──
            # Count insights per module to avoid anchor bias toward dominant themes.
            from collections import Counter as _Ctr
            _mod_counts = _Ctr()
            for ins in all_insights:
                for m in self._insight_modules(ins):
                    _mod_counts[m] += 1
            _n_ins = max(len(all_insights), 1)

            def _is_saturated(ins):
                """True if ALL modules of this insight are over-represented (>25%)."""
                mods = self._insight_modules(ins)
                if not mods:
                    return False
                return all(_mod_counts.get(m, 0) / _n_ins > 0.25 for m in mods)

            # Filter saturated insights from candidates (keep at least 30%)
            def _prefer_unsaturated(candidates):
                unsaturated = [c for c in candidates if not _is_saturated(c)]
                return unsaturated if len(unsaturated) >= max(1, len(candidates) * 0.3) else candidates

            # ── Probabilistic selection ──
            roll = random.random()
            if boundary_insights and roll < 0.4:
                anchor = random.choice(_prefer_unsaturated(boundary_insights))
                return (f"exploring cross-module boundary: {anchor.sentence[:50]}...", [anchor.id])

            if depth_frontiers and roll < 0.7:
                anchor = random.choice(_prefer_unsaturated(depth_frontiers))
                return (f"exploring depth frontier (no DEEPENS child): {anchor.sentence[:50]}...", [anchor.id])

            # ── Isolated / under-explored bands (edge-degree based) ──
            # Ablation A1 skips both: with no edges every node has degree 0, so
            # these bands would silently become "random insight" and blur the
            # remaining probability mass.  The ablation falls through to the
            # explicit random pick below instead.
            if not self.no_graph_structure:
                # ── Isolated insights (20%) ──
                isolated = [
                    ins for ins in all_insights
                    if insight_graph.graph.degree(ins.id) == 0
                ]
                if isolated and roll < 0.9:
                    anchor = random.choice(isolated)
                    return (f"exploring isolated insight: {anchor.sentence[:50]}...", [anchor.id])

                # ── Fallback (10%): random under-explored ──
                under_explored = [
                    ins for ins in all_insights
                    if insight_graph.graph.degree(ins.id) < 3
                ]
                if under_explored:
                    anchor = random.choice(_prefer_unsaturated(under_explored))
                    return (f"exploring under-explored area: {anchor.sentence[:50]}...", [anchor.id])

            anchor = random.choice(_prefer_unsaturated(all_insights))
            return (f"exploring around: {anchor.sentence[:50]}...", [anchor.id])
        
        elif mode == "refinement":
            from core.types import RelationType
            insight_graph = self.historian.insight_graph
            scored_insights = []

            for insight in all_insights:
                overall_score = insight.metadata.get('overall_score', 0.5)
                validity_score = insight.metadata.get('validity_score', 0.5)
                test_type = insight.metadata.get('test_type', '')

                # Chain-aware depth bonus: assoc insights without DEEPENS
                # children are prime targets for interact/heterogeneity deepening.
                # Ablation A1 has no edges, so "already deepened" is unknowable
                # and the bonus is dropped entirely (ranking by score alone).
                depth_bonus = 0.0
                if not self.no_graph_structure:
                    has_deepens_child = any(
                        data.get("relation_type") == RelationType.DEEPENS
                        for _, _, data in insight_graph.graph.in_edges(insight.id, data=True)
                    )
                    if test_type == "assoc" and not has_deepens_child:
                        depth_bonus = 0.25  # Best deepening target
                    elif test_type == "interact" and not has_deepens_child:
                        depth_bonus = 0.15  # Can be deepened to heterogeneity

                combined_score = (overall_score * 0.6 + validity_score * 0.4) + depth_bonus
                scored_insights.append((insight, combined_score))

            scored_insights.sort(key=lambda x: x[1], reverse=True)

            if scored_insights:
                top_score = scored_insights[0][1]
                top_threshold = top_score * 0.7
                candidates = [
                    insight for insight, score in scored_insights
                    if score >= top_threshold
                ]

                if candidates:
                    anchor = random.choice(candidates)
                    # Follow DEEPENS chain for richer context
                    chain_ids = [anchor.id]
                    if not self.no_graph_structure:
                        chain_ancestors = insight_graph.get_chain_ancestors(anchor.id, max_depth=3)
                        for anc in chain_ancestors:
                            if anc.id not in chain_ids:
                                chain_ids.append(anc.id)
                    return (f"refining chain from: {anchor.sentence[:50]}...", chain_ids)
            
            # Fallback: pick random
            anchor = random.choice(all_insights)
            return (f"refining: {anchor.sentence[:50]}...", [anchor.id])
        
        elif mode == "conflict_resolution":
            # Find potentially contradictory insights
            conflicts = self._detect_conflicts(all_insights)
            if conflicts:
                conflict_pair = random.choice(conflicts)
                return (
                    f"resolving conflict between insights",
                    [conflict_pair[0].id, conflict_pair[1].id]
                )
            else:
                # No conflicts found, pick random
                anchor = random.choice(all_insights)
                return (f"exploring for potential conflicts: {anchor.sentence[:50]}...", [anchor.id])
        
        # Default
        if all_insights:
            anchor = random.choice(all_insights)
            return ("general discovery", [anchor.id])
        
        return ("initial exploration", [])
    
    def _detect_conflicts(self, insights: List[Insight]) -> List[Tuple[Insight, Insight]]:
        """Find insight pairs that genuinely contradict each other.

        A conflict is two insights that estimated the **same quantity** — same
        test type, same X/Y pair, same moderator and grouping variable — and
        reported opposite, statistically credible effects.  This is the exact
        rule the edge classifier uses for CONTRADICTS
        (``core.relation_classifier.contradicts``), so the mode the loop enters
        and the edge the graph records cannot disagree.

        Candidate pairs are found by grouping insights by estimand, so no
        pairwise embedding comparison is needed.
        """
        from core.relation_classifier import contradicts, estimand_key
        from itertools import combinations

        by_estimand: Dict[tuple, List[Insight]] = {}
        for insight in insights:
            key = estimand_key(insight)
            if key is None:
                # No parsed DSL roles (seed briefs, pre-fix graphs): the
                # estimand is unknowable, so never assert a conflict.
                continue
            by_estimand.setdefault(key, []).append(insight)

        conflicts: List[Tuple[Insight, Insight]] = []
        for members in by_estimand.values():
            if len(members) < 2:
                continue
            for insight1, insight2 in combinations(members, 2):
                if contradicts(insight1, insight2):
                    conflicts.append((insight1, insight2))

        return conflicts
    
    def plan_next_round(
        self,
        round_result: RoundResult
    ) -> Optional[RoundContext]:
        """Plan the next round based on current round results."""
        if not round_result.accepted_insights:
            # No insights accepted - try exploration
            return self.start_round(round_result.summary, mode="exploration")
        else:
            # Some insights accepted - continue refining
            return self.start_round(round_result.summary, mode="refinement")
    
    def process(self, action: str, **kwargs) -> any:
        """Process orchestrator actions."""
        if action == "start_round":
            return self.start_round(
                kwargs.get("goal", ""),
                kwargs.get("mode")
            )
        elif action == "plan_next_round":
            return self.plan_next_round(kwargs["round_result"])
        else:
            raise ValueError(f"Unknown action: {action}")


