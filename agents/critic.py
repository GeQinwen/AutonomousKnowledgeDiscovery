"""Critic Agent - judges value, novelty, and makes final decisions."""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
from core.types import (
    Hypothesis,
    Evaluation,
    InsightScore,
    Insight,
    RelationType
)
from agents.base_agent import BaseAgent
from agents.historian import Historian
from core.llm_client import create_llm_client
from core.config import config
from core.llm_parse import extract_json, extract_json_or


class Critic(BaseAgent):
    """Critic agent that judges novelty and decides what to store."""
    
    def __init__(self, historian: Historian):
        super().__init__("critic")
        self.historian = historian
        self.min_insight_score = self.get_config("min_insight_score", 0.6)
        self.novelty_weight = self.get_config("novelty_weight", 0.5)
        self.validity_weight = self.get_config("validity_weight", 0.4)
        self.interpretability_weight = self.get_config("interpretability_weight", 0.1)
        
        # Domain description should come from config; keep a survey-safe default.
        self.domain_description = self.get_config(
            "domain_description",
            "Cross-national survey data (e.g., World Values Survey)"
        )
        # Mirror of the evaluator's effect floor: smaller effects are never accepted.
        self.min_meaningful_effect_size = self.get_config("min_meaningful_effect_size", 0.02)
        
        # Initialize LLM client for surprise scoring
        llm_config = config.get_llm_config()
        self.llm = create_llm_client(llm_config)

        # Last-call artifacts (best-effort, for run artifact logging)
        self.last_pre_novelty_call: Optional[Dict[str, Any]] = None
        self.last_surprise_call: Optional[Dict[str, Any]] = None
    
    def estimate_novelty(
        self,
        hypothesis: Hypothesis,
        context,
        retrieved_insights: Optional[List[Insight]] = None,
    ) -> Tuple[float, str]:
        """
        LLM-only novelty estimate BEFORE running experiments.
        
        Returns:
            (novelty_score in [0.0, 1.0], reason string)
        If parsing fails, we fall back to novelty=0.7 so we don't
        accidentally drop everything just because of JSON issues.
        """
        if retrieved_insights is None:
            retrieved_insights = []
        
        # Take a small subset of existing insight sentences as context
        existing_snippets = "\n".join(
            f"- {ins.sentence}" for ins in retrieved_insights[:10]
        )
        
        prompt = f"""
You are acting as a scientific peer reviewer judging the *novelty* of a single
hypothesis about this dataset domain:
{self.domain_description}

Overall goal:
{context.goal}

Existing related insights (may already be known):
{existing_snippets if existing_snippets else "(none)"}

Candidate hypothesis:
"{hypothesis.sentence}"

Rate ONLY its *novelty* relative to the existing insights and to common-sense
expectations about such data.

Respond in strict JSON with fields:
- "novelty": a number between 0.0 and 1.0
- "reason": a short string explanation.

Example:
{{"novelty": 0.72, "reason": "Combines reviewer activity and product format in a non-trivial way."}}
""".strip()
        
        try:
            raw = self.llm.generate(
                prompt,
                temperature=0.2,
                max_tokens=256,
                system="You are a scientific peer reviewer. Output valid JSON only."
            )
            self.last_pre_novelty_call = {"prompt": prompt, "response": raw}

            data = extract_json_or(
                raw,
                fallback={"novelty": 0.5, "reason": "parse_failed"},
            )
            novelty = float(data.get("novelty", 0.5))
            reason = str(data.get("reason", "")).strip()
            if reason == "parse_failed":
                print(f"  [WARN] Critic.estimate_novelty: JSON parse failed, using fallback novelty=0.5")
        except Exception as e:
            print(f"  [WARN] Critic.estimate_novelty failed: {e}")
            # Fallback: neutral score so parse errors don't systematically
            # inflate or kill novelty.
            novelty, reason = 0.5, f"Fallback novelty=0.5 due to error: {e}"
        
        # Clamp to [0, 1]
        novelty = max(0.0, min(1.0, novelty))
        
        # Cache into hypothesis.metadata if it is a dict
        if hasattr(hypothesis, "metadata") and isinstance(hypothesis.metadata, dict):
            hypothesis.metadata["pre_novelty_score"] = novelty
            hypothesis.metadata["pre_novelty_reason"] = reason
        
        return novelty, reason
    
    def judge(
        self,
        hypothesis: Hypothesis,
        evaluation: Evaluation,
        context_insights: List[Insight]
    ) -> InsightScore:
        """Judge a hypothesis and return insight score."""
        novelty_score = self._assess_novelty(hypothesis, context_insights)
        
        interpretability_score = self._assess_interpretability(hypothesis)
        
        overall_score = (
            novelty_score * self.novelty_weight +
            evaluation.validity_score * self.validity_weight +
            interpretability_score * self.interpretability_weight
        )
        
        decision = self._make_decision(overall_score, evaluation, novelty_score)
        
        related_insight_ids = self._find_related_insights(hypothesis, context_insights)
        
        explanation = self._generate_explanation(
            hypothesis,
            overall_score,
            novelty_score,
            evaluation.validity_score,
            related_insight_ids
        )
        
        return InsightScore(
            hypothesis_id=hypothesis.id,
            overall_score=overall_score,
            novelty_score=novelty_score,
            usefulness_score=0.0,  # Not used anymore
            validity_score=evaluation.validity_score,
            interpretability_score=interpretability_score,
            explanation=explanation,
            decision=decision,
            related_insight_ids=related_insight_ids
        )
    
    def _assess_novelty(self, hypothesis: Hypothesis, context_insights: List[Insight]) -> float:
        """Assess novelty using Structural, Embedding Distance, AND Epistemic Surprise."""
        
        # 1. Structural Novelty (Column-based) - Check if we've seen these columns before
        structural_score = self._assess_structural_novelty(hypothesis)
        
        # 2. Semantic Novelty (Embeddings) - Fast check
        embedding_score = self._get_embedding_novelty(hypothesis, context_insights)
        
        # 3. Epistemic Novelty (LLM Surprise) - Deep check
        surprise_data = self._get_epistemic_surprise(hypothesis, context_insights)
        
        # Extract score (0-10 scale -> 0-1 scale)
        llm_score = surprise_data.get("score", 5.0) / 10.0
        
        hypothesis.metadata["critic_reasoning"] = surprise_data.get("reasoning", "")
        hypothesis.metadata["critic_critique"] = surprise_data.get("critique", "")
        hypothesis.metadata["critic_mechanism"] = surprise_data.get("mechanism", "")
        hypothesis.metadata["surprise_class"] = surprise_data.get("classification", "UNKNOWN")
        hypothesis.metadata["llm_surprise_raw"] = surprise_data.get("score", 5.0)

        # If LLM says "TAUTOLOGICAL" (score < 3), force reject regardless.
        if llm_score < 0.3:
            return 0.0

        # Same-column hypotheses are penalized as derivative, except deliberate
        # deepenings, which reuse a stored finding's variables by design.
        is_deepening = bool(hypothesis.metadata.get("is_deepening"))

        if structural_score < 0.3 and not is_deepening:
            # Blend: structural penalty dominates but LLM can partially rescue
            novelty = structural_score * 0.4 + embedding_score * 0.1 + llm_score * 0.5
            return min(novelty, 0.40)  # cap so truly derivative findings stay low

        # TRIVIAL gate: an LLM score below 4 with a TRIVIAL classification caps
        # novelty low.
        if llm_score < 0.4 and surprise_data.get("classification") == "TRIVIAL":
            return max(llm_score, 0.20)

        # Weighted combination: structural 25%, semantic 25%, LLM 50%
        novelty = (structural_score * 0.25) + (embedding_score * 0.25) + (llm_score * 0.50)

        return novelty
    
    @staticmethod
    def _q_number(var: str) -> int:
        """Extract numeric part of a Q-variable (e.g., 'Q113' → 113). Returns -1 if not a Q-var."""
        import re
        m = re.match(r"Q(\d+)", var)
        return int(m.group(1)) if m else -1

    def _assess_structural_novelty(self, hypothesis: Hypothesis) -> float:
        """Return novelty score based on column overlap with existing insights.

        Returns:
            0.0-1.0: 1.0 = completely new columns, 0.0 = exact same columns
        """
        new_cols = set(hypothesis.metadata.get("columns", []))
        if not new_cols:
            return 0.5  # Neutral if parser failed or no columns extracted

        # Same-battery penalty: if any pair of Q-variables in the hypothesis
        # has a Q-number gap ≤ 3, it's likely testing same-construct items.
        # This is a safety net for anything that leaks past validation.
        q_nums = sorted(set(self._q_number(c) for c in new_cols if self._q_number(c) >= 0))
        if len(q_nums) >= 2:
            min_gap = min(q_nums[i+1] - q_nums[i] for i in range(len(q_nums) - 1))
            if min_gap <= 3:
                return 0.1  # Same-battery → force very low structural novelty

        # Same-module penalty: if all core variables share the same
        # thematic module, this is a within-module finding.  Cross-module
        # discoveries are far more valuable.
        from core.relation_classifier import insight_modules as _get_modules
        from core.types import Insight as _Ins
        from datetime import datetime as _dt
        try:
            _tmp = _Ins(id="__check__", sentence="", source="",
                        created_at=_dt.now(),
                        metadata=hypothesis.metadata)
            mods = _get_modules(_tmp)
            if len(mods) == 1 and len(new_cols) >= 2:
                return 0.15  # Same module → low structural novelty
        except Exception:
            pass

        all_insights = self.historian.get_all_insights()
        if not all_insights:
            return 1.0  # No existing insights = high novelty

        max_overlap = 0.0
        exact_matches = 0

        for insight in all_insights:
            existing_cols = set(insight.metadata.get("columns", []))
            if not existing_cols:
                continue

            # Exact match of columns = very low novelty
            if new_cols == existing_cols:
                exact_matches += 1
                max_overlap = 1.0
                continue

            # Jaccard similarity: intersection / union
            intersection = len(new_cols.intersection(existing_cols))
            union = len(new_cols.union(existing_cols))

            if union > 0:
                jaccard = intersection / union
                max_overlap = max(max_overlap, jaccard)

        # If exact match found, return very low novelty
        if exact_matches > 0:
            return 0.1  # Same columns = structurally derivative

        # Map overlap to novelty: high overlap → low novelty
        # max_overlap of 0.8+ means 80%+ column overlap = low novelty
        # max_overlap of 0.0 means no overlap = high novelty
        structural_novelty = max(0.0, 1.0 - max_overlap)

        return structural_novelty
    
    def _get_embedding_novelty(self, hypothesis: Hypothesis, context_insights: List[Insight]) -> float:
        """Get novelty score based on embedding similarity against the full graph."""
        all_insights = self.historian.get_all_insights()
        if not all_insights:
            return 1.0  # No existing insights = high novelty

        # Check for highly similar insights in the full graph
        similar = self.historian.find_similar_insights(hypothesis.sentence, threshold=0.6, max_results=5)

        if not similar:
            return 0.9  # Nothing similar found — genuinely new

        top_score = similar[0][1]

        if top_score >= 0.85:
            return 0.2  # Very similar to existing — low novelty
        elif top_score >= 0.7:
            return 0.6  # Refinement range
        else:
            return 0.75  # Loosely related — moderate-high novelty
    
    def _is_refinement(self, new_sentence: str, existing_sentence: str) -> bool:
        """Check if new hypothesis refines an existing insight (0.6-0.85 similarity)."""
        # Compute embedding similarity between new hypothesis and the specific existing insight
        try:
            model = self.historian.insight_graph._get_embedding_model()
            emb_new = model.encode(new_sentence, convert_to_numpy=True)
            emb_existing = model.encode(existing_sentence, convert_to_numpy=True)
            sim = self.historian.insight_graph._compute_similarity(emb_new, emb_existing)
            return 0.6 <= sim < 0.85
        except Exception:
            return False

    def _is_support(self, new_sentence: str, existing_sentence: str) -> bool:
        """Check if new hypothesis supports an existing insight (0.7-0.9 similarity)."""
        try:
            model = self.historian.insight_graph._get_embedding_model()
            emb_new = model.encode(new_sentence, convert_to_numpy=True)
            emb_existing = model.encode(existing_sentence, convert_to_numpy=True)
            sim = self.historian.insight_graph._compute_similarity(emb_new, emb_existing)
            return 0.7 <= sim < 0.9
        except Exception:
            return False
    
    def _get_epistemic_surprise(self, hypothesis: Hypothesis, context_insights: List[Insight]) -> Dict[str, Any]:
        """
        Uses LLM to judge epistemic surprise vs common sense & context.
        Returns JSON: {score, classification, reasoning, critique}
        """
        if not self.llm.is_available():
            return {"score": 5.0, "classification": "UNKNOWN", "reasoning": "LLM offline", "critique": ""}
        
        known_facts = "\n".join([f"- {i.sentence}" for i in context_insights[:10]])
        if not known_facts:
            known_facts = "No prior knowledge available."

        # Prefer human-readable sentence (with variable labels) over raw DSL
        display_sentence = hypothesis.metadata.get("readable_sentence", hypothesis.sentence)

        prompt = f"""You are a research scientist building a knowledge base of empirical findings from a dataset.
Your task is to evaluate whether a newly discovered pattern is scientifically meaningful.

## INPUT DATA

1. **The Hypothesis:** "{display_sentence}"
2. **Dataset Domain:** {self.domain_description}
3. **Existing Knowledge:**
{known_facts}

## YOUR TASK

First, ask yourself: **what mechanism could explain this relationship?**
If you cannot articulate a plausible causal or theoretical pathway in 1-2 sentences, the finding is INCOHERENT regardless of statistical significance.
Then judge how novel and valuable the finding is.

## SCORING RUBRIC (0-10)

**0-2: TAUTOLOGICAL**
- Variables measure the same underlying construct, or the relationship is a coding artifact, definition, or mathematical identity.
- Classification: TRIVIAL. Action: REJECT.

**3-4: INCOHERENT**
- No mechanism explaining why these variables would be related.
- Combination is arbitrary; no researcher would hypothesize this.
- Classification: TRIVIAL. Action: REJECT.

**5-6: FOUNDATIONAL**
- Clear mechanism; direction a domain researcher would predict.
- Quantifies a known phenomenon; confirming it across contexts is valuable.
- Classification: FOUNDATIONAL. Action: ACCEPT.

**7-8: NOVEL**
- Plausible mechanism; relationship not commonly tested.
- Bridges research areas, reveals a hidden moderator, or quantifies an assumed-but-unverified effect.
- Classification: NOVEL. Action: STRONG ACCEPT.

**9-10: SURPRISING**
- Finding contradicts domain expectations or reveals a reversal/amplification in subgroups.
- Classification: SURPRISING. Action: STRONG ACCEPT.

## RESPONSE FORMAT (JSON ONLY)

{{ "mechanism": "<1-2 sentences on WHY these variables would relate, or 'none' if incoherent>",
"reasoning": "<step-by-step novelty analysis>",
"classification": "TRIVIAL | FOUNDATIONAL | NOVEL | SURPRISING",
"score": <float between 0.0 and 10.0>,
"critique": "<one specific suggestion to improve novelty>" }}
"""
        try:
            response = self.llm.generate(
                prompt, 
                temperature=0.0, 
                system="You are a strict scientific critic. Output valid JSON only."
            )
            self.last_surprise_call = {"prompt": prompt, "response": response}

            _fallback = {"score": 5.0, "classification": "UNKNOWN", "reasoning": "parse_failed", "critique": ""}
            data = extract_json_or(response, fallback=_fallback)
            # Validate and clamp score
            if "score" in data:
                data["score"] = max(0.0, min(10.0, float(data["score"])))
            if data.get("reasoning") == "parse_failed":
                print("    [WARN] Critic epistemic surprise: JSON parse failed, using neutral fallback")
            return data
        except Exception as e:
            print(f"    [WARN] Critic LLM failed: {e}")
            return {"score": 5.0, "classification": "UNKNOWN", "reasoning": f"Error: {e}", "critique": ""}
    
    def _assess_interpretability(self, hypothesis: Hypothesis) -> float:
        """Assess interpretability based on hypothesis structure, not formatting.

        Rewards cross-module scope, explicit controls, and complex test types
        (interaction/heterogeneity) which produce richer scientific findings.
        """
        dsl = hypothesis.metadata.get("dsl_sentence", hypothesis.sentence)
        score = 0.50  # baseline

        # Cross-module hypotheses compare constructs from different domains
        if hypothesis.metadata.get("is_cross_module"):
            score += 0.20

        # Explicit controls improve interpretability (rules out confounds)
        if "controlling for" in dsl:
            score += 0.15

        # Interaction/heterogeneity tests reveal moderators — richer findings
        if any(op in dsl for op in ["interact", "heterogeneity"]):
            score += 0.15

        return min(1.0, score)
    
    def _make_decision(self, overall_score: float, evaluation: Evaluation, novelty_score: float) -> str:
        """Make final decision: accept, refine, or reject.

        Args:
            overall_score: Combined score from all components
            evaluation: Statistical evaluation
            novelty_score: Novelty score (0-1)
        """
        validity = evaluation.validity_score

        # Hard validity floor: never accept findings with very weak
        # statistical evidence, even if novelty is high (prevents
        # non-significant results from entering the graph early on
        # when novelty is inflated due to an empty graph).
        if validity < 0.4:
            if overall_score >= 0.45:
                return "refine"
            return "reject"

        # Hard novelty floor: derivative findings are refined or rejected, never accepted.
        if novelty_score < 0.30:
            if validity >= 0.70 and overall_score >= 0.45:
                return "refine"
            return "reject"

        # Accept gate: insight threshold tau_s = min_insight_score.
        if overall_score >= self.min_insight_score:
            if abs(evaluation.effect_size) >= self.min_meaningful_effect_size:
                return "accept"
            return "refine"
        elif overall_score >= 0.45:
            return "refine"
        else:
            return "reject"
    
    def _find_related_insights(
        self,
        hypothesis: Hypothesis,
        context_insights: List[Insight]
    ) -> List[str]:
        """Connect this hypothesis back to the anchors that inspired it.

        Edges are built **only** to the anchor / context insights that the
        orchestrator selected and the generator used to create this
        hypothesis.  These are the insights that *causally* led to this
        finding, so the edges carry real epistemic meaning:

        - Refinement anchor + same vars + higher depth  → DEEPENS
        - Exploration anchor + different module          → EXTENDS
        - Same theme, added controls                    → NARROWS
        - Opposing direction on shared vars             → CONTRADICTS

        No global similarity search is performed — cross-module bridges
        arise naturally when the orchestrator picks cross-module anchors.

        Relation types are stored in
        ``hypothesis.metadata["_edge_relation_types"]`` for downstream use.
        """
        from core.relation_classifier import classify_relation
        from core.types import Insight as InsightType
        from datetime import datetime

        columns = hypothesis.metadata.get("columns", [])
        test_type = hypothesis.metadata.get("test_type", "")
        graph = self.historian.insight_graph

        src_insight = InsightType(
            id="__hyp__", sentence=hypothesis.sentence, source="",
            created_at=datetime.now(),
            metadata={
                "columns": columns,
                "test_type": test_type,
                "card_modules": hypothesis.metadata.get("card_modules", {}),
                # Without `parsed`, core_columns() falls back to all columns
                # (controls included) for the source while the target uses true
                # cores — an asymmetry that breaks the DEEPENS subset test.
                "parsed": hypothesis.metadata.get("parsed", {}),
                "effect_size": hypothesis.metadata.get("effect_size"),
                "p_value": hypothesis.metadata.get("p_value"),
                "readable_sentence": hypothesis.metadata.get("readable_sentence", ""),
            },
        )

        related_ids: List[str] = []
        edge_types: dict = {}

        anchor_ids = hypothesis.context_insight_ids or []
        for aid in anchor_ids:
            if len(related_ids) >= 3:
                break
            target = graph.get_insight(aid)
            if target is None:
                continue
            if aid not in edge_types:
                rel_type = classify_relation(src_insight, target)
                related_ids.append(aid)
                edge_types[aid] = rel_type

        # ── Deepening parents ────────────────────────────────────────────
        # A round's anchors are often unrelated to the finding a deepening
        # hypothesis builds on, so the assoc() being deepened would never be
        # compared with the new interact()/heterogeneity().  Only strict deepening
        # parents (core columns a proper subset, lower test-type rank) are added,
        # so every other relation type still comes from the anchors alone.
        from core.relation_classifier import test_type_rank

        src_rank = test_type_rank(test_type)
        # Compare CORE variables (x/y/group/moderator), not `columns`: controls
        # inflate the union and push a genuine parent below the similarity threshold.
        _hyp_parsed = hypothesis.metadata.get("parsed") or {}
        src_core = {_hyp_parsed.get(k) for k in ("x", "y", "group", "moderator") if _hyp_parsed.get(k)}
        src_core = {c for c in src_core if c}
        if src_rank >= 2 and src_core and len(related_ids) < 3:
            try:
                candidates = self.historian.find_structurally_similar_insights(
                    columns=sorted(src_core),
                    threshold=0.5,
                    skip_sources=["seed_prescan", "statistical_profiling"],
                )
            except Exception:
                candidates = []
            for cand in candidates:
                if len(related_ids) >= 3:
                    break
                if cand.id in edge_types:
                    continue
                c_meta = cand.metadata or {}
                c_parsed = c_meta.get("parsed") or {}
                c_core = {c_parsed.get(k) for k in ("x", "y", "group", "moderator") if c_parsed.get(k)}
                c_core = {c for c in c_core if c}
                if not c_core or not (c_core < src_core):
                    continue
                if test_type_rank(c_meta.get("test_type", "")) >= src_rank:
                    continue
                rel_type = classify_relation(src_insight, cand)
                related_ids.append(cand.id)
                edge_types[cand.id] = rel_type

        hypothesis.metadata["_edge_relation_types"] = edge_types
        return related_ids
    
    def _generate_explanation(
        self,
        hypothesis: Hypothesis,
        overall_score: float,
        novelty_score: float,
        validity_score: float,
        related_insight_ids: List[str]
    ) -> str:
        """Generate explanation for the judgment."""
        parts = []
        
        if novelty_score > 0.7:
            parts.append("High novelty - introduces new structure")
        elif novelty_score > 0.4:
            parts.append("Moderate novelty - refines or supports existing insights")
        else:
            parts.append("Low novelty - similar to existing insights")
        
        if validity_score > 0.7:
            parts.append("strong statistical validity")
        elif validity_score > 0.5:
            parts.append("moderate statistical validity")
        else:
            parts.append("weak statistical validity")
        
        if related_insight_ids:
            parts.append(f"relates to {len(related_insight_ids)} existing insights")
        
        return ". ".join(parts) + "."
    
    def process(
        self,
        hypothesis: Hypothesis,
        evaluation: Evaluation,
        context_insights: List[Insight]
    ) -> InsightScore:
        """Process: judge hypothesis."""
        return self.judge(hypothesis, evaluation, context_insights)





