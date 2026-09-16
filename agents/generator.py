"""Generator Agent - proposes new hypotheses."""

import uuid
import re
import random
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from core.types import Hypothesis, Insight, RoundContext
from agents.base_agent import BaseAgent
from core.llm_client import create_llm_client, parse_list_response
from core.config import config
from core.llm_parse import extract_json_or


class Generator(BaseAgent):
    """Generator agent that proposes new hypotheses."""
    
    def __init__(self):
        super().__init__("generator")
        self.batch_size = self.get_config("batch_size", 10)
        self.temperature = self.get_config("temperature", 0.75)
        self.max_context_insights = self.get_config("max_context_insights", 5)  # Show fewer existing facts (prevents anchoring)
        
        # Initialize LLM client — use a dedicated generator model if configured,
        # otherwise fall back to the main model.
        llm_config = config.get_llm_config()
        generator_model = self.get_config("model", None)
        if generator_model:
            llm_config = dict(llm_config)
            llm_config["model"] = generator_model
            print(f"[INFO] Generator using dedicated model: {generator_model}")
        self.llm = create_llm_client(llm_config)

        # Last-call artifacts (best-effort, for run artifact logging)
        self.last_generation: Optional[Dict[str, Any]] = None
        self.last_skeptic_edits: List[Dict[str, Any]] = []
        self.last_general_idea_call: Optional[Dict[str, Any]] = None
        self.last_refiner_call: Optional[Dict[str, Any]] = None
    
    def generate_hypotheses(
        self,
        context: RoundContext,
        rejected_sentences: Optional[List[str]] = None,
        rejection_summary: Optional[str] = None,
        insight_graph_summary: Optional[str] = None,
    ) -> List[Hypothesis]:
        """Generate a batch of new hypotheses based on context.
        
        Args:
            context: Round context with goal, focus, mode, etc.
            rejected_sentences: List of hypothesis sentences that were rejected as duplicates
                              in this round. These will be explicitly forbidden in the prompt.
            rejection_summary: LLM-generated summary of cross-round rejection history.
                              This provides high-level guidance on what to avoid.
            insight_graph_summary: LLM-generated summary of what is already stored in the
                                  insight graph (accepted/stored insights). This provides
                                  high-level guidance on what has already been explored.
        """
        # 1. Generate initial raw hypotheses
        prompt, system_prompt = self._build_prompts(
            context,
            rejected_sentences,
            rejection_summary,
            insight_graph_summary,
        )
        raw_hypotheses = self._generate_with_llm(prompt, system_prompt, context)

        # Capture best-effort prompt/response artifacts
        try:
            self.last_generation = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": self._get_mode_temperature(context.mode),
                "raw_response": getattr(self, "_last_llm_raw_response", None),
                "parsed_sentences": [h.sentence for h in raw_hypotheses],
            }
        except Exception:
            self.last_generation = None
        self.last_skeptic_edits = []
        
        # 2. Intra-batch de-duplication: remove structural duplicates within the batch
        deduplicated_hypotheses = self._deduplicate_batch(raw_hypotheses, context)
        if len(deduplicated_hypotheses) < len(raw_hypotheses):
            print(
                f"  [INFO] Removed {len(raw_hypotheses) - len(deduplicated_hypotheses)} duplicate hypotheses from batch"
            )
        
        # 3. Adversarial refinement (skeptic rewrite)
        refined_hypotheses = []
        print(f"  [INFO] The Skeptic is reviewing {len(deduplicated_hypotheses)} hypotheses...")
        
        for h in deduplicated_hypotheses:
            # Only refine if it looks simple (no "controlling for" or "interaction")
            if "control" not in h.sentence.lower() and "when" not in h.sentence.lower() and "accounting for" not in h.sentence.lower():
                original_sentence = h.sentence
                refined_h = self._refine_single_hypothesis(h, context)
                if refined_h.sentence != original_sentence:
                    # best-effort: record only the before/after (prompts can be very large)
                    self.last_skeptic_edits.append(
                        {
                            "hypothesis_id": refined_h.id,
                            "before": original_sentence,
                            "after": refined_h.sentence,
                            "ts": datetime.now().isoformat(),
                        }
                    )
                refined_hypotheses.append(refined_h)
            else:
                refined_hypotheses.append(h)
        
        # 4. Track which insights influenced each hypothesis
        final_hypotheses = self._link_to_context_insights(refined_hypotheses, context)
        
        return final_hypotheses
    
    def _build_prompts(
        self, 
        context: RoundContext, 
        rejected_sentences: Optional[List[str]] = None,
        rejection_summary: Optional[str] = None,
        insight_graph_summary: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Build mode-aware prompts for hypothesis generation."""

        # Parse schema once (used for dataset-aware prompt assembly)
        schema_cols: List[str] = []
        schema_desc: Dict[str, str] = {}
        if context.data_schema:
            column_lines = [line for line in context.data_schema.split("\n") if line.strip().startswith("-")]
            for line in column_lines:
                raw = line.strip().lstrip("-").strip()
                name = raw.split("(")[0].strip()
                desc = ""
                if "(" in raw and raw.endswith(")"):
                    desc = raw[raw.find("(") + 1 : -1].strip()
                if name:
                    schema_cols.append(name)
                    schema_desc[name] = desc

        text_like = [c for c in schema_cols if "text" in (schema_desc.get(c, "") or "").lower()]
        id_like = [
            c
            for c in schema_cols
            if c.lower().endswith("id") or c.lower().endswith("_id") or c.lower() in {"survey_id"}
        ]
        categorical_like = [
            c for c in schema_cols if "categorical" in (schema_desc.get(c, "") or "").lower()
        ]
        time_like = [
            c
            for c in schema_cols
            if any(k in c.lower() for k in ["year", "date", "time", "timestamp", "month", "day"])
        ]

        def _is_wvs_q(name: str) -> bool:
            return bool(re.match(r"^Q\d{1,3}([_][A-Z0-9]+)?$", name))

        has_wvs_like = any(_is_wvs_q(c) for c in schema_cols) or any(
            c in schema_cols for c in ["B_COUNTRY", "B_COUNTRY_ALPHA", "W_WEIGHT", "PWGHT"]
        )
        has_amazon_like = any(c in schema_cols for c in ["review_text", "helpful_votes", "reviewerID", "asin", "rating"])

        # Build data constraints section
        data_constraints = ""
        if context.data_schema:
            extra_notes = []
            if categorical_like:
                extra_notes.append(
                    "- Many columns are categorical codes; treat them as categories (group comparisons, chi-square, regression with dummies), not numeric magnitudes."
                )
            if id_like:
                extra_notes.append(
                    "- ID-like columns may be used for grouping/aggregation only; do NOT treat their numeric magnitude as meaningful."
                )
            if text_like:
                extra_notes.append(
                    "- Text-like columns may be used only via simple pandas string operations (len/contains/count). Do NOT assume external NLP models."
                )
            # Dataset profile hints take precedence over heuristic detection
            from core.dataset_profile import prompt_hints as _profile_hints
            _hints = _profile_hints()
            if _hints:
                for hint in _hints:
                    extra_notes.append(f"- {hint}")
            else:
                # Legacy heuristic fallback
                if has_wvs_like:
                    extra_notes.append(
                        "- WVS-style survey variables are often ordinal/categorical codes. Prefer comparisons across groups, ordered relationships, or treat as categorical; consider adding country/year controls if available."
                    )
                if has_amazon_like:
                    extra_notes.append(
                        "- Amazon-style review datasets may include text + IDs. Use simple string features and group-by aggregations when those columns exist."
                    )

            extra_notes_text = "\n".join(extra_notes) if extra_notes else "- Use only the columns listed below; do not invent columns."

            data_constraints = f"""
DATA CONSTRAINTS (You MUST follow these):

{context.data_schema}

Additional notes:
{extra_notes_text}
"""
        else:
            # Fallback if no schema available
            data_constraints = """
DATA CONSTRAINTS:
You are analyzing a dataset. Focus on measurable, testable hypotheses using available numeric and categorical columns.
Avoid hypotheses that require text analysis or user history data unless explicitly available.
"""
        
        # Base system prompt with data constraints
        system_prompt = f"""You are a scientific hypothesis generator for data analysis.
{data_constraints}

Your task is to generate clear, testable hypotheses relevant to the given discovery goal.

CRITICAL REQUIREMENTS FOR EACH HYPOTHESIS:
1. MUST use only columns from the DATA CONSTRAINTS section above
2. MUST be testable with statistical methods (correlation, t-test, regression, ANOVA)
3. MUST mention specific column names from the available dataset
4. If the schema includes text-like columns, you MAY use them via simple pandas string operations
   (e.g., .str.len(), .str.contains(), basic regex). Do NOT assume external NLP models/libraries.
5. MUST be a single, clear sentence
6. MUST mention specific conditions, thresholds, or comparisons when applicable

VALIDATION CHECKLIST (for every hypothesis you output):
- [ ] Uses only columns listed in the schema.
- [ ] Is testable with pandas/numpy/scipy/statsmodels.
- [ ] If it uses text, the text part can be implemented via simple string operations (str.len, str.contains, etc.).
- [ ] If it uses IDs, they are used as grouping keys or categorical labels, not numeric magnitudes.
- [ ] Is specific enough that a coder can write the test without guessing.

Format your response as a numbered list, one hypothesis per line."""
        
        # Build user prompt with mode-specific instructions
        prompt_parts = [
            f"Goal: {context.goal}",
            f"Focus Area: {context.focus_area}",
            f"Mode: {context.mode}",
            ""
        ]
        
        # === VARIABLE FORCING (Diversity helper) ===
        # Prefer Text-like + Numeric/Categorical pairs when text-like columns exist.
        if context.data_schema:
            available_cols = list(schema_cols)
            id_cols = list(id_like)
            text_cols = list(text_like)
            numeric_or_cat_cols = [c for c in available_cols if c not in id_cols]
            
            forced_pair = None
            # Prefer a Text + Numeric/Categorical pair if available
            if text_cols and numeric_or_cat_cols:
                if random.random() < 0.5:
                    forced_pair = [random.choice(text_cols), random.choice(numeric_or_cat_cols)]
            
            # Fallback: two non-ID numeric/categorical columns
            if forced_pair is None and len(numeric_or_cat_cols) >= 2:
                forced_pair = random.sample(numeric_or_cat_cols, 2)
            
            # As a last resort, sample from all columns except ID-like.
            if forced_pair is None and len(available_cols) >= 2:
                non_id_cols = [c for c in available_cols if c not in id_cols]
                if len(non_id_cols) >= 2:
                    forced_pair = random.sample(non_id_cols, 2)
            
            if forced_pair is not None:
                prompt_parts.append("OPTIONAL FOCUS FOR THIS BATCH:")
                prompt_parts.append(
                    f"In this batch, TRY to include 2–3 hypotheses that explicitly investigate a relationship involving "
                    f"'{forced_pair[0]}' and '{forced_pair[1]}'. You may include additional columns or interaction effects."
                )
                prompt_parts.append(
                    "The remaining hypotheses should intentionally explore DIFFERENT combinations of columns to maximize structural diversity."
                )
                prompt_parts.append("")

            # Diversity requirement: each hypothesis should use a different outcome or predictor pair
            prompt_parts.append(
                "DIVERSITY REQUIREMENT: Each hypothesis MUST use a different primary predictor-outcome pair. "
                "Do NOT generate multiple hypotheses about the same two columns. Spread across ALL available "
                "columns (product metadata, review features, text properties, etc.)."
            )
            prompt_parts.append("")
        
        # Rejected ideas: first the cross-round rejection summary (high-level guidance)
        if rejection_summary:
            prompt_parts.append("CROSS-ROUND REJECTION HISTORY (Summary):")
            prompt_parts.append("The following summary describes patterns in rejected hypotheses from previous rounds:")
            prompt_parts.append("")
            prompt_parts.append(rejection_summary)
            prompt_parts.append("")
            prompt_parts.append("Use this summary to avoid over-explored areas and focus on NEW directions.")
            prompt_parts.append("")

        # Then, add cross-round accepted/stored insights summary (high-level guidance)
        if insight_graph_summary:
            prompt_parts.append("INSIGHT GRAPH (ACCEPTED / STORED INSIGHTS) SUMMARY:")
            prompt_parts.append(
                "The following summary describes what is already in the discovery graph (accepted/stored insights)."
            )
            prompt_parts.append("")
            prompt_parts.append(insight_graph_summary)
            prompt_parts.append("")
            if context.mode == "global_exploration":
                prompt_parts.append(
                    "GLOBAL EXPLORATION DIRECTIVE: Treat the summary as explored territory. "
                    "Actively target hypotheses that are NOT implied by THEMES_ALREADY_COVERED / OVER-EXPLORED, "
                    "and preferentially expand the NEW_DIRECTIONS section (new variable families, new subgroup cuts, or new interactions). "
                    "Do NOT base your hypotheses on any single anchor insight."
                )
            else:
                prompt_parts.append(
                    "INSTRUCTION: Avoid generating hypotheses that are paraphrases of already-stored insights. "
                    "Use the summary to steer into genuinely new variables, conditions, groups, or interactions."
                )
            prompt_parts.append("")
        
        # Then, add current round rejections (specific to avoid)
        if rejected_sentences:
            prompt_parts.append("CURRENT ROUND FAILED ATTEMPTS (DO NOT REPEAT):")
            prompt_parts.append("You just proposed the following hypotheses in this round, and they were REJECTED because they are duplicates or already known:")
            for i, sent in enumerate(rejected_sentences, 1):
                prompt_parts.append(f"  {i}. {sent}")
            prompt_parts.append("")
            prompt_parts.append("INSTRUCTION: You must generate COMPLETELY DIFFERENT hypotheses. Change the variables, change the conditions, or look for interaction effects.")
            prompt_parts.append("")
        
        # Mode-specific instructions
        mode_instructions = self._get_mode_instructions(context.mode)
        prompt_parts.append(mode_instructions)
        prompt_parts.append("")
        
        # Add relevant insights with metadata, grouped by knowledge type
        if context.retrieved_insights:
            # Limit the number of insights for the prompt
            insights_for_prompt = context.retrieved_insights[:self.max_context_insights]
            
            # Group insights by knowledge_type
            descriptive = []
            patterns = []
            others = []
            
            for ins in insights_for_prompt:
                meta = ins.metadata or {}
                kt = meta.get("knowledge_type", "")
                if kt == "descriptive_stat":
                    descriptive.append(ins)
                elif kt in {"phenomenon", "empirical_pattern",
                             "cross_module_profile", "within_module_profile"}:
                    patterns.append(ins)
                else:
                    others.append(ins)
            
            # 1) Data facts: context only, not to be repeated as hypotheses
            if descriptive:
                prompt_parts.append("DATA FACTS (context only):")
                prompt_parts.append(
                    "These summarize basic dataset statistics. You may USE them as known facts, "
                    "but you should NOT propose them again as standalone hypotheses."
                )
                prompt_parts.append("")
                prompt_parts.append(self._format_insights_for_prompt(descriptive, context.mode))
                prompt_parts.append("")
            
            # 2) Observed phenomena: should be explained, refined, or challenged
            if patterns:
                prompt_parts.append("OBSERVED PHENOMENA (explain / refine / challenge):")
                prompt_parts.append(
                    "These are empirical patterns we observed in the data. Your job is to propose NEW hypotheses "
                    "that explain, refine, or challenge them, for example:"
                )
                prompt_parts.append(
                    "- adding boundary conditions (when does the pattern hold / break?),"
                )
                prompt_parts.append("- testing interactions (does a third variable Z change the effect?),")
                prompt_parts.append(
                    "- proposing possible mechanisms that can be examined with the available columns."
                )
                prompt_parts.append("")
                prompt_parts.append(self._format_insights_for_prompt(patterns, context.mode))
                prompt_parts.append("")
            
            # 3) Other existing insights (especially from previous rounds): these should NOT be restated
            if others:
                prompt_parts.append("EXISTING HYPOTHESES / INSIGHTS (do NOT restate):")
                prompt_parts.append(
                    "The following ideas already exist in the discovery graph. "
                    "Do NOT generate hypotheses that are paraphrases of these; instead, propose genuinely new angles."
                )
                prompt_parts.append("")
                prompt_parts.append(self._format_insights_for_prompt(others, context.mode))
                prompt_parts.append("")
        
        # Generation instructions - emphasize diversity AND data constraints
        # Extract column names for examples
        example_cols = ""
        if context.data_schema:
            column_lines = [line for line in context.data_schema.split('\n') if line.strip().startswith('-')]
            col_list = [line.strip().lstrip('-').split('(')[0].strip() for line in column_lines if line.strip().startswith('-')]
            if col_list:
                example_cols = f" (e.g., using columns like: {', '.join(col_list[:5])})"
        
        prompt_parts.extend([
            f"Generate EXACTLY {self.batch_size} new testable hypotheses relevant to the goal: {context.goal}",
            "",
            "## DIVERSITY CONSTRAINTS (IMPORTANT BUT FLEXIBLE):",
            f"Your batch of hypotheses must be diverse. Do NOT generate {self.batch_size} variations of the same idea.",
            "",
            "1. MOST hypotheses (at least half of them) should be SIMPLE main effects or pairwise comparisons.",
            "   - Example: 'X is higher than Y', 'X is positively associated with Y'.",
            "   - You may add AT MOST ONE control variable (e.g., 'even after controlling for Z').",
            "",
            "2. A MINORITY of hypotheses (up to 3) may look for interaction effects",
            "   - Example: 'The effect of X on Y is different for group A vs group B'.",
            "   - Avoid 3-way or very complex interactions.",
            "",
            "3. Try to cover DIFFERENT TYPES of structure across the batch:",
            *(
                [f"   - Some hypotheses about TEXT properties (e.g., {', '.join(text_like[:3])})."]
                if text_like
                else []
            ),
            *(
                [f"   - Some about TIME (e.g., {', '.join(time_like[:3])})."]
                if time_like
                else []
            ),
            *(
                [f"   - Some about GROUP differences or aggregation by keys (e.g., {', '.join(id_like[:3])})."]
                if id_like
                else []
            ),
            *(
                ["   - Some about subgroup comparisons across categorical/ordinal variables (e.g., comparing means across categories)."]
                if (categorical_like and not id_like)
                else []
            ),
            *(
                ["   - Some about interactions (X affects Y differently when Z is high/low)."]
                if not (text_like or time_like or id_like or categorical_like)
                else []
            ),
            "",
            "4. Only a few hypotheses (up to 2) should claim a negative/null result",
            "   (e.g., 'X does NOT affect Y'). Use these sparingly, and only when this would be somewhat surprising.",
            "",
            "In other words: keep most hypotheses simple and robust, and use interactions or non-linear patterns only for a small subset.",
            "",
            "## DATA VALIDATION (MANDATORY - READ BEFORE GENERATING):",
            "BEFORE writing each hypothesis, verify:",
            "1. Does it use ONLY columns from the DATA CONSTRAINTS section above?",
            "2. Can it be tested with statistical methods (correlation, t-test, regression) using pandas/numpy/scipy/statsmodels?",
            "3. If it uses text, can the text part be implemented via simple string operations (str.len, str.contains, etc.)?",
            "4. If it uses IDs, are they used as grouping keys, not numeric values?",
            "",
            "Each hypothesis must:",
            "- Be a clear, simple sentence that can be tested with data",
            "- Mention specific columns from the available dataset",
            "- Be directly related to the discovery goal and focus area",
            "- Use ONLY the columns listed in DATA CONSTRAINTS (no exceptions!)",
            "",
            "Format each hypothesis as a numbered list, one hypothesis per line."
        ])
        
        # Encourage interaction effects
        prompt_parts.append("")
        prompt_parts.append("PRO TIP: To avoid duplicates, look for INTERACTION EFFECTS using ONLY available columns.")
        prompt_parts.append("Instead of 'X affects Y', try 'X affects Y differently when Z is high'.")
        if has_amazon_like and all(c in schema_cols for c in ["rating", "helpful_votes", "verified_purchase"]):
            prompt_parts.append(
                "Example: 'rating affects helpful_votes' is obvious. Try: "
                "'rating affects helpful_votes more strongly for verified_purchase=1 than verified_purchase=0'."
            )
        prompt_parts.append("")
        prompt_parts.append("FINAL REMINDER: Every hypothesis MUST use only the columns from DATA CONSTRAINTS. If you're unsure about a column, check the schema above.")
        
        return "\n".join(prompt_parts), system_prompt
    
    def _get_mode_instructions(self, mode: str) -> str:
        """Get mode-specific generation instructions."""
        instructions = {
            "exploration": """Your task is EXPLORATION: Generate hypotheses that explore new areas or under-explored aspects.
- Focus on novel angles not yet covered by existing insights
- Consider edge cases, boundary conditions, or alternative explanations
- Think about what might happen in different scenarios or contexts
- Aim for diversity in your hypotheses to cover unexplored territory""",

            "global_exploration": """Your task is GLOBAL EXPLORATION: Generate hypotheses that deliberately explore areas NOT covered in the insight-graph summary.
- Do NOT anchor on any specific prior insight; there may be no anchor context
- Treat the insight-graph summary as a map of what is already explored; aim for gaps and orthogonal directions
- Prefer new variable families, new subgroup cuts, new interactions, or new outcome variables not emphasized so far
- Still obey the dataset schema constraints strictly (use only listed columns)""",
            
            "refinement": """Your task is REFINEMENT: Generate hypotheses that refine or extend existing strong insights.
- Build upon the high-quality insights provided
- Make them more specific, precise, or testable
- Consider variations, extensions, or deeper investigations
- Focus on improving clarity, specificity, or adding conditions/contexts""",
            
            "conflict_resolution": """Your task is CONFLICT RESOLUTION: Generate hypotheses that can help resolve contradictory insights.
- Consider hypotheses that could explain or reconcile conflicting findings
- Think about conditions, contexts, or moderating factors that might explain both sides
- Generate hypotheses that could test which insight is correct under what conditions
- Consider meta-hypotheses about when each conflicting insight applies"""
        }
        
        return instructions.get(mode, instructions["exploration"])
    
    def _format_insights_for_prompt(
        self,
        insights: List[Insight],
        mode: str
    ) -> str:
        """Format insights with metadata for prompt."""
        formatted = []
        
        for i, insight in enumerate(insights, 1):
            parts = [f"{i}. {insight.sentence}"]
            
            # Add quality metadata if available
            metadata = insight.metadata
            if metadata:
                scores = []
                if "overall_score" in metadata:
                    scores.append(f"score: {metadata['overall_score']:.2f}")
                if "validity_score" in metadata:
                    scores.append(f"validity: {metadata['validity_score']:.2f}")
                if "novelty_score" in metadata:
                    scores.append(f"novelty: {metadata['novelty_score']:.2f}")
                
                if scores:
                    parts.append(f"   ({', '.join(scores)})")
            
            formatted.append(" ".join(parts))
        
        return "\n".join(formatted)
    
    def _generate_with_llm(
        self,
        prompt: str,
        system_prompt: str,
        context: RoundContext
    ) -> List[Hypothesis]:
        """Generate hypotheses using LLM with mode-aware processing.
        
        Raises:
            RuntimeError: If LLM is not available or generation fails
            ValueError: If insufficient hypotheses are generated or parsing fails
        """
        # Check if LLM is available
        if not self.llm.is_available():
            raise RuntimeError(
                f"LLM is not available. Cannot generate hypotheses. "
                f"Please ensure the LLM service is running and properly configured."
            )
        
        # Adjust temperature based on mode
        temperature = self._get_mode_temperature(context.mode)
        
        # Generate with LLM
        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=temperature,
                system=system_prompt
            )
            # stash raw response for artifact logging
            self._last_llm_raw_response = response
        except Exception as e:
            raise RuntimeError(
                f"LLM generation failed: {e}. "
                f"Please check LLM service connection and configuration."
            ) from e
        
        # Parse the response to extract hypotheses
        hypothesis_sentences = parse_list_response(response, num_items=self.batch_size)
        
        # If parsing failed or returned too few, try to extract sentences
        if len(hypothesis_sentences) < self.batch_size:
            # Try to extract sentences ending with periods
            sentences = re.split(r'[.!?]+', response)
            hypothesis_sentences = [
                s.strip() for s in sentences 
                if s.strip() and len(s.strip()) > 20 and len(s.strip()) < 200
            ][:self.batch_size]
        
        # Validate we have enough hypotheses
        if len(hypothesis_sentences) < self.batch_size:
            raise ValueError(
                f"Insufficient hypotheses generated. Expected {self.batch_size}, "
                f"but only parsed {len(hypothesis_sentences)} valid hypotheses from LLM response. "
                f"LLM response: {response[:200]}..."
            )
        
        # Create Hypothesis objects
        hypotheses = []
        for sentence in hypothesis_sentences[:self.batch_size]:
            if not sentence or len(sentence) <= 10:
                raise ValueError(
                    f"Invalid hypothesis sentence generated: '{sentence}'. "
                    f"Hypotheses must be at least 10 characters long."
                )
            
            hypotheses.append(Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                sentence=sentence,
                context_insight_ids=[],  # Will be filled by _link_to_context_insights
                metadata={
                    "generation_method": "llm",
                    "mode": context.mode,
                    "focus_area": context.focus_area,
                    "model": self.llm.model if hasattr(self.llm, 'model') else "unknown",
                    "temperature": temperature
                }
            ))
        
        # Final validation
        if len(hypotheses) < self.batch_size:
            raise ValueError(
                f"Failed to create sufficient Hypothesis objects. "
                f"Expected {self.batch_size}, got {len(hypotheses)}."
            )
        
        return hypotheses[:self.batch_size]
    
    def _get_mode_temperature(self, mode: str) -> float:
        """Get temperature setting based on mode."""
        # Exploration: higher temperature for more diversity
        # Refinement: lower temperature for more focused generation
        # Conflict resolution: medium temperature
        mode_temperatures = {
            "exploration": min(0.85, self.temperature + 0.1),
            "global_exploration": min(0.95, self.temperature + 0.2),
            "refinement": max(0.5, self.temperature - 0.1),
            "conflict_resolution": self.temperature
        }
        return mode_temperatures.get(mode, self.temperature)
    
    def _link_to_context_insights(
        self,
        hypotheses: List[Hypothesis],
        context: RoundContext
    ) -> List[Hypothesis]:
        """Link hypotheses to context insights that influenced them."""
        if not context.retrieved_insights:
            return hypotheses
        
        # Extract insight IDs from anchor insights (most relevant)
        anchor_ids = set(context.anchor_insight_ids)
        
        # For each hypothesis, find related insights
        for hypothesis in hypotheses:
            related_ids = []
            
            # Always include anchor insights as they're the focus
            related_ids.extend(context.anchor_insight_ids)
            
            # For refinement mode, prioritize high-quality insights
            if context.mode == "refinement":
                # Add top insights by quality score
                scored_insights = [
                    (insight, insight.metadata.get("overall_score", 0.5))
                    for insight in context.retrieved_insights
                    if insight.id not in anchor_ids
                ]
                scored_insights.sort(key=lambda x: x[1], reverse=True)
                related_ids.extend([insight.id for insight, _ in scored_insights[:2]])
            
            # For conflict resolution, include all retrieved insights
            elif context.mode == "conflict_resolution":
                related_ids.extend([insight.id for insight in context.retrieved_insights[:5]])
            
            # For exploration, include diverse insights
            else:
                # Include a mix of insights
                related_ids.extend([insight.id for insight in context.retrieved_insights[:3]])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_ids = []
            for insight_id in related_ids:
                if insight_id not in seen:
                    seen.add(insight_id)
                    unique_ids.append(insight_id)
            
            hypothesis.context_insight_ids = unique_ids[:5]  # Limit to top 5
        
        return hypotheses

    # -------------------------------------------------------------------------
    # Two-phase flow: Phase 1 = general idea, Phase 2 = refiner (idea + RAG → variable-level hypotheses)
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_valid_phase1_json(parsed: Dict[str, Any], concept_menu: List[str]) -> bool:
        """Validate phase-1 JSON structure (primary theme required, secondary optional)."""
        required_str_keys = [
            "research_question",
            "core_constructs",
            "hypothesized_relations",
        ]
        for k in required_str_keys:
            if not str(parsed.get(k, "") or "").strip():
                return False
        # primary_theme_id must be valid int in range
        try:
            n = int(parsed["primary_theme_id"])
            if not (1 <= n <= len(concept_menu)):
                return False
        except (ValueError, KeyError, TypeError):
            return False
        # secondary_theme_id: optional, null allowed
        sec = parsed.get("secondary_theme_id")
        if sec is not None:
            try:
                m = int(sec)
                if not (1 <= m <= len(concept_menu)):
                    return False
            except (ValueError, TypeError):
                return False
        # anchors: should be a list (may be empty)
        anchors = parsed.get("anchors")
        if anchors is not None and not isinstance(anchors, list):
            return False
        return True

    def generate_general_idea(
        self,
        context: RoundContext,
        concept_menu: List[str],
        grounding_feedback: Optional[str] = None,
        avoid_themes: Optional[List[str]] = None,
        required_theme: Optional[str] = None,
        insight_graph_summary: Optional[str] = None,
        refinement_leads: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Phase 1: Produce a general research idea (construct-level only, no variable names).
        Used when variable_catalog is enabled (e.g. WVS). Output is used to query RAG and then refiner.

        Args:
            avoid_themes: Theme strings that have been over-used in recent rounds.
                          Injected as a negative constraint so the LLM diversifies.
            required_theme: If set, the LLM MUST use this as PRIMARY_THEME.
                           Overrides avoid_themes.
            insight_graph_summary: Summary of what's already been discovered, so the
                                   LLM can steer toward genuinely new directions.
            refinement_leads: Thematic summary of borderline findings worth revisiting.
                              Helps steer theme selection toward areas with promising
                              near-miss results that could be deepened.
        """
        # Numbered menu so model can output PRIMARY/SECONDARY ids for stable range resolution.
        menu_text = "\n".join(f"{i}) {m}" for i, m in enumerate(concept_menu, 1))

        avoid_block = ""
        if required_theme:
            # Find the menu index for the required theme
            req_idx = None
            for i, m in enumerate(concept_menu, 1):
                if m == required_theme or required_theme in m:
                    req_idx = i
                    break
            if req_idx is not None:
                avoid_block = (
                    f"\nREQUIRED PRIMARY THEME — You MUST use theme {req_idx} "
                    f"(\"{required_theme}\") as your PRIMARY_THEME. "
                    f"Pick a complementary secondary_theme from a different module.\n"
                )
            # Skip avoid_themes when a theme is explicitly required
        elif avoid_themes:
            avoid_list = "\n".join(f"  - {t}" for t in avoid_themes)
            avoid_block = (
                f"\nTHEME COOLDOWN — the following themes have been heavily used in recent rounds. "
                f"You MUST pick at least one theme NOT on this list:\n{avoid_list}\n"
            )

        explored_block = ""
        if insight_graph_summary:
            explored_block = (
                f"\nALREADY EXPLORED (avoid these themes/directions):\n"
                f"{insight_graph_summary}\n"
                f"Use this to steer toward genuinely NEW research questions, "
                f"not re-examine known findings.\n"
            )

        refinement_block = ""
        if refinement_leads:
            refinement_block = (
                f"\n{refinement_leads}\n"
                f"Consider picking themes that let you deepen these promising near-miss findings "
                f"(e.g., add moderators, test interactions, or examine subgroup heterogeneity).\n"
            )

        grounding_block = ""
        if grounding_feedback:
            grounding_block = (
                f"\nGROUNDING FEEDBACK FROM PREVIOUS ATTEMPT:\n{grounding_feedback}\n"
                "Revise your output so constructs/anchors are mappable to available variables.\n"
            )

        from core.dataset_profile import phase1_persona, phase1_wording_hint

        # The dataset-specific persona is sent as the system message; the user
        # prompt is the Phase-1 template.
        persona = phase1_persona() or "You are a scientist proposing a research idea for a dataset."
        wording = phase1_wording_hint()
        wording_rule = f"\n- {wording}" if wording else ""

        prompt = f"""You are a research scientist proposing potentially novel and scientifically meaningful research ideas for a large-scale empirical dataset. Your goal is to identify high-level research directions that connect conceptually distinct themes and could lead to surprising, testable findings.

Goal: {context.goal}
Focus area: {context.focus_area}
Mode: {context.mode}

CONCEPT MENU (pick up to TWO themes by number):
{menu_text}
{avoid_block}{explored_block}{refinement_block}{grounding_block}
Rules:
- Do NOT mention any variable codes or column names (no Qxx, no B_COUNTRY, etc.).
- Keep this stage high-level and thematic. Do NOT commit to a specific statistical template yet.{wording_rule}
- AIM FOR SURPRISE: prefer ideas that would contradict common assumptions, reveal unexpected connections between distant themes, or show that an effect reverses for certain subgroups.

Return ONLY a JSON object with these fields:
{{
  "primary_theme_id": <int 1..{len(concept_menu)}>,
  "secondary_theme_id": <int 1..{len(concept_menu)} or null>,
  "research_question": "<one sentence>",
  "core_constructs": "<2-4 short noun phrases, comma-separated>",
  "hypothesized_relations": "<1-2 short clauses in natural language>",
  "confounders": "<3-6 items; include demographics if plausible>",
  "anchors": ["<6-12 short codebook-like phrases for retrieval>"],
  "surprise_angle": "<one sentence: what would be counterintuitive>"
}}"""

        if not self.llm.is_available():
            raise RuntimeError("LLM not available for general idea generation.")
        _persona_short = persona.split(".")[0].rstrip() if persona else "You are a scientist"
        system_msg = f"{_persona_short}. Output only valid JSON."
        response = ""
        # Temperature 0.6 for exploration and global exploration, 0.45 otherwise.
        phase1_temperature = 0.6 if str(context.mode or "").lower() in {"exploration", "global_exploration"} else 0.45
        # One attempt plus up to 3 retries on JSON parsing or format failure.
        max_attempts = 4
        first_prompt = prompt
        parsed_json: Dict[str, Any] = {}
        for attempt in range(max_attempts):
            raw = self.llm.generate(
                prompt,
                temperature=phase1_temperature,
                system=system_msg,
            )
            response = (raw or "").strip()
            parsed_json = extract_json_or(response, fallback={})
            if self._is_valid_phase1_json(parsed_json, concept_menu):
                break
            if attempt < max_attempts - 1:
                # Clean retry: rebuild prompt with short error note (don't append)
                prompt = (
                    f"{first_prompt}\n\n"
                    f"Your previous output was not valid JSON or had missing/invalid fields. "
                    f"Output ONLY the JSON object, nothing else."
                )
        self.last_general_idea_call = {
            "prompt": first_prompt,
            "system_prompt": system_msg,
            "response": response,
        }

        # Build idea dict with downstream-compatible keys
        idea: Dict[str, Any] = {
            "primary_theme_id": None,
            "secondary_theme_id": None,
            "primary_theme": "",
            "secondary_theme": "",
            "research_question": "",
            "core_constructs": "",
            "hypothesized_relations": "",
            "candidate_confounders": "",
            "anchors": [],
            "data_requirements": "",
            "raw": response,
        }

        # Extract primary_theme_id
        pid_raw = parsed_json.get("primary_theme_id")
        try:
            pid = int(pid_raw)
            if 1 <= pid <= len(concept_menu):
                idea["primary_theme_id"] = pid
                idea["primary_theme"] = concept_menu[pid - 1]
            else:
                idea["primary_theme_id"] = pid_raw
        except (ValueError, TypeError):
            idea["primary_theme_id"] = pid_raw

        # Extract secondary_theme_id
        sid_raw = parsed_json.get("secondary_theme_id")
        if sid_raw is None:
            idea["secondary_theme_id"] = None
            idea["secondary_theme"] = ""
        else:
            try:
                sid = int(sid_raw)
                if 1 <= sid <= len(concept_menu):
                    idea["secondary_theme_id"] = sid
                    idea["secondary_theme"] = concept_menu[sid - 1]
                else:
                    idea["secondary_theme_id"] = sid_raw
            except (ValueError, TypeError):
                idea["secondary_theme_id"] = None

        # Fallback: ensure primary_theme is set
        if not (idea.get("primary_theme") or "").strip():
            idea["primary_theme"] = concept_menu[0]

        # Copy string fields
        idea["research_question"] = str(parsed_json.get("research_question") or "").strip()
        idea["core_constructs"] = str(parsed_json.get("core_constructs") or "").strip()
        idea["hypothesized_relations"] = str(parsed_json.get("hypothesized_relations") or "").strip()
        idea["candidate_confounders"] = str(parsed_json.get("confounders") or parsed_json.get("candidate_confounders") or "").strip()

        # Anchors: proper list
        anchors_raw = parsed_json.get("anchors", [])
        if isinstance(anchors_raw, list):
            idea["anchors"] = [str(a).strip() for a in anchors_raw if str(a).strip()]
        elif isinstance(anchors_raw, str):
            idea["anchors"] = [a.strip() for a in anchors_raw.split(",") if a.strip()]
        else:
            idea["anchors"] = []

        # Surprise angle
        idea["surprise_angle"] = str(parsed_json.get("surprise_angle") or "").strip()

        return idea

    def refine_idea_to_hypotheses(
        self,
        general_idea: Dict[str, Any],
        schema_with_glossary: str,
        context: RoundContext,
        batch_size: int,
        avoid_variables: Optional[List[str]] = None,
        rejected_sentences: Optional[List[str]] = None,
        rejection_summary: Optional[str] = None,
        insight_graph_summary: Optional[str] = None,
        critic_hints: Optional[str] = None,
    ) -> List[Hypothesis]:
        """
        Phase 2: Turn general idea + variable schema/glossary into concrete, testable hypothesis sentences
        that use ONLY the column names from the schema. Used when variable_catalog is enabled.

        Args:
            avoid_variables: Over-used variable codes to deprioritize. Injected into the
                             prompt so the LLM explores less common columns.
            rejected_sentences: Hypothesis sentences rejected this round (do not repeat).
            rejection_summary: Summary of cross-round rejection history.
            insight_graph_summary: Summary of already-discovered findings.
        """
        surprise_angle = general_idea.get("surprise_angle", "")
        idea_block = (
            f"Research question: {general_idea.get('research_question', '')}\n"
            f"Core constructs: {general_idea.get('core_constructs', '')}\n"
            f"Hypothesized relations: {general_idea.get('hypothesized_relations', '')}\n"
            f"Candidate confounders: {general_idea.get('candidate_confounders', '')}"
            + (f"\nSURPRISE ANGLE: {surprise_angle}" if surprise_angle else "")
        )

        # Build history blocks — summary is already compact (deterministic, no LLM)
        history_block = ""
        if insight_graph_summary:
            history_block += f"ALREADY DISCOVERED:\n{insight_graph_summary}\n\n"
        if rejection_summary:
            history_block += f"CROSS-ROUND REJECTION PATTERNS:\n{rejection_summary}\n\n"
        if critic_hints:
            history_block += f"{critic_hints}\n\n"
        if rejected_sentences:
            # Cap at 5, truncate each line
            reject_list = "\n".join(
                f"  {i}. {s[:80]}" for i, s in enumerate(rejected_sentences[:5], 1)
            )
            history_block += f"DO NOT REPEAT:\n{reject_list}\n\n"

        from core.dataset_profile import (
            dsl_rules as _dsl_rules,
            weight_columns as _weight_columns,
        )

        # --- Profile-driven rules (fall back to legacy WVS defaults) --------
        rules = _dsl_rules()
        if not rules:
            # Legacy WVS default rules for backward compat
            rules = [
                "WVS rule: assume negative/special codes (e.g., -1/-2/-4/-5) are recoded to missing before analysis.",
                "Weight columns (e.g., sampling_weight, PWGHT, W_WEIGHT) may be used only as statistical weights, NOT as main predictors/outcomes.",
                "Country/ID columns (e.g., B_COUNTRY, survey_id) may be used for stratification/grouping (by=..., | ...) or fixed effects, NOT as numeric predictors/outcomes in assoc().",
                "survey_id is a technical identifier; do NOT use it as predictor, outcome, or control.",
                "SAME-BATTERY RULE: Do NOT test assoc() between variables from the same questionnaire section (adjacent Q-numbers like Q113 and Q114, or Q62 and Q63). Their correlation is trivially high by survey design. Your X and Y must come from DIFFERENT sections.",
                "COUNTRY-DIFF RULE: diff(Y, by=B_COUNTRY) MUST include at least one substantive (non-demographic) Q-variable as a control. Bare country differences are trivially significant in a 60-country survey.",
            ]
        rules_block = "\n".join(f"- {r}" for r in rules)

        # --- DSL weight examples --------------------------------------------
        _weights = _weight_columns()
        weight_example = f" [weights={_weights[0]}]" if _weights else ""

        prompt = f"""You are a research scientist translating a high-level research idea into concrete, statistically testable hypotheses over a structured dataset. Convert the GENERAL IDEA into hypotheses using ONLY the provided column names.

GENERAL IDEA:
{idea_block}

{history_block}AVAILABLE VARIABLES (schema + glossary). Use ONLY exact column names:
{schema_with_glossary}

Rules:
- Output exactly {batch_size} hypotheses as a numbered list. Nothing else.
- Each hypothesis must be a single DSL line starting with exactly one of: assoc(...), diff(...), interact(...), or heterogeneity(...).
- Simple associations and group differences are allowed and encouraged when appropriate.
- "controlling for ..." is OPTIONAL. If included, list 1-4 plausible controls (prefer demographics) using ONLY existing column names.
{rules_block}
- Do NOT append explanations after ":"; output only DSL lines.
- Do NOT put weight columns in "controlling for"; if a weight is needed, add weights=<weight_column> at the end.

Internal step (do NOT output):
Map each construct/anchor phrase to 1-2 columns from the glossary, then instantiate kernels.

PRIORITY ORDER (most important first):
1. CROSS-MODULE: X and Y MUST come from different thematic sections of the dataset. Connecting distant themes is far more valuable than testing within-section patterns.
2. NOVELTY: Prioritize hypotheses not already in the existing knowledge base. Simple associations are fine early on; if many are known, explore conditional effects (interact, heterogeneity).
3. SURPRISE: At least 2 hypotheses should test a relationship most domain experts would NOT expect.
4. DIVERSITY: Vary outcomes/predictors across the batch.
5. CORRECTNESS: All hypotheses must use valid column names and DSL syntax.

DSL forms (one per line; examples show optional controls/weights):
- assoc(X, Y) [optional: controlling for C1, C2]{weight_example}
- diff(Y, by=G) [optional: controlling for C1, C2]{weight_example}
- interact(X * Z -> Y) [optional: controlling for C1, C2]{weight_example}
- heterogeneity(Y ~ X | G) [optional: controlling for C1, C2]{weight_example}

Forbidden examples:
- assoc(X, Y | G)
- diff(Y by G)
- ... : explanatory sentence

Output only the numbered hypotheses."""

        if avoid_variables:
            var_list = ", ".join(avoid_variables)
            prompt += (
                f"\n\nVARIABLE COOLDOWN: These variables have been heavily used in previous rounds "
                f"and should be DEPRIORITIZED as main X/Y predictors/outcomes (they are still "
                f"allowed as controls): {var_list}\n"
                f"Try to use DIFFERENT variables from the schema as your main predictors/outcomes."
            )

        if not self.llm.is_available():
            raise RuntimeError("LLM not available for hypothesis refiner.")
        response = ""
        sentences: List[str] = []
        # One attempt plus up to 3 retries if the returned list size mismatches the batch size.
        phase2_attempts = 4
        first_prompt = prompt
        system_msg = "You output only a numbered list of hypothesis sentences using the given variable names."
        # Temperature 0.6 for exploration and 0.4 otherwise.
        phase2_temperature = 0.6 if str(context.mode or "").lower() == "exploration" else 0.4
        for attempt in range(phase2_attempts):
            response = self.llm.generate(
                prompt,
                temperature=phase2_temperature,
                system=system_msg,
            )
            response = (response or "").strip()
            sentences = parse_list_response(response, num_items=batch_size)
            if len(sentences) == batch_size:
                break
            if attempt < phase2_attempts - 1:
                # Clean retry: rebuild prompt with short error note (don't append)
                prompt = (
                    f"{first_prompt}\n\n"
                    f"Your previous output had {len(sentences)} hypotheses instead of {batch_size}. "
                    f"Output ONLY a numbered list of exactly {batch_size} DSL hypotheses."
                )
        self.last_refiner_call = {
            "prompt": first_prompt,
            "system_prompt": system_msg,
            "response": response,
            "parsed_sentences": list(sentences),
        }

        if len(sentences) < 1:
            # Fallback parser when model misses numbered formatting
            for m in re.finditer(r"(?:\d+[.)]\s*)?([^.?!]+[.?!])", response):
                s = m.group(1).strip()
                if 25 < len(s) < 300:
                    sentences.append(s)
            sentences = sentences[:batch_size]

        hypotheses: List[Hypothesis] = []
        for s in sentences:
            if not s or len(s) <= 10:
                continue
            hypotheses.append(
                Hypothesis(
                    id=f"hyp_{uuid.uuid4().hex[:8]}",
                    sentence=s,
                    context_insight_ids=context.anchor_insight_ids[:5],
                    metadata={
                        "generation_method": "refiner",
                        "mode": context.mode,
                        "focus_area": context.focus_area,
                        "general_idea": general_idea,
                    },
                )
            )
        return hypotheses[:batch_size]

    def _refine_single_hypothesis(self, hypothesis: Hypothesis, context: RoundContext) -> Hypothesis:
        """Use an adversarial prompt to make the hypothesis more novel/robust."""
        
        # Extract columns for the prompt
        cols = "available columns"
        col_list = []
        if context.data_schema:
            column_lines = [line for line in context.data_schema.split('\n') if line.strip().startswith('-')]
            col_list = [line.strip().lstrip('-').split('(')[0].strip() for line in column_lines if line.strip().startswith('-')]
            if col_list:
                cols = ", ".join(col_list)
        
        # Build data constraints reminder
        data_constraints_reminder = ""
        if col_list:
            data_constraints_reminder = f"""
CRITICAL DATA CONSTRAINTS:
- You MUST use only columns listed in the provided schema.
- If the dataset includes free-text columns, you MAY use them only via simple pandas string operations
  (length, keyword search, basic pattern checks). Do NOT assume external NLP.
- You MAY use identifier/key columns (if any) as grouping keys to define group-level patterns.
- Do NOT assume external NLP models; everything must be implementable with pandas/numpy/scipy.
- The hypothesis must be concrete enough that a coder can implement it as Python code without guessing.
"""
        
        skeptic_prompt = f"""ACT AS A HARSH SCIENTIFIC REVIEWER.

I am proposing this hypothesis for an experiment:

"{hypothesis.sentence}"
{data_constraints_reminder}
CRITIQUE:

1. Is this obvious? (e.g., "higher rating = more helpful votes" is trivial)

2. Is it a spurious correlation? (e.g., "ice cream causes shark attacks")

3. What is a "Third Variable" from the dataset that might explain this relationship?

TASK:

Rewrite this hypothesis to be scientifically more robust by:
- making it slightly more specific (e.g., specify a subgroup, a time period, or a clear boundary condition), OR
- adding AT MOST ONE clearly defined control variable (e.g., 'even after controlling for Z').

Avoid introducing overly complex interactions or three-way relationships.

You MUST use ONLY the available columns: {cols}

Examples of Refinement (generic):
- Bad: "X affects Y." (too vague)
- Good: "X is associated with Y even after controlling for Z." (adds a concrete control)
- Good: "The association between X and Y is stronger/weaker for group G vs H." (adds a clear subgroup)

VALIDATION: Before returning, verify the refined hypothesis uses ONLY columns from: {cols}

Return ONLY the rewritten hypothesis sentence. Do not add explanations.
"""
        try:
            # Low temp for precise rewriting
            if not self.llm.is_available():
                return hypothesis
            
            refined_sentence = self.llm.generate(skeptic_prompt, temperature=0.3).strip().strip('"').strip("'")
            
            # Clean up common LLM artifacts
            if refined_sentence.startswith("Hypothesis:"):
                refined_sentence = refined_sentence.replace("Hypothesis:", "").strip()
            if refined_sentence.startswith("Rewritten:"):
                refined_sentence = refined_sentence.replace("Rewritten:", "").strip()
            
            # If the LLM returns something wildly different or empty, keep original
            if len(refined_sentence) > 10 and refined_sentence != hypothesis.sentence:
                print(f"    [INFO] Refined: '{hypothesis.sentence[:40]}...' -> '{refined_sentence[:40]}...'")
                hypothesis.sentence = refined_sentence
                # Update metadata
                hypothesis.metadata["refined_by_skeptic"] = True
                
        except Exception as e:
            print(f"    [WARN] Skeptic failed: {e}")
            
        return hypothesis
    
    def _deduplicate_batch(self, hypotheses: List[Hypothesis], context: RoundContext) -> List[Hypothesis]:
        """Remove structural duplicates within a batch of hypotheses.

        Deduplicates on core DSL variables (before "controlling for") so that
        the same X,Y relationship with different controls is treated as one.
        """
        import re

        def _core_signature(sentence: str) -> str:
            """Normalize to core DSL (strip controls, sort assoc args)."""
            s = sentence.strip().lower()
            core = s.split(" controlling for")[0].strip()
            core = re.sub(r"\s*weights\s*=\s*\S+", "", core).strip()
            # Sort assoc arguments (commutative)
            m = re.match(r"assoc\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", core)
            if m:
                args = sorted([m.group(1).strip(), m.group(2).strip()])
                core = f"assoc({args[0]}, {args[1]})" + core[m.end():]
            return core

        seen_signatures = set()
        deduplicated = []

        for hypothesis in hypotheses:
            sig = _core_signature(hypothesis.sentence)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                deduplicated.append(hypothesis)

        return deduplicated
    
    def process(self, context: RoundContext) -> List[Hypothesis]:
        """Process: generate hypotheses from context."""
        return self.generate_hypotheses(context)

