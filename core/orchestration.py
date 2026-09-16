"""Main orchestration loop for AutoKD rounds."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime
from pathlib import Path
import uuid
import re
import math

from core.types import (
    RoundContext,
    RoundResult,
    Hypothesis,
    ExperimentResult,
    Evaluation,
    InsightScore,
    Insight,
    RelationType
)
from memory.insight_graph import InsightGraph
from memory.rejection_storage import RejectionStorage
from memory.refine_storage import RefineStorage
from agents.historian import Historian
from agents.orchestrator import Orchestrator
from agents.generator import Generator
from agents.coder_runner import CoderRunner
from agents.evaluator import Evaluator
from agents.critic import Critic
from core.config import config
from core.llm_client import create_llm_client
from core.artifacts import ArtifactWriter
from core.variable_catalog import (
    build_variable_glossary_detail,
    load_variable_cards_jsonl,
    VariableRetriever,
    parse_q_range_from_theme,
    parse_module_from_theme,
)


class DiscoveryEngine:
    """Main engine that orchestrates discovery rounds."""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        memory_path: Optional[str] = None
    ):
        # Initialize Discovery Memory
        memory_path = memory_path or config.get("memory.storage_path", "./data/memory/insight_graph.json")
        embedding_model = config.get("memory.embedding_model", "all-MiniLM-L6-v2")
        self.insight_graph = InsightGraph(storage_path=memory_path, embedding_model=embedding_model)
        
        # Initialize Rejection Storage (cross-round rejection history)
        rejection_path = str(Path(memory_path).parent / "rejections.json")
        self.rejection_storage = RejectionStorage(storage_path=rejection_path)
        
        # Initialize Refine Storage (cross-round refinement candidates)
        refine_path = str(Path(memory_path).parent / "refinements.json")
        self.refine_storage = RefineStorage(storage_path=refine_path)
        
        # Initialize agents
        self.historian = Historian(self.insight_graph)
        self.orchestrator = Orchestrator(self.historian)
        self.generator = Generator()
        self.coder_runner = CoderRunner(data_path=data_path)
        self.evaluator = Evaluator()
        self.critic = Critic(self.historian)
        
        # Optional: variable catalog retriever for wide datasets (e.g., WVS).
        self.variable_retriever = None
        self._variable_catalog_cfg = config.get("variable_catalog", {}) or {}
        self._use_generalized_glossary = bool(self._variable_catalog_cfg.get("use_generalized_glossary", True))
        self._include_raw_question_fallback = bool(self._variable_catalog_cfg.get("include_raw_question_fallback", True))
        self._include_raw_options_in_glossary = bool(self._variable_catalog_cfg.get("include_raw_options_in_glossary", False))
        self._include_raw_options_in_index_text = bool(self._variable_catalog_cfg.get("include_raw_options_in_index_text", False))
        try:
            if bool(self._variable_catalog_cfg.get("enabled", False)):
                catalog_path = str(self._variable_catalog_cfg.get("catalog_path", "") or "").strip()
                if catalog_path:
                    cards = load_variable_cards_jsonl(catalog_path)
                    self.variable_retriever = VariableRetriever(
                        cards,
                        include_raw_question_fallback=self._include_raw_question_fallback,
                        include_raw_options_in_index_text=self._include_raw_options_in_index_text,
                    )
                    print(f"[INFO] Variable catalog enabled: {catalog_path} ({len(cards)} cards)")
        except Exception as e:
            # Never break discovery due to optional RAG tooling.
            print(f"[WARN] Variable catalog init failed: {type(e).__name__}: {e}")
            self.variable_retriever = None

        # Initialize LLM for rejection summarization
        llm_config = config.get_llm_config()
        self.llm = create_llm_client(llm_config)
        
        self.goal = config.get_goal()
        self.round_history: List[RoundResult] = []

        # Theme cooldown: track recent primary themes to force diversity.
        # When a theme has been used >= cooldown_threshold times in the last
        # cooldown_window rounds, it is injected as "avoid" in Phase 1.
        self._theme_history: List[str] = []  # all themes (primary + secondary)
        self._theme_round_count: int = 0     # actual number of two-phase rounds
        self._theme_cooldown_window = int(self._variable_catalog_cfg.get("theme_cooldown_window", 6))
        self._theme_cooldown_threshold = int(self._variable_catalog_cfg.get("theme_cooldown_threshold", 3))

        # Variable cooldown: track variables in a sliding window of recent
        # accepted insights.  Variables appearing in > max_pct of the window
        # are flagged as over-used.  Using a window prevents the cumulative
        # count from diluting the signal as the graph grows.
        self._variable_history: List[List[str]] = []  # per-insight column lists
        self._variable_cooldown_window = int(self._variable_catalog_cfg.get("variable_cooldown_window", 30))
        self._variable_cooldown_max_pct = float(self._variable_catalog_cfg.get("variable_cooldown_max_pct", 0.25))

        # Variable coverage: track column usage across ALL tested hypotheses (not just accepted).
        # Used to inject under/over-explored guidance into the generator prompt.
        from collections import Counter
        self._column_usage_counter: Counter = Counter()

        # Control-only columns: may appear in "controlling for" but never as
        # main predictor/outcome (X, Y, G, Z).  Fieldwork dates and survey
        # year are classic examples of confounders that should not be treated
        # as substantive variables.
        from core.dataset_profile import control_only_columns as _profile_control_only
        _profile_co = _profile_control_only()
        _config_co = (
            self._variable_catalog_cfg.get("control_only_columns", []) or
            config.get("data.control_only_columns", []) or []
        )
        # Profile takes precedence; fall back to config; legacy default = WVS fieldwork cols
        _co_source = _profile_co if _profile_co else list(_config_co)
        if not _co_source:
            _co_source = ["FW_START", "FW_END", "A_YEAR"]
        self._control_only_cols = set(str(c).lower() for c in _co_source)

        # Toggle for semantic PRE-EXISTING filter during hypothesis generation.
        # Enabled by default, threshold is configurable and can be softer than graph duplicate threshold.
        self.enable_preexisting_filter = bool(config.get("memory.enable_preexisting_filter", True))
        # Ablation A0: no memory across rounds. See config.yaml: ablation.
        self.no_persistence = bool(config.get("ablation.no_persistence", False))
        _pst = config.get("memory.preexisting_similarity_threshold", 0.88)
        try:
            self.preexisting_similarity_threshold = float(_pst)
        except (TypeError, ValueError):
            self.preexisting_similarity_threshold = 0.88
        self.preexisting_similarity_threshold = max(0.0, min(1.0, self.preexisting_similarity_threshold))

        # Artifacts: persist run/round/hypothesis middle outputs for audit/debugging
        art_cfg = config.get("artifacts", {}) or {}
        self.artifacts = ArtifactWriter(
            enabled=art_cfg.get("enabled", True),
            base_dir=art_cfg.get("base_dir", "./data/runs"),
            run_id=art_cfg.get("run_id"),
            save_llm_prompts=art_cfg.get("save_llm_prompts", True),
            save_llm_responses=art_cfg.get("save_llm_responses", True),
            save_generated_code=art_cfg.get("save_generated_code", True),
            save_data_fingerprint=art_cfg.get("save_data_fingerprint", True),
        )
        try:
            self.artifacts.write_run_manifest(
                config_snapshot=dict(config.config) if isinstance(config.config, dict) else {},
                goal=self.goal,
                data_path=data_path,
                memory_path=memory_path or config.get("memory.storage_path"),
            )
        except Exception:
            # Artifacts should never break discovery
            pass

        # Seed the insight graph with descriptive statistics if empty.
        self._seed_insights_from_data()
        # Seed with pre-scanned cross-module associations.
        self._seed_prescan_associations()

    def _seed_insights_from_data(self) -> None:
        """Seed the insight graph with per-module descriptive statistics.

        Only runs when the graph is empty (fresh start). Creates one seed
        insight per concept-menu theme, describing key variables and basic
        stats. This gives the orchestrator diverse anchors from round 1
        and enables frontier detection immediately.

        Works for any dataset — reads concept_menu and column_labels from
        the active dataset profile; falls back to variable_catalog config.
        """
        if self.insight_graph.get_all_insights():
            return  # graph already has insights, skip seeding
        if self.coder_runner.data is None or self.coder_runner.data.empty:
            return

        from core.dataset_profile import profile_get, column_labels as _col_labels
        import pandas as pd
        from datetime import datetime

        concept_menu = (
            self._variable_catalog_cfg.get("concept_menu")
            or profile_get("concept_menu")
            or []
        )
        if not concept_menu:
            return

        labels = _col_labels()
        df = self.coder_runner.data
        available = set(df.columns)

        seeds_added = 0
        for theme in concept_menu:
            # Extract column names from parenthetical: "Trust (Q57, Q58, ...)" or
            # "Rating and Helpfulness (rating, helpful_votes, ...)"
            cols_in_theme = []
            paren_match = re.search(r"\(([^)]+)\)", theme)
            if paren_match:
                raw = paren_match.group(1)
                # Handle Q-ranges like "Q57–Q105"
                range_match = re.match(r"Q(\d+)[–\-]Q?(\d+)", raw.strip())
                if range_match:
                    lo, hi = int(range_match.group(1)), int(range_match.group(2))
                    cols_in_theme = [c for c in available if re.match(r"Q(\d+)", c)
                                     and lo <= int(re.match(r"Q(\d+)", c).group(1)) <= hi]
                else:
                    # Comma-separated column names
                    for token in raw.split(","):
                        col = token.strip()
                        if col in available:
                            cols_in_theme.append(col)
                        else:
                            # Try case-insensitive match
                            for ac in available:
                                if ac.lower() == col.lower():
                                    cols_in_theme.append(ac)
                                    break

            if not cols_in_theme:
                continue

            # Compute basic stats for up to 5 key columns
            sample_cols = cols_in_theme[:5]
            stat_parts = []
            for col in sample_cols:
                label = labels.get(col, col)
                s = df[col]
                n_valid = int(s.notna().sum())
                pct_missing = (1 - n_valid / len(df)) * 100
                if pd.api.types.is_numeric_dtype(s):
                    try:
                        mean = float(s.mean())
                        std = float(s.std())
                        stat_parts.append(f"{col} ({label}): mean={mean:.2f}, std={std:.2f}, {pct_missing:.0f}% missing")
                    except Exception:
                        stat_parts.append(f"{col} ({label}): {n_valid} valid, {pct_missing:.0f}% missing")
                else:
                    nunique = int(s.nunique())
                    stat_parts.append(f"{col} ({label}): {nunique} categories, {pct_missing:.0f}% missing")

            # Theme name without parenthetical
            theme_name = re.sub(r"\s*\([^)]*\)", "", theme).strip()
            sentence = f"Dataset module '{theme_name}': {len(cols_in_theme)} variables. " + "; ".join(stat_parts) + "."

            insight = Insight(
                id=f"seed_{uuid.uuid4().hex[:8]}",
                sentence=sentence,
                source="statistical_profiling",
                created_at=datetime.now(),
                metadata={
                    "knowledge_type": "descriptive_stat",
                    "columns": cols_in_theme[:10],
                    "module": theme_name,
                    "is_seed": True,
                },
            )
            try:
                self.insight_graph.add_insight(insight)
                seeds_added += 1
            except Exception as e:
                print(f"  [WARN] Failed to add seed for '{theme_name}': {e}")

        if seeds_added:
            try:
                self.insight_graph.save()
            except Exception:
                pass
            print(f"[INFO] Seeded insight graph with {seeds_added} descriptive-stat insights from concept menu.")

    def _seed_prescan_associations(self) -> None:
        """Seed the insight graph with pre-scanned cross-module associations.

        Runs fast Pearson correlations on all substantive variable pairs and
        injects significant findings as seed insights.  Eliminates the
        cold-start problem and gives the orchestrator diverse anchors for
        exploration and depth-frontier deepening from round 1.

        Only runs when no ``seed_prescan`` insights exist (idempotent).
        """
        # Skip if prescan seeds already exist
        existing = self.insight_graph.get_all_insights()
        if any(i.source == "seed_prescan" for i in existing):
            return

        if self.coder_runner.data is None or self.coder_runner.data.empty:
            return

        from scipy import stats as sp_stats
        from scripts.seed_prescan import (
            get_substantive_columns,
            get_variable_modules_from_catalog,
            prescan_correlations,
            build_module_pair_seeds,
            inject_prescan_seeds,
        )
        from core.dataset_profile import (
            derived_pairs as _dp,
            column_labels as _col_labels,
        )

        df = self.coder_runner.data
        print("[INFO] Pre-scan seeding: computing cross-module associations...")

        # Get variable module mapping
        var_modules: Dict[str, str] = {}
        if self.variable_retriever is not None:
            try:
                all_cols = list(df.columns)
                cards = self.variable_retriever.get_cards_by_names(all_cols)
                for card in cards:
                    name = (card.var_name or "").strip()
                    mod = (card.module or card.section_theme or "").strip()
                    if name and mod:
                        var_modules[name] = mod
            except Exception:
                pass

        if not var_modules:
            # Fallback: try catalog file directly
            vc_cfg = self._variable_catalog_cfg or {}
            cat_path = str(vc_cfg.get("catalog_path", "") or "").strip()
            if cat_path and Path(cat_path).exists():
                var_modules = get_variable_modules_from_catalog(cat_path)

        cols = get_substantive_columns(df)
        if len(cols) < 2:
            return

        taut_pairs = _dp() or []
        labels = _col_labels() or {}

        results = prescan_correlations(df, cols, var_modules, taut_pairs)
        if not results:
            return

        seeds = build_module_pair_seeds(results, labels)
        count = inject_prescan_seeds(self.insight_graph, seeds)

        if count > 0:
            n_cross = sum(1 for _, info in seeds if info.get("cross"))
            mods = set()
            for _, info in seeds:
                for m in (info.get("mod_a", ""), info.get("mod_b", "")):
                    if m and m not in ("Identifiers", "Unknown"):
                        mods.add(m)
            print(
                f"[INFO] Pre-scan seeding: {count} module-pair briefs "
                f"({n_cross} cross-module), {len(mods)} modules covered."
            )

    def _select_schema_columns_for_round(
        self,
        context: RoundContext,
        idea_query: Optional[str] = None,
        allowed_q_ranges: Optional[Sequence[Tuple[int, int]]] = None,
        allowed_modules: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """
        If a variable catalog is configured, select a small subset of columns to inject
        into the Generator/Coder prompts for this round (token budget gate).
        When idea_query is provided (two-phase flow), use it for RAG instead of goal+focus+mode.
        When allowed_q_ranges or allowed_modules is set, retrieval is restricted accordingly.
        """
        if self.variable_retriever is None:
            return None
        if self.coder_runner.data is None:
            return None

        max_cols = self._variable_catalog_cfg.get("max_schema_columns")
        try:
            max_cols = int(max_cols) if max_cols is not None else 0
        except (TypeError, ValueError):
            max_cols = 0
        # 0 = no cap: use all retrieved + always_keep
        cap_schema = max_cols > 0

        always_keep = list(self._variable_catalog_cfg.get("always_keep", []) or [])
        available = set(self.coder_runner.data.columns)
        keep = [c for c in always_keep if c in available]

        if idea_query and idea_query.strip():
            query = idea_query.strip()
        else:
            insight_hints = ""
            try:
                if getattr(context, "retrieved_insights", None):
                    hint_sents = [i.sentence for i in (context.retrieved_insights or [])[:6] if getattr(i, "sentence", None)]
                    if hint_sents:
                        insight_hints = "\n".join([f"- {s}" for s in hint_sents])
            except Exception:
                insight_hints = ""
            query = (
                f"Goal: {context.goal}\n"
                f"Focus area: {context.focus_area}\n"
                f"Mode: {context.mode}\n"
                + (f"\nAnchor/context hints:\n{insight_hints}\n" if insight_hints else "")
            )
        max_retrieved = self._variable_catalog_cfg.get("max_retrieved_variables")
        try:
            max_retrieved = int(max_retrieved) if max_retrieved is not None else 0
        except (TypeError, ValueError):
            max_retrieved = 0
        cap_retrieved = max_retrieved > 0

        top_k = max(10, max_cols) if cap_schema else max(len(available), 500)
        if cap_retrieved:
            top_k = max(top_k, max_retrieved)
        # Normalize/deduplicate ranges for stable downstream behavior.
        normalized_ranges: List[Tuple[int, int]] = []
        if allowed_q_ranges:
            seen_ranges = set()
            for rg in allowed_q_ranges:
                if not rg or len(rg) != 2:
                    continue
                lo, hi = int(rg[0]), int(rg[1])
                if lo > hi:
                    lo, hi = hi, lo
                key = (lo, hi)
                if key in seen_ranges:
                    continue
                seen_ranges.add(key)
                normalized_ranges.append(key)
        normalized_ranges.sort()

        candidates: List[str] = []
        # Multi-range strategy: guarantee some coverage from each selected theme range,
        # especially important when max_retrieved_variables is small.
        if len(normalized_ranges) > 1:
            target = max_retrieved if cap_retrieved else min(top_k, 80)
            quota = max(1, int(math.ceil(float(target) / float(len(normalized_ranges)))))
            for rg in normalized_ranges:
                try:
                    hits_r = self.variable_retriever.retrieve(
                        query,
                        top_k=max(quota * 3, quota),
                        only_vars=available,
                        allowed_q_ranges=[rg],
                        allowed_modules=allowed_modules,
                    )
                    candidates.extend([r["var_name"] for r in hits_r[:quota]])
                except Exception:
                    continue
            # Fill remaining slots from the union of selected ranges.
            union_hits = self.variable_retriever.retrieve(
                query,
                top_k=top_k,
                only_vars=available,
                allowed_q_ranges=normalized_ranges,
                allowed_modules=allowed_modules,
            )
            candidates.extend([r["var_name"] for r in union_hits])
        else:
            retrieved = self.variable_retriever.retrieve(
                query,
                top_k=top_k,
                only_vars=available,
                allowed_q_ranges=normalized_ranges or None,
                allowed_modules=allowed_modules,
            )
            candidates = [r["var_name"] for r in retrieved]

        # Two-phase guardrail:
        # when retrieval is constrained to a theme range/module, add a small covariate pass
        # so common controls (demographics/country/weights) remain available.
        supplemental: List[str] = []
        if normalized_ranges or allowed_modules:
            covariate_query = (
                str(self._variable_catalog_cfg.get("covariate_query", "") or "").strip()
                or "age gender sex education income employment marital household children urban rural religion country weight survey year"
            )
            cov_top_k = 24 if not cap_retrieved else max(8, min(24, max_retrieved))
            try:
                cov_hits = self.variable_retriever.retrieve(
                    covariate_query,
                    top_k=cov_top_k,
                    only_vars=available,
                    allowed_q_ranges=None,  # intentionally unconstrained by theme
                )
                supplemental = [r["var_name"] for r in cov_hits]
            except Exception:
                supplemental = []

        if cap_retrieved:
            if supplemental:
                reserve = max(1, min(3, max_retrieved // 3))
                core_budget = max(1, max_retrieved - reserve)
                candidates = candidates[:core_budget] + supplemental[:reserve]
            else:
                candidates = candidates[:max_retrieved]
        else:
            candidates = candidates + supplemental

        seen = set()
        selected: List[str] = []
        for c in keep + candidates:
            if c in seen:
                continue
            if c not in available:
                continue
            seen.add(c)
            selected.append(c)
            if cap_schema and len(selected) >= max_cols:
                break

        return selected

    def _build_variable_glossary_for_prompt(
        self,
        *,
        query: str,
        available_cols: Sequence[str],
        max_cards: int = 25,
        allowed_q_ranges: Optional[Sequence[Tuple[int, int]]] = None,
        allowed_modules: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Build a compact variable glossary for LLM grounding.
        Important: do NOT use '-' bullet prefix (Generator/CoderRunner parse columns from '-' lines).
        """
        if self.variable_retriever is None:
            return None
        if not available_cols:
            return None

        top_k = max_cards if max_cards > 0 else max(len(available_cols), 500)
        retrieved = self.variable_retriever.retrieve(
            query,
            top_k=top_k,
            only_vars=set(available_cols),
            allowed_q_ranges=allowed_q_ranges,
            allowed_modules=allowed_modules,
        )
        if not retrieved:
            return None

        lines: List[str] = []
        lines.append("VARIABLE GLOSSARY (for meaning; use column names EXACTLY as in schema):")
        use = retrieved[:max_cards] if max_cards > 0 else retrieved
        for r in use:
            name = r.get("var_name", "")
            detail = self._format_variable_glossary_detail(r)
            if not detail:
                detail = "(no description available)"

            # IMPORTANT: use '•' not '-' to avoid being parsed as schema columns.
            lines.append(f"• {name}: {detail}")

        return "\n".join(lines)

    def _format_variable_glossary_detail(self, record: Dict[str, Any]) -> str:
        return build_variable_glossary_detail(
            record,
            use_generalized=self._use_generalized_glossary,
            include_raw_question_fallback=self._include_raw_question_fallback,
            include_raw_options=self._include_raw_options_in_glossary,
        )

    def _get_variable_key_for_display(
        self,
        selected_cols: Optional[List[str]],
        query: str = "",
        max_items: int = 40,
        max_label_len: Optional[int] = 90,
    ) -> Dict[str, str]:
        """Build var_name -> label for console display (so Qxx is interpretable).

        Uses direct card lookup instead of TF-IDF retrieval — display labels
        don't need relevance ranking, just name → label mapping.
        Use max_label_len=None for full labels.
        """
        if not selected_cols or self.variable_retriever is None:
            return {}
        try:
            cards = self.variable_retriever.get_cards_by_names(selected_cols[:max_items])
            key: Dict[str, str] = {}
            for card in cards:
                name = (card.var_name or "").strip()
                if not name:
                    continue
                label = (card.label or "").strip()
                question = (card.question or "").strip()
                text = label or question or "(no description)"
                key[name] = text if max_label_len is None else text[:max_label_len]
            return key
        except Exception:
            return {}

    def _sentence_with_variable_labels(self, sentence: str, var_key: Dict[str, str]) -> str:
        """Return sentence with variable codes replaced by 'CODE (label)' for readability."""
        if not var_key:
            return sentence
        out = sentence
        # Replace longer names first to avoid Q5 matching inside Q58
        for name in sorted(var_key.keys(), key=len, reverse=True):
            label = var_key[name]
            out = out.replace(name, f"{name} ({label})")
        return out

    @staticmethod
    def _q_number(var: str) -> Optional[int]:
        """Extract the numeric part from a WVS Q-variable name (e.g., 'Q113' → 113)."""
        m = re.match(r"Q(\d+)", var)
        return int(m.group(1)) if m else None

    def _are_same_battery(self, var_a: str, var_b: str) -> bool:
        """Check if two Q-variables are from the same questionnaire battery.

        Heuristic (validated against all known WVS trivial pairs with zero false positives):
        Q-number gap ≤ 3 AND same codebook module → same item battery.
        """
        if self.variable_retriever is None:
            return False
        qa = self._q_number(var_a)
        qb = self._q_number(var_b)
        if qa is None or qb is None:
            return False
        if abs(qa - qb) > 3:
            return False
        cards = self.variable_retriever.get_cards_by_names([var_a, var_b])
        if len(cards) < 2:
            return False
        return bool(cards[0].module and cards[0].module == cards[1].module)

    @staticmethod
    def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return float(len(sa.intersection(sb)) / len(sa.union(sb)))

    @staticmethod
    def _parse_anchors(hyp_relations: str) -> List[str]:
        if not hyp_relations:
            return []
        m = re.search(r"ANCHORS\s*=\s*\[(.*?)\]", hyp_relations, flags=re.IGNORECASE)
        if not m:
            return []
        raw = m.group(1)
        return [p.strip() for p in raw.split(",") if p and p.strip()]

    def _phrase_grounded_by_retrieval(
        self,
        phrase: str,
        available_cols: set,
        min_score: float = 0.05,
    ) -> bool:
        """Check if an anchor phrase retrieves at least one variable with meaningful similarity.

        Queries the retriever *independently* per phrase, avoiding the circularity of
        checking phrases against a corpus that was built from those same phrases.
        """
        if not phrase or not phrase.strip() or self.variable_retriever is None:
            return True
        try:
            hits = self.variable_retriever.retrieve(
                phrase.strip(),
                top_k=3,
                only_vars=available_cols,
            )
            if not hits:
                return False
            # At least one result with a non-trivial similarity score
            return any(float(h.get("score", 0)) >= min_score for h in hits)
        except Exception:
            return True  # Don't block on retriever errors

    @staticmethod
    def _infer_secondary_theme_from_text(
        concept_menu: Sequence[str],
        primary_theme: str,
        text: str,
    ) -> Optional[Tuple[int, str]]:
        """
        Heuristic fallback: when Phase-1 returns SECONDARY_THEME=NONE but the idea text
        clearly spans cross-module concepts, pick one best secondary theme.
        Returns (1-based index, exact menu text) or None.
        """
        t = (text or "").lower()
        if not t:
            return None
        primary_low = (primary_theme or "").lower()

        # --- Build signals from profile theme_keywords if available ----------
        _SIGNALS: List[Tuple[List[str], str, int]] = []
        try:
            from core.dataset_profile import theme_keywords as _ptk
            _profile_kw = _ptk()
        except Exception:
            _profile_kw = {}

        if _profile_kw:
            # Profile-driven: each module_name -> [keywords] becomes a signal
            # using the module name (lowered) as the menu-substring matcher.
            for module_name, kw_list in _profile_kw.items():
                _SIGNALS.append((list(kw_list), module_name.lower(), 4))
        else:
            # Fallback: hardcoded WVS7 keyword -> menu-substring mappings.
            _SIGNALS = [
                # Social values & norms
                (["norm", "stereotype", "gender role", "tolerance", "justifiable", "moral"], "social values", 4),
                # Happiness & wellbeing
                (["well-being", "wellbeing", "happiness", "life satisfaction", "satisfaction with life", "subjective well"], "happiness and wellbeing", 5),
                # Social capital & trust
                (["trust", "social capital", "civic", "community", "organizational membership", "volunteer"], "social capital", 4),
                # Economic values
                (["economic", "inequality", "income distribution", "competition", "private ownership", "wealth"], "economic values", 4),
                # Corruption
                (["corruption", "bribery", "nepotism", "transparency"], "corruption", 4),
                # Migration
                (["migration", "immigration", "immigrant", "refugee", "foreigner", "ethnic diversity"], "migration", 4),
                # Security
                (["security", "terrorism", "crime", "violence", "war", "armed conflict", "safety"], "security", 4),
                # Postmaterialism
                (["postmaterial", "value change", "materialis", "inglehart", "self-expression"], "postmaterialism", 4),
                # Science & technology
                (["science", "technology", "innovation", "artificial intelligence", "biotechnology"], "science and technology", 4),
                # Religious values
                (["religio", "god", "prayer", "faith", "spiritual", "secular", "church", "mosque"], "religious", 4),
                # Ethical values
                (["ethic", "moral", "justifiable", "abortion", "euthanasia", "homosexual", "divorce", "suicide"], "ethical", 4),
                # Political interest & participation
                (["political", "democracy", "election", "voting", "participation", "protest", "petition"], "political interest and participation", 4),
                # Political culture & regimes
                (["regime", "authoritarian", "autocra", "democratic system", "army rule", "government type"], "political culture and political regimes", 4),
                # Demographics
                (["demographic", "education level", "employment", "socioeconomic", "age group", "marital status"], "demographics", 3),
            ]

        scored: List[Tuple[int, int, str]] = []
        for i, item in enumerate(concept_menu, 1):
            ilow = item.lower()
            if ilow == primary_low:
                continue
            score = 0
            for keywords, menu_substr, weight in _SIGNALS:
                if any(k in t for k in keywords):
                    if menu_substr in ilow:
                        score += weight
            if score > 0:
                scored.append((score, i, item))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], x[1]))
        _, idx, theme = scored[0]
        return idx, theme

    def _is_general_idea_grounded(
        self,
        general_idea: Dict[str, Any],
        selected_cols: Optional[List[str]],
        query: str,
    ) -> Tuple[bool, float, List[str]]:
        """
        Check whether Phase-1 constructs/anchors are grounded in available variables.

        Uses per-anchor independent retrieval to avoid the circularity of checking
        phrases against a corpus that was built from the same idea_query.
        Returns (ok, ratio, missing_phrases).
        """
        if not selected_cols or self.variable_retriever is None:
            return True, 1.0, []

        available = set(selected_cols)

        phrases: List[str] = []
        # Prefer ANCHORS (intended for RAG / glossary matching),
        # fall back to CORE_CONSTRUCTS if no anchors are provided.
        anchors_raw = general_idea.get("anchors", [])
        if isinstance(anchors_raw, list):
            anchors = [str(a).strip() for a in anchors_raw if str(a).strip()]
        elif isinstance(anchors_raw, str):
            anchors = [a.strip() for a in anchors_raw.split(",") if a.strip()]
        else:
            # Legacy fallback: try extracting from hypothesized_relations
            anchors = self._parse_anchors(str(general_idea.get("hypothesized_relations") or ""))
        core = str(general_idea.get("core_constructs") or "")
        core_phrases = [p.strip() for p in core.split(",") if p.strip()]
        phrases.extend(anchors if anchors else core_phrases)
        if not phrases:
            return True, 1.0, []

        # Per-anchor independent retrieval: each phrase is queried separately
        # against the variable catalog, constrained to selected columns.
        missing = [p for p in phrases if not self._phrase_grounded_by_retrieval(p, available)]
        ratio = float((len(phrases) - len(missing)) / max(1, len(phrases)))
        min_ratio_cfg = self._variable_catalog_cfg.get("min_grounding_ratio", 0.55)
        try:
            min_ratio = float(min_ratio_cfg)
        except (TypeError, ValueError):
            min_ratio = 0.55
        return ratio >= min_ratio, ratio, missing

    @staticmethod
    def _is_strict_dsl_hypothesis(sentence: str) -> bool:
        s = (sentence or "").strip()
        if not s:
            return False
        if ":" in s:
            return False
        s_low = s.lower()
        if not (
            s_low.startswith("assoc(")
            or s_low.startswith("diff(")
            or s_low.startswith("interact(")
            or s_low.startswith("heterogeneity(")
        ):
            return False
        # "controlling for ..." is optional in the two-phase flow. When present, it must come after the DSL body.
        body = s_low.split(" controlling for ", 1)[0].strip()
        if body.startswith("assoc(") and "|" in body:
            return False
        if body.startswith("diff(") and "by=" not in body:
            return False
        if body.startswith("interact(") and "->" not in body:
            return False
        if body.startswith("heterogeneity(") and ("~" not in body or "|" not in body):
            return False
        return True

    @staticmethod
    def _schema_columns(data_schema: Optional[str]) -> List[str]:
        if not data_schema:
            return []
        out: List[str] = []
        for line in data_schema.splitlines():
            t = line.strip()
            if not t.startswith("-"):
                continue
            name = t.lstrip("-").split("(")[0].strip()
            if name:
                out.append(name)
        return out

    @staticmethod
    def _has_col_token(text: str, col: str) -> bool:
        return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(col)}(?![A-Za-z0-9_])", text))

    def _validate_two_phase_hypothesis(self, sentence: str, data_schema: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Lightweight guardrail for common prompt failures:
        - Weight columns as substantive X/Y variables
        - ID columns used as numeric predictors instead of grouping
        - Text columns used directly (would crash on non-numeric data)
        - Control-only columns as main predictors
        """
        if not sentence:
            return False, "EMPTY_SENTENCE"
        if not self._is_strict_dsl_hypothesis(sentence):
            return False, "INVALID_DSL_FORMAT"
        sentence = self._normalize_dsl_brackets(sentence)
        lower = sentence.lower()
        if " controlling for " in lower:
            parts = lower.split(" controlling for ", 1)
            main_part = parts[0].strip()
            controls_part = parts[1].strip() if len(parts) > 1 else ""
        else:
            main_part = lower.strip()
            controls_part = ""

        schema_cols = self._schema_columns(data_schema)
        schema_cols_low = [c.lower() for c in schema_cols]
        schema_set = set(schema_cols_low)

        # Optional weights=COL slot (must not appear in controls list)
        weight_slot = None
        wm = re.search(r"\bweights\s*=\s*([a-z_][a-z0-9_]*)", lower)
        if wm:
            weight_slot = wm.group(1).strip()
            if weight_slot not in schema_set:
                return False, "INVALID_WEIGHT_SLOT"
            controls_part = re.sub(r"\bweights\s*=\s*[a-z_][a-z0-9_]*", "", controls_part).strip(" ,")
            main_part = re.sub(r"\bweights\s*=\s*[a-z_][a-z0-9_]*", "", main_part).strip(" ,")

        from core.dataset_profile import weight_columns as _pw, id_columns as _pi, text_columns as _ptc

        # Text columns must never appear in DSL hypotheses (non-numeric, will crash executor)
        _text_cols = set(str(c).lower() for c in (_ptc() or []))
        if _text_cols:
            for tc in _text_cols:
                if tc in schema_set and (self._has_col_token(main_part, tc) or self._has_col_token(controls_part, tc)):
                    return False, f"TEXT_COLUMN_IN_DSL:{tc}"

        # High-cardinality categoricals: reject early to avoid slow regressions.
        # Check columns used as group (by=...), moderator (X * Z), or controls.
        from core.dataset_profile import known_categorical as _pkc
        from core.dsl import _MAX_CATEGORICAL_CARDINALITY
        _known_cats = set(str(c).lower() for c in (_pkc() or []))
        if _known_cats and self.coder_runner.data is not None:
            df = self.coder_runner.data
            for cat_col in _known_cats:
                if cat_col not in schema_set:
                    continue
                # Find the actual column name (case-sensitive)
                actual_col = next((c for c in df.columns if c.lower() == cat_col), None)
                if actual_col is None:
                    continue
                card = int(df[actual_col].nunique())
                if card > _MAX_CATEGORICAL_CARDINALITY:
                    if self._has_col_token(main_part, cat_col) or self._has_col_token(controls_part, cat_col):
                        return False, f"HIGH_CARDINALITY_CATEGORICAL:{cat_col}({card})"

        # --- Grouping position cardinality guard ---
        # For diff(Y, by=G) and heterogeneity(Y ~ X | G), G must have
        # 2 <= cardinality <= MAX.  This catches ANY column in grouping
        # position, not just known_categorical.  Also enforces the
        # profile's grouping_excluded list.
        from core.dataset_profile import profile_get as _pg
        _grouping_excluded = set(
            str(c).lower() for c in (_pg("grouping_excluded", []) or [])
        )
        _drop_columns = set(
            str(c).lower() for c in (_pg("drop_columns", []) or [])
        )
        # Reject any dropped column appearing anywhere in the hypothesis
        for dc in _drop_columns:
            if dc in schema_set and (
                self._has_col_token(main_part, dc)
                or self._has_col_token(controls_part, dc)
            ):
                return False, f"DROPPED_COLUMN:{dc}"

        # Extract the grouping variable from by= or | positions
        _grouping_col = None
        by_match = re.search(r"\bby\s*=\s*([a-z_][a-z0-9_]*)", main_part)
        if by_match:
            _grouping_col = by_match.group(1).strip()
        pipe_match = re.search(r"\|\s*([a-z_][a-z0-9_]*)", main_part)
        if pipe_match and _grouping_col is None:
            _grouping_col = pipe_match.group(1).strip()

        if _grouping_col and _grouping_col in schema_set:
            # Check profile exclusion list
            if _grouping_col in _grouping_excluded:
                return False, f"GROUPING_EXCLUDED:{_grouping_col}"
            # Check actual cardinality in data — uses the grouping-specific
            # limit (looser than _MAX_CATEGORICAL_CARDINALITY which governs
            # regression dummy variables).
            from core.dsl import _MAX_GROUPING_CARDINALITY
            if self.coder_runner.data is not None:
                actual_col = next(
                    (c for c in self.coder_runner.data.columns if c.lower() == _grouping_col),
                    None,
                )
                if actual_col is not None:
                    card = int(self.coder_runner.data[actual_col].nunique(dropna=True))
                    if card < 2:
                        return False, f"GROUPING_TOO_FEW_LEVELS:{_grouping_col}({card})"
                    if card > _MAX_GROUPING_CARDINALITY:
                        return False, f"GROUPING_TOO_MANY_LEVELS:{_grouping_col}({card})"

        # --- Interact position guard: both predictor (X) and moderator (Z) ---
        # interact(X * Z -> Y) expands both X and Z into dummies if they are
        # string/categorical columns.  High-cardinality columns in EITHER
        # position create massive design matrices that OOM-kill the process.
        _interact_match = re.match(
            r"^interact\(\s*([a-z_]\w*)\s*\*\s*([a-z_]\w*)\s*->\s*([a-z_]\w*)",
            main_part,
        )
        if _interact_match:
            _predictor_col = _interact_match.group(1).strip()
            _moderator_col = _interact_match.group(2).strip()
            for _icol, _irole in [(_predictor_col, "predictor"), (_moderator_col, "moderator")]:
                if _icol not in schema_set:
                    continue
                if _icol in _grouping_excluded:
                    return False, f"INTERACT_{_irole.upper()}_EXCLUDED:{_icol}"
                # Check actual cardinality for string/object columns —
                # statsmodels auto-dummies them regardless of known_categorical.
                if self.coder_runner.data is not None:
                    from core.dsl import _MAX_CATEGORICAL_CARDINALITY
                    actual_col = next(
                        (c for c in self.coder_runner.data.columns if c.lower() == _icol),
                        None,
                    )
                    if actual_col is not None:
                        col_dtype = self.coder_runner.data[actual_col].dtype
                        if col_dtype == object or str(col_dtype) == "category":
                            card = int(self.coder_runner.data[actual_col].nunique(dropna=True))
                            if card > _MAX_CATEGORICAL_CARDINALITY:
                                return False, f"INTERACT_{_irole.upper()}_TOO_MANY_LEVELS:{_icol}({card})"

        _profile_weights = _pw()
        weight_cols = set(str(w).lower() for w in _profile_weights) if _profile_weights else set(config.get("data.weight_columns", []) or [])
        if not _profile_weights:
            weight_cols.update({"sampling_weight", "pwght", "w_weight"})

        _profile_ids = _pi()
        id_cols = set(str(c).lower() for c in _profile_ids) if _profile_ids else {"b_country", "b_country_alpha", "country", "country_code", "survey_id"}
        id_cols = {c for c in id_cols if c in schema_set}
        forbidden_control_ids = set(str(c).lower() for c in _profile_ids) if _profile_ids else {"survey_id"}

        # Weights are never meaningful outcomes/predictors in the hypothesis body.
        for w in weight_cols:
            wl = str(w).lower()
            if wl and self._has_col_token(main_part, wl):
                return False, f"INVALID_WEIGHT_ROLE:{w}"
            if wl and self._has_col_token(controls_part, wl):
                return False, f"INVALID_WEIGHT_CONTROL:{w}"
        if weight_slot and weight_slot not in {str(w).lower() for w in weight_cols}:
            return False, "INVALID_WEIGHT_SLOT_COLUMN"

        # Control-only columns (fieldwork dates, survey year) must not appear
        # as main predictor/outcome/grouping — they are confounders, not
        # substantive variables.  Unlike ID columns (B_COUNTRY) which are
        # legitimate stratifiers, these should ONLY appear after "controlling for".
        for co in self._control_only_cols:
            if co in schema_set and self._has_col_token(main_part, co):
                return False, f"CONTROL_ONLY_AS_MAIN:{co}"

        # ID/country columns must be used as grouping/stratification only.
        for ident in id_cols:
            if self._has_col_token(main_part, ident):
                allowed = (
                    f"by={ident}" in main_part
                    or f"by {ident}" in main_part
                    or f"| {ident}" in main_part
                    or f"|{ident}" in main_part
                )
                if not allowed:
                    return False, f"INVALID_ID_ROLE:{ident}"
        # Additional hygiene for controls: survey identifiers should not be controls.
        for ident in forbidden_control_ids:
            if ident in schema_set and self._has_col_token(controls_part, ident):
                return False, f"INVALID_ID_CONTROL:{ident}"

        # Explicitly reject assoc(B_COUNTRY, ...) forms.
        if "assoc(" in main_part and any(self._has_col_token(main_part, ident) for ident in id_cols):
            return False, "INVALID_ASSOC_WITH_COUNTRY_ID"

        # Enforce assoc(X, Y) with atomic column terms (no arithmetic expressions).
        if main_part.startswith("assoc(") and ")" in main_part:
            inside = main_part[len("assoc("):main_part.rfind(")")]
            args = [a.strip() for a in inside.split(",") if a.strip()]
            if len(args) != 2:
                return False, "INVALID_ASSOC_ARITY"
            for arg in args:
                if any(op in arg for op in ["+", "-", "*", "/", "|", "~"]):
                    return False, "INVALID_ASSOC_COMPLEX_EXPR"
                if arg not in schema_set:
                    return False, "INVALID_ASSOC_COLUMN"

        # Validate core column names for diff/interact/heterogeneity
        # (assoc is already checked above).  main_part is already lowercased
        # and schema_set contains lowercase column names.
        _dsl_col_patterns = [
            (r"^diff\(\s*([a-z_]\w*)\s*,\s*by\s*=\s*([a-z_]\w*)", ["outcome", "group"]),
            (r"^interact\(\s*([a-z_]\w*)\s*\*\s*([a-z_]\w*)\s*->\s*([a-z_]\w*)", ["predictor", "moderator", "outcome"]),
            (r"^heterogeneity\(\s*([a-z_]\w*)\s*~\s*([a-z_]\w*)\s*\|\s*([a-z_]\w*)", ["outcome", "predictor", "group"]),
        ]
        for pattern, roles in _dsl_col_patterns:
            _m = re.match(pattern, main_part)
            if _m:
                matched_cols = []
                for i, role in enumerate(roles):
                    col_name = _m.group(i + 1).strip()
                    if col_name not in schema_set:
                        return False, f"INVALID_{role.upper()}_COLUMN:{col_name}"
                    matched_cols.append(col_name)
                # Reject if predictor/X == outcome/Y (self-correlation)
                if len(matched_cols) >= 2 and matched_cols[0] == matched_cols[-1]:
                    return False, "SELF_CORRELATION"

        # Ensure at least two schema columns are referenced.
        cols = self._extract_columns_from_sentence(sentence, data_schema)
        if len(cols) < 2:
            return False, "TOO_FEW_SCHEMA_COLUMNS"

        # Same-battery check: assoc(X, Y) where X and Y are adjacent items
        # from the same questionnaire section is a measurement artifact, not a discovery.
        if main_part.startswith("assoc(") and ")" in main_part:
            inside = main_part[len("assoc("):main_part.rfind(")")]
            args = [a.strip() for a in inside.split(",") if a.strip()]
            if len(args) == 2:
                # Resolve to actual column names (case-insensitive)
                resolved = []
                for arg in args:
                    for sc in schema_cols:
                        if sc.lower() == arg:
                            resolved.append(sc)
                            break
                if len(resolved) == 2 and self._are_same_battery(resolved[0], resolved[1]):
                    return False, "SAME_BATTERY_ASSOC"

        # Country-diff substantive-control check: diff(Y, by=GROUP_COL) is trivially
        # significant in multi-group surveys. Require at least one non-demographic
        # variable as control to make it a genuine conditional finding.
        from core.dataset_profile import demographic_synonyms as _pds, known_categorical as _pkc

        # Build demographic/non-substantive column set from profile
        _profile_demo_syns = _pds()
        _profile_cat = _pkc()
        _DEMOGRAPHIC_COLS: set = set()
        if _profile_demo_syns:
            # Add synonym keys (natural language) and values (column names)
            _DEMOGRAPHIC_COLS.update(str(k).lower() for k in _profile_demo_syns.keys())
            _DEMOGRAPHIC_COLS.update(str(v).lower() for v in _profile_demo_syns.values())
        if _profile_cat:
            _DEMOGRAPHIC_COLS.update(str(c).lower() for c in _profile_cat)
        _DEMOGRAPHIC_COLS.update(str(c).lower() for c in (self._control_only_cols or set()))
        _DEMOGRAPHIC_COLS.update(id_cols)
        _DEMOGRAPHIC_COLS.update(weight_cols)
        if not _profile_demo_syns and not _profile_cat:
            # Legacy WVS fallback
            _DEMOGRAPHIC_COLS = {
                "q260", "q262", "q273", "q274", "q275", "q275a", "q279",
                "q287", "q288", "q289", "q290",
                "h_urbrural", "b_country", "b_country_alpha",
                "a_year", "year", "fw_start", "fw_end",
                "sampling_weight", "w_weight", "pwght", "survey_id",
                # Natural-language aliases the LLM sometimes uses
                "gender", "sex", "age", "age_group", "education", "education_level",
                "income", "income_level", "income_category", "income_status",
                "employment_status", "marital_status", "marital status",
                "urban", "rural", "urbanicity",
            }

        # Only apply the diff-by-group guard when id_cols look like grouping
        # columns (e.g. country).  Record-level IDs like "paperid" are never
        # used as ``by=paperid`` so the check would be vacuous.
        _grouping_ids = {c for c in id_cols if any(
            self._has_col_token(main_part, c) and (f"by={c}" in main_part or f"by {c}" in main_part)
            for _ in [None]
        )}
        if main_part.startswith("diff(") and _grouping_ids:
            # This is a diff-by-group/ID. Check for substantive controls.
            ctrl_tokens = [c.strip() for c in controls_part.split(",") if c.strip()] if controls_part else []
            has_substantive = any(
                t.lower() not in _DEMOGRAPHIC_COLS and self._q_number(t) is not None
                for t in ctrl_tokens
            )
            if not has_substantive:
                return False, "BARE_COUNTRY_DIFF"

        # Same guard for heterogeneity(Y ~ X | G): "does X→Y vary across countries?"
        # is trivially true in a multi-country survey, just like bare country-diff.
        if main_part.startswith("heterogeneity(") and "|" in main_part:
            # Extract the grouping variable after |
            pipe_idx = main_part.index("|")
            group_part = main_part[pipe_idx + 1:].strip().rstrip(")")
            group_low = group_part.strip().lower()
            if group_low in {c.lower() for c in id_cols}:
                ctrl_tokens = [c.strip() for c in controls_part.split(",") if c.strip()] if controls_part else []
                has_substantive = any(
                    t.lower() not in _DEMOGRAPHIC_COLS and self._q_number(t) is not None
                    for t in ctrl_tokens
                )
                if not has_substantive:
                    return False, "BARE_COUNTRY_HETEROGENEITY"

        return True, None

    @staticmethod
    def _wrap_print(text: str, indent: str = "      ", width: int = 100) -> None:
        """Print text with word-wrap at width; continuation lines use indent."""
        if not text:
            return
        line = indent
        for word in text.split():
            if line.strip() and len(line) - len(indent) + len(word) + 1 > width:
                print(line.rstrip())
                line = indent + word
            else:
                line = (line + " " + word) if line.strip() else indent + word
        if line.strip():
            print(line.rstrip())
    
    def _extract_columns_from_sentence(self, sentence: str, data_schema: Optional[str]) -> List[str]:
        """
        Heuristic: extract column names by matching known schema columns
        that appear in the sentence (case-insensitive).
        
        Args:
            sentence: Hypothesis sentence to extract columns from
            data_schema: Data schema string with column definitions
            
        Returns:
            List of column names found in the sentence
        """
        if not data_schema:
            return []
        
        # Same way generator extracts column names
        column_lines = [
            line for line in data_schema.split('\n')
            if line.strip().startswith('-')
        ]
        column_names = []
        for line in column_lines:
            col_name = line.strip().lstrip('-').split('(')[0].strip()
            if col_name:
                column_names.append(col_name)
        
        sentence_lower = sentence.lower()
        used_cols = []
        for col in column_names:
            # Use word-boundary matching to avoid "c5" matching inside "log_c5"
            if self._has_col_token(sentence_lower, col.lower()):
                used_cols.append(col)

        return used_cols

    def _extract_core_columns_from_sentence(self, sentence: str, data_schema: Optional[str]) -> List[str]:
        """Extract only the core DSL variables (X, Y, G, Z) — before 'controlling for'.

        Two hypotheses that test the same core relationship but differ only in
        controls should be treated as structural duplicates.
        """
        if not sentence:
            return []
        # Strip everything after "controlling for" to get the core DSL body
        lower = sentence.lower()
        core_part = lower.split(" controlling for")[0].strip()
        # Also strip weights= clause
        core_part = re.sub(r"\s*weights\s*=\s*\S+", "", core_part).strip()
        return self._extract_columns_from_sentence(core_part, data_schema)

    @staticmethod
    def _normalize_dsl(sentence: str) -> str:
        """Normalize a DSL sentence for deduplication.

        - lowercase + strip
        - sort assoc() arguments (assoc is commutative: assoc(X,Y) == assoc(Y,X))
        """
        s = (sentence or "").strip().lower()
        # Normalize assoc(A, B) → assoc(sorted_first, sorted_second)
        m = re.match(r"assoc\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", s)
        if m:
            args = sorted([m.group(1).strip(), m.group(2).strip()])
            s = f"assoc({args[0]}, {args[1]})" + s[m.end():]
        return s

    # Max deepenings admitted per base finding, to bound duplicate inflation.
    MAX_DEEPENINGS_PER_PAIR = 3

    @staticmethod
    def _normalize_dsl_brackets(sentence: str) -> str:
        """Strip square-bracket suffixes the model copies from the prompt template.

        e.g. "assoc(X, Y) [controlling for C1, C2] [weights=W]" ->
             "assoc(X, Y) controlling for C1, C2 weights=W".

        Left unnormalised these are stored verbatim as `dsl_sentence`, graph
        retrieval re-surfaces them as round context, and the model imitates the
        format it sees — a self-reinforcing loop.
        """
        if not sentence or "[" not in sentence:
            return sentence
        s = re.sub(r"\[\s*(?:optional\s*:\s*)?(controlling for|weights\s*=)",
                   r"\1", sentence, flags=re.IGNORECASE)
        s = s.replace("]", " ")
        return re.sub(r"\s{2,}", " ", s).strip()

    @staticmethod
    def _dsl_rank(sentence: str) -> int:
        """Epistemic depth from the DSL head: assoc/diff=1, interact=2, heterogeneity=3.

        Read straight from the sentence because the duplicate filter runs before
        CoderRunner parses the spec, so metadata['test_type'] is not set yet.
        """
        s = (sentence or "").strip().lower()
        if s.startswith("heterogeneity("):
            return 3
        if s.startswith("interact("):
            return 2
        if s.startswith(("assoc(", "diff(")):
            return 1
        return 0

    def _is_strict_deepening(
        self,
        hypothesis: Hypothesis,
        core_cols: List[str],
        matches: List[Insight],
    ) -> bool:
        """True when *hypothesis* strictly deepens EVERY structural match.

        A deeper test over a proper superset of an existing finding's core
        variables asks a genuinely new question (does the effect hold once we
        condition on Z?), so it must not be discarded as a structural duplicate.
        This is the exact condition relation_classifier uses for DEEPENS, so
        admitting it here keeps DEEPENS edges reachable.

        Deliberately narrow — all of these remain duplicates:
          * a bare repeat of the same test (cores equal, so not a superset)
          * the same test with different controls (controls are not core)
          * an equal-or-shallower test on the same variables
        """
        new_core = {c for c in core_cols if c}
        new_rank = self._dsl_rank(hypothesis.sentence)
        if not new_core or new_rank < 2:
            return False

        for ins in matches:
            meta = ins.metadata or {}
            parsed = meta.get("parsed") or {}
            m_core = {parsed.get(k) for k in ("x", "y", "group", "moderator") if parsed.get(k)}
            if not m_core:
                m_core = set(meta.get("columns") or [])
            m_core = {c for c in m_core if c}
            if not m_core:
                return False
            # Must be a PROPER superset at strictly greater depth.
            if not (m_core < new_core):
                return False
            if new_rank <= self._dsl_rank(meta.get("dsl_sentence", "") or ins.sentence):
                return False

            # Bound how many deepenings one base finding may spawn.
            existing = 0
            for other in self.historian.get_all_insights():
                o_meta = other.metadata or {}
                o_parsed = o_meta.get("parsed") or {}
                o_core = {o_parsed.get(k) for k in ("x", "y", "group", "moderator") if o_parsed.get(k)}
                if o_core and m_core < o_core:
                    existing += 1
                    if existing >= self.MAX_DEEPENINGS_PER_PAIR:
                        return False
        return True

    def _filter_hypotheses_novelty_and_duplicates(
        self,
        hypotheses: List[Hypothesis],
        context: RoundContext,
        rejected_history: List[str],
        *,
        use_precritic_novelty_filter: bool = True,
        novelty_threshold: float = 0.25,
    ) -> Tuple[List[Hypothesis], List[Dict[str, Any]]]:
        """
        Apply novelty and duplicate filters to hypotheses; update rejected_history and rejection_storage.
        Returns (list of kept hypotheses, list of status dicts for artifacts).
        """
        # Build cross-round rejection index (normalized) — O(1) lookup per hypothesis.
        cross_round_rejected: set = set()
        try:
            for s in self.rejection_storage.get_all_sentences():
                cross_round_rejected.add(self._normalize_dsl(s))
        except Exception:
            pass

        # Load derived/tautological pairs from profile for early rejection.
        from core.dataset_profile import derived_pairs as _dp
        _tautological_pairs = _dp()

        current_unique: List[Hypothesis] = []
        hypothesis_statuses: List[Dict[str, Any]] = []
        for h in hypotheses:
            # --- Tautology check: reject if core X,Y are a derived pair ---
            if _tautological_pairs:
                core_cols = self._extract_core_columns_from_sentence(
                    h.sentence, context.data_schema
                )
                core_set = frozenset(core_cols)
                is_tautological = any(
                    pair.issubset(core_set) for pair in _tautological_pairs
                )
                if is_tautological:
                    print(f"    [SKIP] TAUTOLOGICAL: {h.sentence[:60]}...")
                    rejected_history.append(h.sentence)
                    self.rejection_storage.add_rejection(
                        h.sentence, "TAUTOLOGICAL_DERIVED_PAIR", 1.0
                    )
                    hypothesis_statuses.append(
                        {
                            "hypothesis_id": h.id,
                            "sentence": h.sentence,
                            "status": "skipped",
                            "reason": "TAUTOLOGICAL_DERIVED_PAIR",
                        }
                    )
                    continue

            # Ablation A0: every filter below consults state carried over from earlier
            # rounds (rejection log, semantic PRE-EXISTING index, structural signature
            # index), so all are skipped.  Only the tautology guard above and DSL/column
            # validation downstream remain, which keep hypotheses runnable.
            if self.no_persistence:
                current_unique.append(h)
                hypothesis_statuses.append(
                    {
                        "hypothesis_id": h.id,
                        "sentence": h.sentence,
                        "status": "kept",
                        "note": "A0_NO_PERSISTENCE_FILTERS_BYPASSED",
                    }
                )
                continue

            # --- Cheapest filter first: cross-round exact duplicate ---
            norm = self._normalize_dsl(h.sentence)
            is_cross_round_duplicate = norm in cross_round_rejected

            if is_cross_round_duplicate:
                print(f"    [SKIP] CROSS-ROUND DUPLICATE: {h.sentence[:60]}...")
                rejected_history.append(h.sentence)
                # Don't re-add to rejection_storage (it's already there).
                hypothesis_statuses.append(
                    {
                        "hypothesis_id": h.id,
                        "sentence": h.sentence,
                        "status": "skipped",
                        "reason": "CROSS_ROUND_DUPLICATE",
                    }
                )
                continue

            low_novelty = False
            novelty_score = None
            novelty_reason = None
            if use_precritic_novelty_filter:
                novelty_score, novelty_reason = self.critic.estimate_novelty(
                    h,
                    context,
                    getattr(context, "retrieved_insights", []),
                )
                if novelty_score is not None and novelty_score < novelty_threshold:
                    low_novelty = True

            is_self_duplicate = any(
                self._normalize_dsl(h.sentence) == self._normalize_dsl(r)
                for r in rejected_history
            )

            struct_duplicate = False
            # Use only CORE variables (X, Y, G, Z — before "controlling for") for
            # structural comparison.  Two hypotheses testing the same relationship
            # with different controls are substantively duplicates.
            core_cols = self._extract_core_columns_from_sentence(h.sentence, context.data_schema)
            # Also strip weight columns
            weight_cols_low = {str(w).lower() for w in (config.get("data.weight_columns", []) or [])}
            weight_cols_low.update({"sampling_weight", "pwght", "w_weight"})
            core_cols = [c for c in core_cols if c.lower() not in weight_cols_low]
            if core_cols:
                # Structural duplicate: core-column Jaccard >= 0.80 with a stored insight.
                _struct_thresh = 0.80
                structurally_similar = self.historian.find_structurally_similar_insights(
                    columns=core_cols,
                    threshold=_struct_thresh,
                    skip_sources=["seed_prescan", "statistical_profiling"],
                )
                if structurally_similar:
                    struct_duplicate = True
                    # Carve-out: a strictly deeper test over a superset of an
                    # existing finding's core variables is a new question, not a
                    # duplicate.
                    if self._is_strict_deepening(h, core_cols, structurally_similar):
                        struct_duplicate = False
                        # Flag it so the Critic does not then penalise the very
                        # property that makes this a deepening (reusing the
                        # parent's variables).  See Critic._assess_novelty.
                        h.metadata["is_deepening"] = True
                        print(f"    [DEEPEN] admitted as deepening: {h.sentence[:60]}...")

            similar_insights = []
            if self.enable_preexisting_filter:
                similar_insights = self.historian.find_similar_insights(
                    h.sentence,
                    threshold=self.preexisting_similarity_threshold,
                    max_results=3,
                )
                # Filter out seed sources — controlled versions of seeds
                # are scientifically different questions, not duplicates.
                _seed_sources = {"seed_prescan", "statistical_profiling"}
                similar_insights = [
                    (ins, score) for ins, score in similar_insights
                    if ins.source not in _seed_sources
                ]
                similar_insights = similar_insights[:1]
            preexisting_match = False
            preexisting_score = None
            preexisting_top = None
            if self.enable_preexisting_filter and similar_insights:
                preexisting_top, preexisting_score = similar_insights[0]
                # Structural gate: only treat as PRE-EXISTING when semantic match
                # also has substantial column overlap (for templated hypothesis sentences).
                if core_cols:
                    top_cols = list((getattr(preexisting_top, "metadata", {}) or {}).get("columns", []) or [])
                    if top_cols:
                        overlap = self._jaccard(core_cols, top_cols)
                        # Adaptive: narrow datasets need higher overlap to confirm duplicate
                        _n_schema2 = len(self._schema_columns(context.data_schema or ""))
                        _preex_thresh = 0.67 if _n_schema2 >= 30 else 0.90
                        preexisting_match = overlap >= _preex_thresh
                    else:
                        preexisting_match = True
                else:
                    preexisting_match = True

            # Cross-format DSL dedup: the semantic check above compares the
            # hypothesis DSL string against the readable sentence stored in
            # the graph, which can miss matches.  Also compare the normalized
            # DSL against stored dsl_sentence metadata for an exact-match gate.
            if not preexisting_match and self.enable_preexisting_filter:
                hyp_norm = self._normalize_dsl(h.sentence)
                for existing in self.historian.get_all_insights():
                    # Skip prescan seeds: a controlled version of a seed is a
                    # different scientific question, not a duplicate.
                    if existing.source in ("seed_prescan", "statistical_profiling"):
                        continue
                    stored_dsl = (getattr(existing, "metadata", {}) or {}).get("dsl_sentence", "")
                    if stored_dsl and self._normalize_dsl(stored_dsl) == hyp_norm:
                        preexisting_match = True
                        preexisting_top = existing
                        preexisting_score = 1.0
                        break

            if low_novelty:
                print(
                    f"    [SKIP] LOW-NOVELTY: {h.sentence[:40]}... "
                    f"(Novelty: {novelty_score:.2f})"
                )
                rejected_history.append(h.sentence)
                self.rejection_storage.add_rejection(
                    sentence=h.sentence,
                    reason="LOW_NOVELTY_PRECRITIC",
                    similarity_score=float(novelty_score) if novelty_score is not None else 0.0,
                    round_id=context.round_id,
                )
                hypothesis_statuses.append(
                    {
                        "hypothesis_id": h.id,
                        "sentence": h.sentence,
                        "status": "skipped",
                        "reason": "LOW_NOVELTY_PRECRITIC",
                        "pre_novelty_score": novelty_score,
                        "pre_novelty_reason": novelty_reason,
                    }
                )
            elif preexisting_match:
                top_match, score = preexisting_top, float(preexisting_score or 0.0)
                print(f"    [SKIP] PRE-EXISTING: {h.sentence[:40]}... (Sim: {score:.2f})")
                rejected_history.append(h.sentence)
                self.rejection_storage.add_rejection(
                    sentence=h.sentence,
                    reason="PRE-EXISTING",
                    similarity_score=score,
                    round_id=context.round_id,
                )
                hypothesis_statuses.append(
                    {
                        "hypothesis_id": h.id,
                        "sentence": h.sentence,
                        "status": "skipped",
                        "reason": "PRE-EXISTING",
                        "similarity_score": score,
                        "top_match_insight_id": getattr(top_match, "id", None),
                    }
                )
            elif is_self_duplicate:
                print(f"    [SKIP] SELF-DUPLICATE: {h.sentence[:40]}...")
                rejected_history.append(h.sentence)
                self.rejection_storage.add_rejection(
                    sentence=h.sentence,
                    reason="SELF-DUPLICATE",
                    round_id=context.round_id,
                )
                hypothesis_statuses.append(
                    {
                        "hypothesis_id": h.id,
                        "sentence": h.sentence,
                        "status": "skipped",
                        "reason": "SELF-DUPLICATE",
                    }
                )
            elif struct_duplicate:
                print(
                    f"    [SKIP] STRUCTURAL DUPLICATE: {h.sentence[:40]}... "
                    f"(columns: {core_cols})"
                )
                rejected_history.append(h.sentence)
                self.rejection_storage.add_rejection(
                    sentence=h.sentence,
                    reason="STRUCTURAL_DUPLICATE",
                    round_id=context.round_id,
                )
                hypothesis_statuses.append(
                    {
                        "hypothesis_id": h.id,
                        "sentence": h.sentence,
                        "status": "skipped",
                        "reason": "STRUCTURAL_DUPLICATE",
                        "columns": core_cols,
                    }
                )
            else:
                current_unique.append(h)
                hypothesis_statuses.append(
                    {
                        "hypothesis_id": h.id,
                        "sentence": h.sentence,
                        "status": "kept",
                    }
                )
        return (current_unique, hypothesis_statuses)

    def _process_single_hypothesis(
        self,
        hypothesis: Hypothesis,
        context: RoundContext,
        var_key: Dict[str, str],
        hyp_index: int,
        hyp_total: int,
    ) -> Tuple[Optional[ExperimentResult], Optional[Evaluation], Optional[InsightScore]]:
        """Run experiment, evaluate, and judge a single hypothesis.

        Returns (ExperimentResult, Evaluation, InsightScore) on success,
        or (None, None, None) if the hypothesis fails at any stage.
        Side-effects: writes artifacts, updates rejection/refine storage.
        """
        disp_text = self._sentence_with_variable_labels(hypothesis.sentence, var_key) if var_key else hypothesis.sentence
        print(f"\n  Hypothesis {hyp_index}/{hyp_total}:")
        self._wrap_print(disp_text, indent="    ", width=100)

        # Save initial hypothesis record
        try:
            self.artifacts.save_hypothesis_json(
                context.round_id,
                hypothesis.id,
                "hypothesis.json",
                {
                    "hypothesis_id": hypothesis.id,
                    "sentence": hypothesis.sentence,
                    "context_insight_ids": hypothesis.context_insight_ids,
                    "metadata": hypothesis.metadata,
                },
            )
        except Exception:
            pass

        try:
            # Coder/Runner: implement and run experiment
            print("    Running experiment...")
            result = self.coder_runner.run_experiment(hypothesis)

            # Track column usage for coverage-based diversity guidance
            _hyp_cols = list(hypothesis.metadata.get("columns", []) or [])
            for _c in _hyp_cols:
                self._column_usage_counter[str(_c)] += 1

            # Persist parse/code/execution artifacts (best-effort)
            try:
                parse_call = getattr(self.coder_runner, "last_parse_call", None) or {}
                if parse_call.get("prompt"):
                    self.artifacts.save_hypothesis_text(context.round_id, hypothesis.id, "parse_prompt.txt", parse_call["prompt"])
                if parse_call.get("response"):
                    self.artifacts.save_hypothesis_text(context.round_id, hypothesis.id, "parse_response.txt", parse_call["response"])
                self.artifacts.save_hypothesis_json(
                    context.round_id,
                    hypothesis.id,
                    "test_spec.json",
                    hypothesis.metadata.get("parsed", {}),
                )

                code_gen = getattr(self.coder_runner, "last_code_gen_call", None) or {}
                if code_gen.get("prompt"):
                    self.artifacts.save_hypothesis_text(context.round_id, hypothesis.id, "code_prompt.txt", code_gen["prompt"])
                if code_gen.get("system_prompt"):
                    self.artifacts.save_hypothesis_text(context.round_id, hypothesis.id, "code_system_prompt.txt", code_gen["system_prompt"])
                if code_gen.get("response"):
                    self.artifacts.save_hypothesis_text(context.round_id, hypothesis.id, "code_raw_response.txt", code_gen["response"])

                if getattr(self.artifacts, "save_generated_code", False):
                    exec_info = getattr(self.coder_runner, "last_execution", None) or {}
                    executed_code = exec_info.get("executed_code")
                    if executed_code:
                        self.artifacts.save_hypothesis_text(context.round_id, hypothesis.id, "code_executed.py", executed_code)

                if getattr(self.coder_runner, "last_code_fix_calls", None):
                    self.artifacts.save_hypothesis_json(
                        context.round_id,
                        hypothesis.id,
                        "code_fix_attempts.json",
                        getattr(self.coder_runner, "last_code_fix_calls", []),
                    )

                self.artifacts.save_hypothesis_json(context.round_id, hypothesis.id, "experiment_result.json", result)
            except Exception:
                pass

            # Evaluator: assess statistical validity
            print("    Evaluating results...")
            evaluation = self.evaluator.evaluate(result)
            print(f"    Validity: {evaluation.validity_score:.2f}, p-value: {evaluation.p_value:.4f}")
            try:
                self.artifacts.save_hypothesis_json(context.round_id, hypothesis.id, "evaluation.json", evaluation)
            except Exception:
                pass

            # Critic: judge value and novelty
            # Attach readable sentence so the surprise prompt can show
            # variable labels instead of opaque Q-codes.
            if var_key:
                hypothesis.metadata["readable_sentence"] = self._sentence_with_variable_labels(
                    hypothesis.sentence, var_key
                )
            # Populate card_modules BEFORE the critic classifies edges, so the
            # hypothesis and the stored insights name modules in the same catalog
            # vocabulary; otherwise relation_classifier.insight_modules() compares
            # different namespaces and the module test becomes uninformative.
            if self.variable_retriever is not None and not hypothesis.metadata.get("card_modules"):
                _hyp_cols = hypothesis.metadata.get("columns", [])
                if _hyp_cols:
                    try:
                        _cards = self.variable_retriever.get_cards_by_names(_hyp_cols)
                        _mod_map = {c.var_name: (c.module or "") for c in _cards if c.module}
                        if _mod_map:
                            hypothesis.metadata["card_modules"] = _mod_map
                    except Exception:
                        pass  # Never break judging for optional metadata
            print("    Judging insight...")
            score = self.critic.judge(hypothesis, evaluation, context.retrieved_insights)
            print(f"    Overall score: {score.overall_score:.2f} - Decision: {score.decision}")

            try:
                self.artifacts.save_hypothesis_json(context.round_id, hypothesis.id, "critic_score.json", score)
                surprise = getattr(self.critic, "last_surprise_call", None) or {}
                if surprise.get("prompt"):
                    self.artifacts.save_hypothesis_text(context.round_id, hypothesis.id, "critic_surprise_prompt.txt", surprise["prompt"])
                if surprise.get("response"):
                    self.artifacts.save_hypothesis_text(context.round_id, hypothesis.id, "critic_surprise_response.txt", surprise["response"])
            except Exception:
                pass

            # Store rejections from Critic
            if score.decision == "reject":
                # `similarity_score` is overloaded to store the critic's overall_score.
                self.rejection_storage.add_rejection(
                    sentence=hypothesis.sentence,
                    reason="REJECTED_BY_CRITIC",
                    similarity_score=float(score.overall_score),
                    round_id=context.round_id
                )

            # Store refinements from Critic
            elif score.decision == "refine":
                readable_sentence = self.evaluator.render_finding_sentence(
                    hypothesis_sentence=hypothesis.sentence,
                    result=result,
                    evaluation=evaluation,
                    var_key=var_key or {},
                )
                # Carry forward refinement_count from graduated hypotheses so
                # the cap in pop_top_for_graduation is enforced across cycles.
                # Fresh (non-graduated) hypotheses have no such metadata, so
                # the default of 0 is used.
                prior_refinement_count = int(
                    hypothesis.metadata.get("refinement_count", 0) or 0
                )
                self.refine_storage.add_refinement(
                    sentence=readable_sentence,
                    validity_score=score.validity_score,
                    novelty_score=score.novelty_score,
                    overall_score=score.overall_score,
                    p_value=evaluation.p_value,
                    effect_size=result.effect_size,
                    n_observations=result.n_observations,
                    critique=hypothesis.metadata.get("critic_critique", ""),
                    columns=hypothesis.metadata.get("columns", []),
                    test_type=hypothesis.metadata.get("test_type", "unknown"),
                    round_id=context.round_id,
                    dsl_sentence=hypothesis.sentence,
                    initial_refinement_count=prior_refinement_count,
                )
                print(f"    [INFO] Stored for refinement: {readable_sentence[:60]}...")

            return result, evaluation, score

        except Exception as e:
            import traceback
            try:
                error_trace = traceback.format_exc()
            except Exception:
                # Rare traceback formatting failures (seen with patsy/exceptiongroup interactions).
                error_trace = f"{type(e).__name__}: {e}"
            print(f"    [WARN] SKIPPING Hypothesis {hyp_index} due to error: {type(e).__name__}: {e}")
            # Print first few lines of traceback to identify source
            trace_lines = error_trace.split('\n')[:10]
            for line in trace_lines:
                if 'coder_runner' in line.lower() or '_analyze' in line.lower() or 'format' in line.lower():
                    print(f"      {line.strip()}")
            try:
                hyp_dir = self.artifacts.hypothesis_dir(context.round_id, hypothesis.id)
                self.artifacts.save_exception(hyp_dir, stage="hypothesis_pipeline", exc=e)
                self.artifacts.save_hypothesis_text(context.round_id, hypothesis.id, "traceback.txt", error_trace)
            except Exception:
                pass
            # Track execution failures so the same hypothesis is not regenerated.
            try:
                self.rejection_storage.add_rejection(
                    sentence=hypothesis.sentence,
                    reason="EXECUTION_ERROR",
                    round_id=context.round_id,
                )
            except Exception:
                pass
            return None, None, None

    def run_round(self, mode: Optional[str] = None) -> RoundResult:
        """Run a single discovery round."""
        # Toggle: enable/disable Critic-based pre-novelty filtering
        use_precritic_novelty_filter = True
        novelty_threshold = 0.25
        
        # 1. Orchestrator picks focus
        context = self.orchestrator.start_round(self.goal, mode=mode)

        # 1.5 Two-phase vs one-phase: get schema subset and optionally general idea
        general_idea: Optional[Dict[str, Any]] = None
        idea_query: Optional[str] = None
        allowed_q_ranges: Optional[List[Tuple[int, int]]] = None
        var_key: Dict[str, str] = {}  # variable code -> short label for flow display
        use_two_phase = self.variable_retriever is not None

        # Generate cross-round summaries EARLY so both Phase 1 and Phase 2 can use them.
        rejection_summary = self._generate_rejection_summary(context)
        try:
            if rejection_summary:
                self.artifacts.save_round_note(context.round_id, "rejection_summary.txt", rejection_summary)
        except Exception:
            pass

        insight_graph_summary = self._generate_insight_graph_summary(context)
        try:
            if insight_graph_summary:
                self.artifacts.save_round_note(context.round_id, "insight_graph_summary.txt", insight_graph_summary)
        except Exception:
            pass

        # Refinement leads: generated once, consumed by Phase 1 (thematic) and Phase 2 (filtered).
        refinement_leads_phase1 = self._generate_refinement_leads_for_phase1()

        if use_two_phase:
            # Phase 1: general idea (construct-level only) chosen from the concept menu
            concept_menu = self._variable_catalog_cfg.get("concept_menu") or [
                "Social values, norms, stereotypes (Q1–Q45)",
                "Happiness and wellbeing (Q46–Q56)",
                "Social capital, trust, and organizational membership (Q57–Q105)",
                "Economic values (Q106–Q111)",
                "Perceptions of corruption (Q112–Q120)",
                "Perceptions of migration (Q121–Q130)",
                "Perceptions of security (Q131–Q151)",
                "Postmaterialism / value change (Q152–Q157)",
                "Science and technology perceptions (Q158–Q163)",
                "Religious values (Q164–Q175)",
                "Ethical values (Q176–Q198)",
                "Political interest and participation (Q199–Q234)",
                "Political culture and political regimes (Q235–Q259)",
                "Demographics and SES (Q260–Q290)",
            ]
            print("  [WVS two-phase] Generating general idea (construct-level)...")

            # --- Theme diversity enforcement ---
            # Two mechanisms work together:
            # 1) Cooldown: block themes used >= threshold times in the recent window
            # 2) Forced exploration: every N rounds, require the least-used theme
            from collections import Counter
            avoid_themes: Optional[List[str]] = None
            required_theme: Optional[str] = None
            is_forced_exploration = False
            total_rounds_so_far = self._theme_round_count

            forced_every = int(self._variable_catalog_cfg.get("forced_explore_every", 3))
            if forced_every > 0 and total_rounds_so_far > 0 and total_rounds_so_far % forced_every == 0:
                # Pick the least-explored theme from the concept menu
                theme_counts = Counter(self._theme_history)
                menu_with_counts = [(t, theme_counts.get(t, 0)) for t in concept_menu]
                menu_with_counts.sort(key=lambda x: x[1])
                least_used = menu_with_counts[0]
                if least_used[1] < total_rounds_so_far / len(concept_menu) + 1:
                    required_theme = least_used[0]
                    is_forced_exploration = True
                    print(f"  [WVS two-phase] FORCED EXPLORATION: requiring theme '{required_theme[:50]}' (used {least_used[1]}x)")

            if required_theme is None and self._theme_history:
                recent = self._theme_history[-self._theme_cooldown_window:]
                counts = Counter(recent)
                overused = [t for t, c in counts.items() if c >= self._theme_cooldown_threshold]
                # Also block any theme that has dominated > 40% of all rounds
                for t in set(self._theme_history):
                    if self._theme_history.count(t) > total_rounds_so_far * 0.4 and t not in overused:
                        overused.append(t)
                if overused:
                    avoid_themes = overused
                    print(f"  [WVS two-phase] Theme cooldown: avoiding {[t[:40] for t in overused]}")

            max_phase1_grounding_retries = 2
            phase1_feedback: Optional[str] = None
            for p1_try in range(max_phase1_grounding_retries + 1):
                general_idea = self.generator.generate_general_idea(
                    context, concept_menu,
                    grounding_feedback=phase1_feedback,
                    avoid_themes=avoid_themes,
                    required_theme=required_theme,
                    insight_graph_summary=insight_graph_summary,
                    refinement_leads=refinement_leads_phase1,
                )
                primary_theme = (general_idea.get("primary_theme") or "").strip()
                secondary_theme = (general_idea.get("secondary_theme") or "").strip()
                if not secondary_theme or secondary_theme.upper() in {"NONE", "NULL", "N/A"}:
                    cross_text = " ".join(
                        [
                            str(context.goal or ""),
                            str(general_idea.get("research_question") or ""),
                            str(general_idea.get("core_constructs") or ""),
                            str(general_idea.get("hypothesized_relations") or ""),
                        ]
                    )
                    inferred = self._infer_secondary_theme_from_text(concept_menu, primary_theme, cross_text)
                    if inferred is not None:
                        sid, stheme = inferred
                        general_idea["secondary_theme_id"] = sid
                        general_idea["secondary_theme"] = stheme
                        secondary_theme = stheme
                        print(f"  [WVS two-phase] Auto-selected SECONDARY_THEME fallback: {stheme}")
                # Strip Q-range parentheticals from theme text — "(Q57–Q105)"
                # creates noise TF-IDF tokens like "q57", "q105" that match
                # variable names rather than semantics.
                _strip_qrange = lambda s: re.sub(r"\s*\(Q\d+[–\-]Q?\d+\)", "", s or "").strip()
                pt_clean = _strip_qrange(primary_theme)
                st_clean = _strip_qrange(secondary_theme)

                # Anchors are the most useful for variable retrieval — weight
                # them by including them explicitly in the query.
                anchor_text = " ".join(general_idea.get("anchors") or [])

                if is_forced_exploration:
                    # For forced exploration, weight the RAG query toward the
                    # required theme to prevent the usual trust/wellbeing
                    # variables from dominating retrieval.
                    idea_query = " ".join(filter(None, [
                        pt_clean,
                        pt_clean,  # double-weight the forced theme
                        general_idea.get("research_question") or "",
                        general_idea.get("core_constructs") or "",
                        anchor_text,
                    ])).strip()
                else:
                    idea_query = " ".join(filter(None, [
                        pt_clean,
                        st_clean,
                        general_idea.get("research_question") or "",
                        general_idea.get("core_constructs") or "",
                        general_idea.get("hypothesized_relations") or "",
                        anchor_text,
                    ])).strip()
                allowed_q_ranges = None
                allowed_modules: Optional[List[str]] = None
                resolved_ranges: List[Tuple[int, int]] = []
                resolved_modules: List[str] = []
                for theme in [primary_theme, secondary_theme]:
                    if not theme or theme.upper() in {"NONE", "NULL", "N/A"}:
                        continue
                    # Module-based filtering (works for any dataset)
                    mod = parse_module_from_theme(theme)
                    if mod:
                        resolved_modules.append(mod)
                    # Q-range filtering (WVS-specific, supplementary)
                    qr = parse_q_range_from_theme(theme)
                    if qr is None:
                        for item in concept_menu:
                            if theme.lower() in item.lower() or item.lower() in theme.lower():
                                qr = parse_q_range_from_theme(item)
                                if qr is not None:
                                    break
                    if qr is not None:
                        resolved_ranges.append(qr)
                if resolved_ranges:
                    allowed_q_ranges = sorted(set(resolved_ranges))
                if resolved_modules:
                    allowed_modules = list(dict.fromkeys(resolved_modules))
                selected_cols = self._select_schema_columns_for_round(
                    context, idea_query=idea_query, allowed_q_ranges=allowed_q_ranges,
                    allowed_modules=allowed_modules
                )
                # Weight columns must be in the round schema whenever the Phase-2
                # prompt advertises the `weights=<col>` slot; otherwise the model is told
                # to use a column outside its allowed vocabulary and the validator rejects
                # the result.
                if selected_cols is not None:
                    from core.dataset_profile import weight_columns as _pwc
                    for _w in (_pwc() or []):
                        if _w and _w not in selected_cols:
                            selected_cols.append(_w)
                grounded_ok, grounded_ratio, missing_phrases = self._is_general_idea_grounded(
                    general_idea,
                    selected_cols,
                    idea_query or context.goal,
                )
                if grounded_ok or p1_try >= max_phase1_grounding_retries:
                    break
                print(
                    f"  [WVS two-phase] Regenerating Phase 1 due to weak grounding "
                    f"(ratio={grounded_ratio:.2f}, missing={len(missing_phrases)})."
                )
                phase1_feedback = (
                    "These anchor phrases could not be matched to any available variable: "
                    + "; ".join(missing_phrases[:8])
                    + ". Revise your anchors to use constructs that map to variables in the selected theme range."
                )
            gi_call = getattr(self.generator, "last_general_idea_call", None) or {}
            gi_prompt = gi_call.get("prompt")
            gi_response = gi_call.get("response")
            if gi_prompt:
                print("\n  " + "=" * 56)
                print("  Phase 1 LLM prompt (high-level, full)")
                print("  " + "=" * 56)
                self._wrap_print(gi_prompt, indent="    ", width=140)
            if gi_response:
                print("\n  Phase 1 LLM raw output (full):")
                self._wrap_print(gi_response, indent="    ", width=140)
            primary_theme = (general_idea.get("primary_theme") or "").strip()
            # Record themes for cooldown tracking (both primary and secondary)
            self._theme_round_count += 1
            if primary_theme:
                self._theme_history.append(primary_theme)
            secondary_theme_final = (general_idea.get("secondary_theme") or "").strip()
            if secondary_theme_final and secondary_theme_final.upper() not in {"NONE", "NULL", "N/A"}:
                self._theme_history.append(secondary_theme_final)
            try:
                self.artifacts.save_round_json(context.round_id, "general_idea.json", general_idea)
            except Exception:
                pass

            # --- Save Phase 1 trace (prompt, response, decisions) ---
            try:
                self.artifacts.save_round_json(context.round_id, "phase1_trace.json", {
                    "theme_cooldown": {
                        "avoid_themes": avoid_themes,
                        "required_theme": required_theme,
                        "theme_history": list(self._theme_history),
                        "total_rounds_so_far": total_rounds_so_far,
                    },
                    "grounding": {
                        "grounded_ok": grounded_ok,
                        "grounded_ratio": grounded_ratio,
                        "missing_phrases": missing_phrases,
                        "retries": p1_try,
                    },
                    "llm_prompt": gi_prompt,
                    "llm_response": gi_response,
                    "parsed_idea": general_idea,
                    "idea_query": idea_query,
                    "allowed_q_ranges": allowed_q_ranges,
                })
            except Exception:
                pass

            # --- Flow: Phase 1 (General concept) - complete text for inspection ---
            print("\n  " + "=" * 56)
            print("  Phase 1: General concept chosen (complete, no truncation)")
            print("  " + "=" * 56)
            for key, label in [
                ("primary_theme_id", "PRIMARY_THEME_ID"),
                ("primary_theme", "PRIMARY_THEME"),
                ("secondary_theme_id", "SECONDARY_THEME_ID"),
                ("secondary_theme", "SECONDARY_THEME"),
                ("research_question", "RESEARCH_QUESTION"),
                ("core_constructs", "CORE_CONSTRUCTS"),
                ("hypothesized_relations", "HYPOTHESIZED_RELATIONS"),
                ("candidate_confounders", "CANDIDATE_CONFOUNDERS"),
                ("anchors", "ANCHORS"),
            ]:
                val = general_idea.get(key)
                if (val is None or val == "") and key in {"secondary_theme_id", "secondary_theme"}:
                    val = "NONE"
                elif val is None or val == "":
                    val = "(none)"
                print(f"    {label} (full):")
                self._wrap_print(str(val), indent="      ", width=120)
            if allowed_q_ranges:
                print(f"    RAG Q-range filter (from PRIMARY+SECONDARY themes): {allowed_q_ranges}")
            print()

            selected_cols = selected_cols if selected_cols is not None else self._select_schema_columns_for_round(
                context, idea_query=idea_query, allowed_q_ranges=allowed_q_ranges
            )

            # --- Flow: RAG (variable selection) - complete query and full variable list ---
            print("\n  " + "=" * 56)
            print("  RAG: What was retrieved (complete)")
            print("  " + "=" * 56)
            print("    RAG query (full sentence used for retrieval):")
            self._wrap_print(idea_query, indent="      ", width=120)
            print(f"    Number of variables selected for this round: {len(selected_cols) if selected_cols else 0}")
            if selected_cols:
                print("    Selected variables (complete list, one per line):")
                for c in selected_cols:
                    print(f"      - {c}")
            var_key_full = self._get_variable_key_for_display(
                selected_cols, idea_query or context.goal, max_items=80, max_label_len=None
            )
            if var_key_full:
                print("    Variable meanings (full label/description per variable):")
                for vname, vlabel in var_key_full.items():
                    print(f"      {vname}:")
                    self._wrap_print(vlabel, indent="        ", width=120)
            var_key = self._get_variable_key_for_display(selected_cols, idea_query or context.goal, max_items=80)
        else:
            selected_cols = self._select_schema_columns_for_round(context)

        # Ensure CoderRunner uses the same subset for parse/code prompts.
        try:
            self.coder_runner.active_columns_subset = selected_cols
        except Exception:
            pass

        data_schema = self.coder_runner.get_data_schema(columns_subset=selected_cols)
        if data_schema:
            context.data_schema = data_schema

        # Attach variable glossary (RAG grounding). Use idea_query when two-phase; same Q-range filter as schema selection.
        try:
            if selected_cols is not None and self.variable_retriever is not None:
                _mc = self._variable_catalog_cfg.get("max_prompt_variable_cards")
                try:
                    max_cards = int(_mc) if _mc is not None else 0
                except (TypeError, ValueError):
                    max_cards = 0
                if max_cards <= 0:
                    max_cards = max(len(selected_cols), 500)
                glossary_query = idea_query if (use_two_phase and idea_query) else f"{context.goal} | {context.focus_area} | {context.mode}"
                glossary = self._build_variable_glossary_for_prompt(
                    query=glossary_query,
                    available_cols=selected_cols,
                    max_cards=max_cards,
                    allowed_q_ranges=allowed_q_ranges if use_two_phase else None,
                    allowed_modules=allowed_modules if use_two_phase else None,
                )
                if glossary and context.data_schema:
                    context.data_schema = context.data_schema + "\n\n" + glossary + "\n"
                if use_two_phase and glossary:
                    print("\n    RAG variable descriptions (glossary attached to prompt, complete):")
                    for line in glossary.splitlines():
                        print(f"      {line}")
        except Exception:
            pass

        # Save variable-catalog selection (best-effort)
        try:
            if selected_cols is not None:
                self.artifacts.save_round_json(
                    context.round_id,
                    "schema_column_subset.json",
                    {
                        "enabled": True,
                        "two_phase": use_two_phase,
                        "max_schema_columns": self._variable_catalog_cfg.get("max_schema_columns"),
                        "selected_num_columns": len(selected_cols),
                        "selected_columns": selected_cols,
                        "query": idea_query if (use_two_phase and idea_query) else f"{context.goal} | {context.focus_area} | {context.mode}",
                    },
                )
                if self.variable_retriever is not None:
                    generalized_cards = [
                        c.to_generalized_dict()
                        for c in self.variable_retriever.get_cards_by_names(selected_cols)
                    ]
                    self.artifacts.save_round_json(
                        context.round_id,
                        "generalized_variable_cards.json",
                        generalized_cards,
                    )
        except Exception:
            pass

        # --- Save RAG trace (query, retrieval scores, glossary) ---
        try:
            if use_two_phase and selected_cols is not None and self.variable_retriever is not None:
                # Re-retrieve with scores for tracing
                rag_query = idea_query if idea_query else f"{context.goal} | {context.focus_area}"
                rag_results = self.variable_retriever.retrieve(
                    rag_query, top_k=30,
                    only_vars=set(self.coder_runner.data.columns) if self.coder_runner.data is not None else None,
                    allowed_q_ranges=allowed_q_ranges,
                    allowed_modules=allowed_modules,
                )
                self.artifacts.save_round_json(context.round_id, "rag_trace.json", {
                    "query": rag_query,
                    "allowed_q_ranges": allowed_q_ranges,
                    "allowed_modules": allowed_modules,
                    "max_retrieved_variables": self._variable_catalog_cfg.get("max_retrieved_variables"),
                    "always_keep": list(self._variable_catalog_cfg.get("always_keep", []) or []),
                    "selected_columns": selected_cols,
                    "retrieval_results_top30": [
                        {"var_name": r.get("var_name"), "label": r.get("label", "")[:80], "score": r.get("score", 0)}
                        for r in rag_results[:30]
                    ],
                    "glossary_text": locals().get("glossary"),
                })
        except Exception:
            pass

        # Save round context artifacts
        try:
            self.artifacts.save_round_context(context.round_id, context, data_schema)
        except Exception:
            pass
        
        print(f"\n{'='*60}")
        print(f"Round {context.round_id}: {context.focus_area}")
        print(f"Mode: {context.mode}")
        print(f"{'='*60}\n")
        
        # Debug note about PRE-EXISTING filter status
        if self.enable_preexisting_filter:
            print(
                f"  [INFO] Semantic PRE-EXISTING filter is ENABLED "
                f"(threshold: {self.preexisting_similarity_threshold:.2f}, with structural overlap gate)."
            )
        else:
            print("  [INFO] Semantic PRE-EXISTING filter is DISABLED for this run_round().")
        print("")
        
        # Debug note about pre-critic novelty filter
        if use_precritic_novelty_filter:
            print(f"  [INFO] Pre-Critic novelty filter is ENABLED (threshold: {novelty_threshold}).")
            print("")
        
        # Retry loop for hypothesis generation
        max_gen_retries = 3
        unique_hypotheses = []
        rejected_history = []  # Keep track of what we rejected this round
        original_temperature = self.generator.temperature  # Store original temperature
        
        if use_two_phase and general_idea is not None:
            # Two-phase: the general idea and schema+glossary are ready; the refiner produces variable-level hypotheses
            print("  [WVS two-phase] Refining idea to variable-level hypotheses...")

            # Collect recent critic critiques to guide the refiner toward more surprising hypotheses.
            critic_hints: Optional[str] = None
            try:
                all_ins = self.historian.get_all_insights()
                critiques = []
                for ins in all_ins[-20:]:
                    c = (ins.metadata or {}).get("critic_critique", "")
                    if c and len(c) > 20:
                        critiques.append(c)
                if critiques:
                    critic_hints = (
                        "CRITIC HINTS (from previous rounds — use these to make hypotheses more surprising):\n"
                        + "\n".join(f"- {c}" for c in critiques[-5:])
                    )
            except Exception:
                critic_hints = None

            # Refinement suggestions: only include leads whose columns overlap
            # with the current round's schema so the generator can act on them.
            refine_suggestions = self._generate_refinement_leads_for_phase2(
                schema_columns=selected_cols,
            )
            if refine_suggestions:
                critic_hints = ((critic_hints or "") + "\n\n" + refine_suggestions).strip()

            # Variable cooldown: sliding-window over recent accepted insights.
            # Variables appearing in > max_pct of the window are deprioritized.
            avoid_variables: Optional[List[str]] = None
            if self._variable_history:
                from collections import Counter
                window = self._variable_history[-self._variable_cooldown_window:]
                n_window = len(window)
                if n_window >= 3:
                    var_counts = Counter(v for cols in window for v in cols)
                    overused = [
                        v for v, c in var_counts.items()
                        if c / n_window > self._variable_cooldown_max_pct
                    ]
                    if overused:
                        avoid_variables = overused
                        print(f"  [WVS two-phase] Variable cooldown: over-used vars to deprioritize: {overused}")

            batch_size = self.generator.batch_size
            max_refiner_retries = 3
            attempt = 0
            hypotheses: List[Hypothesis] = []
            invalid_counts: Dict[str, int] = {}
            while attempt < max_refiner_retries and len(hypotheses) < batch_size:
                attempt += 1
                generated = self.generator.refine_idea_to_hypotheses(
                    general_idea, context.data_schema or "", context, batch_size,
                    avoid_variables=avoid_variables,
                    rejected_sentences=rejected_history,
                    rejection_summary=rejection_summary,
                    insight_graph_summary=insight_graph_summary,
                    critic_hints=critic_hints,
                )
                # Enforce variable-role hygiene before novelty checks.
                for h in generated:
                    # Normalise before validation AND before storage, so a
                    # bracket-copied line cannot reach the graph and seed the
                    # format-contamination loop described in
                    # _normalize_dsl_brackets.
                    h.sentence = self._normalize_dsl_brackets(h.sentence)
                    ok, reason = self._validate_two_phase_hypothesis(h.sentence, context.data_schema)
                    if ok:
                        hypotheses.append(h)
                    else:
                        invalid_counts[reason or "UNKNOWN"] = invalid_counts.get(reason or "UNKNOWN", 0) + 1
                        print(f"    [SKIP] INVALID-HYPOTHESIS: {h.sentence[:60]}... ({reason})")
                        # Feed invalid sentences back so the refiner avoids them on retry.
                        rejected_history.append(h.sentence)
                # Deduplicate by exact sentence each pass.
                seen_sent = set()
                deduped: List[Hypothesis] = []
                for h in hypotheses:
                    key = h.sentence.strip().lower()
                    if key in seen_sent:
                        continue
                    seen_sent.add(key)
                    deduped.append(h)
                hypotheses = deduped
                if len(hypotheses) >= batch_size:
                    hypotheses = hypotheses[:batch_size]
                    break
                if attempt < max_refiner_retries:
                    top_reasons = sorted(invalid_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    reason_str = ", ".join(f"{k}={v}" for k, v in top_reasons) if top_reasons else "none"
                    print(
                        f"    [RETRY] Refiner top-up attempt {attempt + 1}/{max_refiner_retries} "
                        f"(kept={len(hypotheses)}/{batch_size}, invalid: {reason_str})"
                    )
            ref_call = getattr(self.generator, "last_refiner_call", None) or {}
            ref_prompt = ref_call.get("prompt")
            ref_resp = ref_call.get("response")
            if ref_prompt:
                print("\n  " + "=" * 56)
                print("  Phase 2 LLM prompt (low-level, full)")
                print("  " + "=" * 56)
                self._wrap_print(ref_prompt, indent="    ", width=140)
            if ref_resp:
                print("\n  Phase 2 LLM raw output (full):")
                self._wrap_print(ref_resp, indent="    ", width=140)

            # --- Save Phase 2 trace (prompt, response, validation, cooldown) ---
            try:
                all_generated_sentences = [h.sentence for h in generated] if locals().get("generated") else []
                self.artifacts.save_round_json(context.round_id, "phase2_trace.json", {
                    "variable_cooldown": {
                        "avoid_variables": avoid_variables,
                        "variable_history_window": list(self._variable_history[-self._variable_cooldown_window:]) if self._variable_history else [],
                    },
                    "llm_prompt": ref_prompt,
                    "llm_response": ref_resp,
                    "refiner_attempts": attempt,
                    "all_generated_sentences": all_generated_sentences,
                    "validation_results": {
                        "kept": [h.sentence for h in hypotheses],
                        "invalid_counts": dict(invalid_counts),
                    },
                    "data_schema_char_count": len(context.data_schema or ""),
                })
            except Exception:
                pass

            hypotheses = self.generator._deduplicate_batch(hypotheses, context)
            hypotheses = self.generator._link_to_context_insights(hypotheses, context)

            # Graduate borderline items from the refinement queue (refinement
            # mode only).  These flow through the same filter -> execute ->
            # score path as freshly generated hypotheses; the point is that
            # graph state and thresholds have likely moved since they were
            # first parked, so a second pass can admit previously-borderline
            # findings without loosening any quality bar.
            hypotheses = self._inject_refinement_candidates(hypotheses, context)

            # --- Flow: Phase 2 (Hypotheses for CoderRunner) - exact sentences, no truncation ---
            print("\n  " + "=" * 56)
            print("  Phase 2: Hypotheses passed to CoderRunner (exact sentences, complete)")
            print("  " + "=" * 56)
            for i, h in enumerate(hypotheses, 1):
                print(f"    Hypothesis {i} (exact sentence used for coding):")
                self._wrap_print(h.sentence, indent="      ", width=120)
                readable = self._sentence_with_variable_labels(h.sentence, var_key)
                if readable != h.sentence:
                    print("      (readable with variable labels):")
                    self._wrap_print(readable, indent="        ", width=120)
            print("")

            print(f"  Checking novelty for {len(hypotheses)} hypotheses (refiner)...")
            unique_hypotheses, hypothesis_statuses = self._filter_hypotheses_novelty_and_duplicates(
                hypotheses,
                context,
                rejected_history,
                use_precritic_novelty_filter=use_precritic_novelty_filter,
                novelty_threshold=novelty_threshold,
            )
            try:
                if hypothesis_statuses:
                    self.artifacts.save_hypothesis_statuses(context.round_id, hypothesis_statuses)
            except Exception:
                pass
            print(f"  Generated {len(hypotheses)} hypotheses (refiner), {len(unique_hypotheses)} passed filters")
            if unique_hypotheses:
                print("  Hypotheses actually sent to CoderRunner (after novelty filter, complete sentences):")
                for i, h in enumerate(unique_hypotheses, 1):
                    print(f"    {i}. ", end="")
                    self._wrap_print(h.sentence, indent="       ", width=120)
        else:
            # One-phase: inject variable coverage guidance into the prompt
            if self._column_usage_counter and context.data_schema:
                coverage_hint = self._build_variable_coverage_hint(context.data_schema)
                if coverage_hint:
                    context.data_schema = context.data_schema + "\n\n" + coverage_hint

            for attempt in range(max_gen_retries):
                print(f"Generating hypotheses (Attempt {attempt + 1}/{max_gen_retries})...")
                hypotheses = self.generator.generate_hypotheses(
                    context,
                    rejected_sentences=rejected_history,
                    rejection_summary=rejection_summary,
                    insight_graph_summary=insight_graph_summary,
                )
                print(f"  Generated {len(hypotheses)} hypotheses")

                # Save generation prompt/response (best-effort)
                try:
                    gen = getattr(self.generator, "last_generation", None) or {}
                    self.artifacts.save_generation_attempt(
                        context.round_id,
                        attempt + 1,
                        gen.get("prompt"),
                        gen.get("system_prompt"),
                        gen.get("temperature"),
                        gen.get("raw_response"),
                        parsed_hypotheses=gen.get("parsed_sentences", []),
                    )
                    if getattr(self.generator, "last_skeptic_edits", None):
                        self.artifacts.save_round_json(
                            context.round_id,
                            f"skeptic_edits_attempt_{attempt+1:02d}.json",
                            getattr(self.generator, "last_skeptic_edits", []),
                        )
                except Exception:
                    pass

                # Filter for Novelty and duplicates
                print(f"  Checking novelty for {len(hypotheses)} hypotheses...")
                current_unique, hypothesis_statuses = self._filter_hypotheses_novelty_and_duplicates(
                    hypotheses,
                    context,
                    rejected_history,
                    use_precritic_novelty_filter=use_precritic_novelty_filter,
                    novelty_threshold=novelty_threshold,
                )

                if current_unique:
                    unique_hypotheses = current_unique
                    print(f"  [OK] Found {len(unique_hypotheses)} unique hypotheses!")
                    try:
                        self.artifacts.save_hypothesis_statuses(context.round_id, hypothesis_statuses)
                    except Exception:
                        pass
                    break  # Valid hypotheses found; exit the retry loop.
                else:
                    print(f"  [WARN] All hypotheses in attempt {attempt+1} were duplicates. Retrying with negative feedback...")
                    # Temporarily boost temperature for the next attempt to force diversity
                    if attempt < max_gen_retries - 1:  # Don't boost on last attempt
                        self.generator.temperature = min(1.0, self.generator.temperature + 0.2)
                        print(f"  [INFO] Increased temperature to {self.generator.temperature:.2f} for next attempt")

        # Reset temperature after loop
        self.generator.temperature = original_temperature
        
        if not unique_hypotheses:
            print("  [ERROR] CRITICAL: Failed to generate novel hypotheses after all retries.")
            round_result = RoundResult(
                round_id=context.round_id,
                hypotheses=[],
                experiment_results=[],
                evaluations=[],
                insight_scores=[],
                accepted_insights=[],
                summary=f"Round {context.round_id} ({context.mode}): Failed to generate novel hypotheses after {max_gen_retries} retries."
            )
            self.round_history.append(round_result)
            return round_result
        
        hypotheses = unique_hypotheses
        print(f"[OK] Proceeding with {len(hypotheses)} unique hypotheses")
        
        # 3. For each hypothesis: test, evaluate, judge
        experiment_results: List[ExperimentResult] = []
        evaluations: List[Evaluation] = []
        insight_scores: List[InsightScore] = []

        for i, hypothesis in enumerate(hypotheses, 1):
            result, evaluation, score = self._process_single_hypothesis(
                hypothesis, context, var_key, hyp_index=i, hyp_total=len(hypotheses),
            )
            if result is not None:
                experiment_results.append(result)
            if evaluation is not None:
                evaluations.append(evaluation)
            if score is not None:
                insight_scores.append(score)
        
        # 4. Historian: store accepted insights
        print("\n  Storing accepted insights...")
        accepted_insights = []
        duplicates_skipped = 0
        storage_records = []
        for score in insight_scores:
            if score.decision == "accept":
                try:
                    hypothesis = next(h for h in hypotheses if h.id == score.hypothesis_id)
                    
                    new_insight_id = f"insight_{uuid.uuid4().hex[:8]}"
                    eval_result = next((e for e in evaluations if e.hypothesis_id == hypothesis.id), None)
                    exp_result = next((r for r in experiment_results if r.hypothesis_id == hypothesis.id), None)
                    readable_sentence = self.evaluator.render_finding_sentence(
                        hypothesis_sentence=hypothesis.sentence,
                        result=exp_result,
                        evaluation=eval_result,
                        var_key=var_key or {},
                    )
                    # Preserve structural metadata from hypothesis (columns, test_type, etc.)
                    insight_metadata = {
                        "round_id": context.round_id,
                        "validity_score": score.validity_score,
                        "novelty_score": score.novelty_score,
                        "overall_score": score.overall_score,
                        "dsl_sentence": hypothesis.sentence,
                        "readable_sentence": readable_sentence,
                        # Preserve structural information for novelty checks
                        "columns": hypothesis.metadata.get("columns", []),
                        "test_type": hypothesis.metadata.get("test_type", "unknown"),
                        # Parsed DSL spec (x / y / group / moderator).  Without it every
                        # core-variable computation downstream (relation_classifier.core_columns(),
                        # the DEEPENS subset test, historian structural dedup) falls back to ALL
                        # columns including controls, so a deepening never looks like a superset
                        # of its base finding.
                        "parsed": hypothesis.metadata.get("parsed", {}),
                        # Provenance: admitted via the deepening carve-out
                        "is_deepening": bool(hypothesis.metadata.get("is_deepening")),
                        # Preserve critic reasoning if available
                        "critic_reasoning": hypothesis.metadata.get("critic_reasoning", ""),
                        "critic_critique": hypothesis.metadata.get("critic_critique", ""),
                        "critic_mechanism": hypothesis.metadata.get("critic_mechanism", ""),
                        "surprise_class": hypothesis.metadata.get("surprise_class", "UNKNOWN"),
                        "llm_surprise_raw": hypothesis.metadata.get("llm_surprise_raw", 5.0),
                        # Provenance: was this insight produced via refinement graduation?
                        "generation_method": hypothesis.metadata.get("generation_method", "llm"),
                    }
                    # Refinement-graduation provenance (only present for graduated insights)
                    if hypothesis.metadata.get("generation_method") == "refinement_graduation":
                        insight_metadata["original_round_id"] = hypothesis.metadata.get("original_round_id")
                        insight_metadata["prior_overall_score"] = hypothesis.metadata.get("prior_overall_score")
                        insight_metadata["refinement_count"] = hypothesis.metadata.get("refinement_count", 1)
                    # Store experiment results for post-hoc analysis
                    if exp_result is not None:
                        insight_metadata["effect_size"] = exp_result.effect_size
                        insight_metadata["p_value"] = exp_result.p_value
                        insight_metadata["n_observations"] = exp_result.n_observations
                    # Populate card_modules for relation classification
                    # (enables EXTENDS/DEEPENS for non-WVS datasets)
                    if self.variable_retriever is not None:
                        _insight_cols = insight_metadata.get("columns", [])
                        if _insight_cols:
                            try:
                                _cards = self.variable_retriever.get_cards_by_names(_insight_cols)
                                _card_mod_map = {c.var_name: (c.module or "") for c in _cards if c.module}
                                if _card_mod_map:
                                    insight_metadata["card_modules"] = _card_mod_map
                            except Exception:
                                pass  # Never break insight creation for optional metadata
                    insight = Insight(
                        id=new_insight_id,
                        sentence=readable_sentence,
                        source="autokd",
                        created_at=datetime.now(),
                        metadata=insight_metadata
                    )
                    
                    # Determine relation types from pairwise analysis
                    # (computed by Critic._find_related_insights and stored
                    #  in hypothesis metadata as _edge_relation_types)
                    edge_types_map = hypothesis.metadata.get("_edge_relation_types", {})
                    relation_types = []
                    for related_id in score.related_insight_ids:
                        if related_id in edge_types_map:
                            relation_types.append(edge_types_map[related_id])
                        else:
                            # Fallback for edge cases
                            relation_types.append(RelationType.NARROWS)
                    
                    # Inject into Discovery Memory
                    returned_insight_id = self.historian.inject_insight(
                        insight,
                        score.related_insight_ids,
                        relation_types
                    )
                    
                    # Check if this was actually added (new) or was a duplicate
                    if returned_insight_id == new_insight_id:
                        accepted_insights.append(insight)
                        # Track variable usage for sliding-window cooldown
                        cols_for_tracking = insight_metadata.get("columns", [])
                        if cols_for_tracking:
                            self._variable_history.append(list(cols_for_tracking))
                        print(f"    [OK] Stored (NEW): {insight.sentence[:60]}...")
                        storage_records.append(
                            {
                                "hypothesis_id": score.hypothesis_id,
                                "stored": True,
                                "insight_id": new_insight_id,
                                "duplicate_of": None,
                            }
                        )
                    else:
                        # Duplicate detected - existing insight ID returned
                        duplicates_skipped += 1
                        existing_insight = self.historian.get_insight(returned_insight_id)
                        if existing_insight:
                            print(f"    [SKIP] DUPLICATE: {insight.sentence[:60]}...")
                            print(f"      (Similar to: {existing_insight.sentence[:60]}...)")
                        else:
                            print(f"    [SKIP] DUPLICATE: {insight.sentence[:60]}...")
                        storage_records.append(
                            {
                                "hypothesis_id": score.hypothesis_id,
                                "stored": False,
                                "insight_id": new_insight_id,
                                "duplicate_of": returned_insight_id,
                            }
                        )
                except Exception as e:
                    print(f"    [WARN] Failed to store insight: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
        
        if duplicates_skipped > 0:
            print(f"\n  Note: {duplicates_skipped} duplicate insight(s) were skipped (similarity threshold: {self.historian.duplicate_threshold})")
        
        # Force save after storing all insights
        try:
            self.insight_graph.save()
        except Exception as e:
            print(f"    [WARN] Failed to save insights: {e}")

        try:
            self.artifacts.save_round_json(context.round_id, "storage_records.json", storage_records)
            self.artifacts.save_round_json(
                context.round_id,
                "accepted_insights.json",
                accepted_insights,
            )
        except Exception:
            pass
        
        # 5. Create round result
        summary = self._generate_round_summary(
            context,
            hypotheses,
            insight_scores,
            accepted_insights
        )
        
        round_result = RoundResult(
            round_id=context.round_id,
            hypotheses=hypotheses,
            experiment_results=experiment_results,
            evaluations=evaluations,
            insight_scores=insight_scores,
            accepted_insights=accepted_insights,
            summary=summary
        )
        
        self.round_history.append(round_result)
        
        total_accepted_by_critic = len([s for s in insight_scores if s.decision == 'accept'])
        total_refined = len([s for s in insight_scores if s.decision == 'refine'])
        print(f"\n  Round Summary:")
        print(f"    Hypotheses: {len(hypotheses)}")
        print(f"    Accepted by Critic: {total_accepted_by_critic}")
        print(f"    Actually Stored (NEW): {len(accepted_insights)}")
        print(f"    Rejected: {len([s for s in insight_scores if s.decision == 'reject'])}")
        print(f"    Stored for Refinement: {total_refined}")
        
        # Show refine storage statistics
        if total_refined > 0:
            refine_stats = self.refine_storage.get_statistics()
            print(f"    Total refinement candidates in storage: {refine_stats['total']}")
        
        # Print knowledge tree visualization
        self._print_knowledge_tree()
        
        return round_result
    
    def _generate_round_summary(
        self,
        context: RoundContext,
        hypotheses: List[Hypothesis],
        insight_scores: List[InsightScore],
        accepted_insights: List[Insight]
    ) -> str:
        """Generate a summary of the round."""
        parts = [
            f"Round {context.round_id} ({context.mode}):",
            f"Focus: {context.focus_area}",
            f"Generated {len(hypotheses)} hypotheses",
            f"Accepted {len(accepted_insights)} insights",
        ]
        
        if accepted_insights:
            avg_score = sum(s.overall_score for s in insight_scores if s.decision == "accept") / len(accepted_insights)
            parts.append(f"Average score: {avg_score:.2f}")
        
        return " | ".join(parts)
    
    def _print_knowledge_tree(self):
        """Print a simple text-based visualization of the knowledge graph."""
        all_insights = self.historian.get_all_insights()
        
        if not all_insights:
            return
        
        print(f"\nCURRENT KNOWLEDGE TREE (showing last 8 insights):")
        print("   " + "-" * 70)
        
        # Show the most recent insights
        recent_insights = all_insights[-8:] if len(all_insights) > 8 else all_insights
        
        for i, insight in enumerate(recent_insights, 1):
            overall_score = insight.metadata.get("overall_score", 0.0)
            source = insight.source
            is_obvious = insight.metadata.get("is_obvious", False)
            
            source_indicator = ""
            if source == "human_seed":
                source_indicator = " [SEED]"
            elif source == "statistical_profiling":
                source_indicator = " [STATS]"
            
            score_str = f" (score: {overall_score:.2f})" if overall_score > 0 else ""
            
            sentence = insight.sentence[:65] + "..." if len(insight.sentence) > 65 else insight.sentence
            
            print(f"   {i}. [{insight.id[:8]}]{source_indicator} {sentence}{score_str}")
        
        stats = self.insight_graph.get_statistics()
        print(
            f"\n   Graph Stats: {stats['num_insights']} insights, {stats['num_relations']} relations, avg degree: {stats['avg_degree']:.2f}"
        )
    
    def run(self, num_rounds: Optional[int] = None):
        """Run multiple discovery rounds.

        Includes early-stopping: if the last *stall_window* consecutive
        rounds produce zero new accepted insights, the session ends early
        to avoid wasting compute on diminishing returns.
        """
        if num_rounds is None:
            num_rounds = config.get("orchestrator.max_rounds", 10)

        stall_window = int(config.get("orchestrator.early_stop_stall_rounds", 10))
        consecutive_empty = 0

        print(f"\n{'='*60}")
        print(f"Starting AutoKD with goal: {self.goal}")
        print(f"Planning to run up to {num_rounds} rounds (early-stop after {stall_window} empty rounds)")
        print(f"{'='*60}\n")

        try:
            for i in range(num_rounds):
                # A transient agent/LLM failure costs one round, not the whole session.
                # It is treated as an empty round so it still counts toward the stall window.
                try:
                    round_result = self.run_round()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self._failed_rounds = getattr(self, "_failed_rounds", 0) + 1
                    print(f"\n[WARN] Round {i+1} failed ({type(e).__name__}: {e}); "
                          f"skipping. Total failed rounds: {self._failed_rounds}")
                    consecutive_empty += 1
                    if consecutive_empty >= stall_window:
                        print(
                            f"\n[EARLY STOP] {consecutive_empty} consecutive rounds "
                            f"with no new insights. Stopping to avoid diminishing returns."
                        )
                        break
                    if i < num_rounds - 1:
                        print("\n" + "-"*60 + "\n")
                    continue

                if round_result.accepted_insights:
                    consecutive_empty = 0
                else:
                    consecutive_empty += 1
                    if consecutive_empty >= stall_window:
                        print(
                            f"\n[EARLY STOP] {consecutive_empty} consecutive rounds "
                            f"with no new insights. Stopping to avoid diminishing returns."
                        )
                        break

                # Brief pause between rounds
                if i < num_rounds - 1:
                    print("\n" + "-"*60 + "\n")

        except StopIteration:
            print("\nMaximum rounds reached.")
        
        # Final summary
        print(f"\n{'='*60}")
        print("Discovery Session Complete")
        print(f"{'='*60}")
        print(f"Total rounds: {len(self.round_history)}")
        
        total_insights = sum(len(r.accepted_insights) for r in self.round_history)
        print(f"Total insights discovered: {total_insights}")
        
        stats = self.insight_graph.get_statistics()
        print(f"Discovery Memory stats:")
        print(f"  Total insights: {stats['num_insights']}")
        print(f"  Total relations: {stats['num_relations']}")
        print(f"  Average degree: {stats['avg_degree']:.2f}")

        if getattr(self, "no_persistence", False):
            # Ablation A0 summary.
            self._print_a0_summary()

        print(f"{'='*60}\n")

    def _print_a0_summary(self) -> None:
        """Rediscovery accounting for the no-persistence arm.

        Every other arm collapses a re-proposed finding into the existing node,
        so these quantities are unobservable there.  A0 stores and counts them,
        which is the whole point of the arm.
        """
        from collections import Counter

        ins = [
            i for i in self.historian.get_all_insights()
            if i.source not in ("statistical_profiling", "seed_prescan")
        ]
        n = len(ins)
        if not n:
            return

        def parsed(i):
            return (i.metadata or {}).get("parsed") or {}

        pairs, ests, dsl = set(), set(), Counter()
        for i in ins:
            p = parsed(i)
            x, y = p.get("x"), p.get("y")
            if x and y:
                pairs.add(frozenset((x, y)))
                ests.add((
                    (i.metadata or {}).get("test_type", ""),
                    frozenset((x, y)),
                    p.get("moderator") or None,
                    p.get("group") or None,
                ))
            d = (i.metadata or {}).get("dsl_sentence") or i.sentence
            dsl[str(d).strip()] += 1

        redisc = getattr(self.historian, "rediscovery_count", 0)
        exact_dupes = sum(c - 1 for c in dsl.values() if c > 1)
        gaps = [
            (i.metadata or {}).get("rediscovery_round_gap")
            for i in ins if (i.metadata or {}).get("rediscovery_of")
        ]
        gaps = [g for g in gaps if isinstance(g, int)]

        print("\n  A0 (no persistence) — rediscovery accounting:")
        print(f"    insights stored                 : {n}")
        print(f"    distinct pairs / insight        : {len(pairs) / n:.4f}")
        print(f"    distinct estimands / insight    : {len(ests) / n:.4f}")
        print(f"    exact-duplicate DSL rate        : {exact_dupes / n:.1%} "
              f"({exact_dupes} repeats over {len(dsl)} distinct strings)")
        print(f"    flagged rediscoveries           : {redisc} "
              f"({redisc / n:.1%} of stored)")
        if gaps:
            print(f"    median rounds between re-finds  : "
                  f"{sorted(gaps)[len(gaps) // 2]}")
    
    def _build_variable_coverage_hint(self, data_schema: str) -> Optional[str]:
        """Build a prompt hint showing which columns are over/under-explored."""
        if not self._column_usage_counter:
            return None

        from core.dataset_profile import id_columns as _profile_ids, control_only_columns as _profile_co
        _skip = set(c.lower() for c in (_profile_ids() or []))
        _skip |= set(c.lower() for c in (_profile_co() or []))
        _skip |= {"review_text", "summary", "product_title", "date", "timestamp"}

        schema_cols = self._schema_columns(data_schema)
        substantive = [c for c in schema_cols if c.lower() not in _skip]
        if len(substantive) < 4:
            return None

        counts = {c: self._column_usage_counter.get(c, 0) for c in substantive}
        total = sum(counts.values()) or 1
        avg = total / len(counts) if counts else 0

        over = sorted([c for c, n in counts.items() if n > avg * 1.5], key=lambda c: -counts[c])
        under = sorted([c for c, n in counts.items() if n < avg * 0.5], key=lambda c: counts[c])
        unused = [c for c, n in counts.items() if n == 0]

        lines = ["VARIABLE EXPLORATION STATUS (use this to diversify your hypotheses):"]
        if unused:
            lines.append(f"  NEVER EXPLORED (strongly prefer these): {', '.join(unused[:8])}")
        if under:
            under_only = [c for c in under if c not in unused][:6]
            if under_only:
                lines.append(f"  UNDER-EXPLORED: {', '.join(under_only)}")
        if over:
            lines.append(f"  OVER-EXPLORED (avoid as primary pair): {', '.join(over[:6])}")

        return "\n".join(lines) if len(lines) > 1 else None

    def _generate_rejection_summary(self, context: RoundContext) -> Optional[str]:
        """Generate a summary of cross-round rejection history using LLM.
        
        Returns:
            Summary string if rejections exist, None otherwise.
        """
        recent_rejections = self.rejection_storage.get_recent_rejections(max_count=100)
        
        if not recent_rejections:
            return None
        
        # Group rejections by reason
        by_reason = {}
        for r in recent_rejections:
            if r.reason not in by_reason:
                by_reason[r.reason] = []
            by_reason[r.reason].append(r.sentence)
        
        # Build rejection list for LLM
        rejection_text = []
        for reason, sentences in by_reason.items():
            rejection_text.append(f"\n{reason} ({len(sentences)} rejections):")
            # Show up to 5 examples per reason
            for i, sent in enumerate(sentences[:5], 1):
                rejection_text.append(f"  {i}. {sent}")
            if len(sentences) > 5:
                rejection_text.append(f"  ... and {len(sentences) - 5} more")
        
        rejection_list = "\n".join(rejection_text)
        
        # Generate summary using LLM
        summary_prompt = f"""Summarize the following rejected hypotheses to help avoid generating similar ones in the future.

REJECTED HYPOTHESES (from previous rounds):
{rejection_list}

Your task:
1. Identify common patterns in these rejections (e.g., "many hypotheses about rating and helpful_votes", "repeated attempts to test warranty keywords")
2. Identify what types of hypotheses have been over-explored
3. Suggest what NEW directions to explore instead

Format your response as:
- PATTERNS: [list 2-3 common patterns]
- OVER-EXPLORED: [list 2-3 areas that have been tried too many times]
- NEW DIRECTIONS: [suggest 2-3 new directions to explore]

Keep it concise (3-5 sentences total). Focus on actionable guidance for generating novel hypotheses."""

        try:
            if not self.llm.is_available():
                # Fallback: simple text summary if LLM unavailable
                stats = self.rejection_storage.get_statistics()
                return f"Note: {stats['total']} hypotheses were rejected in previous rounds. Avoid repeating similar patterns."
            
            summary = self.llm.generate(
                prompt=summary_prompt,
                temperature=0.3,  # Low temperature for consistent summarization
                system="You are a research assistant helping to avoid redundant hypothesis generation."
            )
            
            # Extract summary (remove markdown if present)
            summary = summary.strip()
            if summary.startswith("```"):
                lines = summary.split('\n')
                summary = '\n'.join([l for l in lines if not l.strip().startswith('```')])
            
            return summary.strip()
        except Exception as e:
            print(f"    [WARN] Failed to generate rejection summary: {e}")
            # Fallback: simple text summary
            stats = self.rejection_storage.get_statistics()
            return f"Note: {stats['total']} hypotheses were rejected in previous rounds. Avoid repeating similar patterns."

    def _compute_graph_frontiers(self, all_insights: List[Insight]) -> str:
        """Deterministic summary of graph structure: coverage, gaps, and depth.

        Uses insight column metadata to find:
        1. Which modules have insights (covered) and which are over-explored
        2. Which module PAIRS have cross-module insights connecting them
        3. Which module pairs are UNEXPLORED FRONTIERS (both covered, no link)
        4. Test-type depth balance (assoc vs interact/heterogeneity)

        Returns a compact text block suitable for both Phase 1 and Phase 2.
        """
        if len(all_insights) < 3:
            return f"{len(all_insights)} insights stored so far. Explore broadly."

        _MODULE_RANGES = [
            ((1, 45), "Social values/norms"),
            ((46, 56), "Happiness/wellbeing"),
            ((57, 105), "Trust/social capital"),
            ((106, 111), "Economic values"),
            ((112, 120), "Corruption"),
            ((121, 130), "Migration"),
            ((131, 151), "Security"),
            ((152, 157), "Postmaterialism"),
            ((158, 163), "Science/technology"),
            ((164, 175), "Religious values"),
            ((176, 198), "Ethical values"),
            ((199, 234), "Political participation"),
            ((235, 259), "Political culture"),
            ((260, 290), "Demographics"),
        ]

        _SKIP_MODULES = {"identifiers", "demographics", "demographics and ses"}

        def _col_to_module(col: str) -> Optional[str]:
            # First try: variable catalog (works for any dataset)
            if self.variable_retriever is not None:
                try:
                    cards = self.variable_retriever.get_cards_by_names([col])
                    if cards:
                        mod = cards[0].section_theme or cards[0].module
                        if mod and mod.lower() not in _SKIP_MODULES:
                            return mod
                        return None
                except Exception:
                    pass  # fall through to Q-range lookup
            # Fallback: WVS Q-number ranges
            qn = self._q_number(col)
            if qn is None:
                return None
            for (lo, hi), name in _MODULE_RANGES:
                if lo <= qn <= hi:
                    return name
            return None

        from collections import Counter
        module_counts: Counter = Counter()
        cross_module_pairs: Counter = Counter()

        for ins in all_insights:
            cols = (ins.metadata or {}).get("columns", [])
            modules = set()
            for c in cols:
                m = _col_to_module(c)
                if m and m != "Demographics":
                    modules.add(m)
            for m in modules:
                module_counts[m] += 1
            mods = sorted(modules)
            for i in range(len(mods)):
                for j in range(i + 1, len(mods)):
                    cross_module_pairs[(mods[i], mods[j])] += 1

        if not module_counts:
            return ""

        covered = [m for m, c in module_counts.most_common() if c >= 2]
        frontiers = []
        for i in range(len(covered)):
            for j in range(i + 1, len(covered)):
                pair = tuple(sorted((covered[i], covered[j])))
                if cross_module_pairs.get(pair, 0) == 0:
                    frontiers.append(pair)

        lines = []
        covered_str = ", ".join(f"{m} ({c})" for m, c in module_counts.most_common())
        lines.append(f"- COVERED MODULES: {covered_str}")

        # Over-explored: modules with disproportionately many insights
        n_total = len(all_insights)
        over_explored = [m for m, c in module_counts.most_common()
                         if c >= 5 and c > n_total * 0.25]
        if over_explored:
            lines.append(f"- OVER-EXPLORED: {', '.join(over_explored)} — avoid unless deepening with interact/heterogeneity")

        if cross_module_pairs:
            top_links = cross_module_pairs.most_common(5)
            links_str = ", ".join(f"{a}↔{b} ({c})" for (a, b), c in top_links)
            lines.append(f"- STRONGEST CROSS-MODULE LINKS: {links_str}")

        if frontiers:
            frontier_str = ", ".join(f"{a}↔{b}" for a, b in frontiers[:8])
            lines.append(f"- UNEXPLORED FRONTIERS (both modules have insights but NO cross-module insight connects them): {frontier_str}")
            lines.append("  Target these frontiers for novel cross-cutting discoveries.")

        # Variable-level CORE usage: count how often each variable appears
        # as a main predictor/outcome (before "controlling for"), not as a
        # control or weight.  This tells the LLM which substantive variables
        # are under-explored so it naturally gravitates toward them.
        var_usage: Counter = Counter()
        for ins in all_insights:
            dsl = (ins.metadata or {}).get("dsl_sentence", "") or ins.sentence
            core = dsl.lower().split(" controlling for")[0].split(" adjusting for")[0]
            core = core.split("weights=")[0]  # strip weight clause
            for c in (ins.metadata or {}).get("columns", []):
                if c.lower() in core:
                    var_usage[c] += 1
        if var_usage and len(var_usage) > 6:
            most_used = var_usage.most_common(5)
            least_used = var_usage.most_common()[:-6:-1]  # bottom 5
            lines.append(
                f"- OVER-USED VARIABLES: "
                + ", ".join(f"{v} ({c}x)" for v, c in most_used)
            )
            lines.append(
                f"- UNDER-EXPLORED VARIABLES (use these): "
                + ", ".join(f"{v} ({c}x)" for v, c in least_used)
            )

        # Depth frontiers: insights that are only simple assoc but could be
        # deepened to interact/heterogeneity.
        type_counts = Counter()
        assoc_only_cols = []
        for ins in all_insights:
            meta = ins.metadata or {}
            tt = meta.get("test_type", "")
            if tt:
                type_counts[tt] += 1
            if tt == "assoc":
                cols = meta.get("columns", [])
                q_cols = [c for c in cols if self._q_number(c) is not None]
                if len(q_cols) >= 2:
                    assoc_only_cols.append((q_cols[0], q_cols[1], ins.sentence[:60]))

        n_assoc = type_counts.get("assoc", 0)
        n_complex = type_counts.get("interact", 0) + type_counts.get("heterogeneity", 0)
        if n_assoc > 0:
            pct_complex = n_complex / (n_assoc + n_complex) * 100
            lines.append(f"- DEPTH ANALYSIS: {n_assoc} assoc, {n_complex} interact/heterogeneity ({pct_complex:.0f}% complex)")
            if pct_complex < 40 and assoc_only_cols:
                depth_examples = assoc_only_cols[:4]
                depth_str = "; ".join(f"{a}↔{b}" for a, b, _ in depth_examples)
                lines.append(f"- DEPTH FRONTIERS: These assoc findings could be deepened with interact() or heterogeneity(): {depth_str}")
                lines.append("  Try: interact(X * MODERATOR -> Y) or heterogeneity(Y ~ X | SUBGROUP) using these variable pairs.")

        return "\n".join(lines)

    def _get_top_refinement_entries(self, max_count: int = 5):
        """Return top refinement candidates that have non-trivial critiques."""
        try:
            entries = self.refine_storage.refinements
        except Exception:
            return []
        if not entries:
            return []
        scored = sorted(entries, key=lambda e: e.overall_score, reverse=True)
        return [e for e in scored if e.critique and len(e.critique) > 10][:max_count]

    def _inject_refinement_candidates(
        self,
        hypotheses: List[Hypothesis],
        context: RoundContext,
        max_count: int = 3,
        min_overall_score: float = 0.50,
    ) -> List[Hypothesis]:
        """Pop top refinement candidates and reintroduce them as fresh Hypotheses.

        Only runs when the orchestrator picked ``refinement`` mode.  The
        popped records are removed from refine_storage, reconstructed as
        ``Hypothesis`` objects using their stored DSL sentence, and prepended
        to the round's candidate list.  They then flow through the normal
        filter -> execute -> evaluate -> critic pipeline against the current
        graph state.  If they still fail (decision="refine"), the critic's
        add_refinement call re-inserts them with refinement_count carried
        forward via initial_refinement_count, so items are capped at
        ``max_refinement_count`` retries to avoid thrashing.

        The point of this mechanism is to drain the refinement queue: without
        it, borderline items [overall in 0.45-0.55] accumulate and are never
        re-evaluated even after the graph has grown or thresholds have moved.
        """
        if context.mode != "refinement":
            return hypotheses
        try:
            popped = self.refine_storage.pop_top_for_graduation(
                max_count=max_count,
                min_overall_score=min_overall_score,
            )
        except Exception as e:
            print(f"    [WARN] refinement graduation failed: {e}")
            return hypotheses
        if not popped:
            return hypotheses

        # Build a normalized-DSL set of the existing batch so we don't
        # double-up if the generator just happened to produce the same
        # sentence as one of the graduated candidates this round.
        existing_norm: set = set()
        for h in hypotheses:
            try:
                existing_norm.add(self._normalize_dsl(h.sentence))
            except Exception:
                pass

        injected: List[Hypothesis] = []
        skipped_dup = 0
        for record in popped:
            dsl = record.dsl_sentence
            if not dsl:
                continue
            try:
                norm = self._normalize_dsl(dsl)
            except Exception:
                norm = dsl.strip().lower()
            if norm in existing_norm:
                skipped_dup += 1
                continue
            # _validate_two_phase_hypothesis is deliberately not called here: it checks
            # against the current round's RAG-selected schema, not the dataset, and would
            # reject graduated candidates whose variables were not selected this round.
            # The DSL parser still catches syntax errors at execution time, and the
            # structural / cross-round dedup filters catch stale items.
            hyp = Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                sentence=dsl,
                context_insight_ids=list(context.anchor_insight_ids[:3]),
                metadata={
                    "generation_method": "refinement_graduation",
                    "original_round_id": record.round_id,
                    "prior_overall_score": record.overall_score,
                    "prior_validity_score": record.validity_score,
                    "prior_novelty_score": record.novelty_score,
                    "refinement_count": record.refinement_count + 1,
                    "test_type": record.test_type or "unknown",
                    "columns": record.columns or [],
                },
            )
            injected.append(hyp)
            existing_norm.add(norm)

        if injected or skipped_dup:
            print(
                f"  [REFINE-GRADUATION] popped={len(popped)} "
                f"injected={len(injected)} skip_dup={skipped_dup} "
                f"queue_now={len(self.refine_storage.refinements)}"
            )
            for i, h in enumerate(injected, 1):
                prior = h.metadata.get("prior_overall_score", 0.0)
                print(f"    {i}. [prior overall={prior:.2f}] {h.sentence[:100]}")
        return list(injected) + list(hypotheses)

    def _generate_refinement_leads_for_phase1(self) -> Optional[str]:
        """Build thematic refinement leads for Phase 1 (no variable codes).

        Phase 1 picks themes, so these leads tell it which thematic areas
        have promising borderline findings worth revisiting.
        """
        top = self._get_top_refinement_entries()
        if not top:
            return None

        lines = ["PROMISING LEADS (borderline findings worth revisiting — consider picking themes that let you deepen these):"]
        for i, e in enumerate(top, 1):
            lines.append(f"  {i}. {e.sentence[:120]}")
            lines.append(f"     Suggestion: {e.critique[:120]}")
        return "\n".join(lines)

    def _generate_refinement_leads_for_phase2(
        self, schema_columns: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Build variable-level refinement leads for Phase 2, filtered to current schema.

        Only includes leads whose columns overlap with the current round's
        available variables, so the generator can actually act on them.
        """
        top = self._get_top_refinement_entries()
        if not top:
            return None

        if schema_columns:
            schema_set = set(schema_columns)
            # Keep entries where at least half of their columns are in the schema
            relevant = []
            for e in top:
                if e.columns:
                    overlap = len(set(e.columns) & schema_set)
                    if overlap >= max(1, len(e.columns) // 2):
                        relevant.append(e)
            top = relevant

        if not top:
            return None

        lines = ["REFINEMENT LEADS (promising but borderline — try to deepen these):"]
        for i, e in enumerate(top, 1):
            tt = e.test_type or "unknown"
            lines.append(
                f"  {i}. [{tt}] {e.sentence[:100]}"
            )
            lines.append(
                f"     Suggestion: {e.critique[:120]}"
            )
        return "\n".join(lines)

    def _generate_insight_graph_summary(self, context: RoundContext) -> Optional[str]:
        """Deterministic summary of what is already stored in the insight graph.

        Covers module coverage, cross-module links, unexplored frontiers,
        over-explored areas, and test-type depth — all computed from metadata
        without an LLM call.
        """
        try:
            all_insights = self.historian.get_all_insights()
        except Exception:
            all_insights = []
        if not all_insights:
            return None

        return self._compute_graph_frontiers(all_insights) or None