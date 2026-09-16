"""Coder/Runner Agent - implements and executes experiments."""

import uuid
import pandas as pd
import numpy as np
import json
import re
import signal
import importlib
import warnings
from typing import Optional, Dict, Any
try:
    from pandas.errors import SettingWithCopyWarning
except ImportError:
    # Removed in pandas 2.0+; define a placeholder so filterwarnings calls are harmless
    SettingWithCopyWarning = FutureWarning
from core.types import Hypothesis, ExperimentResult, ResultStatus
from agents.base_agent import BaseAgent
from core.llm_client import create_llm_client, OllamaClient
from core.config import config
from core.dsl import (
    is_dsl_hypothesis,
    parse_dsl_hypothesis,
    execute_spec,
    replicate_spec,
    build_label_alias_map,
    spec_to_python_code,
    DSLParseError,
    DSLExecutionError,
    WVSTestSpec,
)
from core.llm_parse import extract_json, extract_code_block


class CoderRunner(BaseAgent):
    """Coder/Runner agent that implements and executes experiments."""
    
    def __init__(self, data_path: Optional[str] = None):
        super().__init__("coder_runner")
        self.data_path = data_path
        self.data: Optional[pd.DataFrame] = None
        # Active per-round prompt budget gate (set by DiscoveryEngine).
        # When set, schema/profile prompts will only describe these columns.
        self.active_columns_subset: Optional[list[str]] = None
        
        self.max_retries = self.get_config("max_retries", 3)
        self.retry_on_failure = self.get_config("retry_on_failure", True)
        self.execution_timeout = self.get_config("execution_timeout", 30)  # Default 30 seconds
        
        # Initialize LLM client (prefer code model if available)
        llm_config = config.get_llm_config()
        code_model = llm_config.get("code_model", "codellama")
        main_model = llm_config.get("model", "llama3")
        
        print(
            f"[INFO] CoderRunner LLM Config: Attempting to use code model '{code_model}' (main model: '{main_model}')"
        )
        
        # Try to use code model, but fallback gracefully
        code_config = llm_config.copy()
        code_config["model"] = code_model
        test_client = create_llm_client(code_config)
        
        # Check if code model is actually available
        if isinstance(test_client, OllamaClient) and test_client.is_available():
            self.llm = test_client
            actual_model = test_client.model if hasattr(test_client, 'model') else code_model
            print(f"[OK] Using code model: {actual_model}")
        else:
            # Fallback to main model
            self.llm = create_llm_client(llm_config)
            actual_model = self.llm.model if isinstance(self.llm, OllamaClient) and hasattr(self.llm, 'model') else main_model
            print(f"[WARN] Code model '{code_model}' not available, falling back to main model: {actual_model}")
        
        if data_path:
            try:
                self.data = pd.read_csv(data_path, low_memory=False)
                # Datasets larger than 500k rows are subsampled once (seed 0)
                MAX_ROWS = 500_000
                if len(self.data) > MAX_ROWS:
                    print(f"  Sampling {MAX_ROWS:,} rows from {len(self.data):,} total rows for faster processing...")
                    self.data = self.data.sample(n=MAX_ROWS, random_state=0).reset_index(drop=True)
                # Replace inf/-inf with NaN so statsmodels regressions don't crash
                n_inf = self.data.isin([np.inf, -np.inf]).sum().sum()
                if n_inf > 0:
                    self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
                    print(f"  [INFO] Replaced {n_inf:,} inf values with NaN")
                print(f"[OK] Loaded data from {data_path}: {len(self.data)} rows")
            except Exception as e:
                raise RuntimeError(
                    f"Could not load data from {data_path}: {e}. "
                    f"Please ensure the file exists and is a valid CSV."
                ) from e

        # DSL label alias map (built lazily on first DSL hypothesis)
        self._label_aliases: Optional[Dict[str, str]] = None

        # Last-call artifacts (best-effort, for run artifact logging)
        self.last_parse_call: Optional[Dict[str, Any]] = None
        self.last_code_gen_call: Optional[Dict[str, Any]] = None
        self.last_code_fix_calls: list[Dict[str, Any]] = []
        self.last_execution: Optional[Dict[str, Any]] = None
    
    def run_experiment(self, hypothesis: Hypothesis) -> ExperimentResult:
        """Run an experiment to test a hypothesis with retry logic.

        Routing logic:
          - If the hypothesis sentence matches DSL format (assoc/diff/interact/
            heterogeneity), use the deterministic WVS DSL path: no LLM re-parsing,
            canonical statistical compiler, strict result validation.
          - Otherwise, fall through to the existing LLM-based path (Amazon datasets).

        Raises:
            RuntimeError: If LLM is unavailable or data is missing
            ValueError: If parsing, code generation, or execution fails after all retries
        """
        # ── DSL path (deterministic, no LLM) ──
        if is_dsl_hypothesis(hypothesis.sentence):
            return self._run_dsl_experiment(hypothesis)

        # ── Legacy LLM-based path (free-form hypotheses) ──
        return self._run_llm_experiment(hypothesis)

    # ------------------------------------------------------------------
    # Pre-flight validation (data-driven, dataset-agnostic)
    # ------------------------------------------------------------------

    def _validate_spec_feasibility(self, spec) -> None:
        """Check that a DSL spec can execute before burning retry attempts.

        Catches common runtime crashes: high-cardinality groups, single-level
        groups after cleaning, and insufficient data.  All checks use the
        actual loaded data, so they work for any dataset.
        """
        from core.dsl import _resolve_known_categorical, _MAX_CATEGORICAL_CARDINALITY

        # 1. High-cardinality guard for columns wrapped with C() in formulas.
        #    - diff/heterogeneity: group is ALWAYS wrapped → check cardinality
        #    - interact: moderator is only C() if in known_categorical;
        #      otherwise it's used as continuous numeric → skip check
        _known_cat = _resolve_known_categorical()
        cols_to_check = []
        if spec.group and spec.family in ("diff", "heterogeneity"):
            cols_to_check.append((spec.group, "group"))
        mod = getattr(spec, "moderator", None)
        if mod and spec.family == "interact" and mod in _known_cat:
            cols_to_check.append((mod, "moderator"))

        for col, role in cols_to_check:
            if col not in self.data.columns:
                continue
            card = int(self.data[col].nunique(dropna=True))
            if card > _MAX_CATEGORICAL_CARDINALITY:
                raise ValueError(
                    f"Column '{col}' ({role}) has {card} unique values, "
                    f"exceeding the safe limit of {_MAX_CATEGORICAL_CARDINALITY} "
                    f"for categorical regression. Use it as a continuous "
                    f"predictor instead."
                )

        # 2. Group singleton guard (after simulated cleaning).
        #    Raw data may show 2+ levels, but dropna/coercion can collapse
        #    to 1.  Simulate the cleaning that execute_spec() will do.
        if spec.group and spec.family in ("diff", "heterogeneity"):
            col = spec.group
            if col in self.data.columns:
                s = self.data[col].copy()
                # Mimic _prepare_data: coerce non-categorical to numeric
                if col not in _known_cat:
                    num = pd.to_numeric(s, errors="coerce")
                    if float(num.notna().mean()) >= 0.5:
                        s = num
                n_levels = int(s.dropna().nunique())
                if n_levels < 2:
                    raise ValueError(
                        f"Grouping variable '{col}' has only {n_levels} "
                        f"level(s) after cleaning — need at least 2 for "
                        f"{spec.family} tests."
                    )

    # ------------------------------------------------------------------
    # DSL deterministic path
    # ------------------------------------------------------------------

    def _run_dsl_experiment(self, hypothesis: Hypothesis) -> ExperimentResult:
        """Deterministic pipeline for DSL hypotheses: parse -> compile -> execute -> strict validate.

        No LLM is involved. Unresolved controls / invalid columns cause
        hard failures (DSLParseError) rather than silent degradation.
        """
        if self.data is None or len(self.data) == 0:
            raise RuntimeError(
                "No data available for DSL experiment. "
                "Provide a valid data file path when initializing CoderRunner."
            )

        valid_columns = set(self.data.columns)

        # Lazily build label alias map from variable catalog
        if self._label_aliases is None:
            vc_cfg = config.get("variable_catalog", {}) or {}
            catalog_path = str(vc_cfg.get("catalog_path", "") or "").strip()
            if catalog_path and vc_cfg.get("enabled", False):
                self._label_aliases = build_label_alias_map(catalog_path)
                if self._label_aliases:
                    print(f"  [DSL] Label alias map loaded: {len(self._label_aliases)} entries")
            else:
                self._label_aliases = {}

        # 1. Deterministic parse (with label alias resolution)
        try:
            spec = parse_dsl_hypothesis(hypothesis.sentence, valid_columns, self._label_aliases or None)
        except DSLParseError as exc:
            print(f"  [DSL] Parse failed: {exc}")
            raise ValueError(f"DSL parse error: {exc}") from exc

        # Save spec to hypothesis metadata
        hypothesis.metadata["columns"] = spec.all_columns
        hypothesis.metadata["test_type"] = spec.family
        hypothesis.metadata["parsed"] = spec.to_dict()
        hypothesis.metadata["dsl_path"] = True

        # Pre-flight validation: catch errors before they waste retry attempts.
        self._validate_spec_feasibility(spec)

        # Stash parse artifact (no LLM prompt to record)
        self.last_parse_call = {
            "hypothesis_sentence": hypothesis.sentence,
            "prompt": "(deterministic DSL parser — no LLM prompt)",
            "response": json.dumps(spec.to_dict(), indent=2),
        }

        # 2. Generate equivalent Python code for artifact logging
        equivalent_code = spec_to_python_code(spec)
        self.last_code_gen_call = {
            "test_spec": spec.to_dict(),
            "prompt": "(deterministic compiler — no LLM code generation)",
            "system_prompt": "",
            "response": equivalent_code,
        }
        self.last_code_fix_calls = []

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Apply execution timeout to prevent stuck regressions
                def _timeout_handler(signum, frame):
                    raise TimeoutError(
                        f"DSL experiment exceeded {self.execution_timeout}s timeout"
                    )
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(self.execution_timeout)
                try:
                    exec_result = execute_spec(spec, self.data)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                self.last_execution = {
                    "test_spec": spec.to_dict(),
                    "executed_code": equivalent_code,
                    "formula": exec_result.formula,
                    "test_method": exec_result.test_method,
                }

                result_dict = exec_result.result
                ci = result_dict.get("confidence_interval", (0.0, 0.0))
                if exec_result.warning_severity == "material":
                    result_status = ResultStatus.VALID_WITH_WARNING
                else:
                    result_status = ResultStatus.VALID

                hypothesis.metadata["summary_estimand"] = exec_result.summary_estimand
                hypothesis.metadata["summary_term"] = exec_result.summary_term
                hypothesis.metadata["warning_severity"] = exec_result.warning_severity
                hypothesis.metadata["review_flags"] = list(exec_result.review_flags)
                hypothesis.metadata["sign_semantics"] = exec_result.sign_semantics

                # Split-sample replication (best-effort, never blocks)
                repl_status = replicate_spec(spec, self.data)
                if repl_status != "replicated":
                    print(f"  [DSL] Replication: {repl_status}")

                exp_result = ExperimentResult(
                    hypothesis_id=hypothesis.id,
                    effect_size=float(result_dict["effect_size"]),
                    p_value=float(result_dict["p_value"]),
                    confidence_interval=tuple(ci),
                    n_observations=int(result_dict["n_observations"]),
                    coverage=float(result_dict["coverage"]),
                    replication_status=repl_status,
                    raw_results=result_dict,
                    result_status=result_status,
                    execution_warnings=exec_result.warnings,
                    test_method=exec_result.test_method,
                    formula=exec_result.formula,
                    warning_severity=exec_result.warning_severity,
                    dsl_family=spec.family,
                    review_flags=list(exec_result.review_flags),
                    summary_estimand=exec_result.summary_estimand,
                    summary_term=exec_result.summary_term,
                    sign_semantics=exec_result.sign_semantics,
                )

                if exec_result.warnings:
                    print(f"  [DSL] {len(exec_result.warnings)} warning(s) captured:")
                    for w in exec_result.warnings[:5]:
                        print(f"    ⚠ {w}")

                if attempt > 0:
                    print(f"  [DSL] Succeeded on attempt {attempt + 1}")
                return exp_result

            except (DSLExecutionError, ValueError, KeyError, TimeoutError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    print(f"  [DSL] Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                else:
                    break

        raise RuntimeError(
            f"WVS DSL experiment failed after {self.max_retries} attempts. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    # Backward-compat alias
    _run_wvs_dsl_experiment = _run_dsl_experiment

    # ------------------------------------------------------------------
    # Legacy LLM-based path (free-form hypotheses)
    # ------------------------------------------------------------------

    def _run_llm_experiment(self, hypothesis: Hypothesis) -> ExperimentResult:
        """LLM-based pipeline for non-DSL hypotheses: LLM parse → LLM codegen → exec → sanitise."""
        # Parse hypothesis to extract testable components
        test_spec = self._parse_hypothesis(hypothesis.sentence)
        test_spec["hypothesis_id"] = hypothesis.id
        
        # Structure metadata used by the structural novelty checks
        hypothesis.metadata["columns"] = test_spec.get("columns", [])
        hypothesis.metadata["test_type"] = test_spec.get("type", "unknown")
        hypothesis.metadata["parsed"] = test_spec
        
        # Generate code to test the hypothesis
        code = self._generate_test_code(test_spec)
        self.last_code_fix_calls = []
        
        # Execute the code with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                results = self._execute_test(code, test_spec)
                if attempt > 0:
                    print(f"[OK] Experiment succeeded on attempt {attempt + 1}")
                return results
            except (ValueError, RuntimeError, SyntaxError, NameError) as e:
                last_error = e
                if attempt < self.max_retries - 1 and self.retry_on_failure:
                    print(f"[WARN] Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                    print(f"  Attempting to fix code with LLM...")
                    try:
                        fixed = self._fix_code_with_llm(code, str(e), test_spec)
                        # best-effort: record each fix attempt
                        try:
                            self.last_code_fix_calls.append(
                                {
                                    "attempt": attempt + 1,
                                    "error": f"{type(e).__name__}: {e}",
                                    "fixed_code": fixed,
                                }
                            )
                        except Exception:
                            pass
                        code = fixed
                    except Exception as fix_error:
                        print(f"  [WARN] Code fixing failed: {fix_error}")
                        # Continue with original code for next attempt
                else:
                    # Last attempt or retry disabled
                    break
        
        # All retries exhausted
        raise RuntimeError(
            f"Experiment failed after {self.max_retries} attempts. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        ) from last_error
    
    def _analyze_data_structure(self, df: pd.DataFrame, columns_subset: Optional[list[str]] = None) -> str:
        """Creates a detailed statistical profile of the dataset (Crash-Proof Version)."""
        if df is None or df.empty:
            return "No data available."
        
        buffer = []
        buffer.append(f"Total Rows: {len(df)}")
        total_cols = len(df.columns)
        cols = list(df.columns)
        if columns_subset is not None:
            wanted = [c for c in columns_subset if c in df.columns]
            wanted_set = set(wanted)
            cols = [c for c in df.columns if c in wanted_set]
            buffer.append(f"Profiled Columns: {len(cols)} / {total_cols} (subset for token budget)")
        buffer.append("Column Schema & Statistics:")
        
        for col in cols:
            dtype = df[col].dtype
            nulls = df[col].isnull().sum()
            
            # Smart profiling based on type
            if pd.api.types.is_numeric_dtype(dtype):
                # Show range for numbers (prevents testing conditions on columns with no variation)
                non_null_col = df[col].dropna()
                if len(non_null_col) > 0:
                    try:
                        # Handle inf, nan, and special values safely
                        min_val = non_null_col.min()
                        max_val = non_null_col.max()
                        mean_val = non_null_col.mean()
                        
                        # Check for inf/nan before formatting
                        def safe_format(val):
                            """Safely format a value, handling inf/nan."""
                            if pd.isna(val) or np.isinf(val) or np.isnan(val):
                                return str(val)
                            try:
                                val_float = float(val)
                                if np.isinf(val_float) or np.isnan(val_float):
                                    return str(val)
                                return f"{val_float:.2f}"
                            except (ValueError, TypeError, OverflowError):
                                return str(val)
                        
                        min_str = safe_format(min_val)
                        max_str = safe_format(max_val)
                        mean_str = safe_format(mean_val)
                        details = f"Range=[{min_str} to {max_str}], Mean={mean_str}"
                    except Exception as e:
                        # Last resort: just show the type
                        details = f"Numeric column with {len(non_null_col)} non-null values (format error: {type(e).__name__})"
                else:
                    details = "All values are null"
            else:
                # Show unique values for categories
                unique_vals = list(df[col].dropna().unique())
                if len(unique_vals) < 10:
                    details = f"Values={unique_vals}"
                else:
                    details = f"Examples={unique_vals[:3]}, Total unique={len(unique_vals)}"
            
            # Add special handling for string / object columns
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object":
                # Compute average length as an extra hint
                try:
                    non_null = df[col].dropna().astype(str)
                    if not non_null.empty:
                        avg_len = non_null.str.len().mean()
                        details += f" | text-like, avg length ~{avg_len:.1f} chars"
                except Exception:
                    details += " | text-like"
            
            buffer.append(f"- {col} ({dtype}): {details}, Nulls: {nulls}")
        
        # Add section describing key column types and what the LLM can do with them
        buffer.append(
            "\nDATA ASSETS AVAILABLE:\n"
            "1. TEXT COLUMNS (e.g., 'review_text', 'summary', 'reviewerName', 'product_format'):\n"
            "   - You CAN analyze these using pandas string methods such as .str.len(), "
            ".str.contains(), .str.count(), etc.\n"
            "2. ID COLUMNS ('reviewerID', 'asin'):\n"
            "   - Treat these as identifiers / grouping keys, not numeric values.\n"
            "   - You CAN group by them to analyze user-level or product-level patterns.\n"
            "3. CATEGORICAL COLUMNS (e.g., 'product_format', 'verified_purchase', 'has_image'):\n"
            "   - You CAN compare distributions, averages, or rates across categories.\n"
            "4. NUMERIC COLUMNS (e.g., 'rating', 'helpful_votes', 'review_length', 'year'):\n"
            "   - You CAN compute correlations, regressions, and group comparisons.\n"
        )
        
        # Add NLP-derived features section if they exist
        nlp_feature_cols = []
        nlp_feature_candidates = {
            "sentiment_score", "is_pos_review", "is_neg_review", "exclamation_count",
            "all_caps_ratio", "lexical_richness", "pronoun_ratio", "sentiment_scaled", "rating_sentiment_gap"
        }
        for col in cols:
            if col in nlp_feature_candidates:
                nlp_feature_cols.append(col)
        
        if nlp_feature_cols:
            buffer.append(
                "\n5. NLP-DERIVED REVIEW FEATURES (precomputed offline, use as regular numeric columns):\n"
                "   - sentiment_score: VADER sentiment score, range [-1, 1].\n"
                "   - is_pos_review / is_neg_review: Binary sentiment indicators (0 or 1) based on sentiment_score.\n"
                "   - exclamation_count: Number of '!' characters in the text (integer).\n"
                "   - all_caps_ratio: Ratio of all-caps words (≥3 letters) to total words, range 0–1.\n"
                "   - lexical_richness: Vocabulary diversity = unique tokens / total tokens, range 0–1.\n"
                "   - pronoun_ratio: First-person pronoun (i, me, my, we, us, our) ratio, range 0–1.\n"
                "   - sentiment_scaled: sentiment_score mapped to rating scale [1, 5].\n"
                "   - rating_sentiment_gap: rating - sentiment_scaled, measures gap between rating and text sentiment.\n"
                "   - These features are already computed - use them directly like any numeric column.\n"
                "   - DO NOT recompute sentiment or text analysis - these columns are ready to use.\n"
            )
        
        return "\n".join(buffer)
    
    def _get_weight_column_hint(self) -> str:
        """If config lists weight columns and the current schema has one, return a hint line for the prompt."""
        weight_candidates = config.get("data.weight_columns") or []
        if not weight_candidates or self.data is None:
            return ""
        cols = self.active_columns_subset if self.active_columns_subset is not None else self.data.columns.tolist()
        for w in weight_candidates:
            if w in self.data.columns and w in cols:
                return (
                    f"\nSurvey weight: use column '{w}' in regressions and group comparisons when appropriate "
                    "to get population-representative estimates."
                )
        return ""
    
    def get_data_schema(self, columns_subset: Optional[list[str]] = None) -> Optional[str]:
        """Get the data schema for hypothesis generation constraints.
        
        Returns:
            Schema string describing available columns, or None if data not loaded.
        """
        if self.data is None or len(self.data) == 0:
            return None

        if columns_subset is None:
            columns_subset = self.active_columns_subset

        cols = list(self.data.columns)
        if columns_subset is not None:
            wanted = [c for c in columns_subset if c in self.data.columns]
            # Keep stable order: follow dataset order
            wanted_set = set(wanted)
            cols = [c for c in self.data.columns if c in wanted_set]

        # Exclude columns marked as dropped in the dataset profile
        from core.dataset_profile import drop_columns as _drop_cols
        _drops = set(_drop_cols())
        if _drops:
            cols = [c for c in cols if c not in _drops]
        
        # Create a clean schema description for the Generator
        schema_parts = []
        schema_parts.append("Available columns in the dataset:")

        for col in cols:
            dtype = self.data[col].dtype
            # Get a sample value to understand the type better
            sample = self.data[col].dropna().iloc[0] if not self.data[col].dropna().empty else None
            
            # Create a human-readable description
            if dtype in ['int64', 'int32', 'int']:
                if sample is not None:
                    if isinstance(sample, (int, float)) and 0 <= sample <= 1:
                        desc = f"binary (0 or 1)" if sample in [0, 1] else f"integer"
                    else:
                        desc = "integer"
                else:
                    desc = "integer"
            elif dtype in ['float64', 'float32', 'float']:
                desc = "float"
            elif dtype in ['object', 'string']:
                # Distinguish "coded categorical stored as string" vs free text.
                # WVS-style preprocessing often casts low-cardinality integer codes to string.
                try:
                    s = self.data[col].dropna().astype(str).head(2000)
                    if not s.empty:
                        avg_len = float(s.str.len().mean())
                        # Numeric-looking codes ratio (e.g., "1", "-1", "999")
                        numeric_like = float(s.str.match(r"^-?\d+$").mean())
                        nunique = int(s.nunique(dropna=True))
                        if numeric_like >= 0.8 and nunique <= 50 and avg_len <= 6:
                            desc = "categorical (coded)"
                        elif avg_len >= 25:
                            desc = "free text"
                        else:
                            desc = "string/categorical"
                    else:
                        desc = "string/categorical"
                except Exception:
                    desc = "string/categorical"
            else:
                desc = str(dtype)
            
            # Add value examples for categorical/string columns to prevent hallucination
            if desc in ("string/categorical", "categorical (coded)") and sample is not None:
                try:
                    vc = self.data[col].dropna().astype(str).value_counts()
                    top_vals = vc.head(6).index.tolist()
                    n_unique = len(vc)
                    if top_vals:
                        examples = ", ".join(f'"{v}"' for v in top_vals[:6])
                        desc += f"; top values: [{examples}]"
                    # Warn about single-value columns — useless for grouping
                    if n_unique < 2:
                        desc += " — SINGLE VALUE, do NOT use for grouping (by=, |)"
                except Exception:
                    pass
            elif desc == "binary (0 or 1)":
                pass
            elif desc in ("integer", "float") and sample is not None:
                try:
                    col_data = self.data[col].dropna()
                    if len(col_data) > 0:
                        desc += f"; range [{col_data.min():.4g}, {col_data.max():.4g}]"
                        # Warn about high-cardinality numerics — unsuitable as
                        # grouping/moderation variables (would explode into dummies).
                        n_unique = int(col_data.nunique())
                        if n_unique > 30:
                            desc += " — CONTINUOUS, do NOT use for grouping (by=, |)"
                except Exception:
                    pass

            schema_parts.append(f"- {col} ({desc})")

        # Add note about NLP features if they exist
        from core.dataset_profile import nlp_features as _profile_nlp
        _profile_nlp_set = set(_profile_nlp())
        nlp_feature_candidates = _profile_nlp_set or {
            "sentiment_score", "is_pos_review", "is_neg_review", "exclamation_count",
            "all_caps_ratio", "lexical_richness", "pronoun_ratio", "sentiment_scaled",
            "rating_sentiment_gap", "word_count", "sentence_count", "avg_word_length",
            "readability_score", "question_count", "uppercase_ratio", "lexical_diversity",
        }
        nlp_cols_found = [col for col in cols if col in nlp_feature_candidates]
        if nlp_cols_found:
            schema_parts.append("")
            schema_parts.append(
                f"Note: {', '.join(nlp_cols_found)} are precomputed NLP features that can be used directly "
                "as numeric columns. No need to recompute sentiment analysis or text processing."
            )
        
        return "\n".join(schema_parts)

    def _is_wvs_like_dataset(self) -> bool:
        """Heuristic: detect WVS-like schema by Q-variables / standard metadata columns."""
        if self.data is None:
            return False
        cols = [str(c) for c in self.data.columns]
        q_like = sum(1 for c in cols if re.match(r"^Q\d{1,3}([_][A-Z0-9]+)?$", c))
        has_meta = any(c in {"B_COUNTRY", "B_COUNTRY_ALPHA", "W_WEIGHT", "PWGHT", "A_YEAR"} for c in cols)
        return q_like >= 20 or (q_like >= 5 and has_meta)

    def _is_dsl_dataset(self) -> bool:
        """Profile-driven check: does the active profile declare DSL mode?"""
        from core.dataset_profile import use_dsl
        return use_dsl()

    def _apply_missing_code_cleanup(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Recode dataset-specific missing codes to NaN for hypothesis-involved columns.
        Reads codes from active dataset profile; falls back to WVS defaults.
        """
        if df is None or df.empty or not columns:
            return df
        from core.dataset_profile import missing_codes as _profile_mc
        special_codes = _profile_mc() or {-1, -2, -3, -4, -5}
        if not special_codes:
            return df
        cleaned = df.copy()
        for col in columns:
            if col not in cleaned.columns:
                continue
            s = cleaned[col]
            # Try numeric coercion; skip columns that are predominantly non-numeric text.
            num = pd.to_numeric(s, errors="coerce")
            numeric_ratio = float(num.notna().mean()) if len(num) else 0.0
            if numeric_ratio < 0.5:
                continue
            cleaned[col] = num.mask(num.isin(special_codes), np.nan)
        return cleaned
    
    def _parse_hypothesis(self, sentence: str) -> dict:
        """Parse hypothesis sentence to extract testable components including CONCRETE COLUMNS."""
        
        # Get schema to help LLM map terms to exact column names
        schema_info = self.get_data_schema() or "No schema available"
        
        parse_prompt = f"""Parse the following hypothesis into structured components.

Hypothesis: "{sentence}"

DATA SCHEMA (Map terms to these EXACT column names):
{schema_info}

Extract:
1. Type: correlation, causal, temporal, difference, or interaction
2. Columns: List of EXACT column names from the schema involved in this hypothesis.
   - Map natural language terms to actual column names from the schema (choose exact strings from the schema list).
   - Include ALL columns mentioned: independent variables, dependent variables, control variables, grouping variables
3. Entities: User groups or filters (e.g., "verified users", "expensive items")
4. Conditions: List of conditions or thresholds (e.g., "≥3", ">30%", "within 7 days")
   - If the hypothesis uses a text-like column, represent conditions as string matching on that column
     (e.g., df['<text_col>'].str.contains('keyword', case=False, na=False)).
5. Outcome: The main metric being measured (as a column name from schema)
6. Time window: time period mentioned in days (if any)
7. Weights: optional weight column if explicitly present as `weights=<column_name>`, else null

CRITICAL: The "columns" field must contain ONLY actual column names from the schema above. 
If a term doesn't match a column name exactly, choose the closest matching column name from the schema.

Respond in JSON format:
{{
    "type": "correlation",
    "columns": ["<X_column>", "<Y_column>"],
    "entities": ["<group_or_filter_if_any>"],
    "conditions": ["<condition_if_any>"],
    "outcome": "<Y_column>",
    "time_window": 7,
    "weights": null
}}"""

        if not self.llm.is_available():
            raise RuntimeError(
                "LLM is not available. Cannot parse hypothesis. "
                "Please ensure the LLM service is running and properly configured."
            )
        
        try:
            response = self.llm.generate(
                prompt=parse_prompt,
                temperature=0.3,  # Lower temperature for structured output
                system="You are a data analysis assistant. Parse hypotheses into structured JSON format."
            )
            # stash raw prompt/response for artifacts
            self.last_parse_call = {
                "hypothesis_sentence": sentence,
                "prompt": parse_prompt,
                "response": response,
            }
            
            parsed = extract_json(response)
            parsed["hypothesis_sentence"] = sentence
            
            # Validate columns exist in data
            if self.data is not None:
                valid_cols = set(self.data.columns)
                extracted = set(parsed.get("columns", []))
                # Filter out hallucinations - only keep columns that actually exist
                parsed["columns"] = list(extracted.intersection(valid_cols))
                # Heuristic fallback: extract weights=COL directly from sentence if parser omitted it.
                if not parsed.get("weights"):
                    m = re.search(r"\bweights\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", sentence)
                    if m:
                        parsed["weights"] = m.group(1)
                # Validate optional weights column.
                w = parsed.get("weights")
                if isinstance(w, str) and w in valid_cols:
                    parsed["weights"] = w
                    if w not in parsed["columns"]:
                        parsed["columns"].append(w)
                else:
                    parsed["weights"] = None
            
            return parsed
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to parse hypothesis from LLM response: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Hypothesis parsing failed: {e}. "
                f"Please check LLM service connection and configuration."
            ) from e
    
    def _generate_test_code(self, test_spec: dict) -> str:
        """Generate Python code to test the hypothesis using LLM."""
        hypothesis = test_spec.get("hypothesis_sentence", "")
        weight_col = test_spec.get("weights")
        
        try:
            data_profile = (
                self._analyze_data_structure(self.data, columns_subset=self.active_columns_subset)
                if self.data is not None
                else "No data available."
            )
        except Exception as e:
            # Fallback if data profiling fails
            print(f"[WARN] Data profiling failed: {e}. Using simplified schema.")
            if self.data is not None:
                cols = self.active_columns_subset or self.data.columns.tolist()
                data_profile = f"Total Rows: {len(self.data)}\nColumns: {', '.join([c for c in cols if c in self.data.columns])}"
            else:
                data_profile = "No data available."
        data_profile = data_profile.rstrip() + self._get_weight_column_hint()
        
        # Safely format test_spec values to avoid format specifier errors
        def safe_str(val):
            """Convert value to string safely, handling special cases."""
            if val is None:
                return "None"
            try:
                return str(val)
            except Exception:
                return repr(val)
        
        # Escape braces in the hypothesis to prevent format errors: { and } become {{ and }}
        safe_hypothesis = hypothesis.replace('{', '{{').replace('}', '}}') if hypothesis else ""
        
        # Dynamic hints based on actual columns (avoid dataset-specific hallucinations).
        text_like_cols = []
        id_like_cols = []
        date_like_cols = []
        if self.data is not None:
            for c in self.data.columns:
                cl = str(c).lower()
                if "date" in cl or "time" in cl:
                    date_like_cols.append(str(c))
                if cl.endswith("id") or cl.endswith("_id") or cl in {"survey_id"}:
                    id_like_cols.append(str(c))
                # Heuristic for free-text: object/string with long average length
                try:
                    if (pd.api.types.is_string_dtype(self.data[c]) or self.data[c].dtype == "object"):
                        s = self.data[c].dropna().astype(str).head(2000)
                        if not s.empty and float(s.str.len().mean()) >= 25:
                            text_like_cols.append(str(c))
                except Exception:
                    pass

        text_like_cols = sorted(set(text_like_cols))[:6]
        id_like_cols = sorted(set(id_like_cols))[:8]
        date_like_cols = sorted(set(date_like_cols))[:8]

        text_hint = ""
        if text_like_cols:
            text_hint = (
                "\n5. TEXT COLUMNS:\n"
                f"   - Detected text-like columns: {', '.join(text_like_cols)}\n"
                "   - You MAY use pandas string operations (str.len, str.contains, str.count) on these.\n"
                "   - Avoid external NLP libraries.\n"
            )

        id_hint = ""
        if id_like_cols:
            id_hint = (
                "\n6. IDENTIFIER / KEY COLUMNS:\n"
                f"   - Detected ID-like columns: {', '.join(id_like_cols)}\n"
                "   - Use these for grouping/aggregation only; do NOT treat their numeric magnitude as meaningful.\n"
            )

        date_hint = ""
        if date_like_cols:
            date_hint = (
                "\n11. TIME/DATE COLUMNS:\n"
                f"   - Detected time/date-like columns: {', '.join(date_like_cols)}\n"
                "   - If you need datetime operations, use pd.to_datetime on the appropriate column.\n"
            )

        code_prompt = f"""Generate Python code to test the following hypothesis.

Hypothesis: {safe_hypothesis}

CRITICAL: Use ONLY columns that appear in the schema below. Do NOT invent, assume, or derive
column names that are not listed. If a column does not exist, skip that part of the analysis.

Actual Data Schema (The dataframe is named 'df'):
{data_profile}

Test Specification:
- Type: {safe_str(test_spec.get('type', 'correlation'))}
- Outcome: {safe_str(test_spec.get('outcome', 'unknown'))}
- Time window: {safe_str(test_spec.get('time_window', 7))} days
- Entities: {safe_str(test_spec.get('entities', []))}
- Conditions: {safe_str(test_spec.get('conditions', []))}

Requirements:
1. Write a function 'test_hypothesis(df)' that returns a dictionary with EXACTLY these keys:
   - effect_size (float): STANDARDIZED magnitude — Pearson r, Cohen's d, or standardized beta.
     IMPORTANT: If using OLS regression, return the STANDARDIZED coefficient (divide by outcome std),
     NOT the raw unstandardized coefficient. Effect size must be in [-5, 5].
   - p_value (float, in [0,1]): statistical significance of the test
   - n_observations (int): number of rows used in the test
   - coverage (float, in [0,1]): fraction of the full dataset used (len(subset)/len(df))
   - confidence_interval (tuple of 2 floats): (lower, upper) bound on effect_size

   MANDATORY RETURN EXAMPLE — your function MUST end like this:
   ```
   return {{
       'effect_size': float(r),                          # e.g. correlation, Cohen's d, slope
       'p_value': float(p),                              # from stats test
       'n_observations': int(len(subset)),
       'coverage': float(len(subset) / len(df)),
       'confidence_interval': (float(ci_lower), float(ci_upper)),
   }}
   ```
   You MAY include extra keys for debugging, but the 5 keys above are NON-NEGOTIABLE.
   DO NOT rename them (e.g. do NOT return 'correlation' instead of 'effect_size').

2. IMPORTS - CRITICAL:
   - ALLOWED: pd, np, scipy.stats as stats
   - NEW ALLOWED: statsmodels.api as sm, statsmodels.formula.api as smf (if available)
   - Use statsmodels for regression if the hypothesis implies "controlling for", "accounting for", or "independent of".
   - ALLOWED functions: stats.ttest_ind, stats.pearsonr, stats.chi2_contingency, stats.f_oneway, stats.mannwhitneyu, stats.kruskal, stats.norm.ppf
   - FORBIDDEN (DO NOT USE - THEY DON'T EXIST): 
     * stats.cohen_d, stats.proportion_confint, stats.proportions_ztest, stats.pearsonr_ci
     * calculate_proportion_confint (any variation of this name)
   - If you need Effect Size (Cohen's d) or Confidence Intervals, CALCULATE THEM MANUALLY:
     * Cohen's d: pooled_std = np.sqrt((std1**2 + std2**2) / 2); d = (mean1 - mean2) / pooled_std
     * Proportion CI: se = np.sqrt(p * (1-p) / n); ci_lower = p - 1.96*se; ci_upper = p + 1.96*se

3. ADVANCED ANALYSIS (REGRESSION):
   - If the hypothesis is "X affects Y accounting for Z", use OLS:
     `model = smf.ols('Y ~ X + Z', data=df).fit()`
     `p_value = model.pvalues['X']`
     `effect_size = model.params['X']` (slope)
   - If statsmodels fails/missing, fall back to analyzing subsets (e.g., split Z into bins).

   - If a column is categorical (e.g. country codes, product categories, urban/rural),
     treat it as categorical: use C(col) in formulas, or create dummies.
   - Never use ID-like or coded columns as a numeric magnitude.

3.5 WEIGHTS (if provided):
   - Detected weight column from hypothesis DSL/parsing: {safe_str(weight_col)}
   - If a valid weight column is provided, use it as model/group weight (WLS or weighted aggregation).
   - Do NOT include weight columns as predictors or outcomes.

4. DEFENSIVE CODING (CRITICAL - Prevents ZeroDivisionError, NaN, and empty data errors):
   - ALWAYS check if filtered data is empty: `if len(filtered) < 2: return {{'effect_size': 0.0, 'p_value': 1.0, 'n_observations': 0, 'coverage': 0.0, 'confidence_interval': (0.0, 0.0)}}`
   - ALWAYS check for zero division BEFORE dividing: `if denominator == 0 or np.isclose(denominator, 0): return {{'effect_size': 0.0, 'p_value': 1.0, ...}}`
   - ALWAYS check for zero variance before statistical tests: `if np.var(group1) == 0 or np.var(group2) == 0: return {{...}}`
   - ALWAYS use np.nan_to_num() or explicit checks to prevent NaN propagation: `result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)`
   - If data is insufficient (n < 2), return zero result dictionary immediately
   - Check if groups have enough data before statistical tests (minimum 2 observations per group)
   - After any division operation, check for NaN: `if np.isnan(result): result = 0.0`
   - When calculating standard deviations, use: `std = np.std(data, ddof=1) if len(data) > 1 else 0.0`
   - When creating filtered subsets and then assigning new columns, do:
     `filtered = filtered.copy()` before assignments, or use `.loc[...] = ...` to avoid SettingWithCopyWarning.

{text_hint}{id_hint}

7. NLP-DERIVED FEATURES (if available in the dataset):
   - The dataset may contain precomputed NLP features as numeric columns.
   - These are already computed offline - use them directly. DO NOT recompute.
   - Use them like any other numeric column in correlations, regressions, or group comparisons.

8. PERFORMANCE & SCALE:
   - Assume the DataFrame can have up to ~500k rows after internal sampling.
   - AVOID:
     * O(N^2) operations such as full pairwise distance matrices over rows.
     * Python for-loops over all rows.
   - Prefer:
     * vectorized operations,
     * boolean indexing,
     * groupby / agg,
     * merging precomputed aggregates back onto the main DataFrame.

9. RESULT CONTRACT (MANDATORY):
   - You MUST ensure that, at the end of the code, one of the following is true:
     * A function `test_hypothesis(df)` is defined AND a global variable `result` is set as
       `result = test_hypothesis(df)`, OR
     * A global variable `result` is directly assigned a dict-like object with the final metrics.
   - The execution environment will look for `result` (or 'results'/'output' as a fallback).
   - Code that does not set `result` will be treated as a failure.

9. STATISTICS - CRITICAL:
   - DO NOT call these NON-EXISTENT functions: `calculate_proportion_confint`, `stats.proportions_ztest`, `stats.cohen_d`
   - For proportions/confidence intervals, ALWAYS calculate manually:
     p = proportion
     n = sample_size
     se = np.sqrt(p * (1 - p) / n)
     ci_lower = p - 1.96 * se
     ci_upper = p + 1.96 * se
   - For proportion z-test, use: `from statsmodels.stats.proportion import proportions_ztest` (if available) OR calculate manually
   - For Cohen's d (effect size), calculate manually: `d = (mean1 - mean2) / pooled_std` where `pooled_std = np.sqrt((std1**2 + std2**2) / 2)`
   - Available scipy.stats functions: stats.ttest_ind, stats.pearsonr, stats.chi2_contingency, stats.f_oneway, stats.mannwhitneyu, stats.kruskal, stats.norm.ppf

{date_hint}

11. FORBIDDEN: 
   - Do NOT use try/except blocks. Let errors raise so the system can catch them properly.
   - Do not import matplotlib, seaborn, or plotly. Do not generate charts.
   - Do NOT call functions that don't exist in scipy.stats (check the ALLOWED list above)
   - Do NOT import NLP libraries like nltk, spacy, textblob, or sklearn.text.
   - Use ONLY pandas / numpy / scipy.stats / statsmodels (if already allowed elsewhere in the prompt).
   - Do NOT use external NLP models; all text analysis must be doable with pandas string methods.

12. SAFETY: Do not use 'while' loops. Ensure all loops have explicit bounds (use 'for' loops with iterables or range()).

13. OPTIMIZATION: If using merge/join, ensure you are not creating a Cartesian product (check sizes before joining).

14. COLUMN VALIDATION - CRITICAL:
   - ALWAYS verify column names exist in df.columns before using them
   - DO NOT assume columns like 'total_votes', 'user_id', 'item_id' exist - check the actual schema
   - Use ONLY columns listed in the data schema provided above
   - If you need to check if a column exists: `if 'column_name' in df.columns: ...`
   - Adapt to the actual data structure - do not assume specific column names

16. Return ONLY executable Python code, no markdown, no explanations

Code:"""

        if not self.llm.is_available():
            raise RuntimeError(
                "LLM is not available. Cannot generate test code. "
                "Please ensure the LLM service is running and properly configured."
            )
        
        try:
            system_prompt = """You are an expert Python data scientist. 

CRITICAL RULE: You generally DO NOT have access to functions like 'ttest_ind' directly. 

ALWAYS use the 'stats.' prefix (e.g., 'stats.ttest_ind', 'stats.pearsonr').

NEVER use 'statsmodels' unless you are sure it is installed.

Return only executable code, wrapped in a function if needed. Make sure the code is complete and can be executed."""
            
            code = self.llm.generate(
                prompt=code_prompt,
                temperature=0.2,  # Low temperature for code generation
                system=system_prompt,
                max_tokens=2000
            )
            self.last_code_gen_call = {
                "test_spec": dict(test_spec),
                "prompt": code_prompt,
                "system_prompt": system_prompt,
                "response": code,
            }
            
            code = extract_code_block(code, min_length=50)

            # Pre-validate and auto-fix code for common errors
            code = self._validate_generated_code(code)
            
            return code
        except Exception as e:
            raise RuntimeError(
                f"Code generation failed: {e}. "
                f"Please check LLM service connection and configuration."
            ) from e
    
    def _validate_generated_code(self, code: str) -> str:
        """Pre-validate and auto-fix generated code for common errors before execution.

        Returns:
            Fixed code string with common errors automatically corrected.
        """
        # Strip import statements — all libraries are pre-injected into exec_globals.
        # The LLM often generates import lines despite being told not to.
        stripped_lines = []
        for line in code.splitlines():
            s = line.strip()
            if s.startswith("import ") or (s.startswith("from ") and "import" in s):
                # Keep the line as a comment so indentation isn't broken
                stripped_lines.append(line.replace(s, f"# (stripped) {s}"))
            else:
                stripped_lines.append(line)
        code = "\n".join(stripped_lines)

        forbidden_functions = [
            'calculate_proportion_confint',
            'stats.proportions_ztest',
            'stats.cohen_d',
            'stats.proportion_confint',
            'stats.pearsonr_ci'
        ]

        # Security: block patterns that could escape the data-analysis sandbox.
        # These are never needed for statistical tests on a dataframe.
        forbidden_security = [
            ('os.system',      'shell execution'),
            ('subprocess',     'shell execution'),
            ('shutil.rmtree',  'file deletion'),
            ('__import__',     'dynamic import'),
            ('importlib',      'dynamic import'),
            ('eval(',          'dynamic code execution'),
            ('exec(',          'dynamic code execution'),
            ('compile(',       'dynamic code compilation'),
            ('globals(',       'global scope access'),
            ('breakpoint(',    'debugger'),
            ('socket.',        'network access'),
            ('urllib',         'network access'),
            ('requests.',      'network access'),
            ('http.',          'network access'),
        ]

        for pattern, reason in forbidden_security:
            if pattern in code:
                raise ValueError(
                    f"Generated code contains forbidden pattern: '{pattern}' ({reason}). "
                    f"Only statistical analysis code is allowed."
                )

        # Block file writes (open with write/append modes)
        if re.search(r"""open\s*\([^)]*[\'\"]([wax+])[\'\"]""", code):
            raise ValueError(
                "Generated code attempts to write files. "
                "Only statistical analysis code is allowed."
            )

        # Check for forbidden functions (non-existent API calls)
        for func in forbidden_functions:
            if func in code:
                raise ValueError(
                    f"Generated code contains forbidden function: {func}. "
                    f"This function does not exist. Please calculate manually instead."
                )
        
        # Auto-fix common errors: functions used without stats. prefix
        # This prevents the common "pearsonr" -> "stats.pearsonr" error
        common_stats_functions = [
            'pearsonr', 'ttest_ind', 'ttest_rel', 'ttest_1samp',
            'chi2_contingency', 'f_oneway', 'mannwhitneyu', 'kruskal',
            'spearmanr', 'kendalltau', 'shapiro', 'normaltest'
        ]
        
        for func_name in common_stats_functions:
            # Pattern: function call without stats. prefix (but not in comments or strings)
            # Match: pearsonr( or pearsonr ( but not stats.pearsonr or 'pearsonr'
            # Use word boundary to avoid matching partial words
            pattern = rf'(?<!stats\.)(?<![\'"])\b{re.escape(func_name)}\s*\('
            if re.search(pattern, code):
                # Replace with stats.func_name
                code = re.sub(pattern, f'stats.{func_name}(', code)
        
        # Handle norm separately (it's a module, not a function, so we need to be more careful)
        # Only fix if it's used as norm.ppf or similar
        norm_pattern = r'(?<!stats\.)(?<![\'"])\bnorm\.(ppf|pdf|cdf|isf|isf)'
        if re.search(norm_pattern, code):
            code = re.sub(norm_pattern, r'stats.norm.\1', code)
        
        # Check for column names if data is available
        if self.data is not None:
            available_columns = set(self.data.columns)
            # Common hallucinated column names
            common_hallucinations = ['total_votes', 'user_id', 'item_id', 'timestamp_ms', 'created_at']
            for col in common_hallucinations:
                if col in code and col not in available_columns:
                    # Check if it's actually used (not just in a comment)
                    pattern = rf'\b{re.escape(col)}\b'
                    if re.search(pattern, code) and (f"'{col}'" in code or f'"{col}"' in code or f'[{col}]' in code):
                        raise ValueError(
                            f"Generated code references non-existent column: '{col}'. "
                            f"Available columns: {sorted(available_columns)}"
                        )
        
        return code
    
    def _fix_code_with_llm(self, code: str, error_message: str, test_spec: dict) -> str:
        """Ask LLM to fix code based on error message."""
        if not self.llm.is_available():
            raise RuntimeError("LLM not available for code fixing")
        
        # Get full data profile for fixing
        data_profile = (
            self._analyze_data_structure(self.data, columns_subset=self.active_columns_subset)
            if self.data is not None
            else "No data available."
        )
        data_profile = data_profile.rstrip() + self._get_weight_column_hint()
        
        fix_prompt = f"""The following Python code failed with an error. Please fix it.

Original Code:
```python
{code.replace('{', '{{').replace('}', '}}')}
```

Error Message:
{error_message}

Hypothesis: {test_spec.get('hypothesis_sentence', '')}

Actual Data Schema (The dataframe is named 'df'):
{data_profile}

Requirements:
1. Fix the error while maintaining the original functionality
2. Ensure the function 'test_hypothesis(df)' exists and returns a dictionary with: effect_size, p_value, n_observations, coverage, confidence_interval

3. IMPORTS - CRITICAL:
   - ALLOWED: pd, np, scipy.stats as stats
   - NEW ALLOWED: statsmodels.api as sm, statsmodels.formula.api as smf (if available)
   - Use statsmodels for regression if the hypothesis implies "controlling for", "accounting for", or "independent of".
   - ALLOWED functions: stats.ttest_ind, stats.pearsonr, stats.chi2_contingency, stats.f_oneway, stats.mannwhitneyu, stats.kruskal
   - FORBIDDEN: stats.cohen_d, stats.proportion_confint, stats.pearsonr_ci (Do NOT invent functions!)
   - If you need Effect Size or Confidence Intervals, CALCULATE THEM MANUALLY using numpy

4. ADVANCED ANALYSIS (REGRESSION):
   - If the hypothesis is "X affects Y accounting for Z", use OLS:
     `model = smf.ols('Y ~ X + Z', data=df).fit()`
     `p_value = model.pvalues['X']`
     `effect_size = model.params['X']` (slope)
   - If statsmodels fails/missing, fall back to analyzing subsets (e.g., split Z into bins).
   - If B_COUNTRY or H_URBRURAL are controls, treat them as categorical via C(...) in formula terms.

5. DEFENSIVE CODING (CRITICAL - Prevents ZeroDivisionError, NaN, and empty data errors):
   - ALWAYS check if filtered data is empty: `if len(filtered) < 2: return {{'effect_size': 0.0, 'p_value': 1.0, 'n_observations': 0, 'coverage': 0.0, 'confidence_interval': (0.0, 0.0)}}`
   - ALWAYS check for zero division BEFORE dividing: `if denominator == 0 or np.isclose(denominator, 0): return {{'effect_size': 0.0, 'p_value': 1.0, ...}}`
   - ALWAYS check for zero variance before statistical tests: `if np.var(group1) == 0 or np.var(group2) == 0: return {{...}}`
   - ALWAYS use np.nan_to_num() or explicit checks to prevent NaN propagation: `result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)`
   - If data is insufficient (n < 2), return zero result dictionary immediately
   - Check if groups have enough data before statistical tests (minimum 2 observations per group)
   - After any division operation, check for NaN: `if np.isnan(result): result = 0.0`
   - When calculating standard deviations, use: `std = np.std(data, ddof=1) if len(data) > 1 else 0.0`
   - Before assigning to filtered DataFrames, call `.copy()` or use `.loc` assignments to avoid SettingWithCopyWarning.

5. TEXT ANALYSIS:
   - You ARE allowed to use pandas string operations on text columns such as 'review_text' and 'summary'.
   - Examples:
     df['review_len'] = df['review_text'].str.len()
     df['has_battery'] = df['review_text'].str.contains('battery', case=False, na=False)

6. AGGREGATION WITH IDS:
   - You ARE allowed to group by identifier columns such as 'reviewerID' or 'asin' to compute user-level
     or product-level metrics.
   - Examples:
     user_counts = df.groupby('reviewerID')['rating'].count()
     product_helpful = df.groupby('asin')['helpful_votes'].mean()
   - Treat 'reviewerID' and 'asin' as grouping keys only; do NOT treat their numeric magnitude as meaningful.

7. NLP-DERIVED FEATURES (if available in the dataset):
   - The dataset may contain precomputed NLP features: sentiment_score, is_pos_review, is_neg_review,
     exclamation_count, all_caps_ratio, lexical_richness, pronoun_ratio, sentiment_scaled, rating_sentiment_gap.
   - These are already computed offline - use them directly as numeric columns.
   - Examples:
     correlation = stats.pearsonr(df['sentiment_score'], df['rating'])[0]
     gap_outliers = df[df['rating_sentiment_gap'].abs() > 1.0]
     high_exclamation = df[df['exclamation_count'] > 3]
   - DO NOT recompute sentiment analysis or text processing - these columns are ready to use.

8. PERFORMANCE & SCALE:
   - Assume the DataFrame can have up to ~500k rows after internal sampling.
   - AVOID: O(N^2) operations, Python for-loops over all rows.
   - Prefer: vectorized operations, boolean indexing, groupby / agg.

9. RESULT CONTRACT (MANDATORY):
   - You MUST ensure that, at the end of the code, one of the following is true:
     * A function `test_hypothesis(df)` is defined AND a global variable `result` is set as
       `result = test_hypothesis(df)`, OR
     * A global variable `result` is directly assigned a dict-like object with the final metrics.
   - The execution environment will look for `result` (or 'results'/'output' as a fallback).
   - Code that does not set `result` will be treated as a failure.

9. STATISTICS - CRITICAL:
   - DO NOT call these NON-EXISTENT functions: `calculate_proportion_confint`, `stats.proportions_ztest`, `stats.cohen_d`
   - For proportions/confidence intervals, ALWAYS calculate manually:
     p = proportion; n = sample_size
     se = np.sqrt(p * (1 - p) / n)
     ci_lower = p - 1.96 * se
     ci_upper = p + 1.96 * se
   - For proportion z-test, use statsmodels if available OR calculate manually
   - Available scipy.stats functions: stats.ttest_ind, stats.pearsonr, stats.chi2_contingency, stats.f_oneway, stats.mannwhitneyu, stats.kruskal, stats.norm.ppf

10. DATES: A 'date' column (YYYY-MM-DD string) is already available. Use df['date'] directly, or convert with: df['date_dt'] = pd.to_datetime(df['date'])

11. FORBIDDEN: 
   - Do NOT use try/except blocks. Let errors raise.
   - Do not import matplotlib, seaborn, or plotly. Do not generate charts.
   - Do NOT call functions that don't exist in scipy.stats
   - Do NOT import NLP libraries like nltk, spacy, textblob, or sklearn.text.
   - Use ONLY pandas / numpy / scipy.stats / statsmodels (if already allowed elsewhere in the prompt).
   - Do NOT use external NLP models; all text analysis must be doable with pandas string methods.

12. SAFETY: Do not use 'while' loops. Ensure all loops have explicit bounds.

13. OPTIMIZATION: If using merge/join, ensure you are not creating a Cartesian product.

14. COLUMN VALIDATION - CRITICAL:
   - ALWAYS verify column names exist in df.columns before using them
   - DO NOT assume columns exist - check the actual schema provided
   - Use ONLY columns listed in the data schema above
   - Handle missing columns gracefully - if a column doesn't exist, return default result dict

15. Return ONLY the fixed Python code, no markdown, no explanations

Fixed Code:"""

        try:
            system_prompt = """You are an expert Python debugger. Fix code errors while maintaining functionality.

CRITICAL RULES:
1. ALWAYS use the 'stats.' prefix for scipy.stats functions (e.g., 'stats.ttest_ind', 'stats.pearsonr').
2. NEVER use functions directly like 'ttest_ind' - they are not in scope.
3. NEVER call these NON-EXISTENT functions: 'calculate_proportion_confint', 'stats.proportions_ztest', 'stats.cohen_d'
   Instead, calculate manually: CI = p ± 1.96*sqrt(p*(1-p)/n), Cohen's d = (mean1-mean2)/pooled_std
4. ALWAYS verify column names exist in df.columns before using them
5. You MUST set a global variable 'result' with the final dictionary at the end of your code.

Return only the fixed code, no explanations."""
            
            fixed_code = self.llm.generate(
                prompt=fix_prompt,
                temperature=0.1,  # Very low temperature for fixing
                system=system_prompt,
                max_tokens=2000
            )
            # Best-effort: record the fix call (prompt can be large)
            try:
                self.last_code_fix_calls.append(
                    {
                        "error_message": error_message,
                        "prompt": fix_prompt,
                        "system_prompt": system_prompt,
                        "response": fixed_code,
                    }
                )
            except Exception:
                pass
            
            fixed_code = extract_code_block(fixed_code, min_length=50)

            # Apply auto-fix to the fixed code as well
            fixed_code = self._validate_generated_code(fixed_code)
            
            return fixed_code
        except Exception as e:
            raise RuntimeError(f"Failed to fix code with LLM: {e}") from e
    
    # Maps common LLM-generated key names to the required standard keys.
    _KEY_ALIASES: Dict[str, list] = {
        "effect_size": [
            "correlation", "r", "rho", "coef", "coefficient", "beta", "d",
            "cohens_d", "effect", "cramers_v", "eta_squared", "slope",
            "odds_ratio", "mean_diff", "difference", "delta",
        ],
        "p_value": [
            "p", "pval", "p_val", "pvalue", "significance",
            "p_value_result", "pval_result", "prob",
        ],
        "n_observations": [
            "n", "N", "count", "sample_size", "num_obs", "total",
            "n_obs", "nobs", "n_samples", "num_samples",
        ],
        "coverage": [
            "coverage_rate", "fraction", "prop", "proportion", "cov",
            "data_coverage", "frac",
        ],
        "confidence_interval": [
            "ci", "conf_interval", "ci_bounds", "conf_int", "interval",
            "confidence_interval_95", "ci_95",
        ],
    }

    def _normalize_result_keys(self, results: dict) -> dict:
        """Map common LLM key aliases to the required standard keys.

        Returns a new dict with standard keys filled in from aliases where
        the standard key was absent. Extra keys are preserved unchanged.
        """
        normalized = dict(results)
        for standard_key, aliases in self._KEY_ALIASES.items():
            if standard_key not in normalized:
                for alias in aliases:
                    if alias in results:
                        normalized[standard_key] = results[alias]
                        break
        return normalized

    def _execute_test(self, code: str, test_spec: dict) -> ExperimentResult:
        """Execute test code and return results.
        
        Raises:
            RuntimeError: If no data is available
            ValueError: If code execution fails or returns invalid results
        """
        if self.data is None or len(self.data) == 0:
            raise RuntimeError(
                "No data available for experiment execution. "
                "Please provide a valid data file path when initializing CoderRunner."
            )
        
        # Prepare execution environment with safe imports.
        # Restrict __builtins__ to a safe subset — block file I/O, imports,
        # exec/eval, and other operations that are never needed for
        # statistical analysis on a pre-loaded dataframe.
        _safe_builtins = {
            k: v for k, v in __builtins__.items()
            if k not in {
                '__import__', 'exec', 'eval', 'compile',
                'open', 'input', 'breakpoint', 'exit', 'quit',
                'globals', 'locals', 'vars', 'dir',
                'getattr', 'setattr', 'delattr',
                'memoryview', '__build_class__',
            }
        } if isinstance(__builtins__, dict) else {
            k: getattr(__builtins__, k)
            for k in dir(__builtins__)
            if not k.startswith('_') and k not in {
                '__import__', 'exec', 'eval', 'compile',
                'open', 'input', 'breakpoint', 'exit', 'quit',
                'globals', 'locals', 'vars', 'dir',
                'getattr', 'setattr', 'delattr',
                'memoryview', '__build_class__',
            }
        }
        exec_globals = {
            'pd': pd,
            'np': np,
            'pandas': pd,
            'numpy': np,
            '__builtins__': _safe_builtins,
        }
        
        # Safely import optional libraries - use importlib for proper module loading
        try:
            exec_globals['scipy'] = importlib.import_module('scipy')
            exec_globals['stats'] = importlib.import_module('scipy.stats')
        except ImportError as e:
            raise RuntimeError(
                f"Missing required library: scipy. "
                f"Install with: pip install scipy"
            ) from e
        
        # Import statsmodels for regression analysis (optional)
        try:
            exec_globals['sm'] = importlib.import_module('statsmodels.api')
            exec_globals['smf'] = importlib.import_module('statsmodels.formula.api')
            exec_globals['statsmodels'] = importlib.import_module('statsmodels')
            # Pre-import proportions_ztest if available
            try:
                from statsmodels.stats.proportion import proportions_ztest
                exec_globals['proportions_ztest'] = proportions_ztest
            except Exception:
                pass
        except ImportError:
            # statsmodels is optional, just continue without it
            pass
        
        try:
            exec_globals['sklearn'] = __import__('sklearn')
        except ImportError:
            # sklearn is optional
            pass
        
        # Add networkx for graph analysis (optional, useful for social network data)
        try:
            exec_globals['nx'] = __import__('networkx')
            exec_globals['networkx'] = __import__('networkx')
        except ImportError:
            # networkx is optional
            pass
        
        df_for_test = self.data.copy()
        # Apply missing code cleanup if the profile (or heuristic) says so
        from core.dataset_profile import missing_codes as _profile_mc
        _has_missing = bool(_profile_mc()) or self._is_dsl_dataset() or self._is_wvs_like_dataset()
        if _has_missing:
            try:
                used_cols = list(test_spec.get("columns", []) or [])
                df_for_test = self._apply_missing_code_cleanup(df_for_test, used_cols)
            except Exception:
                # Never fail execution due to optional cleaning.
                pass
        exec_globals['df'] = df_for_test
        exec_locals = {}
        
        # Final auto-fix pass before execution (safety net)
        # This ensures any missed fixes are caught before execution
        code = self._validate_generated_code(code)
        
        # Define timeout handler (only works on Unix systems)
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution timed out after {self.execution_timeout} seconds")
        
        # Set up timeout (only on Unix systems)
        use_timeout = hasattr(signal, 'SIGALRM')
        old_handler = None
        if use_timeout:
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.execution_timeout)
            except (ValueError, OSError):
                # Signal not available or already in use, disable timeout
                use_timeout = False
        
        try:
            # Suppress numpy warnings during execution to prevent RuntimeWarning clutter
            # We'll handle NaN values explicitly in validation
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
                warnings.filterwarnings('ignore', message='divide by zero')
                warnings.filterwarnings('ignore', message='invalid value encountered')
                
                # Execute the code in a controlled environment
                # The timeout alarm is already set and will cover both exec and function call
                exec(code, exec_globals, exec_locals)
                # Save executed code for artifacts (note: may differ from raw due to validation fixes)
                self.last_execution = {
                    "test_spec": dict(test_spec),
                    "executed_code": code,
                }
                
                # Try to get results from various possible return patterns
                results = None
                
                # Check if there's a function result (also within warnings context)
                if 'test_hypothesis' in exec_locals:
                    results = exec_locals['test_hypothesis'](exec_globals['df'])
                elif 'result' in exec_locals:
                    results = exec_locals['result']
                elif 'results' in exec_locals:
                    results = exec_locals['results']
                elif 'output' in exec_locals:
                    results = exec_locals['output']
                else:
                    raise ValueError(
                        "Code execution did not return results. "
                        "Expected function 'test_hypothesis(df)' or variables 'result', 'results', or 'output'. "
                        "Please ensure the generated code returns results in the expected format."
                    )
            
            # Validate and extract results
            if not isinstance(results, dict):
                raise ValueError(
                    f"Code execution returned invalid result type: {type(results)}. "
                    f"Expected dictionary with keys: effect_size, p_value, n_observations, coverage, confidence_interval."
                )

            # Normalize common LLM key aliases before validation.
            results = self._normalize_result_keys(results)

            # effect_size and p_value are required for scoring — raise so the retry/LLM-fix
            # mechanism gets an actionable message instead of silently using wrong defaults.
            missing_critical = [k for k in ("effect_size", "p_value") if k not in results]
            if missing_critical:
                got_keys = sorted(results.keys())
                raise ValueError(
                    f"test_hypothesis() result is missing required keys: {missing_critical}. "
                    f"Got keys: {got_keys}. "
                    f"The function MUST return a dict with AT MINIMUM: "
                    f"effect_size (float, magnitude of effect), p_value (float in [0,1]). "
                    f"Also include: n_observations (int), coverage (float in [0,1]), "
                    f"confidence_interval (tuple of 2 floats). "
                    f"Example: return {{'effect_size': r, 'p_value': p, "
                    f"'n_observations': len(df), 'coverage': 1.0, 'confidence_interval': (r-0.05, r+0.05)}}"
                )

            effect_size = results.get('effect_size')
            p_value = results.get('p_value')
            n_obs = results.get('n_observations')
            coverage = results.get('coverage')
            ci = results.get('confidence_interval')
            
            # Sanitize NaN, inf, and None values before validation so they cannot
            # cause divide-by-zero or invalid calculations
            def sanitize_value(val, default, min_val=None, max_val=None, name="value"):
                """Sanitize a value, replacing NaN/inf/None with safe defaults."""
                if val is None:
                    return default
                if not isinstance(val, (int, float)):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        return default
                # Check for NaN or inf
                if np.isnan(val) or np.isinf(val):
                    return default
                # Clamp to range if specified
                if min_val is not None and val < min_val:
                    return min_val
                if max_val is not None and val > max_val:
                    return max_val
                return val
            
            # Sanitize all values with appropriate defaults
            effect_size = sanitize_value(effect_size, 0.0, name="effect_size")
            # Clamp effect size: LLM code often returns raw OLS coefficients
            # instead of standardized effect sizes. Cap at [-5, 5].
            if abs(effect_size) > 5.0:
                effect_size = np.sign(effect_size) * min(abs(effect_size), 5.0)
            p_value = sanitize_value(p_value, 1.0, min_val=0.0, max_val=1.0, name="p_value")
            n_obs = sanitize_value(n_obs, len(self.data), min_val=0, name="n_observations")
            coverage = sanitize_value(coverage, 1.0, min_val=0.0, max_val=1.0, name="coverage")
            
            # Validate types (after sanitization, should always pass)
            if not isinstance(effect_size, (int, float)):
                raise ValueError(f"Invalid effect_size: {effect_size}. Must be a number.")
            
            if not isinstance(p_value, (int, float)) or p_value < 0 or p_value > 1:
                raise ValueError(f"Invalid p_value: {p_value}. Must be a number between 0 and 1.")
            
            if not isinstance(n_obs, (int, float)) or n_obs < 0:
                raise ValueError(f"Invalid n_observations: {n_obs}. Must be a non-negative number.")
            
            if not isinstance(coverage, (int, float)) or coverage < 0 or coverage > 1:
                raise ValueError(f"Invalid coverage: {coverage}. Must be a number between 0 and 1.")
            
            # Validate confidence interval
            if ci is None or not isinstance(ci, (list, tuple)) or len(ci) != 2:
                # Calculate default CI if missing
                ci = (max(0.0, float(effect_size) - 0.1), min(1.0, float(effect_size) + 0.1))
            else:
                # Sanitize CI values to handle NaN/inf
                ci_lower = sanitize_value(ci[0], float(effect_size) - 0.1, name="ci_lower")
                ci_upper = sanitize_value(ci[1], float(effect_size) + 0.1, name="ci_upper")
                # Ensure lower <= upper
                if ci_lower > ci_upper:
                    ci_lower, ci_upper = ci_upper, ci_lower
                ci = (ci_lower, ci_upper)
            
            return ExperimentResult(
                hypothesis_id=test_spec.get("hypothesis_id", "unknown"),
                effect_size=float(effect_size),
                p_value=float(p_value),
                confidence_interval=tuple(ci),
                n_observations=int(n_obs),
                coverage=float(coverage),
                replication_status="pending",
                raw_results=results
            )
        
        except SyntaxError as e:
            error_msg = (
                f"Code syntax error at line {e.lineno}: {e.msg}. "
                f"Text: {e.text if e.text else 'N/A'}"
            )
            raise ValueError(error_msg) from e
        except NameError as e:
            error_msg = (
                f"Undefined variable or function: {e.name}. "
                f"This usually means a library is not imported or a variable is misspelled. "
                f"Available libraries: pandas (pd), numpy (np), scipy.stats (stats)"
            )
            raise ValueError(error_msg) from e
        except ImportError as e:
            error_msg = (
                f"Import error: {e}. "
                f"Please ensure all required libraries are installed. "
                f"Required: pandas, numpy, scipy, statsmodels. Install with: pip install statsmodels"
            )
            raise RuntimeError(error_msg) from e
        except AttributeError as e:
            error_msg = (
                f"Attribute error: {e}. "
                f"This usually means trying to access a method/attribute that doesn't exist. "
                f"Check column names and data types."
            )
            raise ValueError(error_msg) from e
        except KeyError as e:
            error_msg = (
                f"Key error: {e}. "
                f"This usually means a column name doesn't exist in the data. "
                f"Available columns: {list(self.data.columns) if self.data is not None else 'N/A'}"
            )
            raise ValueError(error_msg) from e
        except TimeoutError as e:
            error_msg = (
                f"Code execution timed out after {self.execution_timeout} seconds. "
                f"This may indicate an infinite loop or inefficient code. "
                f"Error: {e}"
            )
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"Code execution failed: {type(e).__name__}: {e}. "
                f"Please check the generated code and data structure."
            )
            raise RuntimeError(error_msg) from e
        finally:
            # Always disable alarm if it was set
            if use_timeout:
                try:
                    signal.alarm(0)  # Disable alarm
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
                except (ValueError, OSError):
                    pass  # Ignore errors when cleaning up
    
    def process(self, hypothesis: Hypothesis) -> ExperimentResult:
        """Process: run experiment for hypothesis."""
        result = self.run_experiment(hypothesis)
        result.hypothesis_id = hypothesis.id
        return result

