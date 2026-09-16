"""Evaluator Agent - assesses statistical validity."""

import math
import re
from typing import Any, Dict, List, Optional, Tuple
from core.types import ExperimentResult, Evaluation
from agents.base_agent import BaseAgent


class Evaluator(BaseAgent):
    """Evaluator agent that assesses statistical validity of experiment results.

    The Evaluator performs statistical validation focusing on core validity metrics:
    - Validity score computation based on p-value and effect size, with sample-size-adaptive weights
    - Sample size adequacy checking (flags only, not in score)
    - Coverage checking (flags only, indicates generalizability not validity)
    - Replication status checking
    - Confidence interval validation
    - Warning flag collection for potential issues

    The validity score focuses on statistical evidence (p-value and effect size).
    Sample size and coverage are checked separately as flags because:
    - Sample size adequacy depends on test type, dataset size, and effect size
    - Coverage is about generalizability/usefulness, not statistical validity
    """
    
    def __init__(self):
        super().__init__("evaluator")
        self.min_p_value = self.get_config("min_p_value", 0.05)
        self.min_effect_size = self.get_config("min_effect_size", 0.1)
        self.require_replication = self.get_config("require_replication", True)
        self.min_sample_size = self.get_config("min_sample_size", 100)
        self.min_coverage = self.get_config("min_coverage", 0.05)
        # Hard floor: effects below this are substantively meaningless regardless
        # of p-value or sample size, so validity is capped.
        self.min_meaningful_effect_size = self.get_config("min_meaningful_effect_size", 0.02)
        # Sigmoid center for effect-size scoring.
        self.effect_sigmoid_center = self.get_config("effect_sigmoid_center", 0.12)

    def evaluate(self, result: ExperimentResult) -> Evaluation:
        """Evaluate statistical validity of an experiment result.
        
        Args:
            result: The experiment result to evaluate
            
        Returns:
            Evaluation object containing validity score, flags, and metadata
            
        Raises:
            ValueError: If result contains invalid statistical values
        """
        self._validate_result(result)
        
        validity_score = self._compute_validity_score(result)

        is_replicated = self._check_replication(result)

        # Replication penalty: findings that fail split-sample replication are
        # penalized because the effect may be unstable or overfit to a data
        # subset. The penalty is applied here (not in the Critic) because
        # the Critic reads validity_score for its accept/reject decisions.
        if self.require_replication and not is_replicated:
            validity_score *= 0.8  # 20% penalty for non-replicated findings

        ci_valid = self._validate_confidence_interval(result)

        flags = self._collect_flags(result, validity_score, is_replicated, ci_valid)

        return Evaluation(
            hypothesis_id=result.hypothesis_id,
            validity_score=validity_score,
            coverage=result.coverage,
            effect_size=result.effect_size,
            p_value=result.p_value,
            is_replicated=is_replicated,
            flags=flags
        )

    @staticmethod
    def _sentence_with_variable_labels(sentence: str, var_key: Optional[Dict[str, str]] = None) -> str:
        """Replace variable codes with CODE (label) for readability."""
        if not var_key:
            return sentence
        out = sentence
        for name in sorted(var_key.keys(), key=len, reverse=True):
            out = out.replace(name, f"{name} ({var_key[name]})")
        return out

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Make codebook-ish labels read like plain English."""
        t = (label or "").strip()
        if not t:
            return t
        t = re.sub(r"^How much you trust:\s*", "Trust in ", t, flags=re.IGNORECASE)
        t = re.sub(r"^Trust:\s*", "Trust in ", t, flags=re.IGNORECASE)
        t = re.sub(r"^Confidence:\s*", "Confidence in ", t, flags=re.IGNORECASE)
        t = re.sub(r"^Satisfaction with\s+", "Satisfaction with ", t, flags=re.IGNORECASE)
        # Light casing cleanup for common WVS label patterns.
        t = re.sub(r"\bYour\b", "your", t)
        t = re.sub(r"\bPeople\b", "people", t)
        return t.strip()

    @classmethod
    def _human_var(cls, name: str, var_key: Optional[Dict[str, str]] = None) -> str:
        """
        Convert a column name into a human-readable phrase.
        Prefer label text; drop raw codes like Q58 when possible.
        """
        n = (name or "").strip()
        if not n:
            return n
        # Friendly names for columns: prefer profile, fall back to legacy WVS defaults
        from core.dataset_profile import column_labels as _profile_labels
        meta = _profile_labels() or {
            "B_COUNTRY": "country",
            "B_COUNTRY_ALPHA": "country",
            "A_YEAR": "survey year",
            "year": "survey year",
            "H_URBRURAL": "urban vs rural residence",
            "FW_START": "fieldwork start date",
            "FW_END": "fieldwork end date",
            "survey_id": "survey (wave) id",
            "sampling_weight": "survey weights",
            "PWGHT": "survey weights",
            "W_WEIGHT": "survey weights",
        }
        if n in meta:
            return meta[n]
        if var_key and n in var_key and str(var_key[n]).strip():
            return cls._normalize_label(str(var_key[n]))
        return n

    @classmethod
    def _human_list(cls, csv: str, var_key: Optional[Dict[str, str]] = None) -> str:
        parts = [p.strip() for p in (csv or "").split(",") if p.strip()]
        return ", ".join([cls._human_var(p, var_key) for p in parts])

    @staticmethod
    def _format_finding_suffix(
        evaluation: Optional[Evaluation],
        result: Optional[ExperimentResult],
    ) -> str:
        """Build compact statistics suffix for human-readable findings."""
        parts: List[str] = []
        if evaluation is not None and evaluation.p_value is not None:
            try:
                pv = float(evaluation.p_value)
                if pv == 0.0:
                    parts.append("p<1e-300")
                else:
                    parts.append(f"p={pv:.4g}")
            except Exception:
                pass
        if result is not None and result.effect_size is not None:
            try:
                parts.append(f"effect={float(result.effect_size):.4g}")
            except Exception:
                pass
        if result is not None and result.n_observations is not None:
            try:
                parts.append(f"n={int(result.n_observations)}")
            except Exception:
                pass
        return f" ({', '.join(parts)})" if parts else ""

    @staticmethod
    def _get_sign_semantics(result: Optional[ExperimentResult]) -> Dict[str, Any]:
        if result is None:
            return {}
        if getattr(result, "sign_semantics", None):
            return result.sign_semantics
        raw = getattr(result, "raw_results", {}) or {}
        return raw.get("sign_semantics", {}) or {}

    @classmethod
    def _has_direction_caution(cls, result: Optional[ExperimentResult]) -> bool:
        sign_semantics = cls._get_sign_semantics(result)
        return bool(sign_semantics.get("direction_caution"))

    def render_finding_sentence(
        self,
        hypothesis_sentence: str,
        result: Optional[ExperimentResult],
        evaluation: Optional[Evaluation] = None,
        var_key: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Translate a DSL-like hypothesis sentence into a readable finding sentence.
        If parsing fails, return a label-substituted fallback sentence.
        """
        # Fallback: at least expand codes into labels.
        labeled = self._sentence_with_variable_labels(hypothesis_sentence, var_key)
        s = (hypothesis_sentence or "").strip()
        if not s:
            return labeled
        s_low = s.lower()

        controls = ""
        weights = ""
        if " controlling for " in s_low:
            idx = s_low.find(" controlling for ")
            tail = s[idx + len(" controlling for "):].strip()
            wt_match = re.search(r"\bweights\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", tail)
            if wt_match:
                weights = wt_match.group(1).strip()
                tail = re.sub(r"\bweights\s*=\s*[A-Za-z_][A-Za-z0-9_]*", "", tail).strip(" ,")
            controls = tail.strip()
        else:
            wt_match = re.search(r"\bweights\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", s)
            if wt_match:
                weights = wt_match.group(1).strip()

        body = s
        if " controlling for " in s_low:
            body = s[:s_low.find(" controlling for ")].strip()
        body = re.sub(r"\bweights\s*=\s*[A-Za-z_][A-Za-z0-9_]*", "", body).strip(" ,")

        suffix = self._format_finding_suffix(evaluation, result)
        controls_text = (
            f", adjusting for {self._human_list(controls, var_key)}"
            if controls
            else ""
        )
        weights_text = (
            f", weighted by {self._human_var(weights, var_key)}"
            if weights
            else ""
        )

        assoc = re.match(r"^\s*assoc\(\s*([^,]+)\s*,\s*([^)]+)\)\s*$", body, flags=re.IGNORECASE)
        if assoc:
            x_raw = assoc.group(1).strip()
            y_raw = assoc.group(2).strip()
            x = self._human_var(x_raw, var_key)
            y = self._human_var(y_raw, var_key)
            direction = ""
            if not self._has_direction_caution(result):
                try:
                    if result is not None and isinstance(result.effect_size, (int, float)):
                        if result.effect_size > 0:
                            direction = "positively "
                        elif result.effect_size < 0:
                            direction = "negatively "
                except Exception:
                    direction = ""
            return f"{x} is {direction}associated with {y}{controls_text}{weights_text}{suffix}."

        diff = re.match(r"^\s*diff\(\s*([^,]+)\s*,\s*by\s*=\s*([^)]+)\)\s*$", body, flags=re.IGNORECASE)
        if diff:
            y_raw = diff.group(1).strip()
            g_raw = diff.group(2).strip()
            y = self._human_var(y_raw, var_key)
            if g_raw == "H_URBRURAL":
                group_text = "between urban and rural respondents"
            elif g_raw in {"B_COUNTRY", "B_COUNTRY_ALPHA"}:
                group_text = "across countries"
            elif g_raw in {"A_YEAR", "year"}:
                group_text = "across survey years"
            else:
                group_text = f"across groups defined by {self._human_var(g_raw, var_key)}"
            return f"{y} differs {group_text}{controls_text}{weights_text}{suffix}."

        inter = re.match(r"^\s*interact\(\s*([^*]+)\*\s*([^-]+)\s*->\s*([^)]+)\)\s*$", body, flags=re.IGNORECASE)
        if inter:
            x = self._human_var(inter.group(1).strip(), var_key)
            z_raw = inter.group(2).strip()
            y = self._human_var(inter.group(3).strip(), var_key)
            if z_raw in {"B_COUNTRY", "B_COUNTRY_ALPHA"}:
                z = "country"
            elif z_raw == "H_URBRURAL":
                z = "urban vs rural residence"
            elif z_raw in {"A_YEAR", "year"}:
                z = "survey year"
            else:
                z = self._human_var(z_raw, var_key)
            return f"The association between {x} and {y} varies by {z}{controls_text}{weights_text}{suffix}."

        hetero = re.match(r"^\s*heterogeneity\(\s*([^~]+)\s*~\s*([^|]+)\s*\|\s*([^)]+)\)\s*$", body, flags=re.IGNORECASE)
        if hetero:
            y = self._human_var(hetero.group(1).strip(), var_key)
            x = self._human_var(hetero.group(2).strip(), var_key)
            g_raw = hetero.group(3).strip()
            if g_raw in {"B_COUNTRY", "B_COUNTRY_ALPHA"}:
                g = "country"
            elif g_raw == "H_URBRURAL":
                g = "urban vs rural residence"
            elif g_raw in {"A_YEAR", "year"}:
                g = "survey year"
            else:
                g = self._human_var(g_raw, var_key)
            return f"The relationship between {x} and {y} varies by {g}{controls_text}{weights_text}{suffix}."

        fallback = labeled.rstrip(".")
        return f"{fallback}{suffix}."
    
    def _validate_result(self, result: ExperimentResult) -> None:
        """Validate that the experiment result contains valid statistical values.
        
        Args:
            result: The experiment result to validate
            
        Raises:
            ValueError: If any statistical values are invalid
        """
        if result is None:
            raise ValueError("ExperimentResult cannot be None")
        
        # Validate p-value
        if not isinstance(result.p_value, (int, float)):
            raise ValueError(f"Invalid p_value type: {type(result.p_value)}. Must be numeric.")
        if result.p_value < 0 or result.p_value > 1:
            raise ValueError(f"Invalid p_value: {result.p_value}. Must be between 0 and 1.")
        
        # Validate effect size
        if not isinstance(result.effect_size, (int, float)):
            raise ValueError(f"Invalid effect_size type: {type(result.effect_size)}. Must be numeric.")
        if not (-10 <= result.effect_size <= 10):  # Clamp instead of rejecting
            result.effect_size = max(-5.0, min(5.0, result.effect_size))
        
        # Validate sample size
        if not isinstance(result.n_observations, (int, float)):
            raise ValueError(f"Invalid n_observations type: {type(result.n_observations)}. Must be numeric.")
        if result.n_observations < 0:
            raise ValueError(f"Invalid n_observations: {result.n_observations}. Must be non-negative.")
        
        # Validate coverage
        if not isinstance(result.coverage, (int, float)):
            raise ValueError(f"Invalid coverage type: {type(result.coverage)}. Must be numeric.")
        if result.coverage < 0 or result.coverage > 1:
            raise ValueError(f"Invalid coverage: {result.coverage}. Must be between 0 and 1.")
        
        # Validate confidence interval
        if result.confidence_interval is not None:
            if not isinstance(result.confidence_interval, (list, tuple)) or len(result.confidence_interval) != 2:
                raise ValueError(
                    f"Invalid confidence_interval: {result.confidence_interval}. "
                    f"Must be a tuple/list of two numbers."
                )
            lower, upper = result.confidence_interval
            if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
                raise ValueError("Confidence interval bounds must be numeric.")
            if lower > upper:
                raise ValueError(f"Invalid confidence interval: lower ({lower}) > upper ({upper})")
    
    def _compute_validity_score(self, result: ExperimentResult) -> float:
        """Compute validity score (0-1) based on statistical evidence.

        Uses N-adaptive weighting: with large samples almost any non-zero
        effect achieves p < 0.001, so p-value provides no discrimination.
        Weight shifts toward effect size as N grows:
          N ≈ 100  → p 78%, effect 22%  (small-sample: p-value informative)
          N ≈ 1 k  → p 55%, effect 45%
          N ≈ 10 k → p 32%, effect 68%
          N ≈ 97 k → p 30%, effect 70%  (large-sample: effect size dominates)

        The effect-size sigmoid is centered at 0.12 (social-science scale,
        where r ≈ 0.10 is a real finding) with steepness 8 for sharper
        discrimination around the threshold.

        A hard floor caps validity at 0.5 when |effect_size| is below
        min_meaningful_effect_size (default 0.02), and p >= min_p_value caps
        it at 0.3.

        Args:
            result: The experiment result to score

        Returns:
            Validity score between 0 and 1
        """
        p_score = self._score_p_value(result.p_value)

        effect_magnitude = abs(result.effect_size)
        # Sigmoid centered at configurable point, steepness 8
        effect_score = 1.0 / (1.0 + math.exp(-8 * (effect_magnitude - self.effect_sigmoid_center)))

        # --- N-adaptive weight rebalancing ---
        n = max(1, result.n_observations)
        p_weight = 0.3 + 0.5 / (1.0 + (n / 1000.0) ** 1.5)
        e_weight = 1.0 - p_weight

        total_score = p_score * p_weight + effect_score * e_weight

        # --- Hard floor: trivial effects cannot achieve high validity ---
        # Interaction and heterogeneity tests with N > 1000 use a floor of 0.03.
        min_effect = self.min_meaningful_effect_size
        dsl_family = getattr(result, 'dsl_family', None) or ''
        if dsl_family in ('interact', 'heterogeneity') and result.n_observations > 1000:
            min_effect = 0.03
        if effect_magnitude < min_effect:
            total_score = min(total_score, 0.5)

        # --- Hard cap: p >= p_min cannot be accepted ---
        # Validity is capped below the Critic's 0.4 validity floor, so a
        # non-significant result can be refined or rejected but never stored.
        if result.p_value >= self.min_p_value:
            total_score = min(total_score, 0.3)

        warning_severity = getattr(result, "warning_severity", None)
        if warning_severity == "review":
            total_score *= 0.95
        elif warning_severity == "material":
            total_score *= 0.8
        return min(1.0, max(0.0, total_score))
    
    def _score_p_value(self, p_value: float) -> float:
        """Score p-value on a 0-1 scale (lower p-values get higher scores).

        Piecewise linear up to p = 0.1, exponential decay above.

        Args:
            p_value: The p-value to score

        Returns:
            Score between 0 and 1
        """
        if p_value < 0.001:
            return 1.0
        elif p_value < 0.01:
            # Linear interpolation between 0.001 and 0.01
            # p=0.001 -> 1.0, p=0.01 -> 0.95
            return 1.0 - (p_value - 0.001) / (0.01 - 0.001) * (1.0 - 0.95)
        elif p_value <= 0.05:
            # Linear interpolation between 0.01 and 0.05
            # p=0.01 -> 0.95, p=0.05 -> 0.85
            return 0.95 - (p_value - 0.01) / (0.05 - 0.01) * (0.95 - 0.85)
        elif p_value < 0.1:
            # Linear decrease between 0.05 and 0.1
            # p=0.05 -> 0.75, p=0.1 -> 0.65
            return 0.75 + (0.05 - p_value) / (0.1 - 0.05) * 0.1
        else:
            # Exponential decay for p >= 0.1, starting at 0.75
            return max(0.0, 0.75 * math.exp(-3 * (p_value - 0.1)))
    
    def _check_sample_size_adequacy(self, result: ExperimentResult) -> Tuple[bool, Optional[str]]:
        """Check if sample size is adequate for the statistical test.
        
        This is a simple heuristic check. Adequate sample size depends on:
        - Test type (t-test, chi-square, correlation, etc.)
        - Effect size (smaller effects need larger samples)
        - Dataset size (relative to total data)
        
        For now, we use general heuristics and flag potential issues.
        Future: Could infer test type from raw_results or use power analysis.
        
        Args:
            result: The experiment result to check
            
        Returns:
            Tuple of (is_adequate, warning_message)
        """
        n = result.n_observations
        effect_magnitude = abs(result.effect_size)
        
        # Very small samples are always problematic
        if n < 10:
            return False, "extremely_small_sample_size"
        if n < 30:
            return False, "very_small_sample_size"
        
        # For small effects, need larger samples
        # Small effect (< 0.1) typically needs n > 100 for reasonable power
        if effect_magnitude < 0.1 and n < 100:
            return False, "small_sample_for_small_effect"
        
        # For medium effects (0.1-0.3), n >= 30 is usually okay
        # For large effects (> 0.3), even n >= 30 can work
        
        # If we get here, sample size is probably adequate
        # But we still flag if it's borderline
        if n < 100:
            return True, "moderate_sample_size"  # Adequate but not ideal
        else:
            return True, None  # Good sample size
    
    def _check_replication(self, result: ExperimentResult) -> bool:
        """Check if result is replicated.
        
        Currently checks the replication_status field. Future enhancements could:
        - Check against historical results in Discovery Memory
        - Perform cross-validation
        - Check for consistency with related insights
        
        Args:
            result: The experiment result to check
            
        Returns:
            True if replicated, False otherwise
        """
        if result.replication_status == "replicated":
            return True
        elif result.replication_status == "not_replicated":
            return False
        else:
            # Default: assume not replicated if status is pending/unknown
            # This is conservative - we require explicit replication confirmation
            return False
    
    def _validate_confidence_interval(self, result: ExperimentResult) -> bool:
        """Validate that the confidence interval is reasonable.
        
        Checks:
        - CI contains the effect size (or is close to it)
        - CI width is reasonable (not too wide, indicating uncertainty)
        - CI bounds are in reasonable range
        
        Args:
            result: The experiment result to validate
            
        Returns:
            True if CI is valid, False otherwise
        """
        if result.confidence_interval is None:
            return False
        
        lower, upper = result.confidence_interval
        
        # Check if effect size is within or near the CI
        # Allow some tolerance for rounding errors
        tolerance = abs(upper - lower) * 0.1
        effect_in_ci = (lower - tolerance <= result.effect_size <= upper + tolerance)
        
        # Check if CI width is reasonable (not too wide)
        ci_width = abs(upper - lower)
        # For typical effect sizes in [-1, 1], CI width > 2 is suspicious
        reasonable_width = ci_width <= 2.0
        
        return effect_in_ci and reasonable_width
    
    def _collect_flags(self, result: ExperimentResult, validity_score: float, 
                     is_replicated: bool, ci_valid: bool) -> List[str]:
        """Collect warning flags about potential statistical issues.
        
        Flags indicate areas where the statistical evidence may be weak or
        where additional validation might be needed. These are warnings, not
        part of the validity score calculation.
        
        Args:
            result: The experiment result
            validity_score: The computed validity score
            is_replicated: Whether the result is replicated
            ci_valid: Whether the confidence interval is valid
            
        Returns:
            List of flag strings describing potential issues
        """
        flags = []
        
        # P-value flags (statistical significance)
        if result.p_value > self.min_p_value:
            flags.append("high_p_value")
        if result.p_value > 0.1:
            flags.append("very_high_p_value")
        if result.p_value < 0.001:
            flags.append("very_low_p_value")  # Positive flag, but worth noting
        
        # Effect size flags (practical significance)
        if abs(result.effect_size) < self.min_effect_size:
            flags.append("small_effect_size")
        if abs(result.effect_size) < 0.05:
            flags.append("very_small_effect_size")
        if abs(result.effect_size) > 1.0:
            flags.append("very_large_effect_size")  # May indicate data issues
        
        # Sample size flags (using adequacy check)
        is_adequate, sample_warning = self._check_sample_size_adequacy(result)
        if not is_adequate:
            flags.append(sample_warning)
        elif sample_warning:
            flags.append(sample_warning)  # Moderate sample size
        
        # Coverage flags (generalizability/usefulness, not validity)
        # Low coverage means result applies to small portion of data
        if result.coverage < self.min_coverage:
            flags.append("low_coverage")
        if result.coverage < 0.01:
            flags.append("very_low_coverage")
        
        # Replication flags
        if not is_replicated and self.require_replication:
            flags.append("not_replicated")

        # Result-contract / warning flags
        if getattr(result, "summary_estimand", None) == "contrast":
            raw = getattr(result, "raw_results", {}) or {}
            if raw.get("omnibus"):
                flags.append("contrast_summary_with_omnibus_available")
        if getattr(result, "warning_severity", None) == "review":
            flags.append("review_level_warning")
        if getattr(result, "warning_severity", None) == "material":
            flags.append("material_execution_warning")
        for review_flag in getattr(result, "review_flags", []) or []:
            if review_flag not in flags:
                flags.append(review_flag)
        if self._has_direction_caution(result):
            flags.append("direction_requires_codebook_interpretation")
        
        # Confidence interval flags
        if not ci_valid:
            flags.append("invalid_confidence_interval")
        elif result.confidence_interval is not None:
            lower, upper = result.confidence_interval
            ci_width = abs(upper - lower)
            if ci_width > 1.0:
                flags.append("wide_confidence_interval")
        
        # Overall validity flags
        if validity_score < 0.5:
            flags.append("low_validity")
        if validity_score < 0.3:
            flags.append("very_low_validity")
        
        # Statistical power concerns
        # Low power when p-value is borderline and sample size is small
        if (self.min_p_value < result.p_value <= 0.1 and 
            result.n_observations < 100):
            flags.append("possible_low_power")
        
        return flags
    
    def process(self, result: ExperimentResult) -> Evaluation:
        """Process: evaluate experiment result.
        
        This is the standard interface method for BaseAgent compatibility.
        
        Args:
            result: The experiment result to evaluate
            
        Returns:
            Evaluation object with validity assessment
        """
        return self.evaluate(result)





