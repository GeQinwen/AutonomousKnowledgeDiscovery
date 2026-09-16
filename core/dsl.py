"""Deterministic DSL parser and canonical statistical executor for structured hypotheses.

Supports the four DSL templates across any tabular dataset:
  assoc(X, Y)                [controlling for C1, C2] [weights=W]
  diff(Y, by=G)              [controlling for C1, C2] [weights=W]
  interact(X * Z -> Y)       [controlling for C1, C2] [weights=W]
  heterogeneity(Y ~ X | G)   [controlling for C1, C2] [weights=W]

Guarantees:
  - Hard-fail on unresolved control/group/weight terms (no silent dropping)
  - Each family maps to a fixed statistical recipe (no LLM inference of test type)
  - Result validation is strict (no sanitization of invalid values)
"""

import re
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Set, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Optional heavy dependency — needed only for regression families
# ---------------------------------------------------------------------------
try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    _HAS_STATSMODELS = True
except ImportError:
    smf = None  # type: ignore[assignment]
    sm = None  # type: ignore[assignment]
    _HAS_STATSMODELS = False


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_known_categorical() -> Set[str]:
    from core.dataset_profile import known_categorical as _pk
    _p = _pk()
    return set(_p) if _p else {"B_COUNTRY", "B_COUNTRY_ALPHA", "H_URBRURAL", "country", "country_code"}

def _resolve_known_weights() -> Set[str]:
    from core.dataset_profile import weight_columns as _pw
    _p = _pw()
    return set(_p) if _p else {"sampling_weight", "W_WEIGHT", "PWGHT"}

def _resolve_missing_codes() -> Set[int]:
    from core.dataset_profile import missing_codes as _pm, get_profile as _gp
    _p = _pm()
    # If a profile is loaded, trust its missing_codes (even if empty).
    # Only fall back to WVS defaults when no profile is active.
    if _p:
        return _p
    if _gp():
        return set()
    return {-1, -2, -3, -4, -5}

# Module-level constants — resolved lazily on first DSL call.
# For backward compatibility these are still importable by name.
KNOWN_CATEGORICAL: Set[str] = {
    "B_COUNTRY", "B_COUNTRY_ALPHA", "H_URBRURAL",
    "country", "country_code",
}

KNOWN_WEIGHTS: Set[str] = {"sampling_weight", "W_WEIGHT", "PWGHT"}

# Columns that must never serve as predictor (X) or outcome (Y).
META_ONLY: Set[str] = KNOWN_WEIGHTS | {"survey_id"}


def _resolve_meta_only() -> Set[str]:
    """Profile-driven set of columns that must never be X or Y."""
    from core.dataset_profile import id_columns as _pi, weight_columns as _pw
    w = set(_pw()) if _pw() else {"sampling_weight", "W_WEIGHT", "PWGHT"}
    i = set(_pi()) if _pi() else {"survey_id"}
    return w | i


def _resolve_demographic_synonyms() -> Dict[str, str]:
    """Profile-driven demographic synonyms (natural language -> column name)."""
    from core.dataset_profile import demographic_synonyms as _pds
    p = _pds()
    return p if p else dict(_DEMOGRAPHIC_SYNONYMS)


def _resolve_sign_semantics() -> Dict[str, Dict[str, str]]:
    """Profile-driven sign semantics for column interpretation."""
    from core.dataset_profile import sign_semantics as _pss
    p = _pss()
    return p if p else dict(_KNOWN_SIGN_SEMANTICS)

# Resolved at call time via _resolve_missing_codes(); kept for backward compat.
WVS_MISSING_CODES: Set[int] = {-1, -2, -3, -4, -5}

_MIN_N_CORRELATION = 10
_MIN_N_REGRESSION = 20
_MIN_N_PER_GROUP = 5
_MAX_CATEGORICAL_CARDINALITY = 80  # Max unique values for a categorical in regression formulas
_MIN_LEVEL_CELL = 30  # Rows a categorical level needs before it earns its own dummy
_MAX_GROUPING_CARDINALITY = 100   # Max unique values for a grouping variable in diff()/heterogeneity()

# Common demographic synonyms that LLMs tend to output as natural language.
# Maps lowercased synonym → WVS variable code.
_DEMOGRAPHIC_SYNONYMS: Dict[str, str] = {
    "gender": "Q260",
    "sex": "Q260",
    "age": "Q262",
    "education": "Q275",
    "education level": "Q275",
    "educational level": "Q275",
    "highest education": "Q275",
    "employment": "Q279",
    "employment status": "Q279",
    "income": "Q288",
    "income level": "Q288",
    "household income": "Q288",
    "scale of incomes": "Q288",
    "marital status": "Q273",
    "number of children": "Q274",
    "children": "Q274",
    "social class": "Q287",
    "religion": "Q289",
    "religious denomination": "Q289",
    "ethnic group": "Q290",
    "ethnicity": "Q290",
    "urban rural": "H_URBRURAL",
    "urban-rural": "H_URBRURAL",
    "country": "B_COUNTRY",
    "year": "A_YEAR",
    "survey year": "A_YEAR",
}

_KNOWN_SIGN_SEMANTICS: Dict[str, Dict[str, str]] = {
    "Q46": {
        "direction": "higher_is_more_negative",
        "meaning": "Larger values mean less happiness.",
    },
    "Q49": {
        "direction": "higher_is_more_positive",
        "meaning": "Larger values mean greater life satisfaction.",
    },
    "Q50": {
        "direction": "higher_is_more_positive",
        "meaning": "Larger values mean greater financial satisfaction.",
    },
    "Q58": {
        "direction": "higher_is_more_negative",
        "meaning": "Larger values mean less trust in family.",
    },
    "Q62": {
        "direction": "higher_is_more_negative",
        "meaning": "Larger values mean less trust in people you know personally.",
    },
    "Q71": {
        "direction": "higher_is_more_negative",
        "meaning": "Larger values mean less confidence in government.",
    },
    "H_URBRURAL": {
        "direction": "coded_category",
        "meaning": "Category code where 1=urban and 2=rural.",
    },
    "B_COUNTRY": {
        "direction": "coded_category",
        "meaning": "Country code used for grouping, not numeric magnitude.",
    },
    "B_COUNTRY_ALPHA": {
        "direction": "coded_category",
        "meaning": "Country code used for grouping, not numeric magnitude.",
    },
    "A_YEAR": {
        "direction": "coded_time",
        "meaning": "Larger values indicate later survey years.",
    },
    "sampling_weight": {
        "direction": "weight",
        "meaning": "Survey weight for exploratory weighted estimation.",
    },
    "PWGHT": {
        "direction": "weight",
        "meaning": "Survey weight for exploratory weighted estimation.",
    },
    "W_WEIGHT": {
        "direction": "weight",
        "meaning": "Survey weight for exploratory weighted estimation.",
    },
}

_MATERIAL_WARNING_PATTERNS: Tuple[str, ...] = (
    r"covariance of constraints does not have full rank",
    r"singular",
    r"divide by zero",
    r"invalid value encountered",
    r"precision loss",
    r"perfect separation",
)

_REVIEW_WARNING_PATTERNS: Tuple[str, ...] = (
    r"kurtosistest",
    r"omni_normtest",
    r"condition number",
    r"collinearity",
    r"convergence",
)


# ═══════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class DSLParseError(Exception):
    """A DSL hypothesis could not be deterministically parsed or validated."""


class DSLExecutionError(Exception):
    """Deterministic execution of a DSL spec failed (data issue, not code bug)."""


# ═══════════════════════════════════════════════════════════════════════════
# Typed intermediate representation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DSLTestSpec:
    """Strongly-typed spec produced by the deterministic DSL parser."""
    family: str                                   # assoc | diff | interact | heterogeneity
    x: Optional[str] = None                       # primary predictor
    y: Optional[str] = None                       # outcome
    group: Optional[str] = None                   # grouping variable (diff, heterogeneity)
    moderator: Optional[str] = None               # interaction moderator (interact)
    controls: List[str] = field(default_factory=list)
    weight: Optional[str] = None
    unresolved_terms: List[str] = field(default_factory=list)
    raw_sentence: str = ""

    @property
    def all_columns(self) -> List[str]:
        """Ordered, deduplicated list of every resolved column name."""
        seen: Set[str] = set()
        out: List[str] = []
        for c in [self.x, self.y, self.group, self.moderator, self.weight] + self.controls:
            if c is not None and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "x": self.x,
            "y": self.y,
            "group": self.group,
            "moderator": self.moderator,
            "controls": list(self.controls),
            "weight": self.weight,
            "unresolved_terms": list(self.unresolved_terms),
            "raw_sentence": self.raw_sentence,
        }


# Backward-compatibility alias
WVSTestSpec = DSLTestSpec


# ═══════════════════════════════════════════════════════════════════════════
# 1. DSL detection
# ═══════════════════════════════════════════════════════════════════════════

def is_dsl_hypothesis(sentence: str) -> bool:
    """Return True if *sentence* looks like one of the four DSL templates."""
    s = (sentence or "").strip()
    if not s:
        return False
    s_low = s.lower()
    prefixes = ("assoc(", "diff(", "interact(", "heterogeneity(")
    if not any(s_low.startswith(p) for p in prefixes):
        return False
    body = s_low.split(" controlling for ", 1)[0].strip()
    # Minimal structural sanity per family
    if body.startswith("assoc(") and "|" in body:
        return False
    if body.startswith("diff(") and "by=" not in body:
        return False
    if body.startswith("interact(") and "->" not in body:
        return False
    if body.startswith("heterogeneity(") and ("~" not in body or "|" not in body):
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 2. Deterministic parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_dsl_hypothesis(
    sentence: str,
    valid_columns: Set[str],
    label_aliases: Optional[Dict[str, str]] = None,
) -> DSLTestSpec:
    """Parse a DSL hypothesis sentence into a strongly-typed DSLTestSpec.

    Args:
        sentence: The raw DSL string from the generator.
        valid_columns: Set of actual DataFrame column names.
        label_aliases: Optional mapping of lowercased label/synonym → var_name,
            built by ``build_label_alias_map``. Enables resolution of
            natural-language control terms like "age" → "Q262".

    Raises DSLParseError if the sentence cannot be parsed, any column in the
    DSL body is unknown, or any control term cannot be resolved.
    """
    s = (sentence or "").strip()
    if not s:
        raise DSLParseError("Empty hypothesis sentence")

    # Pre-normalize: the LLM sometimes places "controlling for" and "weights="
    # INSIDE the parentheses (e.g., "assoc(X, Y, controlling for C1, weights=W)").
    # Move them outside so the parser can split correctly.
    if re.search(r",\s*controlling\s+for\b", s, flags=re.IGNORECASE):
        s = re.sub(
            r",\s*(controlling\s+for\b)",
            r") \1",
            s,
            count=1,
            flags=re.IGNORECASE,
        )
        # The original closing paren is now orphaned at the end — remove it
        s = s.rstrip().rstrip(")")

    lower = s.lower()
    body = s
    tail = ""

    if " controlling for " in lower:
        idx = lower.index(" controlling for ")
        body = s[:idx].strip()
        tail = s[idx + len(" controlling for "):].strip()

    # --- extract weights=COL from tail (preferred) or body end ---
    weight: Optional[str] = None
    weight_re = re.compile(r"\bweights\s*=\s*([A-Za-z_]\w*)")
    wm = weight_re.search(tail) if tail else weight_re.search(body)
    if wm:
        w_candidate = wm.group(1)
        weight = _resolve_column(w_candidate, valid_columns, label_aliases)
        if weight is None:
            raise DSLParseError(
                f"Weight column '{w_candidate}' not found in dataset. "
                f"Closest columns: {_suggest_columns(w_candidate, valid_columns)}"
            )
        # Remove the weights=... token from whichever string it was in
        if tail and weight_re.search(tail):
            tail = weight_re.sub("", tail).strip(" ,")
        else:
            body = weight_re.sub("", body).strip(" ,")

    # --- parse controls from tail ---
    controls: List[str] = []
    unresolved: List[str] = []
    if tail:
        # Clean: LLM sometimes outputs "income | weights=W" or "Q61 |" — strip
        # stray pipe chars and brackets that leak from DSL/prompt formatting.
        tail = re.sub(r"\s*\|\s*", " ", tail).strip()
        tail = tail.strip("[]")
        for term in (t.strip() for t in tail.split(",") if t.strip()):
            resolved = _resolve_column(term, valid_columns, label_aliases)
            if resolved is not None:
                controls.append(resolved)
            else:
                unresolved.append(term)

    # --- parse DSL body ---
    spec = _parse_body(body, valid_columns, label_aliases)
    spec.controls = controls
    spec.weight = weight
    spec.unresolved_terms = unresolved
    spec.raw_sentence = sentence

    # ── hard-fail on unresolved terms ──
    if unresolved:
        raise DSLParseError(
            f"Unresolved control terms: {unresolved}. "
            f"These could not be mapped to valid column names. "
            f"Suggestion: {_suggest_columns(unresolved[0], valid_columns)}"
        )

    # ── semantic guardrails ──
    if weight and weight in controls:
        raise DSLParseError(f"Weight column '{weight}' must not also appear as a control variable")
    _meta = _resolve_meta_only()
    for col, role in [(spec.x, "predictor (X)"), (spec.y, "outcome (Y)")]:
        if col and col in _meta:
            raise DSLParseError(f"Column '{col}' is a weight/meta column and cannot be used as {role}")

    return spec


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def build_label_alias_map(catalog_path: str) -> Dict[str, str]:
    """Build a lowercased-label → var_name alias map from a JSONL variable catalog.

    The map includes:
      - Each card's label (lowercased) → var_name
      - Leading keyword tokens of long labels  (e.g. "employment status" from
        "Employment status")
      - Hardcoded demographic synonyms (``_DEMOGRAPHIC_SYNONYMS``) filtered to
        var_names that actually appear in the catalog.

    Returns an empty dict if the catalog file does not exist.
    """
    import json
    from pathlib import Path

    p = Path(catalog_path)
    if not p.exists():
        return {}

    aliases: Dict[str, str] = {}
    catalog_vars: Set[str] = set()

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            var = obj.get("var_name", "").strip()
            label = obj.get("label", "").strip()
            if not var:
                continue
            catalog_vars.add(var)
            if label:
                key = label.lower()
                aliases[key] = var
                # Also index the first N words as a shorter alias if label is long
                words = key.split()
                if len(words) >= 2:
                    for length in (2, 3):
                        short = " ".join(words[:length])
                        if short not in aliases:
                            aliases[short] = var

    # Merge demographic synonyms (only if the target var exists in catalog)
    for syn, var in _resolve_demographic_synonyms().items():
        if var in catalog_vars and syn not in aliases:
            aliases[syn] = var

    return aliases


def _resolve_column(
    term: str,
    valid_columns: Set[str],
    label_aliases: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Resolve a term to a valid column name.

    Resolution order:
      1. Exact match against valid_columns
      2. Case-insensitive match against valid_columns
      3. Strip parenthesised annotation, re-try 1 & 2
      4. Lookup in label_aliases (catalog label / synonym → var_name)
      5. Substring containment against label_aliases keys
    """
    t = term.strip()
    if not t:
        return None
    # 1. exact
    if t in valid_columns:
        return t
    # 2. case-insensitive
    t_low = t.lower()
    for col in valid_columns:
        if col.lower() == t_low:
            return col

    # 3. Strip parenthesised annotation (e.g. "Q49 (Satisfaction with your life)" → "Q49")
    bare = re.sub(r"\s*\(.*?\)\s*$", "", t).strip()
    if bare and bare != t:
        if bare in valid_columns:
            return bare
        bare_low = bare.lower()
        for col in valid_columns:
            if col.lower() == bare_low:
                return col

    # 4. Label alias exact lookup
    if label_aliases:
        hit = label_aliases.get(t_low)
        if hit and hit in valid_columns:
            return hit
        # Also try the bare (annotation-stripped) form
        if bare and bare != t:
            hit = label_aliases.get(bare.lower())
            if hit and hit in valid_columns:
                return hit

    # 5. Substring containment against alias keys (short term in long label)
    if label_aliases and len(t_low) >= 3:
        for alias_key, var in label_aliases.items():
            if var not in valid_columns:
                continue
            if t_low in alias_key or alias_key in t_low:
                return var

    return None


def _suggest_columns(term: str, valid_columns: Set[str], n: int = 5) -> List[str]:
    """Return up to *n* valid column names with some string overlap to *term*."""
    t_low = (term or "").lower()
    scored = []
    for c in valid_columns:
        cl = c.lower()
        overlap = len(set(t_low) & set(cl))
        if overlap > 0:
            scored.append((overlap, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:n]]


def _parse_body(
    body: str,
    valid_columns: Set[str],
    label_aliases: Optional[Dict[str, str]] = None,
) -> DSLTestSpec:
    """Parse the main DSL function call (before 'controlling for')."""
    body = body.strip()

    # assoc(X, Y)
    m = re.match(r"assoc\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", body, re.I)
    if m:
        x = _require_col(m.group(1).strip(), valid_columns, "assoc", "predictor", label_aliases)
        y = _require_col(m.group(2).strip(), valid_columns, "assoc", "outcome", label_aliases)
        return DSLTestSpec(family="assoc", x=x, y=y)

    # diff(Y, by=G)
    m = re.match(r"diff\(\s*([^,]+?)\s*,\s*by\s*=\s*([^)]+?)\s*\)", body, re.I)
    if m:
        y = _require_col(m.group(1).strip(), valid_columns, "diff", "outcome", label_aliases)
        g = _require_col(m.group(2).strip(), valid_columns, "diff", "group", label_aliases)
        return DSLTestSpec(family="diff", y=y, group=g)

    # interact(X * Z -> Y)
    m = re.match(r"interact\(\s*([^*]+?)\s*\*\s*([^-]+?)\s*->\s*([^)]+?)\s*\)", body, re.I)
    if m:
        x = _require_col(m.group(1).strip(), valid_columns, "interact", "predictor", label_aliases)
        z = _require_col(m.group(2).strip(), valid_columns, "interact", "moderator", label_aliases)
        y = _require_col(m.group(3).strip(), valid_columns, "interact", "outcome", label_aliases)
        return DSLTestSpec(family="interact", x=x, y=y, moderator=z)

    # heterogeneity(Y ~ X | G)
    m = re.match(r"heterogeneity\(\s*([^~]+?)\s*~\s*([^|]+?)\s*\|\s*([^)]+?)\s*\)", body, re.I)
    if m:
        y = _require_col(m.group(1).strip(), valid_columns, "heterogeneity", "outcome", label_aliases)
        x = _require_col(m.group(2).strip(), valid_columns, "heterogeneity", "predictor", label_aliases)
        g = _require_col(m.group(3).strip(), valid_columns, "heterogeneity", "group", label_aliases)
        return DSLTestSpec(family="heterogeneity", x=x, y=y, group=g)

    raise DSLParseError(
        f"Could not parse DSL body: '{body}'. "
        f"Expected one of: assoc(X,Y) | diff(Y,by=G) | interact(X*Z->Y) | heterogeneity(Y~X|G)"
    )


def _require_col(
    raw: str,
    valid_columns: Set[str],
    family: str,
    role: str,
    label_aliases: Optional[Dict[str, str]] = None,
) -> str:
    resolved = _resolve_column(raw, valid_columns, label_aliases)
    if resolved is None:
        raise DSLParseError(
            f"{family}: {role} '{raw}' not found in dataset columns. "
            f"Suggestion: {_suggest_columns(raw, valid_columns)}"
        )
    return resolved


# ═══════════════════════════════════════════════════════════════════════════
# 3. Canonical statistical executor (compiler table → direct execution)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DSLExecutionResult:
    """Execution result with captured warnings and test metadata."""
    result: Dict[str, Any]
    warnings: List[str]
    test_method: str
    formula: Optional[str] = None
    warning_severity: str = "none"
    review_flags: List[str] = field(default_factory=list)
    summary_estimand: Optional[str] = None
    summary_term: Optional[str] = None
    sign_semantics: Dict[str, Any] = field(default_factory=dict)


def _classify_warning_messages(messages: List[str]) -> Tuple[str, List[str]]:
    """Classify warning severity for audit and downstream flags."""
    review_flags: List[str] = []
    severity = "none"
    for msg in messages:
        lowered = msg.lower()
        if any(re.search(pattern, lowered) for pattern in _MATERIAL_WARNING_PATTERNS):
            severity = "material"
            if "rank_deficiency" not in review_flags:
                review_flags.append("rank_deficiency")
            continue
        if any(re.search(pattern, lowered) for pattern in _REVIEW_WARNING_PATTERNS):
            if severity == "none":
                severity = "review"
            if "statistical_warning" not in review_flags:
                review_flags.append("statistical_warning")
    return severity, review_flags


def _sign_semantics_for_column(col: Optional[str]) -> Dict[str, Any]:
    """Return direction metadata for a column or grouping variable."""
    if not col:
        return {}
    info = dict(_resolve_sign_semantics().get(col, {}))
    if not info:
        info = {
            "direction": "unknown",
            "meaning": "Consult the codebook before interpreting the sign.",
        }
    info["column"] = col
    return info


def _build_sign_semantics(spec: DSLTestSpec) -> Dict[str, Any]:
    """Collect codebook-direction metadata used for audit and evaluator flags."""
    columns = {
        "x": _sign_semantics_for_column(spec.x),
        "y": _sign_semantics_for_column(spec.y),
        "group": _sign_semantics_for_column(spec.group),
        "moderator": _sign_semantics_for_column(spec.moderator),
        "weight": _sign_semantics_for_column(spec.weight),
        "controls": [_sign_semantics_for_column(c) for c in spec.controls],
    }
    caution_columns = [
        entry["column"]
        for entry in [columns["x"], columns["y"], columns["group"], columns["moderator"]]
        if entry and entry.get("direction") in {"higher_is_more_negative", "coded_category"}
    ]
    caution_columns.extend(
        entry["column"]
        for entry in columns["controls"]
        if entry and entry.get("direction") in {"higher_is_more_negative", "coded_category"}
    )
    return {
        "columns": columns,
        "direction_caution": bool(caution_columns),
        "caution_columns": caution_columns,
    }


def _is_continuous_like(series: pd.Series) -> bool:
    """Heuristic for routing binary diff toward Welch vs rank tests."""
    numeric = pd.to_numeric(series, errors="coerce")
    unique_count = int(numeric.nunique(dropna=True))
    return unique_count >= 7


def _sorted_group_values(values: Any) -> List[Any]:
    """Stable ordering for human-readable group contrasts."""
    return sorted(list(values), key=lambda x: str(x))


def _rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U."""
    denom = n1 * n2
    if denom <= 0:
        return 0.0
    return float((2.0 * u_stat / denom) - 1.0)


def execute_spec(spec: DSLTestSpec, df: pd.DataFrame) -> DSLExecutionResult:
    """Execute the canonical statistical test for *spec* on *df*.

    Automatically dispatches logistic regression when the outcome variable
    is binary (detected via variable catalog or data inspection). Effect
    sizes from logistic models are standardized to Cohen's d for
    comparability with OLS results.

    Does NOT sanitise invalid values. If the test cannot produce valid
    scalar results, raises DSLExecutionError.
    """
    total_n = len(df)
    subset = _prepare_data(spec, df)

    # Detect binary outcome → dispatch logistic regression
    _y_is_binary = spec.y and _is_binary_outcome(spec.y, subset)

    caught: List[str] = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        if _y_is_binary and _HAS_STATSMODELS and spec.family in ("assoc", "diff", "interact", "heterogeneity"):
            # Binary Y: use logistic regression
            if (
                spec.family == "assoc"
                and not spec.controls
                and not spec.weight
                and not _is_categorical_col(spec.x, subset)
            ):
                # Simple assoc with binary Y and no controls: still use correlation
                # (point-biserial r is interpretable)
                res = _exec_correlation(spec, subset, total_n)
            elif spec.family == "diff" and not spec.controls and not spec.weight:
                # Simple diff with binary Y and no controls: nonparametric test
                res = _exec_diff_nonparam(spec, subset, total_n)
            else:
                res = _exec_logistic_regression(spec, subset, total_n, formula_type=spec.family)

        elif spec.family == "assoc":
            # A categorical predictor has no Pearson r to report — correlation
            # would try to cast its labels to float.  It asks the same question
            # diff() does, so it takes the regression/omnibus route.
            if spec.controls or spec.weight or _is_categorical_col(spec.x, subset):
                res = _exec_regression(spec, subset, total_n, formula_type="assoc")
            else:
                res = _exec_correlation(spec, subset, total_n)

        elif spec.family == "diff":
            if spec.controls or spec.weight:
                res = _exec_regression(spec, subset, total_n, formula_type="diff")
            else:
                res = _exec_diff_nonparam(spec, subset, total_n)

        elif spec.family == "interact":
            res = _exec_regression(spec, subset, total_n, formula_type="interact")

        elif spec.family == "heterogeneity":
            res = _exec_regression(spec, subset, total_n, formula_type="heterogeneity")

        else:
            raise DSLParseError(f"Unknown family: {spec.family}")

        caught = [f"{x.category.__name__}: {x.message}" for x in w]

    _validate_result_scalars(res.result)

    res.warnings = caught
    res.warning_severity, warning_flags = _classify_warning_messages(caught)
    res.review_flags.extend(flag for flag in warning_flags if flag not in res.review_flags)
    res.sign_semantics = _build_sign_semantics(spec)
    res.result.setdefault("summary_estimand", res.summary_estimand)
    if res.summary_term is not None:
        res.result.setdefault("summary_term", res.summary_term)
    res.result.setdefault("warning_severity", res.warning_severity)
    res.result.setdefault("review_flags", list(res.review_flags))
    res.result.setdefault("has_material_warning", res.warning_severity == "material")
    res.result.setdefault("sign_semantics", res.sign_semantics)
    return res


def replicate_spec(
    spec: DSLTestSpec,
    df: pd.DataFrame,
    seed: int = 42,
    p_threshold: float = 0.10,
    split_fraction: float = 0.7,
) -> str:
    """Split-sample replication: run the same spec on a random 70/30 split.

    1. Randomly split *df* into a ``split_fraction`` part and the remainder
       (stratified by the group column when the spec has one, falling back to
       a simple random split).
    2. Run ``execute_spec`` on each part independently.
    3. Report ``"replicated"`` when **both** parts show:
       - the same sign of effect_size, AND
       - p < *p_threshold* (relaxed vs the full-sample threshold since N is reduced).

    Returns one of: ``"replicated"``, ``"not_replicated"``, ``"replication_error"``.
    """
    try:
        n = len(df)
        if n < 40:
            # Too small to split meaningfully
            return "not_replicated"

        rng = np.random.RandomState(seed)

        # Stratified split by group column if available, else simple random
        group_col = spec.group
        if group_col and group_col in df.columns:
            # Ensure each group is represented in both parts
            idx_a, idx_b = [], []
            for _, grp in df.groupby(group_col):
                perm = rng.permutation(grp.index)
                cut = int(round(split_fraction * len(perm)))
                idx_a.extend(perm[:cut])
                idx_b.extend(perm[cut:])
            part_a = df.loc[idx_a]
            part_b = df.loc[idx_b]
        else:
            perm = rng.permutation(df.index)
            cut = int(round(split_fraction * n))
            part_a = df.loc[perm[:cut]]
            part_b = df.loc[perm[cut:]]

        res_a = execute_spec(spec, part_a)
        res_b = execute_spec(spec, part_b)

        es_a = res_a.result.get("effect_size", 0.0)
        es_b = res_b.result.get("effect_size", 0.0)
        p_a = res_a.result.get("p_value", 1.0)
        p_b = res_b.result.get("p_value", 1.0)

        same_sign = (es_a >= 0 and es_b >= 0) or (es_a < 0 and es_b < 0)
        both_sig = p_a < p_threshold and p_b < p_threshold

        if same_sign and both_sig:
            return "replicated"
        return "not_replicated"

    except Exception:
        # Replication is best-effort; never block the main result.
        return "replication_error"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare_data(spec: DSLTestSpec, df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant columns, coerce numerics, recode missing codes."""
    cols = spec.all_columns
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise DSLExecutionError(
            f"Columns not found in dataframe: {missing_cols}. "
            f"Available: {sorted(df.columns.tolist())}"
        )

    subset = df[cols].copy()

    _missing = _resolve_missing_codes()
    for c in cols:
        if c == spec.weight:
            num = pd.to_numeric(subset[c], errors="coerce")
            if _missing:
                subset[c] = num.where(~num.isin(_missing))
            else:
                subset[c] = num
            continue
        if c in _resolve_known_categorical():
            continue
        # Coerce numeric + recode missing codes
        num = pd.to_numeric(subset[c], errors="coerce")
        numeric_ratio = float(num.notna().mean()) if len(num) else 0.0
        if numeric_ratio >= 0.5:
            if _missing:
                subset[c] = num.where(~num.isin(_missing))
            else:
                subset[c] = num
        # else keep as-is (categorical/string column)

    subset = subset.dropna()
    if len(subset) < _MIN_N_CORRELATION:
        raise DSLExecutionError(
            f"Insufficient data after cleaning: {len(subset)} rows "
            f"(min {_MIN_N_CORRELATION}). Original: {len(df)} rows."
        )

    # Reject diff/heterogeneity when grouping variable has fewer than 2 levels
    if spec.group and spec.family in ("diff", "heterogeneity"):
        n_levels = subset[spec.group].nunique(dropna=True)
        if n_levels < 2:
            raise DSLExecutionError(
                f"Grouping variable '{spec.group}' has only {n_levels} level(s) "
                f"after cleaning — need at least 2 for {spec.family} tests."
            )

    return subset


# ---------------------------------------------------------------------------
# Outcome type detection
# ---------------------------------------------------------------------------

def _is_binary_outcome(y_col: str, df: pd.DataFrame) -> bool:
    """Detect if Y is a binary (0/1) variable.

    Uses two signals:
      1. Variable catalog: if the card's variable_kind == "binary", trust it.
      2. Data inspection fallback: unique non-NaN values are a subset of {0, 1}.
    """
    # Try variable catalog first (most reliable)
    try:
        from core.dataset_profile import get_profile
        profile = get_profile()
        catalog_path = (
            profile.get("catalog_path")
            or (profile.get("variable_catalog", {}) or {}).get("catalog_path", "")
        )
        if catalog_path:
            from core.variable_catalog import load_variable_cards_jsonl
            cards = load_variable_cards_jsonl(str(catalog_path))
            for card in cards:
                if card.var_name == y_col:
                    return card.variable_kind == "binary"
    except Exception:
        pass

    # Fallback: inspect the data
    try:
        vals = df[y_col].dropna()
        if len(vals) == 0:
            return False
        unique = set(vals.unique())
        return unique.issubset({0, 1, 0.0, 1.0})
    except Exception:
        return False


def _log_odds_to_cohens_d(log_odds: float) -> float:
    """Convert a log-odds ratio to Cohen's d (Hasselblad & Hedges, 1995).

    d = log_OR * sqrt(3) / pi

    This puts logistic regression effect sizes on the same scale as OLS
    partial-r / Cohen's d, so the Evaluator and Critic scoring pipelines
    work without modification.
    """
    return log_odds * math.sqrt(3) / math.pi


# ---------------------------------------------------------------------------
# Family executors
# ---------------------------------------------------------------------------

def _exec_correlation(spec: DSLTestSpec, subset: pd.DataFrame, total_n: int) -> DSLExecutionResult:
    """assoc(X, Y) — no controls → Pearson correlation with Fisher-z CI."""
    x_vals = subset[spec.x].astype(float)
    y_vals = subset[spec.y].astype(float)
    n = len(x_vals)

    if x_vals.std() == 0 or y_vals.std() == 0:
        raise DSLExecutionError(
            f"Zero variance in '{spec.x}' or '{spec.y}' — correlation undefined"
        )

    r, p = stats.pearsonr(x_vals, y_vals)

    # Fisher z-transform CI
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3) if n > 3 else float("inf")
    ci_lo = float(np.tanh(z - 1.96 * se))
    ci_hi = float(np.tanh(z + 1.96 * se))

    return DSLExecutionResult(
        result={
            "effect_size": float(r),
            "p_value": float(p),
            "n_observations": int(n),
            "coverage": float(n / total_n),
            "confidence_interval": (ci_lo, ci_hi),
            "effect_size_kind": "pearson_r",
        },
        warnings=[],
        test_method="pearson_correlation",
        summary_estimand="correlation",
        summary_term=f"{spec.x} ~ {spec.y}",
    )


def _exec_logistic_regression(
    spec: DSLTestSpec, subset: pd.DataFrame, total_n: int, *, formula_type: str
) -> DSLExecutionResult:
    """Logistic regression for binary outcome variables.

    Dispatched automatically when Y is detected as binary (0/1).
    Effect sizes are converted from log-odds to Cohen's d for comparability
    with OLS-based results throughout the scoring pipeline.

    A fit that fails, or a template term with no single contrast, raises
    DSLExecutionError (no finding) instead of switching to another model.
    """
    if not _HAS_STATSMODELS:
        raise DSLExecutionError("statsmodels is required for logistic regression.")

    # ── build formula (same structure as OLS) ──
    rhs_parts: List[str] = []
    primary_term: Optional[str] = None

    subset, collapse_notes = _collapse_rare_levels(
        subset, _formula_categoricals(spec, subset, formula_type)
    )

    # ── pre-flight cardinality checks ──
    if formula_type == "diff" or formula_type == "heterogeneity":
        _check_cardinality(spec.group, "group", subset)
    if formula_type == "interact":
        _check_cardinality(spec.moderator, "moderator", subset)
    if formula_type == "assoc":
        _check_cardinality(spec.x, "predictor", subset)
    for ctrl in spec.controls:
        _check_cardinality(ctrl, "control", subset)

    if formula_type == "assoc":
        if _is_categorical_col(spec.x, subset):
            rhs_parts.append(f"C({spec.x})")
        else:
            rhs_parts.append(spec.x)
            primary_term = spec.x
        rhs_parts.extend(_fterm(c, subset) for c in spec.controls)
    elif formula_type == "diff":
        rhs_parts.append(f"C({spec.group})")
        rhs_parts.extend(_fterm(c, subset) for c in spec.controls)
    elif formula_type == "interact":
        x_term = f"C({spec.x})" if _is_categorical_col(spec.x, subset) else spec.x
        z_term = _fterm(spec.moderator, subset)
        rhs_parts.append(f"{x_term} * {z_term}")
        rhs_parts.extend(_fterm(c, subset) for c in spec.controls)
        if not _is_categorical_col(spec.moderator, subset) and x_term == spec.x:
            primary_term = f"{spec.x}:{spec.moderator}"
    elif formula_type == "heterogeneity":
        x_term = f"C({spec.x})" if _is_categorical_col(spec.x, subset) else spec.x
        rhs_parts.append(f"{x_term} * C({spec.group})")
        rhs_parts.extend(_fterm(c, subset) for c in spec.controls)
    else:
        raise DSLParseError(f"Unknown formula_type: {formula_type}")

    formula = f"{spec.y} ~ " + " + ".join(rhs_parts)
    n = len(subset)
    n_predictors = _estimate_actual_predictors(rhs_parts, subset)
    if n < max(_MIN_N_REGRESSION, 10 * n_predictors):
        raise DSLExecutionError(
            f"Insufficient data for logistic regression: {n} rows, "
            f"{n_predictors} predictors after dummy expansion "
            f"(need >= {max(_MIN_N_REGRESSION, 10 * n_predictors)})"
        )

    # ── fit logistic model ──
    # A failed fit (e.g. perfect separation) yields no finding.
    try:
        model = smf.logit(formula, data=subset).fit(disp=0, maxiter=100, warn_convergence=True)
        method = "logistic_regression"
    except Exception as exc:
        raise DSLExecutionError(f"Logistic regression failed to fit: {exc}") from exc

    # ── extract effect / p-value / CI ──
    extra: Dict[str, Any] = {"model_family": "logistic", "effect_size_kind": "cohens_d_from_log_odds"}

    if primary_term is None:
        # Report the parameter the template asks about: the group contrast
        # (diff, categorical assoc) or the interaction (interact, heterogeneity).
        # A multi-level categorical term has no single log-odds contrast, so it
        # yields no finding instead of a positionally chosen coefficient.
        names = [str(name) for name in model.params.index if name != "Intercept"]
        if formula_type in ("interact", "heterogeneity"):
            candidates = [name for name in names if ":" in name]
        elif formula_type == "diff":
            candidates = [name for name in names if name.startswith(f"C({spec.group})[T.") and ":" not in name]
        else:
            candidates = [name for name in names if name.startswith(f"C({spec.x})[T.") and ":" not in name]
        if len(candidates) != 1:
            raise DSLExecutionError(
                f"Logistic {formula_type}: expected one target parameter, found "
                f"{len(candidates)} ({candidates}); multi-level categorical terms "
                f"have no single contrast under a binary outcome."
            )
        primary_term = candidates[0]
    elif primary_term not in model.params:
        raise DSLExecutionError(
            f"Primary term '{primary_term}' not found in logistic model parameters: "
            f"{list(model.params.index)}"
        )

    log_odds = float(model.params[primary_term])
    p = float(model.pvalues[primary_term])
    raw_ci = model.conf_int().loc[primary_term]
    raw_ci_lo, raw_ci_hi = float(raw_ci.iloc[0]), float(raw_ci.iloc[1])
    # Convert log-odds to Cohen's d for standardized comparability
    effect = _log_odds_to_cohens_d(log_odds)
    ci_lo = _log_odds_to_cohens_d(raw_ci_lo)
    ci_hi = _log_odds_to_cohens_d(raw_ci_hi)
    summary_term = primary_term
    summary_estimand = "log_odds_as_d"
    extra["raw_log_odds"] = log_odds
    extra["raw_odds_ratio"] = math.exp(log_odds) if abs(log_odds) < 500 else float("inf")
    extra["raw_ci_log_odds"] = (raw_ci_lo, raw_ci_hi)

    # Pseudo R-squared (McFadden)
    try:
        pseudo_r2 = float(model.prsquared)
    except Exception:
        pseudo_r2 = 0.0
    extra["pseudo_r_squared"] = pseudo_r2

    result_dict: Dict[str, Any] = {
        "effect_size": effect,
        "p_value": p,
        "n_observations": int(model.nobs),
        "coverage": float(n / total_n),
        "confidence_interval": (ci_lo, ci_hi),
        "r_squared": pseudo_r2,
    }
    if collapse_notes:
        extra["collapsed_levels"] = list(collapse_notes)
    result_dict.update(extra)

    return DSLExecutionResult(
        result=result_dict,
        warnings=list(collapse_notes),
        test_method=method,
        formula=formula,
        summary_estimand=summary_estimand,
        summary_term=summary_term,
    )


def _exec_diff_nonparam(spec: DSLTestSpec, subset: pd.DataFrame, total_n: int) -> DSLExecutionResult:
    """diff(Y, by=G) — no controls → schema-dependent binary diff, omnibus+summary for k>2."""
    # The k>2 branch reports a summary contrast between the first two sorted
    # levels.  Without pooling, a level holding a couple of rows lands in that
    # pair and sinks the whole test even though the Kruskal-Wallis omnibus
    # computed fine.
    subset, collapse_notes = _collapse_rare_levels(subset, [spec.group])
    y_vals = subset[spec.y].astype(float)
    g_vals = subset[spec.group]
    groups = _sorted_group_values(g_vals.unique())
    k = len(groups)

    if k < 2:
        raise DSLExecutionError(f"Only {k} group(s) in '{spec.group}' — need ≥ 2")

    if k == 2:
        g1_name, g2_name = groups[0], groups[1]
        g1 = y_vals[g_vals == g1_name]
        g2 = y_vals[g_vals == g2_name]
        if len(g1) < _MIN_N_PER_GROUP or len(g2) < _MIN_N_PER_GROUP:
            raise DSLExecutionError(
                f"Groups too small: n1={len(g1)}, n2={len(g2)} (min {_MIN_N_PER_GROUP})"
            )
        n = len(g1) + len(g2)
        summary_term = f"{spec.group}[{g2_name}] - {spec.group}[{g1_name}]"

        if _is_continuous_like(y_vals):
            mean_diff = float(g2.mean() - g1.mean())
            var1 = float(g1.var(ddof=1))
            var2 = float(g2.var(ddof=1))
            se = math.sqrt((var1 / len(g1)) + (var2 / len(g2)))
            if se == 0:
                raise DSLExecutionError("Welch t-test undefined because the standard error is zero")
            t_stat, p = stats.ttest_ind(g2, g1, equal_var=False)
            num = ((var1 / len(g1)) + (var2 / len(g2))) ** 2
            den = 0.0
            if len(g1) > 1:
                den += ((var1 / len(g1)) ** 2) / (len(g1) - 1)
            if len(g2) > 1:
                den += ((var2 / len(g2)) ** 2) / (len(g2) - 1)
            df = num / den if den > 0 else max(n - 2, 1)
            t_crit = float(stats.t.ppf(0.975, df))
            ci_lo = mean_diff - t_crit * se
            ci_hi = mean_diff + t_crit * se
            pooled_std = math.sqrt(((len(g1) - 1) * var1 + (len(g2) - 1) * var2) / max(n - 2, 1))
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
            # CI for Cohen's d via non-central t approximation
            d_se = math.sqrt((len(g1) + len(g2)) / (len(g1) * len(g2)) + cohens_d ** 2 / (2 * (len(g1) + len(g2))))
            t_crit_d = float(stats.t.ppf(0.975, max(n - 2, 1)))
            d_ci_lo = cohens_d - t_crit_d * d_se
            d_ci_hi = cohens_d + t_crit_d * d_se
            return DSLExecutionResult(
                result={
                    "effect_size": float(cohens_d),
                    "p_value": float(p),
                    "n_observations": int(n),
                    "coverage": float(n / total_n),
                    "confidence_interval": (d_ci_lo, d_ci_hi),
                    "effect_size_kind": "cohens_d",
                    "group_labels": [g1_name, g2_name],
                    "mean_difference": float(mean_diff),
                    "raw_ci": (ci_lo, ci_hi),
                },
                warnings=[],
                test_method="welch_t_test",
                summary_estimand="contrast",
                summary_term=summary_term,
            )

        u_stat, p = stats.mannwhitneyu(g2, g1, alternative="two-sided")
        rank_biserial = _rank_biserial_from_u(float(u_stat), len(g2), len(g1))
        clipped = max(min(rank_biserial, 0.999999), -0.999999)
        z_val = np.arctanh(clipped)
        se_z = 1.0 / math.sqrt(max(n - 3, 1))
        ci_lo = float(np.tanh(z_val - 1.96 * se_z))
        ci_hi = float(np.tanh(z_val + 1.96 * se_z))
        return DSLExecutionResult(
            result={
                "effect_size": float(rank_biserial),
                "p_value": float(p),
                "n_observations": int(n),
                "coverage": float(n / total_n),
                "confidence_interval": (ci_lo, ci_hi),
                "effect_size_kind": "rank_biserial",
                "group_labels": [g1_name, g2_name],
                "u_statistic": float(u_stat),
            },
            warnings=[],
            test_method="mann_whitney_u",
            summary_estimand="contrast",
            summary_term=summary_term,
        )

    # >2 groups → Kruskal-Wallis with epsilon-squared
    arrays = []
    for grp in groups:
        arr = y_vals[g_vals == grp]
        if len(arr) >= 2:
            arrays.append(arr)
    if len(arrays) < 2:
        raise DSLExecutionError("Not enough groups with ≥ 2 observations for Kruskal-Wallis")

    h_stat, p = stats.kruskal(*arrays)
    n = sum(len(a) for a in arrays)
    # Epsilon-squared = H / ((n² - 1)/(n + 1)) = H*(n+1)/(n²-1)
    eps_sq = float(h_stat * (n + 1) / (n * n - 1)) if n > 1 else 0.0
    eps_sq = max(0.0, min(1.0, eps_sq))

    # Approximate SE for epsilon-squared (delta-method)
    se_eps = math.sqrt(4 * eps_sq * (1 - eps_sq) ** 2 / max(n - 1, 1)) if n > 1 else 0.0
    omni_ci_lo = max(0.0, eps_sq - 1.96 * se_eps)
    omni_ci_hi = min(1.0, eps_sq + 1.96 * se_eps)

    # Keep the scalar summary contrast-compatible for evaluator/critic,
    # but also persist the true omnibus test in raw_results.
    base_name, contrast_name = groups[0], groups[1]
    base_arr = y_vals[g_vals == base_name]
    contrast_arr = y_vals[g_vals == contrast_name]
    if len(base_arr) < _MIN_N_PER_GROUP or len(contrast_arr) < _MIN_N_PER_GROUP:
        raise DSLExecutionError(
            f"Reference pair too small for summary contrast: {base_name} n={len(base_arr)}, "
            f"{contrast_name} n={len(contrast_arr)}"
        )
    u_stat, pair_p = stats.mannwhitneyu(contrast_arr, base_arr, alternative="two-sided")
    pair_n = len(base_arr) + len(contrast_arr)
    rank_biserial = _rank_biserial_from_u(float(u_stat), len(contrast_arr), len(base_arr))
    clipped = max(min(rank_biserial, 0.999999), -0.999999)
    z_val = np.arctanh(clipped)
    se_z = 1.0 / math.sqrt(max(pair_n - 3, 1))
    ci_lo = float(np.tanh(z_val - 1.96 * se_z))
    ci_hi = float(np.tanh(z_val + 1.96 * se_z))
    summary_term = f"{spec.group}[{contrast_name}] - {spec.group}[{base_name}]"

    return DSLExecutionResult(
        result={
            # Omnibus statistics as the primary result — these represent
            # the actual research question "does Y differ across groups?"
            "effect_size": float(eps_sq),
            "p_value": float(p),
            "n_observations": int(n),
            "coverage": float(n / total_n),
            "confidence_interval": (omni_ci_lo, omni_ci_hi),
            "n_groups": int(len(arrays)),
            "effect_size_kind": "epsilon_squared",
            "group_labels": groups,
            # Preserve pairwise contrast for reference
            "pairwise_contrast": {
                "summary_term": summary_term,
                "effect_size": float(rank_biserial),
                "effect_size_kind": "rank_biserial",
                "p_value": float(pair_p),
                "n_observations": int(pair_n),
                "confidence_interval": (ci_lo, ci_hi),
            },
        },
        warnings=[],
        test_method="kruskal_wallis",
        summary_estimand="omnibus",
        summary_term=f"C({spec.group})",
    )


def _is_categorical_col(col: str, subset: pd.DataFrame) -> bool:
    """True when the fitted model will dummy-expand *col*.

    ``known_categorical`` is a *declaration*; patsy decides from the dtype and
    expands every non-numeric column into level dummies whether or not the
    profile names it.  Building the formula from the declaration alone lets the
    two disagree, which is what raised "Primary term '<x>:<z>' not found in
    model parameters" on datasets whose string columns are undeclared — e.g.
    Amazon's ``product_format`` and ``product_sub_category``.
    """
    if col in _resolve_known_categorical():
        return True
    if col not in subset.columns:
        return False
    series = subset[col]
    if pd.api.types.is_bool_dtype(series):
        return True
    return not pd.api.types.is_numeric_dtype(series)


def _formula_categoricals(
    spec: DSLTestSpec, subset: pd.DataFrame, formula_type: str
) -> List[str]:
    """Columns this formula will dummy-expand, in the roles that expand them."""
    cols: List[str] = []
    if formula_type in ("diff", "heterogeneity"):
        cols.append(spec.group)
    if formula_type == "interact":
        cols.append(spec.moderator)
    if formula_type == "assoc":
        cols.append(spec.x)
    cols.extend(spec.controls)
    return [
        c for c in dict.fromkeys(cols)
        if c and _is_categorical_col(c, subset)
    ]


def _expanded_terms(model, term: str) -> List[str]:
    """Model parameters *term* expanded into, if it dummy-expanded at all.

    ``x:z`` can arrive as ``x:z[T.b]``, ``x[T.a]:z`` or ``x[T.a]:z[T.b]``
    depending on which side patsy treated as categorical, so the components are
    matched one by one.  Prefix matching only ever caught the first of those
    three shapes.
    """
    want = term.split(":")
    out: List[str] = []
    for param in model.params.index:
        parts = str(param).split(":")
        if len(parts) != len(want):
            continue
        if all(got == exp or got.startswith(f"{exp}[") for got, exp in zip(parts, want)):
            out.append(str(param))
    return out


def _collapse_rare_levels(
    subset: pd.DataFrame, cols: List[str], min_cell: int = _MIN_LEVEL_CELL
) -> Tuple[pd.DataFrame, List[str]]:
    """Fold thin and long-tail levels of *cols* into a single ``Other`` bucket.

    Pooling removes two failure modes that would otherwise return no result:
      * levels holding a handful of rows (``Album n=2``) leave the reference
        contrast unestimable;
      * hundreds of levels overshoot _MAX_CATEGORICAL_CARDINALITY, putting the
        column permanently out of the DSL's reach.

    A level is kept when it holds at least *min_cell* rows, capped at the
    _MAX_CATEGORICAL_CARDINALITY - 1 most frequent.  Rows outside that set
    become ``Other``, or are dropped when even the pooled bucket would be
    thinner than *min_cell*.  The collapse is handed back to the caller so it
    reaches the result's warnings instead of silently changing what the test
    means.
    """
    notes: List[str] = []
    out = subset
    keep_max = max(1, _MAX_CATEGORICAL_CARDINALITY - 1)
    for col in dict.fromkeys(cols):
        if col not in out.columns:
            continue
        counts = out[col].value_counts(dropna=True)
        if counts.empty:
            continue
        keep = counts[counts >= min_cell].index[:keep_max]
        if len(keep) == 0 or len(keep) >= len(counts):
            continue
        if out is subset:
            out = subset.copy()
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(object)
        folded = ~out[col].isin(keep) & out[col].notna()
        n_folded = int(folded.sum())
        if n_folded == 0:
            continue
        pooled = n_folded >= min_cell
        out.loc[folded, col] = "Other" if pooled else np.nan
        notes.append(
            f"{col}: kept {len(keep)} of {len(counts)} levels; "
            f"{'pooled' if pooled else 'dropped'} {n_folded} rows in levels "
            f"under n={min_cell}"
        )
    return out, notes


def _check_cardinality(col: str, role: str, subset: pd.DataFrame) -> None:
    """Raise DSLExecutionError if a categorical column has too many levels."""
    if col not in subset.columns:
        return
    # Only check columns the fitted model will actually dummy-expand.  Group
    # roles are always wrapped in C(); every other role is decided from the
    # dtype, since patsy expands any non-numeric column whether or not the
    # profile declares it.  Continuous moderators in interact() use X:Z terms
    # without C(), so they create no dummies and need no guard.
    if role not in ("group",) and not _is_categorical_col(col, subset):
        return  # continuous column, no dummy expansion
    card = int(subset[col].nunique(dropna=True))
    if card > _MAX_CATEGORICAL_CARDINALITY:
        raise DSLExecutionError(
            f"Column '{col}' ({role}) has {card} unique values, exceeding "
            f"the safe limit of {_MAX_CATEGORICAL_CARDINALITY} for regression. "
            f"This would create {card - 1} dummy variables."
        )


def _estimate_actual_predictors(rhs_parts: List[str], subset: pd.DataFrame) -> int:
    """Count actual predictor columns after dummy variable expansion."""
    n_actual = 0
    for term in rhs_parts:
        if "*" in term:
            # Interaction term: estimate both main effects + interactions
            parts = [p.strip() for p in term.split("*")]
            term_cols = 1
            for p in parts:
                if p.startswith("C(") and p.endswith(")"):
                    col_name = p[2:-1]
                    if col_name in subset.columns:
                        term_cols *= max(1, int(subset[col_name].nunique(dropna=True)) - 1)
                else:
                    term_cols *= 1  # continuous: 1 column
            n_actual += term_cols
            # Also count main effects from interaction (statsmodels adds them)
            for p in parts:
                if p.startswith("C(") and p.endswith(")"):
                    col_name = p[2:-1]
                    if col_name in subset.columns:
                        n_actual += max(1, int(subset[col_name].nunique(dropna=True)) - 1)
                else:
                    n_actual += 1
        elif term.startswith("C(") and term.endswith(")"):
            col_name = term[2:-1]
            if col_name in subset.columns:
                n_actual += max(1, int(subset[col_name].nunique(dropna=True)) - 1)
            else:
                n_actual += 1
        else:
            n_actual += 1
    return n_actual


def _exec_regression(
    spec: DSLTestSpec, subset: pd.DataFrame, total_n: int, *, formula_type: str
) -> DSLExecutionResult:
    """Unified OLS/WLS executor for assoc-with-controls, diff, interact, heterogeneity.

    Compiler table:
      assoc + controls       → Y ~ X + C1 + C2 …
      diff  + controls       → Y ~ C(G) + C1 + C2 …
      interact               → Y ~ X * Z + C1 + C2 …     (Z = moderator)
      heterogeneity          → Y ~ X * C(G) + C1 + C2 …
    """
    if not _HAS_STATSMODELS:
        raise DSLExecutionError(
            "statsmodels is required for regression-based DSL families "
            "(assoc with controls, diff with controls, interact, heterogeneity). "
            "Install with: pip install statsmodels"
        )

    # ── build formula ──
    rhs_parts: List[str] = []
    primary_term: Optional[str] = None
    summary_estimand = "coefficient"
    use_anova_f: bool = False
    anova_target: Optional[str] = None
    contrast_prefix: Optional[str] = None

    # Thin and long-tail levels are pooled before anything measures the
    # column, so the cardinality guard below judges the design that is
    # actually fitted.
    subset, collapse_notes = _collapse_rare_levels(
        subset, _formula_categoricals(spec, subset, formula_type)
    )

    # ── pre-flight cardinality checks ──
    if formula_type == "diff" or formula_type == "heterogeneity":
        _check_cardinality(spec.group, "group", subset)
    if formula_type == "interact":
        _check_cardinality(spec.moderator, "moderator", subset)
    if formula_type == "assoc":
        _check_cardinality(spec.x, "predictor", subset)
    for ctrl in spec.controls:
        _check_cardinality(ctrl, "control", subset)

    if formula_type == "assoc":
        if _is_categorical_col(spec.x, subset):
            # A categorical predictor has no single slope to report; the
            # question it asks is "does Y differ across X", so it takes the
            # same omnibus F route as diff().
            x_term = f"C({spec.x})"
            rhs_parts.append(x_term)
            x_levels = int(subset[spec.x].nunique(dropna=True))
            if x_levels > 2:
                use_anova_f = True
                anova_target = x_term
            else:
                summary_estimand = "contrast"
                contrast_prefix = f"{x_term}[T."
        else:
            rhs_parts.append(spec.x)
            primary_term = spec.x
        rhs_parts.extend(_fterm(c, subset) for c in spec.controls)

    elif formula_type == "diff":
        g_term = f"C({spec.group})"
        rhs_parts.append(g_term)
        rhs_parts.extend(_fterm(c, subset) for c in spec.controls)
        group_levels = int(subset[spec.group].nunique(dropna=True))
        if group_levels > 2:
            use_anova_f = True
            anova_target = g_term
        else:
            summary_estimand = "contrast"
        contrast_prefix = f"{g_term}[T."

    elif formula_type == "interact":
        x_cat = _is_categorical_col(spec.x, subset)
        z_cat = _is_categorical_col(spec.moderator, subset)
        x_term = f"C({spec.x})" if x_cat else spec.x
        z_term = _fterm(spec.moderator, subset)
        rhs_parts.append(f"{x_term} * {z_term}")
        rhs_parts.extend(_fterm(c, subset) for c in spec.controls)
        if z_cat:
            moderator_levels = int(subset[spec.moderator].nunique(dropna=True))
            if moderator_levels > 2:
                use_anova_f = True
                anova_target = f"{x_term}:{z_term}"
            else:
                summary_estimand = "contrast"
            contrast_prefix = f"{x_term}:{z_term}[T."
        elif x_cat:
            # Categorical predictor against a numeric moderator: the
            # interaction is a block of C(x)[T.lvl]:z coefficients, so there is
            # no single slope to name and the block gets an omnibus F.
            use_anova_f = True
            anova_target = f"{x_term}:{spec.moderator}"
        else:
            primary_term = f"{spec.x}:{spec.moderator}"

    elif formula_type == "heterogeneity":
        x_term = f"C({spec.x})" if _is_categorical_col(spec.x, subset) else spec.x
        g_term = f"C({spec.group})"
        rhs_parts.append(f"{x_term} * {g_term}")
        rhs_parts.extend(_fterm(c, subset) for c in spec.controls)
        group_levels = int(subset[spec.group].nunique(dropna=True))
        if group_levels > 2:
            use_anova_f = True
            anova_target = f"{x_term}:{g_term}"
        else:
            summary_estimand = "contrast"
        contrast_prefix = f"{x_term}:{g_term}[T."

    else:
        raise DSLParseError(f"Unknown formula_type: {formula_type}")

    formula = f"{spec.y} ~ " + " + ".join(rhs_parts)
    n_predictors = _estimate_actual_predictors(rhs_parts, subset)
    n = len(subset)
    if n < max(_MIN_N_REGRESSION, 5 * n_predictors):
        raise DSLExecutionError(
            f"Insufficient data for regression: {n} rows, "
            f"{n_predictors} predictors after dummy expansion "
            f"(need ≥ {max(_MIN_N_REGRESSION, 5 * n_predictors)})"
        )

    # ── fit model ──
    if spec.weight and spec.weight in subset.columns:
        model = smf.wls(formula, data=subset, weights=subset[spec.weight]).fit()
        method = "wls_regression"
    else:
        model = smf.ols(formula, data=subset).fit()
        method = "ols_regression"

    # ── extract effect / p-value / CI ──
    extra: Dict[str, Any] = {}
    if use_anova_f:
        summary = _extract_anova_summary(
            model, spec, formula_type, anova_target, contrast_prefix
        )
        omni = summary["omnibus"]
        # Use omnibus statistics as the primary result — these represent
        # the actual research question (e.g., "does Y differ across groups?").
        effect = float(omni["effect_size"])  # partial eta-squared
        p = float(omni["p_value"])
        # CI for partial eta-squared: use non-negative bounds [0, 1]
        # (precise CI would need non-central F, approximate with delta method)
        se_eta = math.sqrt(4 * effect * (1 - effect) ** 2 / max(int(model.nobs) - 1, 1))
        ci_lo = max(0.0, effect - 1.96 * se_eta)
        ci_hi = min(1.0, effect + 1.96 * se_eta)
        extra["omnibus"] = omni
        extra["effect_size_kind"] = "partial_eta_squared"
        # Preserve the single-contrast details for reference
        extra["first_contrast"] = {
            "summary_term": summary["summary_term"],
            "effect_size": summary["effect_size"],
            "effect_size_kind": "partial_r",
            "p_value": summary["p_value"],
            "confidence_interval": summary["confidence_interval"],
            "raw_coefficient": summary.get("raw_coefficient"),
            "raw_ci": summary.get("raw_ci"),
        }
        summary_estimand = "omnibus"
        summary_term = str(omni.get("term", ""))
    else:
        if primary_term is None:
            summary = _extract_first_group_contrast(
                model, spec, formula_type, contrast_prefix
            )
            effect = summary["effect_size"]
            p = summary["p_value"]
            ci_lo, ci_hi = summary["confidence_interval"]
            summary_term = summary["summary_term"]
            extra["effect_size_kind"] = "partial_r"
            summary_estimand = "contrast"
        else:
            expanded = _expanded_terms(model, primary_term)
            if primary_term not in model.params and expanded:
                # The term survived as a set of level dummies.  Its joint Wald
                # test is the honest summary of "does this term matter", and
                # is reported on the same partial-eta-squared scale the
                # omnibus branch uses.
                ftest = model.f_test(" = 0, ".join(expanded) + " = 0")
                f_stat = float(np.squeeze(ftest.fvalue))
                df_num = float(ftest.df_num)
                df_den = float(ftest.df_denom)
                effect = (f_stat * df_num) / (f_stat * df_num + df_den)
                effect = max(0.0, min(1.0, effect))
                p = float(np.squeeze(ftest.pvalue))
                se_eta = math.sqrt(
                    4 * effect * (1 - effect) ** 2 / max(int(model.nobs) - 1, 1)
                )
                ci_lo = max(0.0, effect - 1.96 * se_eta)
                ci_hi = min(1.0, effect + 1.96 * se_eta)
                extra["effect_size_kind"] = "partial_eta_squared"
                extra["omnibus"] = {
                    "test": "wald_joint",
                    "term": primary_term,
                    "statistic": f_stat,
                    "df_num": df_num,
                    "df_den": df_den,
                    "p_value": p,
                    "effect_size": effect,
                    "effect_size_kind": "partial_eta_squared",
                }
                summary_term = primary_term
                summary_estimand = "omnibus"
            elif primary_term not in model.params:
                available = list(model.params.index)
                raise DSLExecutionError(
                    f"Primary term '{primary_term}' not found in model parameters. "
                    f"Available: {available}"
                )
            else:
                effect = _partial_r_from_model(model, primary_term)
                extra["raw_coefficient"] = float(model.params[primary_term])
                p = float(model.pvalues[primary_term])
                ci_lo, ci_hi = _partial_r_ci(effect, int(model.nobs))
                raw_ci = model.conf_int().loc[primary_term]
                extra["raw_ci"] = (float(raw_ci.iloc[0]), float(raw_ci.iloc[1]))
                summary_term = primary_term
                extra["effect_size_kind"] = "partial_r"

    result_dict: Dict[str, Any] = {
        "effect_size": effect,
        "p_value": p,
        "n_observations": int(model.nobs),
        "coverage": float(n / total_n),
        "confidence_interval": (ci_lo, ci_hi),
        "r_squared": float(model.rsquared),
    }
    if collapse_notes:
        extra["collapsed_levels"] = list(collapse_notes)
    result_dict.update(extra)

    return DSLExecutionResult(
        result=result_dict,
        warnings=list(collapse_notes),
        test_method=method,
        formula=formula,
        summary_estimand=summary_estimand,
        summary_term=summary_term,
    )


# ---------------------------------------------------------------------------
# ANOVA helper (for diff / categorical interact / heterogeneity)
# ---------------------------------------------------------------------------

def _extract_anova_summary(
    model,
    spec: DSLTestSpec,
    formula_type: str,
    target_override: Optional[str] = None,
    contrast_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Return contrast-aligned scalar summary plus omnibus ANOVA metadata.

    *target_override* is the term the caller actually put in the formula.
    Re-deriving it here from the declared categorical list is what let the
    lookup drift away from the fitted design, so the caller's string wins.
    """
    try:
        anova_table = sm.stats.anova_lm(model, typ=2)
    except Exception as exc:
        raise DSLExecutionError(f"ANOVA table computation failed: {exc}") from exc

    # Identify the target term for the F-test p-value
    if target_override:
        target = target_override
    elif formula_type == "diff":
        target = f"C({spec.group})"
    elif formula_type == "interact" and spec.moderator in _resolve_known_categorical():
        target = f"{spec.x}:C({spec.moderator})"
    elif formula_type == "heterogeneity":
        target = f"{spec.x}:C({spec.group})"
    else:
        target = None

    matched_row = None
    if target:
        for idx_name in anova_table.index:
            if idx_name == target:
                matched_row = idx_name
                break
        if matched_row is None:
            for idx_name in anova_table.index:
                if target in str(idx_name) or str(idx_name) in target:
                    matched_row = idx_name
                    break

    if matched_row is None:
        for idx_name in anova_table.index:
            if "residual" not in str(idx_name).lower():
                matched_row = idx_name
                break

    if matched_row is None:
        raise DSLExecutionError(
            f"Could not locate target term in ANOVA table. "
            f"Table index: {list(anova_table.index)}"
        )

    ss_term = float(anova_table.loc[matched_row, "sum_sq"])
    ss_resid = float(anova_table.loc["Residual", "sum_sq"])
    p_val = float(anova_table.loc[matched_row, "PR(>F)"])
    f_stat = float(anova_table.loc[matched_row, "F"])
    df_num = float(anova_table.loc[matched_row, "df"])
    df_den = float(anova_table.loc["Residual", "df"])

    # Partial eta-squared (standardised overall effect, stored in raw_results)
    eta_sq = ss_term / (ss_term + ss_resid) if (ss_term + ss_resid) > 0 else 0.0
    eta_sq = max(0.0, min(1.0, eta_sq))

    summary = _extract_first_group_contrast(
        model, spec, formula_type, contrast_prefix
    )
    summary["omnibus"] = {
        "test": "anova_type_2",
        "term": matched_row,
        "statistic": f_stat,
        "df_num": df_num,
        "df_den": df_den,
        "p_value": p_val,
        "effect_size": eta_sq,
        "effect_size_kind": "partial_eta_squared",
    }
    return summary


def _partial_r_from_model(model, term: str) -> float:
    """Compute partial correlation from the model's t-statistic for *term*.

    Formula: r_partial = t / sqrt(t² + df_resid)

    This is bounded [-1, 1] and directly comparable to Pearson r,
    regardless of variable scaling.
    """
    t = float(model.tvalues[term])
    df = float(model.df_resid)
    return t / math.sqrt(t * t + df)


def _partial_r_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute CI for a partial correlation using Fisher z-transform.

    Same technique as for Pearson r: z = atanh(r), SE = 1/sqrt(n-3),
    then back-transform with tanh.
    """
    if n <= 3:
        return (-1.0, 1.0)
    # Clamp to avoid atanh domain errors at exact ±1
    r_clamped = max(-0.9999, min(0.9999, r))
    z = np.arctanh(r_clamped)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_lo = z - z_crit * se
    z_hi = z + z_crit * se
    return (float(np.tanh(z_lo)), float(np.tanh(z_hi)))


def _extract_first_group_contrast(
    model,
    spec: DSLTestSpec,
    formula_type: str,
    prefix_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the first named contrast with matched effect, p-value, and CI.

    *prefix_override* is the dummy prefix implied by the formula the caller
    built; it takes precedence over re-deriving one from the declared list.
    """
    if prefix_override:
        prefix = prefix_override
    elif formula_type == "diff":
        prefix = f"C({spec.group})[T."
    elif formula_type == "interact" and spec.moderator in _resolve_known_categorical():
        prefix = f"{spec.x}:C({spec.moderator})[T."
    elif formula_type == "heterogeneity":
        prefix = f"{spec.x}:C({spec.group})[T."
    else:
        prefix = None

    ci_df = model.conf_int()
    n = int(model.nobs)
    if prefix:
        for idx_name in ci_df.index:
            if str(idx_name).startswith(prefix):
                pr = _partial_r_from_model(model, idx_name)
                return {
                    "summary_term": str(idx_name),
                    "effect_size": pr,
                    "raw_coefficient": float(model.params[idx_name]),
                    "p_value": float(model.pvalues[idx_name]),
                    "confidence_interval": _partial_r_ci(pr, n),
                    "raw_ci": (
                        float(ci_df.loc[idx_name].iloc[0]),
                        float(ci_df.loc[idx_name].iloc[1]),
                    ),
                }

    for idx_name in ci_df.index:
        if idx_name != "Intercept":
            pr = _partial_r_from_model(model, idx_name)
            return {
                "summary_term": str(idx_name),
                "effect_size": pr,
                "raw_coefficient": float(model.params[idx_name]),
                "p_value": float(model.pvalues[idx_name]),
                "confidence_interval": _partial_r_ci(pr, n),
                "raw_ci": (
                    float(ci_df.loc[idx_name].iloc[0]),
                    float(ci_df.loc[idx_name].iloc[1]),
                ),
            }

    raise DSLExecutionError("Could not identify a non-intercept contrast term in the model")


# ---------------------------------------------------------------------------
# Formula term helper
# ---------------------------------------------------------------------------

def _fterm(col: str, subset: Optional[pd.DataFrame] = None) -> str:
    """Wrap categorical columns with C(); leave others as-is.

    With *subset* supplied the decision is dtype-aware and so matches what
    patsy will actually do.  Without it only the declared list is consulted —
    the older, declaration-only behaviour, kept for the codegen path.
    """
    if subset is not None:
        return f"C({col})" if _is_categorical_col(col, subset) else col
    if col in _resolve_known_categorical():
        return f"C({col})"
    return col


# ═══════════════════════════════════════════════════════════════════════════
# 4. Strict result validation
# ═══════════════════════════════════════════════════════════════════════════

def _validate_result_scalars(result: Dict[str, Any]) -> None:
    """Validate that all core result values are finite scalars.

    Raises DSLExecutionError on first violation. Does NOT sanitise.
    """
    required_scalar = ["effect_size", "p_value", "n_observations", "coverage"]
    for key in required_scalar:
        val = result.get(key)
        if val is None:
            raise DSLExecutionError(f"Result missing required key: '{key}'")
        if isinstance(val, (pd.Series, pd.DataFrame, list)):
            raise DSLExecutionError(
                f"Result['{key}'] must be a scalar, got {type(val).__name__}"
            )
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise DSLExecutionError(
                f"Result['{key}'] must be numeric, got {type(val).__name__}: {val!r}"
            )
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            raise DSLExecutionError(
                f"Result['{key}'] is {fval} — statistical test produced invalid output"
            )

    # p_value range
    pv = float(result["p_value"])
    if pv < 0 or pv > 1:
        raise DSLExecutionError(f"p_value={pv} outside [0, 1]")

    # coverage range
    cov = float(result["coverage"])
    if cov < 0 or cov > 1:
        raise DSLExecutionError(f"coverage={cov} outside [0, 1]")

    # confidence_interval
    ci = result.get("confidence_interval")
    if ci is None or not isinstance(ci, (list, tuple)) or len(ci) != 2:
        raise DSLExecutionError(
            f"confidence_interval must be (lower, upper) tuple, got {ci!r}"
        )
    for i, bound in enumerate(ci):
        if isinstance(bound, (pd.Series, pd.DataFrame, list)):
            raise DSLExecutionError(
                f"confidence_interval[{i}] must be scalar, got {type(bound).__name__}"
            )
        fb = float(bound)
        if math.isnan(fb) or math.isinf(fb):
            raise DSLExecutionError(
                f"confidence_interval[{i}] is {fb} — invalid"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Code generation: reconstruct equivalent Python from a typed spec
# ═══════════════════════════════════════════════════════════════════════════

def spec_to_python_code(spec: DSLTestSpec) -> str:
    """Reconstruct the equivalent Python code that execute_spec() would run.

    This is used for artifact logging so every DSL hypothesis has an
    auditable, self-contained Python script alongside its results.
    """
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "from scipy import stats",
    ]

    needs_sm = spec.controls or spec.weight or spec.family in ("interact", "heterogeneity")
    if needs_sm:
        lines.append("import statsmodels.formula.api as smf")
        lines.append("import statsmodels.api as sm")

    lines.append("")
    lines.append("def test_hypothesis(df):")
    lines.append(f'    """DSL: {spec.raw_sentence}"""')

    cols = spec.all_columns
    lines.append(f"    cols = {cols}")
    lines.append("    subset = df[cols].copy()")
    lines.append("")

    _mc = _resolve_missing_codes()
    lines.append(f"    WVS_MISSING = {_mc!r}" if _mc else "    WVS_MISSING = set()")
    for c in cols:
        if c == spec.weight:
            lines.append(f"    subset['{c}'] = pd.to_numeric(subset['{c}'], errors='coerce')")
            if _mc:
                lines.append(f"    subset['{c}'] = subset['{c}'].where(~subset['{c}'].isin(WVS_MISSING))")
        elif c in _resolve_known_categorical():
            lines.append(f"    # {c}: kept as categorical")
        else:
            lines.append(f"    subset['{c}'] = pd.to_numeric(subset['{c}'], errors='coerce')")
            if _mc:
                lines.append(f"    subset['{c}'] = subset['{c}'].where(~subset['{c}'].isin(WVS_MISSING))")
    lines.append("    subset = subset.dropna()")
    lines.append("    n = len(subset)")
    lines.append("    total_n = len(df)")
    lines.append("")

    if spec.family == "assoc" and not spec.controls and not spec.weight:
        lines.append(f"    # Family: assoc (no controls, no weight) -> Pearson correlation")
        lines.append(f"    x_vals = subset['{spec.x}'].astype(float)")
        lines.append(f"    y_vals = subset['{spec.y}'].astype(float)")
        lines.append(f"    r, p = stats.pearsonr(x_vals, y_vals)")
        lines.append(f"    z = np.arctanh(r)")
        lines.append(f"    se = 1.0 / np.sqrt(n - 3)")
        lines.append(f"    ci_lo = float(np.tanh(z - 1.96 * se))")
        lines.append(f"    ci_hi = float(np.tanh(z + 1.96 * se))")
        lines.append(f"    return dict(effect_size=float(r), p_value=float(p),")
        lines.append(f"                n_observations=n, coverage=n/total_n,")
        lines.append(f"                confidence_interval=(ci_lo, ci_hi))")

    elif spec.family == "diff" and not spec.controls and not spec.weight:
        lines.append(f"    # Family: diff (no controls, no weight) -> schema-dependent binary diff / multigroup omnibus")
        lines.append(f"    y_vals = subset['{spec.y}'].astype(float)")
        lines.append(f"    g_vals = subset['{spec.group}']")
        lines.append(f"    groups = sorted(g_vals.unique(), key=lambda x: str(x))")
        lines.append(f"    k = len(groups)")
        lines.append(f"    if k == 2:")
        lines.append(f"        g1 = y_vals[g_vals == groups[0]]")
        lines.append(f"        g2 = y_vals[g_vals == groups[1]]")
        lines.append(f"        if y_vals.nunique(dropna=True) >= 7:")
        lines.append(f"            t_stat, p = stats.ttest_ind(g2, g1, equal_var=False)")
        lines.append(f"            mean_diff = float(g2.mean() - g1.mean())")
        lines.append(f"            var1 = float(g1.var(ddof=1))")
        lines.append(f"            var2 = float(g2.var(ddof=1))")
        lines.append(f"            se = np.sqrt((var1/len(g1)) + (var2/len(g2)))")
        lines.append(f"            df_num = ((var1/len(g1)) + (var2/len(g2)))**2")
        lines.append(f"            df_den = (((var1/len(g1))**2)/max(len(g1)-1,1)) + (((var2/len(g2))**2)/max(len(g2)-1,1))")
        lines.append(f"            df_welch = df_num / df_den if df_den > 0 else max(len(g1)+len(g2)-2, 1)")
        lines.append(f"            t_crit = float(stats.t.ppf(0.975, df_welch))")
        lines.append(f"            ci_lo = mean_diff - t_crit * se")
        lines.append(f"            ci_hi = mean_diff + t_crit * se")
        lines.append(f"            return dict(effect_size=mean_diff, p_value=float(p),")
        lines.append(f"                        n_observations=len(g1)+len(g2), coverage=(len(g1)+len(g2))/total_n,")
        lines.append(f"                        confidence_interval=(ci_lo, ci_hi),")
        lines.append(f"                        effect_size_kind='mean_difference')")
        lines.append(f"        u_stat, p = stats.mannwhitneyu(g2, g1, alternative='two-sided')")
        lines.append(f"        rank_biserial = float((2.0 * u_stat / (len(g1) * len(g2))) - 1.0)")
        lines.append(f"        clipped = max(min(rank_biserial, 0.999999), -0.999999)")
        lines.append(f"        z_val = np.arctanh(clipped)")
        lines.append(f"        se_z = 1.0 / np.sqrt(max(len(g1)+len(g2)-3, 1))")
        lines.append(f"        ci_lo = float(np.tanh(z_val - 1.96 * se_z))")
        lines.append(f"        ci_hi = float(np.tanh(z_val + 1.96 * se_z))")
        lines.append(f"        return dict(effect_size=rank_biserial, p_value=float(p), n_observations=len(g1)+len(g2),")
        lines.append(f"                    coverage=(len(g1)+len(g2))/total_n,")
        lines.append(f"                    confidence_interval=(ci_lo, ci_hi),")
        lines.append(f"                    effect_size_kind='rank_biserial')")
        lines.append(f"    else:  # >2 groups -> Kruskal-Wallis")
        lines.append(f"        arrays = [y_vals[g_vals == g] for g in groups if len(y_vals[g_vals == g]) >= 2]")
        lines.append(f"        h_stat, p = stats.kruskal(*arrays)")
        lines.append(f"        nn = sum(len(a) for a in arrays)")
        lines.append(f"        eps_sq = float(h_stat * (nn+1) / (nn*nn - 1))")
        lines.append(f"        import math")
        lines.append(f"        se_eps = math.sqrt(4*eps_sq*(1-eps_sq)**2 / max(nn-1,1))")
        lines.append(f"        omnibus = dict(test='kruskal_wallis', statistic=float(h_stat), p_value=float(p),")
        lines.append(f"                       effect_size=float(eps_sq), effect_size_kind='epsilon_squared',")
        lines.append(f"                       confidence_interval=(max(0,eps_sq-1.96*se_eps), min(1,eps_sq+1.96*se_eps)))")
        lines.append(f"        base = y_vals[g_vals == groups[0]]")
        lines.append(f"        contrast = y_vals[g_vals == groups[1]]")
        lines.append(f"        u_stat, pair_p = stats.mannwhitneyu(contrast, base, alternative='two-sided')")
        lines.append(f"        rank_biserial = float((2.0 * u_stat / (len(base) * len(contrast))) - 1.0)")
        lines.append(f"        clipped = max(min(rank_biserial, 0.999999), -0.999999)")
        lines.append(f"        z_val = np.arctanh(clipped)")
        lines.append(f"        pair_n = len(base) + len(contrast)")
        lines.append(f"        se_z = 1.0 / np.sqrt(max(pair_n-3, 1))")
        lines.append(f"        ci_lo = float(np.tanh(z_val - 1.96 * se_z))")
        lines.append(f"        ci_hi = float(np.tanh(z_val + 1.96 * se_z))")
        lines.append(f"        return dict(effect_size=rank_biserial, p_value=float(pair_p), n_observations=pair_n,")
        lines.append(f"                    coverage=pair_n/total_n, confidence_interval=(ci_lo, ci_hi),")
        lines.append(f"                    effect_size_kind='rank_biserial', omnibus=omnibus)")

    else:
        _known_cat = _resolve_known_categorical()
        def _fterm(c: str) -> str:
            return f"C({c})" if c in _known_cat else c

        rhs_parts: list[str] = []
        categorical_summary_var = None

        if spec.family == "assoc":
            rhs_parts.append(spec.x)
            primary_term = spec.x
            rhs_parts.extend(_fterm(c) for c in spec.controls)
        elif spec.family == "diff":
            rhs_parts.append(f"C({spec.group})")
            rhs_parts.extend(_fterm(c) for c in spec.controls)
            primary_term = None
            categorical_summary_var = spec.group
        elif spec.family == "interact":
            z_term = _fterm(spec.moderator)
            rhs_parts.append(f"{spec.x} * {z_term}")
            rhs_parts.extend(_fterm(c) for c in spec.controls)
            if spec.moderator in _known_cat:
                primary_term = None
                categorical_summary_var = spec.moderator
            else:
                primary_term = f"{spec.x}:{spec.moderator}"
        elif spec.family == "heterogeneity":
            rhs_parts.append(f"{spec.x} * C({spec.group})")
            rhs_parts.extend(_fterm(c) for c in spec.controls)
            primary_term = None
            categorical_summary_var = spec.group
        else:
            primary_term = spec.x
            rhs_parts.append(spec.x)

        formula = f"{spec.y} ~ " + " + ".join(rhs_parts)
        method = "WLS" if spec.weight else "OLS"

        lines.append(f"    # Family: {spec.family} -> {method} regression")
        lines.append(f"    formula = '{formula}'")
        if spec.weight:
            lines.append(f"    model = smf.wls(formula, data=subset, weights=subset['{spec.weight}']).fit()")
        else:
            lines.append(f"    model = smf.ols(formula, data=subset).fit()")

        lines.append(f"")
        if categorical_summary_var is not None:
            if spec.family == "diff":
                target = f"C({spec.group})"
                prefix = f"C({spec.group})[T."
            elif spec.family == "heterogeneity":
                target = f"{spec.x}:C({spec.group})"
                prefix = f"{spec.x}:C({spec.group})[T."
            elif spec.family == "interact" and spec.moderator in _known_cat:
                target = f"{spec.x}:C({spec.moderator})"
                prefix = f"{spec.x}:C({spec.moderator})[T."
            else:
                target = "?"
                prefix = ""

            lines.append(f"    level_count = subset['{categorical_summary_var}'].nunique(dropna=True)")
            lines.append(f"    ci_df = model.conf_int()")
            lines.append(f"    for idx_name in ci_df.index:")
            lines.append(f"        if str(idx_name).startswith('{prefix}'):")
            lines.append(f"            primary_term = str(idx_name)")
            lines.append(f"            break")
            lines.append(f"    effect = float(model.params[primary_term])")
            lines.append(f"    p_value = float(model.pvalues[primary_term])")
            lines.append(f"    ci = model.conf_int().loc[primary_term]")
            lines.append(f"    ci_lo, ci_hi = float(ci.iloc[0]), float(ci.iloc[1])")
            lines.append(f"    if level_count > 2:")
            lines.append(f"        anova_table = sm.stats.anova_lm(model, typ=2)")
            lines.append(f"        ss_term = anova_table.loc['{target}', 'sum_sq']")
            lines.append(f"        ss_resid = anova_table.loc['Residual', 'sum_sq']")
            lines.append(f"        omnibus = dict(test='anova_type_2', term='{target}', statistic=float(anova_table.loc['{target}', 'F']),")
            lines.append(f"                       p_value=float(anova_table.loc['{target}', 'PR(>F)']),")
            lines.append(f"                       effect_size=float(ss_term / (ss_term + ss_resid)),")
            lines.append(f"                       effect_size_kind='partial_eta_squared')")
            lines.append(f"        return dict(effect_size=effect, p_value=p_value,")
            lines.append(f"                    n_observations=int(model.nobs), coverage=n/total_n,")
            lines.append(f"                    confidence_interval=(ci_lo, ci_hi),")
            lines.append(f"                    r_squared=float(model.rsquared),")
            lines.append(f"                    effect_size_kind='coefficient', summary_term=primary_term, omnibus=omnibus)")
            lines.append(f"    return dict(effect_size=effect, p_value=p_value,")
            lines.append(f"                n_observations=int(model.nobs), coverage=n/total_n,")
            lines.append(f"                confidence_interval=(ci_lo, ci_hi),")
            lines.append(f"                r_squared=float(model.rsquared),")
            lines.append(f"                effect_size_kind='coefficient', summary_term=primary_term)")
        else:
            lines.append(f"    effect = float(model.params['{primary_term}'])")
            lines.append(f"    p_value = float(model.pvalues['{primary_term}'])")
            lines.append(f"    ci = model.conf_int().loc['{primary_term}']")
            lines.append(f"    ci_lo, ci_hi = float(ci.iloc[0]), float(ci.iloc[1])")
            lines.append(f"    return dict(effect_size=effect, p_value=p_value,")
            lines.append(f"                n_observations=int(model.nobs), coverage=n/total_n,")
            lines.append(f"                confidence_interval=(ci_lo, ci_hi),")
            lines.append(f"                r_squared=float(model.rsquared),")
            lines.append(f"                effect_size_kind='coefficient', summary_term='{primary_term}')")

    lines.append("")
    lines.append("result = test_hypothesis(df)")
    return "\n".join(lines)
